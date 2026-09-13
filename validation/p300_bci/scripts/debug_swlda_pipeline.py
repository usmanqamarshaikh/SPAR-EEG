from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import statsmodels.api as sm
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bci_p300_denoise.eeglab_io import channel_indices, get_struct_field, load_eeglab_run, memmap_fdt


DATASET_ROOT = Path(__file__).resolve().parents[3] / "data" / "Won2022_BIDS"
MANIFEST = PROJECT_ROOT / "outputs" / "preprocessed_fp12" / "manifests" / "sub-001_p300_preprocessing_manifest.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "swlda_decoder_debug"
SEQ_CODES = np.arange(1, 13)
FULL_REPEAT = 15
SPELLER_MATRIX = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_")


@dataclass(frozen=True)
class RunFiles:
    subject: str
    task: str
    run: int
    wide_h5: Path
    branches_h5: Path

    @property
    def is_train(self) -> bool:
        return "train" in self.task.lower()

    @property
    def is_test(self) -> bool:
        return "test" in self.task.lower()


@dataclass
class FeatureBlock:
    X: np.ndarray
    y01: np.ndarray
    ypm: np.ndarray
    seq: np.ndarray
    run: np.ndarray
    text: np.ndarray


def decode_attr(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        return item.decode("utf-8") if isinstance(item, bytes) else str(item)
    if arr.dtype.kind in {"S", "U", "O"}:
        return "".join(item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in arr.reshape(-1))
    return str(value)


def load_manifest(path: Path) -> list[RunFiles]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("status") == "ok"]
    return sorted(
        [
            RunFiles(
                subject=row["subject"],
                task=row["task"],
                run=int(row["run"]),
                wide_h5=Path(row["wide_h5"]),
                branches_h5=Path(row["branches_h5"]),
            )
            for row in rows
        ],
        key=lambda item: item.run,
    )


def branch_data(run: RunFiles, branch: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    with h5py.File(run.branches_h5, "r") as hb, h5py.File(run.wide_h5, "r") as hw:
        data = np.asarray(hb[f"{branch}/data"][:], dtype=np.float64)
        seq = np.asarray(hw["events/event_sequence"][:], dtype=np.int16)
        target = np.asarray(hw["events/event_target"][:], dtype=np.int16)
        fs = float(np.asarray(hw["wide/srate"][()]).reshape(-1)[0])
        text = decode_attr(hw.attrs.get("text_to_spell", ""))
    return data, seq, target, fs, text


def raw_data(dataset_root: Path, subject: str, task: str, run: int, channels: list[str] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    info = load_eeglab_run(dataset_root, subject, task, run)
    data = np.asarray(memmap_fdt(info), dtype=np.float64)
    if channels:
        data = data[channel_indices(info.labels, channels), :]
    seq = np.asarray(get_struct_field(info.eeg, "event_sequence", []), dtype=np.int16).reshape(-1)
    target = np.asarray(get_struct_field(info.eeg, "event_target", []), dtype=np.int16).reshape(-1)
    text = str(get_struct_field(info.eeg, "text_to_spell", ""))
    return data, seq, target, float(info.srate), text


def bandpass(data: np.ndarray, fs: float, low_hz: float = 0.5, high_hz: float = 10.0) -> np.ndarray:
    sos = butter(4, [low_hz / (fs / 2), high_hz / (fs / 2)], btype="band", output="sos")
    demeaned = data - np.mean(data, axis=1, keepdims=True)
    return sosfiltfilt(sos, demeaned, axis=1)


def event_indices(seq: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(seq, SEQ_CODES))


def extract_features(
    data: np.ndarray,
    seq_series: np.ndarray,
    target_series: np.ndarray,
    fs: float,
    run_id: int,
    text: str,
    decim: int = 24,
) -> FeatureBlock:
    filt = bandpass(data, fs)
    frame_start = 0
    frame_stop = int(np.floor(0.6 * fs))
    base_start = int(np.floor(-0.2 * fs))
    base_stop = 0
    n_dec = frame_stop // decim

    feats: list[np.ndarray] = []
    labels01: list[int] = []
    seqs: list[int] = []
    runs: list[int] = []
    texts: list[str] = []
    for idx in event_indices(seq_series):
        b0 = idx + base_start
        b1 = idx + base_stop
        f0 = idx + frame_start
        f1 = idx + frame_stop
        if b0 < 0 or b1 <= b0 or f0 < 0 or f1 > filt.shape[1]:
            continue
        code = int(seq_series[idx])
        target = int(target_series[idx])
        if code < 1 or code > 12 or target not in (1, 2):
            continue
        epoch = filt[:, f0:f1] - np.mean(filt[:, b0:b1], axis=1, keepdims=True)
        epoch = epoch[:, : n_dec * decim].reshape(epoch.shape[0], n_dec, decim).mean(axis=2)
        feats.append(epoch.reshape(-1))
        labels01.append(1 if target == 1 else 0)
        seqs.append(code)
        runs.append(run_id)
        texts.append(text)

    X = np.asarray(feats, dtype=np.float64)
    y01 = np.asarray(labels01, dtype=np.int8)
    return FeatureBlock(
        X=X,
        y01=y01,
        ypm=2 * y01.astype(np.float64) - 1,
        seq=np.asarray(seqs, dtype=np.int16),
        run=np.asarray(runs, dtype=np.int16),
        text=np.asarray(texts, dtype=object),
    )


def concat(blocks: list[FeatureBlock]) -> FeatureBlock:
    return FeatureBlock(
        X=np.vstack([b.X for b in blocks]),
        y01=np.concatenate([b.y01 for b in blocks]),
        ypm=np.concatenate([b.ypm for b in blocks]),
        seq=np.concatenate([b.seq for b in blocks]),
        run=np.concatenate([b.run for b in blocks]),
        text=np.concatenate([b.text for b in blocks]),
    )


def train_backward_ols_no_intercept(X: np.ndarray, y: np.ndarray, p_val: float = 0.08, max_features: int = 60):
    selected = np.arange(X.shape[1])
    while selected.size > 1:
        result = sm.OLS(y, X[:, selected]).fit()
        p = np.where(np.isfinite(result.pvalues), result.pvalues, 1.0)
        worst = int(np.argmax(p))
        if p[worst] <= p_val:
            break
        selected = np.delete(selected, worst)
    if selected.size > max_features:
        result = sm.OLS(y, X[:, selected]).fit()
        p = np.where(np.isfinite(result.pvalues), result.pvalues, 1.0)
        selected = selected[np.argsort(p)[:max_features]]
    reg = LinearRegression().fit(X[:, selected], y)
    return selected, lambda Z: reg.predict(Z[:, selected])


def train_stepwise_forward_backward(
    X: np.ndarray,
    y: np.ndarray,
    p_enter: float = 0.08,
    p_remove: float = 0.15,
    max_features: int = 60,
):
    selected: list[int] = []
    remaining = set(range(X.shape[1]))

    while remaining and len(selected) < max_features:
        best_feature = None
        best_p = np.inf
        for feature in sorted(remaining):
            cols = selected + [feature]
            result = sm.OLS(y, sm.add_constant(X[:, cols], has_constant="add")).fit()
            p = float(result.pvalues[-1]) if len(result.pvalues) else np.inf
            if np.isfinite(p) and p < best_p:
                best_p = p
                best_feature = feature

        if best_feature is None or best_p > p_enter:
            break
        selected.append(best_feature)
        remaining.remove(best_feature)

        changed = True
        while changed and selected:
            changed = False
            result = sm.OLS(y, sm.add_constant(X[:, selected], has_constant="add")).fit()
            pvals = np.asarray(result.pvalues[1:], dtype=np.float64)
            pvals = np.where(np.isfinite(pvals), pvals, 1.0)
            worst_idx = int(np.argmax(pvals))
            if pvals[worst_idx] > p_remove:
                removed = selected.pop(worst_idx)
                remaining.add(removed)
                changed = True

    if not selected:
        return train_corr_linear(X, y, max_features=max_features)

    selected_arr = np.asarray(selected[:max_features], dtype=int)
    reg = LinearRegression().fit(X[:, selected_arr], y)
    return selected_arr, lambda Z: reg.predict(Z[:, selected_arr])


def train_corr_linear(X: np.ndarray, y: np.ndarray, max_features: int = 60):
    y0 = y - np.mean(y)
    X0 = X - np.mean(X, axis=0, keepdims=True)
    denom = np.sqrt(np.sum(X0**2, axis=0) * np.sum(y0**2))
    r = np.divide(np.sum(X0 * y0[:, None], axis=0), denom, out=np.zeros(X.shape[1]), where=denom > 0)
    selected = np.argsort(np.abs(r))[::-1][: min(max_features, X.shape[1])]
    reg = LinearRegression().fit(X[:, selected], y)
    return selected, lambda Z: reg.predict(Z[:, selected])


def train_lda(X: np.ndarray, y01: np.ndarray):
    clf = LinearDiscriminantAnalysis(priors=[0.5, 0.5]).fit(X, y01)
    return np.arange(X.shape[1]), lambda Z: clf.decision_function(Z)


def train_logistic(X: np.ndarray, y01: np.ndarray):
    clf = LogisticRegression(class_weight="balanced", max_iter=5000, solver="liblinear").fit(X, y01)
    return np.arange(X.shape[1]), lambda Z: clf.decision_function(Z)


def letter_acc_from_chunks(scores: np.ndarray, seq: np.ndarray, y01: np.ndarray, run_ids: np.ndarray, sign: float = 1.0) -> tuple[np.ndarray, dict[int, str]]:
    scores = sign * scores
    acc_sum = np.zeros(FULL_REPEAT, dtype=np.float64)
    n_letters = 0
    answers: dict[int, str] = {}
    for run_id in np.unique(run_ids):
        idx_run = np.flatnonzero(run_ids == run_id)
        sc = scores[idx_run]
        sq = seq[idx_run]
        yy = y01[idx_run]
        n_chunks = len(idx_run) // (FULL_REPEAT * 12)
        chars_final: list[str] = []
        for letter in range(n_chunks):
            a = letter * FULL_REPEAT * 12
            b = a + FULL_REPEAT * 12
            sc_l = sc[a:b]
            sq_l = sq[a:b]
            yy_l = yy[a:b]
            target_codes = sq_l[yy_l == 1]
            rows = target_codes[(target_codes >= 1) & (target_codes <= 6)]
            cols = target_codes[(target_codes >= 7) & (target_codes <= 12)]
            if rows.size == 0 or cols.size == 0:
                continue
            true_row = int(np.bincount(rows, minlength=13).argmax())
            true_col = int(np.bincount(cols, minlength=13).argmax())
            n_letters += 1
            for rep in range(FULL_REPEAT):
                ssum = np.zeros(12, dtype=np.float64)
                for k in range((rep + 1) * 12):
                    code = int(sq_l[k])
                    if 1 <= code <= 12:
                        ssum[code - 1] += sc_l[k]
                pred_row = int(np.argmax(ssum[:6]) + 1)
                pred_col = int(np.argmax(ssum[6:]) + 7)
                acc_sum[rep] += float(pred_row == true_row and pred_col == true_col)
                if rep == FULL_REPEAT - 1:
                    chars_final.append(SPELLER_MATRIX[(pred_row - 1) * 6 + (pred_col - 7)])
        answers[int(run_id)] = "".join(chars_final)
    return acc_sum / max(1, n_letters), answers


def evaluate(name: str, train: FeatureBlock, test: FeatureBlock, trainer) -> dict[str, Any]:
    selected, scorer = trainer(train.X, train.y01 if name in {"lda", "logistic"} else train.ypm)
    score_tr = scorer(train.X)
    score_te = scorer(test.X)
    pred = (score_te >= 0).astype(int)
    auc = roc_auc_score(test.y01, score_te) if len(np.unique(test.y01)) == 2 else np.nan
    bacc = balanced_accuracy_score(test.y01, pred)
    acc_pos, answers_pos = letter_acc_from_chunks(score_te, test.seq, test.y01, test.run, sign=1.0)
    acc_neg, answers_neg = letter_acc_from_chunks(score_te, test.seq, test.y01, test.run, sign=-1.0)
    if float(np.nanmean(acc_neg)) > float(np.nanmean(acc_pos)):
        sign = -1.0
        acc = acc_neg
        answers = answers_neg
    else:
        sign = 1.0
        acc = acc_pos
        answers = answers_pos
    return {
        "model": name,
        "n_features": int(len(selected)),
        "train_target_score_mean": float(np.mean(score_tr[train.y01 == 1])),
        "train_nontarget_score_mean": float(np.mean(score_tr[train.y01 == 0])),
        "test_target_score_mean": float(np.mean(score_te[test.y01 == 1])),
        "test_nontarget_score_mean": float(np.mean(score_te[test.y01 == 0])),
        "test_auc": float(auc),
        "test_bacc": float(bacc),
        "best_decode_sign": sign,
        "letter_acc_mean": float(np.mean(acc)),
        "letter_acc_rep5": float(acc[4]),
        "letter_acc_rep10": float(acc[9]),
        "letter_acc_rep15": float(acc[14]),
        "answers": ";".join(f"run{run}:{ans}" for run, ans in sorted(answers.items())),
    }


def build_blocks(args: argparse.Namespace) -> tuple[FeatureBlock, FeatureBlock]:
    if args.source == "branch":
        runs = load_manifest(args.manifest)
        train_blocks = []
        test_blocks = []
        for run in runs:
            data, seq, target, fs, text = branch_data(run, args.branch)
            block = extract_features(data, seq, target, fs, run.run, text)
            (train_blocks if run.is_train else test_blocks).append(block)
        return concat(train_blocks), concat(test_blocks)

    train_blocks = []
    test_blocks = []
    tasks = [
        ("P300trainrun1", 6),
        ("P300trainrun2", 7),
        ("P300testrun1", 8),
        ("P300testrun2", 9),
        ("P300testrun3", 10),
        ("P300testrun4", 11),
    ]
    for task, run_id in tasks:
        data, seq, target, fs, text = raw_data(args.dataset_root, args.subject, task, run_id, args.channels)
        block = extract_features(data, seq, target, fs, run_id, text)
        (train_blocks if "train" in task.lower() else test_blocks).append(block)
    return concat(train_blocks), concat(test_blocks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug P300 SWLDA-like feature/classifier/decoder choices.")
    parser.add_argument("--source", choices=["branch", "raw"], default="branch")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--branch", default="baseline")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--subject", default="sub-001")
    parser.add_argument("--channels", nargs="*", default=["FP1", "FP2"], help="Raw source channels. Use no values for all 32.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if args.channels == []:
        args.channels = None

    train, test = build_blocks(args)
    print(f"source={args.source} branch={args.branch} channels={args.channels or 'ALL'}")
    print(f"train X={train.X.shape} target_frac={train.y01.mean():.3f}")
    print(f"test  X={test.X.shape} target_frac={test.y01.mean():.3f}")

    trainers = {
        "swlda_backward_ols_no_intercept": train_backward_ols_no_intercept,
        "swlda_forward_backward": train_stepwise_forward_backward,
        "corr60_linear": train_corr_linear,
        "lda_uniform": train_lda,
        "logistic_balanced": train_logistic,
    }
    rows = [evaluate(name, train, test, trainer) for name, trainer in trainers.items()]
    for row in rows:
        print(
            f"{row['model']}: AUC={row['test_auc']:.3f} BACC={row['test_bacc']:.3f} "
            f"L5={row['letter_acc_rep5']:.3f} L10={row['letter_acc_rep10']:.3f} "
            f"L15={row['letter_acc_rep15']:.3f} sign={row['best_decode_sign']:+.0f} "
            f"features={row['n_features']}"
        )
        print(f"  target/non-target score mean test: {row['test_target_score_mean']:.4g} / {row['test_nontarget_score_mean']:.4g}")
        print(f"  answers: {row['answers']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.branch if args.source == "branch" else ("all32" if args.channels is None else "_".join(args.channels))
    out_csv = args.out_dir / f"{args.subject}_{args.source}_{tag}_debug_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
