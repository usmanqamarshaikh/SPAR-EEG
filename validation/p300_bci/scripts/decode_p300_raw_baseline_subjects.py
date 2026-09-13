from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.signal import butter, sosfiltfilt
from scipy.stats import sem
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bci_p300_denoise.eeglab_io import channel_indices, get_struct_field, load_eeglab_run, memmap_fdt


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "data" / "Won2022_BIDS"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "raw_fp12_baseline_swlda"
SEQ_CODES = np.arange(1, 13)
FULL_REPEAT = 15
SPELLER_MATRIX = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_")


@dataclass(frozen=True)
class SubjectResult:
    subject: str
    n_train_epochs: int
    n_test_epochs: int
    n_features: int
    train_binary_accuracy: float
    epoch_auc: float
    epoch_bacc: float
    total_test_letters: int
    repetition_accuracy: np.ndarray
    elapsed_sec: float
    status: str
    message: str


def matlab_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return value


def matlab_text(value: Any) -> str:
    value = matlab_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "S"}:
        return "".join(arr.astype(str).reshape(-1)).strip()
    return str(value)


def natural_subject_key(name: str) -> tuple[int, str]:
    match = re.search(r"sub-(\d+)$", name)
    if not match:
        return (10**9, name)
    return (int(match.group(1)), name)


def discover_subjects(dataset_root: Path) -> list[str]:
    return sorted(
        [
            item.name
            for item in dataset_root.iterdir()
            if item.is_dir() and re.match(r"^sub-\d+$", item.name)
        ],
        key=natural_subject_key,
    )


def selected_subjects(dataset_root: Path, subjects: list[str] | None, n_subjects: int | None, start_subject: int) -> list[str]:
    if subjects:
        return subjects
    all_subjects = discover_subjects(dataset_root)
    if n_subjects is None:
        return all_subjects
    wanted = {f"sub-{idx:03d}" for idx in range(start_subject, start_subject + n_subjects)}
    return [sub for sub in all_subjects if sub in wanted]


def p300_tasks() -> list[tuple[str, int, str]]:
    return [
        ("P300trainrun1", 6, "train"),
        ("P300trainrun2", 7, "train"),
        ("P300testrun1", 8, "test"),
        ("P300testrun2", 9, "test"),
        ("P300testrun3", 10, "test"),
        ("P300testrun4", 11, "test"),
    ]


def bandpass_p300(data: np.ndarray, fs: float, low_hz: float = 0.5, high_hz: float = 10.0) -> np.ndarray:
    sos = butter(4, [low_hz / (fs / 2), high_hz / (fs / 2)], btype="band", output="sos")
    demeaned = data - np.mean(data, axis=1, keepdims=True)
    return sosfiltfilt(sos, demeaned, axis=1)


def event_indices(event_sequence: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(event_sequence, SEQ_CODES))


def extract_features_from_run(
    dataset_root: Path,
    subject: str,
    task: str,
    run_id: int,
    channels: list[str],
    decim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    info = load_eeglab_run(dataset_root, subject, task, run_id)
    idx_ch = channel_indices(info.labels, channels)
    raw = np.asarray(memmap_fdt(info)[idx_ch, :], dtype=np.float64)
    filtered = bandpass_p300(raw, info.srate)

    event_sequence = np.asarray(get_struct_field(info.eeg, "event_sequence", []), dtype=np.int16).reshape(-1)
    event_target = np.asarray(get_struct_field(info.eeg, "event_target", []), dtype=np.int16).reshape(-1)
    text_to_spell = matlab_text(get_struct_field(info.eeg, "text_to_spell", ""))

    frame_start = 0
    frame_stop = int(np.floor(0.6 * info.srate))
    base_start = int(np.floor(-0.2 * info.srate))
    base_stop = 0
    n_dec = frame_stop // decim

    features: list[np.ndarray] = []
    labels01: list[int] = []
    seq_codes: list[int] = []
    run_codes: list[int] = []

    for idx in event_indices(event_sequence):
        b0 = idx + base_start
        b1 = idx + base_stop
        f0 = idx + frame_start
        f1 = idx + frame_stop
        if b0 < 0 or b1 <= b0 or f0 < 0 or f1 > filtered.shape[1]:
            continue
        code = int(event_sequence[idx])
        target = int(event_target[idx])
        if code < 1 or code > 12 or target not in (1, 2):
            continue

        epoch = filtered[:, f0:f1] - np.mean(filtered[:, b0:b1], axis=1, keepdims=True)
        epoch = epoch[:, : n_dec * decim].reshape(epoch.shape[0], n_dec, decim).mean(axis=2)
        features.append(epoch.reshape(-1))
        labels01.append(1 if target == 1 else 0)
        seq_codes.append(code)
        run_codes.append(run_id)

    if not features:
        raise RuntimeError(f"No valid P300 epochs extracted for {subject} {task} run-{run_id}.")

    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(labels01, dtype=np.int8),
        np.asarray(seq_codes, dtype=np.int16),
        np.asarray(run_codes, dtype=np.int16),
        text_to_spell,
    )


def top_correlated_features(X: np.ndarray, y: np.ndarray, max_features: int) -> np.ndarray:
    y0 = y - np.mean(y)
    X0 = X - np.mean(X, axis=0, keepdims=True)
    denom = np.sqrt(np.sum(X0**2, axis=0) * np.sum(y0**2))
    corr = np.divide(np.sum(X0 * y0[:, None], axis=0), denom, out=np.zeros(X.shape[1]), where=denom > 0)
    return np.argsort(np.abs(corr))[::-1][: min(max_features, X.shape[1])]


def train_swlda_stepwise(
    X: np.ndarray,
    ypm: np.ndarray,
    p_enter: float,
    p_remove: float,
    max_features: int,
) -> tuple[np.ndarray, LinearRegression, float, float]:
    selected: list[int] = []
    remaining = set(range(X.shape[1]))

    while remaining and len(selected) < max_features:
        best_feature = None
        best_p = np.inf
        for feature in sorted(remaining):
            cols = selected + [feature]
            result = sm.OLS(ypm, sm.add_constant(X[:, cols], has_constant="add")).fit()
            p_value = float(result.pvalues[-1]) if len(result.pvalues) else np.inf
            if np.isfinite(p_value) and p_value < best_p:
                best_p = p_value
                best_feature = feature

        if best_feature is None or best_p > p_enter:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)

        changed = True
        while changed and selected:
            changed = False
            result = sm.OLS(ypm, sm.add_constant(X[:, selected], has_constant="add")).fit()
            pvalues = np.asarray(result.pvalues[1:], dtype=np.float64)
            pvalues = np.where(np.isfinite(pvalues), pvalues, 1.0)
            worst = int(np.argmax(pvalues))
            if pvalues[worst] > p_remove:
                removed = selected.pop(worst)
                remaining.add(removed)
                changed = True

    selected_array = np.asarray(selected[:max_features], dtype=int) if selected else top_correlated_features(X, ypm, max_features)
    reg = LinearRegression().fit(X[:, selected_array], ypm)
    train_scores = reg.predict(X[:, selected_array])
    direction = 1.0
    if np.mean(train_scores[ypm == 1]) < np.mean(train_scores[ypm == -1]):
        direction = -1.0
    pred = np.sign(direction * train_scores)
    pred[pred == 0] = 1
    train_acc = float(np.mean(pred == ypm))
    return selected_array, reg, direction, train_acc


def decode_letters_from_events(scores: np.ndarray, seq: np.ndarray, y01: np.ndarray, run_codes: np.ndarray) -> tuple[np.ndarray, int, dict[int, str]]:
    acc_sum = np.zeros(FULL_REPEAT, dtype=np.float64)
    n_letters = 0
    answers: dict[int, str] = {}

    for run_id in np.unique(run_codes):
        idx_run = np.flatnonzero(run_codes == run_id)
        scores_r = scores[idx_run]
        seq_r = seq[idx_run]
        y_r = y01[idx_run]
        chunk_len = FULL_REPEAT * len(SEQ_CODES)
        n_chunks = len(idx_run) // chunk_len
        final_chars: list[str] = []

        for chunk in range(n_chunks):
            begin = chunk * chunk_len
            stop = begin + chunk_len
            scores_l = scores_r[begin:stop]
            seq_l = seq_r[begin:stop]
            y_l = y_r[begin:stop]
            target_codes = seq_l[y_l == 1]
            rows = target_codes[(target_codes >= 1) & (target_codes <= 6)]
            cols = target_codes[(target_codes >= 7) & (target_codes <= 12)]
            if rows.size == 0 or cols.size == 0:
                continue
            true_row = int(np.bincount(rows, minlength=13).argmax())
            true_col = int(np.bincount(cols, minlength=13).argmax())
            n_letters += 1

            for rep in range(FULL_REPEAT):
                score_sum = np.zeros(12, dtype=np.float64)
                for k in range((rep + 1) * 12):
                    code = int(seq_l[k])
                    if 1 <= code <= 12:
                        score_sum[code - 1] += scores_l[k]
                pred_row = int(np.argmax(score_sum[:6]) + 1)
                pred_col = int(np.argmax(score_sum[6:12]) + 7)
                acc_sum[rep] += float(pred_row == true_row and pred_col == true_col)
                if rep == FULL_REPEAT - 1:
                    final_chars.append(SPELLER_MATRIX[(pred_row - 1) * 6 + (pred_col - 7)])
        answers[int(run_id)] = "".join(final_chars)

    return acc_sum / max(1, n_letters), n_letters, answers


def decode_subject(
    dataset_root: Path,
    subject: str,
    channels: list[str],
    decim: int,
    p_enter: float,
    p_remove: float,
    max_features: int,
) -> SubjectResult:
    t0 = time.perf_counter()
    try:
        train_parts = []
        test_parts = []
        for task, run_id, split in p300_tasks():
            X, y01, seq, run_codes, _text = extract_features_from_run(dataset_root, subject, task, run_id, channels, decim)
            item = (X, y01, seq, run_codes)
            (train_parts if split == "train" else test_parts).append(item)

        X_train = np.vstack([item[0] for item in train_parts])
        y_train01 = np.concatenate([item[1] for item in train_parts])
        y_train_pm = 2 * y_train01.astype(np.float64) - 1

        X_test = np.vstack([item[0] for item in test_parts])
        y_test01 = np.concatenate([item[1] for item in test_parts])
        seq_test = np.concatenate([item[2] for item in test_parts])
        run_test = np.concatenate([item[3] for item in test_parts])

        selected, reg, direction, train_acc = train_swlda_stepwise(
            X_train,
            y_train_pm,
            p_enter=p_enter,
            p_remove=p_remove,
            max_features=max_features,
        )
        scores = direction * reg.predict(X_test[:, selected])
        pred = (scores >= 0).astype(np.int8)
        epoch_auc = float(roc_auc_score(y_test01, scores)) if len(np.unique(y_test01)) == 2 else np.nan
        epoch_bacc = float(balanced_accuracy_score(y_test01, pred))
        rep_acc, n_letters, answers = decode_letters_from_events(scores, seq_test, y_test01, run_test)
        elapsed = time.perf_counter() - t0
        message = "; ".join(f"run-{run_id}:{answer}" for run_id, answer in sorted(answers.items()))

        return SubjectResult(
            subject=subject,
            n_train_epochs=int(X_train.shape[0]),
            n_test_epochs=int(X_test.shape[0]),
            n_features=int(selected.size),
            train_binary_accuracy=train_acc,
            epoch_auc=epoch_auc,
            epoch_bacc=epoch_bacc,
            total_test_letters=int(n_letters),
            repetition_accuracy=rep_acc,
            elapsed_sec=elapsed,
            status="ok",
            message=message,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        return SubjectResult(
            subject=subject,
            n_train_epochs=0,
            n_test_epochs=0,
            n_features=0,
            train_binary_accuracy=np.nan,
            epoch_auc=np.nan,
            epoch_bacc=np.nan,
            total_test_letters=0,
            repetition_accuracy=np.full(FULL_REPEAT, np.nan),
            elapsed_sec=elapsed,
            status="failed",
            message=repr(exc),
        )


def result_to_row(result: SubjectResult, channels: list[str]) -> dict[str, object]:
    row: dict[str, object] = {
        "subject": result.subject,
        "channels": ",".join(channels),
        "status": result.status,
        "message": result.message,
        "elapsed_sec": f"{result.elapsed_sec:.3f}",
        "n_train_epochs": result.n_train_epochs,
        "n_test_epochs": result.n_test_epochs,
        "total_test_letters": result.total_test_letters,
        "selected_feature_count": result.n_features,
        "train_binary_accuracy": f"{result.train_binary_accuracy:.6f}",
        "epoch_auc": f"{result.epoch_auc:.6f}",
        "epoch_bacc": f"{result.epoch_bacc:.6f}",
        "auc_repetition_accuracy": f"{float(np.nanmean(result.repetition_accuracy)):.6f}",
        "acc_rep_5": f"{result.repetition_accuracy[4]:.6f}",
        "acc_rep_10": f"{result.repetition_accuracy[9]:.6f}",
        "acc_rep_15": f"{result.repetition_accuracy[14]:.6f}",
    }
    for rep in range(1, FULL_REPEAT + 1):
        row[f"acc_rep_{rep}"] = f"{result.repetition_accuracy[rep - 1]:.6f}"
    return row


def write_subject_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_group_summary(rows: list[dict[str, object]], out_dir: Path) -> None:
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return
    numeric_cols = [
        "auc_repetition_accuracy",
        "acc_rep_5",
        "acc_rep_10",
        "acc_rep_15",
        "epoch_auc",
        "epoch_bacc",
    ] + [f"acc_rep_{rep}" for rep in range(1, FULL_REPEAT + 1)]
    for col in numeric_cols:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")

    summary_rows = []
    for col in numeric_cols:
        values = ok[col].dropna().to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": col,
                "n": values.size,
                "mean": np.nanmean(values),
                "sem": sem(values, nan_policy="omit") if values.size > 1 else np.nan,
                "median": np.nanmedian(values),
                "q25": np.nanpercentile(values, 25),
                "q75": np.nanpercentile(values, 75),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "raw_fp12_baseline_group_summary.csv", index=False)

    reps = np.arange(1, FULL_REPEAT + 1)
    mat = ok[[f"acc_rep_{rep}" for rep in reps]].to_numpy(dtype=float)
    mean = np.nanmean(mat, axis=0)
    n = np.sum(np.isfinite(mat), axis=0)
    err = np.zeros_like(mean)
    valid = n > 1
    err[valid] = np.nanstd(mat[:, valid], axis=0, ddof=1) / np.sqrt(n[valid])

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(reps, mean, marker="o", linewidth=1.8, color="#1f77b4")
    ax.fill_between(reps, mean - err, mean + err, color="#1f77b4", alpha=0.18)
    ax.set_xlabel("Repetition")
    ax.set_ylabel("Letter accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(reps)
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_title(f"Raw FP1/FP2 baseline SWLDA (n={len(ok)})")
    fig.tight_layout()
    fig.savefig(out_dir / "raw_fp12_baseline_accuracy_curve_mean_sem.png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run raw FP1/FP2 P300 SWLDA baseline without denoising or H5 preprocessing.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--n-subjects", type=int, default=None, help="Default is all discovered subjects.")
    parser.add_argument("--start-subject", type=int, default=1)
    parser.add_argument("--channels", nargs="+", default=["FP1", "FP2"])
    parser.add_argument("--decim", type=int, default=24)
    parser.add_argument("--p-enter", type=float, default=0.08)
    parser.add_argument("--p-remove", type=float, default=0.15)
    parser.add_argument("--max-features", type=int, default=60)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    subjects = selected_subjects(args.dataset_root, args.subjects, args.n_subjects, args.start_subject)
    if not subjects:
        raise ValueError("No subjects selected.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Raw FP1/FP2 baseline SWLDA")
    print(f"Dataset: {args.dataset_root}")
    print(f"Channels: {args.channels}")
    print(f"Subjects: {len(subjects)}")
    print(", ".join(subjects))

    rows: list[dict[str, object]] = []
    all_ok = True
    for idx, subject in enumerate(subjects, start=1):
        print(f"\n[{idx}/{len(subjects)}] {subject}")
        result = decode_subject(
            dataset_root=args.dataset_root,
            subject=subject,
            channels=args.channels,
            decim=args.decim,
            p_enter=args.p_enter,
            p_remove=args.p_remove,
            max_features=args.max_features,
        )
        row = result_to_row(result, args.channels)
        rows.append(row)
        subject_csv = args.out_dir / f"{subject}_raw_fp12_baseline_summary.csv"
        write_subject_rows(subject_csv, [row])

        if result.status == "ok":
            print(
                f"  ok | train={result.n_train_epochs} test={result.n_test_epochs} "
                f"letters={result.total_test_letters} features={result.n_features} "
                f"epoch_auc={result.epoch_auc:.3f} "
                f"L5={result.repetition_accuracy[4]:.3f} "
                f"L10={result.repetition_accuracy[9]:.3f} "
                f"L15={result.repetition_accuracy[14]:.3f} "
                f"elapsed={result.elapsed_sec:.1f}s"
            )
        else:
            all_ok = False
            print(f"  failed | {result.message}")
            if not args.keep_going:
                break

    all_csv = args.out_dir / "raw_fp12_baseline_subject_summary.csv"
    write_subject_rows(all_csv, rows)
    write_group_summary(rows, args.out_dir)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    print(f"\nCompleted subjects: {len(ok_rows)}/{len(rows)}")
    if ok_rows:
        df = pd.DataFrame(ok_rows)
        for col in ["acc_rep_5", "acc_rep_10", "acc_rep_15", "auc_repetition_accuracy"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        print(
            "Group means: "
            f"L5={df['acc_rep_5'].mean():.3f}, "
            f"L10={df['acc_rep_10'].mean():.3f}, "
            f"L15={df['acc_rep_15'].mean():.3f}, "
            f"AUC-rep={df['auc_repetition_accuracy'].mean():.3f}"
        )
    print(f"Subject summary: {all_csv}")
    print(f"Group summary: {args.out_dir / 'raw_fp12_baseline_group_summary.csv'}")
    print(f"Curve figure: {args.out_dir / 'raw_fp12_baseline_accuracy_curve_mean_sem.png'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
