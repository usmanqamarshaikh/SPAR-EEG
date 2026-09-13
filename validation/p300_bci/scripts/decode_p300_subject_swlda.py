from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LinearRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "preprocessed_fp12"
    / "manifests"
    / "sub-001_p300_preprocessing_manifest.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "swlda_decoder"

SPELLER_MATRIX = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_")
SEQ_CODES = np.arange(1, 13)
FULL_REPEAT = 15


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
class SwldaModel:
    selected_features: np.ndarray
    regressor: LinearRegression
    train_binary_accuracy: float
    score_direction: float


def decode_attr(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    if arr.dtype.kind in {"S", "U", "O"}:
        return "".join(
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in arr.reshape(-1)
        )
    return str(value)


def load_manifest(path: Path) -> list[RunFiles]:
    runs: list[RunFiles] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            runs.append(
                RunFiles(
                    subject=row["subject"],
                    task=row["task"],
                    run=int(row["run"]),
                    wide_h5=Path(row["wide_h5"]),
                    branches_h5=Path(row["branches_h5"]),
                )
            )
    return sorted(runs, key=lambda item: item.run)


def butter_bandpass_official(data: np.ndarray, fs: float, low_hz: float = 0.5, high_hz: float = 10.0) -> np.ndarray:
    sos = butter(4, [low_hz / (fs / 2), high_hz / (fs / 2)], btype="band", output="sos")
    demeaned = data - np.mean(data, axis=1, keepdims=True)
    return sosfiltfilt(sos, demeaned, axis=1)


def load_branch_data(run: RunFiles, branch: str) -> tuple[np.ndarray, dict[str, Any]]:
    with h5py.File(run.branches_h5, "r") as h5:
        data = np.asarray(h5[f"{branch}/data"][:], dtype=np.float64)
        attrs = dict(h5.attrs.items())
    return data, attrs


def load_event_info(run: RunFiles) -> tuple[np.ndarray, np.ndarray, float, str]:
    with h5py.File(run.wide_h5, "r") as h5:
        event_sequence = np.asarray(h5["events/event_sequence"][:], dtype=np.int16)
        event_target = np.asarray(h5["events/event_target"][:], dtype=np.int16)
        fs = float(np.asarray(h5["wide/srate"][()]).reshape(-1)[0])
        text_to_spell = decode_attr(h5.attrs.get("text_to_spell", ""))
    return event_sequence, event_target, fs, text_to_spell


def event_indices(event_sequence: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(event_sequence, SEQ_CODES))


def extract_epochs(
    data: np.ndarray,
    event_idx: np.ndarray,
    fs: float,
    frame_ms: tuple[float, float] = (0.0, 600.0),
    baseline_ms: tuple[float, float] = (-200.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    frame_start = int(np.floor(frame_ms[0] / 1000.0 * fs))
    frame_stop = int(np.floor(frame_ms[1] / 1000.0 * fs))
    base_start = int(np.floor(baseline_ms[0] / 1000.0 * fs))
    base_stop = int(np.floor(baseline_ms[1] / 1000.0 * fs))
    n_frame = frame_stop - frame_start

    epochs: list[np.ndarray] = []
    kept: list[int] = []
    n = data.shape[1]
    for idx in event_idx:
        f0 = int(idx + frame_start)
        f1 = int(idx + frame_stop)
        b0 = int(idx + base_start)
        b1 = int(idx + base_stop)
        if b0 < 0 or f0 < 0 or b1 <= b0 or f1 > n:
            continue
        epoch = data[:, f0:f1]
        if epoch.shape[1] != n_frame:
            continue
        baseline = np.mean(data[:, b0:b1], axis=1, keepdims=True)
        epochs.append(epoch - baseline)
        kept.append(int(idx))

    if not epochs:
        raise ValueError("No valid epochs extracted.")
    return np.stack(epochs, axis=2), np.asarray(kept, dtype=int)


def decimate_by_average(epochs: np.ndarray, factor: int = 24) -> np.ndarray:
    n_ch, n_frame, n_trial = epochs.shape
    n_dec = int(np.floor(n_frame / factor))
    trimmed = epochs[:, : n_dec * factor, :]
    return trimmed.reshape(n_ch, n_dec, factor, n_trial).mean(axis=2)


def epochs_to_features(epochs: np.ndarray) -> np.ndarray:
    down = decimate_by_average(epochs, factor=24)
    return np.asarray([down[:, :, trial].reshape(-1) for trial in range(down.shape[2])], dtype=np.float64)


def build_training_features(runs: list[RunFiles], branch: str) -> tuple[np.ndarray, np.ndarray]:
    feat_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for run in runs:
        data, _ = load_branch_data(run, branch)
        event_sequence, event_target, fs, _ = load_event_info(run)
        filtered = butter_bandpass_official(data, fs)
        idx_all = event_indices(event_sequence)

        target_idx = idx_all[event_target[idx_all] == 1]
        nontarget_idx = idx_all[event_target[idx_all] == 2]
        target_epochs, _ = extract_epochs(filtered, target_idx, fs)
        nontarget_epochs, _ = extract_epochs(filtered, nontarget_idx, fs)

        feat_parts.append(epochs_to_features(target_epochs))
        label_parts.append(np.ones(feat_parts[-1].shape[0], dtype=np.float64))
        feat_parts.append(epochs_to_features(nontarget_epochs))
        label_parts.append(-np.ones(feat_parts[-1].shape[0], dtype=np.float64))

    X = np.vstack(feat_parts)
    y = np.concatenate(label_parts)
    rng = np.random.default_rng(101)
    order = np.arange(X.shape[0])
    rng.shuffle(order)
    return X[order], y[order]


def top_correlated_features(X: np.ndarray, y: np.ndarray, max_features: int) -> np.ndarray:
    y0 = y - np.mean(y)
    X0 = X - np.mean(X, axis=0, keepdims=True)
    denom = np.sqrt(np.sum(X0**2, axis=0) * np.sum(y0**2))
    corr = np.divide(np.sum(X0 * y0[:, None], axis=0), denom, out=np.zeros(X.shape[1]), where=denom > 0)
    return np.argsort(np.abs(corr))[::-1][: min(max_features, X.shape[1])]


def train_swlda_stepwise_python(
    X: np.ndarray,
    y: np.ndarray,
    p_enter: float = 0.08,
    p_remove: float = 0.15,
    max_features: int = 60,
) -> SwldaModel:
    selected: list[int] = []
    remaining = set(range(X.shape[1]))

    while remaining and len(selected) < max_features:
        best_feature = None
        best_p = np.inf
        for feature in sorted(remaining):
            cols = selected + [feature]
            result = sm.OLS(y, sm.add_constant(X[:, cols], has_constant="add")).fit()
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
            result = sm.OLS(y, sm.add_constant(X[:, selected], has_constant="add")).fit()
            pvalues = np.asarray(result.pvalues[1:], dtype=np.float64)
            pvalues = np.where(np.isfinite(pvalues), pvalues, 1.0)
            worst = int(np.argmax(pvalues))
            if pvalues[worst] > p_remove:
                removed = selected.pop(worst)
                remaining.add(removed)
                changed = True

    if not selected:
        selected_array = top_correlated_features(X, y, max_features)
    else:
        selected_array = np.asarray(selected[:max_features], dtype=int)

    regressor = LinearRegression()
    regressor.fit(X[:, selected_array], y)
    train_scores = regressor.predict(X[:, selected_array])
    direction = 1.0
    if np.mean(train_scores[y == 1]) < np.mean(train_scores[y == -1]):
        direction = -1.0
    pred = np.sign(direction * train_scores)
    pred[pred == 0] = 1
    acc = float(np.mean(pred == y))
    return SwldaModel(
        selected_features=selected_array,
        regressor=regressor,
        train_binary_accuracy=acc,
        score_direction=direction,
    )


def score_test_run(run: RunFiles, branch: str, model: SwldaModel) -> tuple[np.ndarray, np.ndarray, str, str, int]:
    data, _ = load_branch_data(run, branch)
    event_sequence, event_target, fs, text = load_event_info(run)
    filtered = butter_bandpass_official(data, fs)
    idx_all = event_indices(event_sequence)
    epochs, kept_idx = extract_epochs(filtered, idx_all, fs)
    features = epochs_to_features(epochs)
    scores = model.score_direction * model.regressor.predict(features[:, model.selected_features])
    seq = event_sequence[kept_idx].astype(int)
    y01 = (event_target[kept_idx] == 1).astype(int)
    answer, acc, n_letters = detect_letters_from_events(scores, seq, y01)
    return acc, scores, text, answer, n_letters


def detect_letters_from_events(scores: np.ndarray, seq: np.ndarray, y01: np.ndarray) -> tuple[str, np.ndarray, int]:
    chunk_len = FULL_REPEAT * len(SEQ_CODES)
    n_letters = len(scores) // chunk_len
    if n_letters < 1:
        raise ValueError(f"Not enough test scores for one letter: got {len(scores)}, expected at least {chunk_len}")
    acc = np.zeros(FULL_REPEAT, dtype=np.float64)
    final_chars: list[str] = []
    valid_letters = 0

    for letter in range(n_letters):
        begin = chunk_len * letter
        stop = begin + chunk_len
        scores_l = scores[begin:stop]
        seq_l = seq[begin:stop]
        y_l = y01[begin:stop]
        target_codes = seq_l[y_l == 1]
        rows = target_codes[(target_codes >= 1) & (target_codes <= 6)]
        cols = target_codes[(target_codes >= 7) & (target_codes <= 12)]
        if rows.size == 0 or cols.size == 0:
            continue
        true_row = int(np.bincount(rows, minlength=13).argmax())
        true_col = int(np.bincount(cols, minlength=13).argmax())
        valid_letters += 1

        for repeat in range(FULL_REPEAT):
            code_scores = np.zeros(len(SEQ_CODES), dtype=np.float64)
            for j in range((repeat + 1) * len(SEQ_CODES)):
                code = int(seq_l[j])
                if 1 <= code <= 12:
                    code_scores[code - 1] += scores_l[j]
            pred_row = int(np.argmax(code_scores[:6]) + 1)
            pred_col = int(np.argmax(code_scores[6:12]) + 7)
            acc[repeat] += float(pred_row == true_row and pred_col == true_col)
            if repeat == FULL_REPEAT - 1:
                final_chars.append(SPELLER_MATRIX[(pred_row - 1) * 6 + (pred_col - 7)])

    if valid_letters == 0:
        raise ValueError("No valid letter chunks with event-derived target row/column.")
    return "".join(final_chars), acc / valid_letters, valid_letters


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for row in summary_rows:
        reps = np.arange(1, FULL_REPEAT + 1)
        acc = np.asarray([float(row[f"acc_rep_{rep}"]) for rep in reps])
        ax.plot(reps, acc, marker="o", linewidth=1.5, label=row["branch"])
    ax.set_xlabel("Repetition")
    ax.set_ylabel("Letter accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(np.arange(1, FULL_REPEAT + 1))
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official-style SWLDA P300 decoding for one preprocessed subject.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--branches", nargs="+", default=["baseline", "eog", "emg_eog", "full"])
    args = parser.parse_args()

    runs = load_manifest(args.manifest)
    train_runs = [run for run in runs if run.is_train]
    test_runs = [run for run in runs if run.is_test]
    if not train_runs or not test_runs:
        raise ValueError("Need both train and test P300 runs in manifest.")

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    print(f"Training runs: {[run.run for run in train_runs]}")
    print(f"Test runs: {[run.run for run in test_runs]}")

    for branch in args.branches:
        print(f"\nBranch: {branch}")
        X_train, y_train = build_training_features(train_runs, branch)
        model = train_swlda_stepwise_python(X_train, y_train)
        print(
            f"  train shape={X_train.shape}, selected_features={model.selected_features.size}, "
            f"binary_train_acc={model.train_binary_accuracy:.3f}, "
            f"score_direction={model.score_direction:+.0f}"
        )

        run_accs = []
        run_correct = []
        run_word_lengths = []
        for run in test_runs:
            acc, _scores, target, answer, n_letters = score_test_run(run, branch, model)
            run_accs.append(acc)
            run_word_lengths.append(n_letters)
            run_correct.append(acc * n_letters)
            print(f"  run-{run.run:02d} target={target} answer={answer} acc15={acc[-1]:.3f}")
            for rep in range(1, FULL_REPEAT + 1):
                detail_rows.append(
                    {
                        "branch": branch,
                        "subject": run.subject,
                        "task": run.task,
                        "run": run.run,
                        "target_text": target,
                        "answer_rep15": answer,
                        "repetition": rep,
                        "accuracy": f"{acc[rep - 1]:.6f}",
                        "correct": f"{acc[rep - 1] * n_letters:.6f}",
                        "word_len": n_letters,
                        "train_binary_accuracy": f"{model.train_binary_accuracy:.6f}",
                        "selected_feature_count": model.selected_features.size,
                        "score_direction": f"{model.score_direction:.0f}",
                    }
                )

        correct_by_rep = np.sum(np.vstack(run_correct), axis=0)
        total_letters = int(np.sum(run_word_lengths))
        aggregate_acc = correct_by_rep / total_letters
        summary = {
            "branch": branch,
            "subject": train_runs[0].subject,
            "train_runs": ",".join(str(run.run) for run in train_runs),
            "test_runs": ",".join(str(run.run) for run in test_runs),
            "total_test_letters": total_letters,
            "train_binary_accuracy": f"{model.train_binary_accuracy:.6f}",
            "selected_feature_count": model.selected_features.size,
            "score_direction": f"{model.score_direction:.0f}",
            "auc_repetition_accuracy": f"{float(np.mean(aggregate_acc)):.6f}",
            "acc_rep_5": f"{aggregate_acc[4]:.6f}",
            "acc_rep_10": f"{aggregate_acc[9]:.6f}",
            "acc_rep_15": f"{aggregate_acc[14]:.6f}",
        }
        for rep in range(1, FULL_REPEAT + 1):
            summary[f"acc_rep_{rep}"] = f"{aggregate_acc[rep - 1]:.6f}"
            summary[f"correct_rep_{rep}"] = f"{correct_by_rep[rep - 1]:.6f}"
        summary_rows.append(summary)

    subject = runs[0].subject
    detail_csv = args.out_dir / f"{subject}_swlda_repetition_detail.csv"
    summary_csv = args.out_dir / f"{subject}_swlda_summary.csv"
    curve_png = args.out_dir / "figures" / f"{subject}_swlda_accuracy_curves.png"
    write_csv(detail_csv, detail_rows)
    write_csv(summary_csv, summary_rows)
    plot_curves(summary_rows, curve_png)

    print(f"\nDetail CSV: {detail_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Curve figure: {curve_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
