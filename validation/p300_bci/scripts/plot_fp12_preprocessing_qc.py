from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRANCHES = (
    PROJECT_ROOT
    / "outputs"
    / "preprocessed_fp12"
    / "sub-001_task-P300trainrun1_run-6_fp12_branches.h5"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "fp12_preprocessing_qc"


def infer_wide_path(branch_path: Path) -> Path:
    name = branch_path.name.replace("_branches.h5", "_wide.h5")
    return branch_path.with_name(name)


def read_channels(branch_path: Path) -> list[str]:
    wide_path = infer_wide_path(branch_path)
    if wide_path.is_file():
        with h5py.File(wide_path, "r") as h5:
            if "wide/channels" in h5:
                return [
                    label.decode("utf-8") if isinstance(label, bytes) else str(label)
                    for label in h5["wide/channels"][:]
                ]
    return ["FP1", "FP2"]


def scalar_attr(value, default: float) -> float:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def moving_ptp_score(x: np.ndarray, fs: float, win_sec: float = 2.0, step_sec: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    win = max(1, int(round(win_sec * fs)))
    step = max(1, int(round(step_sec * fs)))
    starts = np.arange(0, max(1, x.shape[1] - win + 1), step)
    scores = np.zeros(len(starts), dtype=float)
    for i, start in enumerate(starts):
        seg = x[:, start : start + win]
        scores[i] = float(np.max(np.ptp(seg, axis=1)))
    return starts, scores


def select_windows(
    baseline: np.ndarray,
    fs: float,
    count: int,
    window_sec: float,
    min_gap_sec: float,
) -> list[int]:
    score_starts, scores = moving_ptp_score(baseline, fs)
    order = np.argsort(scores)[::-1]
    selected: list[int] = []
    gap = int(round(min_gap_sec * fs))
    half_window = int(round(window_sec * fs / 2))
    n = baseline.shape[1]

    for idx in order:
        center = int(score_starts[idx] + fs)
        start = max(0, min(n - int(round(window_sec * fs)), center - half_window))
        if all(abs(start - prev) >= gap for prev in selected):
            selected.append(start)
        if len(selected) >= count:
            break

    if not selected:
        selected = [0]
    return sorted(selected)


def robust_ylim(arrays: list[np.ndarray], pad_frac: float = 0.08) -> tuple[float, float]:
    values = np.concatenate([a.reshape(-1) for a in arrays])
    lo, hi = np.percentile(values, [1, 99])
    pad = max(1.0, (hi - lo) * pad_frac)
    return float(lo - pad), float(hi + pad)


def plot_window(
    branch_path: Path,
    data: dict[str, np.ndarray],
    fs: float,
    channels: list[str],
    start: int,
    window_sec: float,
    out_path: Path,
) -> None:
    n = data["baseline"].shape[1]
    stop = min(n, start + int(round(window_sec * fs)))
    t = np.arange(start, stop) / fs
    t = t - t[0]

    colors = {
        "baseline": "#111111",
        "eog": "#1f77b4",
        "emg_eog": "#d62728",
        "full": "#2ca02c",
    }
    labels = {
        "baseline": "Baseline wide",
        "eog": "EOG",
        "emg_eog": "EMG+EOG",
        "full": "Full",
    }

    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 5.8), sharex=True)
    if len(channels) == 1:
        axes = [axes]

    for ch_idx, ax in enumerate(axes):
        segment_arrays = [branch[ch_idx, start:stop] for branch in data.values()]
        ax.set_ylim(*robust_ylim(segment_arrays))
        for name, branch in data.items():
            ax.plot(t, branch[ch_idx, start:stop], lw=1.0, color=colors[name], label=labels[name], alpha=0.95)
        ax.set_ylabel(f"{channels[ch_idx]} (uV)")
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
        if ch_idx == 0:
            base = data["baseline"][:, start:stop]
            branch_stats = []
            for name in ["eog", "emg_eog", "full"]:
                delta = np.sqrt(np.mean((data[name][:, start:stop] - base) ** 2))
                branch_stats.append(f"{labels[name]} RMSΔ={delta:.2f} uV")
            ax.set_title(
                f"{branch_path.stem}: {start / fs:.1f}-{stop / fs:.1f} s | "
                + " | ".join(branch_stats)
            )
            ax.legend(loc="upper right", ncol=4, fontsize=8, frameon=False)

    axes[-1].set_xlabel("Time within window (s)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overview(
    branch_path: Path,
    data: dict[str, np.ndarray],
    fs: float,
    channels: list[str],
    out_path: Path,
) -> None:
    n = data["baseline"].shape[1]
    step = max(1, int(round(fs / 8)))
    t = np.arange(0, n, step) / fs

    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 5.8), sharex=True)
    if len(channels) == 1:
        axes = [axes]

    for ch_idx, ax in enumerate(axes):
        ax.plot(t, data["baseline"][ch_idx, ::step], color="#111111", lw=0.8, label="Baseline wide")
        ax.plot(t, data["eog"][ch_idx, ::step], color="#1f77b4", lw=0.8, alpha=0.9, label="EOG")
        ax.plot(t, data["emg_eog"][ch_idx, ::step], color="#d62728", lw=0.8, alpha=0.9, label="EMG+EOG")
        ax.plot(t, data["full"][ch_idx, ::step], color="#2ca02c", lw=0.8, alpha=0.9, label="Full")
        ax.set_ylabel(f"{channels[ch_idx]} (uV)")
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
        ax.set_ylim(*robust_ylim([branch[ch_idx, ::step] for branch in data.values()]))
        if ch_idx == 0:
            ax.set_title(f"{branch_path.stem}: full-run downsampled overview")
            ax.legend(loc="upper right", ncol=4, fontsize=8, frameon=False)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create FP1/FP2 before-after preprocessing QC time-series figures.")
    parser.add_argument("--branches-h5", type=Path, default=DEFAULT_BRANCHES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--n-windows", type=int, default=3)
    parser.add_argument("--min-gap-sec", type=float, default=20.0)
    args = parser.parse_args()

    with h5py.File(args.branches_h5, "r") as h5:
        fs = scalar_attr(h5.attrs.get("srate"), 512.0)
        data = {
            "baseline": h5["baseline/data"][:],
            "eog": h5["eog/data"][:],
            "emg_eog": h5["emg_eog/data"][:],
            "full": h5["full/data"][:],
        }

    channels = read_channels(args.branches_h5)
    starts = select_windows(
        data["baseline"],
        fs=fs,
        count=args.n_windows,
        window_sec=args.window_sec,
        min_gap_sec=args.min_gap_sec,
    )

    stem = args.branches_h5.stem
    overview_path = args.out_dir / f"{stem}_overview.png"
    plot_overview(args.branches_h5, data, fs, channels, overview_path)
    print(f"Saved overview: {overview_path}")

    for i, start in enumerate(starts, start=1):
        out_path = args.out_dir / f"{stem}_window{i}_{start / fs:.1f}s.png"
        plot_window(args.branches_h5, data, fs, channels, start, args.window_sec, out_path)
        print(f"Saved window {i}: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
