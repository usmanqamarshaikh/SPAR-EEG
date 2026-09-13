from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FILES = [
    "03_denoise-net_emg_-20dB.h5",
    "03_denoise-net_emg_0dB.h5",
    "03_denoise-net_eog_-20dB.h5",
    "03_denoise-net_eog_0dB.h5",
    "03_denoise-net_eog+emg_-20dB.h5",
    "03_denoise-net_eog+emg_0dB.h5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot example generated H5 samples with artifact masks and input SNR labels."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/dataset_examples"))
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    for file_name in args.files:
        path = args.data_dir / file_name
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        examples.append(load_example(path, args.record_index))

    if not examples:
        raise SystemExit("No examples could be loaded.")

    save_individual_figures(examples, args.out_dir)
    save_grid_figure(examples, args.out_dir / "generated_denoisenet_examples_grid.png")
    save_grid_figure(examples, args.out_dir / "generated_denoisenet_examples_grid.pdf")

    print(f"Saved {len(examples)} example figures to {args.out_dir}")


def load_example(path: Path, record_index: int) -> dict[str, object]:
    with h5py.File(path, "r") as h5:
        keys = sorted(h5.keys(), key=natural_key)
        key = keys[min(record_index, len(keys) - 1)]
        group = h5[key]

        signal = as_channel_first(group["eeg_signal"][()])[0]
        reference = as_channel_first(group["eeg_reference"][()])[0]
        mask = np.asarray(group["artifacts"][()]).astype(bool).reshape(-1)
        fs = float(group.attrs.get("freq", h5.attrs.get("freq", 256.0)))

        nominal = group.attrs.get("nominal_snr_db", h5.attrs.get("nominal_snr_db", np.nan))
        measured = group.attrs.get("measured_snr_db", np.nan)
        if not np.isfinite(float_or_nan(measured)):
            measured = calculate_snr(reference[mask], signal[mask] - reference[mask])

        return {
            "file": path.name,
            "record": key,
            "dataset_name": h5.attrs.get("name", path.stem),
            "signal": signal,
            "reference": reference,
            "artifact": signal - reference,
            "mask": mask,
            "fs": fs,
            "nominal_snr_db": float_or_nan(nominal),
            "measured_snr_db": float_or_nan(measured),
            "artifact_layout": h5.attrs.get("artifact_layout", ""),
        }


def save_individual_figures(examples: list[dict[str, object]], out_dir: Path) -> None:
    for ex in examples:
        fig, ax = plt.subplots(figsize=(8.0, 3.2))
        plot_example(ax, ex)
        fig.tight_layout()
        stem = sanitize(str(ex["file"]).replace(".h5", ""))
        fig.savefig(out_dir / f"{stem}_{sanitize(str(ex['record']))}.png", dpi=300)
        fig.savefig(out_dir / f"{stem}_{sanitize(str(ex['record']))}.pdf")
        plt.close(fig)


def save_grid_figure(examples: list[dict[str, object]], out_path: Path) -> None:
    n = len(examples)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 3.0 * nrows), squeeze=False)

    for ax, ex in zip(axes.ravel(), examples):
        plot_example(ax, ex)
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle("Generated DenoiseNet Examples With Centered Artifact Masks", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300 if out_path.suffix.lower() == ".png" else None)
    plt.close(fig)


def plot_example(ax: plt.Axes, ex: dict[str, object]) -> None:
    signal = np.asarray(ex["signal"], dtype=float)
    reference = np.asarray(ex["reference"], dtype=float)
    artifact = np.asarray(ex["artifact"], dtype=float)
    mask = np.asarray(ex["mask"], dtype=bool)
    fs = float(ex["fs"])
    t = np.arange(signal.size) / fs

    shade_mask(ax, t, mask)
    ax.plot(t, reference, color="#1f77b4", linewidth=1.1, label="Clean EEG")
    ax.plot(t, signal, color="#d62728", linewidth=1.0, alpha=0.82, label="Noisy input")
    ax.plot(t, artifact, color="#444444", linewidth=0.9, alpha=0.55, label="Injected artifact")

    nominal = ex["nominal_snr_db"]
    measured = ex["measured_snr_db"]
    snr_text = f"Input SNR: {nominal:.1f} dB"
    if np.isfinite(measured):
        snr_text += f" | measured: {measured:.2f} dB"

    title = f"{str(ex['dataset_name'])} | {ex['record']}"
    ax.set_title(title, fontsize=9)
    ax.text(
        0.01,
        0.95,
        snr_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.88, "pad": 3},
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.grid(alpha=0.22)
    ax.set_xlim(t[0], t[-1])
    set_robust_ylim(ax, np.r_[signal, reference])
    ax.legend(loc="lower right", fontsize=7, frameon=False, ncol=3)


def shade_mask(ax: plt.Axes, t: np.ndarray, mask: np.ndarray) -> None:
    for start, end in mask_runs(mask):
        ax.axvspan(t[start], t[end - 1], color="#f2c14e", alpha=0.28, linewidth=0)


def mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def as_channel_first(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    sig_var = float(np.var(signal))
    noise_var = float(np.var(noise))
    if sig_var <= np.finfo(float).eps or noise_var <= np.finfo(float).eps:
        return np.nan
    return float(10.0 * np.log10(sig_var / noise_var))


def set_robust_ylim(ax: plt.Axes, values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    lo, hi = np.percentile(finite, [1, 99])
    pad = 0.15 * max(hi - lo, np.finfo(float).eps)
    ax.set_ylim(lo - pad, hi + pad)


def natural_key(text: str) -> tuple[object, ...]:
    parts: list[object] = []
    cur = ""
    is_digit = False
    for ch in text:
        if ch.isdigit() == is_digit:
            cur += ch
        else:
            if cur:
                parts.append(int(cur) if is_digit else cur)
            cur = ch
            is_digit = ch.isdigit()
    if cur:
        parts.append(int(cur) if is_digit else cur)
    return tuple(parts)


def float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


if __name__ == "__main__":
    main()
