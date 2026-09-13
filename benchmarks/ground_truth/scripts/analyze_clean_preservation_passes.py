from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from analyze_clean_preservation import (
    compute_spectral_metrics,
    compute_waveform_metrics,
    ensure_2d_rows,
    latex_escape,
    natural_clean_key,
)


PASS_CONFIGS = [
    ("EMG", "proposed_emg"),
    ("EOG", "proposed_eog"),
    ("Slow", "proposed_slow"),
    ("EMG+EOG", "proposed_emg_eog"),
    ("Full", "proposed_full"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare clean-EEG preservation across proposed pass configurations."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("results/clean_preservation/data/04_denoise-net_clean.h5"),
    )
    parser.add_argument(
        "--restored-root",
        type=Path,
        default=Path("results/clean_preservation/restored"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/clean_preservation/analysis_passes"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pass_label, restored_folder in PASS_CONFIGS:
        restored_file = args.restored_root / restored_folder / args.data_file.name
        if not restored_file.exists():
            raise FileNotFoundError(f"Missing restored file: {restored_file}")
        rows.extend(analyze_one_pass(args.data_file, restored_file, pass_label))

    df = pd.DataFrame(rows)
    record_path = args.out_dir / "clean_preservation_pass_record_metrics.csv"
    df.to_csv(record_path, index=False)

    compact = make_compact_table(df)
    compact_path = args.out_dir / "clean_preservation_pass_compact_table.csv"
    compact.to_csv(compact_path, index=False)

    latex_path = args.out_dir / "clean_preservation_pass_compact_table.tex"
    latex_path.write_text(compact_to_latex(compact), encoding="utf-8")

    print(f"Saved {record_path}")
    print(f"Saved {compact_path}")
    print(f"Saved {latex_path}")
    print("\nClean preservation by pass:")
    print(compact.to_string(index=False))


def analyze_one_pass(
    data_file: Path, restored_file: Path, pass_label: str
) -> list[dict[str, float | str | int]]:
    rows = []
    with h5py.File(data_file, "r") as data_h5, h5py.File(restored_file, "r") as restored_h5:
        for record_name in sorted(data_h5.keys(), key=natural_clean_key):
            group = data_h5[record_name]
            reference = ensure_2d_rows(np.asarray(group["eeg_reference"][()]))
            restored = ensure_2d_rows(np.asarray(restored_h5[record_name][()]))
            if restored.shape != reference.shape and restored.T.shape == reference.shape:
                restored = restored.T
            if restored.shape != reference.shape:
                raise RuntimeError(
                    f"{pass_label}/{record_name}: restored shape {restored.shape} "
                    f"does not match reference shape {reference.shape}."
                )

            fs = float(group.attrs.get("freq", 256.0))
            for channel_idx in range(reference.shape[0]):
                x = np.asarray(reference[channel_idx], dtype=float)
                y = np.asarray(restored[channel_idx], dtype=float)
                row = {
                    "pass_config": pass_label,
                    "record": record_name,
                    "channel": channel_idx,
                    "fs": fs,
                }
                row.update(compute_waveform_metrics(x, y))
                row.update(compute_spectral_metrics(x, y, fs))
                rows.append(row)
    return rows


def make_compact_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pass_label, _ in PASS_CONFIGS:
        sub = df[df["pass_config"] == pass_label]
        rows.append(
            {
                "Pass": pass_label,
                "Epochs": int(sub["record"].nunique()),
                "RRMSE (%)": median_iqr(sub["rrmse_percent"], "%"),
                "Preservation SNR (dB)": median_iqr(sub["preservation_snr_db"], "dB"),
                "Pearson r": median_iqr(sub["pearson_r"], ""),
                "PSD change 1--45 Hz (dB)": median_iqr(
                    sub["mean_abs_psd_change_db_1_45"], "dB"
                ),
                "Alpha change (dB)": median_iqr(sub["alpha_power_change_db"], "dB"),
                "Beta change (dB)": median_iqr(sub["beta_power_change_db"], "dB"),
            }
        )
    return pd.DataFrame(rows)


def median_iqr(values: pd.Series, unit: str) -> str:
    raw = pd.to_numeric(values, errors="coerce")
    clean = raw.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return "NA"
    median = empirical_quantile_with_inf(clean, 0.50)
    q25 = empirical_quantile_with_inf(clean, 0.25)
    q75 = empirical_quantile_with_inf(clean, 0.75)
    if np.isposinf(median) and np.isposinf(q25) and np.isposinf(q75):
        return "Inf [Inf, Inf]"
    if np.isneginf(median) and np.isneginf(q25) and np.isneginf(q75):
        return "-Inf [-Inf, -Inf]"
    if unit == "%":
        return f"{median:.2f} [{q25:.2f}, {q75:.2f}]"
    if unit == "dB":
        return f"{median:.2f} [{q25:.2f}, {q75:.2f}]"
    return f"{median:.4f} [{q25:.4f}, {q75:.4f}]"


def empirical_quantile_with_inf(values: np.ndarray, q: float) -> float:
    """Linear empirical quantile with explicit handling of infinite endpoints."""
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size == 0:
        return np.nan
    position = (ordered.size - 1) * q
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    lo = ordered[lower]
    hi = ordered[upper]
    if lo == hi:
        return float(lo)
    fraction = position - lower
    if np.isneginf(lo) or np.isposinf(hi):
        return float(lo if fraction == 0 else hi)
    return float(lo + fraction * (hi - lo))


def compact_to_latex(compact: pd.DataFrame) -> str:
    lines = [
        "\\begin{table*}[!t]",
        "\\caption{Clean-EEG preservation diagnostic by proposed pass configuration. "
        "Each configuration was applied to 3400 clean EEGdenoiseNet epochs without "
        "added artifact. Values are medians [IQR] across epochs.}",
        "\\label{tab:clean_preservation_passes}",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Pass & RRMSE (\\%) & Preservation SNR (dB) & Pearson $r$ & "
        "PSD change (dB) & Alpha change (dB) & Beta change (dB) \\\\",
        "\\midrule",
    ]
    for _, row in compact.iterrows():
        lines.append(
            f"{latex_escape(str(row['Pass']))} & "
            f"{latex_escape(str(row['RRMSE (%)']))} & "
            f"{latex_escape(str(row['Preservation SNR (dB)']))} & "
            f"{latex_escape(str(row['Pearson r']))} & "
            f"{latex_escape(str(row['PSD change 1--45 Hz (dB)']))} & "
            f"{latex_escape(str(row['Alpha change (dB)']))} & "
            f"{latex_escape(str(row['Beta change (dB)']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
