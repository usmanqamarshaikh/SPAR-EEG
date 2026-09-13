from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.signal as ss


BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma45": (30.0, 45.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze clean-epoch preservation after proposed denoising."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("results/clean_preservation/data/04_denoise-net_clean.h5"),
    )
    parser.add_argument(
        "--restored-file",
        type=Path,
        default=Path(
            "results/clean_preservation/restored/proposed_full/04_denoise-net_clean.h5"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/clean_preservation/analysis"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with h5py.File(args.data_file, "r") as data_h5, h5py.File(
        args.restored_file, "r"
    ) as restored_h5:
        record_names = sorted(data_h5.keys(), key=natural_clean_key)
        for record_name in record_names:
            group = data_h5[record_name]
            reference = ensure_2d_rows(np.asarray(group["eeg_reference"][()]))
            restored = ensure_2d_rows(np.asarray(restored_h5[record_name][()]))
            if restored.shape != reference.shape and restored.T.shape == reference.shape:
                restored = restored.T
            if restored.shape != reference.shape:
                raise RuntimeError(
                    f"{record_name}: restored shape {restored.shape} does not match "
                    f"reference shape {reference.shape}."
                )

            fs = float(group.attrs.get("freq", 256.0))
            for channel_idx in range(reference.shape[0]):
                x = np.asarray(reference[channel_idx], dtype=float)
                y = np.asarray(restored[channel_idx], dtype=float)
                rows.append(
                    {
                        "record": record_name,
                        "channel": channel_idx,
                        "fs": fs,
                        **compute_waveform_metrics(x, y),
                        **compute_spectral_metrics(x, y, fs),
                    }
                )

    df = pd.DataFrame(rows)
    records_path = args.out_dir / "clean_preservation_record_metrics.csv"
    df.to_csv(records_path, index=False)

    summary = summarize(df)
    summary_path = args.out_dir / "clean_preservation_summary.csv"
    summary.to_csv(summary_path, index=False)

    compact = make_compact_table(df)
    compact_path = args.out_dir / "clean_preservation_compact_table.csv"
    compact.to_csv(compact_path, index=False)
    latex_path = args.out_dir / "clean_preservation_compact_table.tex"
    latex_path.write_text(compact_to_latex(compact), encoding="utf-8")

    print(f"Saved {records_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {compact_path}")
    print(f"Saved {latex_path}")
    print("\nCompact preservation table:")
    print(compact.to_string(index=False))


def natural_clean_key(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def ensure_2d_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.ndim != 2:
        raise RuntimeError(f"Expected 1D or 2D array, got shape {x.shape}.")
    if x.shape[0] > x.shape[1] and x.shape[1] <= 16:
        return x.T
    return x


def compute_waveform_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    eps = np.finfo(float).eps
    diff = y - x
    signal_energy = float(np.sum(x**2))
    diff_energy = float(np.sum(diff**2))
    rms_input = float(np.sqrt(np.mean(x**2)))
    rms_change = float(np.sqrt(np.mean(diff**2)))
    rrmse = float(np.sqrt(diff_energy / max(signal_energy, eps)))
    preservation_snr = (
        np.inf if diff_energy <= eps else float(10.0 * np.log10(signal_energy / diff_energy))
    )
    if np.std(x) <= eps or np.std(y) <= eps:
        corr = np.nan
    else:
        corr = float(np.corrcoef(x, y)[0, 1])
    return {
        "rms_input": rms_input,
        "rms_change": rms_change,
        "rrmse": rrmse,
        "rrmse_percent": 100.0 * rrmse,
        "preservation_snr_db": preservation_snr,
        "pearson_r": corr,
        "max_abs_change": float(np.max(np.abs(diff))),
    }


def compute_spectral_metrics(x: np.ndarray, y: np.ndarray, fs: float) -> dict[str, float]:
    nperseg = min(256, len(x))
    if nperseg < 16:
        return {}
    freqs, px = ss.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    _, py = ss.welch(y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    eps = np.finfo(float).eps

    valid = (freqs >= 1.0) & (freqs <= 45.0)
    psd_change_db = 10.0 * np.log10((py[valid] + eps) / (px[valid] + eps))
    metrics = {
        "mean_abs_psd_change_db_1_45": float(np.mean(np.abs(psd_change_db))),
        "median_abs_psd_change_db_1_45": float(np.median(np.abs(psd_change_db))),
    }
    for band_name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        p_x = bandpower(freqs[mask], px[mask])
        p_y = bandpower(freqs[mask], py[mask])
        metrics[f"{band_name}_power_change_db"] = float(
            10.0 * np.log10((p_y + eps) / (p_x + eps))
        )
    return metrics


def bandpower(freqs: np.ndarray, power: np.ndarray) -> float:
    if freqs.size == 0:
        return np.nan
    if freqs.size == 1:
        return float(power[0])
    return float(np.trapezoid(power, freqs))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in numeric_metric_columns(df):
        values = df[column].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "metric": column,
                "n": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"record", "channel"}
    return [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]


def make_compact_table(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("Clean epochs", "record_count", "", "count"),
        ("Waveform RRMSE", "rrmse_percent", "%", "median_iqr"),
        ("Preservation SNR", "preservation_snr_db", "dB", "median_iqr"),
        ("Pearson correlation", "pearson_r", "", "median_iqr"),
        (
            "Mean absolute PSD change, 1--45 Hz",
            "mean_abs_psd_change_db_1_45",
            "dB",
            "median_iqr",
        ),
        ("Alpha-band power change", "alpha_power_change_db", "dB", "median_iqr"),
        ("Beta-band power change", "beta_power_change_db", "dB", "median_iqr"),
    ]
    rows = []
    for label, column, unit, mode in specs:
        if mode == "count":
            value = str(df["record"].nunique())
        else:
            values = df[column].replace([np.inf, -np.inf], np.nan).dropna()
            value = format_median_iqr(values, unit)
        rows.append({"Metric": label, "Value": value})
    return pd.DataFrame(rows)


def format_median_iqr(values: pd.Series, unit: str) -> str:
    median = float(values.median())
    q25 = float(values.quantile(0.25))
    q75 = float(values.quantile(0.75))
    if unit == "%":
        return f"{median:.2f}% [{q25:.2f}, {q75:.2f}]"
    if unit == "dB":
        return f"{median:.2f} dB [{q25:.2f}, {q75:.2f}]"
    return f"{median:.5f} [{q25:.5f}, {q75:.5f}]"


def compact_to_latex(compact: pd.DataFrame) -> str:
    lines = [
        "\\begin{table}[!t]",
        "\\caption{Clean-EEG preservation diagnostic on EEGdenoiseNet epochs. "
        "The proposed full chain was applied to clean EEG epochs without added artifact. "
        "Values are medians [IQR] across epochs unless otherwise stated.}",
        "\\label{tab:clean_preservation}",
        "\\centering",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
    ]
    for _, row in compact.iterrows():
        metric = latex_escape(str(row["Metric"]))
        value = latex_escape(str(row["Value"]))
        lines.append(f"{metric} & {value} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


if __name__ == "__main__":
    main()
