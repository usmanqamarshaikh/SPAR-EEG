from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from eeg_eval.datasets import ensure_2d_rows, normalize_mask
from eeg_eval.metrics import evaluate_record


TESTS = {
    "vmd": {
        "prefix": "emg",
        "setting_prefix": "vmd_k",
        "default_depth": 8,
        "label": "VMD mode count",
    },
    "ssa": {
        "prefix": "eog",
        "setting_prefix": "ssa_l",
        "default_depth": 12,
        "label": "SSA embedding length",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exploratory depth sensitivity outputs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--restored-root",
        type=Path,
        default=Path("results/restored_depth_sensitivity_exploratory"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/analysis_depth_sensitivity_exploratory"),
    )
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--snr-levels", type=float, nargs="+", default=[-10.0])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    record_ids = pd.read_csv(args.manifest)["record_id"].astype(int).tolist()
    metrics, debug = load_outputs(
        args.data_dir, args.restored_root, record_ids, args.depths, args.snr_levels
    )

    metrics.to_csv(args.out_dir / "depth_sensitivity_record_metrics.csv", index=False)
    metrics.to_parquet(args.out_dir / "depth_sensitivity_record_metrics.parquet", index=False)
    debug.to_csv(args.out_dir / "depth_sensitivity_debug_summary_by_record.csv", index=False)

    summary = summarize(metrics, debug)
    summary.to_csv(args.out_dir / "depth_sensitivity_summary.csv", index=False)
    comparisons = paired_comparisons(metrics)
    comparisons.to_csv(args.out_dir / "depth_sensitivity_paired_comparisons.csv", index=False)
    make_figure(
        metrics,
        args.depths,
        args.snr_levels,
        args.out_dir / "depth_sensitivity_exploratory.pdf",
    )
    make_figure(
        metrics,
        args.depths,
        args.snr_levels,
        args.out_dir / "depth_sensitivity_exploratory.png",
    )

    print("Exploratory decomposition-depth analysis completed.")
    print(f"  selected records : {len(record_ids)}")
    print(f"  metric rows      : {len(metrics)}")
    print(f"  debug rows       : {len(debug)}")
    print(f"  outputs          : {args.out_dir}")
    print()
    display = summary[summary["region"].isin(["artifact", "clean"])].copy()
    print(display.to_string(index=False))
    print("\nPaired comparisons against each family's submitted default:")
    print(comparisons.to_string(index=False))


def load_outputs(
    data_dir: Path,
    restored_root: Path,
    record_ids: list[int],
    depths: list[int],
    snr_levels: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    debug_rows: list[dict[str, object]] = []

    for family, spec in TESTS.items():
        for snr_db in snr_levels:
            file_name = sensitivity_file_name(str(spec["prefix"]), snr_db)
            source_path = data_dir / file_name
            with h5py.File(source_path, "r") as source_h5:
                for depth in depths:
                    setting = f"{spec['setting_prefix']}{depth}"
                    restored_path = restored_root / setting / file_name
                    if not restored_path.exists():
                        raise FileNotFoundError(restored_path)
                    with h5py.File(restored_path, "r") as restored_h5:
                        for record_id in record_ids:
                            record = f"{spec['prefix']}_{record_id}"
                            if record not in source_h5 or record not in restored_h5:
                                raise KeyError(f"Missing paired record {record} for {setting}")
                            source_group = source_h5[record]
                            restored_ds = restored_h5[record]
                            status = decode(restored_ds.attrs.get("status", ""))
                            if status != "OK":
                                raise RuntimeError(f"{setting}/{record}: {status}")

                            signal = ensure_2d_rows(source_group["eeg_signal"][()])
                            reference = ensure_2d_rows(source_group["eeg_reference"][()])
                            mask = normalize_mask(source_group["artifacts"][()], signal.shape[1])
                            restored = ensure_2d_rows(restored_ds[()])
                            fs = float(source_group.attrs.get("freq", 256.0))

                            for row in evaluate_record(restored, signal, reference, mask, fs):
                                row.update(
                                    {
                                        "family": family,
                                        "setting": setting,
                                        "depth": depth,
                                        "is_submitted_default": depth == spec["default_depth"],
                                        "record_id": record_id,
                                        "record": record,
                                        "noise_type": spec["prefix"],
                                        "nominal_snr_db": float(snr_db),
                                        "contamination_level": contamination_label(snr_db),
                                    }
                                )
                                metric_rows.append(row)

                            debug_rows.append(
                                {
                                    "family": family,
                                    "setting": setting,
                                    "depth": depth,
                                    "is_submitted_default": depth == spec["default_depth"],
                                    "record_id": record_id,
                                    "record": record,
                                    "nominal_snr_db": float(snr_db),
                                    "contamination_level": contamination_label(snr_db),
                                    "configured_depth": attr_float(
                                        restored_ds, "configured_depth"
                                    ),
                                    "realized_depth": attr_float(restored_ds, "realized_depth"),
                                    "detected_regions": attr_float(
                                        restored_ds, "detected_regions"
                                    ),
                                    "clean_samples": attr_float(restored_ds, "clean_samples"),
                                    "mask_fraction": attr_float(restored_ds, "mask_fraction"),
                                    "attenuated_sample_fraction": attr_float(
                                        restored_ds, "attenuated_sample_fraction"
                                    ),
                                    "output_change_rrmse": attr_float(
                                        restored_ds, "output_change_rrmse"
                                    ),
                                    "elapsed_sec": attr_float(restored_ds, "elapsed_sec"),
                                }
                            )

    return pd.DataFrame(metric_rows), pd.DataFrame(debug_rows)


def summarize(metrics: pd.DataFrame, debug: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (family, snr_db, setting, depth, region), group in metrics.groupby(
        ["family", "nominal_snr_db", "setting", "depth", "region"], sort=True
    ):
        delta = finite(group["deltaSNR"])
        change = finite(group["OutputChange_RRMSE"])
        r_after = finite(group["R_after"])
        rrmse_after = finite(group["RRMSE_after"])
        dbg = debug[
            (debug["family"] == family)
            & (debug["nominal_snr_db"] == snr_db)
            & (debug["depth"] == depth)
        ]
        rows.append(
            {
                "family": family,
                "nominal_snr_db": float(snr_db),
                "contamination_level": contamination_label(float(snr_db)),
                "setting": setting,
                "depth": int(depth),
                "is_submitted_default": bool(group["is_submitted_default"].iloc[0]),
                "region": region,
                "n_records": int(group["record_id"].nunique()),
                "median_delta_snr_db": quantile(delta, 0.50),
                "q1_delta_snr_db": quantile(delta, 0.25),
                "q3_delta_snr_db": quantile(delta, 0.75),
                "fraction_delta_snr_positive": float(np.mean(delta > 0)) if len(delta) else np.nan,
                "median_output_change_rrmse": quantile(change, 0.50),
                "q3_output_change_rrmse": quantile(change, 0.75),
                "median_r_after": quantile(r_after, 0.50),
                "median_rrmse_after": quantile(rrmse_after, 0.50),
                "activation_rate": float(np.mean(dbg["output_change_rrmse"] > 1e-12)),
                "median_detected_regions": float(dbg["detected_regions"].median()),
                "median_mask_fraction": float(dbg["mask_fraction"].median()),
                "median_attenuated_sample_fraction": float(
                    dbg["attenuated_sample_fraction"].median()
                ),
                "median_elapsed_sec": float(dbg["elapsed_sec"].median()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["family", "nominal_snr_db", "depth", "region"])
        .reset_index(drop=True)
    )


def paired_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "artifact_delta_snr_db": ("artifact", "deltaSNR"),
        "clean_output_change_rrmse": ("clean", "OutputChange_RRMSE"),
        "whole_rrmse_after": ("whole", "RRMSE_after"),
    }
    rows: list[dict[str, object]] = []
    for family, spec in TESTS.items():
        default_depth = int(spec["default_depth"])
        for snr_db in sorted(
            metrics.loc[metrics["family"] == family, "nominal_snr_db"].unique()
        ):
            for metric_name, (region, column) in definitions.items():
                subset = metrics[
                    (metrics["family"] == family)
                    & (metrics["nominal_snr_db"] == snr_db)
                    & (metrics["region"] == region)
                ]
                pivot = subset.pivot(index="record_id", columns="depth", values=column)
                if default_depth not in pivot:
                    continue
                family_rows = []
                for depth in sorted(
                    int(v) for v in pivot.columns if int(v) != default_depth
                ):
                    pair = (
                        pivot[[depth, default_depth]]
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                    )
                    diff = pair[depth] - pair[default_depth]
                    if len(diff) == 0:
                        stat, p_raw = np.nan, np.nan
                    elif np.allclose(diff, 0):
                        stat, p_raw = 0.0, 1.0
                    else:
                        stat, p_raw = wilcoxon(
                            pair[depth], pair[default_depth], alternative="two-sided"
                        )
                    family_rows.append(
                        {
                            "family": family,
                            "nominal_snr_db": float(snr_db),
                            "contamination_level": contamination_label(float(snr_db)),
                            "metric": metric_name,
                            "alternative_depth": depth,
                            "default_depth": default_depth,
                            "n_pairs": len(pair),
                            "median_alternative": float(pair[depth].median())
                            if len(pair)
                            else np.nan,
                            "median_default": float(pair[default_depth].median())
                            if len(pair)
                            else np.nan,
                            "median_paired_difference_alt_minus_default": float(
                                diff.median()
                            )
                            if len(diff)
                            else np.nan,
                            "wilcoxon_statistic": float(stat),
                            "p_raw_two_sided": float(p_raw),
                        }
                    )
                valid = [
                    row for row in family_rows if np.isfinite(row["p_raw_two_sided"])
                ]
                if valid:
                    adjusted = holm_adjust(
                        np.asarray([row["p_raw_two_sided"] for row in valid], dtype=float)
                    )
                    for row, p_holm in zip(valid, adjusted):
                        row["p_holm_within_family_metric"] = float(p_holm)
                for row in family_rows:
                    row.setdefault("p_holm_within_family_metric", np.nan)
                    rows.append(row)
    return pd.DataFrame(rows)


def make_figure(
    metrics: pd.DataFrame, depths: list[int], snr_levels: list[float], path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=True)
    panels = [
        ("vmd", "artifact", "deltaSNR", "VMD: artifact-region $\\Delta$SNR", "$\\Delta$SNR (dB)"),
        (
            "ssa",
            "artifact",
            "deltaSNR",
            "SSA: artifact-region $\\Delta$SNR",
            "$\\Delta$SNR (dB)",
        ),
        (
            "vmd",
            "clean",
            "OutputChange_RRMSE",
            "VMD: outside-mask modification",
            "Output change (%)",
        ),
        (
            "ssa",
            "clean",
            "OutputChange_RRMSE",
            "SSA: outside-mask modification",
            "Output change (%)",
        ),
    ]
    colors = {4: "#3B6FB6", 8: "#B44A3C", 12: "#2F855A"}
    markers = {4: "o", 8: "s", 12: "^"}
    ordered_snr = sorted(float(v) for v in snr_levels)
    for ax, (family, region, column, title, ylabel) in zip(axes.ravel(), panels):
        subset = metrics[(metrics["family"] == family) & (metrics["region"] == region)]
        for depth in depths:
            medians = []
            q1 = []
            q3 = []
            for snr_db in ordered_snr:
                vals = finite(
                    subset.loc[
                        (subset["depth"] == depth)
                        & (subset["nominal_snr_db"] == snr_db),
                        column,
                    ]
                )
                if column == "OutputChange_RRMSE":
                    vals = 100.0 * vals
                medians.append(quantile(vals, 0.50))
                q1.append(quantile(vals, 0.25))
                q3.append(quantile(vals, 0.75))
            medians_arr = np.asarray(medians)
            q1_arr = np.asarray(q1)
            q3_arr = np.asarray(q3)
            default_depth = int(TESTS[family]["default_depth"])
            label = f"{depth}" + (" (default)" if depth == default_depth else "")
            ax.errorbar(
                ordered_snr,
                medians_arr,
                yerr=np.vstack((medians_arr - q1_arr, q3_arr - medians_arr)),
                color=colors.get(depth, "#4B5563"),
                marker=markers.get(depth, "o"),
                linewidth=2.0 if depth == default_depth else 1.2,
                linestyle="-" if depth == default_depth else "--",
                capsize=3,
                label=label,
            )
        ax.axhline(0, color="#6B7280", linewidth=0.7, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Input SNR (dB): high to low contamination")
        ax.set_ylabel(ylabel)
        ax.set_xticks(
            ordered_snr,
            [f"{snr:g}\n{contamination_label(snr)}" for snr in ordered_snr],
        )
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
        ax.legend(title=str(TESTS[family]["label"]), fontsize=8, title_fontsize=8)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def finite(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else np.nan


def sensitivity_file_name(prefix: str, snr_db: float) -> str:
    return f"03_denoise-net_{prefix}_{snr_db:g}dB.h5"


def contamination_label(snr_db: float) -> str:
    if snr_db <= -15:
        return "High"
    if snr_db <= -5:
        return "Medium"
    return "Low"


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm-adjusted p-values in the original input order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    multipliers = np.arange(len(ranked), 0, -1, dtype=float)
    adjusted_ranked = np.minimum(1.0, np.maximum.accumulate(ranked * multipliers))
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def attr_float(dataset: h5py.Dataset, name: str) -> float:
    value = dataset.attrs.get(name, np.nan)
    try:
        array = np.asarray(value).reshape(-1)
        return float(array[0]) if array.size else np.nan
    except (TypeError, ValueError):
        return np.nan


if __name__ == "__main__":
    main()
