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
    "vmd": {"prefix": "emg", "fixed_depth": 8, "label": "VMD EMG pass"},
    "ssa": {"prefix": "eog", "fixed_depth": 12, "label": "SSA EOG pass"},
}
DEFAULT_ZTHR = 3.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SPAR-EEG threshold sensitivity.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--restored-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--snr-levels", type=float, nargs="+", default=[-20, -10, 0])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    record_ids = pd.read_csv(args.manifest)["record_id"].astype(int).tolist()
    metrics, debug = load_outputs(
        args.data_dir,
        args.restored_root,
        record_ids,
        args.thresholds,
        args.snr_levels,
    )

    metrics.to_csv(args.out_dir / "zthr_sensitivity_record_metrics.csv", index=False)
    metrics.to_parquet(args.out_dir / "zthr_sensitivity_record_metrics.parquet", index=False)
    debug.to_csv(args.out_dir / "zthr_sensitivity_debug_by_record.csv", index=False)

    summary = summarize(metrics, debug)
    summary.to_csv(args.out_dir / "zthr_sensitivity_summary.csv", index=False)
    comparisons = paired_comparisons(metrics)
    comparisons.to_csv(args.out_dir / "zthr_sensitivity_paired_comparisons.csv", index=False)
    compact = make_compact_table(summary)
    compact.to_csv(args.out_dir / "zthr_sensitivity_compact_table.csv", index=False)
    for suffix in ("pdf", "png"):
        make_figure(
            metrics,
            args.thresholds,
            args.snr_levels,
            args.out_dir / f"zthr_sensitivity.{suffix}",
        )

    print("Artifact-threshold sensitivity analysis completed.")
    print(f"  selected records : {len(record_ids)}")
    print(f"  restorations     : {len(debug)}")
    print(f"  metric rows      : {len(metrics)}")
    print(f"  outputs          : {args.out_dir}")
    print("\nCompact descriptive table:")
    print(compact.to_string(index=False))


def load_outputs(
    data_dir: Path,
    restored_root: Path,
    record_ids: list[int],
    thresholds: list[float],
    snr_levels: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    debug_rows: list[dict[str, object]] = []

    for family, spec in TESTS.items():
        prefix = str(spec["prefix"])
        fixed_depth = int(spec["fixed_depth"])
        for snr_db in snr_levels:
            file_name = sensitivity_file_name(prefix, snr_db)
            source_path = data_dir / file_name
            with h5py.File(source_path, "r") as source_h5:
                for zthr in thresholds:
                    setting = setting_key(family, zthr)
                    restored_path = restored_root / setting / file_name
                    if not restored_path.exists():
                        raise FileNotFoundError(restored_path)
                    with h5py.File(restored_path, "r") as restored_h5:
                        validate_root(restored_h5, family, zthr, fixed_depth, snr_db, len(record_ids))
                        for record_id in record_ids:
                            record = f"{prefix}_{record_id}"
                            if record not in source_h5 or record not in restored_h5:
                                raise KeyError(f"Missing paired record {record} for {setting}")
                            source_group = source_h5[record]
                            restored_ds = restored_h5[record]
                            status = decode(restored_ds.attrs.get("status", ""))
                            if status != "OK":
                                raise RuntimeError(f"{setting}/{record}: {status}")
                            assert_close(restored_ds, "configured_zthr", zthr)
                            assert_close(restored_ds, "fixed_depth", fixed_depth)
                            assert_close(restored_ds, "realized_depth", fixed_depth)

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
                                        "zthr": float(zthr),
                                        "is_submitted_default": np.isclose(zthr, DEFAULT_ZTHR),
                                        "fixed_depth": fixed_depth,
                                        "record_id": record_id,
                                        "record": record,
                                        "noise_type": prefix,
                                        "nominal_snr_db": float(snr_db),
                                        "contamination_level": contamination_label(snr_db),
                                    }
                                )
                                metric_rows.append(row)

                            debug_rows.append(
                                {
                                    "family": family,
                                    "setting": setting,
                                    "zthr": float(zthr),
                                    "is_submitted_default": np.isclose(zthr, DEFAULT_ZTHR),
                                    "fixed_depth": fixed_depth,
                                    "record_id": record_id,
                                    "record": record,
                                    "nominal_snr_db": float(snr_db),
                                    "contamination_level": contamination_label(snr_db),
                                    "detected_regions": attr_float(restored_ds, "detected_regions"),
                                    "clean_samples": attr_float(restored_ds, "clean_samples"),
                                    "min_baseline_required": attr_float(
                                        restored_ds, "min_baseline_required"
                                    ),
                                    "bypass_no_region": attr_float(restored_ds, "bypass_no_region"),
                                    "bypass_insufficient_baseline": attr_float(
                                        restored_ds, "bypass_insufficient_baseline"
                                    ),
                                    "initial_exceedance_fraction": attr_float(
                                        restored_ds, "initial_exceedance_fraction"
                                    ),
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
    group_keys = ["family", "nominal_snr_db", "setting", "zthr", "region"]
    for (family, snr_db, setting, zthr, region), group in metrics.groupby(group_keys, sort=True):
        delta = finite(group["deltaSNR"])
        change = finite(group["OutputChange_RRMSE"])
        rrmse_after = finite(group["RRMSE_after"])
        dbg = debug[
            (debug["family"] == family)
            & (debug["nominal_snr_db"] == snr_db)
            & np.isclose(debug["zthr"], zthr)
        ]
        rows.append(
            {
                "family": family,
                "nominal_snr_db": float(snr_db),
                "contamination_level": contamination_label(float(snr_db)),
                "setting": setting,
                "zthr": float(zthr),
                "is_submitted_default": bool(np.isclose(zthr, DEFAULT_ZTHR)),
                "region": region,
                "n_records": int(group["record_id"].nunique()),
                "median_delta_snr_db": quantile(delta, 0.50),
                "q1_delta_snr_db": quantile(delta, 0.25),
                "q3_delta_snr_db": quantile(delta, 0.75),
                "fraction_delta_snr_positive": float(np.mean(delta > 0)) if len(delta) else np.nan,
                "median_output_change_rrmse": quantile(change, 0.50),
                "q3_output_change_rrmse": quantile(change, 0.75),
                "median_rrmse_after": quantile(rrmse_after, 0.50),
                "activation_rate": float(np.mean(dbg["output_change_rrmse"] > 1e-12)),
                "no_region_bypass_rate": mean_numeric(dbg["bypass_no_region"]),
                "insufficient_baseline_bypass_rate": mean_numeric(
                    dbg["bypass_insufficient_baseline"]
                ),
                "median_initial_exceedance_fraction": median_numeric(
                    dbg["initial_exceedance_fraction"]
                ),
                "median_mask_fraction": median_numeric(dbg["mask_fraction"]),
                "median_detected_regions": median_numeric(dbg["detected_regions"]),
                "median_attenuated_sample_fraction": median_numeric(
                    dbg["attenuated_sample_fraction"]
                ),
                "median_elapsed_sec": median_numeric(dbg["elapsed_sec"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["family", "nominal_snr_db", "zthr", "region"]
    ).reset_index(drop=True)


def paired_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "artifact_delta_snr_db": ("artifact", "deltaSNR"),
        "outside_mask_output_change_rrmse": ("clean", "OutputChange_RRMSE"),
        "whole_rrmse_after": ("whole", "RRMSE_after"),
    }
    rows: list[dict[str, object]] = []
    for family in TESTS:
        snr_values = sorted(metrics.loc[metrics["family"] == family, "nominal_snr_db"].unique())
        for snr_db in snr_values:
            for metric_name, (region, column) in definitions.items():
                subset = metrics[
                    (metrics["family"] == family)
                    & (metrics["nominal_snr_db"] == snr_db)
                    & (metrics["region"] == region)
                ]
                pivot = subset.pivot(index="record_id", columns="zthr", values=column)
                if DEFAULT_ZTHR not in pivot:
                    continue
                family_rows: list[dict[str, object]] = []
                for zthr in sorted(float(value) for value in pivot.columns if value != DEFAULT_ZTHR):
                    pair = pivot[[zthr, DEFAULT_ZTHR]].replace([np.inf, -np.inf], np.nan).dropna()
                    diff = pair[zthr] - pair[DEFAULT_ZTHR]
                    if len(diff) == 0:
                        stat, p_raw = np.nan, np.nan
                    elif np.allclose(diff, 0):
                        stat, p_raw = 0.0, 1.0
                    else:
                        stat, p_raw = wilcoxon(
                            pair[zthr], pair[DEFAULT_ZTHR], alternative="two-sided"
                        )
                    family_rows.append(
                        {
                            "family": family,
                            "nominal_snr_db": float(snr_db),
                            "contamination_level": contamination_label(float(snr_db)),
                            "metric": metric_name,
                            "alternative_zthr": zthr,
                            "default_zthr": DEFAULT_ZTHR,
                            "n_pairs": len(pair),
                            "median_alternative": float(pair[zthr].median()) if len(pair) else np.nan,
                            "median_default": float(pair[DEFAULT_ZTHR].median()) if len(pair) else np.nan,
                            "median_paired_difference_alt_minus_default": (
                                float(diff.median()) if len(diff) else np.nan
                            ),
                            "wilcoxon_statistic": float(stat),
                            "p_raw_two_sided": float(p_raw),
                        }
                    )
                valid = [row for row in family_rows if np.isfinite(row["p_raw_two_sided"])]
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


def make_compact_table(summary: pd.DataFrame) -> pd.DataFrame:
    artifact = summary[summary["region"] == "artifact"].copy()
    clean = summary[summary["region"] == "clean"][
        ["family", "nominal_snr_db", "zthr", "median_output_change_rrmse"]
    ].rename(columns={"median_output_change_rrmse": "outside_mask_change_rrmse"})
    whole = summary[summary["region"] == "whole"][
        ["family", "nominal_snr_db", "zthr", "median_rrmse_after"]
    ].rename(columns={"median_rrmse_after": "whole_rrmse_after"})
    keep = [
        "family", "nominal_snr_db", "contamination_level", "zthr",
        "is_submitted_default", "n_records", "median_delta_snr_db",
        "q1_delta_snr_db", "q3_delta_snr_db", "fraction_delta_snr_positive",
        "activation_rate", "no_region_bypass_rate", "insufficient_baseline_bypass_rate",
        "median_initial_exceedance_fraction", "median_mask_fraction",
        "median_detected_regions", "median_attenuated_sample_fraction", "median_elapsed_sec",
    ]
    return (
        artifact[keep]
        .merge(clean, on=["family", "nominal_snr_db", "zthr"], validate="one_to_one")
        .merge(whole, on=["family", "nominal_snr_db", "zthr"], validate="one_to_one")
        .sort_values(["family", "nominal_snr_db", "zthr"])
        .reset_index(drop=True)
    )


def make_figure(
    metrics: pd.DataFrame,
    thresholds: list[float],
    snr_levels: list[float],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=True)
    panels = [
        ("vmd", "artifact", "deltaSNR", "(a) VMD: artifact-region $\\Delta$SNR", "$\\Delta$SNR (dB)"),
        ("ssa", "artifact", "deltaSNR", "(b) SSA: artifact-region $\\Delta$SNR", "$\\Delta$SNR (dB)"),
        ("vmd", "clean", "OutputChange_RRMSE", "(c) VMD: outside-mask modification", "Output change (%)"),
        ("ssa", "clean", "OutputChange_RRMSE", "(d) SSA: outside-mask modification", "Output change (%)"),
    ]
    colors = {-20.0: "#B44A3C", -10.0: "#D18B28", 0.0: "#3B6FB6"}
    markers = {-20.0: "o", -10.0: "s", 0.0: "^"}
    ordered_z = sorted(float(value) for value in thresholds)
    ordered_snr = sorted(float(value) for value in snr_levels)

    for ax, (family, region, column, title, ylabel) in zip(axes.ravel(), panels):
        subset = metrics[(metrics["family"] == family) & (metrics["region"] == region)]
        for snr_db in ordered_snr:
            medians, q1, q3 = [], [], []
            for zthr in ordered_z:
                vals = finite(
                    subset.loc[
                        (subset["nominal_snr_db"] == snr_db)
                        & np.isclose(subset["zthr"], zthr),
                        column,
                    ]
                )
                if column == "OutputChange_RRMSE":
                    vals = 100.0 * vals
                medians.append(quantile(vals, 0.50))
                q1.append(quantile(vals, 0.25))
                q3.append(quantile(vals, 0.75))
            med = np.asarray(medians)
            lower, upper = np.asarray(q1), np.asarray(q3)
            ax.errorbar(
                ordered_z,
                med,
                yerr=np.vstack((med - lower, upper - med)),
                color=colors.get(snr_db, "#4B5563"),
                marker=markers.get(snr_db, "o"),
                linewidth=1.5,
                capsize=3,
                label=f"{snr_db:g} dB ({contamination_label(snr_db)})",
            )
        ax.axvline(DEFAULT_ZTHR, color="#111827", linewidth=1.0, linestyle="--", label="Default")
        if column == "deltaSNR":
            ax.axhline(0, color="#9CA3AF", linewidth=0.7, linestyle=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Artifact threshold $z_{\\mathrm{thr}}$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ordered_z)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
        ax.legend(fontsize=7.5)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def validate_root(
    h5: h5py.File,
    family: str,
    zthr: float,
    fixed_depth: int,
    snr_db: float,
    n_records: int,
) -> None:
    if decode(h5.attrs.get("test_family", "")) != family:
        raise ValueError("Root family metadata mismatch.")
    for name, expected in (
        ("configured_zthr", zthr),
        ("fixed_depth", fixed_depth),
        ("nominal_snr_db", snr_db),
        ("selected_record_count", n_records),
    ):
        value = float(np.asarray(h5.attrs.get(name, np.nan)).reshape(-1)[0])
        if not np.isclose(value, expected):
            raise ValueError(f"Root attribute {name}={value}, expected {expected}")
    failures = float(np.asarray(h5.attrs.get("failed_record_count", np.nan)).reshape(-1)[0])
    if failures != 0:
        raise RuntimeError(f"Output collection reports {failures:g} failures.")


def assert_close(dataset: h5py.Dataset, name: str, expected: float) -> None:
    value = attr_float(dataset, name)
    if not np.isclose(value, expected):
        raise ValueError(f"{dataset.name}: {name}={value}, expected {expected}")


def setting_key(family: str, zthr: float) -> str:
    text = f"{zthr:g}".replace("-", "m").replace(".", "p")
    return f"{family}_z{text}"


def sensitivity_file_name(prefix: str, snr_db: float) -> str:
    return f"03_denoise-net_{prefix}_{snr_db:g}dB.h5"


def contamination_label(snr_db: float) -> str:
    if snr_db <= -15:
        return "High"
    if snr_db <= -5:
        return "Medium"
    return "Low"


def finite(series: pd.Series) -> np.ndarray:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy()
    )


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else np.nan


def median_numeric(series: pd.Series) -> float:
    values = finite(series)
    return quantile(values, 0.5)


def mean_numeric(series: pd.Series) -> float:
    values = finite(series)
    return float(np.mean(values)) if len(values) else np.nan


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = np.minimum(
        1.0,
        np.maximum.accumulate(ranked * np.arange(len(ranked), 0, -1, dtype=float)),
    )
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
