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
        "fixed_depth": 8,
        "default_dilation_sec": 0.14,
        "label": "VMD EMG pass",
    },
    "ssa": {
        "prefix": "eog",
        "fixed_depth": 12,
        "default_dilation_sec": 0.12,
        "label": "SSA EOG pass",
    },
}
DEFAULT_MULTIPLIER = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SPAR-EEG dilation sensitivity.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--restored-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--default-reference-root", type=Path, required=True)
    parser.add_argument("--multipliers", type=float, nargs="+", default=[0, 1, 2])
    parser.add_argument("--snr-levels", type=float, nargs="+", default=[-20, -10, 0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    record_ids = pd.read_csv(args.manifest)["record_id"].astype(int).tolist()
    multipliers = list(dict.fromkeys(float(value) for value in args.multipliers))
    snr_levels = list(dict.fromkeys(float(value) for value in args.snr_levels))

    metrics, debug = load_outputs(
        args.data_dir,
        args.restored_root,
        record_ids,
        multipliers,
        snr_levels,
    )
    default_error = audit_default_equivalence(
        args.restored_root,
        args.default_reference_root,
        record_ids,
        snr_levels,
    )
    audit = audit_outputs(metrics, debug, record_ids, multipliers, snr_levels)

    metrics.to_csv(args.out_dir / "dilation_sensitivity_record_metrics.csv", index=False)
    metrics.to_parquet(
        args.out_dir / "dilation_sensitivity_record_metrics.parquet", index=False
    )
    debug.to_csv(args.out_dir / "dilation_sensitivity_debug_by_record.csv", index=False)
    summary = summarize(metrics, debug)
    summary.to_csv(args.out_dir / "dilation_sensitivity_summary.csv", index=False)
    comparisons = paired_comparisons(metrics)
    comparisons.to_csv(
        args.out_dir / "dilation_sensitivity_paired_comparisons.csv", index=False
    )
    compact = make_compact_table(summary)
    compact.to_csv(args.out_dir / "dilation_sensitivity_compact_table.csv", index=False)
    for suffix in ("pdf", "png"):
        make_figure(
            metrics,
            multipliers,
            snr_levels,
            args.out_dir / f"dilation_sensitivity.{suffix}",
        )
    report = build_report(audit, default_error, compact, comparisons)
    (args.out_dir / "dilation_sensitivity_report.md").write_text(report, encoding="utf-8")

    print(audit)
    print(f"Default-output equivalence maximum error: {default_error:.3g}")
    print("\nCompact descriptive table:")
    print(compact.to_string(index=False))
    print(f"\nOutputs: {args.out_dir}")


def load_outputs(
    data_dir: Path,
    restored_root: Path,
    record_ids: list[int],
    multipliers: list[float],
    snr_levels: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    debug_rows: list[dict[str, object]] = []
    for family, spec in TESTS.items():
        prefix = str(spec["prefix"])
        fixed_depth = int(spec["fixed_depth"])
        default_sec = float(spec["default_dilation_sec"])
        for snr_db in snr_levels:
            file_name = sensitivity_file_name(prefix, snr_db)
            source_path = data_dir / file_name
            with h5py.File(source_path, "r") as source_h5:
                for multiplier in multipliers:
                    dilation_sec = default_sec * multiplier
                    setting = setting_key(family, multiplier)
                    restored_path = restored_root / setting / file_name
                    if not restored_path.exists():
                        raise FileNotFoundError(restored_path)
                    with h5py.File(restored_path, "r") as restored_h5:
                        validate_root(
                            restored_h5,
                            family,
                            multiplier,
                            dilation_sec,
                            default_sec,
                            fixed_depth,
                            snr_db,
                            len(record_ids),
                        )
                        for record_id in record_ids:
                            record = f"{prefix}_{record_id}"
                            if record not in source_h5 or record not in restored_h5:
                                raise KeyError(f"Missing paired record {record} for {setting}")
                            source_group = source_h5[record]
                            restored_ds = restored_h5[record]
                            status = decode(restored_ds.attrs.get("status", ""))
                            if status != "OK":
                                raise RuntimeError(f"{setting}/{record}: {status}")
                            for name, expected in (
                                ("dilation_multiplier", multiplier),
                                ("configured_dilation_sec", dilation_sec),
                                ("submitted_default_dilation_sec", default_sec),
                                ("fixed_depth", fixed_depth),
                                ("realized_depth", fixed_depth),
                                ("fixed_zthr", 3),
                            ):
                                assert_close(restored_ds, name, expected)

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
                                        "dilation_multiplier": multiplier,
                                        "dilation_sec": dilation_sec,
                                        "is_submitted_default": np.isclose(
                                            multiplier, DEFAULT_MULTIPLIER
                                        ),
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
                                    "dilation_multiplier": multiplier,
                                    "dilation_sec": dilation_sec,
                                    "is_submitted_default": np.isclose(
                                        multiplier, DEFAULT_MULTIPLIER
                                    ),
                                    "fixed_depth": fixed_depth,
                                    "record_id": record_id,
                                    "record": record,
                                    "nominal_snr_db": float(snr_db),
                                    "contamination_level": contamination_label(snr_db),
                                    "detected_regions": attr_float(
                                        restored_ds, "detected_regions"
                                    ),
                                    "clean_samples": attr_float(restored_ds, "clean_samples"),
                                    "bypass_no_region": attr_float(
                                        restored_ds, "bypass_no_region"
                                    ),
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
                                    "minimum_realized_gain": attr_float(
                                        restored_ds, "minimum_realized_gain"
                                    ),
                                    "output_change_rrmse": attr_float(
                                        restored_ds, "output_change_rrmse"
                                    ),
                                    "elapsed_sec": attr_float(restored_ds, "elapsed_sec"),
                                }
                            )
    return pd.DataFrame(metric_rows), pd.DataFrame(debug_rows)


def audit_default_equivalence(
    restored_root: Path,
    reference_root: Path,
    record_ids: list[int],
    snr_levels: list[float],
) -> float:
    maximum = 0.0
    for family, spec in TESTS.items():
        file_prefix = str(spec["prefix"])
        for snr_db in snr_levels:
            file_name = sensitivity_file_name(file_prefix, snr_db)
            current_path = restored_root / setting_key(family, 1) / file_name
            reference_path = reference_root / f"{family}_z3" / file_name
            if not reference_path.exists():
                raise FileNotFoundError(reference_path)
            with h5py.File(current_path, "r") as current, h5py.File(
                reference_path, "r"
            ) as reference:
                for record_id in record_ids:
                    record = f"{file_prefix}_{record_id}"
                    difference = np.max(
                        np.abs(
                            np.asarray(current[record][()], dtype=float)
                            - np.asarray(reference[record][()], dtype=float)
                        )
                    )
                    maximum = max(maximum, float(difference))
    if maximum > 1e-10:
        raise RuntimeError(f"Default-output reproduction failed: maximum error {maximum}")
    return maximum


def audit_outputs(
    metrics: pd.DataFrame,
    debug: pd.DataFrame,
    record_ids: list[int],
    multipliers: list[float],
    snr_levels: list[float],
) -> str:
    expected_restorations = len(TESTS) * len(snr_levels) * len(multipliers) * len(record_ids)
    expected_metrics = expected_restorations * 3
    if len(debug) != expected_restorations or len(metrics) != expected_metrics:
        raise RuntimeError(
            f"Expected {expected_restorations} restorations/{expected_metrics} metrics, "
            f"found {len(debug)}/{len(metrics)}"
        )
    if debug[["family", "nominal_snr_db", "dilation_multiplier", "record_id"]].duplicated().any():
        raise RuntimeError("Duplicate restoration diagnostics detected.")
    default_artifact = metrics[
        (metrics["region"] == "artifact")
        & np.isclose(metrics["dilation_multiplier"], DEFAULT_MULTIPLIER)
    ]
    maximum_snr_error = float(
        np.max(np.abs(default_artifact["SNR_before"] - default_artifact["nominal_snr_db"]))
    )
    if maximum_snr_error > 1e-8:
        raise RuntimeError(f"Input artifact-region SNR audit failed: {maximum_snr_error}")
    return (
        f"Audit passed: {expected_restorations} restorations, {expected_metrics} metric rows, "
        f"no failures, and maximum artifact-region input-SNR error "
        f"{maximum_snr_error:.3g} dB."
    )


def summarize(metrics: pd.DataFrame, debug: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = [
        "family",
        "nominal_snr_db",
        "setting",
        "dilation_multiplier",
        "dilation_sec",
        "region",
    ]
    for values, group in metrics.groupby(keys, sort=True):
        family, snr_db, setting, multiplier, dilation_sec, region = values
        delta = finite(group["deltaSNR"])
        change = finite(group["OutputChange_RRMSE"])
        rrmse_after = finite(group["RRMSE_after"])
        dbg = debug[
            (debug["family"] == family)
            & (debug["nominal_snr_db"] == snr_db)
            & np.isclose(debug["dilation_multiplier"], multiplier)
        ]
        rows.append(
            {
                "family": family,
                "nominal_snr_db": float(snr_db),
                "contamination_level": contamination_label(float(snr_db)),
                "setting": setting,
                "dilation_multiplier": float(multiplier),
                "dilation_sec": float(dilation_sec),
                "is_submitted_default": bool(
                    np.isclose(multiplier, DEFAULT_MULTIPLIER)
                ),
                "region": region,
                "n_records": int(group["record_id"].nunique()),
                "median_delta_snr_db": quantile(delta, 0.50),
                "q1_delta_snr_db": quantile(delta, 0.25),
                "q3_delta_snr_db": quantile(delta, 0.75),
                "fraction_delta_snr_positive": (
                    float(np.mean(delta > 0)) if len(delta) else np.nan
                ),
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
                "median_minimum_realized_gain": median_numeric(
                    dbg["minimum_realized_gain"]
                ),
                "median_elapsed_sec": median_numeric(dbg["elapsed_sec"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["family", "nominal_snr_db", "dilation_multiplier", "region"]
    )


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
                pivot = subset.pivot(
                    index="record_id", columns="dilation_multiplier", values=column
                )
                family_rows: list[dict[str, object]] = []
                for alternative in sorted(
                    value for value in pivot.columns if value != DEFAULT_MULTIPLIER
                ):
                    pair = pivot[[alternative, DEFAULT_MULTIPLIER]].replace(
                        [np.inf, -np.inf], np.nan
                    ).dropna()
                    difference = pair[alternative] - pair[DEFAULT_MULTIPLIER]
                    if len(difference) == 0:
                        statistic, p_raw = np.nan, np.nan
                    elif np.allclose(difference, 0):
                        statistic, p_raw = 0.0, 1.0
                    else:
                        statistic, p_raw = wilcoxon(
                            pair[alternative],
                            pair[DEFAULT_MULTIPLIER],
                            alternative="two-sided",
                        )
                    family_rows.append(
                        {
                            "family": family,
                            "nominal_snr_db": float(snr_db),
                            "metric": metric_name,
                            "alternative_multiplier": float(alternative),
                            "default_multiplier": DEFAULT_MULTIPLIER,
                            "n_pairs": len(pair),
                            "median_alternative": pair[alternative].median(),
                            "median_default": pair[DEFAULT_MULTIPLIER].median(),
                            "median_paired_difference_alt_minus_default": difference.median(),
                            "wilcoxon_statistic": float(statistic),
                            "p_raw_two_sided": float(p_raw),
                        }
                    )
                adjusted = holm_adjust(
                    np.asarray([row["p_raw_two_sided"] for row in family_rows])
                )
                for row, p_holm in zip(family_rows, adjusted):
                    row["p_holm_within_family_metric"] = float(p_holm)
                    rows.append(row)
    return pd.DataFrame(rows)


def make_compact_table(summary: pd.DataFrame) -> pd.DataFrame:
    artifact = summary[summary["region"] == "artifact"].copy()
    clean = summary[summary["region"] == "clean"][
        [
            "family",
            "nominal_snr_db",
            "dilation_multiplier",
            "median_output_change_rrmse",
        ]
    ].rename(columns={"median_output_change_rrmse": "outside_mask_change_rrmse"})
    whole = summary[summary["region"] == "whole"][
        ["family", "nominal_snr_db", "dilation_multiplier", "median_rrmse_after"]
    ].rename(columns={"median_rrmse_after": "whole_rrmse_after"})
    keep = [
        "family",
        "nominal_snr_db",
        "contamination_level",
        "dilation_multiplier",
        "dilation_sec",
        "is_submitted_default",
        "n_records",
        "median_delta_snr_db",
        "q1_delta_snr_db",
        "q3_delta_snr_db",
        "fraction_delta_snr_positive",
        "activation_rate",
        "no_region_bypass_rate",
        "median_mask_fraction",
        "median_attenuated_sample_fraction",
        "median_minimum_realized_gain",
        "median_elapsed_sec",
    ]
    return (
        artifact[keep]
        .merge(
            clean,
            on=["family", "nominal_snr_db", "dilation_multiplier"],
            validate="one_to_one",
        )
        .merge(
            whole,
            on=["family", "nominal_snr_db", "dilation_multiplier"],
            validate="one_to_one",
        )
        .sort_values(["family", "nominal_snr_db", "dilation_multiplier"])
        .reset_index(drop=True)
    )


def make_figure(
    metrics: pd.DataFrame,
    multipliers: list[float],
    snr_levels: list[float],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    panels = [
        ("vmd", "artifact", "deltaSNR", "(a) EMG: artifact-region $\\Delta$SNR", "$\\Delta$SNR (dB)"),
        ("ssa", "artifact", "deltaSNR", "(b) EOG: artifact-region $\\Delta$SNR", "$\\Delta$SNR (dB)"),
        ("vmd", "clean", "OutputChange_RRMSE", "(c) EMG: outside-mask modification", "Output change (%)"),
        ("ssa", "clean", "OutputChange_RRMSE", "(d) EOG: outside-mask modification", "Output change (%)"),
    ]
    colors = {-20.0: "#B44A3C", -10.0: "#D18B28", 0.0: "#3B6FB6"}
    markers = {-20.0: "o", -10.0: "s", 0.0: "^"}
    ordered = sorted(multipliers)
    for axis, (family, region, column, title, ylabel) in zip(axes.ravel(), panels):
        subset = metrics[(metrics["family"] == family) & (metrics["region"] == region)]
        for snr_db in sorted(snr_levels):
            medians, q1, q3 = [], [], []
            for multiplier in ordered:
                values = finite(
                    subset.loc[
                        (subset["nominal_snr_db"] == snr_db)
                        & np.isclose(subset["dilation_multiplier"], multiplier),
                        column,
                    ]
                )
                if column == "OutputChange_RRMSE":
                    values = 100 * values
                medians.append(quantile(values, 0.5))
                q1.append(quantile(values, 0.25))
                q3.append(quantile(values, 0.75))
            median = np.asarray(medians)
            low, high = np.asarray(q1), np.asarray(q3)
            axis.errorbar(
                ordered,
                median,
                yerr=np.vstack((median - low, high - median)),
                color=colors.get(snr_db, "#4B5563"),
                marker=markers.get(snr_db, "o"),
                linewidth=1.5,
                capsize=3,
                label=f"{snr_db:g} dB",
            )
        axis.axvline(1, color="#111827", linestyle="--", linewidth=1, label="Default")
        if column == "deltaSNR":
            axis.axhline(0, color="#9CA3AF", linestyle=":", linewidth=0.7)
        default_sec = float(TESTS[family]["default_dilation_sec"])
        labels = [f"{value:g}x\n({default_sec * value:.2f} s)" for value in ordered]
        axis.set_xticks(ordered)
        axis.set_xticklabels(labels)
        axis.set_xlabel("Dilation margin multiplier")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
        axis.legend(fontsize=7.5)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_report(
    audit: str,
    default_error: float,
    compact: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> str:
    display = compact[
        [
            "family",
            "nominal_snr_db",
            "dilation_multiplier",
            "dilation_sec",
            "median_delta_snr_db",
            "outside_mask_change_rrmse",
            "median_attenuated_sample_fraction",
        ]
    ]
    return "\n".join(
        [
            "# Temporal-Dilation Sensitivity",
            "",
            "## Audit",
            "",
            audit,
            f"Submitted-default outputs reproduced the previous threshold study with maximum absolute error `{default_error:.3g}`.",
            "",
            "## Compact outcomes",
            "",
            dataframe_to_markdown(display),
            "",
            "## Interpretation rule",
            "",
            "Dilation is evaluated as a suppression--preservation tradeoff. Zero dilation may leave boundary leakage; "
            "the submitted margin should cover nearby leakage; a doubled margin is useful only if additional "
            "artifact suppression is not offset by greater outside-mask modification.",
            "",
            f"Paired-comparison rows: {len(comparisons)}.",
            "",
        ]
    )


def validate_root(
    h5: h5py.File,
    family: str,
    multiplier: float,
    dilation_sec: float,
    default_sec: float,
    fixed_depth: int,
    snr_db: float,
    n_records: int,
) -> None:
    if decode(h5.attrs.get("test_family", "")) != family:
        raise ValueError("Root family metadata mismatch.")
    for name, expected in (
        ("dilation_multiplier", multiplier),
        ("configured_dilation_sec", dilation_sec),
        ("submitted_default_dilation_sec", default_sec),
        ("fixed_depth", fixed_depth),
        ("fixed_zthr", 3),
        ("nominal_snr_db", snr_db),
        ("selected_record_count", n_records),
        ("failed_record_count", 0),
    ):
        value = float(np.asarray(h5.attrs.get(name, np.nan)).reshape(-1)[0])
        if not np.isclose(value, expected):
            raise ValueError(f"Root attribute {name}={value}, expected {expected}")


def assert_close(dataset: h5py.Dataset, name: str, expected: float) -> None:
    value = attr_float(dataset, name)
    if not np.isclose(value, expected):
        raise ValueError(f"{dataset.name}: {name}={value}, expected {expected}")


def setting_key(family: str, multiplier: float) -> str:
    text = f"{multiplier:g}".replace("-", "m").replace(".", "p")
    return f"{family}_d{text}"


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
    return quantile(finite(series), 0.5)


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


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("NA")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.4g}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
