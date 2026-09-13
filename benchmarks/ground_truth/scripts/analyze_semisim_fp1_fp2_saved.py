from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from analyze_primary_results import (
    DATASET_LABELS,
    METRIC_DIRECTIONS,
    METHOD_LABELS,
    METHOD_ORDER,
    STAT_METRICS,
    SUMMARY_METRICS,
    holm_adjust,
    safe_mean,
    safe_median,
)
from eeg_eval.datasets import ensure_2d_rows, iter_h5_records, parse_dataset_info
from eeg_eval.metrics import evaluate_record, finalize_metrics_table


SEMI_SIM_CHANNELS = [
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
]

SELECTIONS = {
    "fp1_fp2": ["FP1", "FP2"],
    "fp1": ["FP1"],
    "fp2": ["FP2"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute semi-simulated EOG metrics on frontal channels only "
            "from saved restored H5 outputs."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--restored-root",
        type=Path,
        default=Path("results/restored_primary_saved"),
        help="Root containing one restored-output folder per method.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/analysis_semisim_fp1_fp2_saved"),
    )
    parser.add_argument("--h5-file", default="02_semisimulated_eog.h5")
    parser.add_argument(
        "--selections",
        nargs="+",
        default=["fp1_fp2", "fp1", "fp2"],
        choices=sorted(SELECTIONS),
    )
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.data_dir / args.h5_file
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    method_dirs = [
        args.restored_root / method
        for method in METHOD_ORDER
        if (args.restored_root / method / args.h5_file).exists()
    ]
    if not method_dirs:
        raise FileNotFoundError(f"No restored method outputs found in {args.restored_root}")

    rows: list[dict[str, object]] = []
    records = list(iter_h5_records(data_path, max_records=args.max_records))
    file_info = parse_dataset_info(data_path.name)

    for selection_name in args.selections:
        channel_names = SELECTIONS[selection_name]
        channel_indices = [SEMI_SIM_CHANNELS.index(ch) for ch in channel_names]

        for method_dir in method_dirs:
            method_key = method_dir.name
            restored_path = method_dir / args.h5_file
            method_label = METHOD_LABELS.get(method_key, method_key)

            with h5py.File(restored_path, "r") as restored_h5:
                for rec in tqdm(
                    records,
                    desc=f"{selection_name}:{method_key}",
                    leave=False,
                ):
                    if rec.record_name not in restored_h5:
                        print(f"Missing record {rec.record_name} in {restored_path}")
                        continue

                    restored = ensure_2d_rows(np.asarray(restored_h5[rec.record_name][()]))
                    restored_sel = restored[channel_indices, :]
                    signal_sel = rec.signal[channel_indices, :]
                    reference_sel = rec.reference[channel_indices, :]

                    metric_rows = evaluate_record(
                        restored=restored_sel,
                        signal=signal_sel,
                        reference=reference_sel,
                        artifact_mask=rec.artifact_mask,
                        fs=rec.fs,
                    )

                    for row in metric_rows:
                        row.update(
                            {
                                "method_key": method_key,
                                "method": method_label,
                                "h5_file": data_path.name,
                                "dataset": rec.dataset_name,
                                "record": rec.record_name,
                                "fs": rec.fs,
                                "n_channels": len(channel_indices),
                                "n_samples": rec.n_samples,
                                "artifact_samples": int(rec.artifact_mask.sum()),
                                "artifact_fraction": float(rec.artifact_mask.mean()),
                                "channel_selection": selection_name,
                                "channel_names": ",".join(channel_names),
                                "channel_indices_zero_based": ",".join(
                                    str(i) for i in channel_indices
                                ),
                                **file_info,
                            }
                        )
                        rows.append(row)

    metrics = finalize_metrics_table(pd.DataFrame(rows))
    metrics = normalize_semisim_metrics(metrics)
    metrics.to_parquet(args.out_dir / "semisim_fp1_fp2_metrics.parquet", index=False)
    metrics.to_csv(args.out_dir / "semisim_fp1_fp2_metrics.csv", index=False)

    summary = summarize_metrics_by_selection(metrics, SUMMARY_METRICS + ["SNR_before"])
    summary.to_parquet(args.out_dir / "summary_by_selection_region_method.parquet", index=False)
    summary.to_csv(args.out_dir / "summary_by_selection_region_method.csv", index=False)

    ranks = rank_methods_by_selection(summary, SUMMARY_METRICS)
    ranks.to_parquet(args.out_dir / "ranks_by_selection_region_metric.parquet", index=False)
    ranks.to_csv(args.out_dir / "ranks_by_selection_region_metric.csv", index=False)

    stats = paired_wilcoxon_by_selection(metrics, STAT_METRICS)
    stats.to_parquet(args.out_dir / "paired_wilcoxon_proposed_vs_baselines.parquet", index=False)
    stats.to_csv(args.out_dir / "paired_wilcoxon_proposed_vs_baselines.csv", index=False)

    artifact_delta = make_artifact_delta_table(summary)
    artifact_delta.to_csv(args.out_dir / "artifact_deltaSNR_by_selection.csv", index=False)

    proposed_comp = make_proposed_comparison(stats)
    proposed_comp.to_csv(
        args.out_dir / "artifact_deltaSNR_proposed_vs_baselines.csv", index=False
    )

    print("Semi-sim FP1/FP2 saved-output analysis completed.")
    print(f"  records      : {len(records)}")
    print(f"  methods      : {', '.join(method_dir.name for method_dir in method_dirs)}")
    print(f"  selections   : {', '.join(args.selections)}")
    print(f"  metric rows  : {len(metrics)}")
    print(f"  outputs      : {args.out_dir}")


def normalize_semisim_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method_short"] = out["method_key"].map(METHOD_LABELS).fillna(out["method_key"])
    out["dataset_label"] = out["h5_file"].map(DATASET_LABELS).fillna(out["dataset"])
    out["method_order"] = out["method_key"].map(
        {m: i for i, m in enumerate(METHOD_ORDER)}
    ).fillna(999)
    out["selection_order"] = out["channel_selection"].map(
        {name: i for i, name in enumerate(SELECTIONS)}
    ).fillna(999)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out.sort_values(
        ["selection_order", "record", "region", "method_order"]
    ).reset_index(drop=True)


def summarize_metrics_by_selection(
    df: pd.DataFrame, metrics: list[str]
) -> pd.DataFrame:
    group_cols = [
        "channel_selection",
        "channel_names",
        "h5_file",
        "dataset_label",
        "benchmark_family",
        "noise_type",
        "nominal_snr_db",
        "region",
        "method_key",
        "method_short",
    ]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        base["n_records"] = int(group["record"].nunique())
        for metric in metrics:
            vals = group[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row = dict(base)
            row["metric"] = metric
            row["direction"] = METRIC_DIRECTIONS.get(metric, 1)
            row["n_finite"] = int(vals.size)
            if vals.size:
                row["mean"] = float(np.mean(vals))
                row["sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                row["median"] = float(np.median(vals))
                row["q1"] = float(np.percentile(vals, 25))
                row["q3"] = float(np.percentile(vals, 75))
                row["iqr"] = row["q3"] - row["q1"]
            else:
                row.update(
                    {
                        "mean": np.nan,
                        "sd": np.nan,
                        "median": np.nan,
                        "q1": np.nan,
                        "q3": np.nan,
                        "iqr": np.nan,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def rank_methods_by_selection(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    parts = []
    for metric in metrics:
        sub = summary[summary["metric"] == metric].copy()
        direction = METRIC_DIRECTIONS[metric]
        sub["rank_value"] = direction * sub["median"]
        sub["rank"] = sub.groupby(
            ["channel_selection", "h5_file", "region", "metric"]
        )["rank_value"].rank(ascending=False, method="min")
        parts.append(sub)
    ranked = pd.concat(parts, ignore_index=True)
    return ranked.drop(columns=["rank_value"]).sort_values(
        ["channel_selection", "h5_file", "region", "metric", "rank", "method_key"]
    )


def paired_wilcoxon_by_selection(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    rows = []
    baselines = [m for m in METHOD_ORDER if m != "proposed_auto"]
    group_cols = ["channel_selection", "channel_names", "h5_file", "dataset_label", "region"]

    for keys, group in df.groupby(group_cols):
        channel_selection, channel_names, h5_file, dataset_label, region = keys
        for metric in metrics:
            direction = METRIC_DIRECTIONS[metric]
            wide = group.pivot_table(
                index="record", columns="method_key", values=metric, aggfunc="mean"
            )
            if "proposed_auto" not in wide:
                continue
            for baseline in baselines:
                if baseline not in wide:
                    continue
                pair = wide[["proposed_auto", baseline]].dropna()
                prop = pair["proposed_auto"].to_numpy(dtype=float)
                base = pair[baseline].to_numpy(dtype=float)
                finite = np.isfinite(prop) & np.isfinite(base)
                prop = prop[finite]
                base = base[finite]
                raw_diff = prop - base
                oriented_diff = direction * raw_diff
                n = int(oriented_diff.size)
                if n == 0:
                    p_two = np.nan
                    p_better = np.nan
                    stat = np.nan
                elif np.allclose(oriented_diff, 0, equal_nan=False):
                    p_two = 1.0
                    p_better = 1.0
                    stat = 0.0
                else:
                    try:
                        stat, p_two = wilcoxon(
                            raw_diff, alternative="two-sided", zero_method="wilcox"
                        )
                    except ValueError:
                        stat, p_two = np.nan, np.nan
                    try:
                        _, p_better = wilcoxon(
                            oriented_diff, alternative="greater", zero_method="wilcox"
                        )
                    except ValueError:
                        p_better = np.nan

                rows.append(
                    {
                        "channel_selection": channel_selection,
                        "channel_names": channel_names,
                        "h5_file": h5_file,
                        "dataset_label": dataset_label,
                        "region": region,
                        "metric": metric,
                        "direction": direction,
                        "baseline_method_key": baseline,
                        "baseline_method": METHOD_LABELS.get(baseline, baseline),
                        "n_pairs": n,
                        "proposed_median": safe_median(prop),
                        "baseline_median": safe_median(base),
                        "median_raw_difference": safe_median(raw_diff),
                        "median_oriented_difference": safe_median(oriented_diff),
                        "mean_oriented_difference": safe_mean(oriented_diff),
                        "fraction_proposed_better": safe_mean(oriented_diff > 0),
                        "fraction_tied": safe_mean(np.isclose(oriented_diff, 0)),
                        "wilcoxon_statistic": stat,
                        "p_two_sided": p_two,
                        "p_one_sided_proposed_better": p_better,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_two_sided_holm"] = holm_adjust(out["p_two_sided"].to_numpy(dtype=float))
    out["p_one_sided_holm"] = holm_adjust(
        out["p_one_sided_proposed_better"].to_numpy(dtype=float)
    )
    return out.sort_values(
        ["channel_selection", "h5_file", "region", "metric", "baseline_method_key"]
    )


def make_artifact_delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    sub = summary[
        (summary["region"] == "artifact") & (summary["metric"] == "deltaSNR")
    ].copy()
    sub["rank"] = sub.groupby("channel_selection")["median"].rank(
        ascending=False, method="min"
    )
    sub["median_iqr"] = sub.apply(
        lambda r: f"{r['median']:.3f} [{r['q1']:.3f}, {r['q3']:.3f}]",
        axis=1,
    )
    return sub[
        [
            "channel_selection",
            "channel_names",
            "method_key",
            "method_short",
            "n_records",
            "median",
            "q1",
            "q3",
            "median_iqr",
            "rank",
        ]
    ].sort_values(["channel_selection", "rank", "method_key"])


def make_proposed_comparison(stats: pd.DataFrame) -> pd.DataFrame:
    sub = stats[
        (stats["region"] == "artifact") & (stats["metric"] == "deltaSNR")
    ].copy()
    if sub.empty:
        return sub
    sub["baseline_method"] = sub["baseline_method_key"].map(METHOD_LABELS).fillna(
        sub["baseline_method_key"]
    )
    sub["proposed_minus_baseline_median"] = (
        sub["proposed_median"] - sub["baseline_median"]
    )
    return sub[
        [
            "channel_selection",
            "baseline_method_key",
            "baseline_method",
            "n_pairs",
            "proposed_median",
            "baseline_median",
            "proposed_minus_baseline_median",
            "fraction_proposed_better",
            "p_one_sided_proposed_better",
            "p_one_sided_holm",
        ]
    ].sort_values(["channel_selection", "baseline_method_key"])


if __name__ == "__main__":
    main()
