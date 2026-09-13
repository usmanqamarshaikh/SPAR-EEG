from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyze_primary_results import (
    METHOD_LABELS,
    STAT_METRICS,
    SUMMARY_METRICS,
    compact_primary_table,
    make_all_figures,
    normalize_metrics_table,
    paired_wilcoxon_tests,
    rank_methods,
    summarize_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze primary benchmark metrics generated from saved restored H5 outputs."
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("results/metrics_primary_saved"),
        help="Root containing one metrics subfolder per method.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/analysis_primary_saved"),
        help="Directory for combined tables, summaries, statistics, and figures.",
    )
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    metrics = load_saved_metrics(args.metrics_root)
    metrics = normalize_metrics_table(metrics)

    metrics.to_parquet(args.out_dir / "combined_primary_saved_metrics.parquet", index=False)
    metrics.to_csv(args.out_dir / "combined_primary_saved_metrics.csv", index=False)

    summary = summarize_metrics(metrics, SUMMARY_METRICS)
    summary.to_parquet(args.out_dir / "summary_by_dataset_region_method.parquet", index=False)
    summary.to_csv(args.out_dir / "summary_by_dataset_region_method.csv", index=False)

    ranks = rank_methods(summary, SUMMARY_METRICS)
    ranks.to_parquet(args.out_dir / "method_ranks_by_dataset_region_metric.parquet", index=False)
    ranks.to_csv(args.out_dir / "method_ranks_by_dataset_region_metric.csv", index=False)

    stats = paired_wilcoxon_tests(metrics, STAT_METRICS)
    stats.to_parquet(args.out_dir / "paired_wilcoxon_proposed_vs_baselines.parquet", index=False)
    stats.to_csv(args.out_dir / "paired_wilcoxon_proposed_vs_baselines.csv", index=False)

    compact = compact_primary_table(summary)
    compact.to_csv(args.out_dir / "compact_primary_table.csv", index=False)

    artifact_delta = make_artifact_delta_snr_table(summary)
    artifact_delta.to_csv(args.out_dir / "artifact_deltaSNR_median_iqr.csv", index=False)

    proposed_delta = make_proposed_delta_snr_comparison(stats)
    proposed_delta.to_csv(args.out_dir / "artifact_deltaSNR_proposed_vs_baselines.csv", index=False)

    if not args.no_figures:
        make_all_figures(metrics, ranks, fig_dir)

    print("Saved-output primary analysis completed.")
    print(f"  combined rows: {len(metrics)}")
    print(f"  metric files : {len(list(args.metrics_root.rglob('*.parquet')))} parquet")
    print(f"  outputs      : {args.out_dir}")
    if not args.no_figures:
        print(f"  figures      : {fig_dir}")


def load_saved_metrics(metrics_root: Path) -> pd.DataFrame:
    paths = sorted(metrics_root.glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet metric files found under {metrics_root}")
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def make_artifact_delta_snr_table(summary: pd.DataFrame) -> pd.DataFrame:
    sub = summary[
        (summary["region"] == "artifact") & (summary["metric"] == "deltaSNR")
    ].copy()
    sub["median_iqr"] = sub.apply(
        lambda r: f"{r['median']:.3f} [{r['q1']:.3f}, {r['q3']:.3f}]",
        axis=1,
    )
    sub["rank"] = sub.groupby("dataset_label")["median"].rank(
        ascending=False, method="min"
    )
    return sub[
        [
            "dataset_label",
            "h5_file",
            "method_key",
            "method_short",
            "n_records",
            "median",
            "q1",
            "q3",
            "median_iqr",
            "rank",
        ]
    ].sort_values(["dataset_label", "rank", "method_key"])


def make_proposed_delta_snr_comparison(stats: pd.DataFrame) -> pd.DataFrame:
    sub = stats[
        (stats["region"] == "artifact") & (stats["metric"] == "deltaSNR")
    ].copy()
    if sub.empty:
        return sub
    sub["proposed_minus_baseline_median"] = (
        sub["proposed_median"] - sub["baseline_median"]
    )
    sub["baseline_method"] = sub["baseline_method_key"].map(METHOD_LABELS).fillna(
        sub["baseline_method_key"]
    )
    return sub[
        [
            "dataset_label",
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
    ].sort_values(["dataset_label", "baseline_method_key"])


if __name__ == "__main__":
    main()
