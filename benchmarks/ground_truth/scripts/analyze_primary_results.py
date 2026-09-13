from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METHOD_ORDER = [
    "proposed_auto",
    "wqn",
    "wt_soft",
    "wt_hard",
    "emd_cca",
    "emd_ica",
]

METHOD_LABELS = {
    "proposed_auto": "Proposed",
    "wqn": "WQN",
    "wt_soft": "WT-soft",
    "wt_hard": "WT-hard",
    "emd_cca": "EMD-CCA",
    "emd_ica": "EMD-ICA",
}

DATASET_LABELS = {
    "01_physiobank.h5": "PhysioBank motion",
    "02_semisimulated_eog.h5": "Semi-sim EOG",
    "03_denoise-net_emg_-20dB.h5": "DenoiseNet EMG",
    "03_denoise-net_eog_-20dB.h5": "DenoiseNet EOG",
    "03_denoise-net_eog+emg_-20dB.h5": "DenoiseNet EOG+EMG",
}

METRIC_DIRECTIONS = {
    "SNR_after": 1,
    "deltaSNR": 1,
    "NMSE_after": -1,
    "deltaNMSE": -1,
    "RMSE_after": -1,
    "RRMSE_after": -1,
    "R_after": 1,
    "deltaR": 1,
    "Coh_after": 1,
    "deltaCoh": 1,
    "PSDerr_after": -1,
    "deltaPSDerr": -1,
    "MI_after": 1,
    "OutputChange_RRMSE": -1,
}

SUMMARY_METRICS = [
    "SNR_after",
    "deltaSNR",
    "RRMSE_after",
    "R_after",
    "Coh_after",
    "PSDerr_after",
    "OutputChange_RRMSE",
]

STAT_METRICS = [
    "deltaSNR",
    "RRMSE_after",
    "R_after",
    "Coh_after",
    "PSDerr_after",
    "OutputChange_RRMSE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine and analyze primary benchmark metrics."
    )
    parser.add_argument("--sota-dir", type=Path, default=Path("results/metrics_sota_primary"))
    parser.add_argument(
        "--proposed-dir", type=Path, default=Path("results/metrics_proposed_primary")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/analysis_primary")
    )
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    metrics = load_all_metrics(args.sota_dir, args.proposed_dir)
    metrics = normalize_metrics_table(metrics)

    combined_path = args.out_dir / "combined_primary_metrics.parquet"
    metrics.to_parquet(combined_path, index=False)
    metrics.to_csv(args.out_dir / "combined_primary_metrics.csv", index=False)

    summary = summarize_metrics(metrics, SUMMARY_METRICS)
    summary.to_csv(args.out_dir / "summary_by_dataset_region_method.csv", index=False)
    summary.to_parquet(args.out_dir / "summary_by_dataset_region_method.parquet", index=False)

    ranks = rank_methods(summary, SUMMARY_METRICS)
    ranks.to_csv(args.out_dir / "method_ranks_by_dataset_region_metric.csv", index=False)
    ranks.to_parquet(args.out_dir / "method_ranks_by_dataset_region_metric.parquet", index=False)

    stats = paired_wilcoxon_tests(metrics, STAT_METRICS)
    stats.to_csv(args.out_dir / "paired_wilcoxon_proposed_vs_baselines.csv", index=False)
    stats.to_parquet(args.out_dir / "paired_wilcoxon_proposed_vs_baselines.parquet", index=False)

    compact = compact_primary_table(summary)
    compact.to_csv(args.out_dir / "compact_primary_table.csv", index=False)

    if not args.no_figures:
        make_all_figures(metrics, ranks, fig_dir)

    print("Primary analysis completed.")
    print(f"  combined rows: {len(metrics)}")
    print(f"  methods      : {', '.join(metrics['method_key'].drop_duplicates())}")
    print(f"  outputs      : {args.out_dir}")
    print(f"  figures      : {fig_dir}")


def load_all_metrics(sota_dir: Path, proposed_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(sota_dir.glob("*.parquet")):
        frames.append(pd.read_parquet(path))
    for path in sorted(proposed_dir.glob("metrics__proposed_auto__*.parquet")):
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError("No metric parquet files were found.")
    return pd.concat(frames, ignore_index=True)


def normalize_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method_key"] = out["method_key"].astype(str)
    out["method_short"] = out["method_key"].map(METHOD_LABELS).fillna(out["method"])
    out["dataset_label"] = out["h5_file"].map(DATASET_LABELS).fillna(out["dataset"])
    out["method_order"] = out["method_key"].map(
        {m: i for i, m in enumerate(METHOD_ORDER)}
    ).fillna(999)
    out["dataset_order"] = out["h5_file"].map(
        {name: i for i, name in enumerate(DATASET_LABELS)}
    ).fillna(999)

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out.sort_values(
        ["dataset_order", "record", "region", "method_order"]
    ).reset_index(drop=True)


def summarize_metrics(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    group_cols = [
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
            row["direction"] = METRIC_DIRECTIONS[metric]
            row["n_finite"] = int(vals.size)
            if vals.size:
                row["mean"] = float(np.mean(vals))
                row["sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                row["median"] = float(np.median(vals))
                row["q1"] = float(np.percentile(vals, 25))
                row["q3"] = float(np.percentile(vals, 75))
                row["iqr"] = row["q3"] - row["q1"]
            else:
                row.update({"mean": np.nan, "sd": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


def rank_methods(summary: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    parts = []
    for metric in metrics:
        sub = summary[summary["metric"] == metric].copy()
        direction = METRIC_DIRECTIONS[metric]
        sub["rank_value"] = direction * sub["median"]
        sub["rank"] = sub.groupby(["h5_file", "region", "metric"])["rank_value"].rank(
            ascending=False, method="min"
        )
        parts.append(sub)
    ranked = pd.concat(parts, ignore_index=True)
    return ranked.drop(columns=["rank_value"]).sort_values(
        ["h5_file", "region", "metric", "rank", "method_key"]
    )


def paired_wilcoxon_tests(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    baselines = [m for m in METHOD_ORDER if m != "proposed_auto"]
    group_cols = ["h5_file", "dataset_label", "region"]

    for (h5_file, dataset_label, region), group in df.groupby(group_cols):
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
                        stat, p_two = wilcoxon(raw_diff, alternative="two-sided", zero_method="wilcox")
                    except ValueError:
                        stat, p_two = np.nan, np.nan
                    try:
                        _, p_better = wilcoxon(oriented_diff, alternative="greater", zero_method="wilcox")
                    except ValueError:
                        p_better = np.nan

                rows.append(
                    {
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
    return out.sort_values(["h5_file", "region", "metric", "baseline_method_key"])


def compact_primary_table(summary: pd.DataFrame) -> pd.DataFrame:
    chosen = summary[
        summary["metric"].isin(["deltaSNR", "RRMSE_after", "R_after", "OutputChange_RRMSE"])
    ].copy()
    chosen["median_iqr"] = chosen.apply(
        lambda r: f"{r['median']:.3g} [{r['q1']:.3g}, {r['q3']:.3g}]"
        if np.isfinite(r["median"])
        else "",
        axis=1,
    )
    return chosen[
        [
            "dataset_label",
            "region",
            "method_short",
            "metric",
            "n_records",
            "median_iqr",
            "median",
            "q1",
            "q3",
        ]
    ].sort_values(["dataset_label", "region", "metric", "method_short"])


def make_all_figures(df: pd.DataFrame, ranks: pd.DataFrame, fig_dir: Path) -> None:
    plot_box_by_dataset(
        df[df["region"] == "artifact"],
        "deltaSNR",
        "Artifact-region SNR improvement",
        "Delta SNR (dB)",
        fig_dir / "artifact_deltaSNR_by_dataset",
    )
    plot_box_by_dataset(
        df[df["region"] == "clean"],
        "OutputChange_RRMSE",
        "Clean-region output change",
        "Relative L2 change",
        fig_dir / "clean_output_change_by_dataset",
    )
    plot_box_by_dataset(
        df[df["region"] == "whole"],
        "RRMSE_after",
        "Whole-record reconstruction error",
        "RRMSE after denoising",
        fig_dir / "whole_rrmse_after_by_dataset",
    )
    plot_rank_heatmap(
        ranks,
        ["deltaSNR", "RRMSE_after", "R_after", "OutputChange_RRMSE"],
        fig_dir / "median_rank_heatmap",
    )


def plot_box_by_dataset(
    df: pd.DataFrame, metric: str, title: str, ylabel: str, out_base: Path
) -> None:
    datasets = list(DATASET_LABELS.values())
    methods = [m for m in METHOD_ORDER if m in set(df["method_key"])]
    colors = {
        "proposed_auto": "#1f77b4",
        "wqn": "#ff7f0e",
        "wt_soft": "#2ca02c",
        "wt_hard": "#9467bd",
        "emd_cca": "#8c564b",
        "emd_ica": "#7f7f7f",
    }

    fig, axes = plt.subplots(1, len(datasets), figsize=(16, 4.8), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset_label"] == dataset]
        data = []
        labels = []
        used_colors = []
        for method in methods:
            vals = sub[sub["method_key"] == method][metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                data.append(vals)
                labels.append(METHOD_LABELS.get(method, method))
                used_colors.append(colors.get(method, "#333333"))
        if data:
            bp = ax.boxplot(data, patch_artist=True, showfliers=False)
            for patch, color in zip(bp["boxes"], used_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.72)
            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(1.2)
        ax.set_title(dataset, fontsize=10)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, out_base)
    plt.close(fig)


def plot_rank_heatmap(ranks: pd.DataFrame, metrics: list[str], out_base: Path) -> None:
    sub = ranks[(ranks["region"].isin(["artifact", "clean", "whole"])) & ranks["metric"].isin(metrics)].copy()
    sub["row"] = sub["dataset_label"] + " | " + sub["region"] + " | " + sub["metric"]
    pivot = sub.pivot_table(index="row", columns="method_short", values="rank", aggfunc="mean")
    method_cols = [METHOD_LABELS[m] for m in METHOD_ORDER if METHOD_LABELS[m] in pivot.columns]
    pivot = pivot.reindex(columns=method_cols)

    fig_height = max(5.0, 0.28 * len(pivot))
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis_r", aspect="auto", vmin=1, vmax=len(method_cols))
    ax.set_xticks(np.arange(len(method_cols)))
    ax.set_xticklabels(method_cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color="white" if val > 3 else "black", fontsize=7)
    ax.set_title("Median-performance ranks (1 = best)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    save_figure(fig, out_base)
    plt.close(fig)


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    for suffix, kwargs in [(".png", {"dpi": 300}), (".pdf", {})]:
        path = out_base.with_suffix(suffix)
        try:
            fig.savefig(path, **kwargs)
        except PermissionError:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alt_path = out_base.with_name(f"{out_base.name}_{stamp}").with_suffix(suffix)
            fig.savefig(alt_path, **kwargs)
            print(f"Warning: {path} was locked; saved {alt_path} instead.")


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(p))
    if finite_idx.size == 0:
        return out

    finite_p = p[finite_idx]
    order = np.argsort(finite_p)
    sorted_idx = finite_idx[order]
    sorted_p = finite_p[order]
    m = sorted_p.size
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for i, pv in enumerate(sorted_p):
        adj = (m - i) * pv
        running = max(running, adj)
        adjusted_sorted[i] = min(running, 1.0)
    out[sorted_idx] = adjusted_sorted
    return out


def safe_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else np.nan


def safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


if __name__ == "__main__":
    main()
