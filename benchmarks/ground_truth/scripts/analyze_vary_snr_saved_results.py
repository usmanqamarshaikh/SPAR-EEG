from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["proposed_auto", "wqn", "wt_hard", "wt_soft", "emd_cca", "emd_ica"]
METHOD_LABELS = {
    "proposed_auto": "Proposed",
    "wqn": "WQN",
    "wt_hard": "WT-hard",
    "wt_soft": "WT-soft",
    "emd_cca": "EMD-CCA",
    "emd_ica": "EMD-ICA",
}
NOISE_ORDER = ["emg", "eog", "eog+emg"]
NOISE_LABELS = {"emg": "EMG", "eog": "EOG", "eog+emg": "EOG+EMG"}
COLORS = {
    "proposed_auto": "#1f77b4",
    "wqn": "#ff7f0e",
    "wt_hard": "#9467bd",
    "wt_soft": "#2ca02c",
    "emd_cca": "#8c564b",
    "emd_ica": "#7f7f7f",
}
MARKERS = {
    "proposed_auto": "o",
    "wqn": "s",
    "wt_hard": "D",
    "wt_soft": "^",
    "emd_cca": "v",
    "emd_ica": "P",
}
METRICS = {
    "deltaSNR": ("Delta SNR (dB)", "higher"),
    "SNR_after": ("SNR after denoising (dB)", "higher"),
    "RRMSE_after": ("RRMSE after denoising", "lower"),
    "R_after": ("Correlation after denoising", "higher"),
    "PSDerr_after": ("PSD error after denoising", "lower"),
    "OutputChange_RRMSE": ("Output change RRMSE", "lower"),
}
EXCLUDED_NOMINAL_SNRS = {-0.5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze DenoiseNet varying-SNR metrics from saved restored outputs."
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("results/metrics_vary_snr_saved"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/analysis_vary_snr_saved"),
    )
    parser.add_argument(
        "--stat",
        choices=["median", "mean"],
        default="median",
        help="Central tendency used in figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    summary = build_summary(args.metrics_root)
    summary.to_csv(args.out_dir / "summary_by_noise_snr_region_method.csv", index=False)
    summary.to_parquet(args.out_dir / "summary_by_noise_snr_region_method.parquet", index=False)

    ranks = rank_summary(summary)
    ranks.to_csv(args.out_dir / "ranks_by_noise_snr_region_metric.csv", index=False)
    ranks.to_parquet(args.out_dir / "ranks_by_noise_snr_region_metric.parquet", index=False)

    delta_table = compact_delta_snr_table(summary)
    delta_table.to_csv(args.out_dir / "artifact_deltaSNR_curve_table.csv", index=False)

    auc_table = area_under_curve_table(summary)
    auc_table.to_csv(args.out_dir / "artifact_deltaSNR_auc_summary.csv", index=False)

    make_delta_snr_panel(summary, args.stat, fig_dir)
    make_metric_panel(summary, "artifact", ["SNR_after", "RRMSE_after", "R_after", "PSDerr_after"], args.stat, fig_dir)
    make_metric_panel(summary, "clean", ["OutputChange_RRMSE"], args.stat, fig_dir)
    make_rank_heatmap(ranks, fig_dir)

    print("Varying-SNR saved-output analysis completed.")
    print(f"  summary rows : {len(summary)}")
    print(f"  ranks rows   : {len(ranks)}")
    print(f"  outputs      : {args.out_dir}")
    print(f"  figures      : {fig_dir}")


def build_summary(metrics_root: Path) -> pd.DataFrame:
    paths = sorted(metrics_root.glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No metric parquet files found under {metrics_root}")

    needed = [
        *METRICS,
        "region",
        "method_key",
        "method",
        "h5_file",
        "record",
        "benchmark_family",
        "noise_type",
        "nominal_snr_db",
    ]
    rows: list[dict[str, object]] = []
    for path in paths:
        df = pd.read_parquet(path, columns=needed)
        df = df[df["benchmark_family"].astype(str).str.lower() == "denoise-net"].copy()
        df = exclude_nominal_snrs(df)
        if df.empty:
            continue

        df["method_key"] = df["method_key"].astype(str)
        df["noise_type"] = df["noise_type"].astype(str).str.lower()
        df["region"] = df["region"].astype(str)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        group_cols = ["noise_type", "nominal_snr_db", "region", "method_key"]
        for keys, group in df.groupby(group_cols, dropna=False):
            base = dict(zip(group_cols, keys))
            base["noise_label"] = NOISE_LABELS.get(base["noise_type"], base["noise_type"])
            base["method_short"] = METHOD_LABELS.get(base["method_key"], base["method_key"])
            base["n_records"] = int(group["record"].nunique())
            base["n_h5_files"] = int(group["h5_file"].nunique())
            for metric in METRICS:
                vals = group[metric].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                row = dict(base)
                row["metric"] = metric
                row["direction"] = metric_direction(metric)
                row["n_finite"] = int(vals.size)
                if vals.size:
                    row["mean"] = float(np.mean(vals))
                    row["sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                    row["sem"] = float(row["sd"] / np.sqrt(vals.size)) if vals.size > 1 else 0.0
                    row["median"] = float(np.median(vals))
                    row["q1"] = float(np.percentile(vals, 25))
                    row["q3"] = float(np.percentile(vals, 75))
                    row["iqr"] = row["q3"] - row["q1"]
                else:
                    row.update({"mean": np.nan, "sd": np.nan, "sem": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan})
                rows.append(row)

    out = pd.DataFrame(rows)
    out["noise_order"] = out["noise_type"].map({n: i for i, n in enumerate(NOISE_ORDER)}).fillna(999)
    out["method_order"] = out["method_key"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(999)
    return out.sort_values(
        ["noise_order", "nominal_snr_db", "region", "metric", "method_order"]
    ).reset_index(drop=True)


def exclude_nominal_snrs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "nominal_snr_db" not in df:
        return df
    snr = pd.to_numeric(df["nominal_snr_db"], errors="coerce")
    keep = np.ones(len(df), dtype=bool)
    for excluded in EXCLUDED_NOMINAL_SNRS:
        keep &= ~np.isclose(snr.to_numpy(dtype=float), excluded, atol=1e-9, rtol=0)
    return df.loc[keep].copy()


def rank_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["rank_value"] = out["direction"] * out["median"]
    out["rank"] = out.groupby(["noise_type", "nominal_snr_db", "region", "metric"])[
        "rank_value"
    ].rank(ascending=False, method="min")
    return out.drop(columns=["rank_value"]).sort_values(
        ["noise_order", "nominal_snr_db", "region", "metric", "rank", "method_order"]
    )


def compact_delta_snr_table(summary: pd.DataFrame) -> pd.DataFrame:
    sub = summary[(summary["region"] == "artifact") & (summary["metric"] == "deltaSNR")].copy()
    sub["median_iqr"] = sub.apply(
        lambda r: f"{r['median']:.3f} [{r['q1']:.3f}, {r['q3']:.3f}]", axis=1
    )
    return sub[
        [
            "noise_type",
            "noise_label",
            "nominal_snr_db",
            "method_key",
            "method_short",
            "n_records",
            "median",
            "q1",
            "q3",
            "median_iqr",
        ]
    ]


def area_under_curve_table(summary: pd.DataFrame) -> pd.DataFrame:
    sub = summary[(summary["region"] == "artifact") & (summary["metric"] == "deltaSNR")].copy()
    rows = []
    for (noise_type, method_key), group in sub.groupby(["noise_type", "method_key"]):
        group = group.sort_values("nominal_snr_db")
        x = group["nominal_snr_db"].to_numpy(dtype=float)
        y = group["median"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        rows.append(
            {
                "noise_type": noise_type,
                "noise_label": NOISE_LABELS.get(noise_type, noise_type),
                "method_key": method_key,
                "method_short": METHOD_LABELS.get(method_key, method_key),
                "n_snr_points": int(x.size),
                "mean_median_deltaSNR": float(np.mean(y)) if y.size else np.nan,
                "auc_deltaSNR": float(np.trapezoid(y, x)) if y.size > 1 else np.nan,
                "min_median_deltaSNR": float(np.min(y)) if y.size else np.nan,
                "max_median_deltaSNR": float(np.max(y)) if y.size else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["rank_mean_deltaSNR"] = out.groupby("noise_type")["mean_median_deltaSNR"].rank(
        ascending=False, method="min"
    )
    out["noise_order"] = out["noise_type"].map({n: i for i, n in enumerate(NOISE_ORDER)}).fillna(999)
    out["method_order"] = out["method_key"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(999)
    return out.sort_values(["noise_order", "rank_mean_deltaSNR", "method_order"]).drop(
        columns=["noise_order", "method_order"]
    )


def make_delta_snr_panel(summary: pd.DataFrame, stat: str, fig_dir: Path) -> None:
    sub = summary[(summary["region"] == "artifact") & (summary["metric"] == "deltaSNR")]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), sharey=True)
    for ax, noise_type in zip(axes, NOISE_ORDER):
        plot_curves_on_axis(ax, sub[sub["noise_type"] == noise_type], "deltaSNR", stat)
        ax.set_title(f"DenoiseNet {NOISE_LABELS[noise_type]}")
        ax.set_xlabel("Input SNR (dB)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(METRICS["deltaSNR"][0])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.suptitle("Artifact-region SNR improvement across input SNR")
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    save_figure(fig, fig_dir / f"artifact_deltaSNR_3panel_{stat}")
    plt.close(fig)


def make_metric_panel(
    summary: pd.DataFrame, region: str, metrics: list[str], stat: str, fig_dir: Path
) -> None:
    for metric in metrics:
        if metric not in METRICS:
            continue
        sub = summary[(summary["region"] == region) & (summary["metric"] == metric)]
        fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), sharey=False)
        for ax, noise_type in zip(axes, NOISE_ORDER):
            plot_curves_on_axis(ax, sub[sub["noise_type"] == noise_type], metric, stat)
            ax.set_title(f"DenoiseNet {NOISE_LABELS[noise_type]}")
            ax.set_xlabel("Input SNR (dB)")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel(METRICS[metric][0])
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
        fig.suptitle(f"{region.title()} region: {METRICS[metric][0]}")
        fig.tight_layout(rect=[0, 0.12, 1, 0.92])
        save_figure(fig, fig_dir / f"{region}_{metric}_3panel_{stat}")
        plt.close(fig)


def plot_curves_on_axis(ax: plt.Axes, sub: pd.DataFrame, metric: str, stat: str) -> None:
    for method in METHOD_ORDER:
        sm = sub[sub["method_key"] == method].sort_values("nominal_snr_db")
        if sm.empty:
            continue
        x = sm["nominal_snr_db"].to_numpy(dtype=float)
        y = sm[stat].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            "-",
            marker=MARKERS.get(method, "o"),
            color=COLORS.get(method, "#333333"),
            linewidth=1.8,
            markersize=3.8,
            label=METHOD_LABELS.get(method, method),
        )
        if stat == "median":
            y1 = sm["q1"].to_numpy(dtype=float)
            y2 = sm["q3"].to_numpy(dtype=float)
            ax.fill_between(x, y1, y2, color=COLORS.get(method, "#333333"), alpha=0.06, linewidth=0)
        elif stat == "mean":
            sem = sm["sem"].to_numpy(dtype=float)
            ax.fill_between(x, y - sem, y + sem, color=COLORS.get(method, "#333333"), alpha=0.08, linewidth=0)


def make_rank_heatmap(ranks: pd.DataFrame, fig_dir: Path) -> None:
    sub = ranks[(ranks["region"] == "artifact") & (ranks["metric"] == "deltaSNR")].copy()
    sub["row"] = sub["noise_label"] + " | " + sub["nominal_snr_db"].map(lambda x: f"{x:g} dB")
    pivot = sub.pivot_table(index="row", columns="method_short", values="rank", aggfunc="mean")
    ordered_rows = []
    for noise_type in NOISE_ORDER:
        noise_label = NOISE_LABELS[noise_type]
        snrs = sorted(sub[sub["noise_type"] == noise_type]["nominal_snr_db"].unique())
        ordered_rows.extend([f"{noise_label} | {snr:g} dB" for snr in snrs])
    ordered_cols = [METHOD_LABELS[m] for m in METHOD_ORDER if METHOD_LABELS[m] in pivot.columns]
    pivot = pivot.reindex(index=ordered_rows, columns=ordered_cols)

    fig, ax = plt.subplots(figsize=(8.2, max(6, 0.18 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis_r", aspect="auto", vmin=1, vmax=len(ordered_cols))
    ax.set_xticks(np.arange(len(ordered_cols)))
    ax.set_xticklabels(ordered_cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6.5)
    ax.set_title("Artifact delta SNR rank across SNR levels (1 = best)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    save_figure(fig, fig_dir / "artifact_deltaSNR_rank_heatmap")
    plt.close(fig)


def metric_direction(metric: str) -> int:
    return 1 if METRICS[metric][1] == "higher" else -1


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


if __name__ == "__main__":
    main()
