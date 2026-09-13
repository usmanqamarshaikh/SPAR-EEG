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


METHOD_ORDER = ["proposed_auto", "wqn", "wt_soft", "wt_hard", "emd_cca", "emd_ica"]
METHOD_LABELS = {
    "proposed_auto": "Proposed",
    "wqn": "WQN",
    "wt_soft": "WT-soft",
    "wt_hard": "WT-hard",
    "emd_cca": "EMD-CCA",
    "emd_ica": "EMD-ICA",
}
NOISE_LABELS = {
    "emg": "EMG",
    "eog": "EOG",
    "eog+emg": "EOG+EMG",
}
COLORS = {
    "proposed_auto": "#1f77b4",
    "wqn": "#ff7f0e",
    "wt_soft": "#2ca02c",
    "wt_hard": "#9467bd",
    "emd_cca": "#8c564b",
    "emd_ica": "#7f7f7f",
}
MARKERS = {
    "proposed_auto": "o",
    "wqn": "s",
    "wt_soft": "^",
    "wt_hard": "D",
    "emd_cca": "v",
    "emd_ica": "P",
}
METRICS = {
    "deltaSNR": ("Delta SNR (dB)", "higher"),
    "NMSE_after": ("NMSE after (dB)", "lower"),
    "deltaR": ("Delta R", "higher"),
    "deltaCoh_normalized": ("Icoh", "higher"),
    "RRMSE_after": ("RRMSE after", "lower"),
    "PSDerr_after": ("PSD error after", "lower"),
}
EXCLUDED_NOMINAL_SNRS = {-0.5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DenoiseNet varying-SNR curves from common metric files. "
            "The default view uses only the artifact-mask region."
        )
    )
    parser.add_argument("--sota-dir", type=Path, default=Path("results/metrics_sota_vary-snr"))
    parser.add_argument(
        "--proposed-dir", type=Path, default=Path("results/metrics_proposed_vary-snr")
    )
    parser.add_argument(
        "--primary-sota-dir", type=Path, default=Path("results/metrics_sota_primary")
    )
    parser.add_argument(
        "--primary-proposed-dir", type=Path, default=Path("results/metrics_proposed_primary")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis_snr_curves"))
    parser.add_argument("--region", default="artifact", choices=["artifact", "clean", "whole"])
    parser.add_argument("--stat", default="mean", choices=["mean", "median"])
    parser.add_argument("--include-primary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metrics", nargs="+", default=["deltaSNR", "NMSE_after", "deltaR", "deltaCoh_normalized"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    combined = load_metric_sources(args)
    if combined.empty:
        raise SystemExit("No DenoiseNet metric rows found. Run the varying-SNR benchmark first.")

    combined = normalize_table(combined, args.region)
    combined.to_parquet(args.out_dir / f"denoisenet_{args.region}_combined_metrics.parquet", index=False)
    combined.to_csv(args.out_dir / f"denoisenet_{args.region}_combined_metrics.csv", index=False)

    summary = summarize(combined, args.metrics)
    summary_path = args.out_dir / f"denoisenet_{args.region}_snr_summary.csv"
    summary.to_csv(summary_path, index=False)
    summary.to_parquet(summary_path.with_suffix(".parquet"), index=False)

    available_snr = summary.groupby("noise_type")["nominal_snr_db"].nunique().to_dict()
    if any(n < 2 for n in available_snr.values()):
        print("Warning: at least one noise type has fewer than two SNR points.")
        print("Available SNR counts:", available_snr)

    for metric in args.metrics:
        if metric not in METRICS:
            print(f"Skipping unknown metric: {metric}")
            continue
        plot_metric_curves(summary, metric, args.stat, fig_dir)

    plot_panel_curves(summary, args.metrics[:4], args.stat, fig_dir)

    print("DenoiseNet SNR curve analysis completed.")
    print(f"  rows       : {len(combined)}")
    print(f"  summary    : {summary_path}")
    print(f"  figures    : {fig_dir}")


def load_metric_sources(args: argparse.Namespace) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    sources = []
    if args.include_primary:
        sources.extend(
            [
                (args.primary_sota_dir, "primary", 1),
                (args.primary_proposed_dir, "primary", 1),
            ]
        )
    sources.extend(
        [
            (args.sota_dir, "vary_snr", 2),
            (args.proposed_dir, "vary_snr", 2),
        ]
    )

    for folder, source_name, priority in sources:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.parquet")):
            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                print(f"Skipping unreadable file {path}: {exc}")
                continue
            if "benchmark_family" not in df or "region" not in df:
                continue
            df = df[df["benchmark_family"].astype(str).str.lower() == "denoise-net"].copy()
            if df.empty:
                continue
            df["metric_source"] = source_name
            df["source_priority"] = priority
            df["metric_file"] = str(path)
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("source_priority")
    dedup_cols = ["h5_file", "record", "region", "method_key"]
    out = out.drop_duplicates(subset=dedup_cols, keep="last")
    return out


def normalize_table(df: pd.DataFrame, region: str) -> pd.DataFrame:
    out = df[df["region"] == region].copy()
    out = exclude_nominal_snrs(out)
    out["method_key"] = out["method_key"].astype(str)
    out["method_short"] = out["method_key"].map(METHOD_LABELS).fillna(out["method"])
    out["noise_type"] = out["noise_type"].astype(str).str.lower()
    out["noise_label"] = out["noise_type"].map(NOISE_LABELS).fillna(out["noise_type"])
    out["method_order"] = out["method_key"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(999)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["noise_type", "nominal_snr_db", "method_order", "record"]).reset_index(drop=True)


def exclude_nominal_snrs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "nominal_snr_db" not in df:
        return df
    snr = pd.to_numeric(df["nominal_snr_db"], errors="coerce")
    keep = np.ones(len(df), dtype=bool)
    for excluded in EXCLUDED_NOMINAL_SNRS:
        keep &= ~np.isclose(snr.to_numpy(dtype=float), excluded, atol=1e-9, rtol=0)
    return df.loc[keep].copy()


def summarize(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    group_cols = ["noise_type", "noise_label", "nominal_snr_db", "method_key", "method_short"]
    for keys, group in df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        base["n_records"] = int(group["record"].nunique())
        base["h5_files"] = ",".join(sorted(group["h5_file"].dropna().astype(str).unique()))
        for metric in metrics:
            if metric not in group:
                continue
            vals = group[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row = dict(base)
            row["metric"] = metric
            row["n_finite"] = int(vals.size)
            if vals.size:
                row["mean"] = float(np.mean(vals))
                row["sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                row["sem"] = float(row["sd"] / np.sqrt(vals.size)) if vals.size > 1 else 0.0
                row["median"] = float(np.median(vals))
                row["q1"] = float(np.percentile(vals, 25))
                row["q3"] = float(np.percentile(vals, 75))
            else:
                row.update({"mean": np.nan, "sd": np.nan, "sem": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan})
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["noise_type", "metric", "nominal_snr_db", "method_key"])


def plot_metric_curves(summary: pd.DataFrame, metric: str, stat: str, fig_dir: Path) -> None:
    ylabel, _ = METRICS[metric]
    noise_types = [n for n in ["emg", "eog", "eog+emg"] if n in set(summary["noise_type"])]

    for noise_type in noise_types:
        sub = summary[(summary["noise_type"] == noise_type) & (summary["metric"] == metric)]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.5, 4.6))
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
                markersize=4.5,
                label=METHOD_LABELS.get(method, method),
            )
            if stat == "mean" and sm["sem"].notna().any() and len(sm) > 1:
                sem = sm["sem"].to_numpy(dtype=float)
                ax.fill_between(x, y - sem, y + sem, color=COLORS.get(method, "#333333"), alpha=0.12, linewidth=0)

        ax.set_title(f"DenoiseNet {NOISE_LABELS.get(noise_type, noise_type)} | {ylabel}")
        ax.set_xlabel("Nominal input SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        out_base = fig_dir / f"denoisenet_{sanitize(noise_type)}_{metric}_{stat}"
        save_figure(fig, out_base)
        plt.close(fig)


def plot_panel_curves(summary: pd.DataFrame, metrics: list[str], stat: str, fig_dir: Path) -> None:
    metrics = [m for m in metrics if m in METRICS]
    noise_types = [n for n in ["emg", "eog", "eog+emg"] if n in set(summary["noise_type"])]
    if not metrics or not noise_types:
        return

    for noise_type in noise_types:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
        axes = axes.ravel()
        for ax, metric in zip(axes, metrics):
            ylabel, _ = METRICS[metric]
            sub = summary[(summary["noise_type"] == noise_type) & (summary["metric"] == metric)]
            for method in METHOD_ORDER:
                sm = sub[sub["method_key"] == method].sort_values("nominal_snr_db")
                if sm.empty:
                    continue
                ax.plot(
                    sm["nominal_snr_db"],
                    sm[stat],
                    "-",
                    marker=MARKERS.get(method, "o"),
                    color=COLORS.get(method, "#333333"),
                    linewidth=1.6,
                    markersize=4,
                    label=METHOD_LABELS.get(method, method),
                )
            ax.set_title(ylabel, fontsize=10)
            ax.set_xlabel("Nominal input SNR (dB)")
            ax.grid(alpha=0.25)
        for ax in axes[len(metrics):]:
            ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False)
        fig.suptitle(f"DenoiseNet {NOISE_LABELS.get(noise_type, noise_type)} artifact-region curves")
        fig.tight_layout(rect=[0, 0.06, 1, 0.94])
        out_base = fig_dir / f"denoisenet_{sanitize(noise_type)}_panel_{stat}"
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


def sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text))


if __name__ == "__main__":
    main()
