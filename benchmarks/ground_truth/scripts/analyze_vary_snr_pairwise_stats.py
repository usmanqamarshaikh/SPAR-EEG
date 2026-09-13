from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METHOD_ORDER = ["proposed_auto", "wqn", "wt_hard", "wt_soft", "emd_cca", "emd_ica"]
METHOD_LABELS = {
    "proposed_auto": "Proposed",
    "wqn": "WQN",
    "wt_hard": "WT-hard",
    "wt_soft": "WT-soft",
    "emd_cca": "EMD-CCA",
    "emd_ica": "EMD-ICA",
}
NOISE_LABELS = {"emg": "EMG", "eog": "EOG", "eog+emg": "EOG+EMG"}
METRICS = {
    "deltaSNR": 1,
    "SNR_after": 1,
    "RRMSE_after": -1,
    "R_after": 1,
    "PSDerr_after": -1,
}
EXCLUDED_NOMINAL_SNRS = {-0.5}
SEVERITY_BINS = [
    ("severe", -20.0, -15.0),
    ("moderate", -14.0, -5.0),
    ("mild", -4.0, 5.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired proposed-vs-baseline tests for varying-SNR benchmark."
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("results/metrics_vary_snr_saved"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/analysis_vary_snr_saved/stats"),
    )
    parser.add_argument("--region", default="artifact", choices=["artifact", "clean", "whole"])
    parser.add_argument("--metrics", nargs="+", default=["deltaSNR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [m for m in args.metrics if m in METRICS]
    if not metrics:
        raise SystemExit(f"No supported metrics requested. Supported: {sorted(METRICS)}")

    df = load_metric_rows(args.metrics_root, args.region, metrics)
    if df.empty:
        raise SystemExit("No metric rows loaded.")

    per_snr = paired_tests_per_snr(df, metrics)
    per_snr = add_adjusted_p_values(per_snr)
    per_snr.to_csv(args.out_dir / f"{args.region}_paired_tests_by_snr.csv", index=False)
    per_snr.to_parquet(args.out_dir / f"{args.region}_paired_tests_by_snr.parquet", index=False)

    bins = paired_tests_by_severity_bin(df, metrics)
    bins = add_adjusted_p_values(bins)
    bins.to_csv(args.out_dir / f"{args.region}_paired_tests_by_severity_bin.csv", index=False)
    bins.to_parquet(args.out_dir / f"{args.region}_paired_tests_by_severity_bin.parquet", index=False)

    compact = compact_delta_snr_counts(per_snr)
    compact.to_csv(args.out_dir / f"{args.region}_deltaSNR_significance_counts.csv", index=False)

    print("Varying-SNR paired statistics completed.")
    print(f"  rows loaded      : {len(df)}")
    print(f"  per-SNR tests    : {len(per_snr)}")
    print(f"  severity tests   : {len(bins)}")
    print(f"  outputs          : {args.out_dir}")


def load_metric_rows(metrics_root: Path, region: str, metrics: list[str]) -> pd.DataFrame:
    cols = [
        "h5_file",
        "record",
        "region",
        "method_key",
        "benchmark_family",
        "noise_type",
        "nominal_snr_db",
        *metrics,
    ]
    frames = []
    for path in sorted(metrics_root.glob("*/*.parquet")):
        df = pd.read_parquet(path, columns=cols)
        df = df[
            (df["benchmark_family"].astype(str).str.lower() == "denoise-net")
            & (df["region"].astype(str) == region)
        ].copy()
        df = exclude_nominal_snrs(df)
        if df.empty:
            continue
        df["method_key"] = df["method_key"].astype(str)
        df["noise_type"] = df["noise_type"].astype(str).str.lower()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out


def exclude_nominal_snrs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "nominal_snr_db" not in df:
        return df
    snr = pd.to_numeric(df["nominal_snr_db"], errors="coerce")
    keep = np.ones(len(df), dtype=bool)
    for excluded in EXCLUDED_NOMINAL_SNRS:
        keep &= ~np.isclose(snr.to_numpy(dtype=float), excluded, atol=1e-9, rtol=0)
    return df.loc[keep].copy()


def paired_tests_per_snr(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for (noise_type, snr), group in df.groupby(["noise_type", "nominal_snr_db"]):
        for metric in metrics:
            wide = group.pivot_table(
                index=["h5_file", "record"],
                columns="method_key",
                values=metric,
                aggfunc="mean",
            )
            rows.extend(test_wide(wide, metric, noise_type, snr, severity_bin=None))
    return pd.DataFrame(rows)


def paired_tests_by_severity_bin(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    parts = []
    for label, lo, hi in SEVERITY_BINS:
        sub = df[(df["nominal_snr_db"] >= lo) & (df["nominal_snr_db"] <= hi)].copy()
        sub["severity_bin"] = label
        parts.append(sub)
    binned = pd.concat(parts, ignore_index=True)

    rows = []
    for (noise_type, severity_bin), group in binned.groupby(["noise_type", "severity_bin"]):
        for metric in metrics:
            wide = group.pivot_table(
                index=["h5_file", "record"],
                columns="method_key",
                values=metric,
                aggfunc="mean",
            )
            rows.extend(test_wide(wide, metric, noise_type, snr=None, severity_bin=severity_bin))
    return pd.DataFrame(rows)


def test_wide(
    wide: pd.DataFrame,
    metric: str,
    noise_type: str,
    snr: float | None,
    severity_bin: str | None,
) -> list[dict[str, object]]:
    if "proposed_auto" not in wide:
        return []

    rows = []
    direction = METRICS[metric]
    for baseline in [m for m in METHOD_ORDER if m != "proposed_auto"]:
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
            stat_two = p_two = p_greater = np.nan
        elif np.allclose(oriented_diff, 0):
            stat_two = 0.0
            p_two = 1.0
            p_greater = 1.0
        else:
            try:
                stat_two, p_two = wilcoxon(raw_diff, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                stat_two, p_two = np.nan, np.nan
            try:
                _, p_greater = wilcoxon(oriented_diff, alternative="greater", zero_method="wilcox")
            except ValueError:
                p_greater = np.nan

        rows.append(
            {
                "noise_type": noise_type,
                "noise_label": NOISE_LABELS.get(noise_type, noise_type),
                "nominal_snr_db": snr,
                "severity_bin": severity_bin,
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
                "wilcoxon_statistic": stat_two,
                "p_two_sided": p_two,
                "p_one_sided_proposed_better": p_greater,
            }
        )
    return rows


def add_adjusted_p_values(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["p_one_sided_holm_global"] = holm_adjust(out["p_one_sided_proposed_better"])
    out["p_two_sided_holm_global"] = holm_adjust(out["p_two_sided"])

    out["p_one_sided_holm_by_noise_metric"] = np.nan
    out["p_two_sided_holm_by_noise_metric"] = np.nan
    for _, idx in out.groupby(["noise_type", "metric"]).groups.items():
        idx = list(idx)
        out.loc[idx, "p_one_sided_holm_by_noise_metric"] = holm_adjust(
            out.loc[idx, "p_one_sided_proposed_better"]
        )
        out.loc[idx, "p_two_sided_holm_by_noise_metric"] = holm_adjust(
            out.loc[idx, "p_two_sided"]
        )
    return out


def compact_delta_snr_counts(per_snr: pd.DataFrame) -> pd.DataFrame:
    sub = per_snr[per_snr["metric"] == "deltaSNR"].copy()
    if sub.empty:
        return sub
    sub["proposed_higher_median"] = sub["median_raw_difference"] > 0
    sub["significant_by_noise"] = sub["p_one_sided_holm_by_noise_metric"] < 0.05
    sub["significant_global"] = sub["p_one_sided_holm_global"] < 0.05
    rows = []
    for keys, group in sub.groupby(["noise_type", "noise_label", "baseline_method_key", "baseline_method"]):
        rows.append(
            {
                "noise_type": keys[0],
                "noise_label": keys[1],
                "baseline_method_key": keys[2],
                "baseline_method": keys[3],
                "n_snr_levels": int(group["nominal_snr_db"].nunique()),
                "proposed_higher_median_count": int(group["proposed_higher_median"].sum()),
                "proposed_significant_by_noise_count": int(
                    (group["proposed_higher_median"] & group["significant_by_noise"]).sum()
                ),
                "proposed_significant_global_count": int(
                    (group["proposed_higher_median"] & group["significant_global"]).sum()
                ),
                "baseline_higher_median_count": int((~group["proposed_higher_median"]).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["noise_type", "baseline_method_key"])


def holm_adjust(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
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
        adjusted = (m - i) * pv
        running = max(running, adjusted)
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
