from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_DIR = PROJECT_ROOT / "outputs" / "swlda_decoder"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "swlda_decoder" / "pilot_summary"
BRANCH_ORDER = ["baseline", "eog", "emg_eog", "full"]


def natural_subject_key(name: str) -> tuple[int, str]:
    match = re.search(r"sub-(\d+)$", name)
    if not match:
        return (10**9, name)
    return (int(match.group(1)), name)


def read_subject_summaries(summary_dir: Path, subjects: list[str] | None) -> pd.DataFrame:
    paths = sorted(summary_dir.glob("sub-*_swlda_summary.csv"), key=lambda p: natural_subject_key(p.name))
    if subjects:
        wanted = set(subjects)
        paths = [path for path in paths if path.name.split("_swlda_summary.csv")[0] in wanted]
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "subject" not in df.columns:
            df["subject"] = path.name.split("_swlda_summary.csv")[0]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No subject summary CSVs found in {summary_dir}")
    out = pd.concat(frames, ignore_index=True)
    for col in out.columns:
        if col.startswith("acc_rep_") or col in {
            "auc_repetition_accuracy",
            "train_binary_accuracy",
            "selected_feature_count",
            "total_test_letters",
        }:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["branch"] = pd.Categorical(out["branch"], categories=BRANCH_ORDER, ordered=True)
    return out.sort_values(["subject", "branch"]).reset_index(drop=True)


def summarize_group(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for branch, sub in df.groupby("branch", observed=True):
        for metric in metrics:
            vals = sub[metric].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "branch": branch,
                    "metric": metric,
                    "n": vals.size,
                    "mean": np.nanmean(vals) if vals.size else np.nan,
                    "sem": np.nanstd(vals, ddof=1) / np.sqrt(vals.size) if vals.size > 1 else np.nan,
                    "median": np.nanmedian(vals) if vals.size else np.nan,
                    "q25": np.nanpercentile(vals, 25) if vals.size else np.nan,
                    "q75": np.nanpercentile(vals, 75) if vals.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def paired_tests(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    wide = df.pivot(index="subject", columns="branch")
    for branch in [item for item in BRANCH_ORDER if item != "baseline"]:
        if ("acc_rep_15", branch) not in wide.columns:
            continue
        for metric in metrics:
            if (metric, "baseline") not in wide.columns or (metric, branch) not in wide.columns:
                continue
            paired = pd.DataFrame(
                {
                    "baseline": wide[(metric, "baseline")],
                    "branch": wide[(metric, branch)],
                }
            ).dropna()
            if paired.empty:
                continue
            diff = paired["branch"].to_numpy(dtype=float) - paired["baseline"].to_numpy(dtype=float)
            try:
                wilcoxon_p = float(wilcoxon(diff).pvalue) if diff.size >= 2 and np.any(diff != 0) else np.nan
            except ValueError:
                wilcoxon_p = np.nan
            try:
                ttest_p = float(ttest_rel(paired["branch"], paired["baseline"]).pvalue) if diff.size >= 2 else np.nan
            except ValueError:
                ttest_p = np.nan
            rows.append(
                {
                    "comparison": f"{branch}-baseline",
                    "metric": metric,
                    "n": int(diff.size),
                    "baseline_mean": float(paired["baseline"].mean()),
                    "branch_mean": float(paired["branch"].mean()),
                    "mean_delta": float(np.mean(diff)),
                    "median_delta": float(np.median(diff)),
                    "improved_subjects": int(np.sum(diff > 0)),
                    "worse_subjects": int(np.sum(diff < 0)),
                    "unchanged_subjects": int(np.sum(diff == 0)),
                    "wilcoxon_p": wilcoxon_p,
                    "paired_t_p": ttest_p,
                }
            )
    return pd.DataFrame(rows)


def plot_curves(df: pd.DataFrame, out_path: Path) -> None:
    reps = np.arange(1, 16)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for branch in BRANCH_ORDER:
        sub = df[df["branch"] == branch]
        if sub.empty:
            continue
        mat = sub[[f"acc_rep_{rep}" for rep in reps]].to_numpy(dtype=float)
        mean = np.nanmean(mat, axis=0)
        n = np.sum(np.isfinite(mat), axis=0)
        sem = np.zeros_like(mean)
        valid = n > 1
        sem[valid] = np.nanstd(mat[:, valid], axis=0, ddof=1) / np.sqrt(n[valid])
        ax.plot(reps, mean, marker="o", linewidth=1.8, label=branch)
        ax.fill_between(reps, mean - sem, mean + sem, alpha=0.16)
    ax.set_xlabel("Repetition")
    ax.set_ylabel("Letter accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(reps)
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate subject-level P300 SWLDA pilot summaries.")
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    df = read_subject_summaries(args.summary_dir, args.subjects)
    metrics = ["auc_repetition_accuracy", "acc_rep_5", "acc_rep_10", "acc_rep_15"]
    for rep in range(1, 16):
        metric = f"acc_rep_{rep}"
        if metric not in df.columns:
            raise ValueError(f"Missing expected column: {metric}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_csv = args.out_dir / "pilot_swlda_subject_summaries_long.csv"
    group_csv = args.out_dir / "pilot_swlda_group_summary.csv"
    paired_csv = args.out_dir / "pilot_swlda_paired_tests_vs_baseline.csv"
    curve_png = args.out_dir / "pilot_swlda_accuracy_curves_mean_sem.png"

    df.to_csv(all_csv, index=False)
    group = summarize_group(df, metrics)
    paired = paired_tests(df, metrics)
    group.to_csv(group_csv, index=False)
    paired.to_csv(paired_csv, index=False)
    plot_curves(df, curve_png)

    subjects = sorted(df["subject"].astype(str).unique(), key=natural_subject_key)
    print(f"Subjects included: {len(subjects)}")
    print(", ".join(subjects))
    print("\nMean acc_rep_15 by branch:")
    rep15 = group[group["metric"] == "acc_rep_15"].copy()
    for _, row in rep15.iterrows():
        print(f"  {row['branch']}: mean={row['mean']:.3f}, sem={row['sem']:.3f}, n={int(row['n'])}")
    print("\nPaired deltas vs baseline at acc_rep_15:")
    p15 = paired[paired["metric"] == "acc_rep_15"]
    for _, row in p15.iterrows():
        print(
            f"  {row['comparison']}: mean_delta={row['mean_delta']:+.3f}, "
            f"improved={int(row['improved_subjects'])}/{int(row['n'])}, "
            f"wilcoxon_p={row['wilcoxon_p']:.4g}"
        )
    print(f"\nSaved:\n  {all_csv}\n  {group_csv}\n  {paired_csv}\n  {curve_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
