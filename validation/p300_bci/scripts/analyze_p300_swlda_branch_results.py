from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "outputs" / "swlda_decoder" / "pilot_summary" / "pilot_swlda_subject_summaries_long.csv"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "swlda_decoder" / "branch_analysis"

BRANCH_ORDER = ["baseline", "eog", "emg_eog", "full"]
BRANCH_LABELS = {
    "baseline": "Baseline",
    "eog": "EOG",
    "emg_eog": "EMG+EOG",
    "full": "Full",
}


def cohen_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return np.nan
    return float(np.mean(diff) / sd)


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def read_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"subject", "branch", "acc_rep_15", "auc_repetition_accuracy"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    df["branch"] = pd.Categorical(df["branch"], categories=BRANCH_ORDER, ordered=True)
    numeric = ["auc_repetition_accuracy", "acc_rep_5", "acc_rep_10", "acc_rep_15"]
    numeric += [f"acc_rep_{rep}" for rep in range(1, 16)]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["subject", "branch"]).reset_index(drop=True)


def branch_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auc_repetition_accuracy", "acc_rep_5", "acc_rep_10", "acc_rep_15"]
    rows = []
    for branch in BRANCH_ORDER:
        sub = df[df["branch"] == branch]
        for metric in metrics:
            vals = sub[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "branch": branch,
                    "metric": metric,
                    "n": int(np.sum(np.isfinite(vals))),
                    "mean": float(np.nanmean(vals)),
                    "sem": sem(vals),
                    "median": float(np.nanmedian(vals)),
                    "q25": float(np.nanpercentile(vals, 25)),
                    "q75": float(np.nanpercentile(vals, 75)),
                }
            )
    return pd.DataFrame(rows)


def primary_pairwise_tests(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auc_repetition_accuracy", "acc_rep_5", "acc_rep_10", "acc_rep_15"]
    wide = df.pivot(index="subject", columns="branch")
    rows = []
    for branch in BRANCH_ORDER[1:]:
        for metric in metrics:
            paired = pd.DataFrame(
                {
                    "baseline": wide[(metric, "baseline")],
                    "branch": wide[(metric, branch)],
                }
            ).dropna()
            diff = paired["branch"].to_numpy(float) - paired["baseline"].to_numpy(float)
            try:
                w_p = float(wilcoxon(diff).pvalue) if diff.size >= 2 and np.any(diff != 0) else np.nan
            except ValueError:
                w_p = np.nan
            try:
                t_p = float(ttest_rel(paired["branch"], paired["baseline"]).pvalue) if diff.size >= 2 else np.nan
            except ValueError:
                t_p = np.nan
            rows.append(
                {
                    "comparison": f"{branch}-baseline",
                    "branch": branch,
                    "metric": metric,
                    "n": int(diff.size),
                    "baseline_mean": float(paired["baseline"].mean()),
                    "branch_mean": float(paired["branch"].mean()),
                    "mean_delta": float(np.mean(diff)),
                    "median_delta": float(np.median(diff)),
                    "delta_sem": sem(diff),
                    "cohen_dz": cohen_dz(diff),
                    "improved_subjects": int(np.sum(diff > 0)),
                    "worse_subjects": int(np.sum(diff < 0)),
                    "unchanged_subjects": int(np.sum(diff == 0)),
                    "wilcoxon_p": w_p,
                    "paired_t_p": t_p,
                }
            )
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm_primary"] = multipletests(out["wilcoxon_p"].fillna(1.0), method="holm")[1]
    return out


def repetitionwise_tests(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot(index="subject", columns="branch")
    rows = []
    for branch in BRANCH_ORDER[1:]:
        for rep in range(1, 16):
            metric = f"acc_rep_{rep}"
            paired = pd.DataFrame(
                {
                    "baseline": wide[(metric, "baseline")],
                    "branch": wide[(metric, branch)],
                }
            ).dropna()
            diff = paired["branch"].to_numpy(float) - paired["baseline"].to_numpy(float)
            try:
                w_p = float(wilcoxon(diff).pvalue) if diff.size >= 2 and np.any(diff != 0) else np.nan
            except ValueError:
                w_p = np.nan
            rows.append(
                {
                    "branch": branch,
                    "repetition": rep,
                    "baseline_mean": float(paired["baseline"].mean()),
                    "branch_mean": float(paired["branch"].mean()),
                    "mean_delta": float(np.mean(diff)),
                    "cohen_dz": cohen_dz(diff),
                    "improved_subjects": int(np.sum(diff > 0)),
                    "worse_subjects": int(np.sum(diff < 0)),
                    "unchanged_subjects": int(np.sum(diff == 0)),
                    "wilcoxon_p": w_p,
                }
            )
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm_all_reps"] = multipletests(out["wilcoxon_p"].fillna(1.0), method="holm")[1]
    out["wilcoxon_p_holm_within_branch"] = np.nan
    for branch in BRANCH_ORDER[1:]:
        mask = out["branch"] == branch
        out.loc[mask, "wilcoxon_p_holm_within_branch"] = multipletests(
            out.loc[mask, "wilcoxon_p"].fillna(1.0),
            method="holm",
        )[1]
    return out


def friedman_by_metric(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auc_repetition_accuracy", "acc_rep_5", "acc_rep_10", "acc_rep_15"]
    wide = df.pivot(index="subject", columns="branch")
    rows = []
    for metric in metrics:
        arrays = [wide[(metric, branch)].to_numpy(float) for branch in BRANCH_ORDER]
        stat, p_value = friedmanchisquare(*arrays)
        rows.append({"metric": metric, "friedman_chi2": float(stat), "friedman_p": float(p_value), "n": len(arrays[0])})
    return pd.DataFrame(rows)


def subject_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot(index="subject", columns="branch")
    rows = []
    for subject in wide.index:
        base15 = float(wide.loc[subject, ("acc_rep_15", "baseline")])
        base_auc = float(wide.loc[subject, ("auc_repetition_accuracy", "baseline")])
        row = {
            "subject": subject,
            "baseline_acc_rep_15": base15,
            "baseline_auc_repetition_accuracy": base_auc,
        }
        best_branch = "baseline"
        best_l15 = base15
        for branch in BRANCH_ORDER[1:]:
            l15 = float(wide.loc[subject, ("acc_rep_15", branch)])
            auc = float(wide.loc[subject, ("auc_repetition_accuracy", branch)])
            row[f"{branch}_acc_rep_15"] = l15
            row[f"{branch}_delta_acc_rep_15"] = l15 - base15
            row[f"{branch}_auc_repetition_accuracy"] = auc
            row[f"{branch}_delta_auc_repetition_accuracy"] = auc - base_auc
            if l15 > best_l15:
                best_l15 = l15
                best_branch = branch
        row["best_branch_by_acc_rep_15"] = best_branch
        row["best_acc_rep_15"] = best_l15
        row["best_delta_acc_rep_15"] = best_l15 - base15
        rows.append(row)
    return pd.DataFrame(rows).sort_values("best_delta_acc_rep_15", ascending=False)


def make_manuscript_table(group: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    l15 = group[group["metric"] == "acc_rep_15"].set_index("branch")
    auc = group[group["metric"] == "auc_repetition_accuracy"].set_index("branch")
    test_l15 = primary[primary["metric"] == "acc_rep_15"].set_index("branch")
    test_auc = primary[primary["metric"] == "auc_repetition_accuracy"].set_index("branch")
    rows = []
    for branch in BRANCH_ORDER:
        row = {
            "branch": BRANCH_LABELS[branch],
            "letter15_mean_sem": f"{l15.loc[branch, 'mean']:.3f} +/- {l15.loc[branch, 'sem']:.3f}",
            "auc_rep_mean_sem": f"{auc.loc[branch, 'mean']:.3f} +/- {auc.loc[branch, 'sem']:.3f}",
        }
        if branch == "baseline":
            row.update({"delta_letter15": "reference", "holm_p_letter15": "", "delta_auc_rep": "reference", "holm_p_auc_rep": ""})
        else:
            row.update(
                {
                    "delta_letter15": f"{test_l15.loc[branch, 'mean_delta']:+.3f}",
                    "holm_p_letter15": f"{test_l15.loc[branch, 'wilcoxon_p_holm_primary']:.4g}",
                    "delta_auc_rep": f"{test_auc.loc[branch, 'mean_delta']:+.3f}",
                    "holm_p_auc_rep": f"{test_auc.loc[branch, 'wilcoxon_p_holm_primary']:.4g}",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_accuracy_curves(df: pd.DataFrame, out_path: Path) -> None:
    reps = np.arange(1, 16)
    colors = {"baseline": "#1f77b4", "eog": "#ff7f0e", "emg_eog": "#2ca02c", "full": "#d62728"}
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for branch in BRANCH_ORDER:
        sub = df[df["branch"] == branch]
        mat = sub[[f"acc_rep_{rep}" for rep in reps]].to_numpy(float)
        mean = np.nanmean(mat, axis=0)
        err = np.array([sem(mat[:, idx]) for idx in range(mat.shape[1])])
        ax.plot(reps, mean, marker="o", markersize=3.0, linewidth=1.35, color=colors[branch], label=BRANCH_LABELS[branch])
        ax.fill_between(reps, mean - err, mean + err, color=colors[branch], alpha=0.14, linewidth=0)
    ax.set_xlabel("Repetition")
    ax.set_ylabel("Letter accuracy")
    ax.set_ylim(0.0, 0.4)
    ax.set_xlim(1, 15)
    ax.set_xticks([1, 5, 10, 15])
    ax.grid(True, color="#dddddd", linewidth=0.55)
    ax.legend(frameon=False, ncols=2, fontsize=7, handlelength=1.4, columnspacing=0.9)
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_subject_delta_heatmap(subjects: pd.DataFrame, out_path: Path) -> None:
    cols = ["eog_delta_acc_rep_15", "emg_eog_delta_acc_rep_15", "full_delta_acc_rep_15"]
    mat = subjects[cols].to_numpy(float)
    labels = ["EOG", "EMG+EOG", "Full"]
    fig_h = max(6.0, 0.18 * len(subjects))
    fig, ax = plt.subplots(figsize=(5.6, fig_h))
    vmax = max(0.25, float(np.nanmax(np.abs(mat))))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels(subjects["subject"], fontsize=7)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_title("Subject-Level Delta Letter@15 vs Baseline")
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("Delta accuracy")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_paired_letter15(df: pd.DataFrame, out_path: Path) -> None:
    wide = df.pivot(index="subject", columns="branch")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7), sharex=True, sharey=True)
    for ax, branch in zip(axes, BRANCH_ORDER[1:]):
        base = wide[("acc_rep_15", "baseline")].to_numpy(float)
        vals = wide[("acc_rep_15", branch)].to_numpy(float)
        jitter = np.linspace(-0.006, 0.006, len(base))
        ax.scatter(base + jitter, vals, s=18, alpha=0.72)
        ax.plot([0, 1], [0, 1], color="#555555", linewidth=1.0, linestyle="--")
        ax.set_title(BRANCH_LABELS[branch])
        ax.set_xlabel("Baseline Letter@15")
        ax.grid(True, color="#dddddd", linewidth=0.7)
    axes[0].set_ylabel("Denoised Letter@15")
    axes[0].set_xlim(-0.03, 1.03)
    axes[0].set_ylim(-0.03, 1.03)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze SWLDA branch decoding results for the P300 BCI validation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = read_results(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    group = branch_group_summary(df)
    primary = primary_pairwise_tests(df)
    repetition = repetitionwise_tests(df)
    friedman = friedman_by_metric(df)
    subjects = subject_delta_table(df)
    manuscript = make_manuscript_table(group, primary)

    group.to_csv(args.out_dir / "branch_group_summary.csv", index=False)
    primary.to_csv(args.out_dir / "branch_primary_paired_tests.csv", index=False)
    repetition.to_csv(args.out_dir / "branch_repetitionwise_tests.csv", index=False)
    friedman.to_csv(args.out_dir / "branch_friedman_tests.csv", index=False)
    subjects.to_csv(args.out_dir / "branch_subject_deltas.csv", index=False)
    manuscript.to_csv(args.out_dir / "branch_manuscript_table.csv", index=False)

    plot_accuracy_curves(df, args.out_dir / "branch_accuracy_curves_mean_sem.png")
    plot_accuracy_curves(df, args.out_dir / "branch_accuracy_curves_mean_sem_singlecol.pdf")
    plot_subject_delta_heatmap(subjects, args.out_dir / "branch_subject_delta_heatmap_letter15.png")
    plot_paired_letter15(df, args.out_dir / "branch_paired_letter15_scatter.png")

    print(f"Subjects: {df['subject'].nunique()}")
    print("\nManuscript-style table:")
    print(manuscript.to_string(index=False))
    print("\nPrimary Letter@15 comparisons:")
    cols = [
        "comparison",
        "mean_delta",
        "cohen_dz",
        "improved_subjects",
        "worse_subjects",
        "unchanged_subjects",
        "wilcoxon_p",
        "wilcoxon_p_holm_primary",
    ]
    print(primary[primary["metric"] == "acc_rep_15"][cols].to_string(index=False))
    print(f"\nSaved outputs to:\n  {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
