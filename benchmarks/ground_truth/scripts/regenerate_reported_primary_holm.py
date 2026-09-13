from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPORTED_FILES = [
    "01_physiobank.h5",
    "03_denoise-net_emg_-20dB.h5",
    "03_denoise-net_eog_-20dB.h5",
    "03_denoise-net_eog+emg_-20dB.h5",
]
BASELINE_ORDER = ["wqn", "wt_hard", "wt_soft", "emd_cca", "emd_ica"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate Holm-adjusted p values for the 20 primary comparisons reported in the manuscript."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/analysis_primary_saved/paired_wilcoxon_proposed_vs_baselines.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/analysis_primary_saved/revision_reported_family"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input)
    selected = source[
        source["h5_file"].isin(REPORTED_FILES)
        & source["region"].eq("artifact")
        & source["metric"].eq("deltaSNR")
        & source["baseline_method_key"].isin(BASELINE_ORDER)
    ].copy()

    expected = {(file_name, method) for file_name in REPORTED_FILES for method in BASELINE_ORDER}
    observed = set(zip(selected["h5_file"], selected["baseline_method_key"]))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"Primary family mismatch. Missing={missing}, extra={extra}")
    if len(selected) != 20 or selected["p_two_sided"].isna().any():
        raise RuntimeError("Expected exactly 20 finite two-sided primary p values.")

    selected["p_two_sided_holm_reported20"] = holm_adjust(
        selected["p_two_sided"].to_numpy(float)
    )
    selected["significant_reported20"] = selected["p_two_sided_holm_reported20"] < 0.05
    selected["changed_significance_vs_saved_global"] = (
        selected["significant_reported20"]
        != (selected["p_two_sided_holm"] < 0.05)
    )

    file_rank = {name: idx for idx, name in enumerate(REPORTED_FILES)}
    method_rank = {name: idx for idx, name in enumerate(BASELINE_ORDER)}
    selected["_file_rank"] = selected["h5_file"].map(file_rank)
    selected["_method_rank"] = selected["baseline_method_key"].map(method_rank)
    selected = selected.sort_values(["_file_rank", "_method_rank"]).drop(
        columns=["_file_rank", "_method_rank"]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "primary_reported20_holm.csv"
    selected.to_csv(csv_path, index=False)

    summary = selected[
        [
            "dataset_label",
            "baseline_method",
            "n_pairs",
            "median_raw_difference",
            "p_two_sided",
            "p_two_sided_holm_reported20",
            "significant_reported20",
            "p_two_sided_holm",
            "changed_significance_vs_saved_global",
        ]
    ].copy()
    summary_path = args.out_dir / "primary_reported20_holm_summary.csv"
    summary.to_csv(summary_path, index=False)

    n_changed = int(selected["changed_significance_vs_saved_global"].sum())
    print("Regenerated reported primary Holm family")
    print("========================================")
    print(f"Family size: {len(selected)}")
    print(f"Significance changes versus saved global family: {n_changed}")
    print(summary.to_string(index=False))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {summary_path}")


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * p[idx])
        running = max(running, adjusted)
        out[idx] = running
    return out


if __name__ == "__main__":
    main()
