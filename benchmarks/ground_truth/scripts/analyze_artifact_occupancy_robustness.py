from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analyze_artifact_geometry_robustness import (
    BRANCHES,
    calculate_snr,
    configure_plotting,
    correlation,
    dataframe_to_markdown,
    decode,
    holm_adjust,
    numeric_mean,
    relative_error,
    row_vector,
    safe_subtract,
)


OCCUPANCIES = OrderedDict(
    (f"occupancy_{numerator}_9", f"{numerator}/9") for numerator in range(1, 10)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze paired EMG artifact-occupancy robustness."
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--restored-file", type=Path, required=True)
    parser.add_argument("--diagnostics-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = pd.read_csv(args.diagnostics_file)
    metrics = load_metrics(args.data_file, args.restored_file)
    audit = audit_outputs(metrics, diagnostics)

    metrics.to_csv(args.out_dir / "artifact_occupancy_record_metrics.csv", index=False)
    metrics.to_parquet(args.out_dir / "artifact_occupancy_record_metrics.parquet", index=False)
    diagnostics.to_csv(
        args.out_dir / "artifact_occupancy_stage_diagnostics_copy.csv", index=False
    )

    summary = summarize(metrics, diagnostics)
    summary.to_csv(args.out_dir / "artifact_occupancy_summary.csv", index=False)
    adjacent = adjacent_comparisons(metrics)
    adjacent.to_csv(args.out_dir / "artifact_occupancy_adjacent_comparisons.csv", index=False)
    diagnostic_summary = summarize_diagnostics(diagnostics)
    diagnostic_summary.to_csv(
        args.out_dir / "artifact_occupancy_diagnostic_summary.csv", index=False
    )
    transition, bootstrap = estimate_transition(
        metrics, args.bootstrap_repetitions, args.seed
    )
    transition.to_csv(args.out_dir / "artifact_occupancy_transition.csv", index=False)
    bootstrap.to_csv(
        args.out_dir / "artifact_occupancy_transition_bootstrap.csv", index=False
    )

    configure_plotting()
    make_figure(metrics, diagnostics, args.out_dir / "artifact_occupancy_response.pdf")
    make_figure(metrics, diagnostics, args.out_dir / "artifact_occupancy_response.png")
    report = build_report(audit, summary, adjacent, transition)
    (args.out_dir / "artifact_occupancy_report.md").write_text(report, encoding="utf-8")

    print(audit)
    print("\nFull-sequence occupancy summary:")
    print(summary[summary["branch"] == "Full"].to_string(index=False))
    print("\nEmpirical transition diagnostics:")
    print(transition.to_string(index=False))
    print(f"\nOutputs: {args.out_dir}")


def load_metrics(data_file: Path, restored_file: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with h5py.File(data_file, "r") as source, h5py.File(restored_file, "r") as restored:
        source_keys = sorted(source.keys())
        if source_keys != sorted(restored.keys()):
            missing = set(source_keys) ^ set(restored.keys())
            raise RuntimeError(f"Source/restored group mismatch: {sorted(missing)[:10]}")
        for key in source_keys:
            src = source[key]
            out = restored[key]
            status = decode(out.attrs.get("status", ""))
            if status != "OK":
                raise RuntimeError(f"{key}: {status}")
            occupancy = decode(src.attrs["geometry"])
            record_id = int(src.attrs["record_id"])
            fs = float(src.attrs.get("freq", 256))
            signal = row_vector(src["eeg_signal"][()])
            reference = row_vector(src["eeg_reference"][()])
            mask = np.asarray(src["artifacts"][()]).reshape(-1).astype(bool)
            if signal.shape != reference.shape or signal.shape[1] != mask.size:
                raise RuntimeError(f"Shape mismatch for {key}")

            for branch, dataset in BRANCHES.items():
                estimate = row_vector(out[dataset][()])
                for region, region_mask in (
                    ("artifact", mask),
                    ("clean", ~mask),
                    ("whole", np.ones(mask.size, dtype=bool)),
                ):
                    values = region_metrics(signal, reference, estimate, region_mask)
                    rows.append(
                        {
                            "record": key,
                            "record_id": record_id,
                            "occupancy": occupancy,
                            "occupancy_order": int(src.attrs["occupancy_numerator"]),
                            "occupancy_label": decode(src.attrs["geometry_label"]),
                            "target_artifact_fraction": float(
                                src.attrs["target_artifact_fraction"]
                            ),
                            "branch": branch,
                            "region": region,
                            "fs": fs,
                            "nominal_snr_db": float(src.attrs["nominal_snr_db"]),
                            "artifact_fraction": float(mask.mean()),
                            "region_samples": int(region_mask.sum()),
                            **values,
                        }
                    )
    return pd.DataFrame(rows)


def region_metrics(
    signal: np.ndarray, reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    names = (
        "snr_before_db",
        "snr_after_db",
        "delta_snr_db",
        "rrmse_before",
        "rrmse_after",
        "correlation_before",
        "correlation_after",
        "output_change_rrmse",
        "artifact_error_energy_removed_percent",
    )
    if not np.any(mask):
        return {name: np.nan for name in names}
    noisy = signal[:, mask]
    clean = reference[:, mask]
    restored = estimate[:, mask]
    error_before = float(np.sum((noisy - clean) ** 2))
    error_after = float(np.sum((restored - clean) ** 2))
    snr_before = calculate_snr(clean, noisy - clean)
    snr_after = calculate_snr(clean, restored - clean)
    energy_removed = (
        100 * (1 - error_after / error_before)
        if error_before > np.finfo(float).eps
        else np.nan
    )
    return {
        "snr_before_db": snr_before,
        "snr_after_db": snr_after,
        "delta_snr_db": safe_subtract(snr_after, snr_before),
        "rrmse_before": relative_error(clean, noisy),
        "rrmse_after": relative_error(clean, restored),
        "correlation_before": correlation(clean, noisy),
        "correlation_after": correlation(clean, restored),
        "output_change_rrmse": relative_error(noisy, restored),
        "artifact_error_energy_removed_percent": energy_removed,
    }


def audit_outputs(metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> str:
    occupancy_records = metrics[["occupancy", "record_id"]].drop_duplicates()
    n_records = len(occupancy_records)
    expected_metrics = n_records * len(BRANCHES) * 3
    if len(metrics) != expected_metrics:
        raise RuntimeError(f"Expected {expected_metrics} metric rows, found {len(metrics)}")
    if diagnostics["status"].fillna("").ne("OK").any():
        raise RuntimeError("At least one diagnostic row failed.")
    expected_diagnostics = n_records * 3
    if len(diagnostics) != expected_diagnostics:
        raise RuntimeError(
            f"Expected {expected_diagnostics} diagnostic rows, found {len(diagnostics)}"
        )
    artifact = metrics[metrics["region"] == "artifact"]
    maximum_snr_error = float(np.nanmax(np.abs(artifact["snr_before_db"] + 10)))
    if maximum_snr_error > 1e-8:
        raise RuntimeError(f"Input artifact-region SNR audit failed: {maximum_snr_error}")
    per_condition = occupancy_records.groupby("occupancy")["record_id"].nunique()
    if len(per_condition) != len(OCCUPANCIES) or per_condition.nunique() != 1:
        raise RuntimeError("Occupancy conditions do not contain a complete paired record set.")
    return (
        f"Audit passed: {n_records} paired occupancy-records, {len(metrics)} metric rows, "
        f"{len(diagnostics)} stage rows, no failures, and maximum artifact-region "
        f"input-SNR error {maximum_snr_error:.3g} dB."
    )


def summarize(metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for occupancy, label in OCCUPANCIES.items():
        for branch in BRANCHES:
            sub = metrics[
                (metrics["occupancy"] == occupancy) & (metrics["branch"] == branch)
            ]
            artifact = sub[sub["region"] == "artifact"]
            clean = sub[sub["region"] == "clean"]
            whole = sub[sub["region"] == "whole"]
            emg_diag = diagnostics[
                (diagnostics["geometry"] == occupancy) & (diagnostics["node"] == "emg")
            ]
            active = emg_diag[
                pd.to_numeric(emg_diag["detector_positive"], errors="coerce") > 0.5
            ]
            bypass = (
                pd.to_numeric(emg_diag["bypass_no_region"], errors="coerce").fillna(0)
                + pd.to_numeric(
                    emg_diag["bypass_insufficient_baseline"], errors="coerce"
                ).fillna(0)
            )
            rows.append(
                {
                    "occupancy": occupancy,
                    "occupancy_label": label,
                    "occupancy_order": int(occupancy.split("_")[1]),
                    "branch": branch,
                    "records": artifact["record_id"].nunique(),
                    "target_artifact_fraction": artifact[
                        "target_artifact_fraction"
                    ].median(),
                    "actual_artifact_fraction": artifact["artifact_fraction"].median(),
                    "artifact_delta_snr_median_db": artifact["delta_snr_db"].median(),
                    "artifact_delta_snr_q25_db": artifact["delta_snr_db"].quantile(0.25),
                    "artifact_delta_snr_q75_db": artifact["delta_snr_db"].quantile(0.75),
                    "artifact_error_removed_median_percent": artifact[
                        "artifact_error_energy_removed_percent"
                    ].median(),
                    "clean_output_change_median": clean["output_change_rrmse"].median(),
                    "clean_output_change_q95": clean["output_change_rrmse"].quantile(0.95),
                    "whole_snr_before_median_db": whole["snr_before_db"].median(),
                    "whole_delta_snr_median_db": whole["delta_snr_db"].median(),
                    "whole_correlation_before_median": whole[
                        "correlation_before"
                    ].median(),
                    "whole_correlation_after_median": whole["correlation_after"].median(),
                    "emg_detector_positive_percent": 100
                    * numeric_mean(emg_diag["detector_positive"]),
                    "emg_attenuation_percent": 100
                    * numeric_mean(emg_diag["attenuation_applied"]),
                    "emg_total_bypass_percent": 100 * numeric_mean(bypass),
                    "emg_estimated_baseline_active_median": pd.to_numeric(
                        active["baseline_fraction"], errors="coerce"
                    ).median(),
                }
            )
    return pd.DataFrame(rows)


def summarize_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for occupancy in OCCUPANCIES:
        for node in ("emg", "eog_after_emg", "slow_after_emg_eog"):
            sub = diagnostics[
                (diagnostics["geometry"] == occupancy) & (diagnostics["node"] == node)
            ]
            active = sub[
                pd.to_numeric(sub["detector_positive"], errors="coerce") > 0.5
            ]
            bypass = (
                pd.to_numeric(sub["bypass_no_region"], errors="coerce").fillna(0)
                + pd.to_numeric(
                    sub["bypass_insufficient_baseline"], errors="coerce"
                ).fillna(0)
            )
            rows.append(
                {
                    "occupancy": occupancy,
                    "node": node,
                    "records": len(sub),
                    "detector_positive_percent": 100
                    * numeric_mean(sub["detector_positive"]),
                    "attenuation_applied_percent": 100
                    * numeric_mean(sub["attenuation_applied"]),
                    "total_bypass_percent": 100 * numeric_mean(bypass),
                    "estimated_baseline_active_median": pd.to_numeric(
                        active["baseline_fraction"], errors="coerce"
                    ).median(),
                    "stage_rrmse_median_percent": pd.to_numeric(
                        sub["stage_input_rrmse_percent"], errors="coerce"
                    ).median(),
                }
            )
    return pd.DataFrame(rows)


def adjacent_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics[
        (metrics["branch"] == "Full") & (metrics["region"] == "artifact")
    ][["record_id", "occupancy_order", "delta_snr_db"]]
    rows = []
    for lower in range(1, 9):
        first = frame[frame["occupancy_order"] == lower][
            ["record_id", "delta_snr_db"]
        ].rename(columns={"delta_snr_db": "lower"})
        second = frame[frame["occupancy_order"] == lower + 1][
            ["record_id", "delta_snr_db"]
        ].rename(columns={"delta_snr_db": "upper"})
        paired = first.merge(second, on="record_id").dropna()
        difference = paired["upper"] - paired["lower"]
        rows.append(
            {
                "lower_occupancy": f"{lower}/9",
                "upper_occupancy": f"{lower + 1}/9",
                "pairs": len(paired),
                "lower_median_db": paired["lower"].median(),
                "upper_median_db": paired["upper"].median(),
                "paired_difference_median_db": difference.median(),
                "p_raw": paired_wilcoxon(difference.to_numpy()),
            }
        )
    adjusted = holm_adjust(np.asarray([row["p_raw"] for row in rows], dtype=float))
    for row, p_holm in zip(rows, adjusted):
        row["p_holm"] = p_holm
    return pd.DataFrame(rows)


def estimate_transition(
    metrics: pd.DataFrame, repetitions: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = metrics[
        (metrics["branch"] == "Full") & (metrics["region"] == "artifact")
    ][["record_id", "occupancy_order", "delta_snr_db"]]
    pivot = frame.pivot(index="record_id", columns="occupancy_order", values="delta_snr_db")
    pivot = pivot.reindex(columns=range(1, 10)).dropna()
    x = np.arange(1, 10, dtype=float) / 9
    observed = pivot.median(axis=0).to_numpy(dtype=float)
    split, sse = best_two_regime_split(x, observed)
    adjacent_drop = np.diff(observed)
    largest_index = int(np.argmin(adjacent_drop))

    rng = np.random.default_rng(seed)
    boot_rows = []
    values = pivot.to_numpy(dtype=float)
    for repetition in range(repetitions):
        sample = values[rng.integers(0, len(values), len(values))]
        medians = np.median(sample, axis=0)
        boot_split, _ = best_two_regime_split(x, medians)
        boot_drop = np.diff(medians)
        boot_largest = int(np.argmin(boot_drop))
        boot_rows.append(
            {
                "repetition": repetition + 1,
                "two_regime_lower_numerator": boot_split,
                "two_regime_upper_numerator": boot_split + 1,
                "largest_drop_lower_numerator": boot_largest + 1,
                "largest_drop_upper_numerator": boot_largest + 2,
            }
        )
    bootstrap = pd.DataFrame(boot_rows)
    split_pairs = (
        bootstrap.groupby(
            ["two_regime_lower_numerator", "two_regime_upper_numerator"]
        ).size()
        / len(bootstrap)
    )
    drop_pairs = (
        bootstrap.groupby(
            ["largest_drop_lower_numerator", "largest_drop_upper_numerator"]
        ).size()
        / len(bootstrap)
    )
    transition = pd.DataFrame(
        [
            {
                "records": len(pivot),
                "bootstrap_repetitions": repetitions,
                "two_regime_transition_interval": f"{split}/9--{split + 1}/9",
                "two_regime_fit_sse": sse,
                "two_regime_bootstrap_support_percent": 100
                * float(split_pairs.get((split, split + 1), 0)),
                "largest_median_drop_interval": f"{largest_index + 1}/9--{largest_index + 2}/9",
                "largest_median_drop_db": float(adjacent_drop[largest_index]),
                "largest_drop_interval_bootstrap_support_percent": 100
                * float(drop_pairs.get((largest_index + 1, largest_index + 2), 0)),
            }
        ]
    )
    return transition, bootstrap


def best_two_regime_split(x: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    # Allow an abrupt level change because thresholded attenuation can switch
    # operating regimes rather than following a continuous broken line.
    best_split = 2
    best_sse = np.inf
    for split in range(2, len(x) - 1):
        left_design = np.column_stack([np.ones(split), x[:split]])
        right_design = np.column_stack([np.ones(len(x) - split), x[split:]])
        left_coefficients, _, _, _ = np.linalg.lstsq(
            left_design, y[:split], rcond=None
        )
        right_coefficients, _, _, _ = np.linalg.lstsq(
            right_design, y[split:], rcond=None
        )
        residual = np.concatenate(
            [
                y[:split] - left_design @ left_coefficients,
                y[split:] - right_design @ right_coefficients,
            ]
        )
        sse = float(np.sum(residual**2))
        if sse < best_sse:
            best_split = split
            best_sse = sse
    return best_split, best_sse


def paired_wilcoxon(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    if np.all(np.abs(values) <= np.finfo(float).eps):
        return 1.0
    return float(wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue)


def make_figure(metrics: pd.DataFrame, diagnostics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    full = metrics[metrics["branch"] == "Full"]
    x = np.arange(1, 10)
    labels = list(OCCUPANCIES.values())

    plot_quantile_curve(
        axes[0, 0],
        full[full["region"] == "artifact"],
        "delta_snr_db",
        "Artifact-region $\\Delta$SNR (dB)",
    )
    axes[0, 0].axhline(0, color="#777777", linewidth=0.7)
    plot_quantile_curve(
        axes[0, 1],
        full[full["region"] == "artifact"],
        "artifact_error_energy_removed_percent",
        "Artifact error energy removed (%)",
    )
    axes[0, 1].axhline(0, color="#777777", linewidth=0.7)

    emg = diagnostics[diagnostics["node"] == "emg"]
    baseline = []
    bypass = []
    for occupancy in OCCUPANCIES:
        subset = emg[emg["geometry"] == occupancy]
        active = subset[
            pd.to_numeric(subset["detector_positive"], errors="coerce") > 0.5
        ]
        baseline.append(
            pd.to_numeric(active["baseline_fraction"], errors="coerce").median()
        )
        bypass.append(
            100
            * numeric_mean(
                pd.to_numeric(subset["bypass_no_region"], errors="coerce").fillna(0)
                + pd.to_numeric(
                    subset["bypass_insufficient_baseline"], errors="coerce"
                ).fillna(0)
            )
        )
    axes[1, 0].plot(x, baseline, color="#4C78A8", marker="o", label="Estimated baseline")
    axes[1, 0].set_ylabel("Estimated baseline fraction\n(detector-positive cases)")
    axes[1, 0].set_ylim(0, 1)
    twin = axes[1, 0].twinx()
    twin.plot(x, bypass, color="#C83E4D", marker="s", label="EMG bypass")
    twin.set_ylabel("EMG bypass (%)")
    twin.set_ylim(0, 105)
    handles1, labels1 = axes[1, 0].get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axes[1, 0].legend(handles1 + handles2, labels1 + labels2, fontsize=6, loc="best")

    whole = full[full["region"] == "whole"]
    before = grouped_median(whole, "correlation_before")
    after = grouped_median(whole, "correlation_after")
    axes[1, 1].plot(x, before, color="#777777", marker="o", label="Before")
    axes[1, 1].plot(x, after, color="#2A9D8F", marker="o", label="After")
    axes[1, 1].set_ylabel("Whole-epoch correlation")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(fontsize=7)

    for axis in axes.ravel():
        axis.axvline(4.5, color="#8A8A8A", linestyle="--", linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_xlabel("Artifact occupancy")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    fig.suptitle("SPAR-EEG response to centered EMG occupancy at -10 dB local SNR", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=250 if path.suffix.lower() == ".png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_quantile_curve(
    axis: plt.Axes, frame: pd.DataFrame, metric: str, ylabel: str
) -> None:
    grouped = frame.groupby("occupancy_order")[metric]
    median = grouped.median().reindex(range(1, 10)).to_numpy(dtype=float)
    low = grouped.quantile(0.25).reindex(range(1, 10)).to_numpy(dtype=float)
    high = grouped.quantile(0.75).reindex(range(1, 10)).to_numpy(dtype=float)
    x = np.arange(1, 10)
    axis.fill_between(x, low, high, color="#78A8A8", alpha=0.3, linewidth=0)
    axis.plot(x, median, color="#2A6F73", marker="o", linewidth=1.5)
    axis.set_ylabel(ylabel)


def grouped_median(frame: pd.DataFrame, metric: str) -> np.ndarray:
    return (
        frame.groupby("occupancy_order")[metric]
        .median()
        .reindex(range(1, 10))
        .to_numpy(dtype=float)
    )


def build_report(
    audit: str,
    summary: pd.DataFrame,
    adjacent: pd.DataFrame,
    transition: pd.DataFrame,
) -> str:
    full = summary[summary["branch"] == "Full"].copy()
    display = full[
        [
            "occupancy_label",
            "actual_artifact_fraction",
            "whole_snr_before_median_db",
            "artifact_delta_snr_median_db",
            "artifact_delta_snr_q25_db",
            "artifact_delta_snr_q75_db",
            "artifact_error_removed_median_percent",
            "emg_total_bypass_percent",
            "whole_correlation_after_median",
        ]
    ].copy()
    display.columns = [
        "Occupancy",
        "Actual fraction",
        "Whole input SNR (median dB)",
        "Artifact delta SNR (median dB)",
        "Q25",
        "Q75",
        "Artifact error removed (median %)",
        "EMG bypass (%)",
        "Whole correlation after",
    ]
    return "\n".join(
        [
            "# EMG Artifact-Occupancy Robustness",
            "",
            "## Scope",
            "",
            "The same paired EEGdenoiseNet clean/EMG records were evaluated with centered, "
            "nested artifact supports from 1/9 to 9/9 of each epoch. Artifact-region input "
            "SNR was fixed at -10 dB independently for every occupancy. Masks were used only "
            "for synthesis and scoring and were not supplied to SPAR-EEG.",
            "",
            "## Audit",
            "",
            audit,
            "",
            "## Full-sequence outcomes",
            "",
            dataframe_to_markdown(display),
            "",
            "## Adjacent paired comparisons",
            "",
            dataframe_to_markdown(adjacent),
            "",
            "## Empirical transition diagnostics",
            "",
            dataframe_to_markdown(transition),
            "",
            "The two-regime split and largest adjacent drop are descriptive diagnostics. They "
            "should be called an empirical transition region unless bootstrap support is "
            "sufficiently concentrated to justify a more precise knee estimate.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
