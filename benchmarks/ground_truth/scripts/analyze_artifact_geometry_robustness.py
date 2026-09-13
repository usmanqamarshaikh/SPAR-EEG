from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


GEOMETRIES = OrderedDict(
    [
        ("center_third", "Centred\n1/3"),
        ("random_third", "Random\n1/3"),
        ("left_edge_third", "Left edge\n1/3"),
        ("right_edge_third", "Right edge\n1/3"),
        ("three_bursts_third", "Three bursts\ntotal 1/3"),
        ("center_sixth", "Centred\n1/6"),
        ("center_two_thirds", "Centred\n2/3"),
        ("full_epoch", "Full\nepoch"),
    ]
)
BRANCHES = OrderedDict([("EMG", "emg"), ("EMG+EOG", "emg_eog"), ("Full", "full")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired EMG artifact-geometry robustness.")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--restored-file", type=Path, required=True)
    parser.add_argument("--diagnostics-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = pd.read_csv(args.diagnostics_file)
    metrics = load_metrics(args.data_file, args.restored_file)
    audit = audit_outputs(metrics, diagnostics)

    metrics.to_csv(args.out_dir / "artifact_geometry_record_metrics.csv", index=False)
    metrics.to_parquet(args.out_dir / "artifact_geometry_record_metrics.parquet", index=False)
    diagnostics.to_csv(args.out_dir / "artifact_geometry_stage_diagnostics_copy.csv", index=False)

    summary = summarize(metrics, diagnostics)
    summary.to_csv(args.out_dir / "artifact_geometry_summary.csv", index=False)
    comparisons = paired_comparisons(metrics)
    comparisons.to_csv(args.out_dir / "artifact_geometry_paired_comparisons.csv", index=False)
    diagnostic_summary = summarize_diagnostics(diagnostics)
    diagnostic_summary.to_csv(args.out_dir / "artifact_geometry_diagnostic_summary.csv", index=False)

    configure_plotting()
    make_figure(metrics, diagnostics, args.out_dir / "artifact_geometry_robustness.pdf")
    make_figure(metrics, diagnostics, args.out_dir / "artifact_geometry_robustness.png")
    report = build_report(audit, summary, diagnostic_summary)
    (args.out_dir / "artifact_geometry_robustness_report.md").write_text(report, encoding="utf-8")

    print(audit)
    print("\nFull-sequence summary:")
    print(summary[summary["branch"] == "Full"].to_string(index=False))
    print("\nEMG-stage diagnostic summary:")
    print(diagnostic_summary[diagnostic_summary["node"] == "emg"].to_string(index=False))
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
            geometry = decode(src.attrs["geometry"])
            record_id = int(src.attrs["record_id"])
            fs = float(src.attrs.get("freq", 256))
            nominal_snr = float(src.attrs["nominal_snr_db"])
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
                            "geometry": geometry,
                            "geometry_order": list(GEOMETRIES).index(geometry) + 1,
                            "branch": branch,
                            "region": region,
                            "fs": fs,
                            "nominal_snr_db": nominal_snr,
                            "artifact_fraction": float(mask.mean()),
                            "region_samples": int(region_mask.sum()),
                            **values,
                        }
                    )
    return pd.DataFrame(rows)


def region_metrics(
    signal: np.ndarray, reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    if not np.any(mask):
        return {name: np.nan for name in metric_names()}
    noisy = signal[:, mask]
    clean = reference[:, mask]
    restored = estimate[:, mask]
    snr_before = calculate_snr(clean, noisy - clean)
    snr_after = calculate_snr(clean, restored - clean)
    return {
        "snr_before_db": snr_before,
        "snr_after_db": snr_after,
        "delta_snr_db": safe_subtract(snr_after, snr_before),
        "rrmse_before": relative_error(clean, noisy),
        "rrmse_after": relative_error(clean, restored),
        "correlation_before": correlation(clean, noisy),
        "correlation_after": correlation(clean, restored),
        "output_change_rrmse": relative_error(noisy, restored),
    }


def metric_names() -> tuple[str, ...]:
    return (
        "snr_before_db",
        "snr_after_db",
        "delta_snr_db",
        "rrmse_before",
        "rrmse_after",
        "correlation_before",
        "correlation_after",
        "output_change_rrmse",
    )


def audit_outputs(metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> str:
    n_records = metrics[["geometry", "record_id"]].drop_duplicates().shape[0]
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
    return (
        f"Audit passed: {n_records} paired geometry-records, {len(metrics)} metric rows, "
        f"{len(diagnostics)} stage rows, no failures, and maximum input-SNR error "
        f"{maximum_snr_error:.3g} dB."
    )


def summarize(metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for geometry in GEOMETRIES:
        for branch in BRANCHES:
            sub = metrics[(metrics["geometry"] == geometry) & (metrics["branch"] == branch)]
            artifact = sub[sub["region"] == "artifact"]
            clean = sub[sub["region"] == "clean"]
            whole = sub[sub["region"] == "whole"]
            emg_diag = diagnostics[
                (diagnostics["geometry"] == geometry) & (diagnostics["node"] == "emg")
            ]
            active_emg = emg_diag[pd.to_numeric(emg_diag["detector_positive"], errors="coerce") > 0.5]
            rows.append(
                {
                    "geometry": geometry,
                    "geometry_label": GEOMETRIES[geometry].replace("\n", " "),
                    "branch": branch,
                    "records": artifact["record_id"].nunique(),
                    "artifact_fraction": artifact["artifact_fraction"].median(),
                    "artifact_delta_snr_median_db": artifact["delta_snr_db"].median(),
                    "artifact_delta_snr_q25_db": artifact["delta_snr_db"].quantile(0.25),
                    "artifact_delta_snr_q75_db": artifact["delta_snr_db"].quantile(0.75),
                    "artifact_rrmse_after_median": artifact["rrmse_after"].median(),
                    "clean_output_change_median": clean["output_change_rrmse"].median(),
                    "clean_output_change_q95": clean["output_change_rrmse"].quantile(0.95),
                    "whole_delta_snr_median_db": whole["delta_snr_db"].median(),
                    "whole_rrmse_after_median": whole["rrmse_after"].median(),
                    "whole_correlation_after_median": whole["correlation_after"].median(),
                    "emg_detector_positive_percent": 100 * numeric_mean(emg_diag["detector_positive"]),
                    "emg_attenuation_percent": 100 * numeric_mean(emg_diag["attenuation_applied"]),
                    "emg_insufficient_baseline_percent": 100
                    * numeric_mean(emg_diag["bypass_insufficient_baseline"]),
                    "emg_total_bypass_percent": 100
                    * numeric_mean(
                        pd.to_numeric(emg_diag["bypass_no_region"], errors="coerce").fillna(0)
                        + pd.to_numeric(
                            emg_diag["bypass_insufficient_baseline"], errors="coerce"
                        ).fillna(0)
                    ),
                    "emg_baseline_fraction_median": pd.to_numeric(
                        emg_diag["baseline_fraction"], errors="coerce"
                    ).median(),
                    "emg_estimated_baseline_fraction_active_median": pd.to_numeric(
                        active_emg["baseline_fraction"], errors="coerce"
                    ).median(),
                }
            )
    return pd.DataFrame(rows)


def summarize_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for geometry in GEOMETRIES:
        for node in ("emg", "eog_after_emg", "slow_after_emg_eog"):
            sub = diagnostics[(diagnostics["geometry"] == geometry) & (diagnostics["node"] == node)]
            active = sub[pd.to_numeric(sub["detector_positive"], errors="coerce") > 0.5]
            rows.append(
                {
                    "geometry": geometry,
                    "node": node,
                    "records": len(sub),
                    "detector_positive_percent": 100 * numeric_mean(sub["detector_positive"]),
                    "attenuation_applied_percent": 100 * numeric_mean(sub["attenuation_applied"]),
                    "bypass_no_region_percent": 100 * numeric_mean(sub["bypass_no_region"]),
                    "bypass_insufficient_baseline_percent": 100
                    * numeric_mean(sub["bypass_insufficient_baseline"]),
                    "total_bypass_percent": 100
                    * numeric_mean(
                        pd.to_numeric(sub["bypass_no_region"], errors="coerce").fillna(0)
                        + pd.to_numeric(
                            sub["bypass_insufficient_baseline"], errors="coerce"
                        ).fillna(0)
                    ),
                    "baseline_fraction_median": pd.to_numeric(
                        sub["baseline_fraction"], errors="coerce"
                    ).median(),
                    "estimated_baseline_fraction_active_median": pd.to_numeric(
                        active["baseline_fraction"], errors="coerce"
                    ).median(),
                    "stage_rrmse_median_percent": pd.to_numeric(
                        sub["stage_input_rrmse_percent"], errors="coerce"
                    ).median(),
                }
            )
    return pd.DataFrame(rows)


def paired_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    requested = [
        ("artifact", "delta_snr_db"),
        ("clean", "output_change_rrmse"),
        ("whole", "rrmse_after"),
        ("whole", "correlation_after"),
    ]
    rows = []
    for branch in BRANCHES:
        for region, metric in requested:
            base = metrics[
                (metrics["branch"] == branch)
                & (metrics["geometry"] == "center_third")
                & (metrics["region"] == region)
            ][["record_id", metric]].rename(columns={metric: "center"})
            family_rows = []
            for geometry in list(GEOMETRIES)[1:]:
                alt = metrics[
                    (metrics["branch"] == branch)
                    & (metrics["geometry"] == geometry)
                    & (metrics["region"] == region)
                ][["record_id", metric]].rename(columns={metric: "alternative"})
                paired = base.merge(alt, on="record_id").dropna()
                difference = paired["alternative"] - paired["center"]
                p_value = paired_wilcoxon(difference.to_numpy())
                family_rows.append(
                    {
                        "branch": branch,
                        "region": region,
                        "metric": metric,
                        "geometry": geometry,
                        "pairs": len(paired),
                        "center_median": paired["center"].median(),
                        "alternative_median": paired["alternative"].median(),
                        "paired_difference_median": difference.median(),
                        "p_raw": p_value,
                    }
                )
            adjusted = holm_adjust(np.asarray([row["p_raw"] for row in family_rows], dtype=float))
            for row, p_holm in zip(family_rows, adjusted):
                row["p_holm"] = p_holm
                rows.append(row)
    return pd.DataFrame(rows)


def make_figure(metrics: pd.DataFrame, diagnostics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    full = metrics[metrics["branch"] == "Full"]
    panels = [
        (axes[0, 0], full[full["region"] == "artifact"], "delta_snr_db", "Artifact-region $\\Delta$SNR (dB)"),
        (axes[0, 1], full[full["region"] == "clean"], "output_change_rrmse", "Clean-region output change"),
        (axes[1, 0], full[full["region"] == "whole"], "rrmse_after", "Whole-epoch RRMSE after"),
    ]
    for axis, frame, metric, ylabel in panels:
        values = [
            pd.to_numeric(frame[frame["geometry"] == geometry][metric], errors="coerce").dropna()
            for geometry in GEOMETRIES
        ]
        axis.boxplot(values, showfliers=False, widths=0.65, patch_artist=True,
                     boxprops={"facecolor": "#78A8A8", "edgecolor": "#333333"},
                     medianprops={"color": "#C83E4D", "linewidth": 1.2})
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    axis = axes[1, 1]
    emg = diagnostics[diagnostics["node"] == "emg"]
    baseline = [
        pd.to_numeric(
            emg[
                (emg["geometry"] == geometry)
                & (pd.to_numeric(emg["detector_positive"], errors="coerce") > 0.5)
            ]["baseline_fraction"],
            errors="coerce",
        ).median()
        for geometry in GEOMETRIES
    ]
    bypass = [
        100
        * numeric_mean(
            pd.to_numeric(
                emg[emg["geometry"] == geometry]["bypass_no_region"], errors="coerce"
            ).fillna(0)
            + pd.to_numeric(
                emg[emg["geometry"] == geometry]["bypass_insufficient_baseline"],
                errors="coerce",
            ).fillna(0)
        )
        for geometry in GEOMETRIES
    ]
    x = np.arange(1, len(GEOMETRIES) + 1)
    axis.bar(x, baseline, color="#4C78A8", label="Median estimated baseline")
    axis.set_ylabel("Estimated baseline fraction\n(detector-positive cases)")
    axis.set_ylim(0, 1)
    twin = axis.twinx()
    twin.plot(x, bypass, color="#C83E4D", marker="o", linewidth=1.2, label="Total EMG bypass")
    twin.set_ylabel("EMG bypass (%)")
    twin.set_ylim(0, 105)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    labels = [label.replace("\n", " ") for label in GEOMETRIES.values()]
    for axis in axes.ravel():
        axis.set_xticks(np.arange(1, len(labels) + 1))
        axis.set_xticklabels(labels, fontsize=6, rotation=28, ha="right", rotation_mode="anchor")
    handles1, labels1 = axis.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axis.legend(handles1 + handles2, labels1 + labels2, fontsize=6, loc="upper right")
    fig.suptitle("SPAR-EEG robustness to EMG artifact geometry at -10 dB", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=250 if path.suffix.lower() == ".png" else None, bbox_inches="tight")
    plt.close(fig)


def build_report(audit: str, summary: pd.DataFrame, diagnostics: pd.DataFrame) -> str:
    full = summary[summary["branch"] == "Full"].copy()
    display = full[
        [
            "geometry_label",
            "artifact_fraction",
            "artifact_delta_snr_median_db",
            "clean_output_change_median",
            "whole_rrmse_after_median",
            "whole_correlation_after_median",
            "emg_total_bypass_percent",
        ]
    ].copy()
    display.columns = [
        "Geometry",
        "Artifact fraction",
        "Artifact delta SNR (median dB)",
        "Clean output change (median)",
        "Whole RRMSE after (median)",
        "Whole correlation after (median)",
        "EMG bypass (%)",
    ]
    return "\n".join(
        [
            "# EMG Artifact-Geometry Robustness",
            "",
            "## Scope",
            "",
            "Paired EEGdenoiseNet clean/EMG records were mixed at -10 dB with the clean EEG and "
            "artifact carrier held common across position variants. Masks were used only for synthesis "
            "and evaluation and were never supplied to SPAR-EEG.",
            "",
            "## Audit",
            "",
            audit,
            "",
            "## Full-sequence outcomes",
            "",
            dataframe_to_markdown(display),
            "",
            "## Interpretation rule",
            "",
            "The full-epoch condition is a boundary test with no clean exterior. Its clean-region "
            "metric is therefore undefined. Position and fragmentation conditions with one-third "
            "support provide the direct geometry comparison; varying-support conditions characterize "
            "the effect of diminishing within-epoch baseline availability.",
            "",
        ]
    )


def row_vector(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).squeeze()
    if array.ndim != 1:
        raise ValueError(f"Expected one-dimensional signal, found {array.shape}")
    return array[None, :]


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    signal_variance = float(np.var(signal))
    noise_variance = float(np.var(noise))
    if signal_variance <= np.finfo(float).eps or noise_variance <= np.finfo(float).eps:
        return np.nan
    return float(10 * np.log10(signal_variance / noise_variance))


def relative_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    if denominator <= np.finfo(float).eps:
        return np.nan
    return float(np.linalg.norm(estimate - reference) / denominator)


def correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    x = np.asarray(reference, dtype=float).ravel()
    y = np.asarray(estimate, dtype=float).ravel()
    if x.size < 2 or np.std(x) <= np.finfo(float).eps or np.std(y) <= np.finfo(float).eps:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def safe_subtract(after: float, before: float) -> float:
    return float(after - before) if np.isfinite(after) and np.isfinite(before) else np.nan


def numeric_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else np.nan


def paired_wilcoxon(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    if np.all(np.abs(values) <= np.finfo(float).eps):
        return 1.0
    return float(wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(p[finite])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("NA")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.4g}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


if __name__ == "__main__":
    main()
