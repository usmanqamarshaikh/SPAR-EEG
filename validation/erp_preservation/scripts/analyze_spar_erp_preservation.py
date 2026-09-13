from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import ttest_1samp


BRANCHES = OrderedDict(
    [
        ("Reference", "data_reference"),
        ("EMG", "data_emg"),
        ("EOG", "data_eog"),
        ("EMG+EOG", "data_emg_eog"),
        ("Full", "data_full"),
    ]
)
REGIONS = OrderedDict(
    [
        ("Frontal", ["FPZ", "F3", "FZ", "F4"]),
        ("Fronto-central", ["FC5", "FC1", "FC2", "FC6"]),
        ("Central", ["C3", "CZ", "C4"]),
        ("Centro-parietal", ["CP5", "CP1", "CP2", "CP6"]),
        ("Parietal", ["P7", "P3", "PZ", "P4", "P8"]),
        ("Parieto-occipital", ["PO7", "PO3", "POZ", "PO4", "PO8"]),
        ("Occipital", ["O1", "OZ", "O2"]),
        ("Temporal", ["T7", "T8"]),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore stimulus-locked ERP preservation across SPAR branches.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fdr", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = loadmat(args.bundle, simplify_cells=True)
    data = {name: ensure_3d(np.asarray(raw[key], dtype=float)) for name, key in BRANCHES.items()}
    labels = np.atleast_1d(raw["channel_labels"]).astype(str)
    times = np.atleast_1d(raw["times"]).astype(float)
    theta = np.atleast_1d(raw["channel_theta"]).astype(float)
    radius = np.atleast_1d(raw["channel_radius"]).astype(float)
    diagnostics = pd.read_csv(args.diagnostics)

    audit = audit_inputs(data, labels, times, diagnostics)
    reference = data["Reference"]
    rereferenced = {name: average_rereference(values) for name, values in data.items()}

    waveform_rows = waveform_metrics(data, labels)
    waveform_df = pd.DataFrame(waveform_rows)
    waveform_df.to_csv(args.out_dir / "channel_trial_waveform_metrics.csv", index=False)

    sig_rows, sig_sets, mean_z = significance_analysis(rereferenced, labels, times, args.fdr)
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(args.out_dir / "channel_significance_results.csv", index=False)

    summary_df, erps, maps = summarize_branches(
        data, rereferenced, waveform_df, sig_sets, mean_z, labels, times
    )
    summary_df.to_csv(args.out_dir / "erp_preservation_branch_summary.csv", index=False)

    channel_erp_df = channel_erp_metrics(erps, labels, times)
    channel_erp_df.to_csv(args.out_dir / "channel_erp_preservation_metrics.csv", index=False)

    region_df = regional_metrics(rereferenced, labels, times)
    region_df.to_csv(args.out_dir / "regional_erp_preservation_metrics.csv", index=False)

    regional_power_df = regional_field_power_metrics(rereferenced, labels, times)
    regional_power_df.to_csv(args.out_dir / "regional_field_power_metrics.csv", index=False)

    detectability_df = regional_erp_detectability_metrics(rereferenced, labels, times)
    detectability_df.to_csv(args.out_dir / "regional_erp_detectability_metrics.csv", index=False)

    stage_summary = summarize_diagnostics(diagnostics)
    stage_summary.to_csv(args.out_dir / "spar_stage_activation_summary.csv", index=False)

    configure_plotting()
    plot_regional_erps(rereferenced, labels, times, args.out_dir)
    plot_topographic_maps(maps["0--600 ms"], labels, theta, radius, summary_df, args.out_dir)
    plot_summary(summary_df, region_df, args.out_dir)

    report = build_report(audit, summary_df, stage_summary)
    (args.out_dir / "erp_preservation_exploratory_report.md").write_text(report, encoding="utf-8")

    print(audit)
    print("\nBranch summary:")
    print(summary_df.to_string(index=False))
    print("\nStage activation:")
    print(stage_summary.to_string(index=False))


def ensure_3d(values: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        return values[:, :, None]
    if values.ndim != 3:
        raise ValueError(f"Expected 3-D branch data, found {values.shape}.")
    return values


def audit_inputs(
    data: dict[str, np.ndarray], labels: np.ndarray, times: np.ndarray, diagnostics: pd.DataFrame
) -> str:
    shapes = {name: values.shape for name, values in data.items()}
    if len(set(shapes.values())) != 1:
        raise RuntimeError(f"Branch shape mismatch: {shapes}")
    n_channels, n_samples, n_trials = next(iter(shapes.values()))
    if len(labels) != n_channels or len(times) != n_samples:
        raise RuntimeError("Channel labels or time vector do not match branch data.")
    if not all(np.isfinite(values).all() for values in data.values()):
        raise RuntimeError("At least one branch contains nonfinite samples.")
    status = diagnostics["status"].fillna("").astype(str)
    if not status.eq("OK").all():
        raise RuntimeError(f"Diagnostic CSV contains {(~status.eq('OK')).sum()} failed rows.")
    expected = n_channels * n_trials * 4
    if len(diagnostics) != expected:
        raise RuntimeError(f"Expected {expected} diagnostic rows, found {len(diagnostics)}.")
    node_counts = diagnostics.groupby(["trial", "channel_index"])["node"].nunique()
    if not (node_counts == 4).all():
        raise RuntimeError("At least one channel-trial pair lacks a diagnostic stage.")
    return (
        f"Audit passed: {n_channels} channels, {n_samples} samples, {n_trials} trials, "
        f"{len(diagnostics)} diagnostic stage rows, and no failed/nonfinite outputs."
    )


def average_rereference(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=0, keepdims=True)


def waveform_metrics(data: dict[str, np.ndarray], labels: np.ndarray) -> list[dict[str, float | int | str]]:
    reference = data["Reference"]
    rows: list[dict[str, float | int | str]] = []
    for branch, values in data.items():
        for channel in range(reference.shape[0]):
            for trial in range(reference.shape[2]):
                x = reference[channel, :, trial]
                y = values[channel, :, trial]
                rows.append(
                    {
                        "branch": branch,
                        "channel_index": channel + 1,
                        "channel": labels[channel],
                        "trial": trial + 1,
                        "rrmse_percent": 100 * relative_change(x, y),
                        "pearson_r": correlation(x, y),
                        "rms_ratio": rms(y) / max(rms(x), np.finfo(float).eps),
                    }
                )
    return rows


def significance_analysis(
    data: dict[str, np.ndarray], labels: np.ndarray, times: np.ndarray, fdr: float
) -> tuple[list[dict[str, float | bool | str]], dict[str, set[str]], dict[str, np.ndarray]]:
    pre = (times >= -600) & (times < 0)
    post = (times >= 0) & (times <= 600)
    rows: list[dict[str, float | bool | str]] = []
    sets: dict[str, set[str]] = {}
    maps: dict[str, np.ndarray] = {}
    for branch, values in data.items():
        pre_mean = values[:, pre, :].mean(axis=1)
        post_mean = values[:, post, :].mean(axis=1)
        pre_std = values[:, pre, :].std(axis=1, ddof=1) + np.finfo(float).eps
        z_trials = (post_mean - pre_mean) / pre_std
        result = ttest_1samp(z_trials, 0, axis=1, alternative="two-sided")
        p = np.asarray(result.pvalue, dtype=float)
        q, significant = benjamini_hochberg(p, fdr)
        maps[branch] = z_trials.mean(axis=1)
        sets[branch] = set(labels[significant])
        for channel, label in enumerate(labels):
            rows.append(
                {
                    "branch": branch,
                    "channel": label,
                    "mean_z": maps[branch][channel],
                    "p_raw": p[channel],
                    "q_fdr": q[channel],
                    "significant": bool(significant[channel]),
                }
            )
    return rows, sets, maps


def summarize_branches(
    raw_data: dict[str, np.ndarray],
    rereferenced: dict[str, np.ndarray],
    waveform: pd.DataFrame,
    sig_sets: dict[str, set[str]],
    mean_z: dict[str, np.ndarray],
    labels: np.ndarray,
    times: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    baseline_corrected = {name: baseline_correct(values, times) for name, values in rereferenced.items()}
    erps = {name: values.mean(axis=2) for name, values in baseline_corrected.items()}
    windows = OrderedDict(
        [
            ("0--600 ms", (times >= 0) & (times <= 600)),
            ("250--600 ms", (times >= 250) & (times <= 600)),
        ]
    )
    maps = {
        window: {name: erp[:, mask].mean(axis=1) for name, erp in erps.items()}
        for window, mask in windows.items()
    }
    reference_sig = sig_sets["Reference"]
    rows: list[dict[str, float | int | str]] = []
    for branch in BRANCHES:
        branch_wave = waveform[waveform["branch"] == branch]
        row: dict[str, float | int | str] = {
            "Branch": branch,
            "Epoch-channel RRMSE median (%)": branch_wave["rrmse_percent"].median(),
            "Epoch-channel RRMSE q95 (%)": branch_wave["rrmse_percent"].quantile(0.95),
            "Epoch-channel r median": branch_wave["pearson_r"].median(),
            "Epoch-channel r q05": branch_wave["pearson_r"].quantile(0.05),
            "RMS ratio median": branch_wave["rms_ratio"].median(),
            "Significant channels": len(sig_sets[branch]),
            "Reference-significant retained": len(reference_sig & sig_sets[branch]),
            "New significant channels": len(sig_sets[branch] - reference_sig),
            "Significant-set Jaccard": jaccard(reference_sig, sig_sets[branch]),
            "Mean-z topography r": correlation(mean_z["Reference"], mean_z[branch]),
        }
        for window in windows:
            metrics = spatial_metrics(maps[window]["Reference"], maps[window][branch])
            prefix = window.replace("--", "-")
            row[f"Map r ({prefix})"] = metrics["r"]
            row[f"Map GMD ({prefix})"] = metrics["gmd"]
            row[f"GFP ratio ({prefix})"] = metrics["gfp_ratio"]
            row[f"Map RRMSE ({prefix}, %)"] = 100 * metrics["rrmse"]
        rows.append(row)
    return pd.DataFrame(rows), erps, maps


def channel_erp_metrics(
    erps: dict[str, np.ndarray], labels: np.ndarray, times: np.ndarray
) -> pd.DataFrame:
    post = (times >= 0) & (times <= 600)
    rows = []
    reference = erps["Reference"]
    for branch, erp in erps.items():
        for channel, label in enumerate(labels):
            rows.append(
                {
                    "branch": branch,
                    "channel": label,
                    "post_waveform_r": correlation(reference[channel, post], erp[channel, post]),
                    "post_waveform_rrmse_percent": 100
                    * relative_change(reference[channel, post], erp[channel, post]),
                    "reference_mean_250_600_uV": reference[channel, (times >= 250) & (times <= 600)].mean(),
                    "branch_mean_250_600_uV": erp[channel, (times >= 250) & (times <= 600)].mean(),
                }
            )
    return pd.DataFrame(rows)


def regional_metrics(data: dict[str, np.ndarray], labels: np.ndarray, times: np.ndarray) -> pd.DataFrame:
    post = (times >= 0) & (times <= 600)
    p3 = (times >= 250) & (times <= 600)
    label_upper = np.char.upper(labels.astype(str))
    regional_erps: dict[str, dict[str, np.ndarray]] = {}
    for branch, values in data.items():
        bc = baseline_correct(values, times)
        regional_erps[branch] = {}
        for region, wanted in REGIONS.items():
            indices = np.flatnonzero(np.isin(label_upper, wanted))
            regional_erps[branch][region] = bc[indices].mean(axis=(0, 2))

    rows = []
    for branch in BRANCHES:
        for region in REGIONS:
            reference = regional_erps["Reference"][region]
            observed = regional_erps[branch][region]
            rows.append(
                {
                    "branch": branch,
                    "region": region,
                    "post_waveform_r": correlation(reference[post], observed[post]),
                    "post_waveform_rrmse_percent": 100 * relative_change(reference[post], observed[post]),
                    "reference_mean_250_600_uV": reference[p3].mean(),
                    "branch_mean_250_600_uV": observed[p3].mean(),
                    "mean_250_600_difference_uV": observed[p3].mean() - reference[p3].mean(),
                }
            )
    return pd.DataFrame(rows)


def regional_field_power_metrics(
    data: dict[str, np.ndarray], labels: np.ndarray, times: np.ndarray
) -> pd.DataFrame:
    """Summarize local spatial dispersion and overall ERP magnitude by scalp region."""
    p3 = (times >= 250) & (times <= 600)
    label_upper = np.char.upper(labels.astype(str))
    erps = {
        branch: baseline_correct(values, times).mean(axis=2)
        for branch, values in data.items()
    }

    rows = []
    for region, wanted in REGIONS.items():
        indices = np.flatnonzero(np.isin(label_upper, wanted))
        reference = erps["Reference"][indices][:, p3]
        reference_map = reference.mean(axis=1)
        reference_gfp_time = np.sqrt(
            np.mean((reference - reference.mean(axis=0, keepdims=True)) ** 2, axis=0)
        )
        reference_map_gfp = rms(reference_map - reference_map.mean())
        reference_gfp_rms = rms(reference_gfp_time)
        reference_erp_rms = rms(reference)

        for branch in BRANCHES:
            observed = erps[branch][indices][:, p3]
            observed_map = observed.mean(axis=1)
            observed_gfp_time = np.sqrt(
                np.mean((observed - observed.mean(axis=0, keepdims=True)) ** 2, axis=0)
            )
            observed_map_gfp = rms(observed_map - observed_map.mean())
            observed_gfp_rms = rms(observed_gfp_time)
            observed_erp_rms = rms(observed)
            rows.append(
                {
                    "branch": branch,
                    "region": region,
                    "n_channels": len(indices),
                    "reference_map_gfp_uV": reference_map_gfp,
                    "branch_map_gfp_uV": observed_map_gfp,
                    "map_gfp_ratio": observed_map_gfp
                    / max(reference_map_gfp, np.finfo(float).eps),
                    "reference_gfp_time_rms_uV": reference_gfp_rms,
                    "branch_gfp_time_rms_uV": observed_gfp_rms,
                    "gfp_time_rms_ratio": observed_gfp_rms
                    / max(reference_gfp_rms, np.finfo(float).eps),
                    "reference_erp_rms_uV": reference_erp_rms,
                    "branch_erp_rms_uV": observed_erp_rms,
                    "erp_rms_ratio": observed_erp_rms
                    / max(reference_erp_rms, np.finfo(float).eps),
                }
            )
    return pd.DataFrame(rows)


def regional_erp_detectability_metrics(
    data: dict[str, np.ndarray], labels: np.ndarray, times: np.ndarray
) -> pd.DataFrame:
    """Compute regional mean-amplitude SME and ERP SNR for one recording."""
    p3 = (times >= 250) & (times <= 600)
    label_upper = np.char.upper(labels.astype(str))
    baseline_corrected = {
        branch: baseline_correct(values, times)
        for branch, values in data.items()
    }

    rows = []
    for region, wanted in REGIONS.items():
        indices = np.flatnonzero(np.isin(label_upper, wanted))
        reference_scores = baseline_corrected["Reference"][indices][:, p3, :].mean(axis=(0, 1))
        reference_signal = float(reference_scores.mean())
        reference_sme = float(reference_scores.std(ddof=1) / np.sqrt(reference_scores.size))
        reference_snr = abs(reference_signal) / max(reference_sme, np.finfo(float).eps)

        for branch in BRANCHES:
            scores = baseline_corrected[branch][indices][:, p3, :].mean(axis=(0, 1))
            signal = float(scores.mean())
            sme = float(scores.std(ddof=1) / np.sqrt(scores.size))
            snr = abs(signal) / max(sme, np.finfo(float).eps)
            rows.append(
                {
                    "branch": branch,
                    "region": region,
                    "n_channels": len(indices),
                    "n_trials": scores.size,
                    "mean_amplitude_uV": signal,
                    "amplitude_magnitude_ratio": abs(signal)
                    / max(abs(reference_signal), np.finfo(float).eps),
                    "analytic_sme_uV": sme,
                    "sme_ratio": sme / max(reference_sme, np.finfo(float).eps),
                    "snr_sme": snr,
                    "snr_sme_ratio": snr / max(reference_snr, np.finfo(float).eps),
                }
            )
    return pd.DataFrame(rows)


def summarize_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    labels = OrderedDict(
        [
            ("emg_from_reference", "EMG on reference"),
            ("eog_from_reference", "EOG on reference"),
            ("eog_after_emg", "EOG after EMG"),
            ("slow_after_emg_eog", "Slow after sequence"),
        ]
    )
    rows = []
    for node, label in labels.items():
        sub = diagnostics[diagnostics["node"] == node]
        rows.append(
            {
                "Stage": label,
                "Channel-trials": len(sub),
                "Detector positive (%)": percent(sub["detector_positive"]),
                "Attenuation applied (%)": percent(sub["attenuation_applied"]),
                "No-region bypass (%)": percent(sub["bypass_no_region"]),
                "Baseline bypass (%)": percent(sub["bypass_insufficient_baseline"]),
                "Median attenuated samples (%)": 100
                * pd.to_numeric(sub["attenuated_sample_fraction"], errors="coerce").median(),
                "Median stage RRMSE (%)": pd.to_numeric(
                    sub["stage_input_rrmse_percent"], errors="coerce"
                ).median(),
            }
        )
    return pd.DataFrame(rows)


def baseline_correct(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    pre = (times >= -600) & (times < 0)
    return values - values[:, pre, :].mean(axis=1, keepdims=True)


def spatial_metrics(reference: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=float)
    observed = np.asarray(observed, dtype=float)
    ref_centered = reference - reference.mean()
    obs_centered = observed - observed.mean()
    ref_gfp = np.sqrt(np.mean(ref_centered**2))
    obs_gfp = np.sqrt(np.mean(obs_centered**2))
    if ref_gfp <= np.finfo(float).eps or obs_gfp <= np.finfo(float).eps:
        gmd = np.nan
    else:
        gmd = np.sqrt(np.mean((ref_centered / ref_gfp - obs_centered / obs_gfp) ** 2))
    return {
        "r": correlation(ref_centered, obs_centered),
        "gmd": gmd,
        "gfp_ratio": obs_gfp / max(ref_gfp, np.finfo(float).eps),
        "rrmse": relative_change(ref_centered, obs_centered),
    }


def benjamini_hochberg(p: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    sorted_p = p[order]
    m = len(p)
    thresholds = alpha * np.arange(1, m + 1) / m
    passing = np.flatnonzero(sorted_p <= thresholds)
    significant = np.zeros(m, dtype=bool)
    if passing.size:
        significant = p <= sorted_p[passing[-1]]
    adjusted_sorted = np.minimum.accumulate((sorted_p * m / np.arange(1, m + 1))[::-1])[::-1]
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1)
    return adjusted, significant


def relative_change(reference: np.ndarray, observed: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    if denominator <= np.finfo(float).eps:
        return np.nan
    return float(np.linalg.norm(observed - reference) / denominator)


def correlation(reference: np.ndarray, observed: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float).ravel()
    observed = np.asarray(observed, dtype=float).ravel()
    if reference.size < 2 or np.std(reference) <= np.finfo(float).eps or np.std(observed) <= np.finfo(float).eps:
        return 1.0 if np.allclose(reference, observed) else np.nan
    return float(np.corrcoef(reference, observed)[0, 1])


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def percent(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return 100 * float(clean.astype(bool).mean()) if not clean.empty else np.nan


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


def plot_regional_erps(
    data: dict[str, np.ndarray], labels: np.ndarray, times: np.ndarray, out_dir: Path
) -> None:
    label_upper = np.char.upper(labels.astype(str))
    colors = {"Reference": "black", "EMG": "#4C78A8", "EOG": "#72B7B2", "EMG+EOG": "#F2CF5B", "Full": "#E45756"}
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0), sharex=True)
    for axis, (region, wanted) in zip(axes.ravel(), REGIONS.items()):
        indices = np.flatnonzero(np.isin(label_upper, wanted))
        for branch, values in data.items():
            bc = baseline_correct(values, times)
            erp = bc[indices].mean(axis=(0, 2))
            axis.plot(times, erp, color=colors[branch], linewidth=1.1, label=branch)
        axis.axvline(0, color="#888888", linestyle=":", linewidth=0.7)
        axis.axhline(0, color="#BBBBBB", linewidth=0.5)
        axis.axvspan(250, 600, color="#999999", alpha=0.08)
        axis.set_title(region, fontsize=8)
        axis.grid(color="#E1E1E1", linewidth=0.5)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (ms)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Amplitude (uV)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_dir / "regional_erp_overlays.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "regional_erp_overlays.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_topographic_maps(
    maps: dict[str, np.ndarray],
    labels: np.ndarray,
    theta: np.ndarray,
    radius: np.ndarray,
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    x = radius * np.sin(np.deg2rad(theta))
    y = radius * np.cos(np.deg2rad(theta))
    limit = max(abs(np.concatenate(list(maps.values()))))
    fig, axes = plt.subplots(1, len(BRANCHES), figsize=(7.2, 1.9), constrained_layout=True)
    for axis, branch in zip(axes, BRANCHES):
        scatter = axis.scatter(x, y, c=maps[branch], cmap="RdBu_r", vmin=-limit, vmax=limit, s=45, edgecolor="black", linewidth=0.35)
        circle = plt.Circle((0, 0), max(radius) * 1.08, fill=False, color="black", linewidth=0.8)
        axis.add_patch(circle)
        axis.plot([0, -0.035, 0.035, 0], [max(radius) * 1.08, max(radius) * 1.18, max(radius) * 1.18, max(radius) * 1.08], color="black", linewidth=0.7)
        axis.set_aspect("equal")
        axis.axis("off")
        row = summary[summary["Branch"] == branch].iloc[0]
        axis.set_title(f"{branch}\nr={row['Map r (0-600 ms)']:.3f}, GFP={row['GFP ratio (0-600 ms)']:.3f}", fontsize=7)
    fig.colorbar(scatter, ax=axes, shrink=0.72, label="Mean ERP (uV)")
    fig.savefig(out_dir / "erp_topographic_maps_0_600ms.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "erp_topographic_maps_0_600ms.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, regions: pd.DataFrame, out_dir: Path) -> None:
    branches = list(BRANCHES)
    x = np.arange(len(branches))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), constrained_layout=True)
    axes[0].bar(x, summary["Map r (0-600 ms)"], color="#4C78A8")
    axes[0].set_ylim(0, 1.03)
    axes[0].set_ylabel("Spatial correlation")
    axes[0].set_title("ERP topography")
    axes[1].bar(x, summary["GFP ratio (0-600 ms)"], color="#72B7B2")
    axes[1].axhline(1, color="black", linestyle=":", linewidth=0.8)
    axes[1].set_ylabel("GFP ratio")
    axes[1].set_title("Map amplitude")
    axes[2].bar(x, summary["Reference-significant retained"], color="#E45756")
    axes[2].axhline(summary.loc[0, "Significant channels"], color="black", linestyle=":", linewidth=0.8)
    axes[2].set_ylabel("Retained channels")
    axes[2].set_title("FDR-significant set")
    for axis in axes:
        axis.set_xticks(x, branches, rotation=30, ha="right")
        axis.grid(axis="y", color="#E1E1E1", linewidth=0.5)
        axis.set_axisbelow(True)
    fig.savefig(out_dir / "erp_preservation_summary.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "erp_preservation_summary.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_report(audit: str, summary: pd.DataFrame, stages: pd.DataFrame) -> str:
    lines = [
        "# Exploratory ERP Preservation Analysis",
        "",
        "## Scope",
        "",
        "The preprocessed EEGLAB tutorial data were treated as a low-artifact reference. "
        "SPAR-EEG was applied independently to every two-second channel-trial epoch. "
        "For spatial ERP analyses, every branch was average-rereferenced again after "
        "channel-wise denoising so that all topographies shared the original reference convention.",
        "",
        "This is a one-recording mechanistic preservation analysis, not population-level evidence.",
        "",
        "## Audit",
        "",
        audit,
        "",
        "## Branch summary",
        "",
        dataframe_markdown(summary),
        "",
        "## Internal activation",
        "",
        dataframe_markdown(stages),
        "",
        "## Interpretation rule",
        "",
        "Significant-channel retention is treated as a secondary descriptor because the reference "
        "already contains a broad 24/30-channel response. Spatial correlation, global map "
        "dissimilarity, global-field-power ratio, and regional waveform measures should be "
        "considered jointly before selecting manuscript outcomes.",
        "",
    ]
    return "\n".join(lines)


def dataframe_markdown(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda value: f"{value:.4g}" if pd.notna(value) else "NA")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
