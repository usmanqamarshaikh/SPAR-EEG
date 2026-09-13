from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.interpolate import RBFInterpolator
from scipy.io import loadmat
from scipy.stats import pearsonr


BRANCHES = OrderedDict(
    [
        ("Reference", "data_reference"),
        ("EMG", "data_emg"),
        ("EOG", "data_eog"),
        ("EMG+EOG", "data_emg_eog"),
        ("Full", "data_full"),
    ]
)
DISPLAY_BRANCHES = ["Reference", "EMG", "EOG", "Full"]
BASELINE_MS = (-600.0, 0.0)
ERP_WINDOW_MS = (250.0, 600.0)
DISPLAY_WINDOW_MS = (-200.0, 800.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare auditable ERP-preservation revision materials."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-snr-threshold", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = loadmat(args.bundle, simplify_cells=True)
    data = {
        branch: ensure_3d(np.asarray(raw[key], dtype=float))
        for branch, key in BRANCHES.items()
    }
    labels = np.atleast_1d(raw["channel_labels"]).astype(str)
    times = np.atleast_1d(raw["times"]).astype(float)
    theta = np.atleast_1d(raw["channel_theta"]).astype(float)
    radius = np.atleast_1d(raw["channel_radius"]).astype(float)
    audit_inputs(data, labels, times, theta, radius)

    corrected = {
        branch: baseline_correct(average_rereference(values), times)
        for branch, values in data.items()
    }
    metrics = channel_metrics(corrected, labels, times)
    summary = branch_summary(metrics, args.reference_snr_threshold)
    percentages = percentage_table(metrics, args.reference_snr_threshold)

    metrics.to_csv(args.out_dir / "erp_preservation_channel_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "erp_preservation_branch_summary.csv", index=False)
    percentages.to_csv(
        args.out_dir / "erp_preservation_channel_detectability_percent.csv",
        index=False,
    )

    configure_plotting()
    make_summary_figure(
        corrected,
        metrics,
        times,
        theta,
        radius,
        args.out_dir / "supp_erp_preservation_snr",
    )
    print(
        f"Prepared ERP-preservation materials from {data['Reference'].shape[2]} trials "
        f"and {data['Reference'].shape[0]} channels."
    )
    print(summary.to_string(index=False))


def ensure_3d(values: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        return values[:, :, None]
    if values.ndim != 3:
        raise ValueError(f"Expected channels x samples x trials, found {values.shape}.")
    return values


def audit_inputs(
    data: dict[str, np.ndarray],
    labels: np.ndarray,
    times: np.ndarray,
    theta: np.ndarray,
    radius: np.ndarray,
) -> None:
    shapes = {branch: values.shape for branch, values in data.items()}
    if len(set(shapes.values())) != 1:
        raise RuntimeError(f"Branch shape mismatch: {shapes}")
    n_channels, n_samples, _ = next(iter(shapes.values()))
    if len(labels) != n_channels or len(theta) != n_channels or len(radius) != n_channels:
        raise RuntimeError("Channel labels or coordinates do not match branch data.")
    if len(times) != n_samples:
        raise RuntimeError("Time vector does not match branch data.")
    if not all(np.isfinite(values).all() for values in data.values()):
        raise RuntimeError("At least one branch contains nonfinite samples.")


def average_rereference(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=0, keepdims=True)


def baseline_correct(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    baseline = (times >= BASELINE_MS[0]) & (times < BASELINE_MS[1])
    if not baseline.any():
        raise RuntimeError("Prestimulus baseline is unavailable.")
    return values - values[:, baseline, :].mean(axis=1, keepdims=True)


def channel_metrics(
    data: dict[str, np.ndarray], labels: np.ndarray, times: np.ndarray
) -> pd.DataFrame:
    window = (times >= ERP_WINDOW_MS[0]) & (times <= ERP_WINDOW_MS[1])
    rows = []
    for branch, values in data.items():
        for channel, label in enumerate(labels):
            scores = values[channel, window, :].mean(axis=0)
            amplitude = float(scores.mean())
            sme = float(scores.std(ddof=1) / np.sqrt(scores.size))
            rows.append(
                {
                    "branch": branch,
                    "channel_index": channel + 1,
                    "channel": str(label),
                    "n_trials": scores.size,
                    "mean_amplitude_uV": amplitude,
                    "analytic_sme_uV": sme,
                    "signed_snr_sme": amplitude / max(sme, np.finfo(float).eps),
                }
            )
    frame = pd.DataFrame(rows)
    reference = (
        frame[frame["branch"] == "Reference"]
        .set_index("channel")
        [["mean_amplitude_uV", "analytic_sme_uV", "signed_snr_sme"]]
        .add_prefix("reference_")
    )
    frame = frame.join(reference, on="channel")
    epsilon = np.finfo(float).eps
    frame["amplitude_magnitude_retention"] = (
        frame["mean_amplitude_uV"].abs()
        / frame["reference_mean_amplitude_uV"].abs().clip(lower=epsilon)
    )
    frame["analytic_sme_retention"] = (
        frame["analytic_sme_uV"]
        / frame["reference_analytic_sme_uV"].clip(lower=epsilon)
    )
    frame["snr_magnitude_retention"] = (
        frame["signed_snr_sme"].abs()
        / frame["reference_signed_snr_sme"].abs().clip(lower=epsilon)
    )
    return frame


def branch_summary(metrics: pd.DataFrame, threshold: float) -> pd.DataFrame:
    reference = metrics[metrics["branch"] == "Reference"].sort_values("channel_index")
    reference_snr = reference["signed_snr_sme"].to_numpy()
    eligible = np.abs(reference_snr) >= threshold
    rows = []
    for branch in list(BRANCHES)[1:]:
        subset = metrics[metrics["branch"] == branch].sort_values("channel_index")
        observed_snr = subset["signed_snr_sme"].to_numpy()
        snr_retention = subset.loc[eligible, "snr_magnitude_retention"].to_numpy()
        rows.append(
            {
                "branch": branch,
                "n_channels_all": len(subset),
                "n_channels_reference_abs_snr_ge_5": int(eligible.sum()),
                "reference_abs_snr_threshold": threshold,
                "median_amplitude_retention_percent": 100.0
                * subset.loc[eligible, "amplitude_magnitude_retention"].median(),
                "median_analytic_sme_retention_percent": 100.0
                * subset.loc[eligible, "analytic_sme_retention"].median(),
                "median_snr_sme_retention_percent": 100.0 * np.median(snr_retention),
                "minimum_snr_sme_retention_percent": 100.0 * np.min(snr_retention),
                "maximum_snr_sme_retention_percent": 100.0 * np.max(snr_retention),
                "channels_snr_retention_ge_90_percent": int((snr_retention >= 0.90).sum()),
                "all_channel_signed_snr_pattern_r": pearsonr(
                    reference_snr, observed_snr
                ).statistic,
            }
        )
    return pd.DataFrame(rows)


def percentage_table(metrics: pd.DataFrame, threshold: float) -> pd.DataFrame:
    reference = (
        metrics[metrics["branch"] == "Reference"]
        .sort_values("channel_index")
        [["channel_index", "channel", "signed_snr_sme"]]
        .rename(columns={"signed_snr_sme": "reference_signed_snr_sme"})
        .copy()
    )
    reference["eligible_reference_abs_snr_ge_5"] = (
        reference["reference_signed_snr_sme"].abs() >= threshold
    )
    table = reference
    for branch in list(BRANCHES)[1:]:
        subset = metrics[metrics["branch"] == branch].sort_values("channel_index")
        column = branch.lower().replace("+", "_") + "_detectability_percent"
        values = pd.DataFrame(
            {
                "channel": subset["channel"].to_numpy(),
                column: 100.0 * subset["snr_magnitude_retention"].to_numpy(),
            }
        )
        table = table.merge(values, on="channel", how="left", validate="one_to_one")
    return table.sort_values("channel_index").reset_index(drop=True)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "savefig.facecolor": "white",
        }
    )


def make_summary_figure(
    data: dict[str, np.ndarray],
    metrics: pd.DataFrame,
    times: np.ndarray,
    theta: np.ndarray,
    radius: np.ndarray,
    output_stem: Path,
) -> None:
    erps = {branch: values.mean(axis=2) for branch, values in data.items()}
    display = (times >= DISPLAY_WINDOW_MS[0]) & (times <= DISPLAY_WINDOW_MS[1])
    butterfly_limit = 1.06 * max(
        np.max(np.abs(erps["Reference"][:, display])),
        np.max(np.abs(erps["Full"][:, display])),
    )
    maps = {
        branch: metrics[metrics["branch"] == branch]
        .sort_values("channel_index")["signed_snr_sme"]
        .to_numpy()
        for branch in DISPLAY_BRANCHES
    }
    snr_limit = 1.05 * max(np.max(np.abs(values)) for values in maps.values())
    x = radius * np.sin(np.deg2rad(theta))
    y = radius * np.cos(np.deg2rad(theta))

    fig = plt.figure(figsize=(7.2, 4.65), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.08, 1.0])
    butterfly_axes = [fig.add_subplot(grid[0, :2]), fig.add_subplot(grid[0, 2:])]
    for axis, branch, color, title in [
        (butterfly_axes[0], "Reference", "#555555", "(a) Low-artifact reference"),
        (butterfly_axes[1], "Full", "#D9534F", "(b) Full SPAR-EEG"),
    ]:
        for channel in erps[branch]:
            axis.plot(times[display], channel[display], color=color, alpha=0.38, linewidth=0.62)
        axis.axvline(0, color="#777777", linestyle=":", linewidth=0.8)
        axis.axhline(0, color="#AAAAAA", linewidth=0.5)
        axis.axvspan(ERP_WINDOW_MS[0], ERP_WINDOW_MS[1], color="#999999", alpha=0.08)
        axis.set_xlim(DISPLAY_WINDOW_MS)
        axis.set_ylim(-butterfly_limit, butterfly_limit)
        axis.set_title(title, loc="left", fontsize=9)
        axis.grid(axis="y", color="#E3E3E3", linewidth=0.45)
        axis.set_xlabel("Time relative to stimulus (ms)")
    butterfly_axes[0].set_ylabel("Amplitude (uV)")
    butterfly_axes[1].tick_params(labelleft=False)

    titles = ["(c) Reference", "(d) EMG", "(e) EOG", "(f) Full"]
    topography_axes = []
    for column, (branch, title) in enumerate(zip(DISPLAY_BRANCHES, titles)):
        axis = fig.add_subplot(grid[1, column])
        draw_topography(
            axis,
            x,
            y,
            radius,
            maps[branch],
            -snr_limit,
            snr_limit,
            "RdBu_r",
        )
        axis.set_title(title, fontsize=8.5, pad=4)
        topography_axes.append(axis)

    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=-snr_limit, vmax=snr_limit), cmap="RdBu_r"),
        ax=topography_axes,
        orientation="horizontal",
        shrink=0.72,
        pad=0.025,
        aspect=28,
    )
    colorbar.set_label("Signed SME-based ERP SNR, 250-600 ms", fontsize=7.5)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_topography(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    radius: np.ndarray,
    values: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str,
) -> None:
    head_radius = float(np.max(radius) * 1.08)
    grid_axis = np.linspace(-head_radius, head_radius, 180)
    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)
    interpolator = RBFInterpolator(
        np.column_stack([x, y]),
        np.asarray(values, dtype=float),
        kernel="thin_plate_spline",
        smoothing=0.002,
    )
    interpolated = interpolator(
        np.column_stack([grid_x.ravel(), grid_y.ravel()])
    ).reshape(grid_x.shape)
    interpolated = np.ma.array(
        interpolated,
        mask=grid_x**2 + grid_y**2 > head_radius**2,
    )
    axis.contourf(
        grid_x,
        grid_y,
        interpolated,
        levels=np.linspace(vmin, vmax, 41),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extend="both",
    )
    axis.scatter(
        x,
        y,
        s=7,
        color="black",
        edgecolor="white",
        linewidth=0.2,
        zorder=4,
    )
    axis.add_patch(
        plt.Circle(
            (0, 0),
            head_radius,
            fill=False,
            color="black",
            linewidth=0.8,
            zorder=5,
        )
    )
    axis.plot(
        [0, -0.035, 0.035, 0],
        [head_radius, head_radius * 1.09, head_radius * 1.09, head_radius],
        color="black",
        linewidth=0.7,
        zorder=5,
    )
    axis.set_xlim(-head_radius * 1.12, head_radius * 1.12)
    axis.set_ylim(-head_radius * 1.10, head_radius * 1.15)
    axis.set_aspect("equal")
    axis.axis("off")


if __name__ == "__main__":
    main()
