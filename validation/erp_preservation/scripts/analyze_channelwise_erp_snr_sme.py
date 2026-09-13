from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute channel-wise SME-based ERP SNR across SPAR branches."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-start-ms", type=float, default=250.0)
    parser.add_argument("--window-end-ms", type=float, default=600.0)
    parser.add_argument(
        "--stable-reference-snr",
        type=float,
        default=1.0,
        help=(
            "Exploratory threshold used only when summarizing retention ratios. "
            "All channel-wise SNR values are always retained."
        ),
    )
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
    rereferenced = {
        branch: average_rereference(values)
        for branch, values in data.items()
    }
    metrics = channelwise_snr_sme(
        rereferenced,
        labels,
        times,
        args.window_start_ms,
        args.window_end_ms,
        args.stable_reference_snr,
    )
    summary = summarize_retention(metrics, args.stable_reference_snr)
    percentage_table = build_percentage_retention_table(metrics)

    stem = window_stem(args.window_start_ms, args.window_end_ms)
    metrics.to_csv(args.out_dir / f"channelwise_erp_snr_sme_{stem}.csv", index=False)
    summary.to_csv(args.out_dir / f"channelwise_erp_snr_sme_summary_{stem}.csv", index=False)
    percentage_table.to_csv(
        args.out_dir / f"channelwise_erp_detectability_percent_{stem}.csv",
        index=False,
    )

    configure_plotting()
    plot_snr_scatter(metrics, args.out_dir, stem)
    plot_snr_topographies(
        metrics,
        labels,
        theta,
        radius,
        args.stable_reference_snr,
        args.out_dir,
        stem,
    )
    plot_signal_noise_snr_topographies(
        metrics,
        theta,
        radius,
        args.out_dir,
        stem,
    )
    plot_detectability_percentage_topographies(
        metrics,
        theta,
        radius,
        args.stable_reference_snr,
        args.out_dir,
        stem,
    )

    print(
        f"Computed analytic SME and SME-based ERP SNR for {len(labels)} channels, "
        f"{data['Reference'].shape[2]} trials, and "
        f"{args.window_start_ms:g}-{args.window_end_ms:g} ms."
    )
    print("\nExploratory branch summary:")
    print(summary.to_string(index=False))


def ensure_3d(values: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        return values[:, :, None]
    if values.ndim != 3:
        raise ValueError(f"Expected 3-D branch data, found {values.shape}.")
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
    if any(len(values) != n_channels for values in (labels, theta, radius)):
        raise RuntimeError("Channel labels or coordinates do not match the data.")
    if len(times) != n_samples:
        raise RuntimeError("Time vector does not match the data.")
    if not all(np.isfinite(values).all() for values in data.values()):
        raise RuntimeError("At least one branch contains nonfinite samples.")


def average_rereference(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=0, keepdims=True)


def baseline_correct(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    baseline = (times >= -600) & (times < 0)
    if not baseline.any():
        raise RuntimeError("The requested prestimulus baseline is unavailable.")
    return values - values[:, baseline, :].mean(axis=1, keepdims=True)


def channelwise_snr_sme(
    data: dict[str, np.ndarray],
    labels: np.ndarray,
    times: np.ndarray,
    window_start: float,
    window_end: float,
    stable_reference_snr: float,
) -> pd.DataFrame:
    window = (times >= window_start) & (times <= window_end)
    if not window.any():
        raise RuntimeError("The requested ERP measurement window is unavailable.")

    corrected = {
        branch: baseline_correct(values, times)
        for branch, values in data.items()
    }
    branch_rows: dict[str, list[dict[str, float | int | str]]] = {}
    for branch, values in corrected.items():
        rows: list[dict[str, float | int | str]] = []
        for channel, label in enumerate(labels):
            trial_scores = values[channel, window, :].mean(axis=0)
            mean_amplitude = float(trial_scores.mean())
            analytic_sme = float(trial_scores.std(ddof=1) / np.sqrt(trial_scores.size))
            signed_snr = mean_amplitude / max(analytic_sme, np.finfo(float).eps)
            rows.append(
                {
                    "branch": branch,
                    "channel_index": channel + 1,
                    "channel": str(label),
                    "n_trials": trial_scores.size,
                    "mean_amplitude_uV": mean_amplitude,
                    "analytic_sme_uV": analytic_sme,
                    "snr_sme_signed": signed_snr,
                    "snr_sme_magnitude": abs(signed_snr),
                }
            )
        branch_rows[branch] = rows

    metrics = pd.DataFrame(
        [row for branch in BRANCHES for row in branch_rows[branch]]
    )
    reference = (
        metrics[metrics["branch"] == "Reference"]
        .set_index("channel")
        .add_prefix("reference_")
    )
    metrics = metrics.join(
        reference[
            [
                "reference_mean_amplitude_uV",
                "reference_analytic_sme_uV",
                "reference_snr_sme_signed",
                "reference_snr_sme_magnitude",
            ]
        ],
        on="channel",
    )
    epsilon = np.finfo(float).eps
    metrics["snr_sme_magnitude_retention"] = (
        metrics["snr_sme_magnitude"]
        / metrics["reference_snr_sme_magnitude"].clip(lower=epsilon)
    )
    metrics["amplitude_magnitude_retention"] = (
        metrics["mean_amplitude_uV"].abs()
        / metrics["reference_mean_amplitude_uV"].abs().clip(lower=epsilon)
    )
    metrics["analytic_sme_retention"] = (
        metrics["analytic_sme_uV"]
        / metrics["reference_analytic_sme_uV"].clip(lower=epsilon)
    )
    metrics["stable_reference_snr"] = (
        metrics["reference_snr_sme_magnitude"] >= stable_reference_snr
    )
    return metrics


def summarize_retention(
    metrics: pd.DataFrame, stable_reference_snr: float
) -> pd.DataFrame:
    rows = []
    reference = metrics[metrics["branch"] == "Reference"].sort_values("channel_index")
    reference_signed = reference["snr_sme_signed"].to_numpy()
    for branch in list(BRANCHES)[1:]:
        subset = metrics[metrics["branch"] == branch].sort_values("channel_index")
        stable = subset["stable_reference_snr"].to_numpy(dtype=bool)
        retention = subset.loc[stable, "snr_sme_magnitude_retention"].to_numpy()
        amplitude_retention = subset.loc[
            stable, "amplitude_magnitude_retention"
        ].to_numpy()
        sme_retention = subset.loc[stable, "analytic_sme_retention"].to_numpy()
        branch_signed = subset["snr_sme_signed"].to_numpy()
        map_metrics = signed_snr_map_metrics(
            reference_signed[stable], branch_signed[stable]
        )
        rows.append(
            {
                "branch": branch,
                "n_channels": len(subset),
                "n_stable_reference_channels": int(stable.sum()),
                "stable_reference_abs_snr_threshold": stable_reference_snr,
                "channelwise_signed_snr_pearson_r": pearsonr(
                    reference_signed, branch_signed
                ).statistic,
                "eligible_signed_snr_pearson_r": map_metrics["pearson_r"],
                "eligible_global_map_dissimilarity": map_metrics["gmd"],
                "eligible_centered_map_rms_ratio": map_metrics["map_rms_ratio"],
                "eligible_cosine_similarity": map_metrics["cosine_similarity"],
                "eligible_relative_map_rmse": map_metrics["relative_rmse"],
                "median_amplitude_magnitude_retention_stable": np.median(
                    amplitude_retention
                ),
                "median_analytic_sme_retention_stable": np.median(sme_retention),
                "median_abs_snr_retention_stable": np.median(retention),
                "min_abs_snr_retention_stable": np.min(retention),
                "max_abs_snr_retention_stable": np.max(retention),
                "stable_channels_retention_ge_0_90": int((retention >= 0.90).sum()),
                "stable_channels_retention_ge_1_00": int((retention >= 1.00).sum()),
            }
        )
    return pd.DataFrame(rows)


def signed_snr_map_metrics(
    reference: np.ndarray, observed: np.ndarray
) -> dict[str, float]:
    reference = np.asarray(reference, dtype=float)
    observed = np.asarray(observed, dtype=float)
    reference_centered = reference - reference.mean()
    observed_centered = observed - observed.mean()
    reference_rms = np.sqrt(np.mean(reference_centered**2))
    observed_rms = np.sqrt(np.mean(observed_centered**2))
    epsilon = np.finfo(float).eps
    return {
        "pearson_r": pearsonr(reference, observed).statistic,
        "gmd": np.sqrt(
            np.mean(
                (
                    reference_centered / max(reference_rms, epsilon)
                    - observed_centered / max(observed_rms, epsilon)
                )
                ** 2
            )
        ),
        "map_rms_ratio": observed_rms / max(reference_rms, epsilon),
        "cosine_similarity": float(
            np.dot(reference, observed)
            / max(np.linalg.norm(reference) * np.linalg.norm(observed), epsilon)
        ),
        "relative_rmse": float(
            np.sqrt(np.mean((observed - reference) ** 2))
            / max(np.sqrt(np.mean(reference**2)), epsilon)
        ),
    }


def build_percentage_retention_table(metrics: pd.DataFrame) -> pd.DataFrame:
    reference = (
        metrics[metrics["branch"] == "Reference"]
        .sort_values("channel_index")
        [[
            "channel_index",
            "channel",
            "reference_snr_sme_signed",
            "reference_snr_sme_magnitude",
            "stable_reference_snr",
        ]]
        .copy()
    )
    reference = reference.rename(
        columns={
            "reference_snr_sme_signed": "reference_signed_snr_sme",
            "reference_snr_sme_magnitude": "reference_abs_snr_sme",
            "stable_reference_snr": "reference_abs_snr_ge_threshold",
        }
    )
    table = reference
    for branch in list(BRANCHES)[1:]:
        values = (
            metrics[metrics["branch"] == branch]
            .sort_values("channel_index")
            [["channel", "snr_sme_magnitude_retention"]]
            .copy()
        )
        column = branch.lower().replace("+", "_") + "_detectability_percent"
        values[column] = 100.0 * values.pop("snr_sme_magnitude_retention")
        table = table.merge(values, on="channel", how="left", validate="one_to_one")
    return table.sort_values("channel_index").reset_index(drop=True)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
        }
    )


def plot_snr_scatter(metrics: pd.DataFrame, out_dir: Path, stem: str) -> None:
    reference = metrics[metrics["branch"] == "Reference"].sort_values("channel_index")
    x = reference["snr_sme_signed"].to_numpy()
    branches = list(BRANCHES)[1:]
    values = [x]
    for branch in branches:
        values.append(
            metrics[metrics["branch"] == branch]
            .sort_values("channel_index")["snr_sme_signed"]
            .to_numpy()
        )
    limit = 1.08 * max(abs(np.concatenate(values)))

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.4), constrained_layout=True)
    for axis, branch, y in zip(axes.ravel(), branches, values[1:]):
        r = pearsonr(x, y).statistic
        axis.scatter(x, y, s=28, color="#2F6B8A", edgecolor="white", linewidth=0.4)
        axis.plot([-limit, limit], [-limit, limit], color="black", linestyle=":", linewidth=0.8)
        axis.axhline(0, color="#BBBBBB", linewidth=0.5)
        axis.axvline(0, color="#BBBBBB", linewidth=0.5)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{branch}: channel-pattern r={r:.3f}")
        axis.set_xlabel("Reference SNR_SME")
        axis.set_ylabel(f"{branch} SNR_SME")
        axis.grid(color="#E4E4E4", linewidth=0.45)
    fig.savefig(out_dir / f"channelwise_erp_snr_sme_scatter_{stem}.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / f"channelwise_erp_snr_sme_scatter_{stem}.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_snr_topographies(
    metrics: pd.DataFrame,
    labels: np.ndarray,
    theta: np.ndarray,
    radius: np.ndarray,
    stable_reference_snr: float,
    out_dir: Path,
    stem: str,
) -> None:
    x = radius * np.sin(np.deg2rad(theta))
    y = radius * np.cos(np.deg2rad(theta))
    ordered = {
        branch: metrics[metrics["branch"] == branch].sort_values("channel_index")
        for branch in BRANCHES
    }
    snr_limit = 1.05 * max(
        abs(np.concatenate([frame["snr_sme_signed"].to_numpy() for frame in ordered.values()]))
    )
    stable = ordered["Reference"]["snr_sme_magnitude"].to_numpy() >= stable_reference_snr
    stable_ratios = np.concatenate(
        [
            ordered[branch].loc[stable, "snr_sme_magnitude_retention"].to_numpy()
            for branch in list(BRANCHES)[1:]
        ]
    )
    ratio_span = max(
        0.1,
        float(np.nanpercentile(abs(stable_ratios - 1), 95)),
    )
    ratio_span = min(ratio_span, 0.75)

    fig, axes = plt.subplots(2, len(BRANCHES), figsize=(8.0, 3.55), constrained_layout=True)
    for column, branch in enumerate(BRANCHES):
        frame = ordered[branch]
        snr_scatter = axes[0, column].scatter(
            x,
            y,
            c=frame["snr_sme_signed"],
            cmap="RdBu_r",
            vmin=-snr_limit,
            vmax=snr_limit,
            s=42,
            edgecolor="black",
            linewidth=0.3,
        )
        decorate_head(axes[0, column], radius)
        axes[0, column].set_title(branch)

        if branch == "Reference":
            axes[1, column].text(0, 0, "Reference", ha="center", va="center", color="#666666")
            decorate_head(axes[1, column], radius)
            continue
        axes[1, column].scatter(
            x[~stable],
            y[~stable],
            color="#D6D6D6",
            s=42,
            edgecolor="black",
            linewidth=0.3,
        )
        ratio_scatter = axes[1, column].scatter(
            x[stable],
            y[stable],
            c=frame.loc[stable, "snr_sme_magnitude_retention"],
            cmap="RdBu_r",
            vmin=1 - ratio_span,
            vmax=1 + ratio_span,
            s=42,
            edgecolor="black",
            linewidth=0.3,
        )
        decorate_head(axes[1, column], radius)

    axes[0, 0].set_ylabel("Signed SNR_SME", labelpad=7)
    axes[1, 0].set_ylabel("|SNR_SME| retention", labelpad=7)
    fig.colorbar(snr_scatter, ax=axes[0, :], shrink=0.72, label="Signed SNR_SME")
    fig.colorbar(
        ratio_scatter,
        ax=axes[1, 1:],
        shrink=0.72,
        label="Branch/reference |SNR_SME|",
    )
    fig.text(
        0.5,
        -0.01,
        f"Gray: reference |SNR_SME| < {stable_reference_snr:g}; retention not summarized.",
        ha="center",
        fontsize=7,
    )
    fig.savefig(out_dir / f"channelwise_erp_snr_sme_topographies_{stem}.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / f"channelwise_erp_snr_sme_topographies_{stem}.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_signal_noise_snr_topographies(
    metrics: pd.DataFrame,
    theta: np.ndarray,
    radius: np.ndarray,
    out_dir: Path,
    stem: str,
) -> None:
    """Visualize the signal, analytic SME, and SNR terms on common scales."""
    displayed = ["Reference", "EMG", "EOG", "Full"]
    ordered = {
        branch: metrics[metrics["branch"] == branch].sort_values("channel_index")
        for branch in displayed
    }
    x = radius * np.sin(np.deg2rad(theta))
    y = radius * np.cos(np.deg2rad(theta))

    amplitude_limit = 1.05 * max(
        abs(
            np.concatenate(
                [frame["mean_amplitude_uV"].to_numpy() for frame in ordered.values()]
            )
        )
    )
    sme_limit = 1.05 * max(
        np.concatenate([frame["analytic_sme_uV"].to_numpy() for frame in ordered.values()])
    )
    snr_limit = 1.05 * max(
        abs(
            np.concatenate(
                [frame["snr_sme_signed"].to_numpy() for frame in ordered.values()]
            )
        )
    )
    scales = [
        ("mean_amplitude_uV", -amplitude_limit, amplitude_limit, "RdBu_r"),
        ("analytic_sme_uV", 0.0, sme_limit, "viridis"),
        ("snr_sme_signed", -snr_limit, snr_limit, "RdBu_r"),
    ]

    fig, axes = plt.subplots(3, len(displayed), figsize=(7.2, 5.35), constrained_layout=True)
    for column, branch in enumerate(displayed):
        axes[0, column].set_title(branch, fontsize=9, pad=4)
        for row, (field, vmin, vmax, cmap) in enumerate(scales):
            draw_interpolated_topography(
                axes[row, column],
                x,
                y,
                radius,
                ordered[branch][field].to_numpy(),
                vmin,
                vmax,
                cmap,
            )

    row_labels = [
        "Mean ERP amplitude",
        "Analytic SME",
        "Signed SNR_SME",
    ]
    for axis, label in zip(axes[:, 0], row_labels):
        axis.set_ylabel(label, fontsize=8, labelpad=7)

    colorbar_labels = [
        "Mean amplitude (uV)",
        "Analytic SME (uV)",
        "Signed SNR_SME",
    ]
    for row, ((_, vmin, vmax, cmap), label) in enumerate(zip(scales, colorbar_labels)):
        mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        colorbar = fig.colorbar(mappable, ax=axes[row, :], shrink=0.70, pad=0.015)
        colorbar.set_label(label, fontsize=7.5)

    fig.savefig(
        out_dir / f"channelwise_erp_signal_sme_snr_topographies_{stem}.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / f"channelwise_erp_signal_sme_snr_topographies_{stem}.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_detectability_percentage_topographies(
    metrics: pd.DataFrame,
    theta: np.ndarray,
    radius: np.ndarray,
    stable_reference_snr: float,
    out_dir: Path,
    stem: str,
) -> None:
    """Map reference-normalized SNR retention without interpolating excluded sites."""
    displayed = list(BRANCHES)[1:]
    ordered = {
        branch: metrics[metrics["branch"] == branch].sort_values("channel_index")
        for branch in displayed
    }
    x = radius * np.sin(np.deg2rad(theta))
    y = radius * np.cos(np.deg2rad(theta))
    eligible = ordered[displayed[0]]["stable_reference_snr"].to_numpy(dtype=bool)
    percentages = np.concatenate(
        [
            100.0
            * ordered[branch].loc[eligible, "snr_sme_magnitude_retention"].to_numpy()
            for branch in displayed
        ]
    )
    span = max(10.0, 5.0 * np.ceil(np.max(np.abs(percentages - 100.0)) / 5.0))
    norm = TwoSlopeNorm(vmin=100.0 - span, vcenter=100.0, vmax=100.0 + span)

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.8), constrained_layout=True)
    for axis, branch in zip(axes.ravel(), displayed):
        values = 100.0 * ordered[branch]["snr_sme_magnitude_retention"].to_numpy()
        axis.scatter(
            x[~eligible],
            y[~eligible],
            s=270,
            color="#D5D5D5",
            edgecolor="#777777",
            linewidth=0.45,
            zorder=2,
        )
        scatter = axis.scatter(
            x[eligible],
            y[eligible],
            s=310,
            c=values[eligible],
            cmap="RdBu_r",
            norm=norm,
            edgecolor="black",
            linewidth=0.55,
            zorder=3,
        )
        for x_value, y_value, percentage in zip(
            x[eligible], y[eligible], values[eligible]
        ):
            axis.text(
                x_value,
                y_value,
                f"{percentage:.0f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="black",
                zorder=4,
            )
        decorate_head(axis, radius)
        median = np.median(values[eligible])
        axis.set_title(f"{branch} (median {median:.1f}%)", fontsize=9, pad=5)

    colorbar = fig.colorbar(scatter, ax=axes, shrink=0.78, pad=0.02)
    colorbar.set_label("SME-based ERP SNR retention (%)", fontsize=8)
    fig.text(
        0.5,
        -0.01,
        (
            f"Gray sensors: reference |SNR_SME| < {stable_reference_snr:g}; "
            "percentages not interpreted."
        ),
        ha="center",
        fontsize=7,
    )
    fig.savefig(
        out_dir / f"channelwise_erp_detectability_percent_topographies_{stem}.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / f"channelwise_erp_detectability_percent_topographies_{stem}.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def draw_interpolated_topography(
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
    levels = np.linspace(vmin, vmax, 41)
    axis.contourf(
        grid_x,
        grid_y,
        interpolated,
        levels=levels,
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


def decorate_head(axis: plt.Axes, radius: np.ndarray) -> None:
    head_radius = max(radius) * 1.08
    axis.add_patch(
        plt.Circle((0, 0), head_radius, fill=False, color="black", linewidth=0.7)
    )
    axis.plot(
        [0, -0.035, 0.035, 0],
        [head_radius, head_radius * 1.09, head_radius * 1.09, head_radius],
        color="black",
        linewidth=0.6,
    )
    axis.set_aspect("equal")
    axis.axis("off")


def window_stem(start: float, end: float) -> str:
    return f"{start:g}_{end:g}ms".replace(".", "p")


if __name__ == "__main__":
    main()
