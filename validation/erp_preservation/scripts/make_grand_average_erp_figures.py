from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.interpolate import RBFInterpolator
from scipy.io import loadmat


REGIONS = OrderedDict(
    [
        ("Frontal", ["FPZ", "F3", "FZ", "F4"]),
        ("Fronto-central", ["FC5", "FC1", "FC2", "FC6"]),
        ("Central", ["C3", "CZ", "C4"]),
        ("Parieto-occipital", ["PO7", "PO3", "POZ", "PO4", "PO8"]),
        ("Occipital", ["O1", "OZ", "O2"]),
    ]
)
KEY_CHANNELS = ["FZ", "CZ", "PZ", "OZ"]
DISPLAY_WINDOW_MS = (-200.0, 800.0)
ERP_WINDOW_MS = (0.0, 600.0)
BASELINE_WINDOW_MS = (-600.0, 0.0)
TOPOGRAPHY_WINDOWS = OrderedDict(
    [
        ("250-350 ms", (250.0, 350.0)),
        ("350-450 ms", (350.0, 450.0)),
        ("250-600 ms", (250.0, 600.0)),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exploratory grand-average ERP figures before and after SPAR-EEG."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("results/spar_erp_preservation/spar_erp_branch_data.mat"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/spar_erp_preservation/exploratory_figures"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bundle = loadmat(args.bundle, simplify_cells=True)
    reference = ensure_3d(np.asarray(bundle["data_reference"], dtype=float))
    emg = ensure_3d(np.asarray(bundle["data_emg"], dtype=float))
    eog = ensure_3d(np.asarray(bundle["data_eog"], dtype=float))
    full = ensure_3d(np.asarray(bundle["data_full"], dtype=float))
    labels = np.atleast_1d(bundle["channel_labels"]).astype(str)
    times = np.atleast_1d(bundle["times"]).astype(float)
    theta = np.atleast_1d(bundle["channel_theta"]).astype(float)
    radius = np.atleast_1d(bundle["channel_radius"]).astype(float)

    branch_shapes = {values.shape for values in (reference, emg, eog, full)}
    if len(branch_shapes) != 1:
        raise RuntimeError(f"Branch shape mismatch: {sorted(branch_shapes)}")
    if reference.shape[0] != labels.size or reference.shape[1] != times.size:
        raise RuntimeError("Channel labels or time vector do not match the data bundle.")

    reference = baseline_correct(average_rereference(reference), times)
    emg = baseline_correct(average_rereference(emg), times)
    eog = baseline_correct(average_rereference(eog), times)
    full = baseline_correct(average_rereference(full), times)
    erp_reference = reference.mean(axis=2)
    erp_emg = emg.mean(axis=2)
    erp_eog = eog.mean(axis=2)
    erp_full = full.mean(axis=2)

    configure_plotting()
    make_butterfly_figure(erp_reference, erp_full, times, args.out_dir)
    make_channel_figure(erp_reference, erp_full, labels, times, args.out_dir)
    make_region_figure(erp_reference, erp_full, labels, times, args.out_dir)
    make_gfp_figure(erp_reference, erp_full, times, args.out_dir)
    make_topography_figure(
        erp_reference, erp_full, theta, radius, times, args.out_dir
    )
    make_topography_pass_figure(
        OrderedDict(
            [
                ("Low-artifact baseline", erp_reference),
                ("EMG only", erp_emg),
                ("EOG only", erp_eog),
                ("Full SPAR-EEG", erp_full),
            ]
        ),
        theta,
        radius,
        times,
        args.out_dir,
    )
    write_metric_summary(erp_reference, erp_full, labels, times, args.out_dir)

    print(
        f"Generated ERP figures from {reference.shape[2]} trials, "
        f"{reference.shape[0]} channels, and {reference.shape[1]} samples."
    )
    print(f"Output directory: {args.out_dir.resolve()}")


def ensure_3d(values: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        return values[:, :, None]
    if values.ndim != 3:
        raise ValueError(f"Expected channels x samples x trials, found {values.shape}.")
    return values


def average_rereference(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=0, keepdims=True)


def baseline_correct(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    mask = (times >= BASELINE_WINDOW_MS[0]) & (times < BASELINE_WINDOW_MS[1])
    if not mask.any():
        raise RuntimeError("No samples are available in the requested baseline window.")
    return values - values[:, mask, :].mean(axis=1, keepdims=True)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_butterfly_figure(
    reference: np.ndarray, full: np.ndarray, times: np.ndarray, out_dir: Path
) -> None:
    display = display_mask(times)
    combined = np.concatenate([reference[:, display].ravel(), full[:, display].ravel()])
    y_limits = padded_limits(combined, fraction=0.06)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05), sharex=True, sharey=True)
    for axis, values, title, color in (
        (axes[0], reference, "(a) Low-artifact reference", "#333333"),
        (axes[1], full, "(b) Full SPAR-EEG", "#D64F4F"),
    ):
        for channel in values:
            axis.plot(times[display], channel[display], color=color, alpha=0.28, linewidth=0.65)
        decorate_erp_axis(axis, title, y_limits)
    axes[0].set_ylabel("Amplitude (\u00b5V)")
    fig.supxlabel("Time relative to stimulus (ms)", y=0.02)
    save_figure(fig, out_dir / "grand_average_erp_butterfly_before_after")


def make_channel_figure(
    reference: np.ndarray,
    full: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    out_dir: Path,
) -> None:
    lookup = label_lookup(labels)
    selected = [lookup[label] for label in KEY_CHANNELS if label in lookup]
    if not selected:
        raise RuntimeError("None of the requested key channels are available.")
    display = display_mask(times)
    combined = np.concatenate(
        [reference[selected][:, display].ravel(), full[selected][:, display].ravel()]
    )
    y_limits = padded_limits(combined, fraction=0.08)

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.25), sharex=True, sharey=True)
    for panel, (axis, label) in enumerate(zip(axes.flat, KEY_CHANNELS), start=1):
        index = lookup[label]
        plot_overlay(axis, reference[index], full[index], times)
        metrics = waveform_metrics(reference[index], full[index], times)
        axis.set_title(f"({chr(96 + panel)}) {label}", loc="left", fontsize=9)
        axis.text(
            0.98,
            0.96,
            f"r={metrics['r']:.3f}\nRMS ratio={metrics['rms_ratio']:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )
        axis.set_ylim(y_limits)
    axes[0, 0].set_ylabel("Amplitude (\u00b5V)")
    axes[1, 0].set_ylabel("Amplitude (\u00b5V)")
    axes[1, 0].set_xlabel("Time relative to stimulus (ms)")
    axes[1, 1].set_xlabel("Time relative to stimulus (ms)")
    add_legend(axes[0, 1])
    save_figure(fig, out_dir / "grand_average_erp_midline_channels_before_after")


def make_region_figure(
    reference: np.ndarray,
    full: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    out_dir: Path,
) -> None:
    lookup = label_lookup(labels)
    regional: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, requested in REGIONS.items():
        indices = [lookup[label] for label in requested if label in lookup]
        if indices:
            regional.append((name, reference[indices].mean(axis=0), full[indices].mean(axis=0)))

    display = display_mask(times)
    combined = np.concatenate(
        [values[display] for _, before, after in regional for values in (before, after)]
    )
    y_limits = padded_limits(combined, fraction=0.08)

    fig, axes = plt.subplots(3, 2, figsize=(7.1, 6.65), sharex=True)
    for panel, (axis, (name, before, after)) in enumerate(
        zip(axes.flat, regional), start=1
    ):
        plot_overlay(axis, before, after, times)
        metrics = waveform_metrics(before, after, times)
        axis.set_title(f"({chr(96 + panel)}) {name}", loc="left", fontsize=8.5)
        axis.text(
            0.98,
            0.96,
            f"r={metrics['r']:.3f}\nRMS ratio={metrics['rms_ratio']:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
        )
        axis.set_ylim(y_limits)

    gfp_axis = axes.flat[-1]
    reference_gfp = np.sqrt(np.mean(reference**2, axis=0))
    full_gfp = np.sqrt(np.mean(full**2, axis=0))
    display = display_mask(times)
    analysis = erp_mask(times)
    ratio = rms(full_gfp[analysis]) / max(rms(reference_gfp[analysis]), np.finfo(float).eps)
    gfp_axis.plot(
        times[display], reference_gfp[display], color="#222222", linewidth=1.25, label="Reference"
    )
    gfp_axis.plot(
        times[display], full_gfp[display], color="#D64F4F", linewidth=1.15, label="Full SPAR-EEG"
    )
    gfp_axis.axvline(0, color="#777777", linestyle=":", linewidth=0.85)
    gfp_axis.axvspan(250, 600, color="#D9D9D9", alpha=0.22, linewidth=0)
    gfp_axis.set_xlim(DISPLAY_WINDOW_MS)
    gfp_axis.set_ylim(bottom=0)
    gfp_axis.set_title("(f) Global field power", loc="left", fontsize=8.5)
    gfp_axis.text(
        0.98,
        0.96,
        f"RMS ratio={ratio:.3f}",
        transform=gfp_axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
    )
    gfp_axis.set_ylabel("GFP (\u00b5V)")
    gfp_axis.grid(axis="y", color="#E3E3E3", linewidth=0.5)

    for axis in axes[:, 0]:
        axis.set_ylabel("Amplitude (\u00b5V)")
    axes[2, 1].set_ylabel("GFP (\u00b5V)")
    fig.supxlabel("Time relative to stimulus (ms)", y=0.02)
    add_legend(axes[0, 1])
    save_figure(fig, out_dir / "grand_average_erp_regions_before_after")


def make_gfp_figure(
    reference: np.ndarray, full: np.ndarray, times: np.ndarray, out_dir: Path
) -> None:
    reference_gfp = np.sqrt(np.mean(reference**2, axis=0))
    full_gfp = np.sqrt(np.mean(full**2, axis=0))
    display = display_mask(times)
    analysis = erp_mask(times)
    ratio = rms(full_gfp[analysis]) / max(rms(reference_gfp[analysis]), np.finfo(float).eps)

    fig, axis = plt.subplots(figsize=(3.5, 2.8))
    axis.plot(times[display], reference_gfp[display], color="#222222", linewidth=1.5, label="Reference")
    axis.plot(times[display], full_gfp[display], color="#D64F4F", linewidth=1.4, label="Full SPAR-EEG")
    axis.axvline(0, color="#777777", linestyle=":", linewidth=0.9)
    axis.axvspan(250, 600, color="#D9D9D9", alpha=0.25, linewidth=0)
    axis.set_xlim(DISPLAY_WINDOW_MS)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Time relative to stimulus (ms)")
    axis.set_ylabel("Global field power (\u00b5V)")
    axis.text(
        0.98,
        0.96,
        f"0-600 ms RMS ratio={ratio:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
    )
    axis.grid(axis="y", color="#E0E0E0", linewidth=0.55)
    axis.legend(frameon=False, loc="upper left", fontsize=7.5)
    save_figure(fig, out_dir / "grand_average_erp_gfp_before_after")


def make_topography_figure(
    reference: np.ndarray,
    full: np.ndarray,
    theta: np.ndarray,
    radius: np.ndarray,
    times: np.ndarray,
    out_dir: Path,
) -> None:
    if theta.size != reference.shape[0] or radius.size != reference.shape[0]:
        raise RuntimeError("Topographic coordinates do not match the channel count.")

    x = radius * np.sin(np.deg2rad(theta))
    y = radius * np.cos(np.deg2rad(theta))
    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]] = []
    for label, (start, stop) in TOPOGRAPHY_WINDOWS.items():
        mask = (times >= start) & (times <= stop)
        before = reference[:, mask].mean(axis=1)
        after = full[:, mask].mean(axis=1)
        difference = after - before
        rows.append((label, before, after, difference, spatial_metrics(before, after)))

    erp_limit = max(
        float(np.max(np.abs(values)))
        for _, before, after, _, _ in rows
        for values in (before, after)
    )
    difference_limit = max(float(np.max(np.abs(row[3]))) for row in rows)
    erp_limit = max(erp_limit, np.finfo(float).eps)
    difference_limit = max(difference_limit, np.finfo(float).eps)

    fig, axes = plt.subplots(
        len(rows), 3, figsize=(7.1, 6.1), constrained_layout=True
    )
    for row_index, (label, before, after, difference, metrics) in enumerate(rows):
        draw_topography(
            axes[row_index, 0], x, y, radius, before, erp_limit, "RdBu_r"
        )
        draw_topography(
            axes[row_index, 1], x, y, radius, after, erp_limit, "RdBu_r"
        )
        draw_topography(
            axes[row_index, 2],
            x,
            y,
            radius,
            difference,
            difference_limit,
            "PuOr_r",
        )
        axes[row_index, 0].text(
            -0.12,
            0.5,
            label,
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=8,
        )
        axes[row_index, 1].text(
            0.5,
            -0.08,
            f"r={metrics['r']:.3f}, GFP ratio={metrics['gfp_ratio']:.3f}",
            transform=axes[row_index, 1].transAxes,
            va="top",
            ha="center",
            fontsize=6.7,
        )
        axes[row_index, 2].text(
            0.5,
            -0.08,
            f"GMD={metrics['gmd']:.3f}",
            transform=axes[row_index, 2].transAxes,
            va="top",
            ha="center",
            fontsize=6.7,
        )

    for axis, title in zip(
        axes[0], ["Low-artifact reference", "Full SPAR-EEG", "Difference"]
    ):
        axis.set_title(title, fontsize=9, pad=5)

    erp_mappable = ScalarMappable(
        norm=Normalize(vmin=-erp_limit, vmax=erp_limit), cmap="RdBu_r"
    )
    difference_mappable = ScalarMappable(
        norm=Normalize(vmin=-difference_limit, vmax=difference_limit), cmap="PuOr_r"
    )
    erp_colorbar = fig.colorbar(
        erp_mappable,
        ax=axes[:, :2],
        orientation="horizontal",
        shrink=0.72,
        pad=0.035,
        aspect=28,
    )
    erp_colorbar.set_label("Mean ERP amplitude (\u00b5V)", fontsize=7.5)
    difference_colorbar = fig.colorbar(
        difference_mappable,
        ax=axes[:, 2],
        orientation="horizontal",
        shrink=0.82,
        pad=0.035,
        aspect=20,
    )
    difference_colorbar.set_label("SPAR minus reference (\u00b5V)", fontsize=7.5)
    save_figure(
        fig,
        out_dir / "grand_average_erp_topographies_before_after",
        apply_tight_layout=False,
    )

    metric_path = out_dir / "grand_average_erp_topography_metrics.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["window", "map_r", "gmd", "gfp_ratio", "map_rrmse_percent"],
        )
        writer.writeheader()
        for label, _, _, _, metrics in rows:
            writer.writerow(
                {
                    "window": label,
                    "map_r": metrics["r"],
                    "gmd": metrics["gmd"],
                    "gfp_ratio": metrics["gfp_ratio"],
                    "map_rrmse_percent": 100 * metrics["rrmse"],
                }
            )


def make_topography_pass_figure(
    branches: OrderedDict[str, np.ndarray],
    theta: np.ndarray,
    radius: np.ndarray,
    times: np.ndarray,
    out_dir: Path,
) -> None:
    window = (times >= 250.0) & (times <= 450.0)
    maps = OrderedDict(
        (name, values[:, window].mean(axis=1)) for name, values in branches.items()
    )
    reference = next(iter(maps.values()))
    limit = max(float(np.max(np.abs(values))) for values in maps.values())
    limit = max(limit, np.finfo(float).eps)
    x = radius * np.sin(np.deg2rad(theta))
    y = radius * np.cos(np.deg2rad(theta))

    fig, axes = plt.subplots(1, len(maps), figsize=(7.1, 2.55), constrained_layout=True)
    metric_rows: list[dict[str, str | float]] = []
    for axis, (name, values) in zip(axes, maps.items()):
        draw_topography(axis, x, y, radius, values, limit, "RdBu_r")
        metrics = spatial_metrics(reference, values)
        title = name
        if name != "Low-artifact baseline":
            title += f"\nr={metrics['r']:.3f}, GFP={metrics['gfp_ratio']:.3f}"
        axis.set_title(title, fontsize=8, pad=5)
        metric_rows.append(
            {
                "branch": name,
                "window": "250-450 ms",
                "map_r": metrics["r"],
                "gmd": metrics["gmd"],
                "gfp_ratio": metrics["gfp_ratio"],
                "map_rrmse_percent": 100 * metrics["rrmse"],
            }
        )

    mappable = ScalarMappable(
        norm=Normalize(vmin=-limit, vmax=limit), cmap="RdBu_r"
    )
    colorbar = fig.colorbar(
        mappable,
        ax=axes,
        orientation="horizontal",
        shrink=0.72,
        pad=0.05,
        aspect=30,
    )
    colorbar.set_label("Mean ERP amplitude, 250-450 ms (\u00b5V)", fontsize=7.5)
    save_figure(
        fig,
        out_dir / "grand_average_erp_topographies_250_450_passes",
        apply_tight_layout=False,
    )

    metric_path = out_dir / "grand_average_erp_topographies_250_450_passes_metrics.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)


def draw_topography(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    radius: np.ndarray,
    values: np.ndarray,
    limit: float,
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
    outside = grid_x**2 + grid_y**2 > head_radius**2
    interpolated = np.ma.array(interpolated, mask=outside)

    levels = np.linspace(-limit, limit, 41)
    axis.contourf(
        grid_x,
        grid_y,
        interpolated,
        levels=levels,
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        extend="both",
    )
    axis.scatter(x, y, s=8, c="black", edgecolors="white", linewidths=0.25, zorder=4)
    axis.add_patch(
        plt.Circle((0, 0), head_radius, fill=False, color="black", linewidth=0.9, zorder=5)
    )
    axis.plot(
        [0, -0.035, 0.035, 0],
        [head_radius, head_radius * 1.10, head_radius * 1.10, head_radius],
        color="black",
        linewidth=0.8,
        zorder=5,
    )
    axis.set_xlim(-head_radius * 1.15, head_radius * 1.15)
    axis.set_ylim(-head_radius * 1.12, head_radius * 1.16)
    axis.set_aspect("equal")
    axis.axis("off")


def spatial_metrics(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    before = np.asarray(before, dtype=float) - np.mean(before)
    after = np.asarray(after, dtype=float) - np.mean(after)
    before_gfp = rms(before)
    after_gfp = rms(after)
    if before_gfp <= np.finfo(float).eps or after_gfp <= np.finfo(float).eps:
        gmd = np.nan
    else:
        gmd = rms(before / before_gfp - after / after_gfp)
    return {
        "r": correlation(before, after),
        "gmd": gmd,
        "gfp_ratio": after_gfp / max(before_gfp, np.finfo(float).eps),
        "rrmse": relative_change(before, after),
    }


def write_metric_summary(
    reference: np.ndarray,
    full: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    out_dir: Path,
) -> None:
    lookup = label_lookup(labels)
    rows: list[dict[str, str | float | int]] = []
    for label in KEY_CHANNELS:
        index = lookup[label]
        metrics = waveform_metrics(reference[index], full[index], times)
        rows.append({"type": "channel", "label": label, "n_channels": 1, **metrics})
    for name, requested in REGIONS.items():
        indices = [lookup[label] for label in requested if label in lookup]
        before = reference[indices].mean(axis=0)
        after = full[indices].mean(axis=0)
        rows.append(
            {
                "type": "region",
                "label": name,
                "n_channels": len(indices),
                **waveform_metrics(before, after, times),
            }
        )
    path = out_dir / "grand_average_erp_before_after_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def waveform_metrics(before: np.ndarray, after: np.ndarray, times: np.ndarray) -> dict[str, float]:
    mask = erp_mask(times)
    x = before[mask]
    y = after[mask]
    return {
        "r": correlation(x, y),
        "rrmse_percent": 100 * relative_change(x, y),
        "rms_ratio": rms(y) / max(rms(x), np.finfo(float).eps),
    }


def plot_overlay(axis: plt.Axes, before: np.ndarray, after: np.ndarray, times: np.ndarray) -> None:
    display = display_mask(times)
    axis.plot(times[display], before[display], color="#222222", linewidth=1.25, label="Reference")
    axis.plot(times[display], after[display], color="#D64F4F", linewidth=1.15, label="Full SPAR-EEG")
    axis.axvline(0, color="#777777", linestyle=":", linewidth=0.85)
    axis.axhline(0, color="#B0B0B0", linewidth=0.55)
    axis.axvspan(250, 600, color="#D9D9D9", alpha=0.22, linewidth=0)
    axis.set_xlim(DISPLAY_WINDOW_MS)
    axis.grid(axis="y", color="#E3E3E3", linewidth=0.5)


def decorate_erp_axis(axis: plt.Axes, title: str, y_limits: tuple[float, float]) -> None:
    axis.axvline(0, color="#777777", linestyle=":", linewidth=0.85)
    axis.axhline(0, color="#B0B0B0", linewidth=0.55)
    axis.axvspan(250, 600, color="#D9D9D9", alpha=0.22, linewidth=0)
    axis.set_xlim(DISPLAY_WINDOW_MS)
    axis.set_ylim(y_limits)
    axis.set_title(title, loc="left", fontsize=9)
    axis.grid(axis="y", color="#E3E3E3", linewidth=0.5)


def add_legend(axis: plt.Axes) -> None:
    axis.legend(frameon=False, fontsize=7, loc="lower right")


def save_figure(
    fig: plt.Figure, stem: Path, apply_tight_layout: bool = True
) -> None:
    if apply_tight_layout:
        fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def label_lookup(labels: np.ndarray) -> dict[str, int]:
    return {str(label).upper(): index for index, label in enumerate(labels)}


def display_mask(times: np.ndarray) -> np.ndarray:
    return (times >= DISPLAY_WINDOW_MS[0]) & (times <= DISPLAY_WINDOW_MS[1])


def erp_mask(times: np.ndarray) -> np.ndarray:
    return (times >= ERP_WINDOW_MS[0]) & (times <= ERP_WINDOW_MS[1])


def padded_limits(values: np.ndarray, fraction: float) -> tuple[float, float]:
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    padding = max((high - low) * fraction, 0.5)
    return low - padding, high + padding


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) <= np.finfo(float).eps or np.std(y) <= np.finfo(float).eps:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def relative_change(x: np.ndarray, y: np.ndarray) -> float:
    denominator = np.linalg.norm(x)
    if denominator <= np.finfo(float).eps:
        return np.nan
    return float(np.linalg.norm(y - x) / denominator)


if __name__ == "__main__":
    main()
