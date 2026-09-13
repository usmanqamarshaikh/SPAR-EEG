"""Build the consolidated SPAR-EEG parameter-sensitivity figure for Revision R1."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    -20.0: "#B4493A",
    -10.0: "#D18A1F",
    0.0: "#356DB5",
}
LABELS = {
    -20.0: r"$-20$ dB (high)",
    -10.0: r"$-10$ dB (medium)",
    0.0: r"$0$ dB (low)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for supp_parameter_sensitivity.pdf and .png",
    )
    return parser.parse_args()


def add_series(ax, frame: pd.DataFrame, x_col: str) -> None:
    for snr in (-20.0, -10.0, 0.0):
        part = frame[np.isclose(frame["nominal_snr_db"], snr)].sort_values(x_col)
        x = part[x_col].to_numpy(dtype=float)
        median = part["median_delta_snr_db"].to_numpy(dtype=float)
        q1 = part["q1_delta_snr_db"].to_numpy(dtype=float)
        q3 = part["q3_delta_snr_db"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            median,
            yerr=np.vstack((median - q1, q3 - median)),
            color=COLORS[snr],
            marker="o" if snr == -20.0 else ("s" if snr == -10.0 else "^"),
            markersize=4.6,
            linewidth=1.5,
            elinewidth=1.0,
            capsize=2.2,
            label=LABELS[snr],
        )


def style_axis(ax, title: str, xlabel: str, default_x: float) -> None:
    ax.axhline(0, color="#777777", linewidth=0.7, linestyle=":")
    ax.axvline(default_x, color="#222222", linewidth=1.0, linestyle="--")
    ax.set_title(title, pad=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Artifact-region $\Delta$SNR (dB)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9DDE3", linewidth=0.6)
    ax.tick_params(direction="out", length=3)


def main() -> None:
    args = parse_args()
    evaluation_root = Path(__file__).resolve().parents[1]
    results = evaluation_root / "results"

    depth = pd.read_csv(
        results
        / "analysis_depth_sensitivity_multisnr_n500"
        / "depth_sensitivity_summary.csv"
    )
    depth = depth[depth["region"].eq("artifact")].copy()
    depth["nominal_snr_db"] = pd.to_numeric(depth["nominal_snr_db"])
    depth["depth"] = pd.to_numeric(depth["depth"])

    threshold = pd.read_csv(
        results
        / "analysis_zthr_sensitivity_multisnr_n500"
        / "zthr_sensitivity_compact_table.csv"
    )
    threshold["nominal_snr_db"] = pd.to_numeric(threshold["nominal_snr_db"])
    threshold["zthr"] = pd.to_numeric(threshold["zthr"])

    dilation = pd.read_csv(
        results
        / "analysis_dilation_sensitivity_multisnr_n500"
        / "dilation_sensitivity_compact_table.csv"
    )
    dilation["nominal_snr_db"] = pd.to_numeric(dilation["nominal_snr_db"])
    dilation["dilation_multiplier"] = pd.to_numeric(
        dilation["dilation_multiplier"]
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.0), constrained_layout=False)

    panels = [
        (axes[0, 0], depth[depth["family"].eq("vmd")], "depth", "(a) VMD mode count", "Number of VMD modes, $K$", 8),
        (axes[0, 1], depth[depth["family"].eq("ssa")], "depth", "(b) SSA embedding length", "Embedding length, $L$ (samples)", 12),
        (axes[1, 0], threshold[threshold["family"].eq("vmd")], "zthr", "(c) EMG proxy threshold", r"Artifact threshold, $z_{\mathrm{thr}}$", 3),
        (axes[1, 1], threshold[threshold["family"].eq("ssa")], "zthr", "(d) EOG proxy threshold", r"Artifact threshold, $z_{\mathrm{thr}}$", 3),
        (axes[2, 0], dilation[dilation["family"].eq("vmd")], "dilation_multiplier", "(e) EMG temporal dilation", "Dilation multiplier", 1),
        (axes[2, 1], dilation[dilation["family"].eq("ssa")], "dilation_multiplier", "(f) EOG temporal dilation", "Dilation multiplier", 1),
    ]

    for ax, frame, x_col, title, xlabel, default_x in panels:
        add_series(ax, frame, x_col)
        style_axis(ax, title, xlabel, default_x)

    for ax in axes[0, :]:
        ax.set_xticks([4, 8, 12])
    for ax in axes[1, :]:
        ax.set_xticks([1, 2, 3, 4, 5])
    axes[2, 0].set_xticks([0, 1, 2], ["0\n(0 s)", "1\n(0.14 s)", "2\n(0.28 s)"])
    axes[2, 1].set_xticks([0, 1, 2], ["0\n(0 s)", "1\n(0.12 s)", "2\n(0.24 s)"])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.945, bottom=0.07, hspace=0.48, wspace=0.30)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "supp_parameter_sensitivity.pdf"
    png_path = args.output_dir / "supp_parameter_sensitivity.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
