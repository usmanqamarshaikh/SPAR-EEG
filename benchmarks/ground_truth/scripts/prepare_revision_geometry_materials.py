from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GEOMETRY_ORDER = [
    ("center_third", "Centered 1/3"),
    ("random_third", "Random 1/3"),
    ("left_edge_third", "Left edge 1/3"),
    ("right_edge_third", "Right edge 1/3"),
    ("three_bursts_third", "Three bursts, total 1/3"),
    ("center_sixth", "Centered 1/6"),
    ("center_two_thirds", "Centered 2/3"),
    ("full_epoch", "Full epoch"),
]

POSITION_GEOMETRIES = GEOMETRY_ORDER[:5]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    workspace = root.parents[2]
    parser = argparse.ArgumentParser(
        description="Prepare locked artifact-geometry revision materials."
    )
    parser.add_argument(
        "--geometry-dir",
        type=Path,
        default=root / "results/artifact_geometry_emg_minus10_n500/analysis",
    )
    parser.add_argument(
        "--occupancy-dir",
        type=Path,
        default=root / "results/artifact_occupancy_emg_minus10_n500/analysis",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=workspace / "Manuscript/TNSRE_Revision_R1/materials/artifact_geometry",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    geom_records_path = args.geometry_dir / "artifact_geometry_record_metrics.csv"
    geom_diag_path = args.geometry_dir / "artifact_geometry_diagnostic_summary.csv"
    geom_pairs_path = args.geometry_dir / "artifact_geometry_paired_comparisons.csv"
    geom_manifest_path = args.geometry_dir / "selected_record_ids.csv"
    occ_records_path = args.occupancy_dir / "artifact_occupancy_record_metrics.csv"
    occ_diag_path = args.occupancy_dir / "artifact_occupancy_diagnostic_summary.csv"
    occ_pairs_path = args.occupancy_dir / "artifact_occupancy_adjacent_comparisons.csv"
    occ_transition_path = args.occupancy_dir / "artifact_occupancy_transition.csv"
    occ_bootstrap_path = args.occupancy_dir / "artifact_occupancy_transition_bootstrap.csv"
    occ_manifest_path = args.occupancy_dir / "selected_record_ids.csv"

    source_paths = [
        geom_records_path,
        geom_diag_path,
        geom_pairs_path,
        geom_manifest_path,
        occ_records_path,
        occ_diag_path,
        occ_pairs_path,
        occ_transition_path,
        occ_bootstrap_path,
        occ_manifest_path,
    ]
    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    geom_records = pd.read_csv(geom_records_path)
    geom_diag = pd.read_csv(geom_diag_path)
    geom_pairs = pd.read_csv(geom_pairs_path)
    occ_records = pd.read_csv(occ_records_path)
    occ_diag = pd.read_csv(occ_diag_path)
    occ_pairs = pd.read_csv(occ_pairs_path)
    occ_transition = pd.read_csv(occ_transition_path)

    geom_manifest = pd.read_csv(geom_manifest_path)
    occ_manifest = pd.read_csv(occ_manifest_path)
    pd.testing.assert_frame_equal(geom_manifest, occ_manifest)
    if len(geom_manifest) != 500:
        raise AssertionError(f"Expected 500 paired records, found {len(geom_manifest)}")

    audit_record_design(geom_records, "geometry", [key for key, _ in GEOMETRY_ORDER])
    audit_record_design(
        occ_records,
        "occupancy",
        [f"occupancy_{index}_9" for index in range(1, 10)],
    )

    geometry_summary = summarize_geometry(geom_records, geom_diag)
    occupancy_summary = summarize_occupancy(occ_records, occ_diag)

    geometry_summary.to_csv(args.out_dir / "geometry_full_sequence_summary.csv", index=False)
    occupancy_summary.to_csv(args.out_dir / "occupancy_full_sequence_summary.csv", index=False)
    geom_manifest.to_csv(args.out_dir / "selected_record_ids.csv", index=False)
    geom_pairs.loc[geom_pairs["branch"].eq("Full")].to_csv(
        args.out_dir / "geometry_full_sequence_paired_comparisons.csv", index=False
    )
    occ_pairs.to_csv(args.out_dir / "occupancy_adjacent_comparisons.csv", index=False)
    occ_transition.to_csv(args.out_dir / "occupancy_transition_summary.csv", index=False)

    write_geometry_table(geometry_summary, args.out_dir / "geometry_summary_candidate.tex")
    write_occupancy_table(occupancy_summary, args.out_dir / "occupancy_summary_candidate.tex")
    make_figure(
        geom_records,
        occupancy_summary,
        args.out_dir / "supp_artifact_geometry_robustness.pdf",
        args.out_dir / "supp_artifact_geometry_robustness.png",
    )

    hashes = pd.DataFrame(
        {
            "source_file": [str(path.resolve()) for path in source_paths],
            "sha256": [sha256(path) for path in source_paths],
        }
    )
    hashes.to_csv(args.out_dir / "source_hashes.csv", index=False)

    material_names = [
        "geometry_full_sequence_summary.csv",
        "occupancy_full_sequence_summary.csv",
        "selected_record_ids.csv",
        "geometry_full_sequence_paired_comparisons.csv",
        "occupancy_adjacent_comparisons.csv",
        "occupancy_transition_summary.csv",
        "geometry_summary_candidate.tex",
        "occupancy_summary_candidate.tex",
        "supp_artifact_geometry_robustness.pdf",
        "supp_artifact_geometry_robustness.png",
        "source_hashes.csv",
    ]
    material_hashes = pd.DataFrame(
        {
            "material_file": material_names,
            "sha256": [sha256(args.out_dir / name) for name in material_names],
        }
    )
    material_hashes.to_csv(args.out_dir / "material_hashes.csv", index=False)

    print(f"Prepared revision materials: {args.out_dir}")
    print(geometry_summary.to_string(index=False))
    print(occupancy_summary.to_string(index=False))


def audit_record_design(frame: pd.DataFrame, condition: str, expected: list[str]) -> None:
    full_artifact = frame.loc[
        frame["branch"].eq("Full") & frame["region"].eq("artifact")
    ].copy()
    if set(full_artifact[condition]) != set(expected):
        raise AssertionError(f"Unexpected {condition} conditions")
    counts = full_artifact.groupby(condition)["record_id"].nunique()
    if not counts.eq(500).all():
        raise AssertionError(f"Incomplete paired design: {counts.to_dict()}")
    input_error = np.nanmax(np.abs(full_artifact["snr_before_db"].to_numpy() + 10.0))
    if input_error > 1e-9:
        raise AssertionError(f"Local input-SNR audit failed: {input_error}")


def summarize_geometry(records: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for key, label in GEOMETRY_ORDER:
        artifact = select(records, key, "geometry", "artifact")
        whole = select(records, key, "geometry", "whole")
        clean = select(records, key, "geometry", "clean")
        emg_diag = diagnostics.loc[
            diagnostics["geometry"].eq(key) & diagnostics["node"].eq("emg")
        ].iloc[0]
        rows.append(
            {
                "geometry": key,
                "geometry_label": label,
                "records": artifact["record_id"].nunique(),
                "artifact_fraction": artifact["artifact_fraction"].median(),
                "artifact_delta_snr_median_db": artifact["delta_snr_db"].median(),
                "artifact_delta_snr_q25_db": artifact["delta_snr_db"].quantile(0.25),
                "artifact_delta_snr_q75_db": artifact["delta_snr_db"].quantile(0.75),
                "clean_output_change_median": (
                    clean["output_change_rrmse"].median() if not clean.empty else np.nan
                ),
                "whole_correlation_before_median": whole["correlation_before"].median(),
                "whole_correlation_after_median": whole["correlation_after"].median(),
                "emg_bypass_percent": float(emg_diag["total_bypass_percent"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_occupancy(records: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for order in range(1, 10):
        key = f"occupancy_{order}_9"
        artifact = select(records, key, "occupancy", "artifact")
        whole = select(records, key, "occupancy", "whole")
        emg_diag = diagnostics.loc[
            diagnostics["occupancy"].eq(key) & diagnostics["node"].eq("emg")
        ].iloc[0]
        rows.append(
            {
                "occupancy": key,
                "occupancy_label": f"{order}/9",
                "occupancy_order": order,
                "records": artifact["record_id"].nunique(),
                "artifact_fraction": artifact["artifact_fraction"].median(),
                "whole_input_snr_median_db": whole["snr_before_db"].median(),
                "artifact_delta_snr_median_db": artifact["delta_snr_db"].median(),
                "artifact_delta_snr_q25_db": artifact["delta_snr_db"].quantile(0.25),
                "artifact_delta_snr_q75_db": artifact["delta_snr_db"].quantile(0.75),
                "artifact_error_removed_median_percent": artifact[
                    "artifact_error_energy_removed_percent"
                ].median(),
                "artifact_error_removed_q25_percent": artifact[
                    "artifact_error_energy_removed_percent"
                ].quantile(0.25),
                "artifact_error_removed_q75_percent": artifact[
                    "artifact_error_energy_removed_percent"
                ].quantile(0.75),
                "whole_correlation_before_median": whole["correlation_before"].median(),
                "whole_correlation_before_q25": whole["correlation_before"].quantile(0.25),
                "whole_correlation_before_q75": whole["correlation_before"].quantile(0.75),
                "whole_correlation_after_median": whole["correlation_after"].median(),
                "whole_correlation_after_q25": whole["correlation_after"].quantile(0.25),
                "whole_correlation_after_q75": whole["correlation_after"].quantile(0.75),
                "emg_bypass_percent": float(emg_diag["total_bypass_percent"]),
            }
        )
    return pd.DataFrame(rows)


def select(frame: pd.DataFrame, key: str, column: str, region: str) -> pd.DataFrame:
    return frame.loc[
        frame[column].eq(key) & frame["branch"].eq("Full") & frame["region"].eq(region)
    ]


def make_figure(
    geometry_records: pd.DataFrame,
    occupancy: pd.DataFrame,
    pdf_path: Path,
    png_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
        }
    )
    teal = "#28777A"
    orange = "#D66A2C"
    gray = "#6B6B6B"
    grid = "#D9D9D9"

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.3), constrained_layout=True)
    ax = axes[0, 0]
    box_data = []
    labels = []
    for key, label in POSITION_GEOMETRIES:
        values = select(geometry_records, key, "geometry", "artifact")["delta_snr_db"]
        box_data.append(values.to_numpy())
        labels.append(label.replace(", total", "\ntotal"))
    boxes = ax.boxplot(
        box_data,
        tick_labels=labels,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": orange, "linewidth": 1.4},
        whiskerprops={"color": "#333333", "linewidth": 0.8},
        capprops={"color": "#333333", "linewidth": 0.8},
        boxprops={"color": "#333333", "linewidth": 0.8},
    )
    for box in boxes["boxes"]:
        box.set_facecolor("#9CC4C2")
    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.set_ylabel(r"Artifact-region $\Delta$SNR (dB)")
    ax.set_title("(a) Position and fragmentation at one-third occupancy", loc="left")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", color=grid, linewidth=0.6)

    x = occupancy["occupancy_order"].to_numpy()
    labels_x = occupancy["occupancy_label"].tolist()
    add_transition(axes[0, 1])
    axes[0, 1].fill_between(
        x,
        occupancy["artifact_delta_snr_q25_db"],
        occupancy["artifact_delta_snr_q75_db"],
        color=teal,
        alpha=0.18,
        linewidth=0,
    )
    axes[0, 1].plot(
        x,
        occupancy["artifact_delta_snr_median_db"],
        color=teal,
        marker="o",
        linewidth=1.5,
    )
    axes[0, 1].axhline(0, color="#777777", linewidth=0.7)
    axes[0, 1].set_ylabel(r"Artifact-region $\Delta$SNR (dB)")
    axes[0, 1].set_title("(b) Suppression across artifact occupancy", loc="left")
    format_occupancy_axis(axes[0, 1], x, labels_x, grid)

    add_transition(axes[1, 0])
    axes[1, 0].fill_between(
        x,
        occupancy["artifact_error_removed_q25_percent"],
        occupancy["artifact_error_removed_q75_percent"],
        color=teal,
        alpha=0.18,
        linewidth=0,
    )
    axes[1, 0].plot(
        x,
        occupancy["artifact_error_removed_median_percent"],
        color=teal,
        marker="o",
        linewidth=1.5,
    )
    axes[1, 0].axhline(0, color="#777777", linewidth=0.7)
    axes[1, 0].set_ylim(-3, 103)
    axes[1, 0].set_ylabel("Artifact error removed (%)")
    axes[1, 0].set_title("(c) Artifact-error reduction", loc="left")
    format_occupancy_axis(axes[1, 0], x, labels_x, grid)

    add_transition(axes[1, 1])
    for prefix, color, label in (
        ("whole_correlation_before", gray, "Before"),
        ("whole_correlation_after", orange, "After"),
    ):
        axes[1, 1].fill_between(
            x,
            occupancy[f"{prefix}_q25"],
            occupancy[f"{prefix}_q75"],
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        axes[1, 1].plot(
            x,
            occupancy[f"{prefix}_median"],
            color=color,
            marker="o",
            linewidth=1.4,
            label=label,
        )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("Whole-epoch correlation")
    axes[1, 1].set_title("(d) Whole-epoch reconstruction", loc="left")
    axes[1, 1].legend(frameon=False, loc="upper right")
    format_occupancy_axis(axes[1, 1], x, labels_x, grid)

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_transition(ax: plt.Axes) -> None:
    ax.axvspan(4, 5, color="#EFEFEF", alpha=0.8, zorder=0)
    ax.axvline(4.5, color="#777777", linestyle="--", linewidth=0.8, zorder=1)


def format_occupancy_axis(ax: plt.Axes, x: np.ndarray, labels: list[str], grid: str) -> None:
    ax.set_xticks(x, labels)
    ax.set_xlabel("Artifact occupancy")
    ax.grid(axis="y", color=grid, linewidth=0.6)


def write_geometry_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{EMG artifact-geometry robustness at $-10$~dB artifact-region input SNR. Values are full-sequence medians over the same 500 paired records; brackets give the interquartile range. Clean-region output change is the RRMSE between the processed output and clean reference outside the synthesis mask.}",
        r"\label{tab:supp_geometry_robustness}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Geometry & Artifact fraction & Artifact $\Delta$SNR (dB) & Clean-region output-change RRMSE & Whole $r$ after & EMG bypass (\%) \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        clean = "--" if pd.isna(row.clean_output_change_median) else f"{row.clean_output_change_median:.3f}"
        delta = (
            f"{row.artifact_delta_snr_median_db:.2f} "
            f"[{row.artifact_delta_snr_q25_db:.2f}, {row.artifact_delta_snr_q75_db:.2f}]"
        )
        lines.append(
            f"{row.geometry_label} & {row.artifact_fraction:.3f} & {delta} & {clean} & "
            f"{row.whole_correlation_after_median:.3f} & {row.emg_bypass_percent:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_occupancy_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Centered EMG occupancy sweep at fixed $-10$~dB artifact-region input SNR. Values are full-sequence medians over the same 500 paired records; brackets give the interquartile range.}",
        r"\label{tab:supp_occupancy_robustness}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Occupancy & Whole-input SNR (dB) & Artifact $\Delta$SNR (dB) & Error removed (\%) & Whole $r$ after & EMG bypass (\%) \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        delta = (
            f"{row.artifact_delta_snr_median_db:.2f} "
            f"[{row.artifact_delta_snr_q25_db:.2f}, {row.artifact_delta_snr_q75_db:.2f}]"
        )
        lines.append(
            f"{row.occupancy_label} & {row.whole_input_snr_median_db:.2f} & {delta} & "
            f"{row.artifact_error_removed_median_percent:.1f} & "
            f"{row.whole_correlation_after_median:.3f} & {row.emg_bypass_percent:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    main()
