from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NODE_ORDER = [
    "emg_from_clean",
    "eog_from_clean",
    "slow_from_clean",
    "eog_after_emg",
    "slow_after_emg_eog",
]
NODE_LABELS = {
    "emg_from_clean": "EMG on reference",
    "eog_from_clean": "EOG on reference",
    "slow_from_clean": "Slow on reference",
    "eog_after_emg": "EOG after EMG",
    "slow_after_emg_eog": "Slow after EMG+EOG",
}
CONFIG_ORDER = ["EMG", "EOG", "Slow", "EMG+EOG", "Full"]
CONFIG_NODES = {
    "EMG": ["emg_from_clean"],
    "EOG": ["eog_from_clean"],
    "Slow": ["slow_from_clean"],
    "EMG+EOG": ["emg_from_clean", "eog_after_emg"],
    "Full": ["emg_from_clean", "eog_after_emg", "slow_after_emg_eog"],
}
FINAL_NODE = {
    "EMG": "emg_from_clean",
    "EOG": "eog_from_clean",
    "Slow": "slow_from_clean",
    "EMG+EOG": "eog_after_emg",
    "Full": "slow_after_emg_eog",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze clean-input activation and attenuation diagnostics."
    )
    parser.add_argument("--diagnostics-file", type=Path, required=True)
    parser.add_argument("--metrics-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--verification-tolerance", type=float, default=1e-9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = pd.read_csv(args.diagnostics_file)
    metrics = pd.read_csv(args.metrics_file)
    audit = audit_inputs(diagnostics, metrics, args.verification_tolerance)

    stage_summary = make_stage_summary(diagnostics)
    config_activation = make_configuration_activation(diagnostics)
    modification = make_modification_summary(metrics, config_activation)

    stage_summary.to_csv(args.out_dir / "clean_input_activation_stage_summary.csv", index=False)
    config_activation.to_csv(
        args.out_dir / "clean_input_configuration_record_activation.csv", index=False
    )
    modification.to_csv(
        args.out_dir / "clean_input_modification_tail_summary.csv", index=False
    )
    (args.out_dir / "clean_input_activation_stage_summary.tex").write_text(
        stage_table_latex(stage_summary), encoding="utf-8"
    )
    (args.out_dir / "clean_input_modification_tail_summary.tex").write_text(
        modification_table_latex(modification), encoding="utf-8"
    )
    make_figure(diagnostics, metrics, args.out_dir)
    report = make_report(audit, stage_summary, modification)
    (args.out_dir / "clean_input_activation_report.md").write_text(
        report, encoding="utf-8"
    )

    print(audit)
    print("\nStage summary:")
    print(stage_summary.to_string(index=False))
    print("\nConfiguration summary:")
    print(modification.to_string(index=False))


def audit_inputs(diagnostics: pd.DataFrame, metrics: pd.DataFrame, tolerance: float) -> str:
    required_nodes = set(NODE_ORDER)
    status = diagnostics["status"].fillna("").astype(str)
    failed = diagnostics[~status.eq("OK")]
    if not failed.empty:
        raise RuntimeError(f"Diagnostic run contains {len(failed)} failed rows.")

    records = diagnostics["record"].astype(str).unique()
    counts = diagnostics.groupby("record")["node"].nunique()
    if not (counts == len(required_nodes)).all():
        bad = counts[counts != len(required_nodes)]
        raise RuntimeError(f"Incomplete diagnostic nodes: {bad.to_dict()}")
    seen_nodes = set(diagnostics["node"].astype(str).unique())
    if seen_nodes != required_nodes:
        raise RuntimeError(f"Unexpected node set: {sorted(seen_nodes)}")

    max_relative = float(diagnostics["verification_relative_error"].max())
    max_absolute = float(diagnostics["verification_max_abs_error"].max())
    if not np.isfinite(max_relative) or max_relative > tolerance:
        raise RuntimeError(
            f"Saved-output equivalence failed: max relative error {max_relative:.3g} "
            f"> {tolerance:.3g}."
        )

    metric_records = set(metrics["record"].astype(str))
    missing_metrics = set(records) - metric_records
    if missing_metrics:
        raise RuntimeError(
            f"Preservation metrics missing for {len(missing_metrics)} diagnostic records."
        )
    expected_metric_rows = len(records) * len(CONFIG_ORDER)
    available = metrics[
        metrics["record"].astype(str).isin(records)
        & metrics["pass_config"].isin(CONFIG_ORDER)
    ]
    if len(available) != expected_metric_rows:
        raise RuntimeError(
            f"Expected {expected_metric_rows} preservation rows, found {len(available)}."
        )
    return (
        f"Audit passed: {len(records)} records, {len(diagnostics)} stage rows, "
        f"{len(available)} preservation rows; max saved-output relative error "
        f"{max_relative:.3g}, max absolute error {max_absolute:.3g}."
    )


def make_stage_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for node in NODE_ORDER:
        sub = diagnostics[diagnostics["node"] == node]
        rows.append(
            {
                "Stage": NODE_LABELS[node],
                "Epochs": int(sub["record"].nunique()),
                "Detector positive (%)": percent_true(sub["detector_positive"]),
                "Attenuation applied (%)": percent_true(sub["attenuation_applied"]),
                "No-region bypass (%)": percent_true(sub["bypass_no_region"]),
                "Baseline bypass (%)": percent_true(
                    sub["bypass_insufficient_baseline"]
                ),
                "Gate 1 positive (%)": percent_true(sub["gate1_pass"]),
                "Gate 2 positive (%)": percent_true(sub["gate2_pass"]),
                "Mask fraction median (%)": 100 * finite_quantile(sub["mask_fraction"], 0.5),
                "Attenuated samples median (%)": 100
                * finite_quantile(sub["attenuated_sample_fraction"], 0.5),
                "Attenuated samples q95 (%)": 100
                * finite_quantile(sub["attenuated_sample_fraction"], 0.95),
                "Stage RRMSE median (%)": finite_quantile(
                    sub["stage_input_rrmse_percent"], 0.5
                ),
                "Stage RRMSE q95 (%)": finite_quantile(
                    sub["stage_input_rrmse_percent"], 0.95
                ),
                "Median runtime (s)": finite_quantile(sub["elapsed_sec"], 0.5),
            }
        )
    return pd.DataFrame(rows)


def make_configuration_activation(diagnostics: pd.DataFrame) -> pd.DataFrame:
    indexed = diagnostics.set_index(["record", "node"])
    rows: list[dict[str, float | int | str]] = []
    for record in diagnostics["record"].astype(str).unique():
        record_nodes = indexed.xs(record, level="record")
        for config in CONFIG_ORDER:
            nodes = CONFIG_NODES[config]
            sub = record_nodes.loc[nodes, :]
            if isinstance(sub, pd.Series):
                sub = sub.to_frame().T
            final = record_nodes.loc[FINAL_NODE[config], :]
            rows.append(
                {
                    "record": record,
                    "pass_config": config,
                    "detector_positive": int(
                        pd.to_numeric(sub["detector_positive"], errors="coerce")
                        .fillna(0)
                        .astype(bool)
                        .any()
                    ),
                    "attenuation_applied": int(
                        pd.to_numeric(sub["attenuation_applied"], errors="coerce")
                        .fillna(0)
                        .astype(bool)
                        .any()
                    ),
                    "active_stage_count": int(
                        pd.to_numeric(sub["attenuation_applied"], errors="coerce")
                        .fillna(0)
                        .astype(bool)
                        .sum()
                    ),
                    "modified_sample_fraction_vs_clean": float(
                        final["modified_sample_fraction_vs_clean"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_modification_summary(
    metrics: pd.DataFrame, activation: pd.DataFrame
) -> pd.DataFrame:
    merged = metrics.merge(
        activation, on=["record", "pass_config"], how="inner", validate="one_to_one"
    )
    rows: list[dict[str, float | int | str]] = []
    for config in CONFIG_ORDER:
        sub = merged[merged["pass_config"] == config]
        rows.append(
            {
                "Configuration": config,
                "Epochs": int(sub["record"].nunique()),
                "Detector positive (%)": percent_true(sub["detector_positive"]),
                "Attenuation applied (%)": percent_true(sub["attenuation_applied"]),
                "RRMSE median (%)": finite_quantile(sub["rrmse_percent"], 0.5),
                "RRMSE IQR low (%)": finite_quantile(sub["rrmse_percent"], 0.25),
                "RRMSE IQR high (%)": finite_quantile(sub["rrmse_percent"], 0.75),
                "RRMSE q90 (%)": finite_quantile(sub["rrmse_percent"], 0.90),
                "RRMSE q95 (%)": finite_quantile(sub["rrmse_percent"], 0.95),
                "RRMSE q99 (%)": finite_quantile(sub["rrmse_percent"], 0.99),
                "RRMSE >10% (%)": 100 * float((sub["rrmse_percent"] > 10).mean()),
                "RRMSE >25% (%)": 100 * float((sub["rrmse_percent"] > 25).mean()),
                "Preservation SNR median (dB)": quantile_keep_inf(
                    sub["preservation_snr_db"], 0.5
                ),
                "Preservation SNR q10 (dB)": quantile_keep_inf(
                    sub["preservation_snr_db"], 0.10
                ),
                "Preservation SNR q05 (dB)": quantile_keep_inf(
                    sub["preservation_snr_db"], 0.05
                ),
                "Pearson r median": finite_quantile(sub["pearson_r"], 0.5),
                "Pearson r q10": finite_quantile(sub["pearson_r"], 0.10),
                "Pearson r q05": finite_quantile(sub["pearson_r"], 0.05),
                "PSD change median (dB)": finite_quantile(
                    sub["mean_abs_psd_change_db_1_45"], 0.5
                ),
                "PSD change q90 (dB)": finite_quantile(
                    sub["mean_abs_psd_change_db_1_45"], 0.90
                ),
                "PSD change q95 (dB)": finite_quantile(
                    sub["mean_abs_psd_change_db_1_45"], 0.95
                ),
                "Alpha change median (dB)": finite_quantile(
                    sub["alpha_power_change_db"], 0.5
                ),
                "Alpha change q05 (dB)": finite_quantile(
                    sub["alpha_power_change_db"], 0.05
                ),
                "Alpha change q95 (dB)": finite_quantile(
                    sub["alpha_power_change_db"], 0.95
                ),
                "Beta change median (dB)": finite_quantile(
                    sub["beta_power_change_db"], 0.5
                ),
                "Beta change q05 (dB)": finite_quantile(
                    sub["beta_power_change_db"], 0.05
                ),
                "Beta change q95 (dB)": finite_quantile(
                    sub["beta_power_change_db"], 0.95
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figure(diagnostics: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)

    stage = make_stage_summary(diagnostics)
    x = np.arange(len(stage))
    width = 0.36
    axes[0].barh(
        x - width / 2,
        stage["Detector positive (%)"],
        width,
        color="#4C78A8",
        label="Detector positive",
    )
    axes[0].barh(
        x + width / 2,
        stage["Attenuation applied (%)"],
        width,
        color="#E45756",
        label="Attenuation applied",
    )
    axes[0].set_yticks(
        x,
        [
            "EMG on reference",
            "EOG on reference",
            "Slow on reference",
            "EOG after EMG",
            "Slow after sequence",
        ],
        fontsize=7,
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("No-added-artifact epochs (%)")
    axes[0].set_xlim(0, 105)
    axes[0].grid(axis="x", color="#D9D9D9", linewidth=0.6)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    axes[0].set_title("(a) Internal activation", loc="left", fontsize=9)

    data = [
        pd.to_numeric(
            metrics.loc[metrics["pass_config"] == config, "rrmse_percent"],
            errors="coerce",
        ).dropna()
        for config in CONFIG_ORDER
    ]
    box = axes[1].boxplot(
        data,
        tick_labels=CONFIG_ORDER,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        widths=0.62,
    )
    colors = ["#4C78A8", "#72B7B2", "#B9B9B9", "#F2CF5B", "#E45756"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)
    axes[1].set_ylabel("Output-change RRMSE (%)")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[1].set_axisbelow(True)
    axes[1].set_title("(b) Modification distribution", loc="left", fontsize=9)

    for suffix in ("pdf", "png"):
        fig.savefig(
            out_dir / f"clean_input_activation_and_modification.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def make_report(audit: str, stage: pd.DataFrame, modification: pd.DataFrame) -> str:
    full = modification.set_index("Configuration").loc["Full"]
    slow = modification.set_index("Configuration").loc["Slow"]
    emg = stage.set_index("Stage").loc["EMG on reference"]
    eog = stage.set_index("Stage").loc["EOG on reference"]
    lines = [
        "# Clean-Input Activation and Attenuation Diagnostic",
        "",
        "## Scope",
        "",
        "All available clean EEGdenoiseNet reference epochs were processed using the frozen "
        "SPAR-EEG defaults. These epochs contain no synthetically added EOG or EMG, but they "
        "are not assumed to be physiologically artifact-free. Activation is therefore treated "
        "as a proxy for potentially unnecessary attenuation rather than proof of a false-positive "
        "decision.",
        "",
        "## Audit",
        "",
        audit,
        "",
        "## Descriptive findings",
        "",
        f"- The EMG detector was positive in {emg['Detector positive (%)']:.1f}% of clean-input "
        f"epochs and applied attenuation in {emg['Attenuation applied (%)']:.1f}%.",
        f"- The EOG detector was positive in {eog['Detector positive (%)']:.1f}% of clean-input "
        f"epochs and applied attenuation in {eog['Attenuation applied (%)']:.1f}%.",
        f"- The standalone slow configuration applied attenuation in "
        f"{slow['Attenuation applied (%)']:.1f}% of epochs.",
        f"- For the full sequence, median output-change RRMSE was "
        f"{full['RRMSE median (%)']:.2f}% and the 95th percentile was "
        f"{full['RRMSE q95 (%)']:.2f}%. Median preservation SNR was "
        f"{format_value(full['Preservation SNR median (dB)'])} dB, with a 5th-percentile "
        f"value of {format_value(full['Preservation SNR q05 (dB)'])} dB.",
        f"- Median full-sequence correlation was {full['Pearson r median']:.4f}; its "
        f"5th percentile was {full['Pearson r q05']:.4f}.",
        "",
        "## Interpretation boundary",
        "",
        "These results quantify how often the deployed logic activates and how strongly the "
        "output differs when the input contains no added benchmark artifact. They do not establish "
        "that every activation is erroneous, because the reference EEG may contain endogenous "
        "transients or residual recording contamination. The distributions and adverse-tail "
        "quantiles should therefore be reported alongside the contaminated-data improvements.",
        "",
    ]
    return "\n".join(lines)


def stage_table_latex(table: pd.DataFrame) -> str:
    lines = [
        "\\begin{table*}[!t]",
        "\\caption{Clean-input activation of individual SPAR-EEG stages. Detector-positive "
        "and attenuation-applied rates are reported separately because detection can be followed "
        "by a baseline or confirmation-gate bypass.}",
        "\\label{tab:clean_activation_stages}",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Stage & Detector (\\%) & Attenuated (\\%) & No-region bypass (\\%) & "
        "Baseline bypass (\\%) & Support median (\\%) & RRMSE q95 (\\%) \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{latex_escape(str(row['Stage']))} & "
            f"{fmt(row['Detector positive (%)'])} & "
            f"{fmt(row['Attenuation applied (%)'])} & "
            f"{fmt(row['No-region bypass (%)'])} & "
            f"{fmt(row['Baseline bypass (%)'])} & "
            f"{fmt(row['Attenuated samples median (%)'])} & "
            f"{fmt(row['Stage RRMSE q95 (%)'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def modification_table_latex(table: pd.DataFrame) -> str:
    lines = [
        "\\begin{table*}[!t]",
        "\\caption{Distributional clean-input modification by SPAR-EEG configuration. "
        "Values describe reference EEG epochs without synthetically added artifact.}",
        "\\label{tab:clean_modification_tails}",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Configuration & Activated (\\%) & RRMSE median & RRMSE q95 & "
        "SNR median & SNR q05 & $r$ median & $r$ q05 \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{latex_escape(str(row['Configuration']))} & "
            f"{fmt(row['Attenuation applied (%)'])} & "
            f"{fmt(row['RRMSE median (%)'])} & {fmt(row['RRMSE q95 (%)'])} & "
            f"{fmt(row['Preservation SNR median (dB)'])} & "
            f"{fmt(row['Preservation SNR q05 (dB)'])} & "
            f"{fmt(row['Pearson r median'], 4)} & {fmt(row['Pearson r q05'], 4)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def percent_true(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return 100 * float(clean.astype(bool).mean())


def finite_quantile(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float(clean.quantile(q))


def quantile_keep_inf(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    if np.isposinf(clean).all():
        return np.inf
    if np.isneginf(clean).all():
        return -np.inf
    return float(clean.quantile(q))


def fmt(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "--"
    if np.isposinf(value):
        return "$\\infty$"
    if np.isneginf(value):
        return "$-\\infty$"
    return f"{float(value):.{decimals}f}"


def format_value(value: float) -> str:
    if np.isposinf(value):
        return "infinite"
    if np.isneginf(value):
        return "negative infinite"
    return f"{value:.2f}"


def latex_escape(text: str) -> str:
    return text.replace("+", "$+$").replace("_", "\\_")


if __name__ == "__main__":
    main()
