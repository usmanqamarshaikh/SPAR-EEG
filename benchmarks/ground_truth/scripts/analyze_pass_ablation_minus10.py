from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = [
    "proposed_emg",
    "proposed_eog",
    "proposed_slow",
    "proposed_emg_eog",
    "proposed_full",
]
METHOD_LABELS = {
    "proposed_emg": "EMG only",
    "proposed_eog": "EOG only",
    "proposed_slow": "Slow only",
    "proposed_emg_eog": "EMG+EOG",
    "proposed_full": "Full",
}
NOISE_ORDER = ["emg", "eog", "eog+emg"]
NOISE_LABELS = {"emg": "EMG", "eog": "EOG", "eog+emg": "EOG+EMG"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the representative -10 dB proposed pass ablation."
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("results/metrics_pass_ablation_minus10"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/analysis_pass_ablation_minus10"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(args.metrics_root)
    summary = build_summary(metrics)
    summary.to_csv(args.out_dir / "pass_ablation_minus10_summary.csv", index=False)
    summary.to_parquet(args.out_dir / "pass_ablation_minus10_summary.parquet", index=False)

    compact = compact_table(summary)
    compact.to_csv(args.out_dir / "pass_ablation_minus10_compact_table.csv", index=False)

    detailed = detailed_table(summary)
    detailed.to_csv(args.out_dir / "pass_ablation_minus10_detailed_table.csv", index=False)

    write_latex_compact(compact, summary, args.out_dir / "pass_ablation_minus10_compact_table.tex")
    write_latex_detailed(detailed, args.out_dir / "pass_ablation_minus10_detailed_table.tex")

    print("Pass-ablation analysis completed.")
    print(f"  metrics rows : {len(metrics)}")
    print(f"  summary rows : {len(summary)}")
    print(f"  outputs      : {args.out_dir}")
    print()
    print(compact.to_string(index=False))


def load_metrics(metrics_root: Path) -> pd.DataFrame:
    paths = sorted(metrics_root.glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet metric files found under {metrics_root}")

    frames = []
    for path in paths:
        df = pd.read_parquet(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out[
        (out["benchmark_family"].astype(str).str.lower() == "denoise-net")
        & (out["region"].astype(str) == "artifact")
        & (pd.to_numeric(out["nominal_snr_db"], errors="coerce") == -10.0)
    ].copy()
    if out.empty:
        raise ValueError("No -10 dB artifact-region EEGdenoiseNet metrics found.")
    return out


def build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = metrics.copy()
    metrics["method_key"] = metrics["method_key"].astype(str)
    metrics["noise_type"] = metrics["noise_type"].astype(str).str.lower()
    for (noise_type, method_key), group in metrics.groupby(["noise_type", "method_key"], dropna=False):
        vals = pd.to_numeric(group["deltaSNR"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "noise_type": noise_type,
                "artifact": NOISE_LABELS.get(noise_type, noise_type),
                "method_key": method_key,
                "method": METHOD_LABELS.get(method_key, method_key),
                "n_records": int(group["record"].nunique()),
                "median_delta_snr_db": float(vals.median()) if len(vals) else np.nan,
                "mean_delta_snr_db": float(vals.mean()) if len(vals) else np.nan,
                "q1_delta_snr_db": float(vals.quantile(0.25)) if len(vals) else np.nan,
                "q3_delta_snr_db": float(vals.quantile(0.75)) if len(vals) else np.nan,
                "iqr_delta_snr_db": float(vals.quantile(0.75) - vals.quantile(0.25)) if len(vals) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["noise_order"] = out["noise_type"].map({n: i for i, n in enumerate(NOISE_ORDER)}).fillna(999)
    out["method_order"] = out["method_key"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(999)
    out["rank_median_delta_snr"] = out.groupby("noise_type")["median_delta_snr_db"].rank(
        ascending=False, method="min"
    )
    return out.sort_values(["noise_order", "method_order"]).reset_index(drop=True)


def compact_table(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot_table(
        index=["noise_order", "artifact"],
        columns="method_key",
        values="median_delta_snr_db",
        aggfunc="first",
    ).reset_index()
    ordered = ["artifact", *METHOD_ORDER]
    pivot = pivot.rename(columns={m: METHOD_LABELS[m] for m in METHOD_ORDER})
    ordered = ["artifact", *[METHOD_LABELS[m] for m in METHOD_ORDER]]
    return pivot.sort_values("noise_order")[ordered].reset_index(drop=True)


def detailed_table(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["median_iqr_delta_snr_db"] = out.apply(
        lambda r: (
            f"{r['median_delta_snr_db']:.2f} "
            f"[{r['q1_delta_snr_db']:.2f}, {r['q3_delta_snr_db']:.2f}]"
        ),
        axis=1,
    )
    return out[
        [
            "artifact",
            "method",
            "n_records",
            "median_delta_snr_db",
            "q1_delta_snr_db",
            "q3_delta_snr_db",
            "mean_delta_snr_db",
            "rank_median_delta_snr",
            "median_iqr_delta_snr_db",
        ]
    ]


def write_latex_compact(compact: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    best_by_row = {}
    method_cols = [METHOD_LABELS[m] for m in METHOD_ORDER]
    for idx, row in compact.iterrows():
        vals = pd.to_numeric(row[method_cols], errors="coerce")
        best_by_row[idx] = float(vals.max())

    n_text = record_count_text(summary)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Representative proposed-method pass ablation on EEGdenoiseNet at $-10$ dB. Values are median artifact-region $\Delta$SNR in dB across {n_text}.}}",
        r"\label{tab:pass_ablation_minus10}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Artifact & EMG only & EOG only & Slow only & EMG+EOG & Full \\",
        r"\midrule",
    ]
    for idx, row in compact.iterrows():
        vals = []
        for col in method_cols:
            val = float(row[col])
            text = f"{val:.2f}"
            if np.isfinite(val) and abs(val - best_by_row[idx]) < 1e-12:
                text = rf"\textbf{{{text}}}"
            vals.append(text)
        lines.append(f"{row['artifact']} & " + " & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_detailed(detailed: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Detailed representative pass ablation on EEGdenoiseNet at $-10$ dB. Values are artifact-region $\Delta$SNR in dB.}",
        r"\label{tab:supp_pass_ablation_minus10}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Artifact & Pass configuration & $n$ & Median [IQR] & Mean \\",
        r"\midrule",
    ]
    for _, row in detailed.iterrows():
        lines.append(
            f"{row['artifact']} & {row['method']} & {int(row['n_records'])} & "
            f"{row['median_iqr_delta_snr_db']} & {row['mean_delta_snr_db']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def record_count_text(summary: pd.DataFrame) -> str:
    counts = sorted(pd.to_numeric(summary["n_records"], errors="coerce").dropna().astype(int).unique())
    if len(counts) == 1:
        return f"{counts[0]} epochs per artifact condition"
    return "the available epochs in each artifact condition"


if __name__ == "__main__":
    main()
