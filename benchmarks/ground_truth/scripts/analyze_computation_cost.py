from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = [
    "wt_hard",
    "wt_soft",
    "wqn",
    "emd_cca",
    "emd_ica",
    "proposed_emg",
    "proposed_eog",
    "proposed_slow",
    "proposed_emg_eog",
    "proposed_full",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine SOTA and proposed computation-cost timing outputs."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/computation_cost"))
    args = parser.parse_args()

    raw_frames = []
    for name in ["computation_cost_sota_raw.csv", "computation_cost_proposed_raw.csv"]:
        path = args.out_dir / name
        if path.exists():
            raw_frames.append(pd.read_csv(path))

    if not raw_frames:
        raise SystemExit(f"No raw timing files found in {args.out_dir}")

    raw = pd.concat(raw_frames, ignore_index=True)
    raw.to_csv(args.out_dir / "computation_cost_all_raw.csv", index=False)

    summary = summarize(raw)
    summary.to_csv(args.out_dir / "computation_cost_all_summary.csv", index=False)
    write_latex(summary, args.out_dir / "computation_cost_compact_table.tex")

    print(f"Saved combined raw    : {args.out_dir / 'computation_cost_all_raw.csv'}")
    print(f"Saved combined summary: {args.out_dir / 'computation_cost_all_summary.csv'}")
    print(f"Saved LaTeX table     : {args.out_dir / 'computation_cost_compact_table.tex'}")


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    ok = raw[raw["status"].astype(str).eq("OK")].copy()
    if ok.empty:
        return pd.DataFrame()

    rows = []
    for (method_key, method, engine), group in ok.groupby(
        ["method_key", "method", "engine"], dropna=False
    ):
        ms = pd.to_numeric(group["ms_per_epoch"], errors="coerce").to_numpy(dtype=float)
        rtf = pd.to_numeric(group["realtime_factor"], errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "engine": engine,
                "method_key": method_key,
                "method": method,
                "n_timed_runs": int(len(group)),
                "n_records": int(group[["source_file", "record"]].drop_duplicates().shape[0]),
                "median_ms_per_epoch": float(np.nanmedian(ms)),
                "q1_ms_per_epoch": float(np.nanpercentile(ms, 25)),
                "q3_ms_per_epoch": float(np.nanpercentile(ms, 75)),
                "mean_ms_per_epoch": float(np.nanmean(ms)),
                "median_realtime_factor": float(np.nanmedian(rtf)),
                "q1_realtime_factor": float(np.nanpercentile(rtf, 25)),
                "q3_realtime_factor": float(np.nanpercentile(rtf, 75)),
            }
        )

    out = pd.DataFrame(rows)
    out["method_order"] = out["method_key"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(999)
    return out.sort_values(["method_order", "method_key"]).drop(columns=["method_order"]).reset_index(drop=True)


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Representative algorithm-only computation cost on EEGdenoiseNet $-10$ dB two-second epochs. Values are median [IQR] across timed records; lower values indicate faster processing.}",
        r"\label{tab:computation_cost}",
        r"\footnotesize",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & ms/epoch & Real-time factor \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        ms = f"{row['median_ms_per_epoch']:.1f} [{row['q1_ms_per_epoch']:.1f}, {row['q3_ms_per_epoch']:.1f}]"
        rtf = (
            f"{format_rtf(row['median_realtime_factor'])} "
            f"[{format_rtf(row['q1_realtime_factor'])}, {format_rtf(row['q3_realtime_factor'])}]"
        )
        lines.append(f"{escape_tex(str(row['method']))} & {ms} & {rtf} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def escape_tex(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def format_rtf(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
