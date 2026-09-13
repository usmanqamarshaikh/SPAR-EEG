from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from eeg_eval.datasets import iter_h5_records


@dataclass
class StreamConfig:
    min_len_sec: float = 2.0
    max_len_sec: float = 4.0
    search_radius_sec: float = 2.0
    extend_step_sec: float = 1.0
    rms_win_sec: float = 0.10
    smooth_score_sec: float = 0.03
    amp_weight: float = 1.0
    slope_weight: float = 0.5
    rms_weight: float = 2.0
    zc_bonus: float = 0.15
    accept_cost_mode: str = "percentile"
    accept_percentile: float = 10.0
    accept_cost_fixed: float = 10.0
    min_denoise_len_sec: float = 0.20


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Characterize adaptive streaming epoch lengths without applying "
            "any denoising passes. The boundary logic mirrors the MATLAB H5 "
            "runner settings used for continuous PhysioBank processing."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--pattern", default="01_physiobank.h5")
    parser.add_argument("--out-dir", type=Path, default=Path("results/adaptive_epoch_lengths"))
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    epoch_path = args.out_dir / "adaptive_epoch_lengths_epochs.csv"
    record_path = args.out_dir / "adaptive_epoch_lengths_by_record.csv"
    summary_path = args.out_dir / "adaptive_epoch_lengths_summary.csv"
    latex_path = args.out_dir / "adaptive_epoch_lengths_summary_table.tex"
    metadata_path = args.out_dir / "adaptive_epoch_lengths_metadata.json"
    if epoch_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists; use --overwrite to replace: {epoch_path}")

    files = sorted(args.data_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No H5 files matched: {args.data_dir / args.pattern}")

    cfg = StreamConfig()
    rows: list[dict[str, object]] = []
    for h5_path in files:
        records = list(iter_h5_records(h5_path, max_records=args.max_records))
        for rec in tqdm(records, desc=f"adaptive epochs: {h5_path.name}"):
            for ch in range(rec.n_channels):
                x = np.asarray(rec.signal[ch], dtype=float).reshape(-1)
                xn, norm_method = robust_normalize_for_scoring(x)
                score = compute_boundary_score(xn, rec.fs, cfg)
                accept_cost = get_accept_cost(score, cfg)
                epochs = adaptive_epoch_signal_from_score(score, rec.fs, cfg, accept_cost)

                for ep_idx, ep in enumerate(epochs, start=1):
                    start = ep["start"]
                    end = ep["end"]
                    n_samples = end - start + 1
                    duration_sec = n_samples / float(rec.fs)
                    is_terminal = ep_idx == len(epochs)
                    rows.append(
                        {
                            "source_file": h5_path.name,
                            "dataset_name": rec.dataset_name,
                            "record": rec.record_name,
                            "channel": ch,
                            "epoch_index": ep_idx,
                            "start_sample": start,
                            "end_sample_inclusive": end,
                            "n_samples": n_samples,
                            "duration_sec": duration_sec,
                            "is_terminal_segment": is_terminal,
                            "is_short_terminal_remainder": bool(is_terminal and duration_sec < cfg.min_len_sec),
                            "nominal_end_sample": ep["nominal_end"],
                            "cut_score": ep["cut_score"],
                            "accept_cost": accept_cost,
                            "accepted_direct": ep["accepted_direct"],
                            "fs": rec.fs,
                            "record_samples": rec.n_samples,
                            "record_duration_sec": rec.n_samples / float(rec.fs),
                            "normalization": norm_method,
                        }
                    )

    epochs_df = pd.DataFrame(rows)
    if epochs_df.empty:
        raise SystemExit("No adaptive epochs were produced.")

    record_df = summarize_by_record(epochs_df)
    summary_df = summarize_overall(epochs_df, record_df, cfg)

    epochs_df.to_csv(epoch_path, index=False)
    record_df.to_csv(record_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_latex_summary(summary_df, latex_path)
    metadata_path.write_text(
        json.dumps(
            {
                "script": Path(__file__).name,
                "data_dir": str(args.data_dir),
                "pattern": args.pattern,
                "max_records": args.max_records,
                "config": asdict(cfg),
                "platform": platform.platform(),
                "python": sys.version,
                "note": "No denoising passes are applied; only adaptive boundary selection is characterized.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved epoch rows     : {epoch_path}")
    print(f"Saved record summary : {record_path}")
    print(f"Saved overall summary: {summary_path}")
    print(f"Saved LaTeX table    : {latex_path}")
    print(summary_df.to_string(index=False))


def robust_normalize_for_scoring(x: np.ndarray) -> tuple[np.ndarray, str]:
    x = np.asarray(x, dtype=float).reshape(-1)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad < np.finfo(float).eps:
        sx = np.nanstd(x, ddof=1)
        if sx < np.finfo(float).eps or not np.isfinite(sx):
            sx = 1.0
        mu = np.nanmean(x)
        return ((x - mu) / sx).astype(float), "std"

    sc = 1.4826 * mad
    return ((x - med) / sc).astype(float), "mad"


def compute_boundary_score(xn: np.ndarray, fs: float, cfg: StreamConfig) -> np.ndarray:
    xn = np.asarray(xn, dtype=float).reshape(-1)
    rms_w = max(3, int(round(cfg.rms_win_sec * float(fs))))
    dx = np.r_[0.0, np.diff(xn)]
    local_rms = np.sqrt(movmean_shrink(xn**2, rms_w))
    zc = np.zeros(xn.size, dtype=bool)
    zc[1:] = np.sign(xn[1:]) != np.sign(xn[:-1])
    score = (
        cfg.amp_weight * np.abs(xn)
        + cfg.slope_weight * np.abs(dx)
        + cfg.rms_weight * local_rms
        - cfg.zc_bonus * zc.astype(float)
    )
    return np.asarray(score, dtype=float).reshape(-1)


def movmean_shrink(x: np.ndarray, win: int) -> np.ndarray:
    """Approximate MATLAB movmean(x, win, 'Endpoints', 'shrink')."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    win = max(1, int(win))
    if win == 1 or n == 0:
        return x.copy()

    before = win // 2
    after = win - before - 1
    starts = np.maximum(0, np.arange(n) - before)
    stops = np.minimum(n, np.arange(n) + after + 1)
    finite = np.isfinite(x)
    values = np.where(finite, x, 0.0)
    cs = np.r_[0.0, np.cumsum(values)]
    cc = np.r_[0, np.cumsum(finite.astype(int))]
    sums = cs[stops] - cs[starts]
    counts = cc[stops] - cc[starts]
    out = np.divide(sums, counts, out=np.full(n, np.nan), where=counts > 0)
    return out


def get_accept_cost(score: np.ndarray, cfg: StreamConfig) -> float:
    mode = cfg.accept_cost_mode.lower().strip()
    if mode == "percentile":
        return float(np.nanpercentile(score, cfg.accept_percentile))
    if mode == "fixed":
        return float(cfg.accept_cost_fixed)
    raise ValueError(f"Unknown accept_cost_mode: {cfg.accept_cost_mode}")


def adaptive_epoch_signal_from_score(
    score: np.ndarray, fs: float, cfg: StreamConfig, accept_cost: float
) -> list[dict[str, object]]:
    """Mirror MATLAB adaptive_epoch_signal_from_score using 0-based indices."""
    score = np.asarray(score, dtype=float).reshape(-1)
    n = int(score.size)
    min_len = max(1, int(round(cfg.min_len_sec * float(fs))))
    max_len = max(min_len, int(round(cfg.max_len_sec * float(fs))))
    search_r = max(1, int(round(cfg.search_radius_sec * float(fs))))
    extend_st = max(1, int(round(cfg.extend_step_sec * float(fs))))

    epochs: list[dict[str, object]] = []
    start = 0
    while start < n:
        if start + min_len > n:
            epochs.append(
                {
                    "start": start,
                    "end": n - 1,
                    "nominal_end": n - 1,
                    "cut_score": float(score[n - 1]),
                    "accepted_direct": True,
                }
            )
            break

        nominal_end = start + min_len - 1
        best_end = nominal_end
        best_cost = np.inf
        accepted_direct = False
        cur_nominal = nominal_end
        hard_limit = min(n - 1, start + max_len - 1)

        while cur_nominal <= hard_limit:
            lo = max(start + min_len - 1, cur_nominal - search_r)
            hi = min(hard_limit, cur_nominal + search_r)
            cand_end, cand_cost = find_best_boundary_in_range(score, lo, hi)
            if cand_cost < best_cost:
                best_cost = cand_cost
                best_end = cand_end
            if cand_cost <= accept_cost:
                accepted_direct = True
                break
            cur_nominal += extend_st

        epochs.append(
            {
                "start": start,
                "end": best_end,
                "nominal_end": nominal_end,
                "cut_score": float(best_cost),
                "accepted_direct": bool(accepted_direct),
            }
        )
        start = best_end + 1

    return epochs


def find_best_boundary_in_range(score: np.ndarray, lo: int, hi: int) -> tuple[int, float]:
    # MATLAB uses score_raw(lo:hi), inclusive.
    window = score[lo : hi + 1]
    rel_idx = int(np.nanargmin(window))
    best_idx = lo + rel_idx
    return best_idx, float(score[best_idx])


def summarize_by_record(epochs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in epochs.groupby(["source_file", "record", "channel"], sort=True):
        durations = group["duration_sec"].to_numpy(dtype=float)
        rows.append(
            {
                "source_file": keys[0],
                "record": keys[1],
                "channel": keys[2],
                "n_epochs": int(len(group)),
                "record_duration_sec": float(group["record_duration_sec"].iloc[0]),
                "median_duration_sec": float(np.nanmedian(durations)),
                "q1_duration_sec": float(np.nanpercentile(durations, 25)),
                "q3_duration_sec": float(np.nanpercentile(durations, 75)),
                "min_duration_sec": float(np.nanmin(durations)),
                "max_duration_sec": float(np.nanmax(durations)),
                "pct_2_to_2p25_sec": float(np.nanmean((durations >= 2.0) & (durations <= 2.25)) * 100.0),
                "pct_ge_3p75_sec": float(np.nanmean(durations >= 3.75) * 100.0),
                "pct_accepted_direct": float(group["accepted_direct"].mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def summarize_overall(epochs: pd.DataFrame, record_summary: pd.DataFrame, cfg: StreamConfig) -> pd.DataFrame:
    base = {
            "dataset": "PhysioBank motion artifact",
            "n_records": int(record_summary[["source_file", "record", "channel"]].drop_duplicates().shape[0]),
            "total_duration_min": float(epochs[["source_file", "record", "channel", "record_duration_sec"]].drop_duplicates()["record_duration_sec"].sum() / 60.0),
            "min_allowed_sec": cfg.min_len_sec,
            "max_allowed_sec": cfg.max_len_sec,
        }
    rows = []
    for label, sub in [
        ("all_segments", epochs),
        ("full_windows_excluding_short_terminal_remainders", epochs[~epochs["is_short_terminal_remainder"]]),
    ]:
        durations = sub["duration_sec"].to_numpy(dtype=float)
        rows.append(
            {
            **base,
            "segment_set": label,
            "n_epochs": int(len(sub)),
            "n_short_terminal_remainders": int(sub["is_short_terminal_remainder"].sum()),
            "median_duration_sec": float(np.nanmedian(durations)),
            "q1_duration_sec": float(np.nanpercentile(durations, 25)),
            "q3_duration_sec": float(np.nanpercentile(durations, 75)),
            "p05_duration_sec": float(np.nanpercentile(durations, 5)),
            "p95_duration_sec": float(np.nanpercentile(durations, 95)),
            "min_duration_sec": float(np.nanmin(durations)),
            "max_duration_sec": float(np.nanmax(durations)),
            "pct_2_to_2p25_sec": float(np.nanmean((durations >= 2.0) & (durations <= 2.25)) * 100.0),
            "pct_ge_3p75_sec": float(np.nanmean(durations >= 3.75) * 100.0),
            "pct_accepted_direct": float(sub["accepted_direct"].mean() * 100.0),
        }
        )
    return pd.DataFrame(rows)


def write_latex_summary(summary: pd.DataFrame, path: Path) -> None:
    row = summary[summary["segment_set"].eq("full_windows_excluding_short_terminal_remainders")].iloc[0]
    all_row = summary[summary["segment_set"].eq("all_segments")].iloc[0]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Adaptive streaming epoch-length summary on the PhysioBank motion artifact dataset. No denoising passes were applied.}",
        r"\label{tab:adaptive_epoch_lengths}",
        r"\footnotesize",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        f"Records & {int(row['n_records'])} \\\\",
        f"Adaptive epochs & {int(row['n_epochs'])} \\\\",
        f"Short terminal remainders & {int(all_row['n_short_terminal_remainders'])} \\\\",
        f"Total duration & {row['total_duration_min']:.1f} min \\\\",
        f"Median length & {row['median_duration_sec']:.2f} s \\\\",
        f"IQR & [{row['q1_duration_sec']:.2f}, {row['q3_duration_sec']:.2f}] s \\\\",
        f"5th--95th percentile & [{row['p05_duration_sec']:.2f}, {row['p95_duration_sec']:.2f}] s \\\\",
        f"2.00--2.25 s epochs & {row['pct_2_to_2p25_sec']:.1f}\\% \\\\",
        f"$\\geq$3.75 s epochs & {row['pct_ge_3p75_sec']:.1f}\\% \\\\",
        f"Accepted at first candidate & {row['pct_accepted_direct']:.1f}\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
