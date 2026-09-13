from __future__ import annotations

import argparse
import gc
import json
import platform
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from eeg_eval.datasets import iter_h5_records, mask_to_intervals
from eeg_eval.methods import available_methods, get_method


METHOD_LABELS = {
    "wt_hard": "WT-hard",
    "wt_soft": "WT-soft",
    "wqn": "WQN",
    "emd_cca": "EMD-CCA",
    "emd_ica": "EMD-ICA",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure algorithm-only wall-clock runtime for Python benchmark "
            "denoisers. H5 reading, metric computation, and output writing are "
            "excluded from timed sections."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--pattern", default="03_denoise-net_*_-10dB.h5")
    parser.add_argument("--out-dir", type=Path, default=Path("results/computation_cost"))
    parser.add_argument("--methods", nargs="+", default=available_methods())
    parser.add_argument("--max-records", type=int, default=100, help="Maximum records per H5 file.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warm-up calls per method.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "computation_cost_sota_raw.csv"
    summary_path = args.out_dir / "computation_cost_sota_summary.csv"
    metadata_path = args.out_dir / "computation_cost_sota_metadata.json"
    if raw_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists; use --overwrite to replace: {raw_path}")

    files = sorted(args.data_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No H5 files matched: {args.data_dir / args.pattern}")

    records = []
    for h5_path in files:
        records.extend(list(iter_h5_records(h5_path, max_records=args.max_records)))
    if not records:
        raise SystemExit("No records found for timing.")

    rows: list[dict[str, object]] = []
    for method_key in args.methods:
        if method_key not in available_methods():
            print(f"Warning: {method_key!r} is not in the advertised method list.")

        method = get_method(method_key)
        label = METHOD_LABELS.get(method_key, method.name)

        for rec in records[: max(0, args.warmup)]:
            intervals = mask_to_intervals(rec.artifact_mask)
            try:
                _ = method.run(rec.signal, intervals, fs=rec.fs, reference=rec.reference)
            except Exception as exc:
                print(f"Warm-up warning for {method_key}/{rec.record_name}: {exc}")

        desc = f"{method_key} timing"
        for rec in tqdm(records, desc=desc):
            intervals = mask_to_intervals(rec.artifact_mask)
            duration_sec = rec.n_samples / float(rec.fs)
            for repeat_idx in range(args.repeats):
                gc.collect()
                t0 = time.perf_counter()
                status = "OK"
                out_shape = ""
                try:
                    restored = method.run(rec.signal, intervals, fs=rec.fs, reference=rec.reference)
                    out_shape = "x".join(str(v) for v in np.asarray(restored).shape)
                except Exception as exc:
                    status = f"FAIL: {exc}"
                elapsed_sec = time.perf_counter() - t0

                rows.append(
                    {
                        "engine": "python",
                        "method_key": method_key,
                        "method": label,
                        "source_file": rec.file_path.name,
                        "record": rec.record_name,
                        "repeat": repeat_idx + 1,
                        "status": status,
                        "elapsed_sec": elapsed_sec,
                        "ms_per_epoch": elapsed_sec * 1000.0,
                        "duration_sec": duration_sec,
                        "realtime_factor": elapsed_sec / duration_sec if duration_sec > 0 else np.nan,
                        "fs": rec.fs,
                        "n_channels": rec.n_channels,
                        "n_samples": rec.n_samples,
                        "out_shape": out_shape,
                    }
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(raw_path, index=False)
    summarize(raw).to_csv(summary_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "script": Path(__file__).name,
                "data_dir": str(args.data_dir),
                "pattern": args.pattern,
                "methods": args.methods,
                "max_records_per_file": args.max_records,
                "repeats": args.repeats,
                "warmup": args.warmup,
                "n_files": len(files),
                "n_records_loaded": len(records),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python": sys.version,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved raw timings: {raw_path}")
    print(f"Saved summary    : {summary_path}")


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    ok = raw[raw["status"].astype(str).eq("OK")].copy()
    if ok.empty:
        return pd.DataFrame()

    rows = []
    for (method_key, method), group in ok.groupby(["method_key", "method"], sort=False):
        ms = group["ms_per_epoch"].to_numpy(dtype=float)
        rtf = group["realtime_factor"].to_numpy(dtype=float)
        rows.append(
            {
                "engine": group["engine"].iloc[0],
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
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
