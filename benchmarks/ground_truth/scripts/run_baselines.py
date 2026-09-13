from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tqdm import tqdm

from eeg_eval.datasets import iter_h5_records, mask_to_intervals, parse_dataset_info
from eeg_eval.metrics import evaluate_record, finalize_metrics_table
from eeg_eval.methods import available_methods, get_method


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline denoisers and common metrics.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--pattern", default="*.h5")
    parser.add_argument("--methods", nargs="+", default=["wt_hard", "wt_soft", "wqn"])
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.data_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No H5 files matched: {args.data_dir / args.pattern}")

    for method_key in args.methods:
        if method_key not in available_methods():
            print(f"Warning: {method_key!r} is not in the advertised method list.")
        method = get_method(method_key)

        for h5_path in files:
            safe_dataset = h5_path.stem.replace("+", "plus")
            out_path = args.out_dir / f"metrics__{safe_dataset}__{method_key}.parquet"
            if out_path.exists() and not args.overwrite:
                print(f"Skipping existing {out_path}")
                continue

            rows = []
            file_info = parse_dataset_info(h5_path.name)
            records = list(iter_h5_records(h5_path, max_records=args.max_records))
            desc = f"{method_key}: {h5_path.name}"

            for rec in tqdm(records, desc=desc):
                intervals = mask_to_intervals(rec.artifact_mask)
                restored = method.run(rec.signal, intervals, fs=rec.fs, reference=rec.reference)
                metric_rows = evaluate_record(
                    restored=restored,
                    signal=rec.signal,
                    reference=rec.reference,
                    artifact_mask=rec.artifact_mask,
                    fs=rec.fs,
                )
                for row in metric_rows:
                    row.update(
                        {
                            "method_key": method_key,
                            "method": method.name,
                            "h5_file": h5_path.name,
                            "dataset": rec.dataset_name,
                            "record": rec.record_name,
                            "fs": rec.fs,
                            "n_channels": rec.n_channels,
                            "n_samples": rec.n_samples,
                            "artifact_samples": int(rec.artifact_mask.sum()),
                            "artifact_fraction": float(rec.artifact_mask.mean()),
                            **file_info,
                        }
                    )
                    rows.append(row)

            df = finalize_metrics_table(pd.DataFrame(rows))
            df.to_parquet(out_path, index=False)
            df.to_csv(out_path.with_suffix(".csv"), index=False)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
