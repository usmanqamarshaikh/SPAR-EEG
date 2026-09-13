from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from eeg_eval.datasets import ensure_2d_rows, iter_h5_records, parse_dataset_info
from eeg_eval.metrics import evaluate_record, finalize_metrics_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute common metrics from shape-preserving restored H5 files."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--restored-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--pattern", default="*.h5")
    parser.add_argument("--method-key", required=True)
    parser.add_argument("--method-label", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    method_label = args.method_label or args.method_key

    rows = []
    for data_path in sorted(args.data_dir.glob(args.pattern)):
        restored_path = args.restored_dir / data_path.name
        if not restored_path.exists():
            print(f"Missing restored file for {data_path.name}: {restored_path}")
            continue

        file_info = parse_dataset_info(data_path.name)
        records = list(iter_h5_records(data_path, max_records=args.max_records))

        with h5py.File(restored_path, "r") as restored_h5:
            for rec in tqdm(records, desc=f"{args.method_key}: {data_path.name}"):
                if rec.record_name not in restored_h5:
                    print(f"Missing record {rec.record_name} in {restored_path}")
                    continue
                restored = ensure_2d_rows(np.asarray(restored_h5[rec.record_name][()]))
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
                            "method_key": args.method_key,
                            "method": method_label,
                            "h5_file": data_path.name,
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
    suffix = ""
    if args.pattern != "*.h5":
        suffix = "__" + args.pattern.replace("*", "ALL").replace("+", "plus").replace(".h5", "")
    out_path = args.out_dir / f"metrics__{args.method_key}{suffix}.parquet"
    df.to_parquet(out_path, index=False)
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
