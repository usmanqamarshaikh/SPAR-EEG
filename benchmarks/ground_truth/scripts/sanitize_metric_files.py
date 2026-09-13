from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from eeg_eval.metrics import finalize_metrics_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite metric parquet/CSV pairs after replacing infinite values "
            "with NaN and adding input_snr_db/input_snr_source columns."
        )
    )
    parser.add_argument("dirs", nargs="+", type=Path)
    args = parser.parse_args()

    for folder in args.dirs:
        if not folder.exists():
            print(f"Skipping missing folder: {folder}")
            continue

        for parquet_path in sorted(folder.glob("*.parquet")):
            df = pd.read_parquet(parquet_path)
            df = finalize_metrics_table(df)
            df.to_parquet(parquet_path, index=False)
            df.to_csv(parquet_path.with_suffix(".csv"), index=False)
            print(f"Sanitized {parquet_path}")


if __name__ == "__main__":
    main()
