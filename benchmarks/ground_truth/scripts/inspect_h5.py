from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eeg_eval.datasets import inspect_h5_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect H5 benchmark files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--pattern", default="*.h5")
    parser.add_argument("--max-records", type=int, default=3)
    args = parser.parse_args()

    files = sorted(args.data_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found: {args.data_dir / args.pattern}")

    for path in files:
        info = inspect_h5_file(path, max_records=args.max_records)
        print(f"\n{info['file']}")
        print(f"  dataset: {info['dataset_name']}")
        print(f"  records: {info['n_records']}")
        for row in info["examples"]:
            print(
                "  {record}: signal={signal_shape}, reference={reference_shape}, "
                "artifacts={artifact_shape}, fs={fs:g}, art_samples={artifact_samples}".format(**row)
            )


if __name__ == "__main__":
    main()
