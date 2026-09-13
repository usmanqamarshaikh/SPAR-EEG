from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


DEFAULT_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "preprocessed_fp12"
    / "sub-001_task-P300trainrun1_run-6_fp12_first20s_branches.h5"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a saved FP1/FP2 branch H5 file.")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()

    with h5py.File(args.path, "r") as h5:
        names = ["baseline", "eog", "emg_eog", "full"]
        base = h5["baseline/data"][:]
        print(f"File: {args.path}")
        print(f"Groups: {list(h5.keys())}")
        print(f"baseline shape={base.shape} mean={np.mean(base):.6f} std={np.std(base):.6f}")
        ok = base.ndim == 2 and base.shape[0] == 2 and base.shape[1] > 0
        for name in names[1:]:
            data = h5[f"{name}/data"][:]
            rms_delta = float(np.sqrt(np.mean((data - base) ** 2)))
            print(
                f"{name:7s} shape={data.shape} mean={np.mean(data):.6f} "
                f"std={np.std(data):.6f} rms_delta_vs_baseline={rms_delta:.6f}"
            )
            ok = ok and data.shape == base.shape and np.isfinite(data).all()

    if not ok:
        print("\nFAIL: branch file failed shape/finite checks.")
        return 1
    print("\nPASS: branch file has expected FP1/FP2 shapes and finite values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
