from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a clean-only EEGdenoiseNet H5 diagnostic file. The file uses "
            "the same 2-s epoch layout and middle-third normalization convention "
            "as the centered noisy EEGdenoiseNet benchmark, but no artifact is added."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/eeg-denoise-net"),
        help="Directory containing EEG_all_epochs.npy.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/clean_preservation/data"),
    )
    parser.add_argument("--max-records", type=int, default=3400)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "04_denoise-net_clean.h5"
    if out_path.exists() and not args.overwrite:
        print(f"Skipping existing {out_path}")
        return

    fs = 256
    n_samples = 2 * fs
    start = n_samples // 3
    end = 2 * (n_samples // 3)
    normalization_region = np.zeros(n_samples, dtype=bool)
    normalization_region[start:end] = True
    artifact_mask = np.zeros(n_samples, dtype=bool)

    eeg = np.load(args.raw_dir / "EEG_all_epochs.npy")
    n_records = min(args.max_records, eeg.shape[0])

    with h5py.File(out_path, "w") as h5:
        h5.attrs["name"] = "Denoise-Net clean EEG preservation diagnostic"
        h5.attrs["author"] = ""
        h5.attrs["preparation"] = "clean_only_middle_third_scaled"
        h5.attrs["artifact_layout"] = "none"
        h5.attrs["normalization_region"] = "middle_third"
        h5.attrs["normalization_start_sample"] = start
        h5.attrs["normalization_end_sample"] = end
        h5.attrs["nominal_snr_db"] = np.inf

        for n in range(n_records):
            clean = np.asarray(eeg[n], dtype=float).reshape(-1)
            if clean.size != n_samples:
                raise RuntimeError(
                    f"Record {n} has {clean.size} samples; expected {n_samples}."
                )

            scale = float(np.std(clean[normalization_region]))
            if not np.isfinite(scale) or scale <= np.finfo(float).eps:
                scale = float(np.std(clean))
            if not np.isfinite(scale) or scale <= np.finfo(float).eps:
                scale = 1.0

            reference = clean / scale

            group = h5.create_group(f"clean_{n}")
            group["eeg_signal"] = [reference]
            group["eeg_reference"] = [reference]
            group["artifacts"] = artifact_mask
            group.attrs["freq"] = fs
            group.attrs["filtered"] = ""
            group.attrs["artifact_layout"] = "none"
            group.attrs["normalization_start_sample"] = start
            group.attrs["normalization_end_sample"] = end

    print(f"Wrote {out_path} with {n_records} clean epochs.")


if __name__ == "__main__":
    main()
