from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


FS = 256
N_SAMPLES = 2 * FS
THIRD = N_SAMPLES // 3
CENTER_START = THIRD
CENTER_STOP = 2 * THIRD
OCCUPANCIES = OrderedDict(
    (f"occupancy_{numerator}_9", f"{numerator}/9") for numerator in range(1, 10)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare paired centered EEGdenoiseNet EMG occupancy variants."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/eeg-denoise-net"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--snr-db", type=float, default=-10.0)
    parser.add_argument("--max-records", type=int, default=2**31 - 1)
    parser.add_argument(
        "--submitted-file", type=Path, default=Path("data/03_denoise-net_emg_-10dB.h5")
    )
    parser.add_argument("--geometry-file", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_file.exists() and not args.overwrite:
        print(f"Skipping existing occupancy file: {args.out_file}")
        return

    record_ids = pd.read_csv(args.manifest)["record_id"].astype(int).to_numpy()
    record_ids = record_ids[: min(len(record_ids), args.max_records)]
    if record_ids.size == 0:
        raise ValueError("Selection manifest contains no usable records.")

    eeg = np.load(args.raw_dir / "EEG_all_epochs.npy", mmap_mode="r")
    emg = np.load(args.raw_dir / "EMG_all_epochs.npy", mmap_mode="r")
    if np.any(record_ids < 0) or np.any(record_ids >= min(len(eeg), len(emg))):
        raise IndexError("At least one selected record ID is outside the EEG/EMG source arrays.")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = args.out_file.with_suffix(args.out_file.suffix + ".tmp")
    if temp_file.exists():
        temp_file.unlink()

    center_mask = interval_mask(CENTER_START, CENTER_STOP)
    measured = []
    with h5py.File(temp_file, "w") as h5:
        h5.attrs["name"] = "SPAR-EEG paired EMG artifact-occupancy robustness"
        h5.attrs["fs"] = FS
        h5.attrs["samples"] = N_SAMPLES
        h5.attrs["nominal_artifact_region_snr_db"] = args.snr_db
        h5.attrs["selection_manifest"] = str(args.manifest)
        h5.attrs["selected_record_count"] = len(record_ids)
        h5.attrs["occupancy_count"] = len(OCCUPANCIES)
        h5.attrs["reference_normalization"] = "submitted_center_third_clean_std"
        h5.attrs["mask_use"] = "mixture_generation_and_evaluation_only"
        h5.attrs["scaling_policy"] = "fixed_artifact_region_snr_per_occupancy"

        for order, record_id in enumerate(record_ids, start=1):
            clean_raw = np.asarray(eeg[record_id], dtype=float)
            emg_raw = np.asarray(emg[record_id], dtype=float)
            if clean_raw.size != N_SAMPLES or emg_raw.size != N_SAMPLES:
                raise ValueError(f"Record {record_id} does not contain {N_SAMPLES} samples.")

            reference = clean_raw / safe_std(clean_raw[center_mask])
            for numerator, (condition, label) in enumerate(OCCUPANCIES.items(), start=1):
                length = N_SAMPLES * numerator // 9
                # Preserve the submitted [170, 340) one-third convention while
                # expanding nested supports approximately symmetrically.
                start = N_SAMPLES * (9 - numerator) // 18
                stop = start + length
                mask = interval_mask(start, stop)
                raw_noise = np.zeros(N_SAMPLES, dtype=float)
                raw_noise[start:stop] = emg_raw[start:stop]
                scaled_noise = scale_noise(reference, raw_noise, mask, args.snr_db)
                signal = reference + scaled_noise
                local_snr = calculate_snr(reference[mask], scaled_noise[mask])
                whole_snr = calculate_snr(reference, scaled_noise)
                if not np.isfinite(local_snr) or abs(local_snr - args.snr_db) > 1e-9:
                    raise RuntimeError(
                        f"{condition}/emg_{record_id}: local SNR {local_snr} != {args.snr_db}"
                    )

                group = h5.create_group(f"{condition}__emg_{record_id}")
                group.create_dataset("eeg_signal", data=signal[None, :])
                group.create_dataset("eeg_reference", data=reference[None, :])
                group.create_dataset("noise", data=scaled_noise[None, :])
                group.create_dataset("artifacts", data=mask.astype(np.uint8))
                group.attrs["geometry"] = condition
                group.attrs["geometry_label"] = label
                group.attrs["geometry_order"] = numerator
                group.attrs["occupancy_numerator"] = numerator
                group.attrs["target_artifact_fraction"] = numerator / 9
                group.attrs["record_id"] = int(record_id)
                group.attrs["freq"] = FS
                group.attrs["nominal_snr_db"] = args.snr_db
                group.attrs["measured_snr_db"] = local_snr
                group.attrs["whole_epoch_snr_db"] = whole_snr
                group.attrs["artifact_samples"] = int(mask.sum())
                group.attrs["artifact_fraction"] = float(mask.mean())
                group.attrs["placement"] = f"[{start},{stop})"
                measured.append(local_snr)

            if order == 1 or order == len(record_ids) or order % max(1, len(record_ids) // 20) == 0:
                print(f"  prepared [{order}/{len(record_ids)}] record {record_id}")

    if args.out_file.exists():
        args.out_file.unlink()
    temp_file.replace(args.out_file)

    submitted_error = audit_submitted_third(args.out_file, args.submitted_file, record_ids)
    geometry_error = audit_geometry_anchors(args.out_file, args.geometry_file, record_ids)
    print(f"Wrote {args.out_file}")
    print(
        f"Records: {len(record_ids)} | occupancies: {len(OCCUPANCIES)} | "
        f"groups: {len(measured)}"
    )
    print(
        "Maximum artifact-region nominal-SNR error: "
        f"{np.max(np.abs(np.asarray(measured) - args.snr_db)):.3g} dB"
    )
    if submitted_error is not None:
        print(f"Maximum 3/9 error versus submitted input: {submitted_error:.3g}")
    if geometry_error is not None:
        print(f"Maximum 3/9, 6/9, 9/9 error versus geometry study: {geometry_error:.3g}")


def interval_mask(start: int, stop: int) -> np.ndarray:
    mask = np.zeros(N_SAMPLES, dtype=bool)
    mask[start:stop] = True
    return mask


def scale_noise(
    reference: np.ndarray, raw_noise: np.ndarray, mask: np.ndarray, snr_db: float
) -> np.ndarray:
    reference_variance = float(np.var(reference[mask]))
    noise_variance = float(np.var(raw_noise[mask]))
    if reference_variance <= np.finfo(float).eps or noise_variance <= np.finfo(float).eps:
        raise ValueError("Cannot scale an epoch with negligible signal or artifact variance.")
    target_noise_variance = reference_variance / (10 ** (snr_db / 10.0))
    return raw_noise * np.sqrt(target_noise_variance / noise_variance)


def safe_std(values: np.ndarray) -> float:
    value = float(np.std(np.asarray(values, dtype=float)))
    if value <= np.finfo(float).eps:
        raise ValueError("Encountered negligible source standard deviation.")
    return value


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    signal_variance = float(np.var(np.asarray(signal, dtype=float)))
    noise_variance = float(np.var(np.asarray(noise, dtype=float)))
    if signal_variance <= np.finfo(float).eps or noise_variance <= np.finfo(float).eps:
        return np.nan
    return float(10 * np.log10(signal_variance / noise_variance))


def audit_submitted_third(
    prepared_file: Path, submitted_file: Path, record_ids: np.ndarray
) -> float | None:
    if not submitted_file.exists():
        print(f"Submitted input not found; 3/9 reproduction audit skipped: {submitted_file}")
        return None
    maximum = 0.0
    with h5py.File(prepared_file, "r") as prepared, h5py.File(submitted_file, "r") as submitted:
        for record_id in record_ids:
            new = prepared[f"occupancy_3_9__emg_{record_id}"]
            old = submitted[f"emg_{record_id}"]
            maximum = max(maximum, compare_groups(new, old))
    if maximum > 1e-10:
        raise RuntimeError(f"Submitted 3/9 reproduction failed; maximum error {maximum}")
    return maximum


def audit_geometry_anchors(
    prepared_file: Path, geometry_file: Path | None, record_ids: np.ndarray
) -> float | None:
    if geometry_file is None or not geometry_file.exists():
        print("Prior geometry input not found; occupancy anchor audit skipped.")
        return None
    mapping = {
        "occupancy_3_9": "center_third",
        "occupancy_6_9": "center_two_thirds",
        "occupancy_9_9": "full_epoch",
    }
    maximum = 0.0
    with h5py.File(prepared_file, "r") as prepared, h5py.File(geometry_file, "r") as geometry:
        for record_id in record_ids:
            for occupancy, prior in mapping.items():
                maximum = max(
                    maximum,
                    compare_groups(
                        prepared[f"{occupancy}__emg_{record_id}"],
                        geometry[f"{prior}__emg_{record_id}"],
                    ),
                )
    if maximum > 1e-10:
        raise RuntimeError(f"Geometry-anchor reproduction failed; maximum error {maximum}")
    return maximum


def compare_groups(first: h5py.Group, second: h5py.Group) -> float:
    maximum = 0.0
    for dataset in ("eeg_signal", "eeg_reference", "noise", "artifacts"):
        if dataset in first and dataset in second:
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            np.asarray(first[dataset][()], dtype=float)
                            - np.asarray(second[dataset][()], dtype=float)
                        )
                    )
                ),
            )
    return maximum


if __name__ == "__main__":
    main()
