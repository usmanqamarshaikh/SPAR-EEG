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
GEOMETRIES = OrderedDict(
    [
        ("center_third", "Centred one-third"),
        ("random_third", "Random single one-third"),
        ("left_edge_third", "Left-edge one-third"),
        ("right_edge_third", "Right-edge one-third"),
        ("three_bursts_third", "Three separated bursts (total one-third)"),
        ("center_sixth", "Centred one-sixth"),
        ("center_two_thirds", "Centred two-thirds"),
        ("full_epoch", "Full epoch"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare paired EEGdenoiseNet EMG geometry variants.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/eeg-denoise-net"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--snr-db", type=float, default=-10.0)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--max-records", type=int, default=2**31 - 1)
    parser.add_argument("--submitted-file", type=Path, default=Path("data/03_denoise-net_emg_-10dB.h5"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_file.exists() and not args.overwrite:
        print(f"Skipping existing geometry file: {args.out_file}")
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
        h5.attrs["name"] = "SPAR-EEG paired EMG artifact-geometry robustness"
        h5.attrs["fs"] = FS
        h5.attrs["samples"] = N_SAMPLES
        h5.attrs["nominal_snr_db"] = args.snr_db
        h5.attrs["selection_manifest"] = str(args.manifest)
        h5.attrs["selected_record_count"] = len(record_ids)
        h5.attrs["geometry_count"] = len(GEOMETRIES)
        h5.attrs["geometry_seed"] = args.seed
        h5.attrs["reference_normalization"] = "submitted_center_third_clean_std"
        h5.attrs["mask_use"] = "mixture_generation_and_evaluation_only"

        for order, record_id in enumerate(record_ids, start=1):
            clean_raw = np.asarray(eeg[record_id], dtype=float)
            emg_raw = np.asarray(emg[record_id], dtype=float)
            if clean_raw.size != N_SAMPLES or emg_raw.size != N_SAMPLES:
                raise ValueError(f"Record {record_id} does not contain {N_SAMPLES} samples.")

            reference = clean_raw / safe_std(clean_raw[center_mask])
            for geometry, label in GEOMETRIES.items():
                raw_noise, mask, placement = make_geometry(geometry, emg_raw, record_id, args.seed)
                scaled_noise = scale_noise(reference, raw_noise, mask, args.snr_db)
                signal = reference + scaled_noise
                snr = calculate_snr(reference[mask], scaled_noise[mask])
                if not np.isfinite(snr) or abs(snr - args.snr_db) > 1e-9:
                    raise RuntimeError(
                        f"{geometry}/emg_{record_id}: measured SNR {snr} != {args.snr_db}"
                    )

                group = h5.create_group(f"{geometry}__emg_{record_id}")
                group.create_dataset("eeg_signal", data=signal[None, :])
                group.create_dataset("eeg_reference", data=reference[None, :])
                group.create_dataset("noise", data=scaled_noise[None, :])
                group.create_dataset("artifacts", data=mask.astype(np.uint8))
                group.attrs["geometry"] = geometry
                group.attrs["geometry_label"] = label
                group.attrs["geometry_order"] = list(GEOMETRIES).index(geometry) + 1
                group.attrs["record_id"] = int(record_id)
                group.attrs["freq"] = FS
                group.attrs["nominal_snr_db"] = args.snr_db
                group.attrs["measured_snr_db"] = snr
                group.attrs["artifact_samples"] = int(mask.sum())
                group.attrs["artifact_fraction"] = float(mask.mean())
                group.attrs["placement"] = placement
                measured.append(snr)

            if order == 1 or order == len(record_ids) or order % max(1, len(record_ids) // 20) == 0:
                print(f"  prepared [{order}/{len(record_ids)}] record {record_id}")

    if args.out_file.exists():
        args.out_file.unlink()
    temp_file.replace(args.out_file)

    center_error = audit_submitted_center(args.out_file, args.submitted_file, record_ids)
    print(f"Wrote {args.out_file}")
    print(f"Records: {len(record_ids)} | geometries: {len(GEOMETRIES)} | groups: {len(measured)}")
    print(f"Maximum nominal-SNR error: {np.max(np.abs(np.asarray(measured) - args.snr_db)):.3g} dB")
    if center_error is not None:
        print(f"Maximum centred-condition error versus submitted input: {center_error:.3g}")


def make_geometry(
    geometry: str, emg: np.ndarray, record_id: int, seed: int
) -> tuple[np.ndarray, np.ndarray, str]:
    noise = np.zeros(N_SAMPLES, dtype=float)
    canonical = np.asarray(emg[CENTER_START:CENTER_STOP], dtype=float)

    if geometry == "center_third":
        start = CENTER_START
        place(noise, canonical, start)
        return noise, interval_mask(start, start + THIRD), f"[{start},{start + THIRD})"
    if geometry == "random_third":
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(record_id)]))
        start = int(rng.integers(0, N_SAMPLES - THIRD + 1))
        place(noise, canonical, start)
        return noise, interval_mask(start, start + THIRD), f"[{start},{start + THIRD})"
    if geometry == "left_edge_third":
        place(noise, canonical, 0)
        return noise, interval_mask(0, THIRD), f"[0,{THIRD})"
    if geometry == "right_edge_third":
        start = N_SAMPLES - THIRD
        place(noise, canonical, start)
        return noise, interval_mask(start, N_SAMPLES), f"[{start},{N_SAMPLES})"
    if geometry == "three_bursts_third":
        lengths = [THIRD // 3 + 1, THIRD // 3 + 1, THIRD - 2 * (THIRD // 3 + 1)]
        centers = [N_SAMPLES // 6, N_SAMPLES // 2, 5 * N_SAMPLES // 6]
        mask = np.zeros(N_SAMPLES, dtype=bool)
        source_start = 0
        intervals = []
        for length, center in zip(lengths, centers):
            start = center - length // 2
            stop = start + length
            place(noise, canonical[source_start : source_start + length], start)
            mask[start:stop] = True
            intervals.append(f"[{start},{stop})")
            source_start += length
        if mask.sum() != THIRD or source_start != THIRD:
            raise RuntimeError("Three-burst geometry did not preserve the one-third support.")
        return noise, mask, ";".join(intervals)
    if geometry == "center_sixth":
        length = N_SAMPLES // 6
        source = center_crop(canonical, length)
        start = (N_SAMPLES - length) // 2
        place(noise, source, start)
        return noise, interval_mask(start, start + length), f"[{start},{start + length})"
    if geometry == "center_two_thirds":
        length = 2 * N_SAMPLES // 3
        source = center_crop(emg, length)
        start = (N_SAMPLES - length) // 2
        place(noise, source, start)
        return noise, interval_mask(start, start + length), f"[{start},{start + length})"
    if geometry == "full_epoch":
        noise[:] = emg
        return noise, np.ones(N_SAMPLES, dtype=bool), f"[0,{N_SAMPLES})"
    raise KeyError(f"Unknown geometry: {geometry}")


def place(destination: np.ndarray, source: np.ndarray, start: int) -> None:
    stop = start + len(source)
    if start < 0 or stop > len(destination):
        raise ValueError(f"Placement [{start}, {stop}) exceeds the destination.")
    destination[start:stop] = source


def center_crop(values: np.ndarray, length: int) -> np.ndarray:
    start = (len(values) - length) // 2
    return np.asarray(values[start : start + length], dtype=float)


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


def audit_submitted_center(
    prepared_file: Path, submitted_file: Path, record_ids: np.ndarray
) -> float | None:
    if not submitted_file.exists():
        print(f"Submitted centred input not found; exact reproduction audit skipped: {submitted_file}")
        return None
    maximum = 0.0
    with h5py.File(prepared_file, "r") as prepared, h5py.File(submitted_file, "r") as submitted:
        for record_id in record_ids:
            new = prepared[f"center_third__emg_{record_id}"]
            old = submitted[f"emg_{record_id}"]
            for dataset in ("eeg_signal", "eeg_reference", "artifacts"):
                maximum = max(maximum, float(np.max(np.abs(new[dataset][()] - old[dataset][()]))))
    if maximum > 1e-10:
        raise RuntimeError(f"Centred-condition reproduction failed; maximum error {maximum}")
    return maximum


if __name__ == "__main__":
    main()
