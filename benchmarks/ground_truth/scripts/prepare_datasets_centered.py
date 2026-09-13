from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import scipy.ndimage as ndi
import scipy.signal as ss


FILTER_LOW = 0.1
FILTER_HIGH = 100.0
DEFAULT_SNR_LEVELS = [
    *range(-20, 0),
    *range(0, 6),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare benchmark H5 files from raw sources. This follows the "
            "reference WQN preparation script, but centers DenoiseNet artifacts "
            "in the middle third of each 2-s epoch."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("data_centered"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["physiobank", "semisimulated-eog", "denoisenet", "all"],
        default=["all"],
    )
    parser.add_argument(
        "--snr-levels",
        nargs="+",
        type=float,
        default=DEFAULT_SNR_LEVELS,
        help="DenoiseNet SNR levels in dB. Default is the integer grid -20..5.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.datasets)
    if "all" in selected:
        selected = {"physiobank", "semisimulated-eog", "denoisenet"}

    if "physiobank" in selected:
        prepare_physiobank(args.raw_dir, args.out_dir, args.overwrite)
    if "semisimulated-eog" in selected:
        prepare_semisimulated_eog(args.raw_dir, args.out_dir, args.overwrite)
    if "denoisenet" in selected:
        prepare_denoisenet(
            args.raw_dir,
            args.out_dir,
            args.snr_levels,
            args.overwrite,
            max_records=args.max_records,
        )

    print(f"Done. Prepared datasets are in: {args.out_dir}")


def prepare_physiobank(raw_dir: Path, out_dir: Path, overwrite: bool) -> None:
    try:
        import wfdb
    except ImportError as exc:
        raise SystemExit(
            "wfdb is required to regenerate the PhysioBank H5 file. "
            "Install it or run with --datasets denoisenet semisimulated-eog."
        ) from exc

    out_path = out_dir / "01_physiobank.h5"
    if out_path.exists() and not overwrite:
        print(f"Skipping existing {out_path}")
        return

    excluded = {"eeg_21"}
    partially_excluded = {"eeg_10": slice(0, 75000), "eeg_20": slice(25000, None)}
    dataset_path = raw_dir / "physiobank-motion-artifacts"

    with h5py.File(out_path, "w") as h5:
        h5.attrs["name"] = "Physiobank Motion Artifacts"
        h5.attrs["author"] = "Kevin Sweeney et al."
        h5.attrs["preparation"] = "reference_wqn_compatible"

        for path in sorted(dataset_path.glob("*.hea"), key=lambda p: int(p.stem[4:])):
            record_name = path.stem
            if record_name in excluded:
                continue

            select = partially_excluded.get(record_name, slice(None, None))
            record_path = str(path.with_name(record_name))
            record = wfdb.rdrecord(record_path)

            fs = record.fs / 8
            eeg = record.p_signal[:, :2]

            reference = filter_bandpass(eeg[:, 0], FILTER_LOW, FILTER_HIGH, record.fs, 2)[
                ::8
            ].reshape(1, -1)
            signal = filter_bandpass(eeg[:, 1], FILTER_LOW, FILTER_HIGH, record.fs, 2)[
                ::8
            ].reshape(1, -1)

            reference = reference[:, select]
            signal = signal[:, select]

            external_mask = record.p_signal[::8, 9] > 0.5
            external_mask = external_mask[select]

            dist = np.sqrt(np.abs((signal - reference)[0] ** 2))
            dist = ndi.gaussian_filter1d(dist, 2 * fs)
            high = np.quantile(dist, 0.80)

            trig = dist > high
            trig = ndi.binary_closing(trig, np.ones(int(5 * fs)))
            trig = ndi.binary_opening(trig, np.ones(int(5 * fs)))
            trig = ndi.binary_dilation(trig, np.ones(int(3 * fs)))
            trig = trig & ~external_mask

            intervals = mask_to_intervals(trig)[:4]
            artifacts = intervals_to_mask(intervals, trig.size)

            group = h5.create_group(record_name)
            group["eeg_signal"] = signal
            group["eeg_reference"] = reference
            group["artifacts"] = artifacts
            group.attrs["freq"] = fs
            group.attrs["filtered"] = f"BANDPASS {FILTER_LOW}-{FILTER_HIGH} Hz"

    print(f"Wrote {out_path}")


def prepare_semisimulated_eog(raw_dir: Path, out_dir: Path, overwrite: bool) -> None:
    out_path = out_dir / "02_semisimulated_eog.h5"
    if out_path.exists() and not overwrite:
        print(f"Skipping existing {out_path}")
        return

    excluded = {36}
    data_signals = scipy.io.loadmat(raw_dir / "eog-data" / "Pure_Data.mat")
    data_artifact = scipy.io.loadmat(raw_dir / "eog-data" / "Contaminated_Data.mat")
    fs = 200

    with h5py.File(out_path, "w") as h5:
        h5.attrs["name"] = "Semi-simulated EOG"
        h5.attrs["author"] = "Manousos A. Klados and Panagiotis D. Bamidis"
        h5.attrs["preparation"] = "reference_wqn_compatible_middle_third"

        for n in range(1, 55):
            if n in excluded:
                continue
            reference = data_signals[f"sim{n}_resampled"]
            artifact = data_artifact[f"sim{n}_con"] - reference

            keep_len = reference.shape[1] // 3
            start = keep_len
            end = 2 * keep_len
            window = ss.windows.general_gaussian(keep_len, 6, keep_len // 2)

            artifact[:, :start] = 0
            artifact[:, start:end] *= window
            artifact[:, end:] = 0

            signal = reference + artifact
            artifacts = np.zeros(signal.shape[1], dtype=bool)
            artifacts[start:end] = True

            noise = signal - reference
            snr = calculate_snr(reference[:, start:end], noise[:, start:end])
            if snr >= 10:
                print(f"Skipping record {n}: SNR too high ({snr:.2f} dB).")
                continue

            group = h5.create_group(f"sim{n}")
            group["eeg_signal"] = signal
            group["eeg_reference"] = reference
            group["artifacts"] = artifacts
            group.attrs["freq"] = fs
            group.attrs["filtered"] = ""
            group.attrs["artifact_layout"] = "middle_third"
            group.attrs["artifact_start_sample"] = start
            group.attrs["artifact_end_sample"] = end

    print(f"Wrote {out_path}")


def prepare_denoisenet(
    raw_dir: Path,
    out_dir: Path,
    snr_levels: list[float],
    overwrite: bool,
    max_records: int | None = None,
) -> None:
    fs = 256
    eeg = np.load(raw_dir / "eeg-denoise-net" / "EEG_all_epochs.npy")
    eog = np.load(raw_dir / "eeg-denoise-net" / "EOG_all_epochs.npy")
    emg = np.load(raw_dir / "eeg-denoise-net" / "EMG_all_epochs.npy")

    n_samples = 2 * fs
    keep_len = n_samples // 3
    start = keep_len
    end = 2 * keep_len

    artifacts = np.zeros(n_samples, dtype=bool)
    artifacts[start:end] = True

    available_records = min(eeg.shape[0], eog.shape[0], emg.shape[0])
    n_records = available_records if max_records is None else min(max_records, available_records)
    print(
        "DenoiseNet centered-middle-third layout: "
        f"N={n_samples}, artifact=[{start}, {end}), clean_left={start}, "
        f"clean_right={n_samples - end}, records={n_records}"
    )

    for snr in snr_levels:
        snr = float(snr)
        prepare_denoisenet_noise_type(
            out_dir,
            "eog",
            snr,
            eeg,
            eog,
            None,
            artifacts,
            start,
            end,
            fs,
            n_records,
            overwrite,
        )
        prepare_denoisenet_noise_type(
            out_dir,
            "emg",
            snr,
            eeg,
            emg,
            None,
            artifacts,
            start,
            end,
            fs,
            n_records,
            overwrite,
        )
        prepare_denoisenet_noise_type(
            out_dir,
            "eog+emg",
            snr,
            eeg,
            eog,
            emg,
            artifacts,
            start,
            end,
            fs,
            n_records,
            overwrite,
        )


def prepare_denoisenet_noise_type(
    out_dir: Path,
    noise_type: str,
    snr: float,
    eeg: np.ndarray,
    primary_artifact: np.ndarray,
    secondary_artifact: np.ndarray | None,
    artifacts: np.ndarray,
    start: int,
    end: int,
    fs: int,
    n_records: int,
    overwrite: bool,
) -> None:
    out_path = out_dir / f"03_denoise-net_{noise_type}_{format_snr(snr)}dB.h5"
    if out_path.exists() and not overwrite:
        print(f"Skipping existing {out_path}")
        return

    with h5py.File(out_path, "w") as h5:
        h5.attrs["name"] = f"Denoise-Net {noise_type.upper()} ({format_snr(snr)} dB)"
        h5.attrs["author"] = ""
        h5.attrs["preparation"] = "centered_middle_third_artifact"
        h5.attrs["artifact_layout"] = "middle_third"
        h5.attrs["artifact_start_sample"] = start
        h5.attrs["artifact_end_sample"] = end
        h5.attrs["nominal_snr_db"] = snr

        prefix = noise_type
        for n in range(n_records):
            clean = np.asarray(eeg[n], dtype=float).copy()
            clean /= np.std(clean[artifacts])

            reference = clean.copy()
            noise = np.zeros_like(reference)

            if secondary_artifact is None:
                noise[start:end] = primary_artifact[n][start:end]
            else:
                a = primary_artifact[n][start:end]
                b = secondary_artifact[n][start:end]
                noise[start:end] = a / safe_std(a) + b / safe_std(b)

            noise /= safe_std(noise[artifacts])
            noise *= np.sqrt(10 ** (-0.1 * snr))

            signal = reference + noise
            measured_snr = calculate_snr(reference[artifacts], noise[artifacts])
            if not np.isfinite(measured_snr) or abs(measured_snr - snr) >= 1e-3:
                raise RuntimeError(
                    f"{out_path.name}/{n}: measured SNR {measured_snr:.6g} "
                    f"does not match nominal {snr:.6g}"
                )

            group = h5.create_group(f"{prefix}_{n}")
            group["eeg_signal"] = [signal]
            group["eeg_reference"] = [reference]
            group["artifacts"] = artifacts
            group.attrs["freq"] = fs
            group.attrs["filtered"] = ""
            group.attrs["artifact_layout"] = "middle_third"
            group.attrs["artifact_start_sample"] = start
            group.attrs["artifact_end_sample"] = end
            group.attrs["nominal_snr_db"] = snr
            group.attrs["measured_snr_db"] = measured_snr

    print(f"Wrote {out_path}")


def filter_bandpass(
    x: np.ndarray,
    low: float,
    high: float,
    fs: float,
    order: int,
) -> np.ndarray:
    sos = ss.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return ss.sosfiltfilt(sos, x)


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    sig_var = float(np.var(np.asarray(signal, dtype=float)))
    noise_var = float(np.var(np.asarray(noise, dtype=float)))
    if sig_var <= np.finfo(float).eps or noise_var <= np.finfo(float).eps:
        return np.nan
    return float(10.0 * np.log10(sig_var / noise_var))


def mask_to_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(mask):
        return []
    edges = np.flatnonzero(np.diff(np.pad(mask.astype(int), (1, 1))))
    return [(int(i), int(j)) for i, j in edges.reshape(-1, 2)]


def intervals_to_mask(intervals: list[tuple[int, int]], size: int) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    for start, end in intervals:
        mask[max(0, start) : min(size, end)] = True
    return mask


def safe_std(x: np.ndarray) -> float:
    value = float(np.std(np.asarray(x, dtype=float)))
    if value <= np.finfo(float).eps:
        return 1.0
    return value


def format_snr(snr: float) -> str:
    if float(snr).is_integer():
        return str(int(snr))
    return f"{snr:g}"


if __name__ == "__main__":
    main()
