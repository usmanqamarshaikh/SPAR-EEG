from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import h5py
import mne
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bci_p300_denoise.eeglab_io import (
    channel_indices,
    get_struct_field,
    load_eeglab_run,
    memmap_fdt,
    matlab_string,
)


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "data" / "Won2022_BIDS"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "preprocessed_fp12"
MATLAB_WORKER = PROJECT_ROOT / "matlab" / "run_bci_fp12_denoise_branches.m"


def matlab_quote(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def output_stem(subject: str, task: str, run: int, max_seconds: float | None) -> str:
    stem = f"{subject}_task-{task}_run-{run}_fp12"
    if max_seconds is not None:
        stem += f"_first{max_seconds:g}s"
    return stem


def write_wide_h5(
    out_path: Path,
    data: np.ndarray,
    run_info,
    channels: list[str],
    filter_low_hz: float,
    filter_high_hz: float,
    notch_freqs_hz: list[float],
    n_samples_original: int,
) -> None:
    event_sequence = np.asarray(get_struct_field(run_info.eeg, "event_sequence", []), dtype=np.int16).reshape(-1)
    event_target = np.asarray(get_struct_field(run_info.eeg, "event_target", []), dtype=np.int16).reshape(-1)
    n_samples = data.shape[1]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "w") as h5:
        h5.attrs["subject"] = run_info.subject
        h5.attrs["task"] = run_info.task
        h5.attrs["run"] = run_info.run
        h5.attrs["source_set"] = str(run_info.set_path)
        h5.attrs["source_fdt"] = str(run_info.fdt_path)
        h5.attrs["n_samples_original"] = int(n_samples_original)
        h5.attrs["n_samples_saved"] = int(n_samples)
        h5.attrs["reference"] = matlab_string(get_struct_field(run_info.eeg, "ref", ""))
        h5.attrs["text_to_spell"] = matlab_string(get_struct_field(run_info.eeg, "text_to_spell", ""))

        wide = h5.create_group("wide")
        wide.create_dataset("data", data=data.astype(np.float32), compression="gzip", shuffle=True)
        wide.create_dataset("srate", data=float(run_info.srate))
        wide.create_dataset("filter_low_hz", data=float(filter_low_hz))
        wide.create_dataset("filter_high_hz", data=float(filter_high_hz))
        wide.create_dataset("notch_freqs_hz", data=np.asarray(notch_freqs_hz, dtype=np.float32))
        wide.create_dataset("channels", data=np.asarray(channels, dtype=object), dtype=string_dtype)

        events = h5.create_group("events")
        if event_sequence.size == n_samples_original:
            events.create_dataset("event_sequence", data=event_sequence[:n_samples], compression="gzip", shuffle=True)
        if event_target.size == n_samples_original:
            events.create_dataset("event_target", data=event_target[:n_samples], compression="gzip", shuffle=True)


def call_matlab(input_h5: Path, output_h5: Path, matlab_exe: str) -> None:
    if not MATLAB_WORKER.is_file():
        raise FileNotFoundError(MATLAB_WORKER)
    code = (
        f"addpath({matlab_quote(MATLAB_WORKER.parent)}); "
        f"run_bci_fp12_denoise_branches({matlab_quote(input_h5)}, {matlab_quote(output_h5)});"
    )
    subprocess.run([matlab_exe, "-batch", code], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare one FP1/FP2 P300 run with shared wide conditioning and proposed branches."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--subject", default="sub-001")
    parser.add_argument("--task", default="P300trainrun1")
    parser.add_argument("--run", type=int, default=6)
    parser.add_argument("--channels", nargs="+", default=["FP1", "FP2"])
    parser.add_argument("--wide-low-hz", type=float, default=0.5)
    parser.add_argument("--wide-high-hz", type=float, default=70.0)
    parser.add_argument(
        "--notch-freqs",
        type=float,
        nargs="*",
        default=[50.0],
        help="Narrow interference frequencies removed after wide bandpass. Use no values to disable.",
    )
    parser.add_argument("--notch-widths", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument("--skip-matlab", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_info = load_eeglab_run(args.dataset_root, args.subject, args.task, args.run)
    indices = channel_indices(run_info.labels, args.channels)
    n_samples = run_info.pnts
    if args.max_seconds is not None:
        n_samples = min(n_samples, int(round(args.max_seconds * run_info.srate)))
    if n_samples <= 0:
        raise ValueError("Requested sample count is empty.")

    stem = output_stem(args.subject, args.task, args.run, args.max_seconds)
    wide_h5 = args.out_dir / f"{stem}_wide.h5"
    branches_h5 = args.out_dir / f"{stem}_branches.h5"
    if not args.overwrite and (wide_h5.exists() or (branches_h5.exists() and not args.skip_matlab)):
        raise FileExistsError("Output exists. Use --overwrite to replace existing files.")

    raw = np.asarray(memmap_fdt(run_info)[indices, :n_samples], dtype=np.float64)
    wide = mne.filter.filter_data(
        raw,
        sfreq=run_info.srate,
        l_freq=args.wide_low_hz,
        h_freq=args.wide_high_hz,
        method="fir",
        phase="zero-double",
        fir_design="firwin",
        verbose=False,
    )
    if args.notch_freqs:
        wide = mne.filter.notch_filter(
            wide,
            Fs=run_info.srate,
            freqs=np.asarray(args.notch_freqs, dtype=float),
            notch_widths=args.notch_widths,
            method="fir",
            phase="zero-double",
            fir_design="firwin",
            verbose=False,
        )

    write_wide_h5(
        wide_h5,
        data=np.asarray(wide, dtype=np.float32),
        run_info=run_info,
        channels=args.channels,
        filter_low_hz=args.wide_low_hz,
        filter_high_hz=args.wide_high_hz,
        notch_freqs_hz=list(args.notch_freqs),
        n_samples_original=run_info.pnts,
    )
    print(f"Wide-conditioned H5: {wide_h5}")
    print(f"Shape: channels={wide.shape[0]}, samples={wide.shape[1]}, fs={run_info.srate:g} Hz")
    print(f"Notch frequencies: {list(args.notch_freqs) if args.notch_freqs else 'none'}")

    if args.skip_matlab:
        print("Skipped MATLAB denoising branches.")
        return 0

    call_matlab(wide_h5, branches_h5, args.matlab)
    print(f"Denoising branches H5: {branches_h5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
