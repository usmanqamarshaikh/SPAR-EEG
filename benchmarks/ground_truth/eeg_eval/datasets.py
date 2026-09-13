from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator

import h5py
import numpy as np


@dataclass
class EEGRecord:
    file_path: Path
    dataset_name: str
    record_name: str
    signal: np.ndarray
    reference: np.ndarray
    artifact_mask: np.ndarray
    fs: float

    @property
    def n_channels(self) -> int:
        return int(self.signal.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.signal.shape[1])


def ensure_2d_rows(x: np.ndarray) -> np.ndarray:
    """Return data as channels x samples."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")


def normalize_mask(mask: np.ndarray, n_samples: int | None = None) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.dtype == np.bool_:
        out = arr.reshape(-1)
    elif np.issubdtype(arr.dtype, np.number):
        out = (arr.reshape(-1) != 0)
    else:
        out = np.array([str(v).strip().lower() in {"1", "true", "t", "yes"} for v in arr.reshape(-1)])

    if n_samples is not None:
        if out.size > n_samples:
            out = out[:n_samples]
        elif out.size < n_samples:
            out = np.pad(out, (0, n_samples - out.size), constant_values=False)
    return out.astype(bool)


def mask_to_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = normalize_mask(mask)
    if not np.any(mask):
        return []
    edges = np.flatnonzero(np.diff(np.pad(mask.astype(int), (1, 1))))
    return [(int(i), int(j)) for i, j in edges.reshape(-1, 2)]


def intervals_to_mask(intervals: list[tuple[int, int]], size: int) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    for start, end in intervals:
        start = max(0, int(start))
        end = min(size, int(end))
        if end > start:
            mask[start:end] = True
    return mask


def parse_dataset_info(file_name: str) -> dict[str, object]:
    lower = file_name.lower()
    info: dict[str, object] = {
        "benchmark_family": "",
        "noise_type": "",
        "nominal_snr_db": np.nan,
    }

    if lower.startswith("01_physiobank"):
        info["benchmark_family"] = "physiobank"
        info["noise_type"] = "motion"
    elif lower.startswith("02_semisimulated_eog"):
        info["benchmark_family"] = "semisimulated_eog"
        info["noise_type"] = "eog"
    elif "03_denoise-net_eog+emg" in lower:
        info["benchmark_family"] = "denoise-net"
        info["noise_type"] = "eog+emg"
    elif "03_denoise-net_emg" in lower:
        info["benchmark_family"] = "denoise-net"
        info["noise_type"] = "emg"
    elif "03_denoise-net_eog" in lower:
        info["benchmark_family"] = "denoise-net"
        info["noise_type"] = "eog"

    match = re.search(r"_(-?\d+(?:\.\d+)?)db", lower)
    if match:
        info["nominal_snr_db"] = float(match.group(1))
    return info


def iter_h5_records(path: Path, max_records: int | None = None) -> Iterator[EEGRecord]:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        dataset_name = decode_attr(h5.attrs.get("name", path.stem))
        keys = list(h5.keys())
        if max_records is not None:
            keys = keys[:max_records]

        for key in keys:
            group = h5[key]
            signal = ensure_2d_rows(group["eeg_signal"][()])
            reference = ensure_2d_rows(group["eeg_reference"][()])

            n_samples = min(signal.shape[1], reference.shape[1])
            signal = signal[:, :n_samples]
            reference = reference[:, :n_samples]

            mask = normalize_mask(group["artifacts"][()], n_samples)
            fs = float(group.attrs.get("freq", h5.attrs.get("freq", 256.0)))

            if signal.shape != reference.shape:
                raise ValueError(
                    f"{path.name}/{key}: signal/reference shape mismatch "
                    f"{signal.shape} vs {reference.shape}"
                )

            yield EEGRecord(
                file_path=path,
                dataset_name=str(dataset_name),
                record_name=str(key),
                signal=signal,
                reference=reference,
                artifact_mask=mask,
                fs=fs,
            )


def inspect_h5_file(path: Path, max_records: int = 3) -> dict[str, object]:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        keys = list(h5.keys())
        rows = []
        for key in keys[:max_records]:
            group = h5[key]
            signal = group["eeg_signal"]
            reference = group["eeg_reference"]
            artifacts = group["artifacts"]
            mask_sum = int(np.asarray(artifacts[()]).astype(bool).sum())
            rows.append(
                {
                    "record": key,
                    "signal_shape": tuple(signal.shape),
                    "reference_shape": tuple(reference.shape),
                    "artifact_shape": tuple(artifacts.shape),
                    "fs": float(group.attrs.get("freq", h5.attrs.get("freq", 256.0))),
                    "artifact_samples": mask_sum,
                }
            )
        return {
            "file": path.name,
            "dataset_name": decode_attr(h5.attrs.get("name", path.stem)),
            "n_records": len(keys),
            "examples": rows,
        }


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value
