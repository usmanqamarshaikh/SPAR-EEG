from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class EeglabRun:
    subject: str
    task: str
    run: int
    set_path: Path
    fdt_path: Path
    eeg_json_path: Path
    channels_tsv_path: Path
    nbchan: int
    pnts: int
    trials: int
    srate: float
    labels: list[str]
    mat: dict[str, Any]
    eeg: Any


def matlab_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return value


def matlab_string(value: Any) -> str:
    value = matlab_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "S"}:
        return "".join(arr.astype(str).reshape(-1)).strip()
    return str(value)


def get_struct_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    return default


def iter_matlab_struct_array(value: Any) -> list[Any]:
    arr = np.asarray(value)
    if arr.shape == ():
        return [arr.item()]
    return [item for item in arr.reshape(-1)]


def channel_labels(chanlocs: Any) -> list[str]:
    labels: list[str] = []
    for ch in iter_matlab_struct_array(chanlocs):
        labels.append(matlab_string(get_struct_field(ch, "labels", "")))
    return labels


def load_eeglab_run(dataset_root: Path, subject: str, task: str, run: int) -> EeglabRun:
    eeg_dir = dataset_root / subject / "eeg"
    stem = f"{subject}_task-{task}_run-{run}"
    set_path = eeg_dir / f"{stem}_eeg.set"
    fdt_path = eeg_dir / f"{stem}_eeg.fdt"
    eeg_json_path = eeg_dir / f"{stem}_eeg.json"
    channels_tsv_path = eeg_dir / f"{stem}_channels.tsv"

    for path in [set_path, fdt_path, eeg_json_path, channels_tsv_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    mat = {
        key: value
        for key, value in loadmat(set_path, squeeze_me=True, struct_as_record=False).items()
        if not key.startswith("__")
    }
    eeg = mat["EEG"] if "EEG" in mat else mat
    nbchan = int(matlab_scalar(get_struct_field(eeg, "nbchan")))
    pnts = int(matlab_scalar(get_struct_field(eeg, "pnts")))
    trials = int(matlab_scalar(get_struct_field(eeg, "trials")))
    srate = float(matlab_scalar(get_struct_field(eeg, "srate")))
    labels = channel_labels(get_struct_field(eeg, "chanlocs"))

    return EeglabRun(
        subject=subject,
        task=task,
        run=run,
        set_path=set_path,
        fdt_path=fdt_path,
        eeg_json_path=eeg_json_path,
        channels_tsv_path=channels_tsv_path,
        nbchan=nbchan,
        pnts=pnts,
        trials=trials,
        srate=srate,
        labels=labels,
        mat=mat,
        eeg=eeg,
    )


def memmap_fdt(run: EeglabRun) -> np.memmap:
    if run.trials != 1:
        raise ValueError(f"Expected continuous trials=1, got {run.trials}")
    expected_bytes = run.nbchan * run.pnts * 4
    observed_bytes = run.fdt_path.stat().st_size
    if observed_bytes != expected_bytes:
        raise ValueError(
            f"Unexpected .fdt size for {run.fdt_path}: "
            f"observed {observed_bytes}, expected {expected_bytes}"
        )
    # Match EEGLAB pop_loadset/floatread output exactly:
    # MATLAB fread(fid, [nbchan pnts], 'float32=>double').
    return np.memmap(run.fdt_path, dtype="<f4", mode="r", shape=(run.nbchan, run.pnts), order="F")


def channel_indices(labels: list[str], requested: list[str]) -> list[int]:
    lookup = {label.upper(): idx for idx, label in enumerate(labels)}
    missing = [label for label in requested if label.upper() not in lookup]
    if missing:
        raise ValueError(f"Missing channels: {missing}. Available labels include {labels[:10]}")
    return [lookup[label.upper()] for label in requested]
