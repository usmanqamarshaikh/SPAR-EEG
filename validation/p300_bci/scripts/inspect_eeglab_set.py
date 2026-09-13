from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "data" / "Won2022_BIDS"
DEFAULT_SUBJECT = "sub-001"
DEFAULT_TASK = "P300trainrun1"
DEFAULT_RUN = 6
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "smoke_tests" / "eeglab_set_inspection.json"


@dataclass(frozen=True)
class SetInspection:
    set_path: str
    fdt_path: str
    eeg_json_path: str
    channels_tsv_path: str
    nbchan: int
    pnts: int
    trials: int
    srate: float
    recording_duration_sec: float | None
    expected_fdt_bytes: int
    observed_fdt_bytes: int
    channel_count_tsv: int
    first_channels: list[str]
    event_count: int
    numeric_event_types: dict[str, int]
    label_event_count_type_1_or_2: int
    event_sequence_present: bool
    event_sequence_shape: list[int]
    event_sequence_unique_nonzero: list[int]
    event_sequence_nonzero_count: int
    fdt_probe_shape_ch_time: list[int]
    fdt_probe_mean_first_ch: float
    fdt_probe_std_first_ch: float
    pass_status: bool


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


def numeric_event_type(event: Any) -> int | None:
    raw = get_struct_field(event, "type")
    raw = matlab_scalar(raw)
    try:
        return int(raw)
    except (TypeError, ValueError):
        try:
            return int(float(str(raw)))
        except (TypeError, ValueError):
            return None


def channel_labels(chanlocs: Any) -> list[str]:
    labels: list[str] = []
    for ch in iter_matlab_struct_array(chanlocs):
        label = matlab_string(get_struct_field(ch, "labels", ""))
        labels.append(label)
    return labels


def inspect_set(dataset_root: Path, subject: str, task: str, run: int) -> SetInspection:
    eeg_dir = dataset_root / subject / "eeg"
    stem = f"{subject}_task-{task}_run-{run}"
    set_path = eeg_dir / f"{stem}_eeg.set"
    fdt_path = eeg_dir / f"{stem}_eeg.fdt"
    eeg_json_path = eeg_dir / f"{stem}_eeg.json"
    channels_tsv_path = eeg_dir / f"{stem}_channels.tsv"

    if not set_path.is_file():
        raise FileNotFoundError(set_path)
    if not fdt_path.is_file():
        raise FileNotFoundError(fdt_path)

    mat = {
        key: value
        for key, value in loadmat(set_path, squeeze_me=True, struct_as_record=False).items()
        if not key.startswith("__")
    }
    # Some EEGLAB exports store a single EEG struct, while this BIDS conversion
    # stores EEGLAB fields directly as top-level MATLAB variables.
    eeg = mat["EEG"] if "EEG" in mat else mat

    nbchan = int(matlab_scalar(get_struct_field(eeg, "nbchan")))
    pnts = int(matlab_scalar(get_struct_field(eeg, "pnts")))
    trials = int(matlab_scalar(get_struct_field(eeg, "trials")))
    srate = float(matlab_scalar(get_struct_field(eeg, "srate")))
    expected_fdt_bytes = nbchan * pnts * max(trials, 1) * 4
    observed_fdt_bytes = fdt_path.stat().st_size

    channels = pd.read_csv(channels_tsv_path, sep="\t")
    labels = channel_labels(get_struct_field(eeg, "chanlocs"))

    events = iter_matlab_struct_array(get_struct_field(eeg, "event", []))
    event_type_counts: dict[str, int] = {}
    label_count = 0
    for event in events:
        etype = numeric_event_type(event)
        if etype is None:
            key = "non_numeric"
        else:
            key = str(etype)
            if etype in {1, 2}:
                label_count += 1
        event_type_counts[key] = event_type_counts.get(key, 0) + 1

    seq = get_struct_field(eeg, "event_sequence")
    seq_arr = np.asarray(seq) if seq is not None else np.asarray([])
    seq_present = seq_arr.size > 0
    seq_nonzero = sorted(int(v) for v in np.unique(seq_arr) if int(v) != 0)

    with eeg_json_path.open("r", encoding="utf-8") as f:
        eeg_json = json.load(f)
    duration = eeg_json.get("RecordingDuration")

    # Match EEGLAB pop_loadset/floatread output exactly:
    # MATLAB fread(fid, [nbchan pnts], 'float32=>double').
    probe_n = min(pnts, int(round(srate * 5)))
    raw = np.memmap(fdt_path, dtype="<f4", mode="r", shape=(nbchan, pnts), order="F")
    first_channel_probe = np.asarray(raw[0, :probe_n], dtype=np.float64)

    ok = (
        nbchan == 32
        and trials == 1
        and abs(srate - 512.0) < 1e-6
        and observed_fdt_bytes == expected_fdt_bytes
        and len(channels) == nbchan
        and len(labels) == nbchan
        and label_count > 0
        and seq_present
        and set(seq_nonzero).issubset(set(range(1, 13)))
    )

    return SetInspection(
        set_path=str(set_path),
        fdt_path=str(fdt_path),
        eeg_json_path=str(eeg_json_path),
        channels_tsv_path=str(channels_tsv_path),
        nbchan=nbchan,
        pnts=pnts,
        trials=trials,
        srate=srate,
        recording_duration_sec=float(duration) if duration is not None else None,
        expected_fdt_bytes=expected_fdt_bytes,
        observed_fdt_bytes=observed_fdt_bytes,
        channel_count_tsv=len(channels),
        first_channels=labels[:8],
        event_count=len(events),
        numeric_event_types=event_type_counts,
        label_event_count_type_1_or_2=label_count,
        event_sequence_present=seq_present,
        event_sequence_shape=list(seq_arr.shape),
        event_sequence_unique_nonzero=seq_nonzero,
        event_sequence_nonzero_count=int(np.count_nonzero(seq_arr)),
        fdt_probe_shape_ch_time=[nbchan, pnts],
        fdt_probe_mean_first_ch=float(np.mean(first_channel_probe)),
        fdt_probe_std_first_ch=float(np.std(first_channel_probe, ddof=0)),
        pass_status=ok,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect one EEGLAB .set/.fdt recording.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--run", type=int, default=DEFAULT_RUN)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    inspection = inspect_set(args.dataset_root, args.subject, args.task, args.run)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(asdict(inspection), f, indent=2)

    print(f"SET: {inspection.set_path}")
    print(f"Channels x points: {inspection.nbchan} x {inspection.pnts}")
    print(f"Sampling rate: {inspection.srate:g} Hz")
    print(f"Events: {inspection.event_count}")
    print(f"Event types: {inspection.numeric_event_types}")
    print(f"Label events type 1/2: {inspection.label_event_count_type_1_or_2}")
    print(f"event_sequence unique nonzero: {inspection.event_sequence_unique_nonzero}")
    print(f"FDT bytes: observed={inspection.observed_fdt_bytes}, expected={inspection.expected_fdt_bytes}")
    print(f"Output JSON: {args.out_json}")

    if not inspection.pass_status:
        print("\nFAIL: recording did not satisfy expected metadata checks.")
        return 1

    print("\nPASS: EEGLAB .set/.fdt recording is readable and internally consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
