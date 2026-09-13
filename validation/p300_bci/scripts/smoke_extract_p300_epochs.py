from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "data" / "Won2022_BIDS"
DEFAULT_SUBJECT = "sub-001"
DEFAULT_TASK = "P300trainrun1"
DEFAULT_RUN = 6
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "smoke_tests" / "p300_epoch_smoke.json"


@dataclass(frozen=True)
class EpochSmoke:
    set_path: str
    nbchan: int
    pnts: int
    srate: float
    window_sec: list[float]
    baseline_sec: list[float]
    epoch_samples: int
    total_events: int
    valid_epochs: int
    skipped_epochs: int
    target_epochs: int
    nontarget_epochs: int
    event_target_matches_event_type: int
    event_target_mismatches_event_type: int
    p300_channel: str
    p300_channel_index: int
    p300_window_sec: list[float]
    target_mean_uv_250_500ms: float
    nontarget_mean_uv_250_500ms: float
    target_minus_nontarget_peak_uv_250_500ms: float
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


def channel_labels(chanlocs: Any) -> list[str]:
    labels: list[str] = []
    for ch in iter_matlab_struct_array(chanlocs):
        labels.append(matlab_string(get_struct_field(ch, "labels", "")))
    return labels


def load_set(set_path: Path) -> dict[str, Any]:
    mat = loadmat(set_path, squeeze_me=True, struct_as_record=False)
    return {key: value for key, value in mat.items() if not key.startswith("__")}


def event_type(event: Any) -> int | None:
    value = matlab_scalar(get_struct_field(event, "type"))
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return None


def inspect_epochs(
    dataset_root: Path,
    subject: str,
    task: str,
    run: int,
    window_sec: tuple[float, float] = (-0.2, 0.6),
    baseline_sec: tuple[float, float] = (-0.2, 0.0),
) -> EpochSmoke:
    eeg_dir = dataset_root / subject / "eeg"
    stem = f"{subject}_task-{task}_run-{run}"
    set_path = eeg_dir / f"{stem}_eeg.set"
    fdt_path = eeg_dir / f"{stem}_eeg.fdt"

    if not set_path.is_file():
        raise FileNotFoundError(set_path)
    if not fdt_path.is_file():
        raise FileNotFoundError(fdt_path)

    mat = load_set(set_path)
    eeg = mat["EEG"] if "EEG" in mat else mat
    nbchan = int(matlab_scalar(get_struct_field(eeg, "nbchan")))
    pnts = int(matlab_scalar(get_struct_field(eeg, "pnts")))
    srate = float(matlab_scalar(get_struct_field(eeg, "srate")))
    labels = channel_labels(get_struct_field(eeg, "chanlocs"))
    label_lookup = {label.upper(): idx for idx, label in enumerate(labels)}
    p300_channel = "PZ" if "PZ" in label_lookup else "CZ"
    p300_idx = label_lookup[p300_channel]

    # Match EEGLAB pop_loadset/floatread output exactly:
    # MATLAB fread(fid, [nbchan pnts], 'float32=>double').
    data = np.memmap(fdt_path, dtype="<f4", mode="r", shape=(nbchan, pnts), order="F")

    epoch_n = int(round((window_sec[1] - window_sec[0]) * srate))
    start_offset = int(round(window_sec[0] * srate))
    time = window_sec[0] + np.arange(epoch_n) / srate
    baseline_mask = (time >= baseline_sec[0]) & (time < baseline_sec[1])
    p300_mask = (time >= 0.25) & (time <= 0.5)

    sums = {1: np.zeros((nbchan, epoch_n), dtype=np.float64), 2: np.zeros((nbchan, epoch_n), dtype=np.float64)}
    counts = {1: 0, 2: 0}
    skipped = 0
    matches = 0
    mismatches = 0

    event_sequence = np.asarray(get_struct_field(eeg, "event_sequence", []), dtype=np.float64).reshape(-1)
    event_target = np.asarray(get_struct_field(eeg, "event_target", []), dtype=np.float64).reshape(-1)
    events = iter_matlab_struct_array(get_struct_field(eeg, "event", []))

    for event in events:
        etype = event_type(event)
        if etype not in {1, 2}:
            continue
        latency = float(matlab_scalar(get_struct_field(event, "latency")))
        center = int(round(latency)) - 1
        start = center + start_offset
        stop = start + epoch_n
        if start < 0 or stop > pnts:
            skipped += 1
            continue
        if event_target.size == pnts:
            target_value = int(event_target[center])
            if target_value == etype:
                matches += 1
            else:
                mismatches += 1
        if event_sequence.size == pnts and int(event_sequence[center]) == 0:
            skipped += 1
            continue

        epoch = np.asarray(data[:, start:stop], dtype=np.float64)
        epoch = epoch - epoch[:, baseline_mask].mean(axis=1, keepdims=True)
        sums[etype] += epoch
        counts[etype] += 1

    target_avg = sums[1] / max(counts[1], 1)
    nontarget_avg = sums[2] / max(counts[2], 1)
    target_p300 = target_avg[p300_idx, p300_mask]
    nontarget_p300 = nontarget_avg[p300_idx, p300_mask]
    diff_p300 = target_p300 - nontarget_p300

    valid = counts[1] + counts[2]
    pass_status = (
        nbchan == 32
        and abs(srate - 512.0) < 1e-6
        and valid > 0
        and counts[1] > 0
        and counts[2] > 0
        and mismatches == 0
    )

    return EpochSmoke(
        set_path=str(set_path),
        nbchan=nbchan,
        pnts=pnts,
        srate=srate,
        window_sec=list(window_sec),
        baseline_sec=list(baseline_sec),
        epoch_samples=epoch_n,
        total_events=len(events),
        valid_epochs=valid,
        skipped_epochs=skipped,
        target_epochs=counts[1],
        nontarget_epochs=counts[2],
        event_target_matches_event_type=matches,
        event_target_mismatches_event_type=mismatches,
        p300_channel=labels[p300_idx],
        p300_channel_index=p300_idx,
        p300_window_sec=[0.25, 0.5],
        target_mean_uv_250_500ms=float(np.mean(target_p300)),
        nontarget_mean_uv_250_500ms=float(np.mean(nontarget_p300)),
        target_minus_nontarget_peak_uv_250_500ms=float(np.max(diff_p300)),
        pass_status=pass_status,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test P300 epoch extraction from one BIDS EEGLAB run.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--run", type=int, default=DEFAULT_RUN)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    smoke = inspect_epochs(args.dataset_root, args.subject, args.task, args.run)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(asdict(smoke), f, indent=2)

    print(f"SET: {smoke.set_path}")
    print(f"Epochs: valid={smoke.valid_epochs}, skipped={smoke.skipped_epochs}")
    print(f"Class counts: target={smoke.target_epochs}, nontarget={smoke.nontarget_epochs}")
    print(
        "Event target agreement: "
        f"matches={smoke.event_target_matches_event_type}, "
        f"mismatches={smoke.event_target_mismatches_event_type}"
    )
    print(
        f"{smoke.p300_channel} 250-500 ms target/non-target mean: "
        f"{smoke.target_mean_uv_250_500ms:.3f} / {smoke.nontarget_mean_uv_250_500ms:.3f} uV"
    )
    print(
        "Peak target-minus-nontarget in 250-500 ms: "
        f"{smoke.target_minus_nontarget_peak_uv_250_500ms:.3f} uV"
    )
    print(f"Output JSON: {args.out_json}")

    if not smoke.pass_status:
        print("\nFAIL: epoch extraction smoke did not satisfy expected checks.")
        return 1

    print("\nPASS: P300 epochs can be extracted and labels align with event_target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
