from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "preprocessed_fp12"
    / "manifests"
    / "sub-001_p300_preprocessing_manifest.csv"
)
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "preprocessed_fp12" / "manifests" / "sub-001_p300_preprocessing_audit.csv"


def scalar_attr(attrs, key: str, default=np.nan) -> float:
    if key not in attrs:
        return default
    arr = np.asarray(attrs[key])
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def audit_branch_file(path: Path) -> dict[str, object]:
    out: dict[str, object] = {
        "branches_exists": path.is_file(),
        "pass_status": False,
        "message": "",
    }
    if not path.is_file():
        out["message"] = "missing branches file"
        return out

    try:
        with h5py.File(path, "r") as h5:
            required = ["baseline/data", "eog/data", "emg_eog/data", "full/data"]
            missing = [name for name in required if name not in h5]
            if missing:
                out["message"] = f"missing datasets: {missing}"
                return out

            base = h5["baseline/data"][:]
            out["shape"] = f"{base.shape[0]}x{base.shape[1]}"
            out["srate"] = scalar_attr(h5.attrs, "srate")
            out["wide_low_hz"] = scalar_attr(h5.attrs, "wide_filter_low_hz")
            out["wide_high_hz"] = scalar_attr(h5.attrs, "wide_filter_high_hz")
            out["baseline_std"] = float(np.std(base))
            out["baseline_mean"] = float(np.mean(base))

            ok = base.ndim == 2 and base.shape[0] == 2 and np.isfinite(base).all()
            for branch in ["eog", "emg_eog", "full"]:
                data = h5[f"{branch}/data"][:]
                out[f"{branch}_std"] = float(np.std(data))
                out[f"{branch}_mean"] = float(np.mean(data))
                out[f"{branch}_rms_delta"] = float(np.sqrt(np.mean((data - base) ** 2)))
                out[f"{branch}_pass_order"] = h5[branch].attrs.get("pass_order", "")
                ok = ok and data.shape == base.shape and np.isfinite(data).all()

            out["pass_status"] = bool(ok)
            out["message"] = "ok" if ok else "shape or finite-value check failed"
            return out
    except Exception as exc:  # noqa: BLE001
        out["message"] = repr(exc)
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit preprocessed P300 FP1/FP2 branch files for one subject.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    with args.manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            branch_path = Path(row["branches_h5"])
            audit = audit_branch_file(branch_path)
            rows.append({**row, **audit})

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(1 for row in rows if row.get("pass_status") is True)
    print(f"Audited files: {len(rows)}")
    print(f"Passed: {ok_count}/{len(rows)}")
    for row in rows:
        print(
            f"  run-{int(row['run']):02d} {row['task']:14s} "
            f"shape={row.get('shape')} "
            f"baseline_std={float(row.get('baseline_std', np.nan)):.3f} "
            f"full_rms_delta={float(row.get('full_rms_delta', np.nan)):.3f} "
            f"status={row.get('message')}"
        )
    print(f"Audit CSV: {args.out_csv}")

    return 0 if ok_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
