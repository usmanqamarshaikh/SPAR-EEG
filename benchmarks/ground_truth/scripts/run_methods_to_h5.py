from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
from tqdm import tqdm

from eeg_eval.datasets import iter_h5_records, mask_to_intervals
from eeg_eval.methods import available_methods, get_method


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Python benchmark denoisers and save restored H5 outputs."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-root", type=Path, default=Path("results/restored"))
    parser.add_argument("--pattern", default="*.h5")
    parser.add_argument("--methods", nargs="+", default=["wt_hard", "wt_soft", "wqn"])
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = sorted(args.data_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No H5 files matched: {args.data_dir / args.pattern}")

    args.out_root.mkdir(parents=True, exist_ok=True)

    for method_key in args.methods:
        if method_key not in available_methods():
            print(f"Warning: {method_key!r} is not in the advertised method list.")
        method = get_method(method_key)
        method_dir = args.out_root / method_key
        method_dir.mkdir(parents=True, exist_ok=True)

        for h5_path in files:
            out_path = method_dir / h5_path.name
            if out_path.exists() and not args.overwrite:
                print(f"Skipping existing {out_path}")
                continue

            records = list(iter_h5_records(h5_path, max_records=args.max_records))
            temp_path = make_temp_h5_path()
            try:
                with h5py.File(temp_path, "w") as out_h5:
                    out_h5.attrs["source_file"] = h5_path.name
                    out_h5.attrs["method_key"] = method_key
                    out_h5.attrs["method"] = method.name
                    out_h5.attrs["runner"] = "run_methods_to_h5.py"

                    for rec in tqdm(records, desc=f"{method_key}: {h5_path.name}"):
                        intervals = mask_to_intervals(rec.artifact_mask)
                        try:
                            restored = method.run(
                                rec.signal,
                                intervals,
                                fs=rec.fs,
                                reference=rec.reference,
                            )
                            restored = np.asarray(restored, dtype=float)
                            ds = out_h5.create_dataset(rec.record_name, data=restored)
                            ds.attrs["status"] = "OK"
                        except Exception as exc:
                            ds = out_h5.create_dataset(rec.record_name, data=rec.signal)
                            ds.attrs["status"] = f"FAIL: {exc}"
                            print(f"Warning: {method_key}/{h5_path.name}/{rec.record_name} failed: {exc}")

                        ds.attrs["fs"] = rec.fs
                        ds.attrs["method_key"] = method_key
                        ds.attrs["method"] = method.name

                replace_file(temp_path, out_path)
                print(f"Saved {out_path}")
            finally:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)


def make_temp_h5_path() -> Path:
    temp_dir = Path(tempfile.gettempdir()) / "eeg_eval_restored_h5"
    temp_dir.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        suffix=".h5",
        prefix="restored_",
        dir=temp_dir,
        delete=False,
    )
    path = Path(handle.name)
    handle.close()
    path.unlink(missing_ok=True)
    return path


def replace_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
