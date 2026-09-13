from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd


PREFIXES = ("emg", "eog")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPAR-EEG artifact-threshold sensitivity.")
    parser.add_argument("--stage", choices=["restore", "analyze", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--result-tag", default="zthr_sensitivity_multisnr_n500")
    parser.add_argument("--n-records", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--snr-levels", type=float, nargs="+", default=[-20, -10, 0])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument(
        "--package-parent",
        default=Path(__file__).resolve().parents[3] / "matlab",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = resolve(root, args.data_dir)
    restored_root = root / "results" / f"restored_{args.result_tag}"
    analysis_dir = root / "results" / f"analysis_{args.result_tag}"
    manifest = analysis_dir / "selected_record_ids.csv"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    thresholds = list(dict.fromkeys(args.thresholds))
    snr_levels = list(dict.fromkeys(args.snr_levels))
    if any(not np.isfinite(value) or value <= 0 for value in thresholds):
        raise ValueError("All thresholds must be finite and greater than zero.")

    if args.stage in {"restore", "all"}:
        selected_ids = select_common_records(data_dir, args.n_records, args.seed, snr_levels)
        pd.DataFrame(
            {
                "selection_order": np.arange(1, len(selected_ids) + 1),
                "record_id": selected_ids,
            }
        ).to_csv(manifest, index=False)
        run_matlab(
            root,
            args.matlab,
            data_dir,
            restored_root,
            manifest,
            thresholds,
            snr_levels,
            args.overwrite,
            args.package_parent,
        )

    if args.stage in {"analyze", "all"}:
        if not manifest.exists():
            raise FileNotFoundError(f"Missing selection manifest: {manifest}")
        py = root / ".venv" / "Scripts" / "python.exe"
        if not py.exists():
            py = Path(sys.executable)
        cmd = [
            str(py),
            "scripts/analyze_zthr_sensitivity.py",
            "--data-dir", str(data_dir),
            "--restored-root", str(restored_root),
            "--manifest", str(manifest),
            "--out-dir", str(analysis_dir),
            "--thresholds", *[str(value) for value in thresholds],
            "--snr-levels", *[str(value) for value in snr_levels],
        ]
        run(cmd, root)


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def select_common_records(
    data_dir: Path, n_records: int, seed: int, snr_levels: list[float]
) -> np.ndarray:
    available: list[set[int]] = []
    for prefix in PREFIXES:
        for snr_db in snr_levels:
            path = data_dir / sensitivity_file_name(prefix, snr_db)
            if not path.exists():
                raise FileNotFoundError(path)
            with h5py.File(path, "r") as h5:
                ids = {
                    int(key.rsplit("_", 1)[1])
                    for key in h5.keys()
                    if key.startswith(prefix + "_") and key.rsplit("_", 1)[1].isdigit()
                }
            available.append(ids)
    common = np.asarray(sorted(set.intersection(*available)), dtype=int)
    if n_records < 1 or n_records > len(common):
        raise ValueError(f"n_records must be in [1, {len(common)}], got {n_records}")
    return np.sort(np.random.default_rng(seed).choice(common, size=n_records, replace=False))


def run_matlab(
    root: Path,
    matlab: str,
    data_dir: Path,
    restored_root: Path,
    manifest: Path,
    thresholds: list[float],
    snr_levels: list[float],
    overwrite: bool,
    package_parent: str,
) -> None:
    threshold_arg = "[" + " ".join(f"{value:g}" for value in thresholds) + "]"
    snr_arg = "[" + " ".join(f"{value:g}" for value in snr_levels) + "]"
    command = (
        f"addpath({matlab_quote(root / 'matlab')}); run_zthr_sensitivity_h5("
        f"{matlab_quote(data_dir)}, {matlab_quote(restored_root)}, "
        f"{matlab_quote(manifest)}, {threshold_arg}, "
        f"{'true' if overwrite else 'false'}, {matlab_quote(package_parent)}, {snr_arg});"
    )
    run([matlab, "-batch", command], root)


def sensitivity_file_name(prefix: str, snr_db: float) -> str:
    return f"03_denoise-net_{prefix}_{snr_db:g}dB.h5"


def matlab_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


if __name__ == "__main__":
    main()
