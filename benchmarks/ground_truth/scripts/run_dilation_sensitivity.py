from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SPAR-EEG dilation sensitivity.")
    parser.add_argument("--stage", choices=["restore", "analyze", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--result-tag", default="dilation_sensitivity_multisnr_n500")
    parser.add_argument("--n-records", type=int, default=500)
    parser.add_argument("--multipliers", type=float, nargs="+", default=[0, 1, 2])
    parser.add_argument("--snr-levels", type=float, nargs="+", default=[-20, -10, 0])
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("results/analysis_zthr_sensitivity_multisnr_n500/selected_record_ids.csv"),
    )
    parser.add_argument(
        "--default-reference-root",
        type=Path,
        default=Path("results/restored_zthr_sensitivity_multisnr_n500"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument(
        "--package-parent",
        default=Path(__file__).resolve().parents[3] / "matlab",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = resolve(root, args.data_dir)
    restored_root = root / "results" / f"restored_{args.result_tag}"
    analysis_dir = root / "results" / f"analysis_{args.result_tag}"
    manifest = analysis_dir / "selected_record_ids.csv"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    multipliers = list(dict.fromkeys(float(value) for value in args.multipliers))
    snr_levels = list(dict.fromkeys(float(value) for value in args.snr_levels))
    if any(not np.isfinite(value) or value < 0 for value in multipliers):
        raise ValueError("Dilation multipliers must be finite and non-negative.")

    if args.stage in {"restore", "all"}:
        prepare_manifest(
            resolve(root, args.source_manifest), manifest, args.n_records, args.overwrite
        )
        multiplier_arg = "[" + " ".join(f"{value:g}" for value in multipliers) + "]"
        snr_arg = "[" + " ".join(f"{value:g}" for value in snr_levels) + "]"
        command = (
            f"addpath({matlab_quote(root / 'matlab')}); run_dilation_sensitivity_h5("
            f"{matlab_quote(data_dir)}, {matlab_quote(restored_root)}, "
            f"{matlab_quote(manifest)}, {multiplier_arg}, "
            f"{'true' if args.overwrite else 'false'}, "
            f"{matlab_quote(args.package_parent)}, {snr_arg});"
        )
        run([args.matlab, "-batch", command], root)

    if args.stage in {"analyze", "all"}:
        require(manifest)
        python = root / ".venv" / "Scripts" / "python.exe"
        if not python.exists():
            python = Path(sys.executable)
        command = [
            str(python),
            "scripts/analyze_dilation_sensitivity.py",
            "--data-dir", str(data_dir),
            "--restored-root", str(restored_root),
            "--manifest", str(manifest),
            "--out-dir", str(analysis_dir),
            "--default-reference-root", str(resolve(root, args.default_reference_root)),
            "--multipliers", *[str(value) for value in multipliers],
            "--snr-levels", *[str(value) for value in snr_levels],
        ]
        run(command, root)


def prepare_manifest(source: Path, destination: Path, n_records: int, overwrite: bool) -> None:
    require(source)
    frame = pd.read_csv(source)
    if "record_id" not in frame.columns:
        raise ValueError(f"Missing record_id column in {source}")
    if n_records < 1 or n_records > len(frame):
        raise ValueError(f"n-records must be in [1, {len(frame)}]")
    selected = frame.iloc[:n_records].copy()
    selected["selection_order"] = range(1, len(selected) + 1)
    selected = selected[["selection_order", "record_id"]]
    if destination.exists() and not overwrite:
        existing = pd.read_csv(destination)
        if not existing.equals(selected):
            raise RuntimeError(f"Existing manifest differs from requested selection: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".csv.tmp")
    selected.to_csv(temp, index=False)
    shutil.move(temp, destination)


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def matlab_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def run(command: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(shlex.quote(part) for part in command), flush=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if Path(command[0]).name.lower() in {"matlab", "matlab.exe"}:
        workspace_root = cwd.parents[2]
        for variable, relative in (
            ("MATLAB_PREFDIR", ".matlab_prefdir"),
            ("LOCALAPPDATA", ".matlab_localappdata"),
            ("APPDATA", ".matlab_appdata"),
        ):
            directory = workspace_root / relative
            if directory.exists():
                env[variable] = str(directory)
    subprocess.run(command, cwd=cwd, check=True, env=env)


if __name__ == "__main__":
    main()
