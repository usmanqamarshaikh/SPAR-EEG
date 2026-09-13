from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paired SPAR-EEG EMG occupancy study.")
    parser.add_argument("--stage", choices=["prepare", "restore", "analyze", "all"], default="all")
    parser.add_argument("--result-tag", default="artifact_occupancy_emg_minus10_n500")
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--snr-db", type=float, default=-10.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("results/analysis_zthr_sensitivity_multisnr_n500/selected_record_ids.csv"),
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
    result_root = root / "results" / args.result_tag
    data_file = result_root / "data" / "emg_occupancy_minus10.h5"
    restored_file = result_root / "restored" / "emg_occupancy_minus10_restored.h5"
    analysis_dir = result_root / "analysis"
    diagnostics_file = analysis_dir / "artifact_occupancy_stage_diagnostics.csv"
    manifest = analysis_dir / "selected_record_ids.csv"
    prior_geometry = (
        root
        / "results"
        / "artifact_geometry_emg_minus10_n500"
        / "data"
        / "emg_geometry_minus10.h5"
    )
    for directory in (data_file.parent, restored_file.parent, analysis_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_manifest = resolve(root, args.source_manifest)
    if args.stage in {"prepare", "all"}:
        prepare_manifest(source_manifest, manifest, args.max_records, args.overwrite)
        run_python(
            root,
            [
                "scripts/prepare_artifact_occupancy_robustness.py",
                "--raw-dir", str(root / "data" / "eeg-denoise-net"),
                "--manifest", str(manifest),
                "--out-file", str(data_file),
                "--submitted-file", str(root / "data" / "03_denoise-net_emg_-10dB.h5"),
                "--geometry-file", str(prior_geometry),
                "--snr-db", str(args.snr_db),
                "--max-records", str(args.max_records),
                *(["--overwrite"] if args.overwrite else []),
            ],
        )

    if args.stage in {"restore", "all"}:
        require(data_file)
        command = (
            f"addpath({matlab_quote(root / 'matlab')}); "
            "run_artifact_geometry_h5("
            f"{matlab_quote(data_file)}, {matlab_quote(restored_file)}, "
            f"{matlab_quote(diagnostics_file)}, "
            f"{'true' if args.overwrite else 'false'}, {matlab_quote(args.package_parent)});"
        )
        run([args.matlab, "-batch", command], root)

    if args.stage in {"analyze", "all"}:
        require(data_file)
        require(restored_file)
        require(diagnostics_file)
        run_python(
            root,
            [
                "scripts/analyze_artifact_occupancy_robustness.py",
                "--data-file", str(data_file),
                "--restored-file", str(restored_file),
                "--diagnostics-file", str(diagnostics_file),
                "--out-dir", str(analysis_dir),
                "--bootstrap-repetitions", str(args.bootstrap_repetitions),
                "--seed", str(args.seed),
            ],
        )

    print(f"\nArtifact-occupancy workflow complete: {result_root}")


def prepare_manifest(source: Path, destination: Path, n_records: int, overwrite: bool) -> None:
    require(source)
    frame = pd.read_csv(source)
    if "record_id" not in frame.columns:
        raise ValueError(f"Missing record_id column in {source}")
    if n_records < 1 or n_records > len(frame):
        raise ValueError(f"max-records must be in [1, {len(frame)}]")
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


def run_python(root: Path, arguments: list[str]) -> None:
    python = root / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = Path(sys.executable)
    run([str(python), *arguments], root)


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def matlab_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    workspace_root = cwd.parents[2]
    if Path(cmd[0]).name.lower() in {"matlab", "matlab.exe"}:
        for variable, relative in (
            ("MATLAB_PREFDIR", ".matlab_prefdir"),
            ("LOCALAPPDATA", ".matlab_localappdata"),
            ("APPDATA", ".matlab_appdata"),
        ):
            directory = workspace_root / relative
            if directory.exists():
                env[variable] = str(directory)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


if __name__ == "__main__":
    main()
