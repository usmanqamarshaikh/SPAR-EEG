from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and analyze SPAR-EEG clean-input activation diagnostics."
    )
    parser.add_argument("--stage", choices=["diagnose", "analyze", "all"], default="all")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("results/clean_preservation/data/04_denoise-net_clean.h5"),
    )
    parser.add_argument(
        "--restored-root",
        type=Path,
        default=Path("results/clean_preservation/restored"),
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path(
            "results/clean_preservation/analysis_passes/"
            "clean_preservation_pass_record_metrics.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/clean_preservation/activation_diagnostics"),
    )
    parser.add_argument("--max-records", type=int, default=2**31 - 1)
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
    data_file = resolve(root, args.data_file)
    restored_root = resolve(root, args.restored_root)
    metrics_file = resolve(root, args.metrics_file)
    out_dir = resolve(root, args.out_dir)
    diagnostics_file = out_dir / "clean_input_activation_record_diagnostics.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"diagnose", "all"}:
        command = (
            f"addpath({matlab_quote(root / 'matlab')}); "
            "run_clean_input_activation_h5("
            f"{matlab_quote(data_file)}, {matlab_quote(restored_root)}, "
            f"{matlab_quote(out_dir)}, {args.max_records}, "
            f"{'true' if args.overwrite else 'false'}, "
            f"{matlab_quote(args.package_parent)});"
        )
        run([args.matlab, "-batch", command], root)

    if args.stage in {"analyze", "all"}:
        if not diagnostics_file.exists():
            raise FileNotFoundError(diagnostics_file)
        if not metrics_file.exists():
            raise FileNotFoundError(metrics_file)
        python = root / ".venv" / "Scripts" / "python.exe"
        if not python.exists():
            python = Path(sys.executable)
        run(
            [
                str(python),
                "scripts/analyze_clean_input_activation.py",
                "--diagnostics-file",
                str(diagnostics_file),
                "--metrics-file",
                str(metrics_file),
                "--out-dir",
                str(out_dir),
            ],
            root,
        )


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def matlab_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    workspace_root = cwd.parents[2]
    local_matlab_dirs = {
        "MATLAB_PREFDIR": workspace_root / ".matlab_prefdir",
        "LOCALAPPDATA": workspace_root / ".matlab_localappdata",
        "APPDATA": workspace_root / ".matlab_appdata",
    }
    if Path(cmd[0]).name.lower() in {"matlab", "matlab.exe"}:
        for variable, directory in local_matlab_dirs.items():
            if directory.exists():
                env[variable] = str(directory)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


if __name__ == "__main__":
    main()
