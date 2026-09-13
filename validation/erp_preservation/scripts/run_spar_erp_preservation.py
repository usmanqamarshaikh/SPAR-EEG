from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and analyze SPAR-EEG ERP preservation branches.")
    parser.add_argument("--stage", choices=["generate", "analyze", "all"], default="all")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=REPO_ROOT / "data" / "eeglab_sample" / "processedEEG.mat",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "erp_preservation",
    )
    parser.add_argument("--max-trials", type=int, default=2**31 - 1)
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
    input_file = resolve(root, args.input_file)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"generate", "all"}:
        command = (
            f"addpath({matlab_quote(root / 'matlab')}); "
            "run_spar_erp_preservation_branches("
            f"{matlab_quote(input_file)}, {matlab_quote(out_dir)}, {args.max_trials}, "
            f"{'true' if args.overwrite else 'false'}, {matlab_quote(args.package_parent)});"
        )
        run([args.matlab, "-batch", command], root)

    if args.stage in {"analyze", "all"}:
        bundle = out_dir / "spar_erp_branch_data.mat"
        diagnostics = out_dir / "spar_erp_branch_diagnostics.csv"
        if not bundle.exists() or not diagnostics.exists():
            raise FileNotFoundError("Branch bundle or diagnostic CSV is missing.")
        python = Path(sys.executable)
        run(
            [
                str(python),
                "scripts/analyze_spar_erp_preservation.py",
                "--bundle",
                str(bundle),
                "--diagnostics",
                str(diagnostics),
                "--out-dir",
                str(out_dir / "analysis"),
            ],
            root,
        )


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def matlab_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def run(command: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(shlex.quote(part) for part in command), flush=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    workspace = cwd.parents[1]
    if Path(command[0]).name.lower() in {"matlab", "matlab.exe"}:
        for variable, folder in {
            "MATLAB_PREFDIR": workspace / ".matlab_prefdir",
            "LOCALAPPDATA": workspace / ".matlab_localappdata",
            "APPDATA": workspace / ".matlab_appdata",
        }.items():
            if folder.exists():
                env[variable] = str(folder)
    subprocess.run(command, cwd=cwd, check=True, env=env)


if __name__ == "__main__":
    main()
