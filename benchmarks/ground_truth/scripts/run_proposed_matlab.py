from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess


def matlab_quote(value: str | Path | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch MATLAB proposed denoiser H5 runner.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/proposed_restored"))
    parser.add_argument("--pattern", default="*.h5")
    parser.add_argument("--max-records", type=float, default=float("inf"))
    parser.add_argument("--pass-policy", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--package-parent",
        default=Path(__file__).resolve().parents[3] / "matlab",
    )
    parser.add_argument("--matlab", default="matlab")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    matlab_dir = project_root / "matlab"

    max_records_arg = "Inf" if args.max_records == float("inf") else str(args.max_records)
    command = (
        f"addpath({matlab_quote(matlab_dir)}); "
        "run_proposed_h5("
        f"{matlab_quote(args.data_dir)}, "
        f"{matlab_quote(args.out_dir)}, "
        f"{matlab_quote(args.pattern)}, "
        f"{max_records_arg}, "
        f"{matlab_quote(args.pass_policy)}, "
        f"{matlab_quote(args.overwrite)}, "
        f"{matlab_quote(args.package_parent)}"
        ");"
    )

    cmd = [args.matlab, "-batch", command]
    print("Running:")
    print(" ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
