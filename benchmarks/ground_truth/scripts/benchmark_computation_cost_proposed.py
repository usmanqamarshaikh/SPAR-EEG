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


def matlab_cellstr(values: list[str]) -> str:
    return "{" + ", ".join(matlab_quote(v) for v in values) + "}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch MATLAB proposed-method computation-cost benchmark."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/computation_cost"))
    parser.add_argument("--pattern", default="03_denoise-net_*_-10dB.h5")
    parser.add_argument("--max-records", type=float, default=100)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["emg", "eog", "slow", "emg+eog", "full"],
        choices=["emg", "eog", "slow", "emg+eog", "full"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--package-parent",
        default=Path(__file__).resolve().parents[3] / "matlab",
    )
    parser.add_argument("--matlab", default="matlab")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    matlab_dir = root / "matlab"
    max_records_arg = "Inf" if args.max_records == float("inf") else str(args.max_records)

    command = (
        f"addpath({matlab_quote(matlab_dir)}); "
        "benchmark_proposed_cost("
        f"{matlab_quote(args.data_dir)}, "
        f"{matlab_quote(args.out_dir)}, "
        f"{matlab_quote(args.pattern)}, "
        f"{max_records_arg}, "
        f"{matlab_cellstr(args.policies)}, "
        f"{args.repeats}, "
        f"{matlab_quote(args.package_parent)}"
        ");"
    )

    cmd = [args.matlab, "-batch", command]
    print("Running:")
    print(" ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
