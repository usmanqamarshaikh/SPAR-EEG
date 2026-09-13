from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_PRIMARY_PATTERNS = [
    "01_physiobank.h5",
    "02_semisimulated_eog.h5",
    "03_denoise-net_emg_-20dB.h5",
    "03_denoise-net_eog_-20dB.h5",
    "03_denoise-net_eog+emg_-20dB.h5",
]

DEFAULT_VARY_SNR_PATTERNS = [
    "03_denoise-net_emg_*.h5",
    "03_denoise-net_eog_*.h5",
    "03_denoise-net_eog+emg_*.h5",
]


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Orchestrate guided SOTA baselines, proposed MATLAB outputs, and "
            "common metrics for ground-truth benchmark files."
        )
    )
    parser.add_argument("--suite", choices=["primary", "vary-snr"], default="primary")
    parser.add_argument("--stage", choices=["sota", "proposed", "proposed-metrics", "all"], default="all")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--result-tag",
        default=None,
        help=(
            "Suffix used for result folders. Defaults to the suite name. "
            "For centered data, use e.g. --result-tag vary-snr-centered."
        ),
    )
    parser.add_argument("--max-records", type=str, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Baseline methods to run. Defaults to all five methods for the "
            "primary suite and WT/WQN only for the varying-SNR suite."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    py = root / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)

    patterns = DEFAULT_PRIMARY_PATTERNS if args.suite == "primary" else DEFAULT_VARY_SNR_PATTERNS
    result_tag = args.result_tag or args.suite
    methods = args.methods
    if methods is None:
        methods = (
            ["wt_hard", "wt_soft", "wqn", "emd_ica", "emd_cca"]
            if args.suite == "primary"
            else ["wt_hard", "wt_soft", "wqn"]
        )
    max_records_args = [] if args.max_records is None else ["--max-records", args.max_records]
    overwrite_args = ["--overwrite"] if args.overwrite else []

    if args.stage in {"sota", "all"}:
        for pattern in patterns:
            run(
                [
                    str(py),
                    "scripts/run_baselines.py",
                    "--data-dir",
                    args.data_dir,
                    "--out-dir",
                    f"results/metrics_sota_{result_tag}",
                    "--pattern",
                    pattern,
                    "--methods",
                    *methods,
                    *max_records_args,
                    *overwrite_args,
                ],
                root,
            )

    if args.stage in {"proposed", "all"}:
        for pattern in patterns:
            run(
                [
                    str(py),
                    "scripts/run_proposed_matlab.py",
                    "--data-dir",
                    args.data_dir,
                    "--pattern",
                    pattern,
                    "--pass-policy",
                    "auto",
                    "--out-dir",
                    f"results/proposed_restored_{result_tag}",
                    *max_records_args,
                    *overwrite_args,
                ],
                root,
            )

    if args.stage in {"proposed-metrics", "all"}:
        for pattern in patterns:
            run(
                [
                    str(py),
                    "scripts/metrics_from_restored_h5.py",
                    "--data-dir",
                    args.data_dir,
                    "--pattern",
                    pattern,
                    "--restored-dir",
                    f"results/proposed_restored_{result_tag}",
                    "--out-dir",
                    f"results/metrics_proposed_{result_tag}",
                    "--method-key",
                    "proposed_auto",
                    "--method-label",
                    "Proposed (self-sufficient auto)",
                    *max_records_args,
                ],
                root,
            )


if __name__ == "__main__":
    main()
