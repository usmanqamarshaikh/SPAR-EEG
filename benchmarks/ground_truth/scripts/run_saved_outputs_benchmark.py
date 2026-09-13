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

METHOD_LABELS = {
    "wt_hard": "WT-hard",
    "wt_soft": "WT-soft",
    "wqn": "WQN",
    "emd_ica": "EMD-ICA",
    "emd_cca": "EMD-CCA",
    "proposed_auto": "Proposed (self-sufficient auto)",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Save restored outputs for every method, then compute common metrics "
            "from those saved outputs."
        )
    )
    parser.add_argument("--suite", choices=["primary", "vary-snr"], default="primary")
    parser.add_argument(
        "--stage",
        choices=["sota-restore", "proposed-restore", "metrics", "all"],
        default="all",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--result-tag", default=None)
    parser.add_argument("--max-records", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--methods", nargs="+", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    py = root / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)

    result_tag = args.result_tag or args.suite
    patterns = DEFAULT_PRIMARY_PATTERNS if args.suite == "primary" else DEFAULT_VARY_SNR_PATTERNS
    methods = args.methods
    if methods is None:
        methods = (
            ["wt_hard", "wt_soft", "wqn", "emd_ica", "emd_cca"]
            if args.suite == "primary"
            else ["wt_hard", "wt_soft", "wqn"]
        )

    max_records_args = [] if args.max_records is None else ["--max-records", args.max_records]
    overwrite_args = ["--overwrite"] if args.overwrite else []

    restored_root = f"results/restored_{result_tag}"
    metrics_root = f"results/metrics_{result_tag}"

    if args.stage in {"sota-restore", "all"}:
        for pattern in patterns:
            run(
                [
                    str(py),
                    "scripts/run_methods_to_h5.py",
                    "--data-dir",
                    args.data_dir,
                    "--out-root",
                    restored_root,
                    "--pattern",
                    pattern,
                    "--methods",
                    *methods,
                    *max_records_args,
                    *overwrite_args,
                ],
                root,
            )

    if args.stage in {"proposed-restore", "all"}:
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
                    f"{restored_root}/proposed_auto",
                    *max_records_args,
                    *overwrite_args,
                ],
                root,
            )

    if args.stage in {"metrics", "all"}:
        metric_methods = [*methods, "proposed_auto"]
        for method_key in metric_methods:
            restored_dir = f"{restored_root}/{method_key}"
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
                        restored_dir,
                        "--out-dir",
                        f"{metrics_root}/{method_key}",
                        "--method-key",
                        method_key,
                        "--method-label",
                        METHOD_LABELS.get(method_key, method_key),
                        *max_records_args,
                    ],
                    root,
                )


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


if __name__ == "__main__":
    main()
