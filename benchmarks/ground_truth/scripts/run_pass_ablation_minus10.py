from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


PATTERN = "03_denoise-net_*_-10dB.h5"

PASS_POLICIES = {
    "emg": ("proposed_emg", "Proposed EMG only"),
    "eog": ("proposed_eog", "Proposed EOG only"),
    "slow": ("proposed_slow", "Proposed slow only"),
    "emg+eog": ("proposed_emg_eog", "Proposed EMG+EOG"),
    "full": ("proposed_full", "Proposed full"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the proposed-method pass ablation on the representative "
            "-10 dB EEGdenoiseNet files."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["restore", "metrics", "analyze", "all"],
        default="all",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--result-tag", default="pass_ablation_minus10")
    parser.add_argument("--max-records", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--delete-restored-after-metrics",
        action="store_true",
        help="Remove restored ablation H5 folders after metrics are written.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=list(PASS_POLICIES),
        default=list(PASS_POLICIES),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    py = root / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)

    data_files = sorted(args.data_dir.glob(PATTERN))
    expected = {
        "03_denoise-net_emg_-10dB.h5",
        "03_denoise-net_eog_-10dB.h5",
        "03_denoise-net_eog+emg_-10dB.h5",
    }
    found = {p.name for p in data_files}
    missing = sorted(expected - found)
    if missing:
        raise FileNotFoundError(f"Missing required -10 dB files: {missing}")

    restored_root = Path("results") / f"restored_{args.result_tag}"
    metrics_root = Path("results") / f"metrics_{args.result_tag}"
    analysis_dir = Path("results") / f"analysis_{args.result_tag}"
    max_records_args = [] if args.max_records is None else ["--max-records", args.max_records]
    overwrite_args = ["--overwrite"] if args.overwrite else []

    if args.stage in {"restore", "all"}:
        for policy in args.policies:
            method_key, _label = PASS_POLICIES[policy]
            run(
                [
                    str(py),
                    "scripts/run_proposed_matlab.py",
                    "--data-dir",
                    str(args.data_dir),
                    "--pattern",
                    PATTERN,
                    "--pass-policy",
                    policy,
                    "--out-dir",
                    str(restored_root / method_key),
                    *max_records_args,
                    *overwrite_args,
                ],
                root,
            )

    if args.stage in {"metrics", "all"}:
        for policy in args.policies:
            method_key, label = PASS_POLICIES[policy]
            run(
                [
                    str(py),
                    "scripts/metrics_from_restored_h5.py",
                    "--data-dir",
                    str(args.data_dir),
                    "--pattern",
                    PATTERN,
                    "--restored-dir",
                    str(restored_root / method_key),
                    "--out-dir",
                    str(metrics_root / method_key),
                    "--method-key",
                    method_key,
                    "--method-label",
                    label,
                    *max_records_args,
                ],
                root,
            )

        if args.delete_restored_after_metrics:
            for policy in args.policies:
                method_key, _label = PASS_POLICIES[policy]
                target = root / restored_root / method_key
                if target.exists():
                    print(f"Deleting restored H5 folder after metrics: {target}")
                    shutil.rmtree(target)

    if args.stage in {"analyze", "all"}:
        run(
            [
                str(py),
                "scripts/analyze_pass_ablation_minus10.py",
                "--metrics-root",
                str(metrics_root),
                "--out-dir",
                str(analysis_dir),
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
