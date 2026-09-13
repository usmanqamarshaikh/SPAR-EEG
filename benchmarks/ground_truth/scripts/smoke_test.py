from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight Evaluation smoke tests.")
    parser.add_argument("--skip-matlab", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    py = root / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)

    run([str(py), "scripts/inspect_h5.py", "--data-dir", "data", "--pattern", "02_*.h5", "--max-records", "1"], root)
    run(
        [
            str(py),
            "scripts/run_baselines.py",
            "--data-dir",
            "data",
            "--out-dir",
            "results/metrics_smoke",
            "--pattern",
            "03_denoise-net_emg_-20dB.h5",
            "--methods",
            "wt_hard",
            "wqn",
            "--max-records",
            "2",
            "--overwrite",
        ],
        root,
    )

    if not args.skip_matlab:
        run(
            [
                str(py),
                "scripts/run_proposed_matlab.py",
                "--pattern",
                "03_denoise-net_eog_-20dB.h5",
                "--max-records",
                "1",
                "--pass-policy",
                "auto",
                "--out-dir",
                "results/proposed_smoke",
                "--overwrite",
            ],
            root,
        )
        run(
            [
                str(py),
                "scripts/metrics_from_restored_h5.py",
                "--pattern",
                "03_denoise-net_eog_-20dB.h5",
                "--restored-dir",
                "results/proposed_smoke",
                "--method-key",
                "proposed_auto",
                "--method-label",
                "Proposed (auto)",
                "--max-records",
                "1",
            ],
            root,
        )
        run(
            [
                str(py),
                "scripts/run_proposed_matlab.py",
                "--pattern",
                "02_semisimulated_eog.h5",
                "--max-records",
                "1",
                "--pass-policy",
                "auto",
                "--out-dir",
                "results/proposed_smoke_semisim_auto",
                "--overwrite",
            ],
            root,
        )
        run(
            [
                str(py),
                "scripts/metrics_from_restored_h5.py",
                "--pattern",
                "02_semisimulated_eog.h5",
                "--restored-dir",
                "results/proposed_smoke_semisim_auto",
                "--method-key",
                "proposed_auto_semisim",
                "--method-label",
                "Proposed (auto semisim)",
                "--max-records",
                "1",
            ],
            root,
        )

    print("\nSmoke tests completed.")


if __name__ == "__main__":
    main()
