from __future__ import annotations

import ast
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    python_files = [
        path
        for path in REPO_ROOT.rglob("*.py")
        if ".venv" not in path.parts and ".idea" not in path.parts
    ]
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    required = [
        REPO_ROOT / "matlab" / "+burstdenoise" / "denoise_emg_vmd_window.m",
        REPO_ROOT / "matlab" / "+burstdenoise" / "denoise_eog_ssa_window.m",
        REPO_ROOT / "matlab" / "+burstdenoise" / "denoise_slow_ssa_window.m",
        REPO_ROOT / "matlab" / "spar_eeg_denoise.m",
        REPO_ROOT / "matlab" / "spar_eeg_adaptive.m",
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required release files: {missing}")

    print(f"Parsed {len(python_files)} Python files successfully.")
    print("Required MATLAB release files are present.")
    print(f"Python: {sys.version.split()[0]}")


if __name__ == "__main__":
    main()
