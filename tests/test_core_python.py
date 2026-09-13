from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_ROOT = REPO_ROOT / "benchmarks" / "ground_truth"
P300_ROOT = REPO_ROOT / "validation" / "p300_bci"
sys.path.insert(0, str(GROUND_TRUTH_ROOT))
sys.path.insert(0, str(P300_ROOT))

from eeg_eval.datasets import ensure_2d_rows, intervals_to_mask  # noqa: E402
from eeg_eval.methods import get_method  # noqa: E402
from eeg_eval.metrics import evaluate_record  # noqa: E402
from bci_p300_denoise.eeglab_io import channel_indices  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(11)
    fs = 256.0
    reference = rng.normal(0.0, 0.25, (1, 512))
    signal = reference.copy()
    signal[:, 192:320] += rng.normal(0.0, 1.0, (1, 128))
    artifacts = [(192, 320)]
    mask = intervals_to_mask(artifacts, signal.shape[1])

    for name in ("wt_hard", "wt_soft", "wqn"):
        restored = get_method(name).run(signal, artifacts, fs=fs, reference=reference)
        assert ensure_2d_rows(restored).shape == signal.shape
        assert np.isfinite(restored).all()
        rows = evaluate_record(restored, signal, reference, mask, fs)
        assert {row["region"] for row in rows} == {"artifact", "clean", "whole"}

    assert channel_indices(["FP1", "FP2", "Cz"], ["fp1", "FP2"]) == [0, 1]
    print("Python numerical smoke test passed.")


if __name__ == "__main__":
    main()
