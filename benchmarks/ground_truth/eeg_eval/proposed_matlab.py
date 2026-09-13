from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .datasets import ensure_2d_rows


@dataclass
class MatlabBurstDenoise:
    """Optional MATLAB Engine wrapper for the proposed burstdenoise package.

    Add the parent folder of `+burstdenoise` to the MATLAB path, for example:

    C:\\Users\\zrf1847\\Documents\\MATLAB\\plugins\\burstdenoiseMethod

    This class is intentionally lightweight. For full-scale experiments, a
    MATLAB batch script that writes shape-preserving outputs may be more robust.
    """

    matlab_package_parent: str | Path
    pass_order: tuple[str, ...] = ("emg", "eog", "slow")

    def __post_init__(self):
        import matlab.engine

        self.matlab = matlab.engine.start_matlab()
        self.matlab.addpath(str(self.matlab_package_parent), nargout=0)

    @property
    def name(self) -> str:
        return "Proposed (" + "+".join(self.pass_order) + ")"

    def run(self, signal, artifacts=None, fs=None, reference=None):
        signal = ensure_2d_rows(signal)
        out = np.zeros_like(signal, dtype=float)
        for ch in range(signal.shape[0]):
            y = signal[ch].astype(float)
            for name in self.pass_order:
                y = self._run_pass(y, float(fs), name)
            out[ch] = y
        return out

    def _run_pass(self, y: np.ndarray, fs: float, name: str) -> np.ndarray:
        import matlab

        mat_y = matlab.double(y.reshape(-1, 1).tolist())
        if name == "emg":
            result = self.matlab.feval("burstdenoise.denoise_emg_vmd_window", mat_y, fs, nargout=1)
        elif name == "eog":
            result = self.matlab.feval("burstdenoise.denoise_eog_ssa_window", mat_y, fs, nargout=1)
        elif name == "slow":
            result = self.matlab.feval("burstdenoise.denoise_slow_ssa_window", mat_y, fs, nargout=1)
        else:
            raise ValueError(f"Unknown proposed pass: {name}")
        return np.asarray(result, dtype=float).reshape(-1)
