"""Single-channel methods used in the reported academic benchmark.

The WQN-related implementation is subject to the upstream research-use and
patent notice summarized in the repository's THIRD_PARTY_NOTICES.md file; it
is not covered by the SPAR-EEG MIT license.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .datasets import ensure_2d_rows, intervals_to_mask


class Denoiser:
    name = "base"

    def run(
        self,
        signal: np.ndarray,
        artifacts: list[tuple[int, int]],
        fs: float | None = None,
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        signal = ensure_2d_rows(signal)
        reference_2d = ensure_2d_rows(reference) if reference is not None else None

        norm_signal, params = self.normalize(signal)
        if reference_2d is not None:
            mean, std = params
            norm_reference = (reference_2d - mean) / std
        else:
            norm_reference = [None] * norm_signal.shape[0]

        out = np.zeros_like(norm_signal)
        for ch in range(norm_signal.shape[0]):
            ref_ch = None if reference_2d is None else norm_reference[ch]
            out[ch] = self.run_channel(norm_signal[ch], artifacts, fs, ref_ch)
        return self.denormalize(out, params)

    def normalize(self, signal: np.ndarray):
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
        std = np.where(std <= np.finfo(float).eps, 1.0, std)
        return (signal - mean) / std, (mean, std)

    def denormalize(self, signal: np.ndarray, params):
        mean, std = params
        return signal * std + mean

    def run_channel(
        self,
        signal: np.ndarray,
        artifacts: list[tuple[int, int]],
        fs: float | None = None,
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass
class WaveletThresholding(Denoiser):
    wavelet: str = "sym5"
    level: int = 5
    mode: str = "hard"

    @property
    def name(self) -> str:
        return f"WT-{self.mode} ({self.wavelet},L{self.level})"

    def run_channel(self, signal, artifacts, fs=None, reference=None):
        import pywt

        padded = self._pad(signal)
        coeffs = pywt.swt(padded, self.wavelet, self.level, norm=True, trim_approx=True)
        coeffs = np.asarray(coeffs)

        artifact_mask = intervals_to_mask(artifacts, coeffs.shape[1])
        k = np.sqrt(2.0 * np.log(coeffs.shape[1]))
        thresholds = k * np.median(np.abs(coeffs), axis=1) / 0.6745

        for coeff, threshold in zip(coeffs, thresholds):
            coeff[artifact_mask] = self._threshold(coeff[artifact_mask], threshold)

        rec = pywt.iswt(coeffs, wavelet=self.wavelet, norm=True)
        return np.asarray(rec[: signal.size], dtype=float)

    def _pad(self, data):
        min_div = 2**self.level
        remainder = data.size % min_div
        pad_len = (min_div - remainder) % min_div
        return np.pad(data, (0, pad_len))

    def _threshold(self, coeffs, threshold):
        if self.mode == "hard":
            return np.where(np.abs(coeffs) <= threshold, coeffs, 0.0)
        if self.mode == "soft":
            return np.clip(coeffs, -threshold, threshold)
        raise ValueError(f"Unknown wavelet mode: {self.mode}")


@dataclass
class WaveletQuantileNormalization(Denoiser):
    wavelet: str = "sym5"
    mode: str = "periodization"
    alpha: float = 1.0
    n: int = 20

    @property
    def name(self) -> str:
        return f"WQN ({self.wavelet})"

    def run_channel(self, signal, artifacts, fs=None, reference=None):
        import pywt

        restored = signal.copy()
        for idx, (i, j) in enumerate(artifacts):
            min_a = 0 if idx == 0 else artifacts[idx - 1][1]
            max_b = signal.size if idx + 1 == len(artifacts) else artifacts[idx + 1][0]

            size = int(j - i)
            if size <= 0:
                continue
            level = int(np.log2(size / self.n))
            if level < 1:
                continue

            ref_size = max(self.n * 2**level, size)
            a = max(min_a, i - ref_size)
            b = min(max_b, j + ref_size)
            if b <= a:
                continue

            coeffs = pywt.wavedec(signal[a:b], self.wavelet, mode=self.mode, level=level)
            for cs in coeffs:
                k = int(np.round(np.log2(b - a) - np.log2(cs.size)))
                ik, jk = np.array([i - a, j - a]) // 2**k
                refs = [cs[:ik], cs[jk:]]
                if len(refs[0]) == 0 and len(refs[1]) == 0:
                    continue

                target = cs[ik:jk]
                if target.size == 0:
                    continue
                order = np.argsort(np.abs(target))
                inv_order = np.empty_like(order)
                inv_order[order] = np.arange(len(order))

                vals_ref = np.abs(np.concatenate(refs))
                ref_order = np.argsort(vals_ref)
                ref_sp = np.linspace(0, len(inv_order), len(ref_order))
                vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

                denom = np.maximum(np.abs(target), np.finfo(float).eps)
                ratio = vals_norm / denom
                cs[ik:jk] *= np.minimum(1.0, ratio) ** self.alpha

            rec = pywt.waverec(coeffs, self.wavelet, mode=self.mode)
            restored[i:j] = rec[i - a : j - a]
        return restored


class EMDDenoiser(Denoiser):
    def __init__(self):
        from PyEMD import EMD

        self.emd = EMD(
            std_thr=0.2,
            energy_ratio_thr=0.2,
            total_power_thr=0,
            range_thr=0.001,
            DTYPE=np.float32,
            spline_kind="cubic",
        )


class EMDICA(EMDDenoiser):
    name = "EMD-ICA"

    def run_channel(self, signal, artifacts, fs=None, reference=None):
        if reference is None:
            raise ValueError("EMD-ICA benchmark implementation requires reference.")

        from sklearn.decomposition import FastICA

        imf = np.asarray(self.emd(signal, max_imf=10), dtype=float)
        if imf.ndim != 2 or imf.shape[0] < 2:
            return signal.copy()

        ica = FastICA(max_iter=2000, random_state=0)
        ics = ica.fit_transform(imf.T)

        r0 = safe_corr(signal, reference)
        bad = np.zeros(ics.shape[1], dtype=bool)
        for idx in range(ics.shape[1]):
            trial = ics.copy()
            trial[:, idx] = 0.0
            restored_trial = ica.inverse_transform(trial).sum(axis=1)
            bad[idx] = safe_corr(restored_trial, reference) > r0

        ics[:, bad] = 0.0
        reconstructed = ica.inverse_transform(ics).sum(axis=1)
        mask = intervals_to_mask(artifacts, signal.size)
        restored = signal.copy()
        restored[mask] = reconstructed[mask]
        return restored


class EMDCCA(EMDDenoiser):
    name = "EMD-CCA"

    def run_channel(self, signal, artifacts, fs=None, reference=None):
        if reference is None:
            raise ValueError("EMD-CCA benchmark implementation requires reference.")

        imf = np.asarray(self.emd(signal, max_imf=10), dtype=float)
        if imf.ndim != 2 or imf.shape[0] < 2:
            return signal.copy()

        imf_conv = np.array([np.convolve(x, [1, 0, 1], mode="same") for x in imf])
        wa, _, _ = cca(imf, imf_conv)
        ccs = wa.T @ imf
        wa_inv = np.linalg.inv(wa.T)

        r0 = safe_corr(signal, reference)
        bad = np.zeros(ccs.shape[0], dtype=bool)
        for idx in range(ccs.shape[0]):
            trial = ccs.copy()
            trial[idx] = 0.0
            restored_trial = (wa_inv @ trial).sum(axis=0)
            bad[idx] = safe_corr(restored_trial, reference) > r0

        ccs[bad] = 0.0
        reconstructed = (wa_inv @ ccs).sum(axis=0)
        mask = intervals_to_mask(artifacts, signal.size)
        restored = signal.copy()
        restored[mask] = reconstructed[mask]
        return restored


def cca(x: np.ndarray, y: np.ndarray):
    z = np.concatenate([x, y]).T
    cov = np.cov(z.T)
    nx = x.shape[0]
    ny = y.shape[0]
    cxx = cov[:nx, :nx] + 1e-10 * np.eye(nx)
    cxy = cov[:nx, nx : nx + ny]
    cyx = cxy.T
    cyy = cov[nx : nx + ny, nx : nx + ny] + 1e-10 * np.eye(ny)

    cxx_inv = np.linalg.inv(cxx)
    cyy_inv = np.linalg.inv(cyy)

    eigvals, eigvecs = np.linalg.eig(cxx_inv @ cxy @ cyy_inv @ cyx)
    order = eigvals.argsort()[::-1]
    r = np.sqrt(np.maximum(eigvals[order], 0.0))
    wx = np.real(eigvecs[:, order])
    wy = cyy_inv @ cyx @ wx
    wy /= np.sqrt((np.abs(wy) ** 2).sum(axis=0, keepdims=True))
    return wx, wy, r


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= np.finfo(float).eps or np.std(b) <= np.finfo(float).eps:
        return -np.inf
    return float(np.corrcoef(a, b)[0, 1])


def get_method(name: str) -> Denoiser:
    key = name.lower().replace("-", "_")
    if key in {"wt_hard", "hard", "wavelet_hard"}:
        return WaveletThresholding("sym5", level=5, mode="hard")
    if key in {"wt_soft", "soft", "wavelet_soft"}:
        return WaveletThresholding("sym5", level=5, mode="soft")
    if key in {"wqn", "wavelet_quantile_normalization"}:
        return WaveletQuantileNormalization("sym5", n=20)
    if key in {"emd_ica", "emdica"}:
        return EMDICA()
    if key in {"emd_cca", "emdcca"}:
        return EMDCCA()
    raise ValueError(f"Unknown method: {name}")


def available_methods() -> list[str]:
    return ["wt_hard", "wt_soft", "wqn", "emd_ica", "emd_cca"]
