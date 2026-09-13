from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datasets import mask_to_intervals


@dataclass
class AdaptiveConfig:
    min_len_sec: float = 2.0
    max_len_sec: float = 4.0
    search_radius_sec: float = 2.0
    extend_step_sec: float = 1.0
    rms_win_sec: float = 0.10
    smooth_score_sec: float = 0.03
    amp_weight: float = 1.0
    slope_weight: float = 0.5
    rms_weight: float = 2.0
    zc_bonus: float = 0.15
    accept_percentile: float = 10.0


def robust_normalize_for_scoring(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad <= np.finfo(float).eps:
        std = np.std(x)
        if std <= np.finfo(float).eps:
            std = 1.0
        return (x - np.mean(x)) / std
    return (x - med) / (1.4826 * mad)


def boundary_score(xn: np.ndarray, fs: float, cfg: AdaptiveConfig) -> np.ndarray:
    xn = np.asarray(xn, dtype=float).reshape(-1)
    rms_w = max(3, int(round(cfg.rms_win_sec * fs)))
    dx = np.r_[0.0, np.diff(xn)]
    local_rms = moving_average(xn**2, rms_w) ** 0.5
    zc = np.zeros(xn.size, dtype=float)
    zc[1:] = np.sign(xn[1:]) != np.sign(xn[:-1])
    return (
        cfg.amp_weight * np.abs(xn)
        + cfg.slope_weight * np.abs(dx)
        + cfg.rms_weight * local_rms
        - cfg.zc_bonus * zc
    )


def adaptive_epochs(score: np.ndarray, fs: float, cfg: AdaptiveConfig) -> list[tuple[int, int]]:
    score = np.asarray(score, dtype=float).reshape(-1)
    n = score.size
    min_len = max(1, int(round(cfg.min_len_sec * fs)))
    max_len = max(min_len, int(round(cfg.max_len_sec * fs)))
    search_r = max(1, int(round(cfg.search_radius_sec * fs)))
    extend_st = max(1, int(round(cfg.extend_step_sec * fs)))
    accept_cost = np.percentile(score, cfg.accept_percentile)

    epochs: list[tuple[int, int]] = []
    start = 0
    while start < n:
        if start + min_len >= n:
            epochs.append((start, n))
            break

        nominal = start + min_len
        best_end = nominal
        best_cost = np.inf
        cur_nominal = nominal
        hard_limit = min(n, start + max_len)

        while cur_nominal <= hard_limit:
            lo = max(start + min_len, cur_nominal - search_r)
            hi = min(hard_limit, cur_nominal + search_r)
            if hi <= lo:
                break
            rel = int(np.argmin(score[lo:hi]))
            cand_end = lo + rel
            cand_cost = float(score[cand_end])
            if cand_cost < best_cost:
                best_cost = cand_cost
                best_end = cand_end
            if cand_cost <= accept_cost:
                break
            cur_nominal += extend_st

        best_end = max(best_end, start + 1)
        epochs.append((start, best_end))
        start = best_end
    return epochs


def apply_adaptive_channel(method, signal: np.ndarray, artifact_mask: np.ndarray, fs: float, cfg: AdaptiveConfig):
    x = np.asarray(signal, dtype=float).reshape(-1)
    xn = robust_normalize_for_scoring(x)
    score = boundary_score(xn, fs, cfg)
    epochs = adaptive_epochs(score, fs, cfg)

    y = np.zeros_like(x)
    for start, end in epochs:
        seg = x[start:end]
        seg_mask = artifact_mask[start:end]
        intervals = mask_to_intervals(seg_mask)
        if intervals:
            y[start:end] = method.run(seg.reshape(1, -1), intervals, fs=fs)[0]
        else:
            y[start:end] = seg
    return y


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    win = max(1, int(win))
    if win == 1:
        return x
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(x, kernel, mode="same")
