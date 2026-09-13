from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.signal as ss
from scipy.stats import entropy

from .datasets import ensure_2d_rows, normalize_mask


EPS = np.finfo(float).eps


def evaluate_record(
    restored: np.ndarray,
    signal: np.ndarray,
    reference: np.ndarray,
    artifact_mask: np.ndarray,
    fs: float,
) -> list[dict[str, float | str | int]]:
    restored = ensure_2d_rows(restored)
    signal = ensure_2d_rows(signal)
    reference = ensure_2d_rows(reference)

    if restored.shape != signal.shape or restored.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch restored={restored.shape}, signal={signal.shape}, "
            f"reference={reference.shape}"
        )

    n_samples = signal.shape[1]
    artifact_mask = normalize_mask(artifact_mask, n_samples)

    regions = {
        "artifact": artifact_mask,
        "clean": ~artifact_mask,
        "whole": np.ones(n_samples, dtype=bool),
    }

    rows = []
    for region_name, region_mask in regions.items():
        row = calculate_region_metrics(restored, signal, reference, region_mask, fs)
        row["region"] = region_name
        row["region_samples"] = int(region_mask.sum())
        row["region_fraction"] = float(region_mask.mean())
        rows.append(row)
    return rows


def calculate_region_metrics(
    restored: np.ndarray,
    signal: np.ndarray,
    reference: np.ndarray,
    region_mask: np.ndarray,
    fs: float,
) -> dict[str, float]:
    restored = ensure_2d_rows(restored)
    signal = ensure_2d_rows(signal)
    reference = ensure_2d_rows(reference)
    mask = normalize_mask(region_mask, signal.shape[1])

    if not np.any(mask):
        return empty_metrics()

    sig_r = signal[:, mask]
    ref_r = reference[:, mask]
    res_r = restored[:, mask]

    snr_before = calculate_snr(ref_r, sig_r - ref_r)
    snr_after = calculate_snr(ref_r, res_r - ref_r)
    nmse_before = calculate_nmse_db(ref_r, sig_r)
    nmse_after = calculate_nmse_db(ref_r, res_r)

    rmse_before = rmse(ref_r, sig_r)
    rmse_after = rmse(ref_r, res_r)
    rrmse_before = relative_l2_error(ref_r, sig_r)
    rrmse_after = relative_l2_error(ref_r, res_r)

    corr_before = mean_channel_corr(sig_r, ref_r)
    corr_after = mean_channel_corr(res_r, ref_r)

    coh_before = mean_coherence(sig_r, ref_r, fs=fs, max_freq=40.0)
    coh_after = mean_coherence(res_r, ref_r, fs=fs, max_freq=40.0)

    psd_err_before = mean_relative_psd_error(sig_r, ref_r, fs=fs, max_freq=40.0)
    psd_err_after = mean_relative_psd_error(res_r, ref_r, fs=fs, max_freq=40.0)

    mi_after = normalized_mutual_information(ref_r, res_r)

    output_change = relative_l2_error(signal[:, mask], restored[:, mask])

    return {
        "SNR_before": snr_before,
        "SNR_after": snr_after,
        "deltaSNR": safe_sub(snr_after, snr_before),
        "NMSE_before": nmse_before,
        "NMSE_after": nmse_after,
        "deltaNMSE": safe_sub(nmse_after, nmse_before),
        "RMSE_before": rmse_before,
        "RMSE_after": rmse_after,
        "RRMSE_before": rrmse_before,
        "RRMSE_after": rrmse_after,
        "R_before": corr_before,
        "R_after": corr_after,
        "deltaR": safe_sub(corr_after, corr_before),
        "deltaR_normalized": safe_div(corr_after - corr_before, 1.0 - corr_before),
        "Coh_before": coh_before,
        "Coh_after": coh_after,
        "deltaCoh": safe_sub(coh_after, coh_before),
        "deltaCoh_normalized": safe_div(coh_after - coh_before, 1.0 - coh_before),
        "PSDerr_before": psd_err_before,
        "PSDerr_after": psd_err_after,
        "deltaPSDerr": safe_sub(psd_err_after, psd_err_before),
        "MI_after": mi_after,
        "OutputChange_RRMSE": output_change,
    }


def empty_metrics() -> dict[str, float]:
    keys = [
        "SNR_before",
        "SNR_after",
        "deltaSNR",
        "NMSE_before",
        "NMSE_after",
        "deltaNMSE",
        "RMSE_before",
        "RMSE_after",
        "RRMSE_before",
        "RRMSE_after",
        "R_before",
        "R_after",
        "deltaR",
        "deltaR_normalized",
        "Coh_before",
        "Coh_after",
        "deltaCoh",
        "deltaCoh_normalized",
        "PSDerr_before",
        "PSDerr_after",
        "deltaPSDerr",
        "MI_after",
        "OutputChange_RRMSE",
    ]
    return {k: np.nan for k in keys}


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    sig_var = float(np.var(np.asarray(signal, dtype=float)))
    noise_var = float(np.var(np.asarray(noise, dtype=float)))
    if sig_var <= EPS and noise_var <= EPS:
        return np.nan
    if noise_var <= EPS:
        return np.nan
    return float(10.0 * np.log10(sig_var / noise_var))


def calculate_nmse_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float).ravel()
    est = np.asarray(estimate, dtype=float).ravel()
    denom = float(np.sum(ref**2))
    if denom <= EPS:
        return np.nan
    nmse = float(np.sum((ref - est) ** 2) / denom)
    if nmse <= EPS:
        return np.nan
    return float(10.0 * np.log10(nmse))


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    return float(np.sqrt(np.mean((ref - est) ** 2)))


def relative_l2_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float).ravel()
    est = np.asarray(estimate, dtype=float).ravel()
    denom = float(np.linalg.norm(ref))
    if denom <= EPS:
        return np.nan
    return float(np.linalg.norm(ref - est) / denom)


def mean_channel_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = ensure_2d_rows(x)
    y = ensure_2d_rows(y)
    vals = []
    for a, b in zip(x, y):
        if a.size < 2 or np.std(a) <= EPS or np.std(b) <= EPS:
            vals.append(np.nan)
        else:
            vals.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan


def mean_coherence(x: np.ndarray, y: np.ndarray, fs: float, max_freq: float = 40.0) -> float:
    x = ensure_2d_rows(x)
    y = ensure_2d_rows(y)
    vals = []
    for a, b in zip(x, y):
        if a.size < 4:
            vals.append(np.nan)
            continue
        nperseg = min(128, a.size)
        f, coh = ss.coherence(a, b, fs=fs, nperseg=nperseg)
        band = f <= max_freq
        vals.append(float(np.nanmean(coh[band])))
    return float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan


def mean_relative_psd_error(x: np.ndarray, y: np.ndarray, fs: float, max_freq: float = 40.0) -> float:
    x = ensure_2d_rows(x)
    y = ensure_2d_rows(y)
    vals = []
    for a, b in zip(x, y):
        if a.size < 4:
            vals.append(np.nan)
            continue
        nperseg = min(128, a.size)
        f, pxx = ss.welch(a, fs=fs, nperseg=nperseg)
        _, pyy = ss.welch(b, fs=fs, nperseg=nperseg)
        band = f <= max_freq
        vals.append(float(np.linalg.norm(pxx[band] - pyy[band]) / (np.linalg.norm(pyy[band]) + EPS)))
    return float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan


def normalized_mutual_information(reference: np.ndarray, restored: np.ndarray, bins: int = 128) -> float:
    x = np.asarray(reference, dtype=float).ravel()
    y = np.asarray(restored, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        return np.nan
    hist = np.histogram2d(x, y, bins=(bins, bins))[0]
    total = hist.sum()
    if total <= 0:
        return np.nan
    pxy = np.clip(hist / total, EPS, None)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    mi = float(np.sum(pxy * np.log(pxy / (px @ py))))
    hx = float(entropy(px.ravel()))
    hy = float(entropy(py.ravel()))
    if hx <= EPS or hy <= EPS:
        return np.nan
    return float(mi / np.sqrt(hx * hy))


def safe_div(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= EPS:
        return np.nan
    return float(num / den)


def safe_sub(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    return float(a - b)


def finalize_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare metric tables for storage and spreadsheet-friendly analysis."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    if "nominal_snr_db" in out.columns and "SNR_before" in out.columns:
        if "SNR_before_region_empirical" not in out.columns:
            out["SNR_before_region_empirical"] = out["SNR_before"]
        nominal = pd.to_numeric(out["nominal_snr_db"], errors="coerce")
        empirical = pd.to_numeric(out["SNR_before"], errors="coerce")
        out["input_snr_db"] = nominal.where(nominal.notna(), empirical)
        out["input_snr_source"] = np.where(nominal.notna(), "nominal_file_label", "empirical_region")
        nominal_rows = nominal.notna()
        out.loc[nominal_rows, "SNR_before"] = out.loc[nominal_rows, "input_snr_db"]
        if "SNR_after" in out.columns and "deltaSNR" in out.columns:
            snr_after = pd.to_numeric(out["SNR_after"], errors="coerce")
            snr_before = pd.to_numeric(out["SNR_before"], errors="coerce")
            out["deltaSNR"] = snr_after - snr_before

    return out
