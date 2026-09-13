# %%
"""
Step-by-step visual preprocessing demo for the Won P300 FP1/FP2 analysis.

This file is designed for VS Code's Python/Jupyter cell runner.
Run each `# %%` cell one by one and adjust CONFIG values as needed.

Goal:
    Visually diagnose where residual high-frequency noise remains:
    raw -> wide bandpass -> notch -> narrow P300 filter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import spectrogram, welch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bci_p300_denoise.eeglab_io import channel_indices, load_eeglab_run, memmap_fdt  # noqa: E402


# %%
# =========================
# CONFIG: edit these values
# =========================

DATASET_ROOT = Path(__file__).resolve().parents[3] / "data" / "Won2022_BIDS"
SUBJECT = "sub-001"
TASK = "P300trainrun1"
RUN = 6
CHANNELS = ["FP1", "FP2"]

# Window used for detailed time-series plots.
START_SEC = 40.0
WINDOW_SEC = 12.0

# Shared conditioning candidates.
WIDE_LOW_HZ = 0.5
WIDE_HIGH_HZ = 70.0

# Start with the currently suspected narrow interference peaks.
# Change this list interactively if you see other peaks in the PSD.
NOTCH_FREQS = [48.0, 64.0]
NOTCH_WIDTHS = 2.0

# Official-style P300 decoding filter candidate.
P300_LOW_HZ = 0.5
P300_HIGH_HZ = 10.0

OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "preprocessing_debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# %%
# =========================
# Load FP1/FP2 raw signal
# =========================

run_info = load_eeglab_run(DATASET_ROOT, SUBJECT, TASK, RUN)
idx = channel_indices(run_info.labels, CHANNELS)
raw = np.asarray(memmap_fdt(run_info)[idx, :], dtype=np.float64)
fs = float(run_info.srate)
t = np.arange(raw.shape[1]) / fs

print(f"Loaded: {run_info.set_path}")
print(f"Channels: {CHANNELS}")
print(f"Shape: {raw.shape}, fs={fs:g} Hz, duration={raw.shape[1] / fs:.2f} s")
print("Raw first samples:", raw[:, :5])


# %%
# =========================
# Helper plotting functions
# =========================

def robust_ylim(arrays, q=(0.5, 99.5), pad_frac=0.08):
    values = np.concatenate([np.asarray(a).reshape(-1) for a in arrays])
    lo, hi = np.percentile(values, q)
    pad = max(1.0, (hi - lo) * pad_frac)
    return float(lo - pad), float(hi + pad)


def plot_timeseries_compare(signals, labels, start_sec=START_SEC, window_sec=WINDOW_SEC, title=""):
    start = int(round(start_sec * fs))
    stop = min(raw.shape[1], start + int(round(window_sec * fs)))
    tt = np.arange(start, stop) / fs
    tt = tt - tt[0]

    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(13, 6), sharex=True)
    if len(CHANNELS) == 1:
        axes = [axes]

    for ch, ax in enumerate(axes):
        arrs = [sig[ch, start:stop] for sig in signals]
        ax.set_ylim(*robust_ylim(arrs))
        for sig, lab in zip(signals, labels):
            ax.plot(tt, sig[ch, start:stop], lw=0.9, label=lab)
        ax.set_ylabel(f"{CHANNELS[ch]} (uV)")
        ax.grid(True, color="#dddddd", linewidth=0.6)
        if ch == 0:
            ax.legend(loc="upper right", ncol=min(4, len(labels)), fontsize=8, frameon=False)
            ax.set_title(title or f"{SUBJECT} {TASK} run-{RUN}: {start_sec:g}-{start_sec + window_sec:g} s")
    axes[-1].set_xlabel("Time within window (s)")
    fig.tight_layout()
    return fig


def plot_psd_compare(signals, labels, fmax=90.0, title=""):
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(13, 4.5), sharey=True)
    if len(CHANNELS) == 1:
        axes = [axes]

    for ch, ax in enumerate(axes):
        for sig, lab in zip(signals, labels):
            f, pxx = welch(sig[ch], fs=fs, nperseg=min(8192, sig.shape[1]), noverlap=None)
            keep = f <= fmax
            ax.semilogy(f[keep], pxx[keep], lw=1.0, label=lab)
        for nf in NOTCH_FREQS:
            ax.axvline(nf, color="#999999", lw=0.8, ls="--")
        ax.set_title(CHANNELS[ch])
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(True, color="#dddddd", linewidth=0.6)
        if ch == 0:
            ax.set_ylabel("PSD (uV^2/Hz)")
            ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(title or f"{SUBJECT} {TASK} run-{RUN}: PSD comparison", y=1.02)
    fig.tight_layout()
    return fig


def print_top_psd_peaks(signal, name, lo=35.0, hi=90.0, top_n=12):
    f, pxx = welch(signal, fs=fs, nperseg=min(8192, signal.shape[1]), axis=1)
    mean_pxx = pxx.mean(axis=0)
    band = (f >= lo) & (f <= hi)
    fb = f[band]
    pb = mean_pxx[band]
    order = np.argsort(pb)[::-1][:top_n]
    print(f"\nTop PSD bins for {name} between {lo:g}-{hi:g} Hz:")
    for k in order:
        print(f"  {fb[k]:7.3f} Hz : {pb[k]:.6g}")


# %%
# =========================
# Visualize raw data only
# =========================

fig = plot_timeseries_compare([raw], ["raw"], title="Raw FP1/FP2")
fig.savefig(OUT_DIR / f"{SUBJECT}_{TASK}_run-{RUN}_01_raw_timeseries.png", dpi=180)
plt.show()

fig = plot_psd_compare([raw], ["raw"], fmax=90, title="Raw FP1/FP2 PSD")
fig.savefig(OUT_DIR / f"{SUBJECT}_{TASK}_run-{RUN}_01_raw_psd.png", dpi=180)
plt.show()

print_top_psd_peaks(raw, "raw")


# %%
# =========================
# Apply wide 0.5-70 Hz filter only
# =========================

wide = mne.filter.filter_data(
    raw,
    sfreq=fs,
    l_freq=WIDE_LOW_HZ,
    h_freq=WIDE_HIGH_HZ,
    method="fir",
    phase="zero-double",
    fir_design="firwin",
    verbose=False,
)

fig = plot_timeseries_compare([raw, wide], ["raw", f"wide {WIDE_LOW_HZ}-{WIDE_HIGH_HZ} Hz"], title="Raw vs wide bandpass")
fig.savefig(OUT_DIR / f"{SUBJECT}_{TASK}_run-{RUN}_02_raw_vs_wide_timeseries.png", dpi=180)
plt.show()

fig = plot_psd_compare([raw, wide], ["raw", "wide"], fmax=90, title="Raw vs wide bandpass PSD")
fig.savefig(OUT_DIR / f"{SUBJECT}_{TASK}_run-{RUN}_02_raw_vs_wide_psd.png", dpi=180)
plt.show()

print_top_psd_peaks(wide, "wide")


# %%
# =========================
# Apply notch after wide filter
# =========================

wide_notched = mne.filter.notch_filter(
    wide,
    Fs=fs,
    freqs=np.asarray(NOTCH_FREQS, dtype=float),
    notch_widths=NOTCH_WIDTHS,
    method="fir",
    phase="zero-double",
    fir_design="firwin",
    verbose=False,
)

fig = plot_timeseries_compare(
    [wide, wide_notched],
    ["wide", f"wide+notch {NOTCH_FREQS}"],
    title="Wide bandpass vs notch-conditioned",
)
fig.savefig(OUT_DIR / f"{SUBJECT}_{TASK}_run-{RUN}_03_wide_vs_notched_timeseries.png", dpi=180)
plt.show()

fig = plot_psd_compare([wide, wide_notched], ["wide", "wide+notch"], fmax=90, title="Wide vs notch PSD")
fig.savefig(OUT_DIR / f"{SUBJECT}_{TASK}_run-{RUN}_03_wide_vs_notched_psd.png", dpi=180)
plt.show()

print_top_psd_peaks(wide_notched, "wide_notched")


# %%
# ===============================================
# Optional: try a stronger/wider notch interactively
# ===============================================

# Edit these two values and rerun only this cell if the PSD still shows
# high-frequency narrow peaks. This does not affect the official pipeline
# until we copy the chosen settings into preprocess_fp12_one_run.py.
TRY_NOTCH_FREQS = [48.0, 64.0]
TRY_NOTCH_WIDTHS = 4.0

wide_notched_try = mne.filter.notch_filter(
    wide,
    Fs=fs,
    freqs=np.asarray(TRY_NOTCH_FREQS, dtype=float),
    notch_widths=TRY_NOTCH_WIDTHS,
    method="fir",
    phase="zero-double",
    fir_design="firwin",
    verbose=False,
)

fig = plot_psd_compare(
    [wide, wide_notched, wide_notched_try],
    ["wide", f"notch {NOTCH_WIDTHS:g} Hz width", f"try notch {TRY_NOTCH_WIDTHS:g} Hz width"],
    fmax=90,
    title="Notch-width comparison",
)
fig.savefig(OUT_DIR / f"{SUBJECT}_{TASK}_run-{RUN}_04_notch_width_compare_psd.png", dpi=180)
plt.show()

fig = plot_timeseries_compare(
    [wide, wide_notched, wide_notched_try],
    ["wide", f"notch {NOTCH_WIDTHS:g} Hz width", f"try notch {TRY_NOTCH_WIDTHS:g} Hz width"],
    title="Notch-width comparison in time domain",
)
fig.savefig(OUT_DIR / f"{SUBJECT}_{TASK}_run-{RUN}_04_notch_width_compare_timeseries.png", dpi=180)
plt.show()

print_top_psd_peaks(wide_notched_try, "wide_notched_try")


# %%
# ======================================================
# Apply official-style narrow P300 filter for comparison
# ======================================================

p300_narrow = mne.filter.filter_data(
    wide_notched,
    sfreq=fs,
    l_freq=P300_LOW_HZ,
    h_freq=P300_HIGH_HZ,
    method="fir",
    phase="zero-double",
    fir_design="firwin",
    verbose=False,
)

fig = plot_timeseries_compare(
    [raw, wide_notched, p300_narrow],
    ["raw", "wide+notch", f"P300 narrow {P300_LOW_HZ}-{P300_HIGH_HZ} Hz"],
    title="Raw vs denoiser input vs final SWLDA-style narrow signal",
)
fig.savefig(OUT_DIR / f"{SUBJECT}_{TASK}_run-{RUN}_05_raw_wide_notch_narrow_timeseries.png", dpi=180)
plt.show()

fig = plot_psd_compare(
    [raw, wide_notched, p300_narrow],
    ["raw", "wide+notch", "P300 narrow"],
    fmax=90,
    title="Raw vs denoiser input vs narrow P300 PSD",
)
fig.savefig(OUT_DIR / f"{SUBJECT}_{TASK}_run-{RUN}_05_raw_wide_notch_narrow_psd.png", dpi=180)
plt.show()


# %%
# ============================================
# Spectrogram around the selected visual window
# ============================================

start = int(round(START_SEC * fs))
stop = min(raw.shape[1], start + int(round(WINDOW_SEC * fs)))
sig_for_spec = wide_notched[0, start:stop]
f, tt, sxx = spectrogram(sig_for_spec, fs=fs, nperseg=512, noverlap=384)
keep = f <= 90

fig, ax = plt.subplots(figsize=(12, 4.5))
mesh = ax.pcolormesh(tt, f[keep], 10 * np.log10(sxx[keep] + np.finfo(float).eps), shading="auto", cmap="magma")
ax.set_title(f"{CHANNELS[0]} wide+notch spectrogram, {START_SEC:g}-{START_SEC + WINDOW_SEC:g} s")
ax.set_xlabel("Time within window (s)")
ax.set_ylabel("Frequency (Hz)")
fig.colorbar(mesh, ax=ax, label="Power (dB)")
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SUBJECT}_{TASK}_run-{RUN}_06_spectrogram_fp1_wide_notch.png", dpi=180)
plt.show()


# %%
print(f"Saved debug figures to: {OUT_DIR}")
