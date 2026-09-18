# SPAR-EEG

**Selective Pass-Wise Artifact Reduction for Wearable Single-Channel EEG Denoising**

SPAR-EEG is a self-contained MATLAB pipeline for selective attenuation of EMG bursts, blink-like ocular transients, and slow drift artifacts in single-channel EEG. It combines an EMG-oriented variational mode decomposition (VMD) pass with two singular spectrum analysis (SSA) passes for ocular transients and slow drift. Artifact-dominant component regions are attenuated rather than rejecting complete components.

This repository accompanies the IEEE Transactions on Neural Systems and Rehabilitation Engineering article:

> U. Q. Shaikh, A. M. Kalra, A. Lowe, and I. K. Niazi, "SPAR-EEG: Selective Pass-Wise Artifact Reduction for Wearable Single-Channel EEG Denoising," IEEE Transactions on Neural Systems and Rehabilitation Engineering, Early Access, 16 September 2026, doi: [10.1109/TNSRE.2026.3734253](https://doi.org/10.1109/TNSRE.2026.3734253).

The open-access article is available on [IEEE Xplore](https://ieeexplore.ieee.org/document/11693065). Volume and page details will be added when assigned.

The released MATLAB pass files are the frozen production implementations used
for the reported analyses. Their provenance and SHA-256 checksums are recorded
in [docs/PROVENANCE.md](docs/PROVENANCE.md).

## Repository Contents

```text
matlab/+burstdenoise/          Frozen production EMG, EOG, and slow passes
matlab/spar_eeg_denoise.m     Fixed-window public API
matlab/spar_eeg_adaptive.m    Adaptive 2-4 s continuous-signal wrapper
demos/                        MATLAB vector and EEGLAB demonstrations
benchmarks/ground_truth/      EEGdenoiseNet/PhysioBank comparisons and analyses
validation/p300_bci/          RSVP/P300 preprocessing and SWLDA validation
validation/erp_preservation/  EEGLAB ERP-preservation diagnostic
docs/                         Dataset and reproduction instructions
tests/                        Dataset-free smoke tests
```

The private dry-electrode exercise EEG dataset is not distributed. Its reported analysis is therefore not presented as a fully reproducible public-data workflow.

## Quick Start

### MATLAB

Requirements:

- MATLAB R2023b or later
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox for the adaptive percentile rule

Add the repository MATLAB folder to the path:

```matlab
addpath(fullfile(pwd, 'matlab'));
```

Denoise one already-epoched signal using the complete sequence:

```matlab
[cleaned, debug] = spar_eeg_denoise(noisyEEG, Fs, {'emg','eog','slow'});
```

Process a continuous signal using adaptive 2-4 s segmentation:

```matlab
[cleaned, meta] = spar_eeg_adaptive(noisyEEG, Fs, {'emg','eog','slow'});
```

Run the dataset-free demonstration:

```matlab
run(fullfile('demos', 'demo_synthetic_signal.m'));
```

### Python workflows

Python 3.11 was used for the reported analyses.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tests\smoke_python.py
```

Dataset downloads and complete reproduction commands are documented in [docs/DATASETS.md](docs/DATASETS.md) and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
The package versions used for release verification are listed in
[docs/TESTED_ENVIRONMENT.md](docs/TESTED_ENVIRONMENT.md).

## Method Defaults

The frozen pass defaults are defined directly in the three files under `matlab/+burstdenoise`. The reported experiments used these internal defaults unless a sensitivity experiment explicitly changed one parameter. The default pass order is:

```text
EMG -> EOG -> slow
```

The passes can also be called independently or in a user-selected sequence.

## Third-Party Benchmark Methods

The benchmark folder contains research code for reproducing comparisons with wavelet thresholding, WQN, EMD-ICA, and EMD-CCA. WQN has upstream research-use and patent conditions and is not covered by the SPAR-EEG software license. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before running or redistributing those components.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). Please cite the associated article when using SPAR-EEG:

```bibtex
@article{shaikh2026spareeg,
  author = {Shaikh, Usman Qamar and Kalra, Anubha Manju and Lowe, Andrew and Niazi, Imran Khan},
  title = {{SPAR-EEG}: Selective Pass-Wise Artifact Reduction for Wearable Single-Channel {EEG} Denoising},
  journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  year = {2026},
  month = sep,
  note = {Early Access},
  doi = {10.1109/TNSRE.2026.3734253},
  url = {https://ieeexplore.ieee.org/document/11693065}
}
```

## Contact

- Scientific correspondence: Imran Khan Niazi, `imran.niazi@nzchiro.co.nz`
- Repository and implementation inquiries: Usman Qamar Shaikh, `usman.shaikh@autuni.ac.nz`
