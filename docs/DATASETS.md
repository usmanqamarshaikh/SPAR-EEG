# Dataset Setup

No EEG dataset is redistributed with SPAR-EEG. Download each public dataset from its original source and retain its license and citation information.

## EEGdenoiseNet

Place the clean EEG, EOG, and EMG source arrays under `data/eegdenoisenet/`. The ground-truth preparation script creates centered artifact mixtures and H5 files used by the comparison workflows.

Run from `benchmarks/ground_truth`:

```powershell
python scripts/prepare_datasets_centered.py --help
```

Use explicit input and output paths when preparing the full dataset. The generated H5 files should be placed in `data/prepared_h5/` or supplied through each script's `--data-dir` option.

## PhysioBank Motion Artifact Dataset

Download the public motion-artifact recordings from PhysioNet/PhysioBank and place them under `data/physiobank/`. The preparation workflow records the contaminated signal, reference signal, sampling frequency, and artifact mask in the common H5 layout.

## Won et al. RSVP/P300 Dataset

Official repository:

https://github.com/Kyungho-Won/EEG-dataset-for-RSVP-P300-speller

Place the BIDS/EEGLAB dataset at:

```text
data/Won2022_BIDS/
```

The validation scripts accept `--dataset-root`, so another location may be used without editing source code.

## EEGLAB Sample ERP Recording

Obtain the EEGLAB sample recording and prepare the independently cleaned `EEG` structure described in the article and supplementary material. Place the MAT file at:

```text
data/eeglab_sample/processedEEG.mat
```

The exercise EEG dataset is not publicly released and is not required for the public benchmark, sensitivity, geometry, P300, or ERP-preservation workflows.
