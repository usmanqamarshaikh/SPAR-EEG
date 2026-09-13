# Implementation Provenance

The three files under `matlab/+burstdenoise/` are the frozen production pass
implementations used throughout the reported full evaluations. The public API
and adaptive wrapper call these files without replacing their internal
defaults. Sensitivity scripts alter only the parameter named by the relevant
experiment.

The release files were compared byte-for-byte with the authors' installed
MATLAB package before publication of this repository. Their SHA-256 checksums
are:

| File | SHA-256 |
|---|---|
| `denoise_emg_vmd_window.m` | `53B95CFD5E820E63491AF7B3E58FE0D0B1A70EC2998C6A29F49C5398985A33E0` |
| `denoise_eog_ssa_window.m` | `2C288D6F04AEEE32161BD343EAE2EE882C5A9B35926693F9370E88FBD4F0BFA1` |
| `denoise_slow_ssa_window.m` | `45E9CA3872609D3EBECA9D0525BF2B4B2BD2977E2182ABB570F24B4A10C8EDA6` |

These checksums allow users to verify that the scientific implementation has
not changed. Future modifications should receive a new software version and
updated checksums.

No private exercise EEG or third-party public dataset is included in this
repository. Dataset acquisition and placement are described in `DATASETS.md`.
