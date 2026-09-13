# EEGLAB ERP-Preservation Diagnostic

This workflow applies the frozen SPAR-EEG branches to an independently pre-cleaned EEGLAB sample recording and compares ERP amplitude, standardized measurement error, SNR retention, and signed-SNR scalp patterns.

Place `processedEEG.mat`, containing an `EEG` structure, under `data/eeglab_sample` at the repository root or provide `--input-file`.

```powershell
python scripts/run_spar_erp_preservation.py --stage all --overwrite
python scripts/prepare_erp_preservation_revision_materials.py --help
```

The workflow saves generated branch data and analysis outputs under the ignored `results/` directory.
