# RSVP/P300 BCI Validation

This workflow reproduces the FP1/FP2 preprocessing comparison and common-protocol SWLDA analysis reported in the article.

Place the Won et al. BIDS/EEGLAB dataset under `data/Won2022_BIDS` at the repository root, or provide `--dataset-root` explicitly.

```powershell
python scripts/check_dataset_files.py --dataset-root ../../data/Won2022_BIDS
python scripts/preprocess_p300_subject.py --help
python scripts/decode_p300_subject_swlda.py --help
python scripts/analyze_p300_swlda_branch_results.py --help
```

The preprocessing worker stores baseline, EOG, EMG+EOG, and full SPAR-EEG branches. Only FP1 and FP2 are denoised. The official-style narrow decoder filter is applied identically after branch generation.
