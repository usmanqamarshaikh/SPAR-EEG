# Ground-Truth Benchmark

This folder contains the common H5 data model, baseline implementations, MATLAB launchers, metric computation, statistical analysis, ablation, sensitivity, artifact-geometry, adaptive-window, and computation-cost workflows used in the article.

Run commands from this directory with the repository environment:

```powershell
..\..\.venv\Scripts\python.exe scripts\smoke_test.py --skip-matlab
```

Prepared H5 files are not distributed. Supply their location with `--data-dir`. SPAR-EEG outputs and SOTA outputs retain the original channel-by-sample layout, and all paired metrics are computed by the common Python implementation in `eeg_eval/metrics.py`.

The WQN, EMD-ICA, and EMD-CCA benchmark branches use artifact masks and/or clean references as required by the reproduced comparison protocol. SPAR-EEG does not receive those oracle inputs. See `THIRD_PARTY_NOTICES.md` at the repository root.

Start with:

```powershell
python scripts/prepare_datasets_centered.py --help
python scripts/run_methods_to_h5.py --help
python scripts/run_proposed_matlab.py --help
python scripts/metrics_from_restored_h5.py --help
```

Full workflow mapping is provided in `docs/REPRODUCIBILITY.md`.
