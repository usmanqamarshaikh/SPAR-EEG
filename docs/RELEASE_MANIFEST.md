# Release Manifest

This manifest maps the accepted article's availability statement to public repository content.

| Promised material | Repository location |
|---|---|
| Frozen SPAR-EEG pass implementation | `matlab/+burstdenoise/` |
| EEGLAB and vector deployment demonstrations | `demos/` |
| EEGdenoiseNet and PhysioBank comparison workflows | `benchmarks/ground_truth/` |
| Parameter-sensitivity analyses | `benchmarks/ground_truth/scripts/run_*sensitivity*.py` and paired analysis scripts |
| Artifact-geometry and occupancy analyses | `benchmarks/ground_truth/scripts/*artifact_geometry*` and `*artifact_occupancy*` |
| Computation-cost analysis | `benchmarks/ground_truth/scripts/benchmark_computation_cost_*.py` |
| EEGLAB ERP-preservation diagnostic | `validation/erp_preservation/` |
| RSVP/P300 BCI validation | `validation/p300_bci/` |
| Dataset acquisition and placement | `docs/DATASETS.md` |
| End-to-end workflow guidance | `docs/REPRODUCIBILITY.md` |
| Frozen-code checksums and provenance | `docs/PROVENANCE.md` |
| Verified software environment | `docs/TESTED_ENVIRONMENT.md` |

The exercise EEG dataset is not publicly distributed. The DOI and final IEEE citation should be added after publication.
