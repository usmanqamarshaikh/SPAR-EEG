# Release Manifest

This manifest maps the published article's availability statement to public repository content.

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

The exercise EEG dataset is not publicly distributed. The article was published
in IEEE TNSRE Early Access on 16 September 2026 with DOI
[10.1109/TNSRE.2026.3734253](https://doi.org/10.1109/TNSRE.2026.3734253).
Volume and page details will be added when assigned.
