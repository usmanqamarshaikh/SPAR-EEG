# Tested Environment

The dataset-free release checks passed with the following Python environment:

| Component | Version |
|---|---:|
| Python | 3.11.9 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| pandas | 2.3.3 |
| h5py | 3.16.0 |
| Matplotlib | 3.11.2 |
| PyWavelets | 1.8.0 |
| scikit-learn | 1.9.1 |
| statsmodels | 0.15.0 |
| MNE-Python | 1.13.2 |
| WFDB | 4.3.1 |

The MATLAB smoke test passed with MATLAB R2023b and exercised each standalone
pass, the complete pass sequence, and the adaptive continuous-signal wrapper.
The MATLAB implementation also requires the Signal Processing Toolbox. The
adaptive percentile rule uses functionality from the Statistics and Machine
Learning Toolbox.

Run the same release checks locally with:

```powershell
.\.venv\Scripts\python.exe tests\smoke_python.py
.\.venv\Scripts\python.exe tests\test_core_python.py
.\.venv\Scripts\python.exe tests\check_release.py
matlab -batch "run('tests/smoke_matlab.m')"
```
