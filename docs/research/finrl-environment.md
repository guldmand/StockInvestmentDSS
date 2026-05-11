# FinRL Environment

## Purpose

Verify that FinRL can be installed and imported in the project environment without blocking the local PoC application.

This is a smoke-test task only. Full RL training, hyperparameter tuning and model registry integration belong to later tasks.

## Environment

- Conda env: `stockdss`
- Python: `3.11`
- Initial machine: Windows AMD workstation
- GPU: AMD Radeon RX 7900 XTX
- PoC mode: CPU-first

## Install commands

Run from the project root after activating the environment:

```powershell
conda activate stockdss
conda install -c conda-forge yfinance matplotlib scikit-learn jupyterlab ipykernel -y
pip install gymnasium stable-baselines3
pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
```

If FinRL installation fails, keep the exact error output and document it under "Known issues".

## Verification script

Use:

```powershell
python research/experiments/finrl_baseline/smoke_test.py
```

The script should print:

- Python version
- platform
- package versions
- import status for core dependencies
- whether PyTorch sees CUDA/GPU
- FinRL import status

## Result

Paste the smoke-test output here after running it.

```txt
Python: 3.11.15
Platform: Windows-10-10.0.26200-SP0

duckdb: 1.5.2
numpy: 2.4.3
pandas: 2.2.3
yfinance: 0.2.57
gymnasium: 1.2.3
stable-baselines3: 2.8.0
torch: 2.10.0
finrl: 0.3.8

Import checks:
duckdb import: OK
yfinance import: OK
gymnasium import: OK
stable_baselines3 import: OK
torch import: OK
torch cuda available: False
finrl import: OK
```

## Known issues

```txt
Initial pip-based torch 2.11.0+cpu installation failed on Windows with WinError 127 when loading torch/lib/shm.dll.

Resolved by uninstalling pip torch/stable-baselines3 and installing PyTorch CPU + stable-baselines3 from conda-forge:

conda install -c conda-forge pytorch-cpu stable-baselines3 -y

FinRL was then installed with pip from the GitHub repository.
```

## Notes

- FinRL belongs primarily to the research/slow-layer pipeline.
- The local PoC application must not depend on live FinRL training.
- AMD GPU acceleration is not required for this task.
- CPU-first verification is enough for this issue.
- Heavier training can later run on the NVIDIA GPU machine, cloud GPU or scheduled infrastructure.