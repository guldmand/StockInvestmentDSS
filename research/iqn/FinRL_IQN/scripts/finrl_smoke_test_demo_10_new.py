# ---------------------------------------------------------------------
# FinRL Baseline Suite Smoke Test on demo_10_new
#
# Cross-platform notebook cell (Windows / Mac / Linux).
# Verifies the V2 FinRL pipeline can read demo_10_new dataset and writes
# outputs to the V2 canonical pattern (outputs/runs/) under the
# post-Plan-2 architecture.
#
# Reduced configuration (1 seed, 500 timesteps, all 6 agents):
# - estimated runtime 5-10 minutes
# - purpose: validate pipeline before launching the full multiseed run
# ---------------------------------------------------------------------

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path.cwd()

assert (REPO_ROOT / "src" / "stock_investment_dss").exists(), (
    f"Notebook must be run from repo root. Current cwd: {REPO_ROOT}"
)

# ---------------------------------------------------------------------
# Environment variables for demo_10_new dataset
# Matches the dataset specification used by clean_25k_baseline_v1.json
# ---------------------------------------------------------------------

smoke_test_env = {
    # PYTHONPATH for `python -m stock_investment_dss...`
    "PYTHONPATH": "src",

    # Dataset specification (demo_10_new, 10-ticker universe, full history)
    "STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE": "demo_10_new",
    "STOCK_INVESTMENT_DSS_DAILY_DATASET_ID": "demo_10_new_long_2010_2026_finrl_smoke",
    "STOCK_INVESTMENT_DSS_DAILY_DATA_START": "2010-01-01",
    "STOCK_INVESTMENT_DSS_DAILY_DATA_END": "2026-12-31",
    "STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE": "data/market/daily/imports/market_data_demo10_new_2010_2026.csv",
    "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE": "true",
    "STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD": "false",
    "STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD": "false",

    # Point-in-time split (eval window 2024-01-01 to 2026-12-31)
    "STOCK_INVESTMENT_DSS_PIT_SPLIT_ID": "demo_10_new_long_2010_2026_finrl_smoke_pit",
    "STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME": "2024-01-01",
    "STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE": "2026-12-31",

    # FinRL config — smoke test sized (500 steps, not 25000)
    "STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS": "a2c,ddpg,td3,ppo,sac,mvo",
    "STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO": "true",
    "STOCK_INVESTMENT_DSS_FINRL_BASELINE_TOTAL_TIMESTEPS": "500",
    "STOCK_INVESTMENT_DSS_FINRL_BASELINE_DEVICE": "auto",

    # FinRL environment parameters
    "STOCK_INVESTMENT_DSS_FINRL_ENV_INITIAL_AMOUNT": "1000000",
    "STOCK_INVESTMENT_DSS_FINRL_ENV_HMAX": "10000",
    "STOCK_INVESTMENT_DSS_FINRL_ENV_BUY_COST_PCT": "0.001",
    "STOCK_INVESTMENT_DSS_FINRL_ENV_SELL_COST_PCT": "0.001",
    "STOCK_INVESTMENT_DSS_FINRL_ENV_REWARD_SCALING": "0.0001",
}

# Apply env vars to current process environment
for key, value in smoke_test_env.items():
    os.environ[key] = value

print("=" * 60)
print("FinRL Baseline Suite Smoke Test — demo_10_new")
print("=" * 60)
print()
print(f"Repo root:    {REPO_ROOT}")
print(f"Python:       {sys.executable}")
print(f"Start time:   {datetime.now():%Y-%m-%d %H:%M:%S}")
print()
print("Environment variables applied:")
for key, value in smoke_test_env.items():
    print(f"  {key} = {value}")
print()
print("Starting smoke test (1 seed, 500 steps, 6 agents)...")
print("Estimated duration: 5-10 minutes")
print("-" * 60)

# Run smoke test as subprocess (works on Windows/Mac/Linux)
start_time = time.time()

result = subprocess.run(
    [sys.executable, "-u", "-m", "stock_investment_dss.runner.run_finrl_baseline_suite_smoke_test"],
    cwd=str(REPO_ROOT),
    env=os.environ.copy(),
)

duration_seconds = time.time() - start_time
duration_minutes = duration_seconds / 60.0

print()
print("=" * 60)
print(f"Smoke test finished — Duration: {duration_minutes:.1f} min ({duration_seconds:.0f} sec)")
print(f"Return code: {result.returncode}")
print("=" * 60)
print()

# Locate the latest run directory
runs_dir = REPO_ROOT / "outputs" / "runs"
latest_run = None
if runs_dir.exists():
    candidates = [
        d for d in runs_dir.iterdir()
        if d.is_dir() and "finrl_baseline_suite_smoke_test" in d.name
    ]
    if candidates:
        latest_run = max(candidates, key=lambda p: p.stat().st_mtime)

if latest_run is not None:
    print(f"Latest smoke test run: {latest_run.name}")
    print()

    # Inspect summary.json
    summary_json = latest_run / "summary" / "finrl_baseline_suite_smoke_summary.json"
    if summary_json.exists():
        print("--- Summary JSON (first 60 lines) ---")
        lines = summary_json.read_text(encoding="utf-8").splitlines()
        for line in lines[:60]:
            print(line)
        print()

    # Inspect comparison snapshot CSV
    comp_csv = latest_run / "summary" / "finrl_baseline_suite_comparison_snapshot.csv"
    if comp_csv.exists():
        print("--- Comparison snapshot (first 10 lines) ---")
        lines = comp_csv.read_text(encoding="utf-8").splitlines()
        for line in lines[:10]:
            print(line)
        print()

    # Show directory structure
    print("--- Run directory structure ---")
    for subdir in sorted(latest_run.iterdir()):
        if subdir.is_dir():
            count = sum(1 for _ in subdir.rglob("*") if _.is_file())
            if count > 0:
                print(f"  {subdir.name}/: {count} files")
else:
    print("ERROR: No FinRL suite smoke test run found in outputs/runs/")
