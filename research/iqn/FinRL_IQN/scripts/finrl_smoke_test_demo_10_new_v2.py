"""
FinRL Baseline Suite Smoke Test on demo_10_new

Cross-platform smoke test for the V2 FinRL pipeline. Verifies that the
pipeline can read the demo_10_new dataset and writes outputs to the V2
canonical pattern (outputs/runs/) under the post-Plan-2 architecture.

Reduced configuration:
  - 1 seed (set_global_seed=42 internally)
  - 500 timesteps per agent (vs. 25000 for full run)
  - All 6 agents (A2C, DDPG, PPO, TD3, SAC, MVO)

Estimated runtime: 5-10 minutes
Purpose: Validate pipeline before launching the full multiseed run

Usage:
  - From repo root:    python scripts/finrl_smoke_test_demo_10_new.py
  - From notebook:     paste this file's contents into a cell

Note on universe_id 'demo_10_new':
  The FinRL pipeline does not have 'demo_10_new' as a predefined universe.
  Available predefined: demo_2, demo_5, demo_10, demo_30.
  The 'demo_10_new' universe is specified via the explicit ticker list
  environment variable STOCK_INVESTMENT_DSS_FINRL_TICKERS, which takes
  priority over the universe_id lookup. The 10 tickers match those used
  by clean_25k_baseline_v1.json: COST, AVGO, LLY, ORCL, CAT, BA, KO,
  MCD, WMT, PG.
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path


def find_repo_root() -> Path:
    """Locate the repository root by looking for src/stock_investment_dss/."""
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "stock_investment_dss").exists():
            return candidate
    raise RuntimeError(
        f"Could not find repo root from cwd={current}. "
        "Run this script from inside the FinRL_IQN repository."
    )


def main() -> int:
    repo_root = find_repo_root()

    # -------------------------------------------------------------------
    # Environment variables for demo_10_new dataset
    # Matches the dataset specification used by clean_25k_baseline_v1.json
    # -------------------------------------------------------------------
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

        # Explicit ticker list for demo_10_new
        # (overrides universe_id lookup since 'demo_10_new' is not in PREDEFINED_UNIVERSES)
        # Same 10 tickers as clean_25k_baseline_v1.json
        "STOCK_INVESTMENT_DSS_FINRL_TICKERS": "COST,AVGO,LLY,ORCL,CAT,BA,KO,MCD,WMT,PG",

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

    # -------------------------------------------------------------------
    # Guard: verify frozen data file exists
    # -------------------------------------------------------------------
    data_file = repo_root / "data/market/daily/imports/market_data_demo10_new_2010_2026.csv"
    if not data_file.exists():
        print(f"[ABORT] Frozen data file not found: {data_file}", file=sys.stderr)
        return 1

    print("[OK] Frozen data file found:", data_file.relative_to(repo_root))

    # -------------------------------------------------------------------
    # Print configuration
    # -------------------------------------------------------------------
    print()
    print("=" * 60)
    print("FinRL Baseline Suite Smoke Test - demo_10_new")
    print("=" * 60)
    print()
    print(f"Repo root:    {repo_root}")
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

    # -------------------------------------------------------------------
    # Run smoke test (cross-platform via subprocess)
    # -------------------------------------------------------------------
    start_time = time.time()

    result = subprocess.run(
        [sys.executable, "-u", "-m",
         "stock_investment_dss.runner.run_finrl_baseline_suite_smoke_test"],
        cwd=str(repo_root),
        env=os.environ.copy(),
    )

    duration_seconds = time.time() - start_time
    duration_minutes = duration_seconds / 60.0

    print()
    print("=" * 60)
    print(f"Smoke test finished - Duration: {duration_minutes:.1f} min "
          f"({duration_seconds:.0f} sec)")
    print(f"Return code: {result.returncode}")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------
    # Inspect output
    # -------------------------------------------------------------------
    runs_dir = repo_root / "outputs" / "runs"
    latest_run = None
    if runs_dir.exists():
        candidates = [
            d for d in runs_dir.iterdir()
            if d.is_dir() and "finrl_baseline_suite_smoke_test" in d.name
        ]
        if candidates:
            latest_run = max(candidates, key=lambda p: p.stat().st_mtime)

    if latest_run is None:
        print("ERROR: No FinRL suite smoke test run found in outputs/runs/")
        return 1

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

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
