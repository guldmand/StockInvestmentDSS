"""
FinRL Baseline Multiseed Run on demo_10_new

Cross-platform multiseed run for the V2 FinRL pipeline. Produces
thesis-grade FinRL parametric RL baseline evidence on demo_10_new
(same dataset as etape 3 algorithmic baselines and IQN clean_25k baseline).

Full configuration:
  - 5 seeds (1, 2, 3, 4, 5) - matches V2 IQN convention
  - 25,000 timesteps per agent (matches IQN clean_25k_baseline_v1)
  - All 6 agents (A2C, DDPG, PPO, TD3, SAC, MVO)

Estimated runtime: 60-90 minutes (based on smoke test extrapolation:
  smoke test = 22 sec for 500 steps × 6 agents × 1 seed
  full run = 22 × 50 × 5 = ~92 minutes)

After completion, the launcher automatically invokes
run_finrl_baseline_multiseed_summary, producing:
  - finrl_baseline_multiseed_aggregate_by_agent.csv
  - finrl_baseline_multiseed_aggregate_by_strategy.csv
  - finrl_baseline_multiseed_summary.json
  - 4 mean±std plots (total_return, max_drawdown, sharpe, cvar)

Usage:
  - From repo root:    python scripts/finrl_multiseed_demo_10_new.py
  - From notebook:     !python -u scripts/finrl_multiseed_demo_10_new.py
                       (or via subprocess.run)

Cancel-safe: Each seed runs as a separate subprocess. Stopping with Ctrl+C
between seeds preserves all completed seed runs in outputs/runs/.
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
    # Environment variables for demo_10_new dataset (FULL multiseed run)
    # Matches the dataset specification used by clean_25k_baseline_v1.json
    # -------------------------------------------------------------------
    multiseed_env = {
        # PYTHONPATH for `python -m stock_investment_dss...`
        "PYTHONPATH": "src",

        # Dataset specification (demo_10_new, 10-ticker universe, full history)
        "STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE": "demo_10_new",
        "STOCK_INVESTMENT_DSS_DAILY_DATASET_ID": "demo_10_new_long_2010_2026_finrl_multiseed",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_START": "2010-01-01",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_END": "2026-12-31",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE": "data/market/daily/imports/market_data_demo10_new_2010_2026.csv",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE": "true",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD": "false",
        "STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD": "false",

        # Explicit ticker list for demo_10_new
        # Same 10 tickers as clean_25k_baseline_v1.json
        "STOCK_INVESTMENT_DSS_FINRL_TICKERS": "COST,AVGO,LLY,ORCL,CAT,BA,KO,MCD,WMT,PG",

        # Point-in-time split (eval window 2024-01-01 to 2026-12-31)
        "STOCK_INVESTMENT_DSS_PIT_SPLIT_ID": "demo_10_new_long_2010_2026_finrl_multiseed_pit",
        "STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME": "2024-01-01",
        "STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE": "2026-12-31",

        # FinRL config — FULL multiseed (25000 steps, matching IQN clean_25k)
        "STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS": "a2c,ddpg,td3,ppo,sac",
        "STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO": "true",
        "STOCK_INVESTMENT_DSS_FINRL_BASELINE_TOTAL_TIMESTEPS": "25000",
        "STOCK_INVESTMENT_DSS_FINRL_BASELINE_DEVICE": "auto",

        # Multiseed config
        "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_LIST": "1,2,3,4,5",
        "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_STOP_ON_FAILURE": "false",
        "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_RUN_SUMMARY_AFTER": "true",

        # FinRL environment parameters
        "STOCK_INVESTMENT_DSS_FINRL_ENV_INITIAL_AMOUNT": "1000000",
        "STOCK_INVESTMENT_DSS_FINRL_ENV_HMAX": "10000",
        "STOCK_INVESTMENT_DSS_FINRL_ENV_BUY_COST_PCT": "0.001",
        "STOCK_INVESTMENT_DSS_FINRL_ENV_SELL_COST_PCT": "0.001",
        "STOCK_INVESTMENT_DSS_FINRL_ENV_REWARD_SCALING": "0.0001",
    }

    # Apply env vars to current process environment
    for key, value in multiseed_env.items():
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
    print("=" * 70)
    print("FinRL Baseline MULTISEED Run - demo_10_new")
    print("=" * 70)
    print()
    print(f"Repo root:    {repo_root}")
    print(f"Python:       {sys.executable}")
    print(f"Start time:   {datetime.now():%Y-%m-%d %H:%M:%S}")
    print()
    print("Configuration:")
    print(f"  Seeds:           {multiseed_env['STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_LIST']}")
    print(f"  Agents:          {multiseed_env['STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS']}")
    print(f"  Include MVO:     {multiseed_env['STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO']}")
    print(f"  Timesteps:       {multiseed_env['STOCK_INVESTMENT_DSS_FINRL_BASELINE_TOTAL_TIMESTEPS']}")
    print(f"  Tickers:         {multiseed_env['STOCK_INVESTMENT_DSS_FINRL_TICKERS']}")
    print(f"  PIT date:        {multiseed_env['STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME']}")
    print(f"  Trade end:       {multiseed_env['STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE']}")
    print(f"  Run summary:     {multiseed_env['STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_RUN_SUMMARY_AFTER']}")
    print()
    print("Estimated runtime: 60-90 minutes (5 seeds × 25k steps × 6 agents)")
    print("Cancel-safe: completed seeds are preserved on Ctrl+C")
    print("-" * 70)
    print()

    # -------------------------------------------------------------------
    # Run multiseed launcher (cross-platform via subprocess)
    # The launcher invokes the suite smoke test runner once per seed,
    # then runs the multiseed_summary aggregator at the end.
    # -------------------------------------------------------------------
    start_time = time.time()

    result = subprocess.run(
        [sys.executable, "-u", "-m",
         "stock_investment_dss.runner.run_finrl_baseline_multiseed_launcher"],
        cwd=str(repo_root),
        env=os.environ.copy(),
    )

    duration_seconds = time.time() - start_time
    duration_minutes = duration_seconds / 60.0

    print()
    print("=" * 70)
    print(f"Multiseed run finished - Duration: {duration_minutes:.1f} min "
          f"({duration_seconds:.0f} sec)")
    print(f"Return code: {result.returncode}")
    print("=" * 70)
    print()

    # -------------------------------------------------------------------
    # Inspect output — show the multiseed summary aggregate
    # -------------------------------------------------------------------
    runs_dir = repo_root / "outputs" / "runs"

    # Find latest multiseed launcher run
    launcher_runs = [
        d for d in runs_dir.iterdir()
        if d.is_dir() and "finrl_baseline_multiseed_launcher" in d.name
    ]
    latest_launcher = max(launcher_runs, key=lambda p: p.stat().st_mtime) if launcher_runs else None

    # Find latest multiseed summary run
    summary_runs = [
        d for d in runs_dir.iterdir()
        if d.is_dir() and "finrl_baseline_multiseed_summary" in d.name
    ]
    latest_summary = max(summary_runs, key=lambda p: p.stat().st_mtime) if summary_runs else None

    if latest_launcher:
        print(f"Latest multiseed launcher run: {latest_launcher.name}")
        launcher_summary = latest_launcher / "summary" / "finrl_baseline_multiseed_launcher_summary.json"
        if launcher_summary.exists():
            print("--- Launcher summary (first 30 lines) ---")
            for line in launcher_summary.read_text(encoding="utf-8").splitlines()[:30]:
                print(line)
            print()

    if latest_summary:
        print(f"Latest multiseed summary run: {latest_summary.name}")
        print()

        # Show aggregate_by_agent CSV
        agg_csv = latest_summary / "summary" / "finrl_baseline_multiseed_aggregate_by_agent.csv"
        if agg_csv.exists():
            print("--- Aggregate by agent (mean ± std across 5 seeds) ---")
            lines = agg_csv.read_text(encoding="utf-8").splitlines()
            for line in lines[:8]:  # Header + 6 agent rows
                print(line)
            print()

        # Show summary plots present
        plots = list(latest_summary.glob("summary/*.png"))
        if plots:
            print("--- Generated plots ---")
            for plot in sorted(plots):
                print(f"  {plot.name}")
            print()

        # Show directory structure
        print("--- Multiseed summary directory structure ---")
        for subdir in sorted(latest_summary.iterdir()):
            if subdir.is_dir():
                count = sum(1 for _ in subdir.rglob("*") if _.is_file())
                if count > 0:
                    print(f"  {subdir.name}/: {count} files")
    else:
        print("WARNING: No multiseed summary run found in outputs/runs/")
        print("The launcher may have failed before the summary step.")

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
