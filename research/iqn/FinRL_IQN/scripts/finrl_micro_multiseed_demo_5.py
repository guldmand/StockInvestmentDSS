"""FinRL Micro Multiseed — fast end-to-end pipeline validation.

This is NOT thesis evidence. It is a sanity check that exercises the full
multiseed pipeline (launcher -> N seeds -> suite runner -> aggregator)
in ~2-3 minutes total instead of ~58 minutes.

Configuration designed for speed (not statistical power):
  - 3 seeds (1, 2, 3) instead of 5
  - 3 agents (a2c, ppo, mvo) instead of 6
  - 500 timesteps per agent instead of 25000
  - 5 tickers instead of 10 (subset of demo_10_new)

Estimated total runtime: 2-3 minutes
  Each seed run: ~25 sec (3 agents x ~8 sec each)
  3 seeds total: ~75 sec + overhead = ~2 min

Success criteria after fix (run AFTER applying seed injection patch):
  - All 3 seeds complete with return_code 0
  - PPO action_memory hashes DIFFER across seeds (std != 0)
  - Aggregator can read all 3 runs (if we run it after)

If this script succeeds, the full multiseed run is safe to execute.

Usage:
  python scripts/finrl_micro_multiseed_demo_5.py
"""

from __future__ import annotations

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path


def find_repo_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "stock_investment_dss").exists():
            return candidate
    raise RuntimeError(f"Could not find repo root from cwd={current}")


def main() -> int:
    repo_root = find_repo_root()

    # ---- Configuration (designed for speed, NOT thesis power) ----
    micro_env = {
        "PYTHONPATH": "src",
        "PYTHONUNBUFFERED": "1",

        # Dataset (still demo_10_new but with explicit 5-ticker subset)
        "STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE": "demo_10_new",
        "STOCK_INVESTMENT_DSS_DAILY_DATASET_ID": "demo_10_new_long_2010_2026_finrl_micro",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_START": "2010-01-01",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_END": "2026-12-31",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE": "data/market/daily/imports/market_data_demo10_new_2010_2026.csv",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE": "true",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD": "false",
        "STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD": "false",

        # Subset of demo_10_new tickers (first 5: COST, AVGO, LLY, ORCL, CAT)
        "STOCK_INVESTMENT_DSS_FINRL_TICKERS": "COST,AVGO,LLY,ORCL,CAT",

        # PIT split — same as full run
        "STOCK_INVESTMENT_DSS_PIT_SPLIT_ID": "demo_10_new_long_2010_2026_finrl_micro_pit",
        "STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME": "2024-01-01",
        "STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE": "2026-12-31",

        # FinRL config — reduced for speed
        "STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS": "a2c,ppo",
        "STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO": "true",
        "STOCK_INVESTMENT_DSS_FINRL_BASELINE_TOTAL_TIMESTEPS": "500",
        "STOCK_INVESTMENT_DSS_FINRL_BASELINE_DEVICE": "auto",

        # Multiseed — only 3 seeds for speed
        "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_LIST": "1,2,3",
        "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_STOP_ON_FAILURE": "false",
        "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_RUN_SUMMARY_AFTER": "true",

        # FinRL env parameters (same as full run)
        "STOCK_INVESTMENT_DSS_FINRL_ENV_INITIAL_AMOUNT": "1000000",
        "STOCK_INVESTMENT_DSS_FINRL_ENV_HMAX": "10000",
        "STOCK_INVESTMENT_DSS_FINRL_ENV_BUY_COST_PCT": "0.001",
        "STOCK_INVESTMENT_DSS_FINRL_ENV_SELL_COST_PCT": "0.001",
        "STOCK_INVESTMENT_DSS_FINRL_ENV_REWARD_SCALING": "0.0001",
    }

    # Apply env vars
    for key, value in micro_env.items():
        os.environ[key] = value

    # Guard: verify frozen data file
    data_file = repo_root / "data/market/daily/imports/market_data_demo10_new_2010_2026.csv"
    if not data_file.exists():
        print(f"[ABORT] Frozen data file not found: {data_file}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("FinRL MICRO Multiseed Pipeline Validation")
    print("=" * 70)
    print()
    print("This is NOT thesis evidence. It validates the pipeline end-to-end")
    print("after the seed injection fix, in ~2-3 minutes instead of ~58 min.")
    print()
    print(f"Repo root:    {repo_root}")
    print(f"Python:       {sys.executable}")
    print(f"Start time:   {datetime.now():%Y-%m-%d %H:%M:%S}")
    print()
    print("Configuration:")
    print(f"  Seeds:        {micro_env['STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_LIST']}")
    print(f"  Agents:       {micro_env['STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS']} + MVO")
    print(f"  Timesteps:    {micro_env['STOCK_INVESTMENT_DSS_FINRL_BASELINE_TOTAL_TIMESTEPS']}")
    print(f"  Tickers (5):  {micro_env['STOCK_INVESTMENT_DSS_FINRL_TICKERS']}")
    print()
    print("Estimated runtime: 2-3 minutes")
    print("-" * 70)
    print(flush=True)

    # Run multiseed launcher with line-streaming output
    process = subprocess.Popen(
        [sys.executable, "-u", "-m",
         "stock_investment_dss.runner.run_finrl_baseline_multiseed_launcher"],
        cwd=str(repo_root),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopping subprocess...", flush=True)
        process.terminate()
        process.wait()
        return 130

    return_code = process.wait()
    print()
    print("=" * 70)
    print(f"Micro multiseed finished — Return code: {return_code}")
    print("=" * 70)
    print(flush=True)

    if return_code != 0:
        print("[FAIL] Pipeline failed. Do not start full multiseed.", flush=True)
        return return_code

    # Quick post-flight check: are PPO action hashes different across seeds?
    print()
    print("=" * 70)
    print("Post-flight check: PPO action hashes across seeds")
    print("=" * 70)
    print()

    runs_dir = repo_root / "outputs" / "runs"
    # Find latest launcher and its 3 seed children
    launcher_runs = sorted(
        (d for d in runs_dir.iterdir()
         if d.is_dir() and "finrl_baseline_multiseed_launcher" in d.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not launcher_runs:
        print("[WARN] No launcher run found")
        return 0

    latest_launcher = launcher_runs[0]
    print(f"Latest launcher: {latest_launcher.name}")
    launcher_start_time = latest_launcher.stat().st_mtime

    # Find 3 suite runs spawned by this launcher (created after launcher start)
    suite_runs = sorted(
        (d for d in runs_dir.iterdir()
         if d.is_dir()
         and "finrl_baseline_suite_smoke_test" in d.name
         and d.stat().st_mtime > launcher_start_time),
        key=lambda p: p.stat().st_mtime,
    )

    print(f"Suite runs spawned: {len(suite_runs)}")

    if len(suite_runs) < 2:
        print("[WARN] Need at least 2 suite runs to compare hashes")
        return 0

    # Compare PPO action hashes
    import hashlib
    ppo_hashes = []
    for i, suite_run in enumerate(suite_runs, 1):
        ppo_actions = (
            suite_run / "data" / "finrl_baseline_suite" / "ppo" / "ppo_action_memory.csv"
        )
        if not ppo_actions.exists():
            print(f"  Seed {i}: PPO actions not found at {ppo_actions}")
            continue
        h = hashlib.sha256(ppo_actions.read_bytes()).hexdigest()
        ppo_hashes.append((i, h, suite_run.name))
        print(f"  Seed {i}: PPO hash {h[:16]}... ({suite_run.name})")

    print()
    unique_hashes = set(h for _, h, _ in ppo_hashes)
    if len(unique_hashes) == len(ppo_hashes):
        print(f"[OK] All {len(ppo_hashes)} PPO hashes are UNIQUE — seed injection works!")
        print("     Safe to launch full multiseed run.")
    elif len(unique_hashes) > 1:
        print(f"[PARTIAL] {len(unique_hashes)}/{len(ppo_hashes)} unique hashes")
        print("          Some seeds produced identical outputs — investigate.")
    else:
        print(f"[FAIL] All {len(ppo_hashes)} PPO hashes are IDENTICAL")
        print("       Seed injection still broken. Do NOT launch full multiseed.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
