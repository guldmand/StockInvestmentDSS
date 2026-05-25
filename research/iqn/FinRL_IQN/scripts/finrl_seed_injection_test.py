"""Test seed injection fix for FinRL suite runner.

Runs the suite smoke test TWICE with different seeds (500 steps each)
and verifies that PPO action_memory.csv hashes are DIFFERENT.

This is the verification test BEFORE we burn another 58 min on multiseed.

Expected outcome AFTER fix:
  - Test 1 (seed=1): produces hash A
  - Test 2 (seed=42): produces hash B
  - hash A != hash B  -> fix works

If hashes are identical, fix is NOT working and patch must be re-checked.

Usage:
  python scripts/finrl_seed_injection_test.py
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def find_repo_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "stock_investment_dss").exists():
            return candidate
    raise RuntimeError(f"Could not find repo root from cwd={current}")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_latest_suite_run(project_root: Path) -> Path | None:
    runs_dir = project_root / "outputs" / "runs"
    candidates = [
        d for d in runs_dir.iterdir()
        if d.is_dir() and "finrl_baseline_suite_smoke_test" in d.name
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def run_one_seed(repo_root: Path, seed: int) -> Path:
    """Run suite smoke test with a specific seed, return ppo action_memory path."""
    print(f"\n{'=' * 60}")
    print(f"Running suite smoke test with seed={seed} (500 steps)")
    print('=' * 60)

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": "src",
        # Inject seed via all 3 names launcher uses
        "STOCK_INVESTMENT_DSS_RANDOM_SEED": str(seed),
        "STOCK_INVESTMENT_DSS_FINRL_SEED": str(seed),
        "STOCK_INVESTMENT_DSS_SB3_SEED": str(seed),

        # Dataset (demo_10_new)
        "STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE": "demo_10_new",
        "STOCK_INVESTMENT_DSS_DAILY_DATASET_ID": "demo_10_new_long_2010_2026_finrl_seed_test",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_START": "2010-01-01",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_END": "2026-12-31",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE": "data/market/daily/imports/market_data_demo10_new_2010_2026.csv",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE": "true",
        "STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD": "false",
        "STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD": "false",
        "STOCK_INVESTMENT_DSS_FINRL_TICKERS": "COST,AVGO,LLY,ORCL,CAT,BA,KO,MCD,WMT,PG",
        "STOCK_INVESTMENT_DSS_PIT_SPLIT_ID": "demo_10_new_long_2010_2026_finrl_seed_test_pit",
        "STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME": "2024-01-01",
        "STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE": "2026-12-31",

        # Super-fast smoke test config
        "STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS": "ppo",
        "STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO": "false",
        "STOCK_INVESTMENT_DSS_FINRL_BASELINE_TOTAL_TIMESTEPS": "500",
        "STOCK_INVESTMENT_DSS_FINRL_BASELINE_DEVICE": "auto",

        "STOCK_INVESTMENT_DSS_FINRL_ENV_INITIAL_AMOUNT": "1000000",
        "STOCK_INVESTMENT_DSS_FINRL_ENV_HMAX": "10000",
        "STOCK_INVESTMENT_DSS_FINRL_ENV_BUY_COST_PCT": "0.001",
        "STOCK_INVESTMENT_DSS_FINRL_ENV_SELL_COST_PCT": "0.001",
        "STOCK_INVESTMENT_DSS_FINRL_ENV_REWARD_SCALING": "0.0001",
    })

    result = subprocess.run(
        [sys.executable, "-u", "-m",
         "stock_investment_dss.runner.run_finrl_baseline_suite_smoke_test"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[FAIL] subprocess failed with code {result.returncode}")
        print("--- stdout ---")
        print(result.stdout[-1000:])
        print("--- stderr ---")
        print(result.stderr[-1000:])
        raise RuntimeError(f"Seed {seed} run failed")

    # Look for "Random seed: N" in stderr (the new log line)
    seed_log_found = "Random seed:" in (result.stdout + result.stderr)
    print(f"  Log line 'Random seed: {seed}' found: {seed_log_found}")

    # Find the run directory and ppo action_memory
    run_dir = find_latest_suite_run(repo_root)
    if run_dir is None:
        raise RuntimeError("No suite run dir found")
    ppo_actions = (
        run_dir / "data" / "finrl_baseline_suite" / "ppo" / "ppo_action_memory.csv"
    )
    if not ppo_actions.exists():
        raise RuntimeError(f"ppo_action_memory.csv not found in {run_dir}")

    print(f"  Run dir: {run_dir.name}")
    print(f"  PPO actions: {ppo_actions.name}")
    return ppo_actions


def main() -> int:
    repo_root = find_repo_root()

    print("=" * 60)
    print("FinRL Seed Injection Fix Verification Test")
    print("=" * 60)
    print(f"Repo root: {repo_root}")
    print(f"Start:     {datetime.now():%Y-%m-%d %H:%M:%S}")
    print()
    print("This test runs PPO twice with different seeds (500 steps each)")
    print("and compares the resulting action_memory.csv files.")
    print()
    print("Expected after fix:")
    print("  Test 1 (seed=1)  -> hash A")
    print("  Test 2 (seed=42) -> hash B")
    print("  hash A != hash B  -> fix works")
    print()
    print("If hashes are identical: seed injection still broken.")
    print()

    # Run with seed=1
    actions_seed_1 = run_one_seed(repo_root, seed=1)
    hash_seed_1 = file_hash(actions_seed_1)
    print(f"  Hash:    {hash_seed_1[:16]}...")

    # Run with seed=42
    actions_seed_42 = run_one_seed(repo_root, seed=42)
    hash_seed_42 = file_hash(actions_seed_42)
    print(f"  Hash:    {hash_seed_42[:16]}...")

    print()
    print("=" * 60)
    print("Verification Result")
    print("=" * 60)
    print(f"Seed 1 hash:  {hash_seed_1}")
    print(f"Seed 42 hash: {hash_seed_42}")
    print()

    if hash_seed_1 == hash_seed_42:
        print("[FAIL] Hashes are IDENTICAL")
        print("       Seed injection is NOT working.")
        print("       Patch did not take effect, or there is another bug.")
        return 1
    else:
        print("[OK] Hashes are DIFFERENT")
        print("     Seed injection fix is WORKING.")
        print("     Safe to re-run multiseed for thesis-grade evidence.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
