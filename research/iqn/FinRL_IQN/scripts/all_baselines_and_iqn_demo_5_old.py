"""All-Baselines-and-IQN Micro — three-layer end-to-end pipeline validation.

This is NOT thesis evidence. It is a sanity check that exercises the entire
thesis comparison stack (algorithmic + FinRL parametric RL + IQN distributional
RL) in roughly 15-20 minutes total, instead of the 1-2 hours a full thesis
evidence run would take.

The three layers it validates:
    1) Algorithmic trading baselines (non-RL)
       buy_and_hold, sma_crossover, ema_crossover, macd_signal,
       rsi_mean_reversion, bollinger_mean_reversion, breakout, momentum,
       volatility_filter, plus portfolio variants
    2) FinRL parametric RL baselines (a2c, ppo, mvo)
       3 seeds × 5000 timesteps × 5 tickers
    3) IQN distributional RL (3 seeds × 5000 steps × 5 tickers)

Training budget for the RL layers (FinRL and IQN):
    Both RL layers receive the same nominal training budget of 5000 environment
    steps per seed. This is the most common fair-comparison convention in the
    RL literature (training-step budget is reported inclusive of any warm-up
    such as IQN's learning_starts).

    Note on step semantics across layers:
        - Algorithmic baselines are rule-based (no weights, no gradient updates).
          They have no training step concept. Each baseline performs a single
          deterministic pass over the full PIT trade window. Giving them a step
          parameter would be meaningless.
        - FinRL: 5000 SB3 environment steps in agent.learn(), no warm-up.
        - IQN:   5000 environment steps in the agent training loop, of which
          the first 2000 (learning_starts) are replay warm-up with no gradient
          updates. So IQN gets ~3000 effective gradient steps versus FinRL's
          5000. This asymmetry is inherent to the algorithms and is documented
          as such in the thesis methodology.

    Fairness in the etape 5 comparison rests on the PIT split, not the step
    count: all three layers are evaluated on the same trade window
    (2024-01-01 → 2026-12-31) with the same 5-ticker universe.

Configuration designed for speed, not statistical power:
    - 5 tickers instead of 10 (COST, AVGO, LLY, ORCL, CAT)
    - FinRL: 3 seeds, 5000 timesteps, a2c+ppo+mvo
    - IQN: 3 seeds, 5000 steps (learning_starts=2000 → 3000 effective updates)
    - reuses the same frozen import file as finrl_micro_multiseed_demo_5.py

Failure handling:
    All three layers run independently. A failure in one layer does NOT
    stop the others. Final summary reports status per layer with return
    codes and output paths.

Estimated total runtime: 15-20 minutes
    Algorithmic:   ~30-60 sec   (5 tickers, all single-ticker + 2 portfolio baselines)
    FinRL micro:   ~8-12 min    (3 seeds × 3 agents × 5000 steps)
    IQN micro:     ~6-10 min    (3 seeds × 5000 steps with multiseed launcher)

If all three layers succeed, the full comparison pipeline is ready for
etape 5 summary dashboard work.

Usage:
    python scripts/all_baselines_and_iqn_demo_5.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ----------------------------------------------------------------------
# Shared configuration (5-ticker subset, frozen import file)
# ----------------------------------------------------------------------

TICKERS = "COST,AVGO,LLY,ORCL,CAT"
TICKER_LIST = TICKERS.split(",")
# Short dataset_id to stay safely under Windows MAX_PATH (260 chars) for
# deeply-nested algorithmic baseline outputs. The longest strategy name in
# the grid is 'vol_filter_m20_v20_0.4' (22 chars).
DATASET_ID = "demo_10_new_micro"
DATA_IMPORT_FILE = "data/market/daily/imports/market_data_demo10_new_2010_2026.csv"
PIT_POINT_IN_TIME = "2024-01-01"
PIT_TRADE_END_DATE = "2026-12-31"


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------


def find_repo_root() -> Path:
    """Locate the v2 repo root by walking up from cwd."""
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "stock_investment_dss").exists():
            return candidate
    raise RuntimeError(f"Could not find repo root from cwd={current}")


def banner(title: str, char: str = "=") -> None:
    print(char * 78, flush=True)
    print(title, flush=True)
    print(char * 78, flush=True)


def stream_subprocess(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    label: str,
) -> int:
    """Run cmd in a subprocess, stream stdout/stderr line by line, return code."""
    print(f"[{label}] CMD: {' '.join(cmd)}", flush=True)
    print(f"[{label}] CWD: {cwd}", flush=True)
    print(f"[{label}] START: {datetime.now():%H:%M:%S}", flush=True)
    print("-" * 78, flush=True)

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
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
        print(f"\n[{label}] INTERRUPTED — terminating subprocess", flush=True)
        process.terminate()
        process.wait()
        return 130

    rc = process.wait()
    print("-" * 78, flush=True)
    print(f"[{label}] END:   {datetime.now():%H:%M:%S}  RC={rc}", flush=True)
    return rc


# ----------------------------------------------------------------------
# Layer 1: Algorithmic baselines
# ----------------------------------------------------------------------


def run_algorithmic_baselines(repo_root: Path) -> tuple[int, Optional[Path]]:
    """Run the full algorithmic baseline grid on the 5-ticker subset."""
    banner("LAYER 1 / 3 — ALGORITHMIC TRADING BASELINES", char="=")
    print()
    print("Note on training budget:")
    print("  Algorithmic baselines are rule-based (SMA, RSI, Bollinger, MACD,")
    print("  momentum, breakout, volatility filter, etc.) — no weights to learn,")
    print("  no gradient updates. They perform a single deterministic pass over")
    print("  the PIT trade window (2024-01-01 → 2026-12-31) using only data")
    print("  available at each decision point.")
    print()
    print("  The RL layers (FinRL, IQN) each receive 5000 training steps. Fair")
    print("  comparison in the etape 5 dashboard rests on the shared PIT eval")
    print("  window, not on a shared step count — which would be meaningless")
    print("  for the non-parametric baselines.")
    print()

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["PYTHONUNBUFFERED"] = "1"

    expected_output_root = (
        repo_root / "outputs" / "run_registry" / "algorithmic_baselines" / DATASET_ID
    )

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "stock_investment_dss.algorithmic_trading.experiments.run_all_algorithmic_experiments",
        "--trade-data",
        DATA_IMPORT_FILE,
        "--dataset-tag",
        DATASET_ID,
        "--ticker",
        "ALL",
        "--continue-on-error",
    ]

    rc = stream_subprocess(cmd, repo_root, env, label="ALGORITHMIC")
    return rc, expected_output_root if expected_output_root.exists() else None


# ----------------------------------------------------------------------
# Layer 2: FinRL parametric RL baselines
# ----------------------------------------------------------------------


def run_finrl_baselines(repo_root: Path) -> tuple[int, Optional[Path]]:
    """Run FinRL multiseed micro pipeline (3 seeds × 500 steps × a2c+ppo+mvo)."""
    banner("LAYER 2 / 3 — FINRL PARAMETRIC RL BASELINES", char="=")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "PYTHONUNBUFFERED": "1",
            # Dataset
            "STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE": "demo_10_new",
            "STOCK_INVESTMENT_DSS_DAILY_DATASET_ID": DATASET_ID,
            "STOCK_INVESTMENT_DSS_DAILY_DATA_START": "2010-01-01",
            "STOCK_INVESTMENT_DSS_DAILY_DATA_END": "2026-12-31",
            "STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE": DATA_IMPORT_FILE,
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE": "true",
            "STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD": "false",
            "STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD": "false",
            # 5-ticker subset
            "STOCK_INVESTMENT_DSS_FINRL_TICKERS": TICKERS,
            # PIT split
            "STOCK_INVESTMENT_DSS_PIT_SPLIT_ID": f"{DATASET_ID}_pit",
            "STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME": PIT_POINT_IN_TIME,
            "STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE": PIT_TRADE_END_DATE,
            # FinRL micro config
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS": "a2c,ppo",
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO": "true",
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_TOTAL_TIMESTEPS": "5000",
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_DEVICE": "auto",
            # Multiseed: 2 seeds for bug-bash micro validation (production: 3)
            "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_LIST": "1,2",
            "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_STOP_ON_FAILURE": "false",
            "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_RUN_SUMMARY_AFTER": "true",
            # FinRL env parameters
            "STOCK_INVESTMENT_DSS_FINRL_ENV_INITIAL_AMOUNT": "1000000",
            "STOCK_INVESTMENT_DSS_FINRL_ENV_HMAX": "10000",
            "STOCK_INVESTMENT_DSS_FINRL_ENV_BUY_COST_PCT": "0.001",
            "STOCK_INVESTMENT_DSS_FINRL_ENV_SELL_COST_PCT": "0.001",
            "STOCK_INVESTMENT_DSS_FINRL_ENV_REWARD_SCALING": "0.0001",
        }
    )

    launcher_start = datetime.now().timestamp()
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "stock_investment_dss.runner.run_finrl_baseline_multiseed_launcher",
    ]
    rc = stream_subprocess(cmd, repo_root, env, label="FINRL")

    # Locate the launcher run mapped to this invocation
    runs_dir = repo_root / "outputs" / "runs"
    latest_launcher: Optional[Path] = None
    if runs_dir.exists():
        candidates = [
            d
            for d in runs_dir.iterdir()
            if d.is_dir()
            and "finrl_baseline_multiseed_launcher" in d.name
            and d.stat().st_mtime >= launcher_start - 5
        ]
        if candidates:
            latest_launcher = max(candidates, key=lambda p: p.stat().st_mtime)

    return rc, latest_launcher


# ----------------------------------------------------------------------
# Layer 3: IQN distributional RL
# ----------------------------------------------------------------------


def run_iqn_micro_multiseed(repo_root: Path) -> tuple[int, Optional[Path]]:
    """Run IQN multiseed micro pipeline (3 seeds × 5000 steps × 5 tickers)."""
    banner("LAYER 3 / 3 — IQN DISTRIBUTIONAL RL", char="=")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "PYTHONUNBUFFERED": "1",
            # Dataset
            "STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE": "demo_10_new",
            "STOCK_INVESTMENT_DSS_DAILY_DATASET_ID": DATASET_ID,
            "STOCK_INVESTMENT_DSS_DAILY_DATA_START": "2010-01-01",
            "STOCK_INVESTMENT_DSS_DAILY_DATA_END": "2026-12-31",
            "STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE": DATA_IMPORT_FILE,
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE": "true",
            "STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD": "false",
            "STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD": "false",
            # 5-ticker subset (reused from FinRL micro pattern)
            "STOCK_INVESTMENT_DSS_FINRL_TICKERS": TICKERS,
            # PIT split — same as FinRL micro
            "STOCK_INVESTMENT_DSS_PIT_SPLIT_ID": f"{DATASET_ID}_pit",
            "STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME": PIT_POINT_IN_TIME,
            "STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE": PIT_TRADE_END_DATE,
            # IQN micro training config
            # learning_starts=2000 (from iqn_stockdss_default.json),
            # so 5000 steps gives ~3000 actual training steps
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_TOTAL_STEPS": "5000",
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_STARTS": "2000",
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_EVAL_INTERVAL": "1000",
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_MAX_EVAL_STEPS": "2000",
            "STOCK_INVESTMENT_DSS_IQN_BACKTEST_SCORE_MODE": "q50_minus_cvar_penalty",
            "STOCK_INVESTMENT_DSS_IQN_BACKTEST_RISK_LAMBDA": "0.75",
            "STOCK_INVESTMENT_DSS_IQN_DISABLE_CHANGE_STRATEGY": "true",
            "STOCK_INVESTMENT_DSS_IQN_DEVICE": "auto",
            # Multiseed: 2 seeds for bug-bash micro validation (production: 3)
            "STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST": "1,2",
            "STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_STOP_ON_FAILURE": "false",
            "STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_RUN_SUMMARY_AFTER": "true",
        }
    )

    launcher_start = datetime.now().timestamp()
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "stock_investment_dss.runner.run_iqn_learning_curve_multiseed_launcher",
    ]
    rc = stream_subprocess(cmd, repo_root, env, label="IQN")

    # Locate the launcher run mapped to this invocation
    runs_dir = repo_root / "outputs" / "runs"
    latest_launcher: Optional[Path] = None
    if runs_dir.exists():
        candidates = [
            d
            for d in runs_dir.iterdir()
            if d.is_dir()
            and "iqn_learning_curve_multiseed_launcher" in d.name
            and d.stat().st_mtime >= launcher_start - 5
        ]
        if candidates:
            latest_launcher = max(candidates, key=lambda p: p.stat().st_mtime)

    return rc, latest_launcher


# ----------------------------------------------------------------------
# Main orchestrator
# ----------------------------------------------------------------------


def main() -> int:
    repo_root = find_repo_root()

    # Guard: frozen data file
    data_file = repo_root / DATA_IMPORT_FILE
    if not data_file.exists():
        print(f"[ABORT] Frozen data file not found: {data_file}", file=sys.stderr)
        return 1

    banner("ALL-BASELINES-AND-IQN MICRO — three-layer pipeline validation")
    print()
    print("This is NOT thesis evidence. It validates the full comparison stack")
    print("(algorithmic + FinRL + IQN) end-to-end in roughly 15-20 minutes.")
    print()
    print(f"Repo root:    {repo_root}")
    print(f"Python:       {sys.executable}")
    print(f"Start time:   {datetime.now():%Y-%m-%d %H:%M:%S}")
    print()
    print("Shared configuration:")
    print(f"  Dataset ID: {DATASET_ID}")
    print(f"  Tickers:    {TICKERS}")
    print(f"  PIT split:  {PIT_POINT_IN_TIME} -> {PIT_TRADE_END_DATE}")
    print()
    print("Per-layer configuration:")
    print("  Algorithmic: all single-ticker + portfolio baselines (~30-60 sec)")
    print("               rule-based, no training steps (see layer 1 note)")
    print("  FinRL:       2 seeds × 5000 steps × a2c+ppo+mvo (~6-8 min)")
    print("  IQN:         2 seeds × 5000 steps × learning_starts=2000 (~4-7 min)")
    print()
    print("Failure policy: each layer runs independently, status reported at end.")
    print()

    pipeline_start = datetime.now()

    # Run all three layers regardless of individual failures
    print()
    algo_rc, algo_path = run_algorithmic_baselines(repo_root)
    print()
    finrl_rc, finrl_path = run_finrl_baselines(repo_root)
    print()
    iqn_rc, iqn_path = run_iqn_micro_multiseed(repo_root)
    print()

    pipeline_end = datetime.now()
    total_seconds = (pipeline_end - pipeline_start).total_seconds()

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    banner("FINAL SUMMARY — ALL-BASELINES-AND-IQN MICRO")
    print()
    print(f"  Pipeline start: {pipeline_start:%Y-%m-%d %H:%M:%S}")
    print(f"  Pipeline end:   {pipeline_end:%Y-%m-%d %H:%M:%S}")
    print(f"  Total runtime:  {int(total_seconds // 60)}m {int(total_seconds % 60)}s")
    print()

    layers = [
        ("LAYER 1  Algorithmic", algo_rc, algo_path),
        ("LAYER 2  FinRL      ", finrl_rc, finrl_path),
        ("LAYER 3  IQN        ", iqn_rc, iqn_path),
    ]

    print(f"  {'Layer':<22} {'RC':>4}   {'Output':<60}")
    print(f"  {'-' * 22} {'-' * 4}   {'-' * 60}")
    for name, rc, path in layers:
        status = "OK" if rc == 0 else f"FAIL({rc})"
        path_str = str(path.relative_to(repo_root)) if path else "(not found)"
        print(f"  {name} {status:>4}   {path_str:<60}")
    print()

    passed = sum(1 for _, rc, _ in layers if rc == 0)
    total = len(layers)

    if passed == total:
        banner(f"ALL {total} LAYERS PASSED — pipeline ready for etape 5", char="=")
        return 0

    banner(f"{passed}/{total} LAYERS PASSED — review failed layers above", char="!")
    return 1 if passed == 0 else 2  # 2 = partial success


if __name__ == "__main__":
    sys.exit(main())
