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
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure src/ is on sys.path so consolidation imports work in the main process
# without requiring the caller to set PYTHONPATH manually.
_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

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

    launcher_start = datetime.now().timestamp()
    rc = stream_subprocess(cmd, repo_root, env, label="ALGORITHMIC")

    # Locate the algorithmic baseline grid run directory
    runs_dir = repo_root / "outputs" / "runs"
    latest_run: Optional[Path] = None
    if runs_dir.exists():
        candidates = [
            d
            for d in runs_dir.iterdir()
            if d.is_dir()
            and "algorithmic_baseline_grid" in d.name
            and d.stat().st_mtime >= launcher_start - 5
        ]
        if candidates:
            latest_run = max(candidates, key=lambda p: p.stat().st_mtime)

    return rc, latest_run


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
    # Post-run consolidation — copy key artifacts into one readable dir
    # ------------------------------------------------------------------
    consolidated_path: Optional[Path] = None
    try:
        consolidated_path = consolidate_run_outputs(
            repo_root=repo_root,
            algo_path=algo_path,
            finrl_launcher_path=finrl_path,
            iqn_launcher_path=iqn_path,
            pipeline_start=pipeline_start,
        )
    except Exception as exc:  # noqa: BLE001
        # Consolidation is non-essential; never let it break the pipeline
        print(f"  [warn] Consolidation step failed: {exc}")

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

    if consolidated_path is not None:
        print()
        print(f"  Consolidated outputs: {consolidated_path.relative_to(repo_root)}")
    print()

    if passed == total:
        banner(f"ALL {total} LAYERS PASSED — pipeline ready for etape 5", char="=")
        return 0

    banner(f"{passed}/{total} LAYERS PASSED — review failed layers above", char="!")
    return 1 if passed == 0 else 2  # 2 = partial success


# ----------------------------------------------------------------------
# Post-run consolidation
# ----------------------------------------------------------------------


def _find_sibling_summary(launcher_path: Path, summary_pattern: str) -> Optional[Path]:
    """Find the multiseed_summary directory created after the launcher.

    Multiseed launchers create a sibling directory in outputs/runs/ with a
    matching base prefix and a '..._multiseed_summary' suffix. This helper
    scans the launcher's parent for that sibling, modified at or after the
    launcher itself.
    """
    runs_dir = launcher_path.parent
    launcher_mtime = launcher_path.stat().st_mtime
    candidates = [
        d
        for d in runs_dir.iterdir()
        if d.is_dir()
        and summary_pattern in d.name
        and d.stat().st_mtime >= launcher_mtime
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _copy_files(src_dir: Path, dst_dir: Path, patterns: tuple[str, ...]) -> int:
    """Copy files matching any pattern from src to dst. Returns count."""
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for pattern in patterns:
        for src_file in src_dir.glob(pattern):
            if src_file.is_file():
                shutil.copy2(src_file, dst_dir / src_file.name)
                count += 1
    return count


def consolidate_run_outputs(
    repo_root: Path,
    algo_path: Optional[Path],
    finrl_launcher_path: Optional[Path],
    iqn_launcher_path: Optional[Path],
    pipeline_start: datetime,
) -> Optional[Path]:
    """Consolidate key artifacts from the 3-layer run into a single readable dir.

    Raw manifests in outputs/runs/ are not modified — this function only
    copies final aggregated artifacts (CSVs, PNGs) into a flat structure
    that mirrors the conceptual pipeline (algorithmic / finrl / iqn).
    Cross-platform: uses shutil.copy2() exclusively (no symlinks, no junctions).
    """
    timestamp = pipeline_start.strftime("%Y_%m_%d_%H%M%S")
    consolidated = (
        repo_root / "outputs" / "runs" / f"{timestamp}_combined_micro_validation"
    )

    if consolidated.exists():
        # Avoid colliding with a previous attempt at the same timestamp
        consolidated = consolidated.with_name(consolidated.name + "_retry")

    consolidated.mkdir(parents=True)

    raw_manifest_paths: list[str] = []

    # --- Layer 1: Algorithmic ---
    if algo_path is not None and algo_path.exists():
        n = _copy_files(
            algo_path / "summary",
            consolidated / "algorithmic",
            patterns=("*.csv",),
        )
        if n:
            raw_manifest_paths.append(f"Algorithmic: `outputs/runs/{algo_path.name}/`")

    # --- Layer 2: FinRL multiseed summary (sibling of launcher) ---
    if finrl_launcher_path is not None and finrl_launcher_path.exists():
        finrl_summary = _find_sibling_summary(
            finrl_launcher_path, "finrl_baseline_multiseed_summary"
        )
        if finrl_summary is not None:
            _copy_files(
                finrl_summary / "summary",
                consolidated / "finrl",
                patterns=("*.csv", "*.png", "*.json"),
            )
            _copy_files(
                finrl_summary / "data",
                consolidated / "finrl",
                patterns=("*.csv",),
            )
            raw_manifest_paths.append(
                f"FinRL launcher: `outputs/runs/{finrl_launcher_path.name}/`"
            )
            raw_manifest_paths.append(
                f"FinRL summary:  `outputs/runs/{finrl_summary.name}/`"
            )

    # --- Layer 3: IQN multiseed summary (sibling of launcher) ---
    if iqn_launcher_path is not None and iqn_launcher_path.exists():
        iqn_summary = _find_sibling_summary(
            iqn_launcher_path, "iqn_learning_curve_multiseed_summary"
        )
        if iqn_summary is not None:
            _copy_files(
                iqn_summary / "summary",
                consolidated / "iqn",
                patterns=("*.csv", "*.png", "*.json"),
            )
            _copy_files(
                iqn_summary / "data",
                consolidated / "iqn",
                patterns=("*.csv",),
            )
            raw_manifest_paths.append(
                f"IQN launcher: `outputs/runs/{iqn_launcher_path.name}/`"
            )
            raw_manifest_paths.append(
                f"IQN summary:  `outputs/runs/{iqn_summary.name}/`"
            )

    # --- Thesis plots (etape 5a) ---
    thesis_plots_dir = consolidated / "thesis_plots"

    if algo_path is not None and algo_path.exists():
        try:
            from stock_investment_dss.visualization.algorithmic_multi_ticker_plots import (
                generate_algorithmic_multi_ticker_plots,
            )

            generate_algorithmic_multi_ticker_plots(
                algorithmic_run_root=algo_path,
                output_dir=thesis_plots_dir / "algorithmic_multi_ticker",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] Algorithmic multi-ticker plot generation failed: {exc}")

    if finrl_launcher_path is not None and finrl_launcher_path.exists():
        try:
            runs_dir = finrl_launcher_path.parent
            launcher_mtime = finrl_launcher_path.stat().st_mtime
            finrl_seed_runs = sorted(
                [
                    d
                    for d in runs_dir.iterdir()
                    if d.is_dir()
                    and "finrl_baseline_suite_smoke_test" in d.name
                    and d.stat().st_mtime >= launcher_mtime
                ]
            )
            if finrl_seed_runs:
                from stock_investment_dss.visualization.finrl_multi_baseline_plot import (
                    generate_finrl_multi_baseline_plot,
                )

                generate_finrl_multi_baseline_plot(
                    finrl_seed_run_roots=finrl_seed_runs,
                    output_dir=thesis_plots_dir / "finrl_multi_baseline",
                )
            else:
                print("  [warn] No FinRL seed runs found for multi-baseline plot.")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] FinRL multi-baseline plot generation failed: {exc}")

    # --- Transaction logs (etape 5a fix-4) ---
    transaction_logs_dir = consolidated / "transaction_logs"

    if algo_path is not None and algo_path.exists():
        try:
            from stock_investment_dss.visualization.transaction_logs import (
                generate_algorithmic_transaction_logs,
            )

            generate_algorithmic_transaction_logs(
                algorithmic_run_root=algo_path,
                output_dir=transaction_logs_dir / "algorithmic",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] Algorithmic transaction log generation failed: {exc}")

    if finrl_launcher_path is not None and finrl_launcher_path.exists():
        try:
            runs_dir = finrl_launcher_path.parent
            launcher_mtime = finrl_launcher_path.stat().st_mtime
            finrl_seed_runs_for_logs = sorted(
                [
                    d
                    for d in runs_dir.iterdir()
                    if d.is_dir()
                    and "finrl_baseline_suite_smoke_test" in d.name
                    and d.stat().st_mtime >= launcher_mtime
                ]
            )
            if finrl_seed_runs_for_logs:
                from stock_investment_dss.visualization.transaction_logs import (
                    generate_finrl_transaction_logs,
                )

                generate_finrl_transaction_logs(
                    finrl_seed_run_roots=finrl_seed_runs_for_logs,
                    output_dir=transaction_logs_dir / "finrl",
                )
            else:
                print("  [warn] No FinRL seed runs found for transaction logs.")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] FinRL transaction log generation failed: {exc}")

    if iqn_launcher_path is not None and iqn_launcher_path.exists():
        try:
            runs_dir = iqn_launcher_path.parent
            launcher_mtime = iqn_launcher_path.stat().st_mtime
            iqn_seed_runs_for_logs = sorted(
                [
                    d
                    for d in runs_dir.iterdir()
                    if d.is_dir()
                    and "iqn_learning_curve_smoke_test" in d.name
                    and d.stat().st_mtime >= launcher_mtime
                ]
            )
            if iqn_seed_runs_for_logs:
                from stock_investment_dss.visualization.transaction_logs import (
                    generate_iqn_transaction_logs,
                )

                generate_iqn_transaction_logs(
                    iqn_seed_run_roots=iqn_seed_runs_for_logs,
                    output_dir=transaction_logs_dir / "iqn",
                )
            else:
                print("  [warn] No IQN seed runs found for transaction logs.")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] IQN transaction log generation failed: {exc}")

    # --- README ---
    readme_lines = [
        f"# Combined Micro Validation — {pipeline_start:%Y-%m-%d %H:%M:%S}",
        "",
        "This directory contains a consolidated, human-readable view of the",
        "key artifacts produced by the 3-layer pipeline.",
        "",
        "Raw manifests (not modified) live in `outputs/runs/`:",
        "",
    ]
    for line in raw_manifest_paths:
        readme_lines.append(f"- {line}")
    readme_lines.extend(
        [
            "",
            "## Layout",
            "",
            "- `algorithmic/` — algorithmic baselines summary CSVs",
            "- `finrl/`       — FinRL multiseed aggregate CSVs and plots",
            "- `iqn/`         — IQN multiseed aggregate CSVs and plots",
            "",
            "## Thesis plots",
            "",
            "- `thesis_plots/algorithmic_multi_ticker/single_ticker_strategies/` — 24 PNGs, one per single-ticker strategy variant, each with one line per ticker (PIT trade window only)",
            "- `thesis_plots/algorithmic_multi_ticker/portfolio_strategies/` — 2 PNGs, one per portfolio-level strategy (PIT trade window only)",
            "- `thesis_plots/finrl_multi_baseline/` — 1 PNG, FinRL agents (a2c, ppo, mvo) mean ± std across seeds, PIT trade window",
            "",
            "## Transaction logs",
            "",
            "- `transaction_logs/algorithmic/single_ticker/` — one .md per (strategy, ticker) pair with entry/exit event table",
            "- `transaction_logs/algorithmic/portfolio/` — one .md per portfolio strategy with initial holdings or rebalance events",
            "- `transaction_logs/finrl/` — one .md per (agent, seed) with daily action table and cumulative holdings",
            "- `transaction_logs/iqn/` — one .md per seed with action distribution and non-HOLD decision table",
            "",
            "## Dataset",
            "",
            f"- Dataset ID: `{DATASET_ID}`",
            f"- Tickers:    `{TICKERS}`",
            f"- PIT split:  {PIT_POINT_IN_TIME} -> {PIT_TRADE_END_DATE}",
        ]
    )
    (consolidated / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    return consolidated


if __name__ == "__main__":
    sys.exit(main())
