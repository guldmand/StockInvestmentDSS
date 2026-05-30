"""All-Baselines-and-IQN S&P 500 v2 — Fair Portfolio Comparison Across All Tiers.

Extends the v1 three-layer demo with two new components that enable a
head-to-head comparison of ALL strategy tiers on the same PIT eval window,
the same universe, and the same initial capital (1 M USD):

    Layer 1a  Algorithmic baselines (run_all_algorithmic_experiments)
              — buy_and_hold, sma_crossover, ema_crossover, macd_signal,
                rsi_mean_reversion, bollinger_mean_reversion, momentum,
                breakout, volatility_filter × full universe (all 467 tickers)

    Layer 1b  NEW — Single-ticker-strategy equal-weight portfolio aggregation
              (single_ticker_portfolio_wrapper) × 9 strategies
              Each strategy is run on every ticker with capital / N tickers;
              per-ticker series are summed to produce portfolio-level metrics.
              This lets rule-based baselines be compared directly to RL methods.

    Layer 2   FinRL parametric RL baselines (run_finrl_baseline_multiseed_launcher)
              — a2c, ppo, ddpg, sac, td3 + MVO × SEED_LIST seeds

    Layer 3   IQN distributional RL (run_iqn_learning_curve_multiseed_launcher)
              — 25 000 steps × SEED_LIST seeds × full universe

    Layer 4   NEW — IQN+HDP+EDL portfolio backtest (run_iqn_hdp_edl_portfolio_backtest)
              Replays pre-computed combined_with_counterfactual_labels.csv across
              4 ablation tiers (A1 IQN-only, A2 IQN+HDP, A3 IQN+EDL, A4 full)
              in a live FinRL environment to produce fair portfolio metrics.

All layers share:
    - Universe: SP500 (467 tickers, frozen import file)
    - PIT eval window: 2024-01-01 → 2026-05-26
    - Initial capital: 1 000 000 USD
    - Transaction cost: 0.1% (0.001)

Bug fixes vs B.6.4/B.6.5 (transparent in Layer 4):
    - Correct IDX_TO_ACTION = {0:"HOLD", 1:"BUY", 2:"SELL"} (training order)
    - No double +1 in EDL ensemble inference

Usage::

    # Full run (all seeds, all strategies, all tickers)
    python scripts/all_baselines_and_iqn_demo_sp500_v2.py

    # Smoke test (fast end-to-end validation, ~5-10 min)
    python scripts/all_baselines_and_iqn_demo_sp500_v2.py --smoke

Pre-requisites:
    Before running Layer 4 you must have:
    - COMBINED_CSV produced by phase B.6 (run_phase_b6_* or the oracle runner)
    - MERGED_DIR produced by EDL training (B.3 v3) — MERGED_..._COMPLETE folder

    If either path is missing, Layer 4 is skipped with a warning (other layers
    proceed normally).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# ---------------------------------------------------------------------------
# Shared configuration (matches sp500_8020_v1.json and frozen market file)
# ---------------------------------------------------------------------------

TICKERS = "ALL"
DATASET_ID = "sp500"
DATA_IMPORT_FILE = "data/market/daily/imports/market_data_full_500.csv"
MANIFEST_PATH = "configs/experiments/sp500_8020_v1.json"
PIT_POINT_IN_TIME = "2024-01-01"
PIT_TRADE_END_DATE = "2026-05-26"
INITIAL_AMOUNT = 1_000_000
TRANSACTION_COST_PCT = 0.001

# Pre-computed artifacts required by Layer 4 (read-only; never overwritten here)
COMBINED_CSV = (
    "outputs/runs/"
    "2026_05_28_144213_d_iqn_dss_edl_counterfactual_oracle_production/"
    "audit/combined_with_counterfactual_labels.csv"
)
MERGED_DIR = (
    "outputs/runs/" "MERGED_2026_05_28_d_iqn_dss_edl_action_training_v3_COMPLETE"
)

# Number of seeds for Layers 2 + 3.  Start at 1 for a first end-to-end timing
# run; raise to "1,2,...,10" for the full 10-seed comparison.
SEED_LIST = "1,2,3,4,5,6,7,8,9,10"

# Single-ticker strategies to aggregate as portfolios in Layer 1b
SINGLE_TICKER_STRATEGIES = [
    "buy_and_hold",
    "sma_crossover",
    "ema_crossover",
    "macd_signal",
    "rsi_mean_reversion",
    "bollinger_mean_reversion",
    "momentum",
    "breakout",
    "volatility_filter",
]

# Smoke-mode overrides
_SMOKE_STRATEGIES = ["buy_and_hold", "sma_crossover", "ema_crossover", "macd_signal", "rsi_mean_reversion", "bollinger_mean_reversion", "momentum", "breakout", "volatility_filter"]
_SMOKE_TICKERS = "AAPL,MSFT,GOOG,AMZN,TSLA"
_SMOKE_SEED_LIST = "1"


# ---------------------------------------------------------------------------
# Utilities (copied from v1)
# ---------------------------------------------------------------------------


def find_repo_root() -> Path:
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
    """Run cmd in a subprocess, stream stdout/stderr, return exit code."""
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


def _base_env(repo_root: Path) -> dict[str, str]:
    """Build base environment dict shared by all layers."""
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": f"{repo_root}/src:{repo_root}/external/FinRL",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    return env


# ---------------------------------------------------------------------------
# Layer 1a: Algorithmic baselines (rule-based)
# ---------------------------------------------------------------------------


def run_algorithmic_baselines(
    repo_root: Path, smoke: bool = False
) -> tuple[int, Optional[Path]]:
    banner("LAYER 1a / 5 — ALGORITHMIC TRADING BASELINES (rule-based)", char="=")

    env = _base_env(repo_root)
    env.update(
        {
            "STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME": PIT_POINT_IN_TIME,
            "STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE": PIT_TRADE_END_DATE,
        }
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
        _SMOKE_TICKERS if smoke else "ALL",
        "--continue-on-error",
    ]

    launcher_start = datetime.now().timestamp()
    rc = stream_subprocess(cmd, repo_root, env, label="ALGO")

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


# ---------------------------------------------------------------------------
# Layer 1b: Single-ticker strategy → equal-weight portfolio aggregation
# ---------------------------------------------------------------------------


def run_single_ticker_portfolio_wrapper(
    repo_root: Path,
    strategies: list[str],
    tickers_arg: str,
) -> tuple[dict[str, int], Optional[Path]]:
    """Run each single-ticker strategy as an equal-weight portfolio.

    Returns a dict of {strategy_name: rc} and the common output parent dir.
    """
    banner("LAYER 1b / 5 — SINGLE-TICKER PORTFOLIO AGGREGATION", char="=")

    env = _base_env(repo_root)

    output_parent = (
        repo_root
        / "outputs"
        / "runs"
        / f"{datetime.now():%Y_%m_%d_%H%M%S}_d_iqn_dss_single_ticker_portfolios_v2"
    )
    output_parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, int] = {}
    for strategy in strategies:
        label = f"STPW/{strategy[:12]}"
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "stock_investment_dss.algorithmic_trading.baselines.single_ticker_portfolio_wrapper",
            "--strategy-name",
            strategy,
            "--market-data",
            DATA_IMPORT_FILE,
            "--tickers",
            tickers_arg,
            "--pit-start",
            PIT_POINT_IN_TIME,
            "--pit-end",
            PIT_TRADE_END_DATE,
            "--initial-amount",
            str(INITIAL_AMOUNT),
            "--transaction-cost-pct",
            str(TRANSACTION_COST_PCT),
            "--dataset-tag",
            DATASET_ID,
            "--output-dir",
            str(output_parent / strategy),
        ]
        rc = stream_subprocess(cmd, repo_root, env, label=label)
        results[strategy] = rc
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"  [{label}] {strategy}: {status}", flush=True)

    passed = sum(1 for rc in results.values() if rc == 0)
    print(
        f"\n  Layer 1b: {passed}/{len(strategies)} strategies succeeded.",
        flush=True,
    )
    return results, output_parent


# ---------------------------------------------------------------------------
# Layer 2: FinRL parametric RL baselines
# ---------------------------------------------------------------------------


def run_finrl_baselines(repo_root: Path, seed_list: str) -> tuple[int, Optional[Path]]:
    banner("LAYER 2 / 5 — FINRL PARAMETRIC RL BASELINES", char="=")

    env = _base_env(repo_root)
    env.update(
        {
            "STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE": "sp500",
            "STOCK_INVESTMENT_DSS_DAILY_DATASET_ID": DATASET_ID,
            "STOCK_INVESTMENT_DSS_DAILY_DATA_START": "2016-01-01",
            "STOCK_INVESTMENT_DSS_DAILY_DATA_END": "2026-05-26",
            "STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE": DATA_IMPORT_FILE,
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE": "true",
            "STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD": "false",
            "STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD": "false",
            "STOCK_INVESTMENT_DSS_FINRL_TICKERS": TICKERS,
            "STOCK_INVESTMENT_DSS_PIT_SPLIT_ID": f"{DATASET_ID}_pit",
            "STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME": PIT_POINT_IN_TIME,
            "STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE": PIT_TRADE_END_DATE,
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS": "a2c,ppo,ddpg,sac,td3",
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO": "true",
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_TOTAL_TIMESTEPS": "25000",
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_DEVICE": "auto",
            "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_LIST": seed_list,
            "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_STOP_ON_FAILURE": "false",
            "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_RUN_SUMMARY_AFTER": "true",
            "STOCK_INVESTMENT_DSS_FINRL_ENV_INITIAL_AMOUNT": str(INITIAL_AMOUNT),
            "STOCK_INVESTMENT_DSS_FINRL_ENV_HMAX": "10000",
            "STOCK_INVESTMENT_DSS_FINRL_ENV_BUY_COST_PCT": str(TRANSACTION_COST_PCT),
            "STOCK_INVESTMENT_DSS_FINRL_ENV_SELL_COST_PCT": str(TRANSACTION_COST_PCT),
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


# ---------------------------------------------------------------------------
# Layer 3: IQN distributional RL
# ---------------------------------------------------------------------------


def run_iqn_multiseed(
    repo_root: Path, seed_list: str, smoke: bool = False
) -> tuple[int, Optional[Path]]:
    banner("LAYER 3 / 5 — IQN DISTRIBUTIONAL RL (25 000 steps)", char="=")

    env = _base_env(repo_root)
    env.update(
        {
            "STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE": "sp500",
            "STOCK_INVESTMENT_DSS_DAILY_DATASET_ID": DATASET_ID,
            "STOCK_INVESTMENT_DSS_DAILY_DATA_START": "2016-01-01",
            "STOCK_INVESTMENT_DSS_DAILY_DATA_END": "2026-05-26",
            "STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE": DATA_IMPORT_FILE,
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE": "true",
            "STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD": "false",
            "STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD": "false",
            "STOCK_INVESTMENT_DSS_FINRL_TICKERS": TICKERS,
            "STOCK_INVESTMENT_DSS_PIT_SPLIT_ID": f"{DATASET_ID}_pit",
            "STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME": PIT_POINT_IN_TIME,
            "STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE": PIT_TRADE_END_DATE,
            # 25k steps — required to avoid HOLD-collapse on full 467-ticker universe
            # smoke mode uses 5k steps / 500 learning_starts for fast end-to-end validation
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_TOTAL_STEPS": (
                "5000" if smoke else "25000"
            ),
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_STARTS": "500" if smoke else "2000",
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_EVAL_INTERVAL": "1000",
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_MAX_EVAL_STEPS": "2000",
            "STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM": "true",
            "STOCK_INVESTMENT_DSS_IQN_BACKTEST_SCORE_MODE": "q50",
            "STOCK_INVESTMENT_DSS_IQN_BACKTEST_RISK_LAMBDA": "0.0",
            "STOCK_INVESTMENT_DSS_IQN_DISABLE_CHANGE_STRATEGY": "true",
            "STOCK_INVESTMENT_DSS_IQN_DEVICE": "auto",
            "STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST": seed_list,
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


# ---------------------------------------------------------------------------
# Layer 4: IQN+HDP+EDL portfolio backtest (4 ablations)
# ---------------------------------------------------------------------------


def run_iqn_hdp_edl_backtest(
    repo_root: Path,
    combined_csv: Path,
    merged_dir: Path,
    smoke: bool,
) -> tuple[int, Optional[Path]]:
    banner("LAYER 4 / 5 — IQN+HDP+EDL PORTFOLIO BACKTEST (4 ablations)", char="=")

    output_dir = (
        repo_root
        / "outputs"
        / "runs"
        / f"{datetime.now():%Y_%m_%d_%H%M%S}_d_iqn_dss_edl_portfolio_backtest_v2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env = _base_env(repo_root)

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "stock_investment_dss.runner.run_iqn_hdp_edl_portfolio_backtest",
        "--combined-csv",
        str(combined_csv),
        "--merged-dir",
        str(merged_dir),
        "--market-data",
        DATA_IMPORT_FILE,
        "--manifest",
        MANIFEST_PATH,
        "--output-dir",
        str(output_dir),
        "--ablations",
        "a1,a2,a3,a4",
        "--initial-amount",
        str(INITIAL_AMOUNT),
        "--transaction-cost-pct",
        str(TRANSACTION_COST_PCT),
    ]
    if smoke:
        cmd.append("--smoke")

    rc = stream_subprocess(cmd, repo_root, env, label="IQN_HDP_EDL")
    return rc, output_dir


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


def _copy_files(src_dir: Path, dst_dir: Path, patterns: tuple[str, ...]) -> int:
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


def _find_sibling_summary(launcher_path: Path, summary_pattern: str) -> Optional[Path]:
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


def consolidate_v2(
    repo_root: Path,
    algo_path: Optional[Path],
    stpw_path: Optional[Path],
    finrl_launcher_path: Optional[Path],
    iqn_launcher_path: Optional[Path],
    edl_backtest_path: Optional[Path],
    pipeline_start: datetime,
) -> Optional[Path]:
    """Consolidate key artifacts from all 5 layers into a single readable dir."""
    timestamp = pipeline_start.strftime("%Y_%m_%d_%H%M%S")
    consolidated = (
        repo_root
        / "outputs"
        / "runs"
        / f"{timestamp}_combined_v2_fair_portfolio_comparison"
    )
    if consolidated.exists():
        consolidated = consolidated.with_name(consolidated.name + "_retry")
    consolidated.mkdir(parents=True)

    # Layer 1a: algorithmic baselines
    if algo_path is not None and algo_path.exists():
        _copy_files(
            algo_path / "summary",
            consolidated / "algorithmic",
            patterns=("*.csv",),
        )

    # Layer 1b: single-ticker portfolio wrappers
    if stpw_path is not None and stpw_path.exists():
        for strategy_dir in stpw_path.iterdir():
            if strategy_dir.is_dir():
                for metrics_csv in strategy_dir.rglob("portfolio_metrics.csv"):
                    dst = consolidated / "single_ticker_portfolios" / strategy_dir.name
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(metrics_csv, dst / metrics_csv.name)
                for account_csv in strategy_dir.rglob("portfolio_account_value.csv"):
                    dst = consolidated / "single_ticker_portfolios" / strategy_dir.name
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(account_csv, dst / account_csv.name)

    # Layer 2: FinRL multiseed summary
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

    # Layer 3: IQN multiseed summary
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

    # Layer 4: IQN+HDP+EDL ablations
    if edl_backtest_path is not None and edl_backtest_path.exists():
        for ablation_dir in ("a1", "a2", "a3", "a4"):
            _copy_files(
                edl_backtest_path / ablation_dir,
                consolidated / "iqn_hdp_edl" / ablation_dir,
                patterns=("*.csv",),
            )
        ablation_summary = edl_backtest_path / "ablation_summary.csv"
        if ablation_summary.exists():
            (consolidated / "iqn_hdp_edl").mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                ablation_summary, consolidated / "iqn_hdp_edl" / "ablation_summary.csv"
            )

    # Write a pipeline manifest
    manifest = {
        "pipeline": "all_baselines_and_iqn_demo_sp500_v2",
        "pipeline_start": pipeline_start.isoformat(),
        "pit_eval_window": f"{PIT_POINT_IN_TIME} → {PIT_TRADE_END_DATE}",
        "initial_amount": INITIAL_AMOUNT,
        "transaction_cost_pct": TRANSACTION_COST_PCT,
        "layers": {
            "1a_algorithmic": str(algo_path or ""),
            "1b_single_ticker_portfolios": str(stpw_path or ""),
            "2_finrl": str(finrl_launcher_path or ""),
            "3_iqn": str(iqn_launcher_path or ""),
            "4_edl_backtest": str(edl_backtest_path or ""),
        },
    }
    import json

    with open(consolidated / "pipeline_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return consolidated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Fair portfolio comparison across all thesis strategy tiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke mode: 1 seed, 5 tickers for Layer 1b, --smoke for Layer 4. "
            "Fast end-to-end validation (~5-10 min)."
        ),
    )
    args = p.parse_args()

    repo_root = find_repo_root()

    # Guard: frozen market data
    data_file = repo_root / DATA_IMPORT_FILE
    if not data_file.exists():
        print(f"[ABORT] Frozen data file not found: {data_file}", file=sys.stderr)
        return 1

    # Resolve Layer 4 inputs (optional; Layer 4 is skipped if missing)
    combined_csv = repo_root / COMBINED_CSV
    merged_dir = repo_root / MERGED_DIR
    layer4_available = combined_csv.exists() and merged_dir.exists()

    if not layer4_available:
        missing = []
        if not combined_csv.exists():
            missing.append(str(combined_csv.relative_to(repo_root)))
        if not merged_dir.exists():
            missing.append(str(merged_dir.relative_to(repo_root)))
        print(
            f"[WARN] Layer 4 will be SKIPPED — pre-requisites not found:\n"
            + "\n".join(f"  {m}" for m in missing),
            flush=True,
        )

    seed_list = _SMOKE_SEED_LIST if args.smoke else SEED_LIST
    stpw_strategies = _SMOKE_STRATEGIES if args.smoke else SINGLE_TICKER_STRATEGIES
    stpw_tickers = _SMOKE_TICKERS if args.smoke else "ALL"

    banner("ALL-BASELINES-AND-IQN SP500 v2 — FAIR PORTFOLIO COMPARISON")
    print()
    print(f"Repo root:        {repo_root}")
    print(f"Python:           {sys.executable}")
    print(f"Start time:       {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Smoke mode:       {args.smoke}")
    print()
    print("Shared configuration:")
    print(f"  PIT eval:       {PIT_POINT_IN_TIME} → {PIT_TRADE_END_DATE}")
    print(f"  Initial amount: {INITIAL_AMOUNT:,} USD")
    print(f"  Transaction:    {TRANSACTION_COST_PCT:.3f}")
    print(f"  Seed list:      {seed_list}")
    print()
    print("Layer configuration:")
    print(f"  1a  Algorithmic baselines: all rules, ALL tickers")
    print(f"  1b  Portfolio aggregation: {stpw_strategies} × {stpw_tickers}")
    print(f"  2   FinRL multiseed: {seed_list} seeds × 25000 steps")
    print(f"  3   IQN multiseed:   {seed_list} seeds × 25000 steps")
    print(
        f"  4   IQN+HDP+EDL:     {'--smoke (100 rows)' if args.smoke else '4 ablations, all rows'}"
    )
    print(f"  4   Layer 4 inputs available: {layer4_available}")
    print()
    print("Failure policy: each layer runs independently (failures reported at end).")
    print()

    pipeline_start = datetime.now()

    # Run all layers independently (failures do not stop subsequent layers)
    algo_rc, algo_path = run_algorithmic_baselines(repo_root, smoke=args.smoke)
    print()

    stpw_results, stpw_path = run_single_ticker_portfolio_wrapper(
        repo_root, stpw_strategies, stpw_tickers
    )
    stpw_rc = 0 if all(rc == 0 for rc in stpw_results.values()) else 1
    print()

    finrl_rc, finrl_path = run_finrl_baselines(repo_root, seed_list)
    print()

    iqn_rc, iqn_path = run_iqn_multiseed(repo_root, seed_list, smoke=args.smoke)
    print()

    edl_rc: int = -1
    edl_path: Optional[Path] = None
    if layer4_available:
        edl_rc, edl_path = run_iqn_hdp_edl_backtest(
            repo_root, combined_csv, merged_dir, smoke=args.smoke
        )
        print()
    else:
        print("[LAYER 4] SKIPPED — pre-requisites not found.", flush=True)
        print()

    pipeline_end = datetime.now()
    total_seconds = (pipeline_end - pipeline_start).total_seconds()

    # Consolidate
    consolidated_path: Optional[Path] = None
    try:
        consolidated_path = consolidate_v2(
            repo_root=repo_root,
            algo_path=algo_path,
            stpw_path=stpw_path,
            finrl_launcher_path=finrl_path,
            iqn_launcher_path=iqn_path,
            edl_backtest_path=edl_path,
            pipeline_start=pipeline_start,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] Consolidation failed: {exc}", flush=True)

    # Summary
    banner("FINAL SUMMARY — ALL-BASELINES-AND-IQN SP500 v2")
    print()
    print(f"  Pipeline start: {pipeline_start:%Y-%m-%d %H:%M:%S}")
    print(f"  Pipeline end:   {pipeline_end:%Y-%m-%d %H:%M:%S}")
    print(f"  Total runtime:  {int(total_seconds // 60)}m {int(total_seconds % 60)}s")
    print()

    layers = [
        ("LAYER 1a  Algorithmic", algo_rc, algo_path),
        (
            "LAYER 1b  SingleTicker",
            stpw_rc,
            stpw_path,
        ),
        ("LAYER 2   FinRL       ", finrl_rc, finrl_path),
        ("LAYER 3   IQN         ", iqn_rc, iqn_path),
        (
            "LAYER 4   IQN+HDP+EDL",
            edl_rc if layer4_available else None,
            edl_path,
        ),
    ]

    print(f"  {'Layer':<25} {'RC':>6}   {'Output':<55}")
    print(f"  {'-' * 25} {'-' * 6}   {'-' * 55}")
    for name, rc, path in layers:
        if rc is None:
            status = "SKIP"
        else:
            status = "OK" if rc == 0 else f"FAIL({rc})"
        path_str = (
            str(path.relative_to(repo_root))
            if path and path.exists()
            else "(not found)"
        )
        print(f"  {name} {status:>6}   {path_str:<55}")
    print()

    if consolidated_path is not None:
        print(f"  Consolidated: {consolidated_path.relative_to(repo_root)}")
    print()

    runnable_layers = [(n, rc, p) for n, rc, p in layers if rc is not None]
    passed = sum(1 for _, rc, _ in runnable_layers if rc == 0)
    total = len(runnable_layers)

    if passed == total:
        banner(f"ALL {total} LAYERS PASSED — v2 pipeline ready", char="=")
        return 0

    banner(f"{passed}/{total} LAYERS PASSED — review failed layers above", char="!")
    return 1 if passed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
