"""
Grid runner orchestrator for algorithmic trading baselines.

Reproduces the exact 24 single-ticker algorithmic baseline configurations from the
V1 evidence package, plus two portfolio-level baselines, for a given dataset.

Audit result:
    The V1 evidence grid was confirmed from:
        external/ObjectRL_style/outputs/runs/
            test_algorithmic_trading_2025_grid_final/
            algorithmic_trading/summary/algorithmic_trading_summary.md

    24 single-ticker configurations were identified (see SINGLE_TICKER_CONFIGS below).
    The candidate grid matches the audited V1 evidence exactly.

Design decisions:
    - Run functions are imported directly (not launched as subprocesses) to avoid
      serialisation overhead and to keep error handling straightforward.
    - The ``strategy_folder`` additive parameter added to all 10 baseline functions
      in Etape 3e routes each configuration's output to a dedicated subfolder, leaving
      the existing Etape 3a-3d default-config outputs untouched.
    - Aggregated summary CSVs are written after all runs complete (or after the last
      attempted run when --continue-on-error is used).

Output paths:
    Single-ticker:
        outputs/run_registry/algorithmic_baselines/<dataset_tag>/<strategy_folder>/<ticker>/
    Portfolio:
        outputs/run_registry/algorithmic_baselines/<dataset_tag>/<strategy_folder>/portfolio/

Summary CSVs:
    Per-ticker:
        outputs/run_registry/algorithmic_baselines/<dataset_tag>/<ticker>_strategy_grid_summary.csv
    Portfolio:
        outputs/run_registry/algorithmic_baselines/<dataset_tag>/portfolio_strategy_summary.csv
    Combined:
        outputs/run_registry/algorithmic_baselines/<dataset_tag>/algorithmic_baselines_summary.csv

Usage:
    python -m stock_investment_dss.algorithmic_trading.experiments.run_all_algorithmic_experiments \\
        --trade-data data/market/daily/imports/market_data_demo10_new_2010_2026.csv \\
        --dataset-tag demo_10_new \\
        --ticker KO

    # Run all tickers
    python -m stock_investment_dss.algorithmic_trading.experiments.run_all_algorithmic_experiments \\
        --trade-data data/market/daily/imports/market_data_demo10_new_2010_2026.csv \\
        --dataset-tag demo_10_new
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from stock_investment_dss.algorithmic_trading.baselines.bollinger_mean_reversion_pit_single_ticker import (
    run_bollinger_mean_reversion,
)
from stock_investment_dss.algorithmic_trading.baselines.breakout_pit_single_ticker import (
    run_breakout,
)
from stock_investment_dss.algorithmic_trading.baselines.buy_and_hold_pit_single_ticker import (
    run_buy_and_hold,
)
from stock_investment_dss.algorithmic_trading.baselines.ema_crossover_pit_single_ticker import (
    run_ema_crossover,
)
from stock_investment_dss.algorithmic_trading.baselines.equal_weight_buy_and_hold_pit_portfolio import (
    run_equal_weight_buy_and_hold_portfolio,
)
from stock_investment_dss.algorithmic_trading.baselines.macd_signal_pit_single_ticker import (
    run_macd_signal,
)
from stock_investment_dss.algorithmic_trading.baselines.momentum_pit_single_ticker import (
    run_momentum,
)
from stock_investment_dss.algorithmic_trading.baselines.naive_one_over_n_rebalanced_pit_portfolio import (
    run_naive_one_over_n_rebalanced,
)
from stock_investment_dss.algorithmic_trading.baselines.rsi_mean_reversion_pit_single_ticker import (
    run_rsi_mean_reversion,
)
from stock_investment_dss.algorithmic_trading.baselines.sma_crossover_pit_single_ticker import (
    run_sma_crossover,
)
from stock_investment_dss.algorithmic_trading.baselines.volatility_filter_pit_single_ticker import (
    run_volatility_filter,
)
from stock_investment_dss.utilities.paths import create_run_paths, RunPaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fix 1: CSV read-through cache
# ---------------------------------------------------------------------------
# Strategy functions call pd.read_csv internally and cannot be modified.
# Patching pd.read_csv here means every strategy call for the same path
# returns a .copy() of the DataFrame already in memory — zero disk I/O.
# With 467 tickers × 24 strategies = 11,208 calls, this eliminates ~56 min
# of redundant reads (300 ms × 11,208 ≈ 56 min → < 1 s).

_CSV_CACHE: Dict[str, "pd.DataFrame"] = {}


def _install_csv_cache() -> None:
    """Patch pd.read_csv to serve results from an in-process cache.

    Call once in main() *before* the grid runs.  The first pd.read_csv call
    for a given resolved file path reads from disk and populates the cache;
    all subsequent calls return a .copy() (filtered to usecols if given).
    Fork-spawned worker processes inherit the patched function and the
    populated cache via copy-on-write — no disk reads in workers.
    """
    if getattr(pd, "_csv_cache_installed", False):
        return  # idempotent
    _orig = pd.read_csv

    def _cached_read_csv(filepath_or_buffer, *args, usecols=None, **kwargs):
        if not isinstance(filepath_or_buffer, (str, Path)):
            # StringIO or other file-like — not cacheable
            return _orig(filepath_or_buffer, *args, usecols=usecols, **kwargs)
        key = str(Path(filepath_or_buffer).resolve())
        if key not in _CSV_CACHE:
            _CSV_CACHE[key] = _orig(filepath_or_buffer, *args, **kwargs)
        cached = _CSV_CACHE[key]
        if usecols is not None:
            return cached[list(usecols)].copy()
        return cached.copy()

    pd.read_csv = _cached_read_csv  # type: ignore[assignment]
    pd._csv_cache_installed = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Audited 24 single-ticker configurations (V1 evidence confirmed)
# ---------------------------------------------------------------------------
# Source audit:
#   external/ObjectRL_style/outputs/runs/
#       test_algorithmic_trading_2025_grid_final/
#       algorithmic_trading/summary/algorithmic_trading_summary.md
#
# The candidate grid matches the audited V1 evidence exactly.
# ---------------------------------------------------------------------------

SINGLE_TICKER_CONFIGS: List[Dict[str, Any]] = [
    # --- buy-and-hold (1 config) -------------------------------------------
    {
        "fn": run_buy_and_hold,
        "params": {},
        "folder": "buy_and_hold",
        "type": "buy_and_hold",
    },
    # --- SMA crossover (4 configs) -----------------------------------------
    {
        "fn": run_sma_crossover,
        "params": {"fast_window": 5, "slow_window": 20},
        "folder": "sma_5_20",
        "type": "sma_crossover",
    },
    {
        "fn": run_sma_crossover,
        "params": {"fast_window": 10, "slow_window": 30},
        "folder": "sma_10_30",
        "type": "sma_crossover",
    },
    {
        "fn": run_sma_crossover,
        "params": {"fast_window": 20, "slow_window": 50},
        "folder": "sma_20_50",
        "type": "sma_crossover",
    },
    {
        "fn": run_sma_crossover,
        "params": {"fast_window": 50, "slow_window": 200},
        "folder": "sma_50_200",
        "type": "sma_crossover",
    },
    # --- EMA crossover (3 configs) -----------------------------------------
    {
        "fn": run_ema_crossover,
        "params": {"fast_window": 5, "slow_window": 20},
        "folder": "ema_5_20",
        "type": "ema_crossover",
    },
    {
        "fn": run_ema_crossover,
        "params": {"fast_window": 12, "slow_window": 26},
        "folder": "ema_12_26",
        "type": "ema_crossover",
    },
    {
        "fn": run_ema_crossover,
        "params": {"fast_window": 20, "slow_window": 50},
        "folder": "ema_20_50",
        "type": "ema_crossover",
    },
    # --- Momentum (4 configs) -----------------------------------------------
    {
        "fn": run_momentum,
        "params": {"lookback_window": 5},
        "folder": "momentum_5",
        "type": "momentum",
    },
    {
        "fn": run_momentum,
        "params": {"lookback_window": 10},
        "folder": "momentum_10",
        "type": "momentum",
    },
    {
        "fn": run_momentum,
        "params": {"lookback_window": 20},
        "folder": "momentum_20",
        "type": "momentum",
    },
    {
        "fn": run_momentum,
        "params": {"lookback_window": 60},
        "folder": "momentum_60",
        "type": "momentum",
    },
    # --- RSI mean-reversion (2 configs) ------------------------------------
    {
        "fn": run_rsi_mean_reversion,
        "params": {"rsi_window": 14, "oversold": 30.0, "overbought": 70.0},
        "folder": "rsi_14_30_70",
        "type": "rsi_mean_reversion",
    },
    {
        "fn": run_rsi_mean_reversion,
        "params": {"rsi_window": 14, "oversold": 25.0, "overbought": 75.0},
        "folder": "rsi_14_25_75",
        "type": "rsi_mean_reversion",
    },
    # --- MACD signal (2 configs) -------------------------------------------
    {
        "fn": run_macd_signal,
        "params": {"fast_window": 8, "slow_window": 21, "signal_window": 9},
        "folder": "macd_8_21_9",
        "type": "macd_signal",
    },
    {
        "fn": run_macd_signal,
        "params": {"fast_window": 12, "slow_window": 26, "signal_window": 9},
        "folder": "macd_12_26_9",
        "type": "macd_signal",
    },
    # --- Bollinger mean-reversion (2 configs) ------------------------------
    {
        "fn": run_bollinger_mean_reversion,
        "params": {"window": 20, "num_std": 2.0, "force_recompute_bands": True},
        "folder": "bollinger_mr_20_2",
        "type": "bollinger_mean_reversion",
    },
    {
        "fn": run_bollinger_mean_reversion,
        "params": {"window": 20, "num_std": 2.5, "force_recompute_bands": True},
        "folder": "bollinger_mr_20_2.5",
        "type": "bollinger_mean_reversion",
    },
    # --- Breakout (3 configs) -----------------------------------------------
    {
        "fn": run_breakout,
        "params": {"lookback_window": 10},
        "folder": "breakout_10",
        "type": "breakout",
    },
    {
        "fn": run_breakout,
        "params": {"lookback_window": 20},
        "folder": "breakout_20",
        "type": "breakout",
    },
    {
        "fn": run_breakout,
        "params": {"lookback_window": 55},
        "folder": "breakout_55",
        "type": "breakout",
    },
    # --- Volatility filter (3 configs) -------------------------------------
    {
        "fn": run_volatility_filter,
        "params": {
            "momentum_window": 5,
            "volatility_window": 10,
            "max_annualized_volatility": 0.4,
        },
        "folder": "vol_filter_m5_v10_0.4",
        "type": "volatility_filter",
    },
    {
        "fn": run_volatility_filter,
        "params": {
            "momentum_window": 10,
            "volatility_window": 20,
            "max_annualized_volatility": 0.4,
        },
        "folder": "vol_filter_m10_v20_0.4",
        "type": "volatility_filter",
    },
    {
        "fn": run_volatility_filter,
        "params": {
            "momentum_window": 20,
            "volatility_window": 20,
            "max_annualized_volatility": 0.4,
        },
        "folder": "vol_filter_m20_v20_0.4",
        "type": "volatility_filter",
    },
]

# ---------------------------------------------------------------------------
# Portfolio-level configurations (2 configs)
# ---------------------------------------------------------------------------

PORTFOLIO_CONFIGS: List[Dict[str, Any]] = [
    {
        "fn": run_equal_weight_buy_and_hold_portfolio,
        "params": {},
        "folder": "equal_weight_buy_and_hold",
        "type": "equal_weight_buy_and_hold",
    },
    {
        "fn": run_naive_one_over_n_rebalanced,
        "params": {"rebalance_frequency_days": 21},
        "folder": "naive_one_over_n_rebalanced_21d",
        "type": "naive_one_over_n_rebalanced",
    },
]


# ---------------------------------------------------------------------------
# Ticker discovery
# ---------------------------------------------------------------------------


def discover_tickers(trade_data: str | Path) -> List[str]:
    """Return sorted list of unique tickers found in the trade data CSV."""
    df = pd.read_csv(trade_data, usecols=["tic"])
    return sorted(df["tic"].astype(str).str.upper().unique().tolist())


# ---------------------------------------------------------------------------
# Summary CSV helpers
# ---------------------------------------------------------------------------

_METRICS_COLUMNS_MAP = {
    "total_return_pct": "total_return_pct",
    "max_drawdown_pct": "max_drawdown_pct",
    "annualized_sharpe": "annualized_sharpe",
    "days": "days",
}


def _read_metrics_row(metrics_path: Optional[Path]) -> Dict[str, Any]:
    """Read the first row of a metrics CSV and return a dict of key values."""
    empty = {k: None for k in _METRICS_COLUMNS_MAP}
    if metrics_path is None or not Path(metrics_path).exists():
        return empty
    try:
        df = pd.read_csv(metrics_path)
        if df.empty:
            return empty
        row = df.iloc[0].to_dict()
        return {local: row.get(src) for local, src in _METRICS_COLUMNS_MAP.items()}
    except Exception:  # noqa: BLE001
        return empty


def _write_per_ticker_summary(
    records: List[Dict[str, Any]], run_paths: RunPaths
) -> Optional[Path]:
    """Group records by ticker and write per-ticker summary CSVs."""
    if not records:
        return None
    df = pd.DataFrame(records)
    summary_dir = run_paths.summary_directory
    summary_dir.mkdir(parents=True, exist_ok=True)
    for ticker, grp in df.groupby("ticker", sort=True):
        out_path = summary_dir / f"{str(ticker).lower()}_strategy_grid_summary.csv"
        grp.sort_values("total_return_pct", ascending=False, na_position="last").to_csv(
            out_path, index=False
        )
        log.info("Wrote per-ticker summary: %s", out_path)
    return summary_dir


def _write_portfolio_summary(
    records: List[Dict[str, Any]], run_paths: RunPaths
) -> Optional[Path]:
    if not records:
        return None
    df = pd.DataFrame(records)
    out_path = run_paths.summary_directory / "portfolio_strategy_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("total_return_pct", ascending=False, na_position="last").to_csv(
        out_path, index=False
    )
    log.info("Wrote portfolio summary: %s", out_path)
    return out_path


def _write_combined_summary(
    ticker_records: List[Dict[str, Any]],
    portfolio_records: List[Dict[str, Any]],
    run_paths: RunPaths,
) -> Optional[Path]:
    # Fix 3: vectorised row construction via pd.concat instead of a Python loop.
    if not ticker_records and not portfolio_records:
        return None
    dfs: List[pd.DataFrame] = []
    if ticker_records:
        df_t = pd.DataFrame(ticker_records)
        df_t.insert(0, "scope", "single_ticker")
        dfs.append(df_t)
    if portfolio_records:
        df_p = pd.DataFrame(portfolio_records)
        df_p.insert(0, "scope", "portfolio")
        df_p["ticker"] = "portfolio"
        dfs.append(df_p)
    df = pd.concat(dfs, ignore_index=True)
    out_path = run_paths.summary_directory / "algorithmic_baselines_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("total_return_pct", ascending=False, na_position="last").to_csv(
        out_path, index=False
    )
    log.info("Wrote combined summary: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Fix 2: top-level worker function (must be at module scope for pickling)
# ---------------------------------------------------------------------------


def _run_ticker_task(args: tuple) -> List[Dict[str, Any]]:
    """Run all single-ticker configs for one ticker.

    Receives a plain tuple so multiprocessing.Pool.map can call it without
    **kwargs.  When dispatched by ProcessPoolExecutor with fork context, the
    worker process inherits the patched pd.read_csv and the pre-populated
    _CSV_CACHE from the parent — no disk reads occur here.
    """
    (
        ticker,
        trade_data_str,
        dataset_tag,
        initial_amount,
        skip_buy_and_hold,
        continue_on_error,
        run_paths,
        pit_start_date,
        pit_end_date,
    ) = args
    records: List[Dict[str, Any]] = []
    for cfg in SINGLE_TICKER_CONFIGS:
        if skip_buy_and_hold and cfg["folder"] == "buy_and_hold":
            continue
        folder = cfg["folder"]
        fn = cfg["fn"]
        params = cfg["params"]
        log.info("[single-ticker] %s / %s ...", ticker, folder)
        try:
            result = fn(
                trade_data=trade_data_str,  # path string; pd.read_csv hits cache
                dataset_tag=dataset_tag,
                ticker=ticker,
                initial_amount=initial_amount,
                strategy_folder=folder,
                run_paths=run_paths,
                output_subpath=f"algorithmic_baselines/{folder}/{ticker.lower()}",
                pit_start_date=pit_start_date,
                pit_end_date=pit_end_date,
                **params,
            )
            m = _read_metrics_row(result.get("metrics"))
            records.append(
                {
                    "ticker": ticker.lower(),
                    "strategy_type": cfg["type"],
                    "config_label": folder,
                    "strategy_folder": folder,
                    "status": "ok",
                    "source_metrics_csv": str(result.get("metrics", "")),
                    **m,
                }
            )
        except Exception as exc:  # noqa: BLE001
            if continue_on_error:
                log.warning(
                    "ERROR %s / %s: %s",
                    ticker,
                    folder,
                    exc,
                    exc_info=True,
                )
                records.append(
                    {
                        "ticker": ticker.lower(),
                        "strategy_type": cfg["type"],
                        "config_label": folder,
                        "strategy_folder": folder,
                        "status": "error",
                        "total_return_pct": None,
                        "max_drawdown_pct": None,
                        "annualized_sharpe": None,
                        "days": None,
                        "source_metrics_csv": None,
                    }
                )
            else:
                raise
    return records


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def _run_single_ticker_grid(
    *,
    trade_data: str | Path,
    dataset_tag: str,
    tickers: List[str],
    initial_amount: float,
    configs: List[Dict[str, Any]],
    skip_buy_and_hold: bool,
    continue_on_error: bool,
    run_paths: RunPaths,
    pit_start_date: str | None = None,
    pit_end_date: str | None = None,
    workers: int = 1,
) -> List[Dict[str, Any]]:
    """Run the full single-ticker strategy grid.

    When workers > 1, each ticker is dispatched to a separate worker process
    (fork context, inheriting the CSV cache) so all available CPU cores are
    utilised.  The CSV cache installed in main() ensures workers never read
    from disk.
    """
    trade_data_str = str(Path(trade_data).resolve())
    tasks = [
        (
            ticker,
            trade_data_str,
            dataset_tag,
            initial_amount,
            skip_buy_and_hold,
            continue_on_error,
            run_paths,
            pit_start_date,
            pit_end_date,
        )
        for ticker in tickers
    ]
    if workers > 1:
        log.info(
            "Parallel dispatch: %d tickers × %d configs on %d workers",
            len(tickers),
            len(SINGLE_TICKER_CONFIGS),
            workers,
        )
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            nested = list(executor.map(_run_ticker_task, tasks))
    else:
        log.info(
            "Sequential dispatch: %d tickers × %d configs",
            len(tickers),
            len(SINGLE_TICKER_CONFIGS),
        )
        nested = [_run_ticker_task(t) for t in tasks]
    return [record for ticker_records in nested for record in ticker_records]


def _run_portfolio_grid(
    *,
    trade_data: str | Path,
    dataset_tag: str,
    initial_amount: float,
    rebalance_frequency_days: int,
    configs: List[Dict[str, Any]],
    continue_on_error: bool,
    run_paths: RunPaths,
    pit_start_date: str | None = None,
    pit_end_date: str | None = None,
) -> List[Dict[str, Any]]:
    records = []
    for cfg in configs:
        folder = cfg["folder"]
        fn = cfg["fn"]
        params = dict(cfg["params"])
        if cfg["type"] == "naive_one_over_n_rebalanced":
            params.setdefault("rebalance_frequency_days", rebalance_frequency_days)
        log.info("[portfolio] %s ...", folder)
        try:
            result = fn(
                trade_data=trade_data,
                dataset_tag=dataset_tag,
                initial_amount=initial_amount,
                strategy_folder=folder,
                run_paths=run_paths,
                output_subpath=f"algorithmic_baselines/{folder}",
                pit_start_date=pit_start_date,
                pit_end_date=pit_end_date,
                **params,
            )
            m = _read_metrics_row(result.get("metrics"))
            records.append(
                {
                    "strategy_type": cfg["type"],
                    "config_label": folder,
                    "strategy_folder": folder,
                    "status": "ok",
                    "source_metrics_csv": str(result.get("metrics", "")),
                    **m,
                }
            )
        except Exception as exc:  # noqa: BLE001
            if continue_on_error:
                log.warning("ERROR portfolio / %s: %s", folder, exc, exc_info=True)
                records.append(
                    {
                        "strategy_type": cfg["type"],
                        "config_label": folder,
                        "strategy_folder": folder,
                        "status": "error",
                        "total_return_pct": None,
                        "max_drawdown_pct": None,
                        "annualized_sharpe": None,
                        "days": None,
                        "source_metrics_csv": None,
                    }
                )
            else:
                raise
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the full algorithmic baseline grid and write aggregated summary CSVs."
        )
    )
    p.add_argument(
        "--trade-data",
        required=True,
        metavar="PATH",
        help="Path to the trade data CSV.",
    )
    p.add_argument(
        "--dataset-tag",
        required=True,
        metavar="STR",
        help="Short dataset identifier used in output paths.",
    )
    p.add_argument(
        "--ticker",
        metavar="STR",
        default="ALL",
        help="Ticker to run (uppercase). Use ALL to run every ticker in the dataset.",
    )
    p.add_argument(
        "--initial-amount",
        type=float,
        default=1_000_000.0,
        metavar="FLOAT",
        help="Initial portfolio capital. Default: 1,000,000.",
    )
    p.add_argument(
        "--rebalance-frequency-days",
        type=int,
        default=21,
        metavar="INT",
        help="Rebalance frequency for the 1/N rebalanced baseline. Default: 21.",
    )
    p.add_argument(
        "--single-ticker-only",
        action="store_true",
        help="Skip all portfolio-level baseline configurations.",
    )
    p.add_argument(
        "--portfolio-only",
        action="store_true",
        help="Skip all single-ticker configurations.",
    )
    p.add_argument(
        "--skip-buy-and-hold",
        action="store_true",
        help="Skip the buy-and-hold entry from the single-ticker grid.",
    )
    p.add_argument(
        "--skip-portfolio-baselines",
        action="store_true",
        help="Alias for --single-ticker-only.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log errors and continue rather than raising on first failure.",
    )
    p.add_argument(
        "--pit-start",
        default=None,
        metavar="DATE",
        help="PIT trade window start date (ISO format, e.g. 2024-01-01).",
    )
    p.add_argument(
        "--pit-end",
        default=None,
        metavar="DATE",
        help="PIT trade window end date, exclusive (ISO format, e.g. 2026-12-31).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Number of parallel worker processes for the single-ticker grid. "
            "Default: os.cpu_count()-1 (leaves one core free). "
            "Use 1 for fully sequential execution."
        ),
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    pit_start_date = args.pit_start or os.environ.get(
        "STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME"
    )
    pit_end_date = args.pit_end or os.environ.get(
        "STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE"
    )

    trade_data = Path(args.trade_data)
    if not trade_data.exists():
        log.error("Trade data not found: %s", trade_data)
        return 1

    # Fix 1: install the CSV cache BEFORE any pd.read_csv call so every
    # subsequent read (including those inside strategy functions and
    # trading_metrics.py) returns a .copy() from memory — zero disk I/O.
    _install_csv_cache()

    # Pre-load the full market data CSV once.  This single disk read populates
    # the cache; all 11,208 strategy calls then hit the in-memory cache.
    log.info("Pre-loading market data: %s", trade_data)
    trade_df = pd.read_csv(trade_data)
    log.info("Cache warm: %d rows × %d cols", len(trade_df), len(trade_df.columns))

    # Fix 2: worker count — default to cpu_count()-1, minimum 1.
    workers = (
        args.workers if args.workers is not None else max(1, (os.cpu_count() or 1) - 1)
    )
    log.info("Workers: %d", workers)

    # Discover tickers from the pre-loaded DataFrame (no second disk read).
    if args.ticker.upper() == "ALL":
        tickers = sorted(trade_df["tic"].astype(str).str.upper().unique().tolist())
        log.info("Discovered %d tickers", len(tickers))
    elif "," in args.ticker:
        tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]
    else:
        tickers = [args.ticker.upper()]

    run_single = not args.portfolio_only
    run_portfolio = not args.single_ticker_only and not args.skip_portfolio_baselines

    shared_run_paths = create_run_paths(
        f"d_iqn_dss_algorithmic_baseline_grid_{args.dataset_tag}"
    )
    shared_run_paths.logs_directory.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        shared_run_paths.logs_directory / "grid_runner.log", encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
        )
    )
    logging.getLogger().addHandler(file_handler)

    import json as _json

    shared_run_paths.config_directory.mkdir(parents=True, exist_ok=True)
    grid_config = {
        "dataset_tag": args.dataset_tag,
        "trade_data": str(trade_data),
        "tickers": tickers if args.ticker.upper() == "ALL" else [args.ticker.upper()],
        "initial_amount": args.initial_amount,
        "run_single": run_single,
        "run_portfolio": run_portfolio,
        "skip_buy_and_hold": args.skip_buy_and_hold,
        "rebalance_frequency_days": args.rebalance_frequency_days,
        "pit_start_date": pit_start_date,
        "pit_end_date": pit_end_date,
    }
    with open(shared_run_paths.config_directory / "grid_config.json", "w") as _fh:
        _json.dump(grid_config, _fh, indent=2)
    log.info("Grid run directory: %s", shared_run_paths.run_directory)

    ticker_records: List[Dict[str, Any]] = []
    portfolio_records: List[Dict[str, Any]] = []

    if run_single:
        ticker_records = _run_single_ticker_grid(
            trade_data=trade_data,
            dataset_tag=args.dataset_tag,
            tickers=tickers,
            initial_amount=args.initial_amount,
            configs=SINGLE_TICKER_CONFIGS,
            skip_buy_and_hold=args.skip_buy_and_hold,
            continue_on_error=args.continue_on_error,
            run_paths=shared_run_paths,
            pit_start_date=pit_start_date,
            pit_end_date=pit_end_date,
            workers=workers,
        )

    if run_portfolio:
        portfolio_records = _run_portfolio_grid(
            trade_data=trade_data,
            dataset_tag=args.dataset_tag,
            initial_amount=args.initial_amount,
            rebalance_frequency_days=args.rebalance_frequency_days,
            configs=PORTFOLIO_CONFIGS,
            continue_on_error=args.continue_on_error,
            run_paths=shared_run_paths,
            pit_start_date=pit_start_date,
            pit_end_date=pit_end_date,
        )

    if ticker_records:
        _write_per_ticker_summary(ticker_records, shared_run_paths)
    if portfolio_records:
        _write_portfolio_summary(portfolio_records, shared_run_paths)
    if ticker_records or portfolio_records:
        _write_combined_summary(ticker_records, portfolio_records, shared_run_paths)

    n_ok = sum(1 for r in ticker_records + portfolio_records if r.get("status") == "ok")
    n_err = sum(
        1 for r in ticker_records + portfolio_records if r.get("status") == "error"
    )
    log.info("Grid runner complete. OK=%d  ERROR=%d", n_ok, n_err)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
