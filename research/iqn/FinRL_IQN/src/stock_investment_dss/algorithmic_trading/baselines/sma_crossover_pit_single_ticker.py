"""
SMA crossover point-in-time baseline for a single ticker.

Signal logic:
    - Compute fast and slow simple moving averages from close prices.
    - If fast SMA > slow SMA, target position is 100% invested.
    - Otherwise, target position is 100% cash.
    - Position is shifted by one trading day to avoid same-day look-ahead.

Point-in-time safety:
    rolling().mean() with min_periods=window and a subsequent shift(1) ensures
    that the position on day t uses only information available up to day t-1.

Transaction cost:
    Optional proportional cost applied to each change in position.
    Default is 0.0 (no transaction cost).

Output contract:
    Writes four files to:
        outputs/run_registry/algorithmic_baselines/<dataset_tag>/sma_crossover/<ticker>/

    - <ticker>_sma_crossover_account_values.csv   — account value time series with signals
    - <ticker>_sma_crossover_metrics.csv          — performance metrics (Etape 2 canonical)
    - <ticker>_sma_crossover_config.json          — run parameters
    - <ticker>_sma_crossover_account_value.png    — portfolio value plot

    Performance metrics are computed via
    stock_investment_dss.metrics.trading_metrics.calculate_account_metrics,
    which wraps the canonical V2 evaluation module.  No metric formulas are
    duplicated in this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from stock_investment_dss.metrics.trading_metrics import (
    calculate_account_metrics,
    load_trade_data_single_ticker,
    save_account_value_plot,
)
from stock_investment_dss.utilities.paths import create_run_paths, RunPaths


def run_sma_crossover(
    *,
    trade_data: str | Path,
    dataset_tag: str,
    ticker: str,
    initial_amount: float = 1_000_000.0,
    fast_window: int = 50,
    slow_window: int = 200,
    transaction_cost_pct: float = 0.001,
    pit_start_date: str | None = None,
    pit_end_date: str | None = None,
    strategy_folder: str | None = None,
    run_paths: Optional[RunPaths] = None,
    output_subpath: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Run SMA crossover for one ticker and write outputs to the V2 canonical run directory.

    Parameters
    ----------
    trade_data:
        Path to a CSV file with columns ``date``, ``tic``, ``close``.
    dataset_tag:
        Short identifier for the dataset, used in the output directory path.
    ticker:
        Ticker symbol (case-insensitive).
    initial_amount:
        Initial portfolio capital.
    fast_window:
        Lookback window for the fast SMA (must be < slow_window).
    slow_window:
        Lookback window for the slow SMA.
    transaction_cost_pct:
        Proportional one-way transaction cost.  Default 0.0.

    Returns
    -------
    dict[str, Path]
        Keys: ``account``, ``metrics``, ``config``, ``plot``.
    """
    if fast_window <= 0:
        raise ValueError("fast_window must be positive.")
    if slow_window <= 0:
        raise ValueError("slow_window must be positive.")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window.")
    if transaction_cost_pct < 0:
        raise ValueError("transaction_cost_pct must be non-negative.")

    ticker = ticker.upper()
    df = load_trade_data_single_ticker(trade_data, ticker)

    df["fast_sma"] = (
        df["close"].rolling(window=fast_window, min_periods=fast_window).mean()
    )
    df["slow_sma"] = (
        df["close"].rolling(window=slow_window, min_periods=slow_window).mean()
    )
    df["signal"] = (df["fast_sma"] > df["slow_sma"]).astype(int)

    # Shift by one day to avoid look-ahead bias.
    df["position"] = df["signal"].shift(1).fillna(0.0)

    if pit_start_date is not None:
        df = df[df["date"] >= pd.Timestamp(pit_start_date)].copy()
    if pit_end_date is not None:
        df = df[df["date"] < pd.Timestamp(pit_end_date)].copy()

    returns = df["close"].pct_change().fillna(0.0)
    position_change = df["position"].diff().abs().fillna(df["position"].abs())
    strategy_return = (df["position"] * returns) - (
        position_change * float(transaction_cost_pct)
    )
    account_value = float(initial_amount) * (1.0 + strategy_return).cumprod()

    account = pd.DataFrame(
        {
            "date": df["date"].dt.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "close": df["close"],
            "fast_sma": df["fast_sma"],
            "slow_sma": df["slow_sma"],
            "signal": df["signal"],
            "position": df["position"],
            "strategy_return": strategy_return,
            "account_value": account_value,
        }
    )

    # Force exact first value for clean comparison.
    account.loc[account.index[0], "account_value"] = float(initial_amount)

    if run_paths is None:
        _folder = (
            strategy_folder
            if (strategy_folder and strategy_folder.strip())
            else "sma_crossover"
        )
        run_name = (
            f"d_iqn_dss_algorithmic_baseline_{_folder}_{dataset_tag}_{ticker.lower()}"
        )
        run_paths = create_run_paths(run_name)
    _sub = Path(output_subpath) if output_subpath else Path("")
    data_dir = run_paths.data_directory / _sub
    metrics_dir = run_paths.metrics_directory / _sub
    config_dir = run_paths.config_directory / _sub
    plots_dir = run_paths.plots_directory / _sub
    for d in [data_dir, metrics_dir, config_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    strategy = f"{ticker}_sma_{fast_window}_{slow_window}"
    source = "Algorithmic Trading / non-RL"

    metrics = calculate_account_metrics(
        account,
        strategy=strategy,
        source=source,
        initial_amount=initial_amount,
    )

    config = {
        "strategy": strategy,
        "source": source,
        "dataset_tag": dataset_tag,
        "ticker": ticker,
        "trade_data": str(trade_data),
        "initial_amount": float(initial_amount),
        "fast_window": int(fast_window),
        "slow_window": int(slow_window),
        "transaction_cost_pct": float(transaction_cost_pct),
        "pit_start_date": pit_start_date,
        "pit_end_date": pit_end_date,
        "signal_rule": (
            "invested if fast SMA > slow SMA; otherwise cash;"
            " signal shifted by 1 day"
        ),
    }

    account_path = data_dir / f"{ticker.lower()}_sma_crossover_account_values.csv"
    metrics_path = metrics_dir / f"{ticker.lower()}_sma_crossover_metrics.csv"
    config_path = config_dir / f"{ticker.lower()}_sma_crossover_config.json"
    plot_path = plots_dir / f"{ticker.lower()}_sma_crossover_account_value.png"

    account.to_csv(account_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    save_account_value_plot(
        account, output_path=plot_path, title=f"{strategy} — account value"
    )

    return {
        "account": account_path,
        "metrics": metrics_path,
        "config": config_path,
        "plot": plot_path,
    }
