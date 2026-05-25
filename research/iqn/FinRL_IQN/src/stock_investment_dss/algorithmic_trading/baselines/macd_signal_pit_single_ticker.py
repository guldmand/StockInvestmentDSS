"""
MACD signal point-in-time baseline for a single ticker.

Signal logic:
    - Compute the MACD line (fast EMA minus slow EMA) from close prices.
    - Compute the MACD signal line (EMA of the MACD line).
    - If MACD line > MACD signal line, target position is 100% invested.
    - Otherwise, target position is 100% cash.
    - Position is shifted by one trading day to avoid same-day look-ahead.

Point-in-time safety:
    All EMA computations use min_periods equal to their respective windows.
    The final position shift(1) ensures no day-t information is used to
    determine the position on day t.

Output contract:
    Writes four files to:
        outputs/run_registry/algorithmic_baselines/<dataset_tag>/macd_signal/<ticker>/

    - <ticker>_macd_signal_account_values.csv   — account value time series with signals
    - <ticker>_macd_signal_metrics.csv          — performance metrics (Etape 2 canonical)
    - <ticker>_macd_signal_config.json          — run parameters
    - <ticker>_macd_signal_account_value.png    — portfolio value plot

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


def run_macd_signal(
    *,
    trade_data: str | Path,
    dataset_tag: str,
    ticker: str,
    initial_amount: float = 1_000_000.0,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
    transaction_cost_pct: float = 0.001,
    pit_start_date: str | None = None,
    pit_end_date: str | None = None,
    strategy_folder: str | None = None,
    run_paths: Optional[RunPaths] = None,
    output_subpath: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Run MACD signal crossover for one ticker and write outputs to the V2 canonical
    run directory.

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
        Span for the fast EMA (must be < slow_window).
    slow_window:
        Span for the slow EMA.
    signal_window:
        Span for the MACD signal line EMA.

    Returns
    -------
    dict[str, Path]
        Keys: ``account``, ``metrics``, ``config``, ``plot``.
    """
    if min(fast_window, slow_window, signal_window) <= 0:
        raise ValueError("fast_window, slow_window and signal_window must be positive.")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window.")
    if transaction_cost_pct < 0:
        raise ValueError("transaction_cost_pct must be non-negative.")

    ticker = ticker.upper()
    df = load_trade_data_single_ticker(trade_data, ticker)

    fast_ema = (
        df["close"].ewm(span=fast_window, adjust=False, min_periods=fast_window).mean()
    )
    slow_ema = (
        df["close"].ewm(span=slow_window, adjust=False, min_periods=slow_window).mean()
    )
    df["macd_line"] = fast_ema - slow_ema
    df["macd_signal_line"] = (
        df["macd_line"]
        .ewm(
            span=signal_window,
            adjust=False,
            min_periods=signal_window,
        )
        .mean()
    )
    df["macd_histogram"] = df["macd_line"] - df["macd_signal_line"]

    df["signal"] = (df["macd_line"] > df["macd_signal_line"]).astype(int)
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
            "macd_line": df["macd_line"],
            "macd_signal_line": df["macd_signal_line"],
            "macd_histogram": df["macd_histogram"],
            "signal": df["signal"],
            "position": df["position"],
            "account_value": account_value,
        }
    )

    if run_paths is None:
        _folder = (
            strategy_folder
            if (strategy_folder and strategy_folder.strip())
            else "macd_signal"
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

    strategy = f"{ticker}_macd_{fast_window}_{slow_window}_{signal_window}"
    metrics = calculate_account_metrics(
        account,
        strategy=strategy,
        source="Algorithmic Trading / non-RL",
        initial_amount=initial_amount,
    )

    config = {
        "strategy": strategy,
        "dataset_tag": dataset_tag,
        "ticker": ticker,
        "trade_data": str(trade_data),
        "initial_amount": float(initial_amount),
        "fast_window": int(fast_window),
        "slow_window": int(slow_window),
        "signal_window": int(signal_window),
        "transaction_cost_pct": float(transaction_cost_pct),
        "pit_start_date": pit_start_date,
        "pit_end_date": pit_end_date,
        "signal_rule": (
            "invested if MACD line > MACD signal line; otherwise cash;"
            " signal shifted by 1 day"
        ),
    }

    account_path = data_dir / f"{ticker.lower()}_macd_signal_account_values.csv"
    metrics_path = metrics_dir / f"{ticker.lower()}_macd_signal_metrics.csv"
    config_path = config_dir / f"{ticker.lower()}_macd_signal_config.json"
    plot_path = plots_dir / f"{ticker.lower()}_macd_signal_account_value.png"

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
