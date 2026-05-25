"""
Breakout point-in-time baseline for a single ticker.

Signal logic:
    - Compute a rolling maximum (resistance) and rolling minimum (support)
      of close prices over a lookback window.
    - If today's close breaks above the prior-period rolling high, enter a
      long position (invested).
    - If today's close breaks below the prior-period rolling low, exit to cash.
    - Between breakout levels, hold the previous position.
    - Rolling high and low are computed up to day t-1 (shift(1)) to avoid
      same-day look-ahead, and the position itself is also shifted by 1 day.

Point-in-time safety:
    rolling().max() and rolling().min() use min_periods=lookback_window so that
    levels are NaN until a full window is available.  The .shift(1) on the
    rolling series guarantees that no day-t close price is used in the
    resistance/support levels for day t.  The position shift(1) provides an
    additional one-day lag on the signal itself.

Output contract:
    Writes four files to:
        outputs/run_registry/algorithmic_baselines/<dataset_tag>/breakout/<ticker>/

    - <ticker>_breakout_account_values.csv   — time series with breakout levels and signals
    - <ticker>_breakout_metrics.csv          — performance metrics (Etape 2 canonical)
    - <ticker>_breakout_config.json          — run parameters
    - <ticker>_breakout_account_value.png    — portfolio value plot

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


def run_breakout(
    *,
    trade_data: str | Path,
    dataset_tag: str,
    ticker: str,
    initial_amount: float = 1_000_000.0,
    lookback_window: int = 20,
    strategy_folder: str | None = None,
    run_paths: Optional[RunPaths] = None,
    output_subpath: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Run a breakout strategy for one ticker and write outputs to the V2 canonical
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
    lookback_window:
        Rolling window used to compute the resistance (high) and support (low)
        levels.

    Returns
    -------
    dict[str, Path]
        Keys: ``account``, ``metrics``, ``config``, ``plot``.
    """
    if lookback_window <= 0:
        raise ValueError("lookback_window must be positive.")

    ticker = ticker.upper()
    df = load_trade_data_single_ticker(trade_data, ticker)

    df["rolling_high"] = (
        df["close"].rolling(lookback_window, min_periods=lookback_window).max().shift(1)
    )
    df["rolling_low"] = (
        df["close"].rolling(lookback_window, min_periods=lookback_window).min().shift(1)
    )

    signal = []
    current_position = 0
    for close, high, low in zip(df["close"], df["rolling_high"], df["rolling_low"]):
        if pd.notna(high) and close > high:
            current_position = 1
        elif pd.notna(low) and close < low:
            current_position = 0
        signal.append(current_position)

    df["signal"] = signal
    df["position"] = df["signal"].shift(1).fillna(0.0)

    returns = df["close"].pct_change().fillna(0.0)
    account_value = float(initial_amount) * (1.0 + df["position"] * returns).cumprod()

    account = pd.DataFrame(
        {
            "date": df["date"].dt.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "close": df["close"],
            "rolling_high": df["rolling_high"],
            "rolling_low": df["rolling_low"],
            "signal": df["signal"],
            "position": df["position"],
            "account_value": account_value,
        }
    )

    if run_paths is None:
        _folder = strategy_folder if (strategy_folder and strategy_folder.strip()) else "breakout"
        run_name = f"d_iqn_dss_algorithmic_baseline_{_folder}_{dataset_tag}_{ticker.lower()}"
        run_paths = create_run_paths(run_name)
    _sub = Path(output_subpath) if output_subpath else Path("")
    data_dir = run_paths.data_directory / _sub
    metrics_dir = run_paths.metrics_directory / _sub
    config_dir = run_paths.config_directory / _sub
    plots_dir = run_paths.plots_directory / _sub
    for d in [data_dir, metrics_dir, config_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    strategy = f"{ticker}_breakout_{lookback_window}"
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
        "lookback_window": int(lookback_window),
        "signal_rule": (
            "buy/invested if close breaks above prior rolling high;"
            " sell/cash if close breaks below prior rolling low;"
            " signal shifted by 1 day"
        ),
    }

    account_path = data_dir / f"{ticker.lower()}_breakout_account_values.csv"
    metrics_path = metrics_dir / f"{ticker.lower()}_breakout_metrics.csv"
    config_path = config_dir / f"{ticker.lower()}_breakout_config.json"
    plot_path = plots_dir / f"{ticker.lower()}_breakout_account_value.png"

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
