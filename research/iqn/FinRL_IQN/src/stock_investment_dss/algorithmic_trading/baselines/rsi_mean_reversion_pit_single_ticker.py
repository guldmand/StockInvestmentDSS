"""
RSI mean-reversion point-in-time baseline for a single ticker.

Signal logic:
    - Compute RSI from close prices using an exponential Wilder smoothing.
    - If RSI falls below ``oversold``, enter a long position (invested).
    - If RSI rises above ``overbought``, exit to cash.
    - Between thresholds, hold the previous position.
    - Position is shifted by one trading day to avoid same-day look-ahead.

Pre-computed column optimisation:
    If the input DataFrame contains an ``rsi_30`` column and ``rsi_window == 30``,
    the pre-computed series is used directly instead of recomputing RSI.
    This behaviour is data-aware and is preserved from the V1 implementation.
    If the column is absent or the window differs, RSI is computed from close prices.

Point-in-time safety:
    The Wilder EWM (ewm with alpha=1/window) uses min_periods=window so that
    the RSI series is NaN until a full window of data is available.  The
    position shift(1) prevents any day-t signal from affecting the day-t position.

Output contract:
    Writes four files to:
        outputs/run_registry/algorithmic_baselines/<dataset_tag>/rsi_mean_reversion/<ticker>/

    - <ticker>_rsi_mean_reversion_account_values.csv   — time series with RSI and signals
    - <ticker>_rsi_mean_reversion_metrics.csv          — performance metrics (Etape 2 canonical)
    - <ticker>_rsi_mean_reversion_config.json          — run parameters
    - <ticker>_rsi_mean_reversion_account_value.png    — portfolio value plot

    Performance metrics are computed via
    stock_investment_dss.metrics.trading_metrics.calculate_account_metrics,
    which wraps the canonical V2 evaluation module.  No metric formulas are
    duplicated in this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from stock_investment_dss.metrics.trading_metrics import (
    calculate_account_metrics,
    load_trade_data_single_ticker,
    save_account_value_plot,
)
from stock_investment_dss.utilities.paths import create_run_paths, RunPaths


def _compute_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def run_rsi_mean_reversion(
    *,
    trade_data: str | Path,
    dataset_tag: str,
    ticker: str,
    initial_amount: float = 1_000_000.0,
    rsi_window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    transaction_cost_pct: float = 0.001,
    strategy_folder: str | None = None,
    run_paths: Optional[RunPaths] = None,
    output_subpath: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Run RSI mean-reversion for one ticker and write outputs to the V2 canonical
    run directory.

    Parameters
    ----------
    trade_data:
        Path to a CSV file with columns ``date``, ``tic``, ``close``.
        An ``rsi_30`` column is used automatically when ``rsi_window == 30``.
    dataset_tag:
        Short identifier for the dataset, used in the output directory path.
    ticker:
        Ticker symbol (case-insensitive).
    initial_amount:
        Initial portfolio capital.
    rsi_window:
        RSI lookback window in trading days.
    oversold:
        RSI threshold below which the strategy enters a long position.
    overbought:
        RSI threshold above which the strategy exits to cash.

    Returns
    -------
    dict[str, Path]
        Keys: ``account``, ``metrics``, ``config``, ``plot``.
    """
    if rsi_window <= 0:
        raise ValueError("rsi_window must be positive.")
    if oversold >= overbought:
        raise ValueError("oversold must be smaller than overbought.")
    if transaction_cost_pct < 0:
        raise ValueError("transaction_cost_pct must be non-negative.")

    ticker = ticker.upper()
    df = load_trade_data_single_ticker(trade_data, ticker)

    if "rsi_30" in df.columns and int(rsi_window) == 30:
        df["rsi"] = df["rsi_30"].astype(float)
    else:
        df["rsi"] = _compute_rsi(df["close"], rsi_window)

    signal = []
    current_position = 0
    for value in df["rsi"]:
        if value < oversold:
            current_position = 1
        elif value > overbought:
            current_position = 0
        signal.append(current_position)

    df["signal"] = signal
    df["position"] = df["signal"].shift(1).fillna(0.0)

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
            "rsi": df["rsi"],
            "signal": df["signal"],
            "position": df["position"],
            "account_value": account_value,
        }
    )

    if run_paths is None:
        _folder = (
            strategy_folder
            if (strategy_folder and strategy_folder.strip())
            else "rsi_mean_reversion"
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

    strategy = f"{ticker}_rsi_{rsi_window}_{int(oversold)}_{int(overbought)}"
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
        "rsi_window": int(rsi_window),
        "oversold": float(oversold),
        "overbought": float(overbought),
        "transaction_cost_pct": float(transaction_cost_pct),
        "signal_rule": (
            "buy/invested if RSI < oversold; sell/cash if RSI > overbought;"
            " otherwise hold previous position; signal shifted by 1 day"
        ),
    }

    account_path = data_dir / f"{ticker.lower()}_rsi_mean_reversion_account_values.csv"
    metrics_path = metrics_dir / f"{ticker.lower()}_rsi_mean_reversion_metrics.csv"
    config_path = config_dir / f"{ticker.lower()}_rsi_mean_reversion_config.json"
    plot_path = plots_dir / f"{ticker.lower()}_rsi_mean_reversion_account_value.png"

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
