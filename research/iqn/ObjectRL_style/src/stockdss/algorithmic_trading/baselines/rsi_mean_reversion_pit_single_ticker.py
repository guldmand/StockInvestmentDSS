from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from stockdss.algorithmic_trading.metrics.trading_metrics import (
    calculate_account_metrics,
    load_trade_data_single_ticker,
    make_single_ticker_output_dirs,
    save_account_value_plot,
)


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
    run_name: str,
    run_root: str | Path | None = None,
    initial_amount: float = 1_000_000.0,
    rsi_window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> Dict[str, Path]:
    if rsi_window <= 0:
        raise ValueError("rsi_window must be positive.")
    if oversold >= overbought:
        raise ValueError("oversold must be smaller than overbought.")

    ticker = ticker.upper()
    df = load_trade_data_single_ticker(trade_data, ticker)
    files_dir, plots_dir = make_single_ticker_output_dirs(
        dataset_tag=dataset_tag,
        run_name=run_name,
        run_root=run_root,
        strategy_folder="rsi_mean_reversion",
    )

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
    account_value = float(initial_amount) * (1.0 + df["position"] * returns).cumprod()

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
        "run_name": run_name,
        "run_root": str(run_root) if run_root else None,
        "initial_amount": float(initial_amount),
        "rsi_window": int(rsi_window),
        "oversold": float(oversold),
        "overbought": float(overbought),
        "signal_rule": "buy/invested if RSI < oversold; sell/cash if RSI > overbought; otherwise hold previous position; signal shifted by 1 day",
    }

    account_path = files_dir / f"{ticker.lower()}_rsi_mean_reversion_account_values.csv"
    metrics_path = files_dir / f"{ticker.lower()}_rsi_mean_reversion_metrics.csv"
    config_path = files_dir / f"{ticker.lower()}_rsi_mean_reversion_config.json"
    plot_path = plots_dir / f"{ticker.lower()}_rsi_mean_reversion_account_value.png"

    # Create output folders before writing files.
    files_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    account.to_csv(account_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    pd.Series(config).to_json(config_path, indent=2)
    save_account_value_plot(account, output_path=plot_path, title=f"{strategy} account value")

    return {"account": account_path, "metrics": metrics_path, "config": config_path, "plot": plot_path}
