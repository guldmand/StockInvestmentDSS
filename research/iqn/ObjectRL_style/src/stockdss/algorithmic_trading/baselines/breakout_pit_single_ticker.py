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


def run_breakout(
    *,
    trade_data: str | Path,
    dataset_tag: str,
    ticker: str,
    run_name: str,
    run_root: str | Path | None = None,
    initial_amount: float = 1_000_000.0,
    lookback_window: int = 20,
) -> Dict[str, Path]:
    if lookback_window <= 0:
        raise ValueError("lookback_window must be positive.")

    ticker = ticker.upper()
    df = load_trade_data_single_ticker(trade_data, ticker)
    files_dir, plots_dir = make_single_ticker_output_dirs(
        dataset_tag=dataset_tag,
        run_name=run_name,
        run_root=run_root,
        strategy_folder="breakout",
    )

    df["rolling_high"] = df["close"].rolling(lookback_window, min_periods=lookback_window).max().shift(1)
    df["rolling_low"] = df["close"].rolling(lookback_window, min_periods=lookback_window).min().shift(1)

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
        "run_name": run_name,
        "run_root": str(run_root) if run_root else None,
        "initial_amount": float(initial_amount),
        "lookback_window": int(lookback_window),
        "signal_rule": "buy/invested if close breaks above prior rolling high; sell/cash if close breaks below prior rolling low; signal shifted by 1 day",
    }

    account_path = files_dir / f"{ticker.lower()}_breakout_account_values.csv"
    metrics_path = files_dir / f"{ticker.lower()}_breakout_metrics.csv"
    config_path = files_dir / f"{ticker.lower()}_breakout_config.json"
    plot_path = plots_dir / f"{ticker.lower()}_breakout_account_value.png"

    # Create output folders before writing files.
    files_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    account.to_csv(account_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    pd.Series(config).to_json(config_path, indent=2)
    save_account_value_plot(account, output_path=plot_path, title=f"{strategy} account value")

    return {"account": account_path, "metrics": metrics_path, "config": config_path, "plot": plot_path}
