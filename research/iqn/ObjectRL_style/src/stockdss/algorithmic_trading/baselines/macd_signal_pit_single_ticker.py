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


def run_macd_signal(
    *,
    trade_data: str | Path,
    dataset_tag: str,
    ticker: str,
    run_name: str,
    run_root: str | Path | None = None,
    initial_amount: float = 1_000_000.0,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> Dict[str, Path]:
    if min(fast_window, slow_window, signal_window) <= 0:
        raise ValueError("fast_window, slow_window and signal_window must be positive.")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window.")

    ticker = ticker.upper()
    df = load_trade_data_single_ticker(trade_data, ticker)
    files_dir, plots_dir = make_single_ticker_output_dirs(
        dataset_tag=dataset_tag,
        run_name=run_name,
        run_root=run_root,
        strategy_folder="macd_signal",
    )

    fast_ema = df["close"].ewm(span=fast_window, adjust=False, min_periods=fast_window).mean()
    slow_ema = df["close"].ewm(span=slow_window, adjust=False, min_periods=slow_window).mean()
    df["macd_line"] = fast_ema - slow_ema
    df["macd_signal_line"] = df["macd_line"].ewm(
        span=signal_window,
        adjust=False,
        min_periods=signal_window,
    ).mean()
    df["macd_histogram"] = df["macd_line"] - df["macd_signal_line"]

    df["signal"] = (df["macd_line"] > df["macd_signal_line"]).astype(int)
    df["position"] = df["signal"].shift(1).fillna(0.0)

    returns = df["close"].pct_change().fillna(0.0)
    account_value = float(initial_amount) * (1.0 + df["position"] * returns).cumprod()

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
        "run_name": run_name,
        "run_root": str(run_root) if run_root else None,
        "initial_amount": float(initial_amount),
        "fast_window": int(fast_window),
        "slow_window": int(slow_window),
        "signal_window": int(signal_window),
        "signal_rule": "invested if MACD line > MACD signal line; otherwise cash; signal shifted by 1 day",
    }

    account_path = files_dir / f"{ticker.lower()}_macd_signal_account_values.csv"
    metrics_path = files_dir / f"{ticker.lower()}_macd_signal_metrics.csv"
    config_path = files_dir / f"{ticker.lower()}_macd_signal_config.json"
    plot_path = plots_dir / f"{ticker.lower()}_macd_signal_account_value.png"

    # Create output folders before writing files.
    files_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    account.to_csv(account_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    pd.Series(config).to_json(config_path, indent=2)
    save_account_value_plot(account, output_path=plot_path, title=f"{strategy} account value")

    return {"account": account_path, "metrics": metrics_path, "config": config_path, "plot": plot_path}
