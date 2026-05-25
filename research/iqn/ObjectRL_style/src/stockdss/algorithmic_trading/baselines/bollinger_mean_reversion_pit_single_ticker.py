from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from stockdss.algorithmic_trading.metrics.trading_metrics import (
    calculate_account_metrics,
    load_trade_data_single_ticker,
    make_single_ticker_output_dirs,
    save_account_value_plot,
)


def run_bollinger_mean_reversion(
    *,
    trade_data: str | Path,
    dataset_tag: str,
    ticker: str,
    run_name: str,
    run_root: str | Path | None = None,
    initial_amount: float = 1_000_000.0,
    window: int = 20,
    num_std: float = 2.0,
) -> Dict[str, Path]:
    if window <= 0:
        raise ValueError("window must be positive.")
    if num_std <= 0:
        raise ValueError("num_std must be positive.")

    ticker = ticker.upper()
    df = load_trade_data_single_ticker(trade_data, ticker)
    results_dir, plots_dir = make_single_ticker_output_dirs(
        dataset_tag=dataset_tag,
        run_name=run_name,
        run_root=run_root,
        strategy_folder="bollinger_mr",
    )

    if {"boll_ub", "boll_lb"}.issubset(df.columns):
        df["boll_upper"] = df["boll_ub"].astype(float)
        df["boll_lower"] = df["boll_lb"].astype(float)
        df["boll_middle"] = df["close"].rolling(window, min_periods=window).mean()
    else:
        df["boll_middle"] = df["close"].rolling(window, min_periods=window).mean()
        rolling_std = df["close"].rolling(window, min_periods=window).std(ddof=0)
        df["boll_upper"] = df["boll_middle"] + num_std * rolling_std
        df["boll_lower"] = df["boll_middle"] - num_std * rolling_std

    signal = []
    current_position = 0
    for close, lower, upper in zip(df["close"], df["boll_lower"], df["boll_upper"]):
        if pd.notna(lower) and close < lower:
            current_position = 1
        elif pd.notna(upper) and close > upper:
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
            "boll_lower": df["boll_lower"],
            "boll_middle": df["boll_middle"],
            "boll_upper": df["boll_upper"],
            "signal": df["signal"],
            "position": df["position"],
            "account_value": account_value,
        }
    )

    strategy = f"{ticker}_bollinger_mr_{window}_{num_std:g}"
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
        "window": int(window),
        "num_std": float(num_std),
        "signal_rule": "buy/invested if close < lower band; sell/cash if close > upper band; otherwise hold previous position; signal shifted by 1 day",
    }

    account_path = results_dir / f"{ticker.lower()}_bollinger_mr_account_values.csv"
    metrics_path = results_dir / f"{ticker.lower()}_bollinger_mr_metrics.csv"
    config_path = results_dir / f"{ticker.lower()}_bollinger_mr_config.json"
    plot_path = plots_dir / f"{ticker.lower()}_bollinger_mr_account_value.png"

    # Create output folders before writing files.
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    account.to_csv(account_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    pd.Series(config).to_json(config_path, indent=2)
    save_account_value_plot(
        account, output_path=plot_path, title=f"{strategy} account value"
    )

    return {
        "account": account_path,
        "metrics": metrics_path,
        "config": config_path,
        "plot": plot_path,
    }
