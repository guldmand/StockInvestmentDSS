from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from stockdss.algorithmic_trading.metrics.trading_metrics import (
    calculate_account_metrics,
    load_trade_data_single_ticker,
    make_single_ticker_output_dirs,
    save_account_value_plot,
)


def run_sma_crossover(
    *,
    trade_data: str | Path,
    dataset_tag: str,
    ticker: str,
    run_name: str,
    run_root: str | Path | None = None,
    initial_amount: float = 1_000_000.0,
    fast_window: int = 50,
    slow_window: int = 200,
    transaction_cost_pct: float = 0.0,
) -> Dict[str, Path]:
    """
    Run a point-in-time single-ticker SMA crossover baseline.

    Signal logic:
    - Compute fast and slow moving averages from close prices.
    - If fast SMA > slow SMA, target position is 100% invested.
    - Otherwise, target position is 100% cash.
    - Position is shifted by 1 day to avoid same-day lookahead.
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
    files_dir, plots_dir = make_single_ticker_output_dirs(
        dataset_tag=dataset_tag,
        run_name=run_name,
        run_root=run_root,
        strategy_folder="sma_crossover",
    )

    df["fast_sma"] = df["close"].rolling(
        window=fast_window, min_periods=fast_window
    ).mean()
    df["slow_sma"] = df["close"].rolling(
        window=slow_window, min_periods=slow_window
    ).mean()
    df["signal"] = (df["fast_sma"] > df["slow_sma"]).astype(int)

    # Shift by one day to avoid lookahead bias.
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
        "run_name": run_name,
        "run_root": str(run_root) if run_root else None,
        "initial_amount": float(initial_amount),
        "fast_window": int(fast_window),
        "slow_window": int(slow_window),
        "transaction_cost_pct": float(transaction_cost_pct),
        "signal_rule": "invested if fast SMA > slow SMA; otherwise cash; signal shifted by 1 day",
    }

    account_path = files_dir / f"{ticker.lower()}_sma_crossover_account_values.csv"
    metrics_path = files_dir / f"{ticker.lower()}_sma_crossover_metrics.csv"
    config_path = files_dir / f"{ticker.lower()}_sma_crossover_config.json"
    plot_path = plots_dir / f"{ticker.lower()}_sma_crossover_account_value.png"

    # Create output folders before writing files.
    files_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    account.to_csv(account_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    pd.Series(config).to_json(config_path, indent=2)
    save_account_value_plot(account, output_path=plot_path, title=f"{strategy} account value")

    return {
        "account": account_path,
        "metrics": metrics_path,
        "config": config_path,
        "plot": plot_path,
    }
