from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_COLUMNS = {"date", "tic", "close"}


def _validate_input_data(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _calculate_metrics(
    account_values: pd.DataFrame,
    strategy_name: str,
    source: str,
    initial_amount: float,
) -> pd.DataFrame:
    values = account_values["account_value"].astype(float)

    start_value = float(values.iloc[0])
    end_value = float(values.iloc[-1])
    profit_loss = end_value - initial_amount
    total_return_pct = (profit_loss / initial_amount) * 100.0

    running_max = values.cummax()
    drawdown = (values / running_max) - 1.0
    max_drawdown_pct = float(drawdown.min() * 100.0)

    daily_returns = values.pct_change().dropna()
    if daily_returns.std() == 0 or daily_returns.empty:
        annualized_sharpe = float("nan")
    else:
        annualized_sharpe = float(
            (daily_returns.mean() / daily_returns.std()) * (252**0.5)
        )

    metrics = pd.DataFrame(
        [
            {
                "strategy": strategy_name,
                "source": source,
                "start_value": start_value,
                "end_value": end_value,
                "profit_loss": profit_loss,
                "total_return_pct": total_return_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "annualized_sharpe": annualized_sharpe,
                "days": int(len(account_values)),
                "ended_above_initial": bool(end_value > initial_amount),
            }
        ]
    )

    return metrics


def run_sma_crossover(
    trade_data: str | Path,
    dataset_tag: str,
    ticker: str,
    run_name: str,
    files_dir: str | Path,
    plots_dir: str | Path,
    initial_amount: float = 1_000_000.0,
    fast_window: int = 50,
    slow_window: int = 200,
    transaction_cost_pct: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str | int | float]]:
    """
    Run a point-in-time single-ticker SMA crossover baseline.

    Signal logic:
    - Compute fast and slow moving averages from close prices.
    - If fast SMA > slow SMA, target position is 100% invested.
    - Otherwise, target position is 100% cash.
    - Position is shifted by 1 day to avoid same-day lookahead.

    Parameters
    ----------
    trade_data:
        PIT trade CSV.
    dataset_tag:
        Dataset identifier.
    ticker:
        Ticker to trade, e.g. AAPL.
    run_name:
        Human-readable run name.
    files_dir:
        Directory for CSV/JSON outputs.
    plots_dir:
        Directory for plot outputs.
    initial_amount:
        Starting capital.
    fast_window:
        Fast SMA window.
    slow_window:
        Slow SMA window.
    transaction_cost_pct:
        Cost applied when switching position.
        Example: 0.001 = 0.1% per full position change.
    """

    if fast_window <= 0:
        raise ValueError("fast_window must be positive.")
    if slow_window <= 0:
        raise ValueError("slow_window must be positive.")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window.")

    trade_data = Path(trade_data)
    files_dir = Path(files_dir)
    plots_dir = Path(plots_dir)

    files_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(trade_data)
    _validate_input_data(df)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    ticker_upper = ticker.upper()
    ticker_df = (
        df[df["tic"].astype(str).str.upper() == ticker_upper]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if ticker_df.empty:
        available = sorted(df["tic"].astype(str).str.upper().unique())[:20]
        raise ValueError(
            f"Ticker {ticker_upper} not found in trade data. "
            f"First available tickers: {available}"
        )

    ticker_df["close"] = ticker_df["close"].astype(float)

    ticker_df["fast_sma"] = (
        ticker_df["close"].rolling(window=fast_window, min_periods=fast_window).mean()
    )
    ticker_df["slow_sma"] = (
        ticker_df["close"].rolling(window=slow_window, min_periods=slow_window).mean()
    )

    ticker_df["raw_signal"] = (ticker_df["fast_sma"] > ticker_df["slow_sma"]).astype(
        int
    )

    # Shift by one day to avoid lookahead bias.
    # We only trade based on information known after the previous close.
    ticker_df["position"] = ticker_df["raw_signal"].shift(1).fillna(0).astype(float)

    ticker_df["asset_return"] = ticker_df["close"].pct_change().fillna(0.0)

    ticker_df["position_change"] = (
        ticker_df["position"].diff().abs().fillna(ticker_df["position"].abs())
    )
    ticker_df["transaction_cost"] = ticker_df["position_change"] * float(
        transaction_cost_pct
    )

    ticker_df["strategy_return"] = (
        ticker_df["position"] * ticker_df["asset_return"]
    ) - ticker_df["transaction_cost"]
    ticker_df["account_value"] = (
        initial_amount * (1.0 + ticker_df["strategy_return"]).cumprod()
    )

    # Force exact first value for clean comparison.
    ticker_df.loc[ticker_df.index[0], "account_value"] = initial_amount

    strategy_name = f"{ticker_upper}_sma_{fast_window}_{slow_window}"
    source = "Algorithmic Trading / non-RL"

    account_values = ticker_df[
        [
            "date",
            "tic",
            "close",
            "fast_sma",
            "slow_sma",
            "raw_signal",
            "position",
            "asset_return",
            "strategy_return",
            "account_value",
        ]
    ].copy()

    metrics = _calculate_metrics(
        account_values=account_values,
        strategy_name=strategy_name,
        source=source,
        initial_amount=initial_amount,
    )

    config = {
        "strategy": strategy_name,
        "source": source,
        "dataset_tag": dataset_tag,
        "ticker": ticker_upper,
        "run_name": run_name,
        "trade_data": str(trade_data),
        "initial_amount": float(initial_amount),
        "fast_window": int(fast_window),
        "slow_window": int(slow_window),
        "transaction_cost_pct": float(transaction_cost_pct),
        "lookahead_protection": "position shifted by one day",
        "files_dir": str(files_dir),
        "plots_dir": str(plots_dir),
    }

    prefix = f"{ticker_upper.lower()}_sma_{fast_window}_{slow_window}"

    account_values_path = files_dir / f"{prefix}_account_values.csv"
    metrics_path = files_dir / f"{prefix}_metrics.csv"
    config_path = files_dir / f"{prefix}_config.json"
    plot_path = plots_dir / f"{prefix}_account_value.png"

    account_values.to_csv(account_values_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    pd.Series(config).to_json(config_path, indent=2)

    plt.figure(figsize=(12, 6))
    plt.plot(
        account_values["date"], account_values["account_value"], label=strategy_name
    )
    plt.title(f"{strategy_name} account value")
    plt.xlabel("Date")
    plt.ylabel("Account value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    config["account_values_path"] = str(account_values_path)
    config["metrics_path"] = str(metrics_path)
    config["config_path"] = str(config_path)
    config["plot_path"] = str(plot_path)

    return account_values, metrics, config
