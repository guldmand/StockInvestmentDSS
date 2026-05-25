"""Volatility-filtered momentum baseline for one ticker on PIT trade data.

Rule:
- Invest when recent momentum is positive AND realised volatility is below threshold.
- Otherwise stay in cash.
- Signal is shifted one day to avoid same-day look-ahead.

This file is intentionally self-contained and does not import project-local
baseline helper modules. That keeps it robust against helper-file naming
mismatches while still using the shared trading_metrics implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockdss.algorithmic_trading.metrics.trading_metrics import (
    calculate_account_metrics,
)

SOURCE = "Algorithmic Trading / non-RL"
STRATEGY_FOLDER = "vol_filter"


def _find_ticker_column(df: pd.DataFrame) -> str:
    for candidate in ("tic", "ticker", "symbol"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Trade data must contain one ticker column named one of: tic, ticker, symbol."
    )


def _load_trade_data_single_ticker(trade_data: str | Path, ticker: str) -> pd.DataFrame:
    path = Path(trade_data)
    if not path.exists():
        raise FileNotFoundError(f"Trade data file not found: {path}")

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Trade data must contain a 'date' column.")
    if "close" not in df.columns:
        raise ValueError("Trade data must contain a 'close' column.")

    ticker_col = _find_ticker_column(df)
    ticker = ticker.upper()

    df[ticker_col] = df[ticker_col].astype(str).str.upper()
    df = df[df[ticker_col] == ticker].copy()
    if df.empty:
        raise ValueError(f"Ticker {ticker} not found in trade data: {path}")

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"Ticker {ticker} has no usable date/close rows in: {path}")

    return df


def _make_output_dirs(
    *,
    dataset_tag: str,
    run_name: str,
    run_root: str | Path | None,
    strategy_folder: str,
) -> Tuple[Path, Path]:
    if run_root is None:
        base = Path("outputs") / "runs" / str(dataset_tag)
    else:
        base = Path(run_root)

    results_dir = base / "algorithmic_trading" / "results" / strategy_folder / run_name
    plots_dir = base / "algorithmic_trading" / "plots" / strategy_folder / run_name

    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    return results_dir, plots_dir


def _save_account_value_plot(
    account: pd.DataFrame, *, output_path: str | Path, title: str
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = account.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])

    plt.figure(figsize=(10, 5))
    plt.plot(plot_df["date"], plot_df["account_value"])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Account value")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_volatility_filter(
    *,
    trade_data: str | Path,
    dataset_tag: str,
    ticker: str,
    run_name: str,
    run_root: str | Path | None = None,
    initial_amount: float = 1_000_000.0,
    momentum_window: int = 20,
    volatility_window: int = 20,
    max_annualized_volatility: float = 0.40,
) -> Dict[str, Path]:
    if momentum_window <= 0:
        raise ValueError("momentum_window must be positive.")
    if volatility_window <= 0:
        raise ValueError("volatility_window must be positive.")
    if max_annualized_volatility <= 0:
        raise ValueError("max_annualized_volatility must be positive.")

    ticker = ticker.upper()
    df = _load_trade_data_single_ticker(trade_data, ticker)

    results_dir, plots_dir = _make_output_dirs(
        dataset_tag=dataset_tag,
        run_name=run_name,
        run_root=run_root,
        strategy_folder=STRATEGY_FOLDER,
    )

    daily_return = df["close"].pct_change().fillna(0.0)
    momentum = df["close"].pct_change(periods=momentum_window)
    annualized_volatility = (
        daily_return.rolling(volatility_window, min_periods=volatility_window)
        .std(ddof=0)
        .mul(np.sqrt(252.0))
    )

    signal = (
        (momentum > 0.0) & (annualized_volatility <= float(max_annualized_volatility))
    ).astype(float)

    # Shift by one trading day to avoid look-ahead bias.
    position = signal.shift(1).fillna(0.0)
    strategy_return = position * daily_return
    account_value = float(initial_amount) * (1.0 + strategy_return).cumprod()

    account = pd.DataFrame(
        {
            "date": df["date"].dt.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "close": df["close"],
            "daily_return": daily_return,
            "momentum": momentum,
            "annualized_volatility": annualized_volatility,
            "signal": signal,
            "position": position,
            "strategy_return": strategy_return,
            "account_value": account_value,
        }
    )

    strategy = (
        f"{ticker}_vol_filter_"
        f"m{momentum_window}_v{volatility_window}_{max_annualized_volatility:g}"
    )
    metrics = calculate_account_metrics(
        account,
        strategy=strategy,
        source=SOURCE,
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
        "momentum_window": int(momentum_window),
        "volatility_window": int(volatility_window),
        "max_annualized_volatility": float(max_annualized_volatility),
        "signal_rule": (
            "invested if momentum > 0 and annualized volatility <= threshold; "
            "otherwise cash; signal shifted by 1 day"
        ),
    }

    account_path = results_dir / f"{ticker.lower()}_vol_filter_account_values.csv"
    metrics_path = results_dir / f"{ticker.lower()}_vol_filter_metrics.csv"
    config_path = results_dir / f"{ticker.lower()}_vol_filter_config.json"
    plot_path = plots_dir / f"{ticker.lower()}_vol_filter_account_value.png"

    # Hard guarantee: create folders immediately before writing.
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    account_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    if not account_path.parent.exists():
        raise RuntimeError(f"Output folder was not created: {account_path.parent}")

    account.to_csv(str(account_path), index=False)
    metrics.to_csv(str(metrics_path), index=False)
    pd.Series(config).to_json(str(config_path), indent=2)

    _save_account_value_plot(
        account,
        output_path=plot_path,
        title=f"{strategy} account value",
    )

    return {
        "account": account_path,
        "metrics": metrics_path,
        "config": config_path,
        "plot": plot_path,
    }
