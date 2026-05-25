"""
Buy-and-hold point-in-time baseline for a single ticker.

Strategy:
    - Buys all available capital at the first price in the trade window.
    - Holds until the final date in the window.
    - Does not rebalance.

Point-in-time safety:
    The strategy is trivially PIT-safe: no look-ahead information is used.
    The first trade price is determined by the first row of the supplied data
    period, and all subsequent prices are taken in chronological order.

Output contract:
    Writes four files to:
        outputs/run_registry/algorithmic_baselines/<dataset_tag>/buy_and_hold/<ticker>/

    - <ticker>_buy_and_hold_account_value.csv   — daily portfolio value time series
    - <ticker>_buy_and_hold_metrics.csv         — performance metrics (Etape 2 canonical)
    - <ticker>_buy_and_hold_config.json         — run parameters
    - <ticker>_buy_and_hold_account_value_plot.png

    Performance metrics are computed via
    stock_investment_dss.metrics.trading_metrics.calculate_account_metrics,
    which wraps the canonical V2 evaluation module.  No metric formulas are
    duplicated in this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from stock_investment_dss.metrics.trading_metrics import (
    calculate_account_metrics,
    save_account_value_plot,
)
from stock_investment_dss.utilities.paths import create_run_paths, RunPaths


def run_buy_and_hold(
    trade_data: Union[str, Path, pd.DataFrame],
    ticker: str,
    *,
    dataset_tag: str,
    initial_amount: float = 1_000_000.0,
    price_column: str | None = None,
    transaction_cost_pct: float = 0.001,
    pit_start_date: str | None = None,
    pit_end_date: str | None = None,
    strategy_folder: str | None = None,
    run_paths: Optional[RunPaths] = None,
    output_subpath: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Run buy-and-hold for one ticker and write outputs to the V2 canonical run directory.

    Parameters
    ----------
    trade_data:
        Path to a CSV file or a pre-loaded DataFrame containing at least date,
        ticker-identifier, and price columns.
    ticker:
        Ticker symbol to evaluate (e.g. ``"KO"``).  Case-insensitive.
    dataset_tag:
        Short identifier for the dataset, used in the output directory path.
        Example: ``"demo_10_new"``.
    initial_amount:
        Initial portfolio capital in portfolio-value units.
    price_column:
        Optional explicit price column name.  If *None*, the function prefers
        ``adj_close``, then ``close``.

    Returns
    -------
    dict[str, Path]
        Keys: ``account``, ``metrics``, ``config``, ``plot``.
        Values: absolute paths to the written output files.
    """
    if isinstance(trade_data, (str, Path)):
        path = Path(trade_data)
        if not path.exists():
            raise FileNotFoundError(f"Trade data not found: {path}")
        df_raw = pd.read_csv(path)
    else:
        df_raw = trade_data.copy()

    if df_raw.empty:
        raise ValueError("trade_data is empty.")

    df = df_raw.copy()

    ticker_col = _detect_ticker_column(df)
    date_col = _detect_date_column(df)
    price_col = price_column or _detect_price_column(df)

    df = df[df[ticker_col].astype(str).str.upper() == ticker.upper()].copy()

    if df.empty:
        raise ValueError(f"No rows found for ticker: {ticker!r}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])

    if df.empty:
        raise ValueError(f"No valid prices found for ticker: {ticker!r}")

    if pit_start_date is not None:
        df = df[df[date_col] >= pd.Timestamp(pit_start_date)].copy()
    if pit_end_date is not None:
        df = df[df[date_col] < pd.Timestamp(pit_end_date)].copy()
    if df.empty:
        raise ValueError(f"No rows in trade window for ticker: {ticker!r}")

    first_price = float(df[price_col].iloc[0])
    if first_price <= 0:
        raise ValueError(f"First price must be positive.  Got: {first_price}")
    if transaction_cost_pct < 0:
        raise ValueError("transaction_cost_pct must be non-negative.")

    # Apply transaction cost at entry: capital reduced by cost before share purchase.
    capital_after_cost = float(initial_amount) * (1.0 - float(transaction_cost_pct))
    shares = capital_after_cost / first_price
    account_values = shares * df[price_col].astype(float)

    account = pd.DataFrame(
        {
            "date": df[date_col].dt.strftime("%Y-%m-%d"),
            "ticker": ticker.upper(),
            "price": df[price_col].astype(float).values,
            "shares": shares,
            "cash": 0.0,
            "account_value": account_values.values,
            "strategy": f"{ticker.upper()}_buy_and_hold",
        }
    )

    if run_paths is None:
        _folder = (
            strategy_folder
            if (strategy_folder and strategy_folder.strip())
            else "buy_and_hold"
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

    strategy = f"{ticker.upper()}_buy_and_hold"
    metrics = calculate_account_metrics(
        account,
        strategy=strategy,
        source="Algorithmic Trading / non-RL",
        initial_amount=initial_amount,
    )

    config = {
        "strategy": strategy,
        "dataset_tag": dataset_tag,
        "ticker": ticker.upper(),
        "initial_amount": float(initial_amount),
        "price_column_used": price_col,
        "transaction_cost_pct": float(transaction_cost_pct),
        "pit_start_date": pit_start_date,
        "pit_end_date": pit_end_date,
        "signal_rule": ("buy at first available price and hold until final trade date"),
    }

    account_path = data_dir / f"{ticker.lower()}_buy_and_hold_account_value.csv"
    metrics_path = metrics_dir / f"{ticker.lower()}_buy_and_hold_metrics.csv"
    config_path = config_dir / f"{ticker.lower()}_buy_and_hold_config.json"
    plot_path = plots_dir / f"{ticker.lower()}_buy_and_hold_account_value_plot.png"

    account.to_csv(account_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    save_account_value_plot(
        account,
        output_path=plot_path,
        title=f"{strategy} — account value",
    )

    return {
        "account": account_path,
        "metrics": metrics_path,
        "config": config_path,
        "plot": plot_path,
    }


# ---------------------------------------------------------------------------
# Private column-detection helpers
# ---------------------------------------------------------------------------


def _detect_ticker_column(df: pd.DataFrame) -> str:
    for candidate in ("tic", "ticker", "symbol"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not detect ticker column.  Expected one of: tic, ticker, symbol."
    )


def _detect_date_column(df: pd.DataFrame) -> str:
    for candidate in ("date", "datetime", "timestamp"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not detect date column.  Expected one of: date, datetime, timestamp."
    )


def _detect_price_column(df: pd.DataFrame) -> str:
    for candidate in ("adj_close", "close", "Close", "Adj Close"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not detect price column.  "
        "Expected one of: adj_close, close, Close, Adj Close."
    )
