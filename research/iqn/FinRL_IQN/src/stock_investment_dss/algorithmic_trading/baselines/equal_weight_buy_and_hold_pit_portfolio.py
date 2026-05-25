"""
Equal-weight buy-and-hold point-in-time baseline for a portfolio.

Strategy:
    - Allocates equal capital to every ticker present in the trade data.
    - Buys on the first date and holds until the final date.
    - Does not rebalance; weights drift with prices.

Point-in-time safety:
    Trivially PIT-safe: all allocation decisions use only the first row price
    per ticker, and no future data is accessed.

Output contract:
    Writes five files to:
        outputs/run_registry/algorithmic_baselines/<dataset_tag>/
            equal_weight_buy_and_hold/portfolio/

    - equal_weight_buy_and_hold_account_values.csv  — portfolio value time series
    - equal_weight_buy_and_hold_metrics.csv         — performance metrics (Etape 2 canonical)
    - equal_weight_buy_and_hold_weights.csv         — initial allocation per ticker
    - equal_weight_buy_and_hold_config.json         — run parameters
    - equal_weight_buy_and_hold_account_value.png   — portfolio value plot

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
    save_account_value_plot,
)
from stock_investment_dss.utilities.paths import create_run_paths, RunPaths


def run_equal_weight_buy_and_hold_portfolio(
    *,
    trade_data: str | Path,
    dataset_tag: str,
    initial_amount: float = 1_000_000.0,
    transaction_cost_pct: float = 0.001,
    pit_start_date: str | None = None,
    pit_end_date: str | None = None,
    strategy_folder: str | None = None,
    run_paths: Optional[RunPaths] = None,
    output_subpath: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Run equal-weight buy-and-hold for all tickers in *trade_data* and write outputs.

    Parameters
    ----------
    trade_data:
        Path to a CSV file with columns ``date``, ``tic``, ``close``.
    dataset_tag:
        Short identifier for the dataset, used in the output directory path.
        Example: ``"demo_10_new"``.
    initial_amount:
        Initial portfolio capital in portfolio-value units.

    Returns
    -------
    dict[str, Path]
        Keys: ``account``, ``metrics``, ``weights``, ``config``, ``plot``.
        Values: absolute paths to the written output files.
    """
    path = Path(trade_data)
    if not path.exists():
        raise FileNotFoundError(f"Trade data not found: {path}")

    df = pd.read_csv(path)
    required = {"date", "tic", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trade data is missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df["tic"] = df["tic"].astype(str).str.upper()
    df["close"] = df["close"].astype(float)
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    price_wide = df.pivot_table(
        index="date", columns="tic", values="close", aggfunc="last"
    ).sort_index()
    price_wide = price_wide.dropna(axis=1, how="any")

    if pit_start_date is not None:
        price_wide = price_wide[price_wide.index >= pd.Timestamp(pit_start_date)]
    if pit_end_date is not None:
        price_wide = price_wide[price_wide.index < pd.Timestamp(pit_end_date)]

    if price_wide.empty:
        raise ValueError(
            "No complete ticker price matrix could be built for"
            " equal-weight buy-and-hold."
        )

    tickers = price_wide.columns.tolist()
    first_prices = price_wide.iloc[0]
    if transaction_cost_pct < 0:
        raise ValueError("transaction_cost_pct must be non-negative.")
    # Apply transaction cost at initial allocation (one buy per ticker).
    capital_after_cost = float(initial_amount) * (1.0 - float(transaction_cost_pct))
    capital_per_asset = capital_after_cost / len(tickers)
    shares = capital_per_asset / first_prices

    portfolio_values = price_wide.mul(shares, axis=1).sum(axis=1)

    account = pd.DataFrame(
        {
            "date": portfolio_values.index.strftime("%Y-%m-%d"),
            "account_value": portfolio_values.values,
        }
    )

    weights = pd.DataFrame(
        {
            "ticker": tickers,
            "initial_price": first_prices.values,
            "initial_capital": capital_per_asset,
            "shares": shares.values,
            "initial_weight": 1.0 / len(tickers),
        }
    )

    if run_paths is None:
        _folder = (
            strategy_folder
            if (strategy_folder and strategy_folder.strip())
            else "equal_weight_buy_and_hold"
        )
        run_name = f"d_iqn_dss_algorithmic_baseline_{_folder}_{dataset_tag}"
        run_paths = create_run_paths(run_name)
    _sub = Path(output_subpath) if output_subpath else Path("")
    data_dir = run_paths.data_directory / _sub
    metrics_dir = run_paths.metrics_directory / _sub
    config_dir = run_paths.config_directory / _sub
    plots_dir = run_paths.plots_directory / _sub
    for d in [data_dir, metrics_dir, config_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    strategy = f"equal_weight_buy_and_hold_{len(tickers)}"
    metrics = calculate_account_metrics(
        account,
        strategy=strategy,
        source="Algorithmic Trading / non-RL",
        initial_amount=initial_amount,
    )

    config = {
        "strategy": strategy,
        "dataset_tag": dataset_tag,
        "trade_data": str(trade_data),
        "initial_amount": float(initial_amount),
        "stock_dimension": int(len(tickers)),
        "transaction_cost_pct": float(transaction_cost_pct),
        "pit_start_date": pit_start_date,
        "pit_end_date": pit_end_date,
        "tickers": tickers,
        "signal_rule": (
            "buy equal-dollar allocation on first trade date"
            " and hold until final trade date"
        ),
    }

    account_path = data_dir / "equal_weight_buy_and_hold_account_values.csv"
    metrics_path = metrics_dir / "equal_weight_buy_and_hold_metrics.csv"
    weights_path = data_dir / "equal_weight_buy_and_hold_weights.csv"
    config_path = config_dir / "equal_weight_buy_and_hold_config.json"
    plot_path = plots_dir / "equal_weight_buy_and_hold_account_value.png"

    account.to_csv(account_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    weights.to_csv(weights_path, index=False)
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
        "weights": weights_path,
        "config": config_path,
        "plot": plot_path,
    }
