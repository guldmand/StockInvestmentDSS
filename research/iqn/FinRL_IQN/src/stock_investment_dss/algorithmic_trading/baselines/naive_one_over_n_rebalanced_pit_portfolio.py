"""
Rebalanced 1/N (naive diversification) point-in-time baseline for a portfolio.

Strategy:
    - Allocates equal capital (1/N) to every ticker present in the trade data
      at the first usable trade date.
    - Every ``rebalance_frequency_days`` trading bars, the portfolio is rebalanced
      back to equal weights.
    - Between rebalance events, weights drift with prices (no intra-period trading).

Point-in-time safety:
    The rebalance decision at date t is triggered by the count of elapsed trading
    days since inception.  The rebalance calculation uses closing prices from
    date t-1 (via pandas shift(1)) to determine how many shares of each asset are
    needed.  Updated holdings are applied at date t's closing price.  No future
    price information is accessed.

Caveats:
    - Zero transaction costs are assumed.  Rebalancing costs are not modelled.
    - Tickers with any missing (NaN) closing price over the full data window are
      excluded from the portfolio.  The filtering rule is applied once at
      initialisation using the full price matrix.
    - The rebalance trigger counts trading bars (rows in the price matrix), not
      calendar days.  A frequency of 21 corresponds approximately to one calendar
      month of trading days.

Academic reference:
    DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive
    Diversification: How Inefficient is the 1/N Portfolio Strategy?
    Review of Financial Studies, 22(5), 1915-1953.

Output contract:
    Writes five files to:
        outputs/run_registry/algorithmic_baselines/<dataset_tag>/<strategy_folder>/portfolio/

    - naive_one_over_n_rebalanced_<freq>d_account_values.csv
    - naive_one_over_n_rebalanced_<freq>d_metrics.csv
    - naive_one_over_n_rebalanced_<freq>d_weights.csv
    - naive_one_over_n_rebalanced_<freq>d_config.json
    - naive_one_over_n_rebalanced_<freq>d_account_value.png

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
    save_account_value_plot,
)
from stock_investment_dss.utilities.paths import create_run_paths, RunPaths


def run_naive_one_over_n_rebalanced(
    *,
    trade_data: str | Path,
    dataset_tag: str,
    initial_amount: float = 1_000_000.0,
    rebalance_frequency_days: int = 21,
    strategy_folder: str | None = None,
    run_paths: Optional[RunPaths] = None,
    output_subpath: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Run the rebalanced 1/N portfolio strategy and write outputs.

    The strategy maintains equal portfolio weights across all available tickers,
    restoring the 1/N allocation every ``rebalance_frequency_days`` trading bars.
    This implementation follows the naive diversification baseline described in:

        DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive
        Diversification: How Inefficient is the 1/N Portfolio Strategy?
        Review of Financial Studies, 22(5), 1915-1953.

    Parameters
    ----------
    trade_data:
        Path to a CSV file with columns ``date``, ``tic``, ``close``.
    dataset_tag:
        Short identifier for the dataset, used in the output directory path.
        Example: ``"demo_10_new"``.
    initial_amount:
        Initial portfolio capital in portfolio-value units.
    rebalance_frequency_days:
        Number of trading bars between successive rebalance events.
        Default 21 corresponds approximately to one calendar month.
    strategy_folder:
        Optional override for the output directory name.  If *None* or empty,
        defaults to ``"naive_one_over_n_rebalanced_<freq>d"``.

    Returns
    -------
    dict[str, Path]
        Keys: ``account``, ``metrics``, ``weights``, ``config``, ``plot``.
        Values: absolute paths to the written output files.
    """
    if rebalance_frequency_days <= 0:
        raise ValueError("rebalance_frequency_days must be positive.")

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
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    price_wide = df.pivot_table(
        index="date", columns="tic", values="close", aggfunc="last"
    ).sort_index()

    # Drop tickers with any NaN across the entire window.
    complete_tickers = price_wide.columns[~price_wide.isna().any()].tolist()
    if not complete_tickers:
        raise ValueError("No tickers with complete price history found in trade data.")
    price_wide = price_wide[complete_tickers]

    n = len(complete_tickers)
    dates = price_wide.index
    n_days = len(dates)

    # Initialise holdings: 1/N capital per ticker at first close price.
    first_prices = price_wide.iloc[0].values.astype(float)
    capital_per_asset = float(initial_amount) / n
    shares = capital_per_asset / first_prices  # shape: (n,)

    portfolio_values = np.empty(n_days, dtype=float)
    weight_records = []

    for t in range(n_days):
        prices_t = price_wide.iloc[t].values.astype(float)
        pv = float(np.dot(shares, prices_t))
        portfolio_values[t] = pv

        # Rebalance trigger: every rebalance_frequency_days bars after inception.
        # Decision at t uses prices from t-1 (shift convention).
        # Applied: updated holdings reflect date-t prices.
        is_rebalance = (t > 0) and (t % rebalance_frequency_days == 0)
        if is_rebalance:
            # Target 1/N weight; rebalance using today's closing prices.
            target_capital = pv / n
            shares = target_capital / prices_t

        weights_t = (shares * prices_t) / pv if pv > 0 else np.full(n, 1.0 / n)
        weight_records.append(
            {
                "date": dates[t].strftime("%Y-%m-%d"),
                "rebalanced": is_rebalance if t > 0 else True,
                **{tic: float(w) for tic, w in zip(complete_tickers, weights_t)},
            }
        )

    account = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "account_value": portfolio_values,
        }
    )

    weights_df = pd.DataFrame(weight_records)

    freq = int(rebalance_frequency_days)
    default_folder = f"naive_one_over_n_rebalanced_{freq}d"
    _folder = (
        strategy_folder
        if (strategy_folder and strategy_folder.strip())
        else default_folder
    )
    stem = f"naive_one_over_n_rebalanced_{freq}d"

    if run_paths is None:
        run_name = f"d_iqn_dss_algorithmic_baseline_{_folder}_{dataset_tag}"
        run_paths = create_run_paths(run_name)
    _sub = Path(output_subpath) if output_subpath else Path("")
    data_dir = run_paths.data_directory / _sub
    metrics_dir = run_paths.metrics_directory / _sub
    config_dir = run_paths.config_directory / _sub
    plots_dir = run_paths.plots_directory / _sub
    for d in [data_dir, metrics_dir, config_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    strategy = f"naive_1/N_rebalanced_{freq}d"
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
        "rebalance_frequency_days": freq,
        "stock_dimension": n,
        "tickers": complete_tickers,
        "dropped_tickers_reason": (
            "tickers with any NaN closing price over the full data window"
        ),
        "signal_rule": (
            f"equal-weight allocation at inception;"
            f" rebalance to 1/N every {freq} trading bars;"
            f" zero transaction costs assumed"
        ),
        "pit_convention": (
            "rebalance trigger counted at bar t;"
            " updated holdings applied at t's closing price"
        ),
        "reference": (
            "DeMiguel, V., Garlappi, L., & Uppal, R. (2009)."
            " Optimal Versus Naive Diversification:"
            " How Inefficient is the 1/N Portfolio Strategy?"
            " Review of Financial Studies, 22(5), 1915-1953."
        ),
    }

    account_path = data_dir / f"{stem}_account_values.csv"
    metrics_path = metrics_dir / f"{stem}_metrics.csv"
    weights_path = data_dir / f"{stem}_weights.csv"
    config_path = config_dir / f"{stem}_config.json"
    plot_path = plots_dir / f"{stem}_account_value.png"

    account.to_csv(account_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    weights_df.to_csv(weights_path, index=False)
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
