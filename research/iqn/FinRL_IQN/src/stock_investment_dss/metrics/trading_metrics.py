# src/stock_investment_dss/metrics/trading_metrics.py
"""Canonical trading metrics module for the D-IQN-DSS thesis pipeline.

This module provides utilities needed by Etape 3 (algorithmic trading baselines) and
Etape 4 (FinRL baselines):

    load_trade_data_single_ticker   — loads and validates single-ticker PIT trade data
    make_single_ticker_output_dirs  — creates stable output directories for single-ticker runs
    make_portfolio_output_dirs      — creates stable output directories for portfolio-level runs
    save_account_value_plot         — saves an account/portfolio value curve as a PNG file
    calculate_account_metrics       — thin wrapper around V2's compute_portfolio_metrics that
                                      returns a 1-row DataFrame for cross-strategy comparison

Methodology decisions (documented in outputs/run_registry/etape_2_metrics_decision.md):

1. Return denominator
   total_return_pct uses the first row of the account_values series as denominator
   (V2 convention: evaluation.portfolio_metrics.compute_portfolio_metrics). The
   initial_amount parameter is accepted for API compatibility and validated against
   the first row; a UserWarning is emitted if they diverge by more than 0.1%.
   This convention is canonical for all Etape 3 and Etape 4 strategy comparisons.

2. Drawdown unit
   max_drawdown_pct is always in percent, range [-100, 0].
   Example: -15.4 represents a 15.4% peak-to-trough drawdown.
   This matches V2's _compute_max_drawdown_pct (returns drawdown.min() * 100.0).

3. Sharpe undefined value
   annualized_sharpe is float("nan") when the return series has fewer than 2 valid
   observations or exhibits zero volatility. This is defined behaviour, not a data
   error. V2's compute_portfolio_metrics returns None in these cases; this wrapper
   normalizes None to float("nan") so callers can use pd.isna() consistently.

4. Annualization
   Sharpe and volatility use 252 trading days per year (default for all functions).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Must precede pyplot import; required for headless execution.
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from stock_investment_dss.evaluation.portfolio_metrics import compute_portfolio_metrics

_REQUIRED_TRADE_COLUMNS: frozenset[str] = frozenset({"date", "tic", "close"})


def _normalise_run_root(run_root: str | Path | None) -> Path | None:
    if run_root is None:
        return None
    p = Path(run_root)
    if str(p).strip() == "":
        return None
    return p


def make_single_ticker_output_dirs(
    *,
    dataset_tag: str,
    run_name: str,
    run_root: str | Path | None,
    strategy_folder: str,
) -> tuple[Path, Path]:
    """Create and return output directories for a single-ticker algorithmic trading run.

    With a central run root, outputs are written to::

        <run_root>/algorithmic_trading/results/<strategy_folder>/<run_name>/
        <run_root>/algorithmic_trading/plots/<strategy_folder>/<run_name>/

    Without a run root, outputs are written to::

        outputs/algorithmic_trading/<dataset_tag>/<run_name>/results/<strategy_folder>/
        outputs/algorithmic_trading/<dataset_tag>/<run_name>/plots/<strategy_folder>/

    Both directories are created with ``parents=True, exist_ok=True``.

    Returns:
        (results_dir, plots_dir) as Path objects.
    """
    root = _normalise_run_root(run_root)
    if root is not None:
        base = root / "algorithmic_trading"
        results_dir = base / "results" / strategy_folder / run_name
        plots_dir = base / "plots" / strategy_folder / run_name
    else:
        base = Path("outputs") / "algorithmic_trading" / dataset_tag / run_name
        results_dir = base / "results" / strategy_folder
        plots_dir = base / "plots" / strategy_folder

    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, plots_dir


def make_portfolio_output_dirs(
    *,
    dataset_tag: str,
    run_name: str,
    run_root: str | Path | None,
    strategy_folder: str,
) -> tuple[Path, Path]:
    """Create and return output directories for a portfolio-level algorithmic trading run.

    Delegates to :func:`make_single_ticker_output_dirs`; see that function for the
    directory path conventions.

    Returns:
        (results_dir, plots_dir) as Path objects. Both directories are created.
    """
    return make_single_ticker_output_dirs(
        dataset_tag=dataset_tag,
        run_name=run_name,
        run_root=run_root,
        strategy_folder=strategy_folder,
    )


def load_trade_data_single_ticker(trade_data: str | Path, ticker: str) -> pd.DataFrame:
    """Load and validate point-in-time trade data for a single ticker.

    Args:
        trade_data: Path to a CSV file containing at least ``{date, tic, close}`` columns
                    and any additional market-data columns. Rows for other tickers are
                    discarded.
        ticker:     Ticker symbol to select (case-insensitive).

    Returns:
        DataFrame with all original columns, filtered to ``ticker``, sorted by ``date``
        ascending, index reset.  The ``date`` column is ``datetime64``, ``tic`` is
        upper-case ``str``, ``close`` is ``float64``.

    Raises:
        FileNotFoundError: if the CSV path does not exist.
        ValueError: if required columns are missing, no rows match the ticker, or any
                    close price is non-positive.
    """
    path = Path(trade_data)
    if not path.exists():
        raise FileNotFoundError(f"Trade data file not found: {path}")

    df = pd.read_csv(path)
    missing = _REQUIRED_TRADE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Trade data is missing required columns: {sorted(missing)}")

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["tic"] = result["tic"].astype(str).str.upper()
    result["close"] = pd.to_numeric(result["close"], errors="coerce")

    ticker_upper = ticker.upper()
    result = result[result["tic"] == ticker_upper].copy()
    result = result.dropna(subset=["date", "close"])
    result = result.sort_values("date").reset_index(drop=True)

    if result.empty:
        raise ValueError(f"No rows found for ticker '{ticker_upper}' in {path.name}")

    if (result["close"] <= 0).any():
        bad_dates = result.loc[result["close"] <= 0, "date"].head(5).tolist()
        raise ValueError(
            f"Ticker '{ticker_upper}' has non-positive close prices. "
            f"First affected dates: {bad_dates}"
        )

    return result


def calculate_account_metrics(
    account_values: pd.DataFrame,
    *,
    strategy: str,
    source: str,
    initial_amount: float,
) -> pd.DataFrame:
    """Compute scalar performance metrics for an account/portfolio value time series.

    Wraps V2's canonical :func:`~stock_investment_dss.evaluation.portfolio_metrics.compute_portfolio_metrics`.
    This wrapper exists for API compatibility with Etape 3 and Etape 4 baseline runners
    that expect the V1-style interface (single 1-row DataFrame return value).

    Args:
        account_values:  DataFrame containing an ``account_value`` column (or any column
                         recognised by the auto-detection in ``compute_portfolio_metrics``:
                         account_value, portfolio_value, total_asset, asset_value, value).
        strategy:        Strategy label for the output row, e.g. ``'buy_and_hold'``.
        source:          Data-source label for the output row, e.g. ``'demo_10_new'``.
        initial_amount:  Configured starting capital.  Used to:

                         - validate alignment with the first row of ``account_values``
                           (a :class:`UserWarning` is emitted if they diverge by > 0.1%);
                         - compute ``ended_above_initial``.

                         Note: ``total_return_pct`` uses the first-row denominator
                         (V2 canonical convention), not ``initial_amount``.

    Returns:
        1-row :class:`pandas.DataFrame` with columns:

            strategy, source, start_value, end_value, profit_loss, total_return_pct,
            max_drawdown_pct, annualized_sharpe, days, ended_above_initial

        Units:

            - ``total_return_pct``: percent (e.g., 15.4 means +15.4 %).
            - ``max_drawdown_pct``: percent in range [-100, 0].
            - ``annualized_sharpe``: float, or ``float('nan')`` when the return series has
              fewer than 2 valid observations or zero volatility.  NaN is defined
              behaviour, not a data error.

    Implementation note:
        ``annualized_sharpe`` uses 252 trading days per year.  Drawdown is the minimum
        of ``(value / running_max) - 1``, multiplied by 100.
    """
    result = compute_portfolio_metrics(account_values)
    s = result.summary

    initial_value = s.get("initial_value")
    first_row_value = (
        float(initial_value) if initial_value is not None else float("nan")
    )

    if not np.isnan(first_row_value):
        divergence = abs(initial_amount - first_row_value) / max(
            abs(initial_amount), 1e-10
        )
        if divergence > 0.001:
            warnings.warn(
                f"initial_amount ({initial_amount:.4f}) differs from the first account "
                f"value ({first_row_value:.4f}) by {divergence * 100:.2f}%. "
                "total_return_pct uses the first-row denominator (V2 canonical). "
                "Ensure initial_amount equals the configured starting capital.",
                UserWarning,
                stacklevel=2,
            )

    sharpe = s.get("annualized_sharpe")
    sharpe = float("nan") if sharpe is None else float(sharpe)

    end_value = s.get("final_value")
    end_value = float("nan") if end_value is None else float(end_value)

    return pd.DataFrame(
        [
            {
                "strategy": strategy,
                "source": source,
                "start_value": first_row_value,
                "end_value": end_value,
                "profit_loss": s.get("profit_loss", float("nan")),
                "total_return_pct": s.get("total_return_pct", float("nan")),
                "max_drawdown_pct": s.get("max_drawdown_pct", float("nan")),
                "annualized_sharpe": sharpe,
                "days": s.get("row_count", 0),
                "ended_above_initial": (
                    bool(end_value > initial_amount)
                    if not np.isnan(end_value)
                    else False
                ),
            }
        ]
    )


def save_account_value_plot(
    account_values: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> None:
    """Save a portfolio/account value curve as a PNG file.

    Args:
        account_values:  DataFrame with ``date`` and ``account_value`` columns.
        output_path:     Destination PNG path. Parent directories are created if absent.
        title:           Plot title string.

    Raises:
        ValueError: if ``date`` or ``account_value`` columns are absent.

    Notes:
        - Uses the Agg backend (set at module level); safe for headless and server execution.
        - Does not call ``plt.show()``; suitable for automated thesis pipeline runs.
        - The figure is closed immediately after saving to release memory.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        "date" not in account_values.columns
        or "account_value" not in account_values.columns
    ):
        raise ValueError(
            "account_values must contain 'date' and 'account_value' columns."
        )

    plot_df = account_values.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plot_df["date"], plot_df["account_value"], linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Account value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
