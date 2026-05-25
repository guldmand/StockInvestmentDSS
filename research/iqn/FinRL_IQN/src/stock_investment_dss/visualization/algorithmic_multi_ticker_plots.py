# src/stock_investment_dss/visualization/algorithmic_multi_ticker_plots.py
"""Multi-ticker account value plots for algorithmic trading baselines.

Generates one PNG per strategy variant by combining all individual ticker
time series onto a single figure. The resulting plots are suitable for
thesis presentation and cross-ticker comparison within a single strategy.

Supported layout conventions (auto-detected from directory structure):

    Single-ticker strategies
        data/algorithmic_baselines/<strategy>/<ticker>/*account_value*.csv
        One line per ticker. Output → single_ticker_strategies/<strategy>.png

    Portfolio strategies
        data/algorithmic_baselines/<strategy>/*account_value*.csv
        Single line (portfolio already aggregates across tickers).
        Output → portfolio_strategies/<strategy>.png

All figures are restricted to the PIT trade window to prevent pre-2024 data
from appearing in thesis materials.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

logger = logging.getLogger(__name__)

_INITIAL_AMOUNT = 1_000_000


def _title_from_name(strategy_name: str) -> str:
    """Convert snake_case strategy name to title-case display string."""
    return " ".join(w.capitalize() for w in strategy_name.split("_"))


def _load_account_value(csv_path: Path, pit_start: str, pit_end: str) -> pd.Series:
    """Load account_value column from a CSV, clipped to [pit_start, pit_end]."""
    df = pd.read_csv(csv_path, usecols=["date", "account_value"], parse_dates=["date"])
    df = df.sort_values("date")
    mask = (df["date"] >= pit_start) & (df["date"] <= pit_end)
    df = df.loc[mask].reset_index(drop=True)
    return df.set_index("date")["account_value"]


def _make_legend_label(ticker: str, series: pd.Series) -> str:
    """Return '<TICKER>: $<final:,.0f> (+<ret:.1f>%)' legend entry."""
    final = series.iloc[-1]
    ret_pct = (final / _INITIAL_AMOUNT - 1.0) * 100.0
    return f"{ticker.upper()}: \\${final:,.0f} ({ret_pct:+.1f}%)"


def _apply_date_formatting(ax: plt.Axes) -> None:
    """Apply AutoDateLocator + ConciseDateFormatter to x-axis."""
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _plot_single_ticker_strategy(
    strategy_dir: Path,
    output_path: Path,
    strategy_name: str,
    pit_start: str,
    pit_end: str,
) -> int:
    """Plot all ticker series for a single-ticker strategy.  Returns ticker count."""
    ticker_dirs = sorted([d for d in strategy_dir.iterdir() if d.is_dir()])
    if not ticker_dirs:
        logger.warning("No ticker subdirectories found in %s — skipping.", strategy_dir)
        return 0

    series_list: list[tuple[str, pd.Series]] = []
    for ticker_dir in ticker_dirs:
        csv_candidates = sorted(ticker_dir.glob("*account_value*.csv"))
        if not csv_candidates:
            logger.warning("No account_value CSV in %s — skipping ticker.", ticker_dir)
            continue
        series = _load_account_value(csv_candidates[0], pit_start, pit_end)
        if series.empty:
            logger.warning(
                "Empty series after PIT clip for %s — skipping.", ticker_dir.name
            )
            continue
        pit_start_value = series.iloc[0]
        if pit_start_value <= 0:
            logger.warning("Degenerate start value for %s — skipping.", ticker_dir.name)
            continue
        series = series / pit_start_value * _INITIAL_AMOUNT
        series_list.append((ticker_dir.name, series))

    if not series_list:
        logger.warning(
            "No valid series for strategy %s — skipping plot.", strategy_name
        )
        return 0

    display_title = _title_from_name(strategy_name)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title(
        f"{display_title} — PIT Backtest Account Value",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    ax.text(
        0.5,
        1.0,
        f"PIT trade window: {pit_start} → {pit_end}  •  Re-baselined to \\$1,000,000 at PIT start",
        fontsize=9,
        color="gray",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
    )

    for ticker, series in series_list:
        label = _make_legend_label(ticker, series)
        ax.plot(series.index, series.values, linewidth=1.5, label=label)

    ax.axhline(
        _INITIAL_AMOUNT,
        linestyle="--",
        color="gray",
        linewidth=1,
        alpha=0.6,
        label="Initial amount",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Account value ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)
    _apply_date_formatting(ax)
    fig.autofmt_xdate()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return len(series_list)


def _plot_portfolio_strategy(
    strategy_dir: Path,
    output_path: Path,
    strategy_name: str,
    pit_start: str,
    pit_end: str,
) -> int:
    """Plot a portfolio-level strategy (single CSV, single line).  Returns 1."""
    csv_candidates = sorted(strategy_dir.glob("*account_value*.csv"))
    if not csv_candidates:
        logger.warning(
            "No account_value CSV in portfolio strategy dir %s — skipping.",
            strategy_dir,
        )
        return 0

    series = _load_account_value(csv_candidates[0], pit_start, pit_end)
    if series.empty:
        logger.warning("Empty series after PIT clip for %s — skipping.", strategy_name)
        return 0

    pit_start_value = series.iloc[0]
    if pit_start_value <= 0:
        logger.warning("Degenerate start value for %s — skipping.", strategy_name)
        return 0
    series = series / pit_start_value * _INITIAL_AMOUNT

    display_title = _title_from_name(strategy_name)
    label = _make_legend_label(display_title, series)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title(
        f"{display_title} — PIT Backtest Account Value",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    ax.text(
        0.5,
        1.0,
        f"PIT trade window: {pit_start} → {pit_end}  •  Re-baselined to \\$1,000,000 at PIT start",
        fontsize=9,
        color="gray",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
    )

    ax.plot(series.index, series.values, linewidth=1.5, label=label)
    ax.axhline(
        _INITIAL_AMOUNT,
        linestyle="--",
        color="gray",
        linewidth=1,
        alpha=0.6,
        label="Initial amount",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(True, alpha=0.25)
    _apply_date_formatting(ax)
    fig.autofmt_xdate()

    existing_handles, existing_labels = ax.get_legend_handles_labels()
    composition_handles: list[Line2D] = []
    weights_csv = strategy_dir / f"{strategy_name}_weights.csv"
    if weights_csv.exists():
        weights_df = pd.read_csv(weights_csv)
        if weights_df.columns[0] == "ticker":
            for _, row in weights_df.iterrows():
                lbl = f"  {row['ticker']}: {int(round(row['shares'])):,} shares  ({row['initial_weight'] * 100:.1f}%)"
                composition_handles.append(Line2D([0], [0], linestyle="none", marker="", label=lbl))
        elif weights_df.columns[0] == "date":
            last_row = weights_df.iloc[-1]
            ticker_cols = [c for c in weights_df.columns if c not in ("date", "rebalanced")]
            for ticker in ticker_cols:
                lbl = f"  {ticker}: {float(last_row[ticker]) * 100:.1f}%"
                composition_handles.append(Line2D([0], [0], linestyle="none", marker="", label=lbl))
    else:
        logger.warning("Weights CSV not found for %s — composition entries omitted.", strategy_name)

    all_handles = existing_handles + composition_handles
    all_labels = existing_labels + [h.get_label() for h in composition_handles]
    ax.legend(all_handles, all_labels, loc="upper left", frameon=True, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return 1


def generate_algorithmic_multi_ticker_plots(
    algorithmic_run_root: Path,
    output_dir: Path,
    pit_start_date: str = "2024-01-01",
    pit_end_date: str = "2026-05-22",
) -> dict[str, int]:
    """Generate one multi-ticker plot per algorithmic strategy variant.

    Args:
        algorithmic_run_root: Path to the algorithmic baseline grid run directory
            (e.g., outputs/runs/<timestamp>_d_iqn_dss_algorithmic_baseline_grid_<id>/).
            Must contain ``data/algorithmic_baselines/`` subdirectory.
        output_dir: Destination directory for the generated PNGs. Subdirectories
            ``single_ticker_strategies/`` and ``portfolio_strategies/`` will be created
            as needed.
        pit_start_date: ISO date string. Time series are clipped to >= this date.
        pit_end_date: ISO date string. Time series are clipped to <= this date.

    Returns:
        Dict mapping strategy_name -> number of ticker lines plotted.
        For portfolio strategies the count is 1.

    Raises:
        ValueError: if ``algorithmic_run_root/data/algorithmic_baselines/`` does not exist.

    Notes:
        All matplotlib figures are explicitly closed after savefig() to prevent
        memory accumulation when generating 26 plots in sequence. Strategy directories
        with zero discoverable account_value CSVs are skipped with a warning log line.
        PIT clipping uses inclusive bounds on both ends.
    """
    baselines_dir = algorithmic_run_root / "data" / "algorithmic_baselines"
    if not baselines_dir.exists():
        raise ValueError(f"algorithmic_baselines directory not found: {baselines_dir}")

    single_out = output_dir / "single_ticker_strategies"
    portfolio_out = output_dir / "portfolio_strategies"

    results: dict[str, int] = {}

    for strategy_dir in sorted(baselines_dir.iterdir()):
        if not strategy_dir.is_dir():
            continue

        strategy_name = strategy_dir.name
        subdirs = [d for d in strategy_dir.iterdir() if d.is_dir()]

        if subdirs:
            output_path = single_out / f"{strategy_name}.png"
            count = _plot_single_ticker_strategy(
                strategy_dir, output_path, strategy_name, pit_start_date, pit_end_date
            )
        else:
            output_path = portfolio_out / f"{strategy_name}.png"
            count = _plot_portfolio_strategy(
                strategy_dir, output_path, strategy_name, pit_start_date, pit_end_date
            )

        if count > 0:
            results[strategy_name] = count
            logger.info("  %s → %d line(s)", strategy_name, count)

    return results
