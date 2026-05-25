# src/stock_investment_dss/visualization/finrl_multi_baseline_plot.py
"""Aggregated portfolio value plot for FinRL baseline agents.

Produces a single PNG showing mean portfolio value over time for each FinRL
agent (a2c, ppo, mvo), with ±1 standard deviation bands computed across
multiple seed runs. The figure uses the PIT trade window to match the
evaluation period used throughout the thesis.

Aggregation:
    For each agent, one ``<agent>_asset_memory.csv`` is loaded per seed run.
    All seeds are concatenated and grouped by date. Mean and std are computed
    per date. When only a single seed is available, the std column contains NaN
    and the shaded band is omitted. When std is uniformly zero (deterministic
    agents such as MVO), the band collapses to zero width and is also omitted.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

_INITIAL_AMOUNT = 1_000_000
_AGENTS: dict[str, str] = {
    "a2c": "tab:blue",
    "ppo": "tab:green",
    "mvo": "tab:brown",
}


def _load_asset_memory(seed_root: Path, agent: str) -> pd.DataFrame | None:
    """Load a single seed's asset_memory CSV for the given agent.

    Returns a DataFrame with columns [date, account_value], or None if the
    file does not exist.
    """
    csv_path = (
        seed_root
        / "data"
        / "finrl_baseline_suite"
        / agent
        / f"{agent}_asset_memory.csv"
    )
    if not csv_path.exists():
        logger.warning("Asset memory not found: %s", csv_path)
        return None
    df = pd.read_csv(csv_path, usecols=["date", "account_value"], parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _aggregate_agent(
    seed_roots: list[Path],
    agent: str,
    pit_start: str,
    pit_end: str,
) -> pd.DataFrame | None:
    """Load, concatenate, and aggregate account_value across seeds for one agent.

    Returns a DataFrame with index=date and columns [mean, std], clipped to
    the PIT window, or None if no data could be loaded.
    """
    frames: list[pd.DataFrame] = []
    for seed_root in seed_roots:
        df = _load_asset_memory(seed_root, agent)
        if df is not None:
            frames.append(df)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[(combined["date"] >= pit_start) & (combined["date"] <= pit_end)]
    agg = combined.groupby("date")["account_value"].agg(["mean", "std"])
    agg["std"] = agg["std"].fillna(0.0)
    return agg.sort_index()


def _apply_date_formatting(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def generate_finrl_multi_baseline_plot(
    finrl_seed_run_roots: list[Path],
    output_dir: Path,
    pit_start_date: str = "2024-01-02",
    pit_end_date: str = "2026-05-22",
) -> Path:
    """Generate a single FinRL multi-baseline portfolio value plot.

    Aggregates account_value time series across all provided seed run roots,
    computing mean and ±1 std per date per agent. Output is a single PNG with
    one line per agent and a shaded band representing the std spread.

    Args:
        finrl_seed_run_roots: List of paths to FinRL seed run directories.
            Each must contain ``data/finrl_baseline_suite/<agent>/<agent>_asset_memory.csv``
            for at least one of {a2c, ppo, mvo}.
        output_dir: Destination directory. Will be created if absent.
        pit_start_date: ISO date string. Time series clipped to >= this date.
        pit_end_date: ISO date string. Time series clipped to <= this date.

    Returns:
        Path to the generated PNG file.

    Raises:
        ValueError: if ``finrl_seed_run_roots`` is empty.
        ValueError: if no agent CSVs can be loaded from any provided seed root.

    Notes:
        Scales correctly to any seed count. MVO's deterministic nature (std=0
        across seeds) is handled naturally: the fill_between band collapses to
        zero width and is omitted. When seed_count is 1, all stds are NaN and
        bands are omitted.
    """
    if not finrl_seed_run_roots:
        raise ValueError("finrl_seed_run_roots must not be empty.")

    seed_count = len(finrl_seed_run_roots)
    single_seed = seed_count == 1

    agent_data: dict[str, pd.DataFrame] = {}
    for agent in _AGENTS:
        agg = _aggregate_agent(
            finrl_seed_run_roots, agent, pit_start_date, pit_end_date
        )
        if agg is not None and not agg.empty:
            agent_data[agent] = agg

    if not agent_data:
        raise ValueError(
            "No agent asset_memory CSVs could be loaded from the provided seed roots."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pit_backtest_portfolio_value_mean_std.png"

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title(
        "PIT Backtest — FinRL Baselines Portfolio Value Over Time",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    subtitle = f"Mean ± 1 standard deviation across {seed_count} seed" + (
        "s" if seed_count != 1 else ""
    )
    ax.text(
        0.5,
        1.0,
        subtitle,
        fontsize=9,
        color="gray",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
    )

    for agent, color in _AGENTS.items():
        if agent not in agent_data:
            continue
        agg = agent_data[agent]
        dates = agg.index
        mean_vals = agg["mean"]
        std_vals = agg["std"]

        final_mean = mean_vals.iloc[-1]
        final_std = std_vals.iloc[-1]
        ret_pct = (final_mean / _INITIAL_AMOUNT - 1.0) * 100.0

        has_band = (not single_seed) and (std_vals.abs().max() > 0)

        if has_band:
            label = (
                f"{agent.upper()}: \\${final_mean:,.0f} ({ret_pct:+.1f}%)"
                f" ± \\${final_std:,.0f}"
            )
        else:
            label = f"{agent.upper()}: \\${final_mean:,.0f} ({ret_pct:+.1f}%)"

        ax.plot(dates, mean_vals, linewidth=1.5, color=color, label=label)

        if has_band:
            ax.fill_between(
                dates,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color=color,
                alpha=0.20,
            )

    ax.axhline(
        _INITIAL_AMOUNT,
        linestyle="--",
        color="gray",
        linewidth=1,
        alpha=0.6,
        label="Initial amount",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value (\\$)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"\\${x:,.0f}"))
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.30)
    _apply_date_formatting(ax)
    fig.autofmt_xdate()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("FinRL multi-baseline plot → %s", output_path)
    return output_path
