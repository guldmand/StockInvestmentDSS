# src/stock_investment_dss/visualization/transaction_logs.py
"""Transaction log generators for algorithmic, FinRL, and IQN strategy outputs.

Each public function reads existing manifest CSV / JSON files, aggregates the
per-decision data, and writes one human-readable markdown file per
strategy/ticker, agent/seed, or IQN seed.  Raw source files are never modified.

Supported output structures:

    Algorithmic (single-ticker strategies)
        transaction_logs/algorithmic/single_ticker/<strategy>/<ticker>.md

    Algorithmic (portfolio strategies)
        transaction_logs/algorithmic/portfolio/<strategy>.md

    FinRL baselines
        transaction_logs/finrl/<agent>_seed_<N>.md

    IQN agent
        transaction_logs/iqn/seed_<N>.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_PIT_START_DEFAULT = "2024-01-01"
_PIT_END_DEFAULT = "2026-05-22"
_INITIAL_AMOUNT = 1_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _title_case(name: str) -> str:
    """Convert snake_case strategy name to Title Case display string."""
    return name.replace("_", " ").title()


def _fmt_usd(value: float) -> str:
    return f"${value:,.0f}"


def _fmt_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def _pit_clip(df: pd.DataFrame, date_col: str, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(df[date_col])
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Algorithmic — single-ticker
# ---------------------------------------------------------------------------


def _write_single_ticker_log(
    csv_path: Path,
    output_path: Path,
    strategy_name: str,
    ticker: str,
    pit_start: str,
    pit_end: str,
) -> int:
    """Write one markdown transaction log for a single-ticker strategy/ticker pair.

    Returns the number of transaction rows written (0 if no transactions found).
    """
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except Exception as exc:
        logger.warning("Cannot read %s: %s — skipping.", csv_path, exc)
        return 0

    pit = _pit_clip(df, "date", pit_start, pit_end)
    if pit.empty:
        logger.warning("Empty PIT slice for %s / %s — skipping.", strategy_name, ticker)
        return 0

    if "position" not in pit.columns:
        # Alternative schema (e.g. buy_and_hold): uses 'shares' column instead.
        # Position is always 100% invested; treat as a single INITIAL_BUY.
        if "shares" not in pit.columns:
            logger.warning("No 'position' or 'shares' column in %s — skipping.", csv_path)
            return 0

        trading_days = len(pit)
        final_value = pit["account_value"].iloc[-1]
        final_return_pct = (final_value / _INITIAL_AMOUNT - 1) * 100
        price_col = "price" if "price" in pit.columns else ("close" if "close" in pit.columns else None)

        lines: list[str] = [
            f"# {_title_case(strategy_name)} — {ticker} — Transaction Log",
            "",
            f"**Strategy**: `{strategy_name}`",
            f"**Ticker**: `{ticker}`",
            f"**PIT trade window**: {pit_start} → {pit_end} (`{trading_days}` trading days)",
            f"**Initial capital**: {_fmt_usd(_INITIAL_AMOUNT)}",
            "",
            "## Summary",
            "",
            "Buy-and-hold strategy. All shares purchased at strategy inception; position held throughout without change.",
            "",
            f"- Days in market: {trading_days} (100%)",
            "- Total transactions: 1 (initial purchase)",
            f"- Final account value: {_fmt_usd(final_value)} ({_fmt_pct(final_return_pct)})",
            "",
            "## Transactions",
            "",
        ]

        first_row = pit.iloc[0]
        date_str = pd.Timestamp(first_row["date"]).strftime("%Y-%m-%d")
        shares = first_row["shares"]
        price_str = _fmt_usd(first_row[price_col]) if price_col else "—"
        lines += [
            "| # | Date | Action | Price | Shares | Account value |",
            "| :--- | :--- | :--- | ---: | ---: | ---: |",
            f"| 1 | {date_str} | INITIAL_BUY | {price_str} | {shares:,.0f} | {_fmt_usd(first_row['account_value'])} |",
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 1

    pit = pit.reset_index(drop=True)
    trading_days = len(pit)
    days_long = int((pit["position"] == 1.0).sum())
    days_cash = trading_days - days_long

    final_value = pit["account_value"].iloc[-1]
    final_return_pct = (final_value / _INITIAL_AMOUNT - 1) * 100

    # Detect transaction rows: position changed from previous row.
    pit["_pos_prev"] = pit["position"].shift(1)
    # First row is always included if position != 0 (initial entry on day 0).
    first_entry_mask = (pit.index == 0) & (pit["position"] != 0.0)
    change_mask = pit["position"] != pit["_pos_prev"]
    tx_mask = first_entry_mask | (change_mask & (pit.index > 0))
    transactions = pit.loc[tx_mask].copy()

    entry_count = int((transactions["position"] == 1.0).sum())
    exit_count = int((transactions["position"] == 0.0).sum())

    lines: list[str] = [
        f"# {_title_case(strategy_name)} — {ticker} — Transaction Log",
        "",
        f"**Strategy**: `{strategy_name}`",
        f"**Ticker**: `{ticker}`",
        f"**PIT trade window**: {pit_start} → {pit_end} (`{trading_days}` trading days)",
        f"**Initial capital**: {_fmt_usd(_INITIAL_AMOUNT)}",
        "",
        "## Summary",
        "",
        f"- Total entry events: {entry_count}",
        f"- Total exit events: {exit_count}",
        f"- Days in market (long): {days_long} ({days_long / trading_days * 100:.1f}%)",
        f"- Days in cash: {days_cash} ({days_cash / trading_days * 100:.1f}%)",
        f"- Final account value: {_fmt_usd(final_value)} ({_fmt_pct(final_return_pct)})",
        "",
        "## Transactions",
        "",
    ]

    has_signal = "signal" in pit.columns
    has_close = "close" in pit.columns

    if transactions.empty:
        lines.append("_No transactions detected in the PIT trade window._")
    else:
        header_cols = ["#", "Date", "Action"]
        if has_signal:
            header_cols.append("Signal")
        if has_close:
            header_cols.append("Price")
        header_cols += ["Position", "Account value"]

        sep_cols = [":---"] * len(header_cols)
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("| " + " | ".join(sep_cols) + " |")

        for i, (_, row) in enumerate(transactions.iterrows(), start=1):
            action = "ENTRY" if row["position"] == 1.0 else "EXIT"
            date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            cols = [str(i), date_str, action]
            if has_signal:
                sig = row["signal"]
                sig_str = (
                    f"{sig:.4f}"
                    if pd.notna(sig) and isinstance(sig, float)
                    else str(sig)
                )
                cols.append(f"`{sig_str}`")
            if has_close:
                cols.append(_fmt_usd(row["close"]))
            cols.append(f"{int(row['position'])}")
            cols.append(_fmt_usd(row["account_value"]))
            lines.append("| " + " | ".join(cols) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(transactions)


# ---------------------------------------------------------------------------
# Algorithmic — portfolio
# ---------------------------------------------------------------------------


def _write_portfolio_log_static(
    account_values_csv: Path,
    weights_csv: Path,
    output_path: Path,
    strategy_name: str,
    pit_start: str,
    pit_end: str,
) -> int:
    """Write log for a static equal-weight buy-and-hold portfolio."""
    try:
        av_df = pd.read_csv(account_values_csv, parse_dates=["date"])
        weights_df = pd.read_csv(weights_csv)
    except Exception as exc:
        logger.warning("Cannot read portfolio CSVs for %s: %s", strategy_name, exc)
        return 0

    pit = _pit_clip(av_df, "date", pit_start, pit_end)
    if pit.empty:
        logger.warning("Empty PIT slice for portfolio %s.", strategy_name)
        return 0

    trading_days = len(pit)
    # Rebase to $1M at PIT start
    start_val = pit["account_value"].iloc[0]
    final_rebased = pit["account_value"].iloc[-1] / start_val * _INITIAL_AMOUNT
    final_return_pct = (final_rebased / _INITIAL_AMOUNT - 1) * 100

    tickers = list(weights_df["ticker"])
    ticker_list = ", ".join(tickers)

    lines: list[str] = [
        f"# {_title_case(strategy_name)} — Transaction Log",
        "",
        f"**Strategy**: `{strategy_name}`",
        f"**Universe**: {ticker_list} ({len(tickers)} tickers, 10% each)",
        f"**PIT trade window**: {pit_start} → {pit_end}",
        f"**Initial capital**: {_fmt_usd(_INITIAL_AMOUNT)} ({_fmt_usd(_INITIAL_AMOUNT // len(tickers))} per ticker)",
        "",
        "## Summary",
        "",
        "Static allocation strategy. All purchases happen at strategy inception; no rebalancing occurs.",
        "",
        f"- Trading days in PIT window: {trading_days}",
        "- Trades during PIT window: 0 (positions established before PIT start)",
        f"- Final account value (rebased): {_fmt_usd(final_rebased)} ({_fmt_pct(final_return_pct)})",
        "",
        "## Initial portfolio (set at strategy inception)",
        "",
        "| Ticker | Shares | Initial price | Initial capital | Initial weight |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]

    for _, row in weights_df.iterrows():
        ticker = row["ticker"]
        shares = int(round(row["shares"]))
        price = row["initial_price"]
        capital = row["initial_capital"]
        weight_pct = row["initial_weight"] * 100
        lines.append(
            f"| {ticker} | {shares:,} | {_fmt_usd(price)} | {_fmt_usd(capital)} | {weight_pct:.1f}% |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(weights_df)


def _write_portfolio_log_dynamic(
    account_values_csv: Path,
    weights_csv: Path,
    output_path: Path,
    strategy_name: str,
    pit_start: str,
    pit_end: str,
) -> int:
    """Write log for a dynamic rebalancing portfolio (naive 1/N)."""
    try:
        av_df = pd.read_csv(account_values_csv, parse_dates=["date"])
        weights_df = pd.read_csv(weights_csv, parse_dates=["date"])
    except Exception as exc:
        logger.warning("Cannot read portfolio CSVs for %s: %s", strategy_name, exc)
        return 0

    pit_av = _pit_clip(av_df, "date", pit_start, pit_end)
    pit_w = _pit_clip(weights_df, "date", pit_start, pit_end)
    if pit_av.empty:
        logger.warning("Empty PIT slice for portfolio %s.", strategy_name)
        return 0

    trading_days = len(pit_av)
    start_val = pit_av["account_value"].iloc[0]
    final_rebased = pit_av["account_value"].iloc[-1] / start_val * _INITIAL_AMOUNT
    final_return_pct = (final_rebased / _INITIAL_AMOUNT - 1) * 100

    ticker_cols = [c for c in weights_df.columns if c not in ("date", "rebalanced")]
    rebalance_rows = (
        pit_w[pit_w["rebalanced"] == True] if not pit_w.empty else pit_w
    )  # noqa: E712
    rebalance_count = len(rebalance_rows)

    lines: list[str] = [
        f"# {_title_case(strategy_name)} — Transaction Log",
        "",
        f"**Strategy**: `{strategy_name}`",
        f"**Universe**: {', '.join(ticker_cols)} ({len(ticker_cols)} tickers)",
        f"**Rebalancing**: every 21 trading days, back to equal weight",
        f"**PIT trade window**: {pit_start} → {pit_end}",
        "",
        "## Summary",
        "",
        f"- Trading days in PIT window: {trading_days}",
        f"- Rebalance events during PIT window: {rebalance_count}",
        f"- Final account value (rebased): {_fmt_usd(final_rebased)} ({_fmt_pct(final_return_pct)})",
        "",
        "## Rebalance events",
        "",
    ]

    if rebalance_rows.empty:
        lines.append("_No rebalance events in the PIT trade window._")
    else:
        header = ["#", "Date"] + ticker_cols
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join([":---"] + ["---:"] * (len(header) - 1)) + " |")
        for i, (_, row) in enumerate(rebalance_rows.iterrows(), start=1):
            date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            weight_strs = [f"{float(row[t]) * 100:.1f}%" for t in ticker_cols]
            lines.append("| " + " | ".join([str(i), date_str] + weight_strs) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rebalance_count


# ---------------------------------------------------------------------------
# Public: algorithmic
# ---------------------------------------------------------------------------


def generate_algorithmic_transaction_logs(
    algorithmic_run_root: Path,
    output_dir: Path,
    pit_start_date: str = _PIT_START_DEFAULT,
    pit_end_date: str = _PIT_END_DEFAULT,
) -> dict[str, int]:
    """Generate one markdown transaction log per algorithmic strategy / ticker pair.

    Single-ticker strategies produce one file per ticker under
    ``output_dir/single_ticker/<strategy>/``.  Portfolio strategies produce one
    file under ``output_dir/portfolio/``.

    Returns a dict mapping output file stem to the number of transaction rows
    written (0 for files with no transactions).
    """
    baselines_dir = algorithmic_run_root / "data" / "algorithmic_baselines"
    if not baselines_dir.exists():
        logger.warning("Algorithmic baselines dir not found: %s", baselines_dir)
        return {}

    results: dict[str, int] = {}

    for strategy_dir in sorted(baselines_dir.iterdir()):
        if not strategy_dir.is_dir():
            continue
        strategy_name = strategy_dir.name
        subdirs = [d for d in strategy_dir.iterdir() if d.is_dir()]

        if subdirs:
            # Single-ticker strategy: subdirs are ticker directories.
            for ticker_dir in sorted(subdirs):
                ticker = ticker_dir.name
                csv_candidates = sorted(ticker_dir.glob("*.csv"))
                if not csv_candidates:
                    logger.warning("No CSV in %s — skipping.", ticker_dir)
                    continue
                out_path = (
                    output_dir
                    / "single_ticker"
                    / strategy_name
                    / f"{ticker.lower()}.md"
                )
                n = _write_single_ticker_log(
                    csv_candidates[0],
                    out_path,
                    strategy_name,
                    ticker,
                    pit_start_date,
                    pit_end_date,
                )
                results[f"{strategy_name}/{ticker}"] = n
        else:
            # Portfolio strategy: CSV + weights CSV at strategy_dir level.
            av_candidates = sorted(strategy_dir.glob("*account_value*.csv"))
            weights_candidates = sorted(strategy_dir.glob("*weights*.csv"))
            if not av_candidates:
                logger.warning("No account_value CSV in %s — skipping.", strategy_dir)
                continue
            if not weights_candidates:
                logger.warning(
                    "No weights CSV in %s — static table unavailable.", strategy_dir
                )

            out_path = output_dir / "portfolio" / f"{strategy_name}.md"
            if weights_candidates:
                weights_df_peek = pd.read_csv(weights_candidates[0], nrows=1)
                if weights_df_peek.columns[0] == "ticker":
                    n = _write_portfolio_log_static(
                        av_candidates[0],
                        weights_candidates[0],
                        out_path,
                        strategy_name,
                        pit_start_date,
                        pit_end_date,
                    )
                elif weights_df_peek.columns[0] == "date":
                    n = _write_portfolio_log_dynamic(
                        av_candidates[0],
                        weights_candidates[0],
                        out_path,
                        strategy_name,
                        pit_start_date,
                        pit_end_date,
                    )
                else:
                    logger.warning(
                        "Unknown weights CSV schema for %s — skipping.", strategy_name
                    )
                    n = 0
            else:
                n = 0
            results[strategy_name] = n

    logger.info(
        "Algorithmic transaction logs: %d files written to %s", len(results), output_dir
    )
    return results


# ---------------------------------------------------------------------------
# FinRL
# ---------------------------------------------------------------------------


def _write_finrl_log(
    seed_run_root: Path,
    agent: str,
    seed_index: int,
    output_path: Path,
    pit_start: str,
    pit_end: str,
) -> int:
    """Write one markdown transaction log for a FinRL agent/seed pair.

    Returns the number of non-zero action rows written (0 for MVO / no actions).
    """
    agent_dir = seed_run_root / "data" / "finrl_baseline_suite" / agent
    if not agent_dir.exists():
        logger.warning("FinRL agent dir not found: %s", agent_dir)
        return 0

    action_csv = agent_dir / f"{agent}_action_memory.csv"
    asset_csv = agent_dir / f"{agent}_asset_memory.csv"
    weights_csv = agent_dir / f"{agent}_weights.csv"

    has_actions = action_csv.exists()
    is_mvo = not has_actions and weights_csv.exists()

    # Load asset memory for account values.
    try:
        asset_df = pd.read_csv(asset_csv, parse_dates=["date"])
        pit_asset = _pit_clip(asset_df, "date", pit_start, pit_end)
    except Exception as exc:
        logger.warning(
            "Cannot read asset memory for %s seed %d: %s", agent, seed_index, exc
        )
        pit_asset = pd.DataFrame()

    final_value = (
        pit_asset["account_value"].iloc[-1] if not pit_asset.empty else float("nan")
    )
    final_return_pct = (
        (final_value / _INITIAL_AMOUNT - 1) * 100 if not pit_asset.empty else 0.0
    )
    trading_days = len(pit_asset)

    agent_display = agent.upper()

    lines: list[str] = [
        f"# {agent_display} Agent — Seed {seed_index} — Transaction Log",
        "",
        f"**Agent**: {agent_display}",
        f"**Seed**: {seed_index}",
        f"**PIT trade window**: {pit_start} → {pit_end} (`{trading_days}` trading days)",
        f"**Initial capital**: {_fmt_usd(_INITIAL_AMOUNT)}",
        "",
        "## Summary",
        "",
    ]

    if is_mvo:
        # MVO: static optimization, no daily actions.
        try:
            w_df = pd.read_csv(weights_csv)
        except Exception as exc:
            logger.warning("Cannot read MVO weights for seed %d: %s", seed_index, exc)
            w_df = pd.DataFrame()

        lines += [
            "MVO is a static mean-variance optimization strategy. Portfolio weights are set",
            "once at the start of the PIT window and held without rebalancing.",
            "",
            f"- Trading days: {trading_days}",
            "- Daily trades: 0 (static allocation)",
            f"- Final account value: {_fmt_usd(final_value)} ({_fmt_pct(final_return_pct)})",
            "",
            "## Portfolio allocation",
            "",
        ]
        if not w_df.empty:
            lines += [
                "| Ticker | Weight | Shares |",
                "| :--- | ---: | ---: |",
            ]
            for _, row in w_df.iterrows():
                lines.append(
                    f"| {row['ticker']} | {float(row['weight']) * 100:.1f}% | {float(row['shares']):,.1f} |"
                )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0

    # A2C / PPO: load action memory.
    try:
        action_df = pd.read_csv(action_csv, parse_dates=["date"])
    except Exception as exc:
        logger.warning(
            "Cannot read action memory for %s seed %d: %s", agent, seed_index, exc
        )
        return 0

    pit_actions = _pit_clip(action_df, "date", pit_start, pit_end)
    ticker_cols = [c for c in action_df.columns if c != "date"]

    # Non-zero action days.
    nonzero_mask = pit_actions[ticker_cols].abs().sum(axis=1) > 0
    active_days = pit_actions.loc[nonzero_mask]

    # Cumulative holdings (running sum of all action_memory rows).
    cum_holdings = (
        pit_actions[ticker_cols].cumsum().iloc[-1] if not pit_actions.empty else {}
    )

    most_traded_ticker = (
        pit_actions[ticker_cols].abs().sum().idxmax() if not pit_actions.empty else "—"
    )
    total_volume = int(pit_actions[ticker_cols].abs().sum().sum())

    lines += [
        f"- Days with trades: {len(active_days)} / {trading_days}",
        f"- Total trade volume (sum of absolute share counts): {total_volume:,}",
        f"- Most-traded ticker: `{most_traded_ticker}`",
        f"- Final account value: {_fmt_usd(final_value)} ({_fmt_pct(final_return_pct)})",
        "",
        "## Final estimated holdings",
        "",
        "| Ticker | Shares held (cumulative) |",
        "| :--- | ---: |",
    ]
    for t in ticker_cols:
        shares = cum_holdings[t] if isinstance(cum_holdings, pd.Series) else 0
        lines.append(f"| {t} | {int(shares):,} |")

    lines += [
        "",
        "## Transactions (non-zero action days only)",
        "",
    ]

    if active_days.empty:
        lines.append("_No non-zero action days in the PIT trade window._")
    else:
        delta_headers = [f"{t} Δ" for t in ticker_cols]
        if not pit_asset.empty:
            # Merge account value onto action rows.
            merged = active_days.merge(
                pit_asset[["date", "account_value"]], on="date", how="left"
            )
        else:
            merged = active_days.copy()
            merged["account_value"] = float("nan")

        header = ["Date"] + delta_headers + ["Account value"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append(
            "| " + " | ".join([":---"] + ["---:"] * (len(delta_headers) + 1)) + " |"
        )

        for _, row in merged.iterrows():
            date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            deltas = []
            for t in ticker_cols:
                v = int(row[t])
                deltas.append(f"{v:+,}" if v != 0 else "0")
            av = row.get("account_value", float("nan"))
            av_str = _fmt_usd(av) if pd.notna(av) else "—"
            lines.append("| " + " | ".join([date_str] + deltas + [av_str]) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(active_days)


def generate_finrl_transaction_logs(
    finrl_seed_run_roots: list[Path],
    output_dir: Path,
    pit_start_date: str = "2024-01-02",
    pit_end_date: str = _PIT_END_DEFAULT,
) -> dict[str, int]:
    """Generate one markdown transaction log per (FinRL agent, seed) pair.

    ``finrl_seed_run_roots`` is a list of per-seed run directories.  Agents are
    auto-detected from subdirectory names under ``data/finrl_baseline_suite/``.

    Returns a dict mapping output file stem to the number of transaction rows
    written.
    """
    results: dict[str, int] = {}

    for seed_index, seed_root in enumerate(sorted(finrl_seed_run_roots), start=1):
        suite_dir = seed_root / "data" / "finrl_baseline_suite"
        if not suite_dir.exists():
            logger.warning("FinRL suite dir not found in %s — skipping.", seed_root)
            continue

        agents = sorted([d.name for d in suite_dir.iterdir() if d.is_dir()])
        for agent in agents:
            out_path = output_dir / f"{agent}_seed_{seed_index}.md"
            try:
                n = _write_finrl_log(
                    seed_root, agent, seed_index, out_path, pit_start_date, pit_end_date
                )
                results[out_path.stem] = n
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FinRL log generation failed for %s seed %d: %s",
                    agent,
                    seed_index,
                    exc,
                )

    logger.info(
        "FinRL transaction logs: %d files written to %s", len(results), output_dir
    )
    return results


# ---------------------------------------------------------------------------
# IQN
# ---------------------------------------------------------------------------


def _write_iqn_log(
    seed_run_root: Path,
    seed_index: int,
    output_path: Path,
) -> int:
    """Write one markdown transaction log for an IQN seed run.

    Returns the number of non-HOLD decision rows written.
    """
    json_candidates = sorted((seed_run_root / "data").glob("*decision_memory*.json"))
    if not json_candidates:
        logger.warning("No decision_memory JSON in %s — skipping.", seed_run_root)
        return 0

    try:
        raw: dict[str, Any] = json.loads(json_candidates[0].read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Cannot parse decision JSON in %s: %s", seed_run_root, exc)
        return 0

    decisions: list[dict[str, Any]] = raw.get("decisions", [])
    if not decisions:
        logger.warning("Empty decisions list in %s.", json_candidates[0])
        return 0

    # Action distribution.
    action_counts: dict[str, int] = {}
    for d in decisions:
        label = d.get("effective_decision_action_label", "UNKNOWN")
        action_counts[label] = action_counts.get(label, 0) + 1

    total_decisions = len(decisions)
    all_labels = ["HOLD", "BUY", "SELL", "REBALANCE", "CHANGE_STRATEGY"]

    non_hold = [
        d
        for d in decisions
        if d.get("effective_decision_action_label", "HOLD") != "HOLD"
    ]

    lines: list[str] = [
        f"# IQN Agent — Seed {seed_index} — Transaction Log",
        "",
        "**Agent**: IQN (Implicit Quantile Network, distributional RL)",
        f"**Seed**: {seed_index}",
        f"**Total decision steps recorded**: {total_decisions}",
        "",
        "## Action distribution",
        "",
        "| Action | Count | Percentage |",
        "| :--- | ---: | ---: |",
    ]

    for label in all_labels:
        count = action_counts.get(label, 0)
        pct = count / total_decisions * 100 if total_decisions > 0 else 0.0
        lines.append(f"| {label} | {count:,} | {pct:.1f}% |")

    lines += [
        "",
        "## Non-HOLD transactions",
        "",
    ]

    if not non_hold:
        lines.append("_No non-HOLD decisions recorded._")
    else:
        lines += [
            "| Step | Action | Ticker | Requested shares | Reward |",
            "| ---: | :--- | :--- | ---: | ---: |",
        ]
        for d in non_hold:
            step = d.get("decision_step", "—")
            label = d.get("effective_decision_action_label", "—")
            resolved = d.get("resolved_action", {})
            ticker = resolved.get("selected_ticker") or "—"
            shares = resolved.get("requested_shares", 0)
            reward = d.get("reward", 0.0)
            lines.append(
                f"| {step} | {label} | {ticker} | {int(shares):,} | {reward:.4f} |"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(non_hold)


def generate_iqn_transaction_logs(
    iqn_seed_run_roots: list[Path],
    output_dir: Path,
) -> dict[str, int]:
    """Generate one markdown transaction log per IQN seed run.

    ``iqn_seed_run_roots`` is a list of per-seed run directories, each
    containing ``data/iqn_learning_curve_train_decision_memory.json``.

    Returns a dict mapping output file stem to the number of non-HOLD decision
    rows written.
    """
    results: dict[str, int] = {}

    for seed_index, seed_root in enumerate(sorted(iqn_seed_run_roots), start=1):
        out_path = output_dir / f"seed_{seed_index}.md"
        try:
            n = _write_iqn_log(seed_root, seed_index, out_path)
            results[out_path.stem] = n
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "IQN log generation failed for seed %d (%s): %s",
                seed_index,
                seed_root,
                exc,
            )

    logger.info(
        "IQN transaction logs: %d files written to %s", len(results), output_dir
    )
    return results
