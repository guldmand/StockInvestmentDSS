# src/stock_investment_dss/runner/run_hierarchical_policy_smoke_test.py
"""
Standalone smoke test for the D-IQN-DSS Hierarchical Decision Policy (v3.0 PoC).

This runner does NOT require a trained IQN model. It simulates the action
type via environment variable and exercises all 5 stages of the hierarchical
policy on real market data.

Usage
-----
From repository root with PYTHONPATH=src:

    # BUY scenario
    $env:PYTHONPATH = "src"
    $env:STOCK_INVESTMENT_DSS_HIERARCHICAL_ACTION_TYPE = "BUY"
    python -m stock_investment_dss.runner.run_hierarchical_policy_smoke_test

    # HOLD scenario (cash-only portfolio)
    $env:PYTHONPATH = "src"
    $env:STOCK_INVESTMENT_DSS_HIERARCHICAL_ACTION_TYPE = "HOLD"
    $env:STOCK_INVESTMENT_DSS_HIERARCHICAL_CASH_WEIGHT = "1.0"
    python -m stock_investment_dss.runner.run_hierarchical_policy_smoke_test

    # SELL scenario (with held positions)
    $env:PYTHONPATH = "src"
    $env:STOCK_INVESTMENT_DSS_HIERARCHICAL_ACTION_TYPE = "SELL"
    python -m stock_investment_dss.runner.run_hierarchical_policy_smoke_test

Environment variables
---------------------
STOCK_INVESTMENT_DSS_HIERARCHICAL_ACTION_TYPE   : HOLD | BUY | SELL | REBALANCE (default: BUY)
STOCK_INVESTMENT_DSS_HIERARCHICAL_CASH_WEIGHT   : float 0-1 (default: 0.80)
STOCK_INVESTMENT_DSS_HIERARCHICAL_TICKERS       : comma-separated (default: AAPL,MSFT,NVDA,AMZN,GOOGL)
STOCK_INVESTMENT_DSS_HIERARCHICAL_STRATEGY      : balanced_v1 | defensive_v1 | aggressive_v1 (default: balanced_v1)
STOCK_INVESTMENT_DSS_HIERARCHICAL_DECISION_DATE : ISO date (default: latest in market data)
STOCK_INVESTMENT_DSS_HIERARCHICAL_STEPS         : number of decision steps to simulate (default: 5)

Outputs (written to outputs/runs/<timestamp>_d_iqn_dss_hierarchical_policy_smoke_test/)
-----------
audit/hierarchical_decision_by_step.csv
audit/ticker_score_table.csv
audit/size_score_table.csv
data/hierarchical_technical_features.csv
data/hierarchical_fundamental_features.csv
data/hierarchical_joined_features.csv
summary/hierarchical_policy_summary.json
summary/hierarchical_policy_summary.md
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Market data loading
# ---------------------------------------------------------------------------

_DEFAULT_IMPORT_FILE = "data/market/daily/imports/market_data_full_500.csv"

_REQUIRED_COLUMNS = {"date", "tic", "close"}
_OPTIONAL_COLUMNS = {
    "macd",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
}


def load_market_data(import_file: str, tickers: list[str]) -> pd.DataFrame:
    path = Path(import_file)
    if not path.exists():
        logger.error(
            "Market data file not found: %s\n"
            "Expected file: %s\n"
            "This file must exist for the smoke test to run.\n"
            "It is typically the Mode B frozen import file:\n"
            "  data/market/daily/imports/market_data_full_500.csv",
            import_file,
            path.resolve(),
        )
        raise FileNotFoundError(
            f"Market data not found at: {path.resolve()}\n"
            "Please ensure the frozen import CSV exists. "
            "Run Mode A data download first, or copy the file manually."
        )

    logger.info("Loading market data from: %s", path)
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower()

    missing_req = _REQUIRED_COLUMNS - set(df.columns)
    if missing_req:
        raise ValueError(
            f"Market data missing required columns: {missing_req}. "
            f"Available: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"])

    tic_col = "tic" if "tic" in df.columns else "ticker"
    df_filtered = df[df[tic_col].isin(tickers)].copy()
    if df_filtered.empty:
        available = sorted(df[tic_col].unique().tolist())[:20]
        raise ValueError(
            f"No rows found for tickers {tickers} in market data.\n"
            f"Available tickers (first 20): {available}"
        )

    found = sorted(df_filtered[tic_col].unique().tolist())
    missing_tickers = [t for t in tickers if t not in found]
    if missing_tickers:
        logger.warning("Missing tickers in market data: %s", missing_tickers)

    logger.info("Loaded %d rows for tickers: %s", len(df_filtered), found)
    return df_filtered


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------


def _build_demo_portfolio(
    cash_weight: float,
    tickers: list[str],
    features: pd.DataFrame,
    total_value: float = 1_000_000.0,
) -> "PortfolioState":
    from stock_investment_dss.decision.hierarchical_decision_policy import (
        PortfolioState,
    )

    cash = cash_weight * total_value
    equity_value = (1.0 - cash_weight) * total_value

    # Distribute remaining equity evenly across tickers (simplified PoC allocation)
    n = len(tickers)
    if n == 0 or equity_value < 1.0:
        return PortfolioState(total_value=total_value, cash=total_value)

    per_ticker = equity_value / n
    holding_values = {t: per_ticker for t in tickers}

    return PortfolioState(
        total_value=total_value,
        cash=cash,
        holdings={t: 0 for t in tickers},  # shares not tracked in PoC
        holding_values=holding_values,
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _decisions_to_df(decisions: list[dict]) -> pd.DataFrame:
    rows = []
    for d in decisions:
        flat = {
            "decision_id": d["decision_id"],
            "date": d["date"],
            "visible_data_cutoff": d["visible_data_cutoff"],
            "strategy_id": d["strategy_id"],
            "score_mode": d["score_mode"],
            "selected_action_type": d["selected_action_type"],
            "selected_ticker": d["selected_ticker"],
            "selected_size": d["selected_size"],
            "risk_adjusted_allocation_fraction": d["risk_adjusted_allocation_fraction"],
            "portfolio_total_value": d["portfolio_state"]["total_value"],
            "portfolio_cash_weight": d["portfolio_state"]["cash_weight"],
        }
        # Risk checks
        for k, v in d.get("risk_checks", {}).items():
            if not isinstance(v, list):
                flat[f"risk_{k}"] = v
        # Recommendation
        rec = d.get("final_recommendation", {})
        flat["recommendation_explanation"] = rec.get("explanation", "")
        flat["recommendation_warnings"] = " | ".join(rec.get("warnings", []))
        # Execution
        exc = d.get("execution_result", {})
        flat["trade_value_estimate"] = exc.get("trade_value_estimate", 0.0)
        flat["transaction_cost_estimate"] = exc.get("transaction_cost_estimate", 0.0)
        rows.append(flat)
    return pd.DataFrame(rows)


def _ticker_scores_to_df(decisions: list[dict]) -> pd.DataFrame:
    rows = []
    for d in decisions:
        for ts in d.get("stage_2_ticker_scores", []):
            row = {
                "decision_id": d["decision_id"],
                "date": d["date"],
                "action_type": d["selected_action_type"],
            }
            row.update(ts)
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _size_scores_to_df(decisions: list[dict]) -> pd.DataFrame:
    rows = []
    for d in decisions:
        for ss in d.get("stage_3_size_scores", []):
            row = {
                "decision_id": d["decision_id"],
                "date": d["date"],
                "action_type": d["selected_action_type"],
            }
            row.update(ss)
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_summary_md(decisions: list[dict], run_id: str, config: dict) -> str:
    lines = [
        f"# Hierarchical Policy Smoke Test — {run_id}",
        "",
        "## Configuration",
        f"- Strategy: `{config.get('strategy_id', 'N/A')}`",
        f"- Tickers: `{', '.join(config.get('tickers', []))}`",
        f"- Action type (forced): `{config.get('action_type', 'N/A')}`",
        f"- Initial cash weight: `{config.get('cash_weight', 'N/A')}`",
        f"- Steps simulated: `{len(decisions)}`",
        "",
        "## Decision Summary",
        "",
        "| # | Date | Action | Ticker | Size | Allocation | Cash % |",
        "|---|------|--------|--------|------|------------|--------|",
    ]
    for i, d in enumerate(decisions, 1):
        lines.append(
            f"| {i} | {d['date']} | {d['selected_action_type']} | "
            f"{d['selected_ticker'] or '—'} | {d['selected_size'] or '—'} | "
            f"{d['risk_adjusted_allocation_fraction']:.1%} | "
            f"{d['portfolio_state']['cash_weight']:.1%} |"
        )

    lines += [
        "",
        "## Notes",
        "- This smoke test does NOT use a trained IQN model.",
        "- Action type is forced via `STOCK_INVESTMENT_DSS_HIERARCHICAL_ACTION_TYPE`.",
        "- Fundamentals are `frozen_snapshot_placeholder` (v3.0 PoC).",
        "- Technical features are computed from historical market data.",
        "- No training was run.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    _setup_logging()

    # --- Configuration from environment ---
    action_type_str = _env(
        "STOCK_INVESTMENT_DSS_HIERARCHICAL_ACTION_TYPE", "BUY"
    ).upper()
    cash_weight = float(_env("STOCK_INVESTMENT_DSS_HIERARCHICAL_CASH_WEIGHT", "0.80"))
    tickers_str = _env(
        "STOCK_INVESTMENT_DSS_HIERARCHICAL_TICKERS", "AAPL,MSFT,NVDA,AMZN,GOOGL"
    )
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    strategy_str = _env("STOCK_INVESTMENT_DSS_HIERARCHICAL_STRATEGY", "balanced_v1")
    decision_date_str = _env("STOCK_INVESTMENT_DSS_HIERARCHICAL_DECISION_DATE", "")
    n_steps = int(_env("STOCK_INVESTMENT_DSS_HIERARCHICAL_STEPS", "5"))
    import_file = _env(
        "STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE", _DEFAULT_IMPORT_FILE
    )

    logger.info("=== Hierarchical Policy Smoke Test (v3.0 PoC) ===")
    logger.info("Action type (forced): %s", action_type_str)
    logger.info("Tickers: %s", tickers)
    logger.info("Strategy: %s", strategy_str)
    logger.info("Cash weight: %.2f", cash_weight)
    logger.info("Steps: %d", n_steps)

    # --- Imports (deferred to catch import errors cleanly) ---
    try:
        from stock_investment_dss.decision.decision_actions import (
            DSSDecisionAction,
            parse_action_label,
        )
        from stock_investment_dss.decision.hierarchical_decision_policy import (
            HierarchicalDecisionPolicy,
            PortfolioState,
        )
        from stock_investment_dss.decision.investor_risk_profile import (
            InvestorRiskProfile,
        )
        from stock_investment_dss.data.technical_feature_builder import (
            TechnicalFeatureBuilder,
        )
        from stock_investment_dss.data.fundamental_feature_store import (
            FundamentalFeatureStore,
        )
        from stock_investment_dss.utilities.paths import create_run_paths
    except ImportError as e:
        logger.error("Import error — is PYTHONPATH set to 'src'?\n  %s", e)
        return 1

    # --- Parse action type ---
    try:
        action_type = parse_action_label(action_type_str)
    except ValueError as e:
        logger.error("Invalid action type: %s", e)
        return 1

    # --- Risk profile ---
    if strategy_str == "defensive_v1":
        risk_profile = InvestorRiskProfile.defensive()
        defensive = True
    elif strategy_str == "aggressive_v1":
        risk_profile = InvestorRiskProfile.aggressive()
        defensive = False
    else:
        risk_profile = InvestorRiskProfile.balanced()
        defensive = False

    # --- Run paths ---
    run_paths = create_run_paths("d_iqn_dss_hierarchical_policy_smoke_test")
    run_paths.run_directory.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_paths.run_directory)

    # --- Load market data ---
    try:
        raw_df = load_market_data(import_file, tickers)
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1

    # --- Build technical features ---
    logger.info("Building technical features ...")
    builder = TechnicalFeatureBuilder()
    tech_df = builder.build(raw_df)
    _write_csv(
        run_paths.data_directory / "hierarchical_technical_features.csv", tech_df
    )
    logger.info(
        "Technical features: %d rows, %d columns", len(tech_df), len(tech_df.columns)
    )

    # --- Load fundamental features ---
    logger.info("Loading fundamental features (frozen placeholder) ...")
    fund_store = FundamentalFeatureStore()

    # --- Determine decision dates ---
    tech_df["date"] = pd.to_datetime(tech_df["date"])
    all_dates = sorted(tech_df["date"].unique())

    if decision_date_str:
        start_date = pd.Timestamp(decision_date_str)
    else:
        start_date = all_dates[-n_steps] if len(all_dates) >= n_steps else all_dates[0]

    decision_dates = [d for d in all_dates if d >= start_date][:n_steps]
    logger.info(
        "Decision dates: %s → %s (%d steps)",
        decision_dates[0].date(),
        decision_dates[-1].date(),
        len(decision_dates),
    )

    # --- Build policy ---
    policy = HierarchicalDecisionPolicy(
        risk_profile=risk_profile,
        strategy_id=strategy_str,
        defensive_strategy=defensive,
    )

    # --- Portfolio state (fixed for smoke test) ---
    # Get the snapshot features at first date to populate initial portfolio
    first_snap = builder.build_latest_snapshot(raw_df, str(decision_dates[0].date()))
    portfolio = _build_demo_portfolio(
        cash_weight=cash_weight,
        tickers=tickers,
        features=first_snap,
        total_value=1_000_000.0,
    )

    # --- Run decision loop ---
    all_decisions: list[dict] = []
    all_fund_rows: list[pd.DataFrame] = []

    for step_date in decision_dates:
        date_str = str(step_date.date())

        # Snapshot features at this date (point-in-time)
        snap = builder.build_latest_snapshot(raw_df, date_str)

        # Fundamental features (PIT filtered)
        fund_df = fund_store.get_scores_as_of(date_str, tickers)
        all_fund_rows.append(fund_df.assign(decision_date=date_str))

        # Join technical + fundamental on ticker
        tic_col_snap = "tic" if "tic" in snap.columns else "ticker"
        tic_col_fund = "ticker"
        joined = snap.merge(
            fund_df.rename(columns={"ticker": tic_col_snap}),
            on=tic_col_snap,
            how="left",
            suffixes=("", "_fund"),
        )

        # Run the hierarchical policy
        decision = policy.decide(
            action_type=action_type,
            features=joined,
            portfolio=portfolio,
            decision_date=date_str,
            visible_data_cutoff=date_str,
            score_mode="smoke_test_forced_action",
        )
        all_decisions.append(decision.to_dict())

        logger.info(
            "Step %s: %s %s %s (cash: %.1f%%)",
            date_str,
            decision.selected_action_type,
            decision.selected_ticker or "—",
            decision.selected_size or "—",
            decision.portfolio_state["cash_weight"] * 100,
        )

    # --- Save joined features (last step snapshot) ---
    last_date = str(decision_dates[-1].date())
    last_snap = builder.build_latest_snapshot(raw_df, last_date)
    last_fund = fund_store.get_scores_as_of(last_date, tickers)
    tic_col_snap = "tic" if "tic" in last_snap.columns else "ticker"
    joined_final = last_snap.merge(
        last_fund.rename(columns={"ticker": tic_col_snap}),
        on=tic_col_snap,
        how="left",
        suffixes=("", "_fund"),
    )
    _write_csv(
        run_paths.data_directory / "hierarchical_joined_features.csv", joined_final
    )

    # Save fundamental features across all dates
    if all_fund_rows:
        fund_all = pd.concat(all_fund_rows, ignore_index=True)
        _write_csv(
            run_paths.data_directory / "hierarchical_fundamental_features.csv", fund_all
        )

    # --- Audit outputs ---
    decisions_df = _decisions_to_df(all_decisions)
    _write_csv(
        run_paths.audit_directory / "hierarchical_decision_by_step.csv", decisions_df
    )

    ticker_scores_df = _ticker_scores_to_df(all_decisions)
    if not ticker_scores_df.empty:
        _write_csv(
            run_paths.audit_directory / "ticker_score_table.csv", ticker_scores_df
        )

    size_scores_df = _size_scores_to_df(all_decisions)
    if not size_scores_df.empty:
        _write_csv(run_paths.audit_directory / "size_score_table.csv", size_scores_df)

    # --- Summary ---
    config_summary = {
        "action_type": action_type_str,
        "strategy_id": strategy_str,
        "tickers": tickers,
        "cash_weight": cash_weight,
        "n_steps": n_steps,
        "decision_dates": [str(d.date()) for d in decision_dates],
        "risk_profile": risk_profile.to_dict(),
        "import_file": import_file,
        "run_id": run_paths.run_id,
        "run_directory": str(run_paths.run_directory),
        "patch_version": "v3.0",
        "fundamentals_source": "frozen_snapshot_placeholder",
        "iqn_model": "NOT_USED (smoke test — forced action type)",
        "decisions": all_decisions,
    }

    _write_json(
        run_paths.summary_directory / "hierarchical_policy_summary.json", config_summary
    )

    summary_md = _build_summary_md(all_decisions, run_paths.run_id, config_summary)
    _write_md(
        run_paths.summary_directory / "hierarchical_policy_summary.md", summary_md
    )

    logger.info("=== Smoke test complete ===")
    logger.info("Output: %s", run_paths.run_directory)
    logger.info("Decisions written: %d", len(all_decisions))

    # Print brief result table
    print("\n--- Decision Summary ---")
    for d in all_decisions:
        print(
            f"  {d['date']}  {d['selected_action_type']:10s}  "
            f"ticker={d['selected_ticker'] or 'NONE':6s}  "
            f"size={d['selected_size'] or 'N/A':8s}  "
            f"cash={d['portfolio_state']['cash_weight']:.1%}"
        )
    print(f"\nOutput directory: {run_paths.run_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
