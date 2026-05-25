# src/stock_investment_dss/runner/run_combined_iqn_hdp_audit_smoke_test.py
"""
Combined IQN + HDP Audit Smoke Test runner.

Combines:
  1. IQN decision/distribution outputs from an existing IQN run (if discoverable).
  2. Latest HDP enriched feature table (from FMP/HDP feature smoke test).
  3. HDP ticker/size selection via HierarchicalDecisionPolicy.
  4. A combined audit CSV for EDL-A counterfactual supervised learning.

If no IQN decision files are found, a manual fallback action type is used.
This allows end-to-end testing of the HDP join/audit plumbing without a trained model.

Constraints
-----------
- Does NOT retrain IQN.
- Does NOT modify IQN core or EDL files.
- Does NOT run training.
- Does NOT make live FMP calls.
- Uses only cached/frozen HDP joined feature table.

Usage
-----
From repository root with PYTHONPATH=src:

    $env:PYTHONPATH = "src"
    $env:STOCK_INVESTMENT_DSS_COMBINED_START_DATE = "2024-01-01"
    $env:STOCK_INVESTMENT_DSS_COMBINED_END_DATE   = "2024-02-01"
    $env:STOCK_INVESTMENT_DSS_COMBINED_ACTION_TYPE = "BUY"
    python -m stock_investment_dss.runner.run_combined_iqn_hdp_audit_smoke_test

Environment variables
---------------------
STOCK_INVESTMENT_DSS_COMBINED_IQN_SOURCE_RUN_ID        : explicit IQN run directory name under outputs/runs/
STOCK_INVESTMENT_DSS_COMBINED_IQN_DECISION_CSV         : direct path to a date-indexed IQN decision CSV (overrides run discovery)
STOCK_INVESTMENT_DSS_COMBINED_HDP_FEATURE_CSV              : direct path to hdp_joined_features.csv (overrides run discovery)
STOCK_INVESTMENT_DSS_COMBINED_HDP_FEATURE_SOURCE_RUN_ID: explicit HDP run directory name under outputs/runs/
STOCK_INVESTMENT_DSS_COMBINED_START_DATE               : ISO date (default: 2024-01-01)
STOCK_INVESTMENT_DSS_COMBINED_END_DATE                 : ISO date (default: 2024-02-01)
STOCK_INVESTMENT_DSS_COMBINED_TICKERS                  : comma-separated (default: AAPL,MSFT,NVDA,AMZN,GOOGL)
STOCK_INVESTMENT_DSS_COMBINED_ACTION_TYPE              : fallback action if no IQN source found (default: BUY)
STOCK_INVESTMENT_DSS_COMBINED_CASH_WEIGHT              : demo portfolio cash weight (default: 0.80)

Outputs (written to outputs/runs/<timestamp>_d_iqn_dss_combined_iqn_hdp_audit_smoke_test/)
-----------
audit/combined_iqn_hdp_decision_audit.csv
audit/combined_iqn_hdp_ticker_candidates.csv
audit/combined_iqn_hdp_size_candidates.csv
summary/combined_iqn_hdp_audit_summary.json
summary/combined_iqn_hdp_audit_summary.md
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


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


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# IQN source discovery
# ---------------------------------------------------------------------------

_IQN_RUN_PATTERN = re.compile(
    r"iqn_learning_curve_smoke_test|iqn_eval|iqn_backtest|iqn_decision"
)

_DECISION_CSV_KEYWORDS = [
    "decision_table",
    "decisions",
    "decision",
    "step_table",
    "eval_step",
]
_DIST_CSV_KEYWORDS = [
    "distribution_table",
    "distributions",
    "quantile_table",
    "quantile",
]


def _find_latest_run(runs_dir: Path, pattern: re.Pattern) -> Optional[Path]:
    candidates = [
        d for d in runs_dir.iterdir() if d.is_dir() and pattern.search(d.name)
    ]
    return sorted(candidates)[-1] if candidates else None


def _csv_has_date(csv_path: Path) -> bool:
    try:
        df = pd.read_csv(csv_path, nrows=1)
        return "date" in df.columns
    except Exception:
        return False


def _discover_iqn_source(
    runs_dir: Path,
    explicit_run_id: str,
    direct_csv_path: str = "",
) -> tuple[bool, str, dict[str, str]]:
    """
    Search for an IQN run directory containing decision/distribution CSVs.

    Priority:
    1. Direct CSV path override (STOCK_INVESTMENT_DSS_COMBINED_IQN_DECISION_CSV).
    2. Explicit run directory override.
    3. Auto-discover latest matching IQN run.

    Returns
    -------
    (iqn_source_available, resolved_run_id, {kind: csv_path_str})
    """
    if direct_csv_path:
        p = Path(direct_csv_path)
        if p.exists() and _csv_has_date(p):
            logger.info("Using direct IQN decision CSV: %s", p.name)
            return True, f"direct_csv:{p.parent.parent.name}", {"decision": str(p)}
        logger.warning("Direct IQN CSV not found or has no date column: %s", p)

    if explicit_run_id:
        run_dir = runs_dir / explicit_run_id
        if not run_dir.exists():
            logger.warning("Explicit IQN source run not found: %s", run_dir)
            return False, explicit_run_id, {}
    else:
        run_dir = _find_latest_run(runs_dir, _IQN_RUN_PATTERN)
        if run_dir is None:
            logger.info("No IQN run directories found in %s", runs_dir)
            return False, "", {}

    logger.info("Searching IQN run for decision/distribution files: %s", run_dir.name)
    iqn_data: dict[str, str] = {}

    for csv_file in sorted(run_dir.rglob("*.csv")):
        stem = csv_file.stem.lower()
        if "decision" not in iqn_data and any(
            k in stem for k in _DECISION_CSV_KEYWORDS
        ):
            if _csv_has_date(csv_file):
                iqn_data["decision"] = str(csv_file)
        if "distribution" not in iqn_data and any(
            k in stem for k in _DIST_CSV_KEYWORDS
        ):
            iqn_data["distribution"] = str(csv_file)

    available = "decision" in iqn_data
    if not available:
        logger.info(
            "IQN run found (%s) but no usable decision CSV with date column.",
            run_dir.name,
        )
    return available, run_dir.name, iqn_data


def _load_iqn_decisions(
    decision_csv: str,
    start_date: str,
    end_date: str,
    tickers: list[str],
) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(decision_csv, low_memory=False)
    except Exception as e:
        logger.warning("Failed to load IQN decision CSV: %s", e)
        return None
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= pd.Timestamp(start_date)) & (
        df["date"] <= pd.Timestamp(end_date)
    )
    tic_col = (
        "ticker" if "ticker" in df.columns else ("tic" if "tic" in df.columns else None)
    )
    if tic_col:
        mask &= df[tic_col].isin(tickers)
    result = df[mask].copy()
    logger.info("Loaded %d IQN decision rows (filtered by date + tickers)", len(result))
    return result if not result.empty else None


# ---------------------------------------------------------------------------
# HDP source discovery
# ---------------------------------------------------------------------------

_HDP_RUN_PATTERN = re.compile(r"fmp_hdp_feature_smoke_test")


def _discover_hdp_source(
    runs_dir: Path,
    explicit_run_id: str,
    data_dir: Path,
    direct_csv_path: str = "",
) -> tuple[Optional[Path], str]:
    """
    Find the HDP joined features CSV.

    Priority:
    1. Direct CSV path override (STOCK_INVESTMENT_DSS_COMBINED_HDP_FEATURE_CSV).
    2. Explicit run directory override.
    3. Project data/ directory (data/hdp_joined_features.csv).
    4. Latest fmp_hdp_feature_smoke_test run in project outputs/runs/.
    5. Same search one level up (D-IQN-DSS/outputs/runs/), handles cases where
       the FMP/HDP runner used a different project root.

    Returns (csv_path, run_id_label)
    """
    if direct_csv_path:
        p = Path(direct_csv_path)
        if p.exists():
            return p, f"direct_csv:{p.name}"
        logger.warning("Direct HDP CSV path not found: %s", p)

    if explicit_run_id:
        candidate = runs_dir / explicit_run_id / "data" / "hdp_joined_features.csv"
        if candidate.exists():
            return candidate, explicit_run_id
        logger.warning(
            "Explicit HDP source run not found or missing CSV: %s", candidate
        )

    root_candidate = data_dir / "hdp_joined_features.csv"
    if root_candidate.exists():
        return root_candidate, "project_data"

    # Search project-level runs/ first, then parent-level runs/ as fallback
    parent_runs = runs_dir.parent.parent.parent / "outputs" / "runs"
    for search_dir in [runs_dir, parent_runs]:
        if not search_dir.exists():
            continue
        run_dir = _find_latest_run(search_dir, _HDP_RUN_PATTERN)
        if run_dir:
            candidate = run_dir / "data" / "hdp_joined_features.csv"
            if candidate.exists():
                logger.info("Found HDP features in: %s", run_dir)
                return candidate, run_dir.name

    return None, ""


# ---------------------------------------------------------------------------
# Demo portfolio construction
# ---------------------------------------------------------------------------


def _build_demo_portfolio(
    cash_weight: float,
    tickers: list[str],
    total_value: float = 1_000_000.0,
) -> "PortfolioState":
    from stock_investment_dss.decision.hierarchical_decision_policy import (
        PortfolioState,
    )

    cash = cash_weight * total_value
    equity_value = (1.0 - cash_weight) * total_value
    n = len(tickers)
    if n == 0 or equity_value < 1.0:
        return PortfolioState(total_value=total_value, cash=total_value)
    per_ticker = equity_value / n
    return PortfolioState(
        total_value=total_value,
        cash=cash,
        holdings={t: 0 for t in tickers},
        holding_values={t: per_ticker for t in tickers},
    )


# ---------------------------------------------------------------------------
# PIT validation helpers
# ---------------------------------------------------------------------------


def _pit_check_date(
    features: pd.DataFrame,
    date_ts: pd.Timestamp,
) -> list[str]:
    """Return list of tickers with PIT violation on this date."""
    violations = []
    if "known_at_effective_date" not in features.columns:
        return violations
    for _, row in features.iterrows():
        ked = row.get("known_at_effective_date")
        ticker = row.get("ticker", row.get("tic", "?"))
        if pd.notna(ked) and pd.Timestamp(ked) > date_ts:
            violations.append(str(ticker))
    return violations


# ---------------------------------------------------------------------------
# Audit row builders
# ---------------------------------------------------------------------------

_HDP_SELECTED_TICKER_COLS = [
    "known_at",
    "known_at_effective_date",
    "fundamental_lag_days",
    "point_in_time_quality",
    "close",
    "ma50",
    "ma200",
    "sma50",
    "sma200",
    "price_vs_ma50",
    "price_vs_ma200",
    "recent_return_20d",
    "volatility_20d",
    "drawdown_from_recent_high",
    "momentum_score",
    "technical_risk_score",
    "revenue",
    "revenue_growth",
    "earnings_growth",
    "gross_margin",
    "operating_margin",
    "profit_margin",
    "roe",
    "current_ratio",
    "debt_ratio",
    "free_cash_flow",
    "fcf_margin",
    "annualized_revenue",
    "annualized_net_income",
    "annualized_free_cash_flow",
    "market_cap_estimate",
    "enterprise_value_estimate",
    "pe_ratio",
    "ps_ratio",
    "ev_ebitda",
    "fcf_yield",
    "value_score",
    "quality_score",
    "profitability_score",
    "balance_sheet_strength_score",
    "ticker_score",
    "valuation_method",
    "valuation_warning",
    "feature_warning",
]


def _hdp_row_fields(
    features: pd.DataFrame,
    ticker: Optional[str],
) -> dict:
    if ticker is None:
        return {col: None for col in _HDP_SELECTED_TICKER_COLS}
    tic_col = "ticker" if "ticker" in features.columns else "tic"
    rows = features[features[tic_col] == ticker]
    if rows.empty:
        return {col: None for col in _HDP_SELECTED_TICKER_COLS}
    r = rows.iloc[0]
    return {
        col: (r[col] if col in features.columns else None)
        for col in _HDP_SELECTED_TICKER_COLS
    }


def _iqn_fields_from_row(iqn_row: Optional[pd.Series]) -> dict:
    if iqn_row is None:
        return {
            "iqn_selected_action": None,
            "iqn_action_score_hold": None,
            "iqn_action_score_buy": None,
            "iqn_action_score_sell": None,
            "iqn_action_score_rebalance": None,
            "q10_hold": None,
            "q50_hold": None,
            "q90_hold": None,
            "cvar_hold": None,
            "q10_buy": None,
            "q50_buy": None,
            "q90_buy": None,
            "cvar_buy": None,
            "q10_sell": None,
            "q50_sell": None,
            "q90_sell": None,
            "cvar_sell": None,
            "q10_rebalance": None,
            "q50_rebalance": None,
            "q90_rebalance": None,
            "cvar_rebalance": None,
            "iqn_uncertainty_proxy": None,
            "iqn_risk_score": None,
        }
    r = iqn_row

    def _g(*keys):
        for k in keys:
            v = r.get(k)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                return v
        return None

    action_label = _g(
        "chosen_action_label",
        "effective_action",
        "iqn_selected_action",
        "selected_action_type",
        "action_label",
    )
    return {
        "iqn_selected_action": action_label,
        "iqn_action_score_hold": _g("score_hold", "q_hold", "action_score_hold"),
        "iqn_action_score_buy": _g("score_buy", "q_buy", "action_score_buy"),
        "iqn_action_score_sell": _g("score_sell", "q_sell", "action_score_sell"),
        "iqn_action_score_rebalance": _g(
            "score_rebalance", "q_rebalance", "action_score_rebalance"
        ),
        "q10_hold": _g("q10_hold"),
        "q50_hold": _g("q50_hold"),
        "q90_hold": _g("q90_hold"),
        "cvar_hold": _g("cvar_hold"),
        "q10_buy": _g("q10_buy"),
        "q50_buy": _g("q50_buy"),
        "q90_buy": _g("q90_buy"),
        "cvar_buy": _g("cvar_buy"),
        "q10_sell": _g("q10_sell"),
        "q50_sell": _g("q50_sell"),
        "q90_sell": _g("q90_sell"),
        "cvar_sell": _g("cvar_sell"),
        "q10_rebalance": _g("q10_rebalance"),
        "q50_rebalance": _g("q50_rebalance"),
        "q90_rebalance": _g("q90_rebalance"),
        "cvar_rebalance": _g("cvar_rebalance"),
        "iqn_uncertainty_proxy": _g("uncertainty_proxy", "iqn_uncertainty_proxy"),
        "iqn_risk_score": _g("risk_score", "iqn_risk_score"),
    }


def _size_reason_codes(ss_dict: dict) -> str:
    penalties = []
    for key in (
        "cash_buffer_penalty",
        "concentration_penalty",
        "volatility_penalty",
        "drawdown_penalty",
        "confidence_penalty",
        "trend_guard_penalty",
    ):
        val = ss_dict.get(key, 0.0) or 0.0
        if abs(val) > 0.001:
            label = key.replace("_penalty", "")
            penalties.append(f"{label}={val:.3f}")
    return "|".join(penalties) if penalties else "none"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    _setup_logging()

    iqn_source_run_id = _env("STOCK_INVESTMENT_DSS_COMBINED_IQN_SOURCE_RUN_ID", "")
    iqn_direct_csv = _env("STOCK_INVESTMENT_DSS_COMBINED_IQN_DECISION_CSV", "")
    hdp_source_run_id = _env(
        "STOCK_INVESTMENT_DSS_COMBINED_HDP_FEATURE_SOURCE_RUN_ID", ""
    )
    hdp_direct_csv = _env("STOCK_INVESTMENT_DSS_COMBINED_HDP_FEATURE_CSV", "")
    start_date = _env("STOCK_INVESTMENT_DSS_COMBINED_START_DATE", "2024-01-01")
    end_date = _env("STOCK_INVESTMENT_DSS_COMBINED_END_DATE", "2024-02-01")
    tickers_str = _env(
        "STOCK_INVESTMENT_DSS_COMBINED_TICKERS", "AAPL,MSFT,NVDA,AMZN,GOOGL"
    )
    manual_action_str = _env("STOCK_INVESTMENT_DSS_COMBINED_ACTION_TYPE", "BUY").upper()
    cash_weight = float(_env("STOCK_INVESTMENT_DSS_COMBINED_CASH_WEIGHT", "0.80"))

    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]

    logger.info("=== Combined IQN+HDP Audit Smoke Test ===")
    logger.info("Date range: %s to %s", start_date, end_date)
    logger.info("Tickers: %s", tickers)
    logger.info("Manual fallback action: %s", manual_action_str)
    logger.info("Cash weight (demo portfolio): %.2f", cash_weight)

    # --- Deferred imports ---
    try:
        from stock_investment_dss.decision.decision_actions import (
            DSSDecisionAction,
            parse_action_label,
            action_to_label,
        )
        from stock_investment_dss.decision.hierarchical_decision_policy import (
            HierarchicalDecisionPolicy,
            PortfolioState,
        )
        from stock_investment_dss.decision.investor_risk_profile import (
            InvestorRiskProfile,
        )
        from stock_investment_dss.utilities.paths import (
            create_run_paths,
            RUNS_DIRECTORY,
            DATA_DIRECTORY,
        )
    except ImportError as e:
        logger.error("Import error — is PYTHONPATH set to 'src'?\n  %s", e)
        return 1

    # Validate manual action string
    try:
        manual_action_type = parse_action_label(manual_action_str)
    except ValueError as e:
        logger.error("Invalid STOCK_INVESTMENT_DSS_COMBINED_ACTION_TYPE: %s", e)
        return 1

    runs_dir = RUNS_DIRECTORY

    # --- Discover IQN source ---
    iqn_available, iqn_run_id, iqn_data = _discover_iqn_source(
        runs_dir, iqn_source_run_id, iqn_direct_csv
    )
    iqn_decision_df: Optional[pd.DataFrame] = None

    if iqn_available:
        iqn_decision_df = _load_iqn_decisions(
            iqn_data["decision"], start_date, end_date, tickers
        )
        if iqn_decision_df is None:
            logger.warning(
                "IQN decision CSV found but no rows for date range/tickers. "
                "Falling back to manual action: %s",
                manual_action_str,
            )
            iqn_available = False

    action_source = "iqn_decision_table" if iqn_available else "manual_fallback"
    logger.info(
        "IQN source available: %s | Action source: %s", iqn_available, action_source
    )

    # --- Discover HDP source ---
    hdp_csv_path, hdp_run_id = _discover_hdp_source(
        runs_dir, hdp_source_run_id, DATA_DIRECTORY, hdp_direct_csv
    )
    if hdp_csv_path is None:
        logger.error(
            "No HDP joined feature table found.\n"
            "Run run_fmp_hdp_feature_smoke_test first to generate:\n"
            "  outputs/runs/<ts>_d_iqn_dss_fmp_hdp_feature_smoke_test/data/hdp_joined_features.csv"
        )
        return 1

    logger.info("Loading HDP features from: %s", hdp_csv_path)
    hdp_df_full = pd.read_csv(hdp_csv_path, low_memory=False)
    hdp_df_full["date"] = pd.to_datetime(hdp_df_full["date"])
    if "known_at_effective_date" in hdp_df_full.columns:
        hdp_df_full["known_at_effective_date"] = pd.to_datetime(
            hdp_df_full["known_at_effective_date"]
        )

    tic_col_hdp = "ticker" if "ticker" in hdp_df_full.columns else "tic"
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    hdp_df_full = hdp_df_full[
        hdp_df_full[tic_col_hdp].isin(tickers)
        & (hdp_df_full["date"] >= start_ts)
        & (hdp_df_full["date"] <= end_ts)
    ].copy()

    if hdp_df_full.empty:
        logger.error(
            "No HDP feature rows found for tickers=%s, dates=%s to %s.\n"
            "The HDP table may not cover this date range. Try widening the range or "
            "re-running run_fmp_hdp_feature_smoke_test with a matching date range.",
            tickers,
            start_date,
            end_date,
        )
        return 1

    logger.info("HDP feature rows loaded: %d", len(hdp_df_full))

    # --- IQN decision index (by date) ---
    iqn_by_date: dict[str, pd.Series] = {}
    if iqn_available and iqn_decision_df is not None:
        iqn_date_col = "date"
        for _, row in iqn_decision_df.iterrows():
            date_key = str(row[iqn_date_col])[:10]
            if date_key not in iqn_by_date:
                iqn_by_date[date_key] = row

    # --- HDP policy setup ---
    risk_profile = InvestorRiskProfile.balanced()
    policy = HierarchicalDecisionPolicy(risk_profile=risk_profile)

    # --- Demo portfolio ---
    portfolio = _build_demo_portfolio(cash_weight, tickers)

    # --- Date loop ---
    dates = sorted(hdp_df_full["date"].dt.strftime("%Y-%m-%d").unique().tolist())
    logger.info("Decision dates: %d", len(dates))

    audit_rows: list[dict] = []
    ticker_candidate_rows: list[dict] = []
    size_candidate_rows: list[dict] = []
    pit_violations_total = 0

    for date_str in dates:
        date_ts = pd.Timestamp(date_str)
        features = hdp_df_full[hdp_df_full["date"] == date_ts].copy()
        if features.empty:
            continue

        # PIT check for this date
        pit_viols = _pit_check_date(features, date_ts)
        pit_violations_total += len(pit_viols)
        if pit_viols:
            logger.warning("PIT violations on %s: %s", date_str, pit_viols)

        # Resolve action type
        iqn_row: Optional[pd.Series] = iqn_by_date.get(date_str)
        if iqn_available and iqn_row is not None:
            raw_label = str(
                iqn_row.get(
                    "chosen_action_label",
                    iqn_row.get(
                        "effective_action",
                        iqn_row.get(
                            "iqn_selected_action",
                            iqn_row.get("selected_action_type", manual_action_str),
                        ),
                    ),
                )
            ).upper()
            try:
                action_type = parse_action_label(raw_label)
            except ValueError:
                action_type = manual_action_type
        else:
            action_type = manual_action_type

        # IQN fields for this date
        iqn_f = _iqn_fields_from_row(iqn_row if iqn_available else None)

        # Call HDP decision policy
        try:
            dec = policy.decide(
                action_type=action_type,
                features=features,
                portfolio=portfolio,
                decision_date=date_str,
                iqn_model_run_id=iqn_run_id if iqn_available else None,
            )
        except Exception as exc:
            logger.error("Policy.decide() failed on %s: %s", date_str, exc)
            continue

        selected_ticker = dec.selected_ticker
        selected_action = dec.selected_action_type
        alloc_frac = dec.risk_adjusted_allocation_fraction or 0.0
        recommendation = (dec.final_recommendation or {}).get("explanation", "")

        # HDP features for selected ticker
        hdp_f = _hdp_row_fields(features, selected_ticker)

        # Core audit row
        audit_row: dict = {
            "decision_id": dec.decision_id,
            "date": date_str,
            "ticker_universe": "|".join(tickers),
            "selected_action_type": selected_action,
            "selected_ticker": selected_ticker,
            "selected_size": dec.selected_size,
            "selected_allocation_fraction": alloc_frac,
            "recommendation": recommendation,
            # Source metadata
            "iqn_source_run_id": iqn_run_id if iqn_available else "",
            "iqn_source_available": iqn_available,
            "action_source": action_source,
            "hdp_feature_source_run_id": hdp_run_id,
            "pit_violations_this_date": len(pit_viols),
        }
        audit_row.update(iqn_f)
        audit_row.update(hdp_f)
        audit_rows.append(audit_row)

        # Ticker candidate rows (all scored tickers)
        for ts_dict in dec.stage_2_ticker_scores:
            cand_ticker = ts_dict.get("ticker", "")
            cand_feat_rows = features[features[tic_col_hdp] == cand_ticker]
            cand_close = None
            cand_ked = None
            cand_tech_risk = None
            cand_bal_sheet = None
            if not cand_feat_rows.empty:
                cr = cand_feat_rows.iloc[0]
                cand_close = cr.get("close") if "close" in features.columns else None
                cand_ked = (
                    cr.get("known_at_effective_date")
                    if "known_at_effective_date" in features.columns
                    else None
                )
                cand_tech_risk = (
                    cr.get("technical_risk_score")
                    if "technical_risk_score" in features.columns
                    else None
                )
                cand_bal_sheet = (
                    cr.get("balance_sheet_strength_score")
                    if "balance_sheet_strength_score" in features.columns
                    else None
                )

            ticker_candidate_rows.append(
                {
                    "date": date_str,
                    "action_type": selected_action,
                    "ticker": cand_ticker,
                    "allowed": not ts_dict.get("rejected", False),
                    "close": cand_close,
                    "known_at_effective_date": (
                        str(cand_ked)[:10]
                        if pd.notna(cand_ked) and cand_ked is not None
                        else None
                    ),
                    "momentum_score": ts_dict.get("momentum_score"),
                    "technical_risk_score": cand_tech_risk,
                    "value_score": ts_dict.get("value_score"),
                    "quality_score": ts_dict.get("quality_score"),
                    "profitability_score": ts_dict.get("profitability_score"),
                    "balance_sheet_strength_score": cand_bal_sheet,
                    "ticker_score": ts_dict.get("final_ticker_score"),
                    "rank": ts_dict.get("rank"),
                    "reason_codes": ts_dict.get("rejection_reason", "") or "",
                }
            )

        # Size candidate rows
        for ss_dict in dec.stage_3_size_scores:
            size_candidate_rows.append(
                {
                    "date": date_str,
                    "selected_action_type": selected_action,
                    "selected_ticker": selected_ticker,
                    "size_option": ss_dict.get("size_label"),
                    "allocation_fraction": ss_dict.get("fraction"),
                    "allowed": ss_dict.get("selected", False),
                    "size_score": ss_dict.get("final_size_score"),
                    "reason_codes": _size_reason_codes(ss_dict),
                }
            )

    # --- Write output files ---
    run_paths = create_run_paths("d_iqn_dss_combined_iqn_hdp_audit_smoke_test")
    audit_dir = run_paths.audit_directory
    summary_dir = run_paths.summary_directory

    audit_df = pd.DataFrame(audit_rows)
    ticker_cand_df = (
        pd.DataFrame(ticker_candidate_rows) if ticker_candidate_rows else pd.DataFrame()
    )
    size_cand_df = (
        pd.DataFrame(size_candidate_rows) if size_candidate_rows else pd.DataFrame()
    )

    _write_csv(audit_dir / "combined_iqn_hdp_decision_audit.csv", audit_df)
    _write_csv(audit_dir / "combined_iqn_hdp_ticker_candidates.csv", ticker_cand_df)
    _write_csv(audit_dir / "combined_iqn_hdp_size_candidates.csv", size_cand_df)
    logger.info("Audit CSVs written to: %s", audit_dir)

    # --- Validation summary ---
    selected_tickers = (
        sorted(audit_df["selected_ticker"].dropna().unique().tolist())
        if not audit_df.empty
        else []
    )

    action_counts: dict = {}
    valuation_method_counts: dict = {}
    valuation_warning_counts: dict = {}
    if not audit_df.empty:
        action_counts = audit_df["selected_action_type"].value_counts().to_dict()
        if "valuation_method" in audit_df.columns:
            valuation_method_counts = (
                audit_df["valuation_method"].dropna().value_counts().to_dict()
            )
        if "valuation_warning" in audit_df.columns:
            valuation_warning_counts = (
                audit_df["valuation_warning"].dropna().value_counts().to_dict()
            )

    ked_counts: dict = {}
    if "known_at_effective_date" in hdp_df_full.columns:
        ked_counts = (
            hdp_df_full.groupby(tic_col_hdp)["known_at_effective_date"]
            .nunique()
            .to_dict()
        )

    edl_a_ready = pit_violations_total == 0 and len(audit_rows) > 0

    # --- Check HDP decision sanity ---
    sanity_notes: list[str] = []
    if not audit_df.empty:
        buy_rows = audit_df[audit_df["selected_action_type"] == "BUY"]
        if not buy_rows.empty and "ticker_score" in audit_df.columns:
            buy_scores = buy_rows["ticker_score"].dropna()
            if not buy_scores.empty:
                sanity_notes.append(
                    f"BUY selected_ticker avg ticker_score: {buy_scores.mean():.3f}"
                )

        hold_rows = audit_df[audit_df["selected_action_type"] == "HOLD"]
        sanity_notes.append(
            f"HOLD rows: {len(hold_rows)} (ticker_universe still logged in candidates)"
        )

    summary = {
        "run_id": run_paths.run_id,
        "source_iqn_run": iqn_run_id if iqn_available else None,
        "source_hdp_feature_run": hdp_run_id,
        "iqn_source_available": iqn_available,
        "action_source": action_source,
        "manual_fallback_action": manual_action_str if not iqn_available else None,
        "start_date": start_date,
        "end_date": end_date,
        "tickers": tickers,
        "dates_covered": len(dates),
        "audit_rows": len(audit_rows),
        "ticker_candidate_rows": len(ticker_candidate_rows),
        "size_candidate_rows": len(size_candidate_rows),
        "pit_violations_total": pit_violations_total,
        "selected_tickers": selected_tickers,
        "action_type_counts": action_counts,
        "valuation_method_counts": valuation_method_counts,
        "valuation_warning_counts": valuation_warning_counts,
        "unique_known_at_effective_date_per_ticker": ked_counts,
        "hdp_decision_sanity_notes": sanity_notes,
        "edl_a_readiness": "ready" if edl_a_ready else "not_ready",
        "edl_a_readiness_notes": (
            []
            if edl_a_ready
            else (["pit_violations > 0"] if pit_violations_total > 0 else [])
            + (["no audit rows produced"] if len(audit_rows) == 0 else [])
        ),
    }

    _write_json(summary_dir / "combined_iqn_hdp_audit_summary.json", summary)

    # --- Markdown summary ---
    md_lines = [
        f"# Combined IQN+HDP Audit Smoke Test — {run_paths.run_id}",
        "",
        "## Sources",
        f"- IQN source available: `{iqn_available}`",
        f"- IQN source run: `{iqn_run_id or 'N/A'}`",
        f"- Action source: `{action_source}`",
        f"- Manual fallback action: `{manual_action_str if not iqn_available else 'N/A'}`",
        f"- HDP feature source run: `{hdp_run_id}`",
        "",
        "## Configuration",
        f"- Date range: `{start_date}` → `{end_date}`",
        f"- Tickers: `{', '.join(tickers)}`",
        f"- Cash weight (demo portfolio): `{cash_weight:.0%}`",
        "",
        "## Results",
        f"- Dates covered: **{len(dates)}**",
        f"- Audit rows: **{len(audit_rows)}**",
        f"- Ticker candidate rows: **{len(ticker_candidate_rows)}**",
        f"- Size candidate rows: **{len(size_candidate_rows)}**",
        f"- PIT violations total: **{pit_violations_total}**",
        f"- Selected tickers: `{', '.join(selected_tickers) or 'none'}`",
        "",
        "## Action Type Counts",
        "",
        "| Action | Count |",
        "|--------|-------|",
    ]
    for action, count in sorted(action_counts.items()):
        md_lines.append(f"| {action} | {count} |")

    md_lines += [
        "",
        "## Valuation Method Counts",
        "",
        "| Method | Count |",
        "|--------|-------|",
    ]
    for method, count in sorted(valuation_method_counts.items()):
        md_lines.append(f"| {method} | {count} |")

    md_lines += [
        "",
        "## Unique known_at_effective_date per Ticker",
        "",
        "| Ticker | Unique effective dates |",
        "|--------|------------------------|",
    ]
    for ticker, cnt in sorted(ked_counts.items()):
        md_lines.append(f"| {ticker} | {cnt} |")

    md_lines += [
        "",
        "## HDP Decision Sanity",
        "",
    ]
    for note in sanity_notes:
        md_lines.append(f"- {note}")

    md_lines += [
        "",
        "## EDL-A Readiness",
        f"- **{summary['edl_a_readiness'].upper()}**",
    ]
    for note in summary["edl_a_readiness_notes"]:
        md_lines.append(f"- ⚠️ {note}")

    _write_md(summary_dir / "combined_iqn_hdp_audit_summary.md", "\n".join(md_lines))

    # --- Final log ---
    logger.info("=== Combined IQN+HDP Audit complete ===")
    logger.info("Run directory:        %s", run_paths.run_directory)
    logger.info("IQN source available: %s", iqn_available)
    logger.info("Action source:        %s", action_source)
    logger.info("Audit rows:           %d", len(audit_rows))
    logger.info("Ticker candidates:    %d", len(ticker_candidate_rows))
    logger.info("Size candidates:      %d", len(size_candidate_rows))
    logger.info("PIT violations:       %d", pit_violations_total)
    logger.info("Selected tickers:     %s", selected_tickers)
    logger.info("EDL-A readiness:      %s", summary["edl_a_readiness"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
