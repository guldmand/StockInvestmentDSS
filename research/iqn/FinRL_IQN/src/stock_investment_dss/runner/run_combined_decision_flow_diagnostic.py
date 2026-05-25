# src/stock_investment_dss/runner/run_combined_decision_flow_diagnostic.py
"""
Combined Decision Flow Diagnostic runner for D-IQN-DSS.

For each decision date, traces the full IQN → action_mask → HDP → EDL/gate
decision pipeline and records *where* each action originates or is overridden.
Does NOT assume a root cause — it measures each stage objectively.

Stage sequence
--------------
0. Load IQN export CSV (date-indexed) and HDP feature CSV.
1. IQN stage   : raw action from q50 argmax + score margins.
2. Action mask : portfolio constraints (no-cash / no-holdings) may override.
3. HDP stage   : TickerSelector + SizeSelector.
4. EDL gate    : null_gate (edl_available=false) unless real EDL inference
                 artifact is found alongside the IQN export.
5. Final action: result after all stages.

hold_reason_category is assigned to each HOLD row based on the first stage
that introduced or maintained HOLD.

Constraints
-----------
- Does NOT retrain IQN.
- Does NOT modify EDL training or model architecture.
- Does NOT make live FMP calls.
- Does NOT invent EDL effects — edl_available=false if no artifact found.

Usage
-----
From repository root with PYTHONPATH=src:

    $env:PYTHONPATH = "src"
    $env:STOCK_INVESTMENT_DSS_DIAG_IQN_DECISION_CSV  = "<path/to/iqn_decision_export.csv>"
    $env:STOCK_INVESTMENT_DSS_DIAG_HDP_FEATURE_CSV   = "<path/to/hdp_joined_features.csv>"
    python -m stock_investment_dss.runner.run_combined_decision_flow_diagnostic

Environment variables
---------------------
STOCK_INVESTMENT_DSS_DIAG_IQN_DECISION_CSV   : path to date-indexed IQN decision CSV
STOCK_INVESTMENT_DSS_DIAG_HDP_FEATURE_CSV    : path to HDP joined features CSV
STOCK_INVESTMENT_DSS_DIAG_START_DATE         : ISO date (default: 2024-01-01)
STOCK_INVESTMENT_DSS_DIAG_END_DATE           : ISO date (default: 2024-02-01)
STOCK_INVESTMENT_DSS_DIAG_TICKERS            : comma-separated (default: AAPL,MSFT,NVDA,AMZN,GOOGL)
STOCK_INVESTMENT_DSS_DIAG_CASH_WEIGHT        : demo portfolio cash weight (default: 0.80)

Outputs
-------
outputs/runs/<timestamp>_d_iqn_dss_combined_decision_flow_diagnostic/
  audit/combined_decision_flow_by_date.csv
  audit/hold_reason_breakdown.csv
  audit/action_transition_table.csv
  summary/combined_decision_flow_diagnostic_summary.json
  summary/combined_decision_flow_diagnostic_summary.md
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOLD_REASON_IQN = "IQN_SELECTED_HOLD"
HOLD_REASON_MASK = "ACTION_MASK_FORCED_HOLD"
HOLD_REASON_NO_TICKER = "HDP_NO_VALID_TICKER"
HOLD_REASON_SIZE = "SIZE_SELECTOR_REJECTED"
HOLD_REASON_EDL = "EDL_FORCED_HOLD"
HOLD_REASON_CONSTRAINT = "PORTFOLIO_CONSTRAINT"
HOLD_REASON_FEATURE = "FEATURE_MISSING"
HOLD_REASON_IQN_MISSING = "IQN_ROW_MISSING"
HOLD_REASON_UNKNOWN = "UNKNOWN"

_ACTION_COLS = ["hold", "buy", "sell", "rebalance"]

_IQN_EXPORT_PATTERN = re.compile(r"iqn_decision_export|iqn_learning_curve")
_HDP_EXPORT_PATTERN = re.compile(r"fmp_hdp_feature_smoke_test")


# ---------------------------------------------------------------------------
# Utilities
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


def _find_latest_run(runs_dir: Path, pattern: re.Pattern) -> Optional[Path]:
    candidates = [
        d for d in runs_dir.iterdir() if d.is_dir() and pattern.search(d.name)
    ]
    return sorted(candidates)[-1] if candidates else None


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


def _resolve_iqn_csv(direct_csv: str, runs_dir: Path) -> Optional[Path]:
    """Find IQN decision export CSV with a 'date' column."""
    if direct_csv:
        p = Path(direct_csv)
        if p.exists():
            return p
        logger.warning("Direct IQN CSV path not found: %s", p)

    # Auto-discover latest IQN export run
    run_dir = _find_latest_run(runs_dir, _IQN_EXPORT_PATTERN)
    if run_dir is None:
        # Try parent-level runs/
        parent_runs = runs_dir.parent.parent.parent / "outputs" / "runs"
        if parent_runs.exists():
            run_dir = _find_latest_run(parent_runs, _IQN_EXPORT_PATTERN)
    if run_dir is None:
        return None

    for csv_path in sorted(run_dir.rglob("*.csv")):
        try:
            sample = pd.read_csv(csv_path, nrows=1)
            if "date" in sample.columns and "iqn_selected_action" in sample.columns:
                logger.info("Auto-discovered IQN export: %s", csv_path)
                return csv_path
        except Exception:
            continue
    return None


def _resolve_hdp_csv(direct_csv: str, runs_dir: Path, data_dir: Path) -> Optional[Path]:
    """Find HDP joined features CSV."""
    if direct_csv:
        p = Path(direct_csv)
        if p.exists():
            return p
        logger.warning("Direct HDP CSV path not found: %s", p)

    candidate = data_dir / "hdp_joined_features.csv"
    if candidate.exists():
        return candidate

    parent_runs = runs_dir.parent.parent.parent / "outputs" / "runs"
    for search in [runs_dir, parent_runs]:
        if not search.exists():
            continue
        run_dir = _find_latest_run(search, _HDP_EXPORT_PATTERN)
        if run_dir:
            c = run_dir / "data" / "hdp_joined_features.csv"
            if c.exists():
                logger.info("Auto-discovered HDP features: %s", c)
                return c
    return None


# ---------------------------------------------------------------------------
# IQN row helpers
# ---------------------------------------------------------------------------


def _safe_float(row: pd.Series, *keys) -> Optional[float]:
    for k in keys:
        v = row.get(k)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return None


def _detect_score_mode(df: pd.DataFrame) -> str:
    """Detect which score mode was used from column names."""
    if "q50_hold" in df.columns and "cvar_hold" in df.columns:
        # Both present — check if action_score == q50 or q50 - cvar_penalty
        sample = df.dropna(subset=["q50_hold", "action_score_hold"]).head(20)
        if sample.empty:
            return "unknown"
        diff = (sample["action_score_hold"] - sample["q50_hold"]).abs().mean()
        if diff < 0.001:
            return "q50"
        # Check q50 - 0.5*|cvar|
        q50c = sample["q50_hold"] - 0.5 * sample["cvar_hold"].abs()
        diff2 = (sample["action_score_hold"] - q50c).abs().mean()
        if diff2 < 0.001:
            return "q50_minus_cvar_penalty"
        return "unknown"
    if "action_score_hold" in df.columns:
        return "action_score_only"
    return "unknown"


def _get_iqn_scores(row: pd.Series) -> dict[str, Optional[float]]:
    """Extract q50 and cvar per action from an IQN row."""
    scores = {}
    for act in _ACTION_COLS:
        scores[f"q50_{act}"] = _safe_float(row, f"q50_{act}")
        scores[f"cvar_{act}"] = _safe_float(row, f"cvar_{act}")
        scores[f"iqn_score_{act}"] = _safe_float(
            row, f"action_score_{act}", f"score_{act}", f"q_{act}"
        )
    return scores


def _iqn_margin(scores: dict) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Compute:
    - iqn_margin_hold_vs_best_non_hold: q50_hold − max(non-hold q50)
    - iqn_best_non_hold_action
    - iqn_best_action_by_score (argmax q50 across all 4 actions)
    """
    non_hold = ["buy", "sell", "rebalance"]
    q50_hold = scores.get("q50_hold")
    q50_non = {a: scores.get(f"q50_{a}") for a in non_hold}
    valid_non = {a: v for a, v in q50_non.items() if v is not None}
    if not valid_non:
        return None, None, None
    best_non_hold = max(valid_non, key=lambda a: valid_non[a])
    best_non_val = valid_non[best_non_hold]
    margin = (q50_hold - best_non_val) if q50_hold is not None else None

    # Best action by q50 across all
    all_q50 = {a: scores.get(f"q50_{a}") for a in _ACTION_COLS}
    valid_all = {a: v for a, v in all_q50.items() if v is not None}
    best_all = max(valid_all, key=lambda a: valid_all[a]) if valid_all else None

    return margin, best_non_hold, best_all


# ---------------------------------------------------------------------------
# Portfolio + HDP helpers
# ---------------------------------------------------------------------------


def _build_demo_portfolio(cash_weight: float, tickers: list[str]) -> "PortfolioState":
    from stock_investment_dss.decision.hierarchical_decision_policy import (
        PortfolioState,
    )

    total = 1_000_000.0
    cash = cash_weight * total
    equity = (1.0 - cash_weight) * total
    n = len(tickers)
    per = equity / n if n > 0 else 0.0
    return PortfolioState(
        total_value=total,
        cash=cash,
        holdings={t: 0 for t in tickers},
        holding_values={t: per for t in tickers},
    )


def _action_mask(
    iqn_action: str,
    portfolio: "PortfolioState",
    risk_profile: "InvestorRiskProfile",
) -> dict[str, bool]:
    """
    Compute per-action availability mask based on portfolio constraints.
    Returns dict: {hold: True, buy: bool, sell: bool, rebalance: bool}
    """
    buy_ok = portfolio.cash_weight >= risk_profile.min_cash_weight
    sell_ok = any(v > 0.001 for v in portfolio.holding_values.values())
    return {
        "hold": True,
        "buy": buy_ok,
        "sell": sell_ok,
        "rebalance": True,
    }


# ---------------------------------------------------------------------------
# hold_reason_category logic
# ---------------------------------------------------------------------------


def _classify_hold_reason(
    iqn_action: Optional[str],
    action_mask: dict[str, bool],
    hdp_applicable: bool,
    hdp_ticker: Optional[str],
    size_rejected: bool,
    edl_available: bool,
    edl_forced_hold: bool,
    feature_missing: bool,
) -> str:
    if iqn_action is None:
        return HOLD_REASON_IQN_MISSING

    ua = iqn_action.upper()

    if ua == "HOLD":
        return HOLD_REASON_IQN

    # IQN wanted a trade
    if ua in ("BUY", "SELL", "REBALANCE"):
        mask_key = ua.lower()
        if not action_mask.get(mask_key, True):
            return HOLD_REASON_MASK

        if feature_missing:
            return HOLD_REASON_FEATURE

        if hdp_applicable and hdp_ticker is None:
            return HOLD_REASON_NO_TICKER

        if hdp_applicable and hdp_ticker is not None and size_rejected:
            return HOLD_REASON_SIZE

        if edl_available and edl_forced_hold:
            return HOLD_REASON_EDL

        return HOLD_REASON_UNKNOWN

    return HOLD_REASON_UNKNOWN


# ---------------------------------------------------------------------------
# Per-date trace
# ---------------------------------------------------------------------------


def _trace_date(
    date_str: str,
    iqn_row: Optional[pd.Series],
    hdp_features: pd.DataFrame,
    portfolio: "PortfolioState",
    policy: "HierarchicalDecisionPolicy",
    risk_profile: "InvestorRiskProfile",
    iqn_source_run_id: str,
    score_mode: str,
) -> dict:
    from stock_investment_dss.decision.hierarchical_decision_policy import (
        DSSDecisionAction,
        action_to_label,
    )
    from stock_investment_dss.uncertainty.edl_gate import EDLGate

    _STR_TO_ACTION = {
        "HOLD": DSSDecisionAction.HOLD,
        "BUY": DSSDecisionAction.BUY,
        "SELL": DSSDecisionAction.SELL,
        "REBALANCE": DSSDecisionAction.REBALANCE,
        "CHANGE_STRATEGY": DSSDecisionAction.CHANGE_STRATEGY,
    }

    def label_to_action(label: str) -> DSSDecisionAction:
        return _STR_TO_ACTION.get(label.upper(), DSSDecisionAction.HOLD)

    # --- Step 1: IQN scores ---
    scores: dict = {}
    raw_iqn_action: Optional[str] = None
    iqn_action_source = "missing"

    if iqn_row is not None:
        scores = _get_iqn_scores(iqn_row)
        raw_iqn_action = _safe_str(
            iqn_row, "iqn_selected_action", "selected_action_type"
        )
        iqn_action_source = "iqn_export"
    else:
        iqn_action_source = "no_iqn_row"

    margin, best_non_hold, best_all = _iqn_margin(scores)

    # --- Step 2: Action mask ---
    mask = _action_mask(raw_iqn_action or "HOLD", portfolio, risk_profile)
    mask_hold = mask["hold"]
    mask_buy = mask["buy"]
    mask_sell = mask["sell"]
    mask_rebalance = mask["rebalance"]

    # --- Step 3a: Determine action before HDP ---
    # Mask can override IQN's choice
    selected_before_hdp: str
    if raw_iqn_action is None:
        selected_before_hdp = "HOLD"
    else:
        ua = raw_iqn_action.upper()
        mask_key = ua.lower()
        if ua != "HOLD" and not mask.get(mask_key, True):
            selected_before_hdp = "HOLD"
        else:
            selected_before_hdp = ua

    # --- Step 3b: Feature availability check ---
    feature_missing = hdp_features.empty
    required_cols = ["momentum_score", "value_score", "quality_score"]
    if not feature_missing:
        missing = [c for c in required_cols if c not in hdp_features.columns]
        feature_missing = len(missing) > 0

    # --- Step 3c: HDP decision ---
    hdp_applicable = selected_before_hdp in ("BUY", "SELL", "REBALANCE")
    hdp_ticker: Optional[str] = None
    hdp_size: Optional[str] = None
    hdp_top_score: Optional[float] = None
    hdp_top_candidate: Optional[str] = None
    hdp_reason_codes = "n/a"
    size_rejected = False
    final_action = selected_before_hdp
    final_ticker: Optional[str] = None
    final_size: Optional[str] = None
    final_fraction: float = 0.0
    final_reason = "iqn_passthrough"

    if hdp_applicable and not feature_missing:
        try:
            action_enum = label_to_action(selected_before_hdp)
            dec = policy.decide(
                action_type=action_enum,
                features=hdp_features,
                portfolio=portfolio,
                decision_date=date_str,
            )
            final_action = dec.selected_action_type
            final_ticker = dec.selected_ticker
            final_size = dec.selected_size
            final_fraction = dec.selected_fraction or 0.0
            final_reason = "hdp_pipeline"

            # Detect ticker rejection
            if selected_before_hdp in ("BUY", "SELL") and final_ticker is None:
                hdp_ticker = None
                hdp_reason_codes = "no_valid_ticker"
            else:
                hdp_ticker = final_ticker
                # Extract top candidate info from stage_2 scores
                stage2 = dec.stage_2_ticker_scores
                if stage2:
                    valid = [r for r in stage2 if not r.get("rejected", False)]
                    if valid:
                        top = valid[0]
                        hdp_top_candidate = top.get("ticker")
                        hdp_top_score = top.get("final_ticker_score")
                    # Collect reason codes from rejections
                    rejected = [r for r in stage2 if r.get("rejected", False)]
                    codes = [
                        r.get("rejection_reason", "")
                        for r in rejected
                        if r.get("rejection_reason")
                    ]
                    hdp_reason_codes = "|".join(codes) if codes else "none"

            # Detect size rejection
            if hdp_ticker is not None and (final_fraction == 0.0 or final_size is None):
                size_rejected = True
                if final_action in ("BUY", "SELL"):
                    final_action = "HOLD"
                    final_reason = "size_rejected_to_hold"

            hdp_size = final_size
        except Exception as exc:
            logger.warning("HDP error on %s: %s", date_str, exc)
            final_reason = f"hdp_error:{exc}"

    elif hdp_applicable and feature_missing:
        final_action = "HOLD"
        final_reason = "feature_missing"
        hdp_reason_codes = "feature_missing"

    # --- Step 4: EDL (null gate — edl_available always false here) ---
    edl_available = False
    edl_predicted_action: Optional[str] = None
    edl_confidence: Optional[float] = None
    edl_vacuity: Optional[float] = None
    edl_gate_decision: Optional[str] = None
    edl_forced_hold = False

    # null_gate — pass through unchanged
    gate = EDLGate().null_gate(
        selected_action=final_action,
        selected_size=final_size or "",
        original_fraction=final_fraction,
    )
    # final_action is unchanged by null gate
    final_action_after_edl = gate.final_action_after_edl_gate
    final_size_after_edl = gate.final_size_after_edl_gate
    final_fraction_after_edl = gate.final_fraction_after_edl_gate

    # --- Step 5: Assign hold_reason_category ---
    hold_reason: Optional[str] = None
    if final_action_after_edl == "HOLD":
        hold_reason = _classify_hold_reason(
            iqn_action=raw_iqn_action,
            action_mask=mask,
            hdp_applicable=hdp_applicable,
            hdp_ticker=hdp_ticker,
            size_rejected=size_rejected,
            edl_available=edl_available,
            edl_forced_hold=edl_forced_hold,
            feature_missing=feature_missing,
        )

    return {
        # IQN
        "date": date_str,
        "raw_iqn_selected_action": raw_iqn_action,
        "iqn_action_source": iqn_action_source,
        "iqn_source_run_id": iqn_source_run_id,
        "iqn_q50_hold": scores.get("q50_hold"),
        "iqn_q50_buy": scores.get("q50_buy"),
        "iqn_q50_sell": scores.get("q50_sell"),
        "iqn_q50_rebalance": scores.get("q50_rebalance"),
        "iqn_cvar_hold": scores.get("cvar_hold"),
        "iqn_cvar_buy": scores.get("cvar_buy"),
        "iqn_cvar_sell": scores.get("cvar_sell"),
        "iqn_cvar_rebalance": scores.get("cvar_rebalance"),
        "iqn_score_hold": scores.get("iqn_score_hold"),
        "iqn_score_buy": scores.get("iqn_score_buy"),
        "iqn_score_sell": scores.get("iqn_score_sell"),
        "iqn_score_rebalance": scores.get("iqn_score_rebalance"),
        "iqn_margin_hold_vs_best_non_hold": (
            round(margin, 6) if margin is not None else None
        ),
        "iqn_best_non_hold_action": best_non_hold,
        "iqn_best_action_by_score": best_all,
        "iqn_score_mode_detected": score_mode,
        # Action mask
        "action_mask_hold": mask_hold,
        "action_mask_buy": mask_buy,
        "action_mask_sell": mask_sell,
        "action_mask_rebalance": mask_rebalance,
        # Pre-HDP
        "selected_action_before_hdp": selected_before_hdp,
        # HDP
        "hdp_applicable": hdp_applicable,
        "hdp_selected_ticker": hdp_ticker,
        "hdp_selected_size": hdp_size,
        "hdp_top_ticker_score": hdp_top_score,
        "hdp_top_candidate": hdp_top_candidate,
        "hdp_reason_codes": hdp_reason_codes,
        # EDL
        "edl_available": edl_available,
        "edl_predicted_action": edl_predicted_action,
        "edl_confidence": edl_confidence,
        "edl_vacuity": edl_vacuity,
        "edl_gate_decision": edl_gate_decision,
        # Final
        "final_action": final_action_after_edl,
        "final_selected_ticker": final_ticker,
        "final_selected_size": final_size_after_edl,
        "final_fraction": round(final_fraction_after_edl, 6),
        "final_reason": final_reason,
        "hold_reason_category": hold_reason,
    }


def _safe_str(row: pd.Series, *keys) -> Optional[str]:
    for k in keys:
        v = row.get(k)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            return str(v)
    return None


# ---------------------------------------------------------------------------
# Transition / aggregation helpers
# ---------------------------------------------------------------------------


def _build_transition_table(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of raw_iqn_selected_action × final_action."""
    df2 = df.copy()
    df2["raw_iqn"] = df2["raw_iqn_selected_action"].fillna("MISSING")
    df2["final"] = df2["final_action"].fillna("MISSING")
    pivot = df2.groupby(["raw_iqn", "final"]).size().reset_index(name="count")
    return pivot


def _build_hold_reason_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    hold_rows = df[df["final_action"] == "HOLD"].copy()
    if hold_rows.empty:
        return pd.DataFrame(
            columns=["hold_reason_category", "count", "pct_of_all_holds"]
        )
    counts = hold_rows["hold_reason_category"].value_counts().reset_index()
    counts.columns = ["hold_reason_category", "count"]
    counts["pct_of_all_holds"] = (counts["count"] / counts["count"].sum() * 100).round(
        1
    )
    return counts


def _transition_counts(df: pd.DataFrame) -> dict:
    raw_iqn = df["raw_iqn_selected_action"].str.upper().fillna("MISSING")
    final = df["final_action"].str.upper().fillna("MISSING")
    iqn_buy_sell_to_hold = int(
        ((raw_iqn.isin(["BUY", "SELL", "REBALANCE"])) & (final == "HOLD")).sum()
    )
    iqn_hold_to_trade = int(((raw_iqn == "HOLD") & (final != "HOLD")).sum())
    hdp_changed = int((df["selected_action_before_hdp"] != df["final_action"]).sum())
    edl_changed = 0  # edl_available=False throughout; no EDL changes
    return {
        "iqn_buy_sell_became_hold_count": iqn_buy_sell_to_hold,
        "iqn_hold_became_trade_count": iqn_hold_to_trade,
        "hdp_changed_action_count": hdp_changed,
        "edl_changed_action_count": edl_changed,
    }


def _score_diagnostics(df: pd.DataFrame) -> dict:
    margin_col = "iqn_margin_hold_vs_best_non_hold"
    margins = df[margin_col].dropna()
    if margins.empty:
        return {"iqn_margin_stats": "no_data"}
    best_non_hold_dist = df["iqn_best_non_hold_action"].value_counts().to_dict()
    return {
        "iqn_margin_hold_vs_best_non_hold_mean": round(float(margins.mean()), 6),
        "iqn_margin_hold_vs_best_non_hold_std": round(float(margins.std()), 6),
        "iqn_margin_hold_vs_best_non_hold_min": round(float(margins.min()), 6),
        "iqn_margin_hold_vs_best_non_hold_max": round(float(margins.max()), 6),
        "iqn_margin_positive_count": int((margins > 0).sum()),
        "iqn_margin_negative_count": int((margins < 0).sum()),
        "iqn_best_non_hold_action_distribution": best_non_hold_dist,
        "iqn_score_mode_detected": (
            df["iqn_score_mode_detected"].iloc[0] if not df.empty else "unknown"
        ),
    }


# ---------------------------------------------------------------------------
# Markdown summary builder
# ---------------------------------------------------------------------------


def _build_summary_md(s: dict) -> str:
    def _pct(a, b):
        return f"{a / b * 100:.1f}%" if b > 0 else "N/A"

    n_dates = s["n_decision_dates"]
    n_iqn = s["n_iqn_rows"]
    n_hdp = s["n_hdp_feature_rows"]
    n_final_hold = s["final_action_counts"].get("HOLD", 0)
    n_total = s["n_decision_dates"]
    transitions = s["transition_counts"]
    hold_reasons = s.get("hold_reason_breakdown", {})
    margins = s.get("score_diagnostics", {})

    lines = [
        "# Combined Decision Flow Diagnostic — Summary",
        "",
        f"Run: `{s['run_id']}`",
        f"IQN source: `{s['iqn_source_path']}`",
        f"HDP source: `{s['hdp_source_path']}`",
        f"Date range: {s['start_date']} → {s['end_date']}",
        f"Tickers: {s['ticker_universe']}",
        "",
        "## Data Coverage",
        f"- Decision dates: {n_dates}",
        f"- IQN rows loaded: {n_iqn}",
        f"- HDP feature rows loaded: {n_hdp}",
        f"- Missing IQN dates: {s['n_missing_iqn_dates']}",
        f"- Missing HDP dates: {s['n_missing_hdp_dates']}",
        f"- IQN ticker universe: {s['iqn_ticker_universe']}",
        f"- HDP ticker universe: {s['hdp_ticker_universe']}",
        f"- Ticker universe mismatch: {s['ticker_universe_mismatch']}",
        "",
        "## Action Counts",
        f"- Raw IQN action counts: {s['raw_iqn_action_counts']}",
        f"- Final action counts:   {s['final_action_counts']}",
        "",
        "## Transition Summary",
        f"- IQN BUY/SELL/REBALANCE → final HOLD: {transitions['iqn_buy_sell_became_hold_count']} dates",
        f"- IQN HOLD → final trade (BUY/SELL):   {transitions['iqn_hold_became_trade_count']} dates",
        f"- HDP/size stage changed action:        {transitions['hdp_changed_action_count']} dates",
        f"- EDL gate changed action:              {transitions['edl_changed_action_count']} dates (edl_available=false in this run)",
        "",
        "## HOLD Breakdown",
        f"- Total HOLD in final output: {n_final_hold} / {n_total} ({_pct(n_final_hold, n_total)})",
        "- hold_reason_category breakdown:",
    ]
    if hold_reasons:
        for cat, cnt in hold_reasons.items():
            lines.append(f"  - `{cat}`: {cnt}")
    else:
        lines.append("  - (no HOLD rows)")
    lines += [
        "",
        "## IQN Score Diagnostics",
        f"- Score mode detected: `{margins.get('iqn_score_mode_detected', 'unknown')}`",
        f"- HOLD-vs-best margin mean: {margins.get('iqn_margin_hold_vs_best_non_hold_mean', 'N/A')}",
        f"- HOLD-vs-best margin std:  {margins.get('iqn_margin_hold_vs_best_non_hold_std', 'N/A')}",
        f"- Dates where HOLD margin > 0 (HOLD genuinely wins): {margins.get('iqn_margin_positive_count', 'N/A')}",
        f"- Dates where HOLD margin ≤ 0 (non-HOLD has equal/higher q50): {margins.get('iqn_margin_negative_count', 'N/A')}",
        f"- Best non-HOLD action distribution: {margins.get('iqn_best_non_hold_action_distribution', {})}",
        "",
        "## EDL Status",
        "- EDL gate: **not integrated** (`edl_available=false`). No EDL inference artifact was found.",
        "- `EDL_FORCED_HOLD` count = 0 (by definition).",
        "- To integrate EDL: run EDL inference and point this runner to the output.",
        "",
        "## Cautious Interpretation",
        f"- In this run (IQN source: `{s['iqn_source_path'].split(chr(47))[-1] if s['iqn_source_path'] else 'unknown'}`),",
        f"  HOLD is introduced at the following stages based on measured transition counts:",
    ]
    for cat, cnt in hold_reasons.items():
        lines.append(f"  - {cat}: {cnt}")
    lines += [
        "",
        "- These results are specific to the IQN checkpoint and seed used.",
        "  Results may differ for other seeds or training steps.",
        "",
        "## 4-Way Ablation Plan (not yet implemented)",
        "| Variant | IQN | HDP | EDL |",
        "|---------|-----|-----|-----|",
        "| A | ✅ | ❌ | ❌ |",
        "| B | ✅ | ✅ | ❌ |",
        "| C | ✅ | ❌ | ✅ |",
        "| D | ✅ | ✅ | ✅ |",
        "",
        "All variants must use the same IQN export, date range, ticker universe, and PIT rules.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    _setup_logging()

    iqn_direct_csv = _env("STOCK_INVESTMENT_DSS_DIAG_IQN_DECISION_CSV")
    hdp_direct_csv = _env("STOCK_INVESTMENT_DSS_DIAG_HDP_FEATURE_CSV")
    start_date = _env("STOCK_INVESTMENT_DSS_DIAG_START_DATE", "2024-01-01")
    end_date = _env("STOCK_INVESTMENT_DSS_DIAG_END_DATE", "2024-02-01")
    tickers_str = _env("STOCK_INVESTMENT_DSS_DIAG_TICKERS", "AAPL,MSFT,NVDA,AMZN,GOOGL")
    cash_weight = float(_env("STOCK_INVESTMENT_DSS_DIAG_CASH_WEIGHT", "0.80"))

    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]

    # Resolve project root
    try:
        from stock_investment_dss.utils.paths import get_project_root

        project_root = get_project_root()
    except Exception:
        project_root = Path(__file__).resolve().parents[3]

    runs_dir = project_root / "outputs" / "runs"
    data_dir = project_root / "data"

    # Output dir
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_name = f"{timestamp}_d_iqn_dss_combined_decision_flow_diagnostic"
    out_dir = runs_dir / run_name
    logger.info("Output dir: %s", out_dir)

    # --- Resolve sources ---
    iqn_csv = _resolve_iqn_csv(iqn_direct_csv, runs_dir)
    hdp_csv = _resolve_hdp_csv(hdp_direct_csv, runs_dir, data_dir)

    if iqn_csv is None:
        logger.error(
            "No IQN decision export CSV found. Set STOCK_INVESTMENT_DSS_DIAG_IQN_DECISION_CSV "
            "or run run_iqn_decision_export_smoke_test.py first."
        )
        return 1

    if hdp_csv is None:
        logger.warning(
            "No HDP feature CSV found. HDP stages will show feature_missing=True. "
            "Set STOCK_INVESTMENT_DSS_DIAG_HDP_FEATURE_CSV or run fmp_hdp_feature_smoke_test.py first."
        )

    logger.info("IQN source: %s", iqn_csv)
    logger.info("HDP source: %s", hdp_csv)

    # --- Load IQN ---
    iqn_df = pd.read_csv(str(iqn_csv), low_memory=False)
    iqn_df["date"] = pd.to_datetime(iqn_df["date"]).dt.strftime("%Y-%m-%d")
    iqn_source_run_id = str(iqn_csv.parent.parent.name) if iqn_csv else "unknown"

    score_mode = _detect_score_mode(iqn_df)
    if score_mode == "unknown":
        logger.warning(
            "Could not determine IQN score mode from column names. "
            "Check whether action_score_hold == q50_hold or q50_hold - 0.5*|cvar_hold|."
        )

    # Filter IQN to date range
    iqn_start = pd.Timestamp(start_date)
    iqn_end = pd.Timestamp(end_date)
    iqn_df_filtered = iqn_df[
        (pd.to_datetime(iqn_df["date"]) >= iqn_start)
        & (pd.to_datetime(iqn_df["date"]) <= iqn_end)
    ].copy()
    iqn_by_date = {row["date"]: row for _, row in iqn_df_filtered.iterrows()}

    # IQN ticker universe from full dataset
    iqn_ticker_universe: list[str] = []
    for col in ("tic", "ticker"):
        if col in iqn_df.columns:
            iqn_ticker_universe = sorted(iqn_df[col].dropna().unique().tolist())
            break

    # --- Load HDP ---
    hdp_features_full: Optional[pd.DataFrame] = None
    hdp_ticker_universe: list[str] = []
    if hdp_csv is not None:
        hdp_features_full = pd.read_csv(str(hdp_csv), low_memory=False)
        tic_col = (
            "ticker"
            if "ticker" in hdp_features_full.columns
            else ("tic" if "tic" in hdp_features_full.columns else None)
        )
        if tic_col:
            hdp_ticker_universe = sorted(
                hdp_features_full[tic_col].dropna().unique().tolist()
            )

    # --- Build decision dates ---
    dates = (
        pd.date_range(start=start_date, end=end_date, freq="B")
        .strftime("%Y-%m-%d")
        .tolist()
    )
    # Remove end_date if it's exclusive (match combined runner behavior)
    dates = [d for d in dates if d < end_date or d == start_date]
    if not dates:
        dates = [start_date]

    # --- Setup policy ---
    from stock_investment_dss.decision.hierarchical_decision_policy import (
        HierarchicalDecisionPolicy,
    )
    from stock_investment_dss.decision.investor_risk_profile import InvestorRiskProfile

    risk_profile = InvestorRiskProfile.balanced()
    policy = HierarchicalDecisionPolicy(risk_profile=risk_profile)
    portfolio = _build_demo_portfolio(cash_weight, tickers)

    # Data coverage diagnostics
    missing_iqn_dates = [d for d in dates if d not in iqn_by_date]
    missing_hdp_dates: list[str] = []
    if hdp_features_full is not None and "date" in hdp_features_full.columns:
        hdp_dates = set(
            pd.to_datetime(hdp_features_full["date"]).dt.strftime("%Y-%m-%d").tolist()
        )
        missing_hdp_dates = [d for d in dates if d not in hdp_dates]

    ticker_mismatch = list(set(iqn_ticker_universe) ^ set(hdp_ticker_universe))

    # --- Per-date trace ---
    logger.info("Tracing %d decision dates...", len(dates))
    rows = []
    for d in dates:
        iqn_row = iqn_by_date.get(d)
        # Get HDP features for this date (or latest available — PIT)
        hdp_features_date: pd.DataFrame = pd.DataFrame()
        if hdp_features_full is not None:
            if "date" in hdp_features_full.columns:
                tic_col = "ticker" if "ticker" in hdp_features_full.columns else "tic"
                sub = hdp_features_full[
                    pd.to_datetime(hdp_features_full["date"]) <= pd.Timestamp(d)
                ]
                if not sub.empty and tic_col in sub.columns:
                    # Latest row per ticker up to decision_date
                    hdp_features_date = (
                        sub.sort_values("date").groupby(tic_col, as_index=False).last()
                    )
                    # Filter to configured tickers
                    hdp_features_date = hdp_features_date[
                        hdp_features_date[tic_col].isin(tickers)
                    ].reset_index(drop=True)
            else:
                hdp_features_date = hdp_features_full[
                    hdp_features_full.get(
                        "ticker", hdp_features_full.get("tic", pd.Series())
                    ).isin(tickers)
                ].copy()

        row = _trace_date(
            date_str=d,
            iqn_row=iqn_row,
            hdp_features=hdp_features_date,
            portfolio=portfolio,
            policy=policy,
            risk_profile=risk_profile,
            iqn_source_run_id=iqn_source_run_id,
            score_mode=score_mode,
        )
        rows.append(row)

    trace_df = pd.DataFrame(rows)

    # --- Aggregations ---
    raw_counts = trace_df["raw_iqn_selected_action"].value_counts().to_dict()
    final_counts = trace_df["final_action"].value_counts().to_dict()
    transitions = _transition_counts(trace_df)
    hold_breakdown_df = _build_hold_reason_breakdown(trace_df)
    hold_reasons_dict = dict(
        zip(hold_breakdown_df["hold_reason_category"], hold_breakdown_df["count"])
    )
    transition_df = _build_transition_table(trace_df)
    score_diag = _score_diagnostics(trace_df)

    # --- Summary dict ---
    summary = {
        "run_id": run_name,
        "iqn_source_path": str(iqn_csv),
        "hdp_source_path": str(hdp_csv) if hdp_csv else "not_found",
        "start_date": start_date,
        "end_date": end_date,
        "ticker_universe": tickers,
        "cash_weight": cash_weight,
        "score_mode_detected": score_mode,
        "edl_available": False,
        "n_decision_dates": len(dates),
        "n_iqn_rows": len(iqn_df_filtered),
        "n_hdp_feature_rows": (
            len(hdp_features_full) if hdp_features_full is not None else 0
        ),
        "n_missing_iqn_dates": len(missing_iqn_dates),
        "n_missing_hdp_dates": len(missing_hdp_dates),
        "missing_iqn_dates": missing_iqn_dates,
        "iqn_ticker_universe": iqn_ticker_universe,
        "hdp_ticker_universe": hdp_ticker_universe,
        "ticker_universe_mismatch": ticker_mismatch,
        "raw_iqn_action_counts": raw_counts,
        "final_action_counts": final_counts,
        "transition_counts": transitions,
        "hold_reason_breakdown": hold_reasons_dict,
        "score_diagnostics": score_diag,
    }

    # --- Write outputs ---
    _write_csv(out_dir / "audit" / "combined_decision_flow_by_date.csv", trace_df)
    logger.info("Wrote combined_decision_flow_by_date.csv (%d rows)", len(trace_df))

    _write_csv(out_dir / "audit" / "hold_reason_breakdown.csv", hold_breakdown_df)
    _write_csv(out_dir / "audit" / "action_transition_table.csv", transition_df)

    _write_json(
        out_dir / "summary" / "combined_decision_flow_diagnostic_summary.json", summary
    )
    _write_md(
        out_dir / "summary" / "combined_decision_flow_diagnostic_summary.md",
        _build_summary_md(summary),
    )

    logger.info("=== Diagnostic complete ===")
    logger.info("Raw IQN action counts:   %s", raw_counts)
    logger.info("Final action counts:     %s", final_counts)
    logger.info(
        "IQN BUY/SELL -> HOLD:    %d", transitions["iqn_buy_sell_became_hold_count"]
    )
    logger.info(
        "IQN HOLD -> trade:       %d", transitions["iqn_hold_became_trade_count"]
    )
    logger.info("HDP changed action:      %d", transitions["hdp_changed_action_count"])
    logger.info("HOLD reason breakdown:   %s", hold_reasons_dict)
    logger.info(
        "IQN margin mean:         %s",
        score_diag.get("iqn_margin_hold_vs_best_non_hold_mean"),
    )
    logger.info("Output: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
