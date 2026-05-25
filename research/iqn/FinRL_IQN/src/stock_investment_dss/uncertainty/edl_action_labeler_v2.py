# src/stock_investment_dss/uncertainty/edl_action_labeler_v2.py
"""
EDL Action Labeler v2 for D-IQN-DSS (v3.3).

Generates training labels for the EDL action uncertainty classifier
from a combined IQN + HierarchicalDecisionPolicy audit CSV.

Supports three label modes:

EDL-A: hindsight
    Labels derived from future realized risk-adjusted return over horizon h.
    NOT available in this runner — requires a future-price oracle pass.
    Returns unavailable status with a summary warning.

EDL-B: rules
    Transparent, deterministic labels derived from IQN score features.
    Uses IQN action scores to determine the "correct" action by rule.

    Rules (applied in priority order):
    1. REBALANCE — if iqn_score_rebalance is the max score AND
                   cash_weight is between 0.10 and 0.85 (invested portfolio)
    2. SELL      — if iqn_score_sell is the max score AND
                   cash_weight < 0.85 (has something to sell)
    3. BUY       — if iqn_score_buy is the max score AND
                   cash_weight >= 0.15 (has cash available) AND
                   price_vs_ma50 is not extreme overbought (> 0.25)
    4. HOLD      — default fallback

    Margin threshold: a rule-based label is emitted only if
    iqn_action_margin >= EDL_B_MIN_MARGIN (default 0.005).
    Below threshold → HOLD (uncertain IQN signal).

EDL-C: iqn_teacher
    Labels from edl_c_teacher_label column (= HDP final action recommendation).
    Fallback: hierarchical_action_type if edl_c_teacher_label is blank/null.

References
----------
See docs/EDL_Action_Dataset_v3_3.md and
copilot-diagnostics/design/edl_uncertainty_poc/edl_v3_3_reference_repo_alignment_and_integration_plan.md
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from stock_investment_dss.uncertainty.edl_action_classes import (
    EDL_ACTION_CLASSES_4,
    action_to_idx,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EDL-B rule parameters
# ---------------------------------------------------------------------------

# Minimum IQN action margin for a non-HOLD rule label
EDL_B_MIN_MARGIN: float = 0.005

# Cash weight below which we consider the portfolio "invested" (may need SELL/REBALANCE)
EDL_B_INVESTED_THRESHOLD: float = 0.85

# Cash weight above which we consider the portfolio "cash-heavy" (ready to BUY)
EDL_B_CASH_THRESHOLD: float = 0.15

# Price vs MA50 overbought threshold — suppress BUY
EDL_B_OVERBOUGHT_THRESHOLD: float = 0.25

# EDL_B score column names
_EDL_B_SCORE_COLS = {
    "HOLD": "iqn_score_hold",
    "BUY": "iqn_score_buy",
    "SELL": "iqn_score_sell",
    "REBALANCE": "iqn_score_rebalance",
}

# ---------------------------------------------------------------------------
# Unavailable status sentinel
# ---------------------------------------------------------------------------

EDL_LABEL_UNAVAILABLE = "UNAVAILABLE"


# ---------------------------------------------------------------------------
# EDL-C labeler
# ---------------------------------------------------------------------------


def label_edl_c(df: pd.DataFrame) -> Tuple[List[Optional[str]], int, List[str]]:
    """
    Generate EDL-C (iqn_teacher) labels from combined audit DataFrame.

    Uses edl_c_teacher_label column (= HDP final action recommendation).
    Falls back to hierarchical_action_type if blank/null.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    labels : list[str | None]
        Label string per row (None = unavailable after fallback).
    n_unavailable : int
        Count of rows with no label.
    warnings : list[str]
        Non-fatal warnings.
    """
    labels: List[Optional[str]] = []
    n_unavailable = 0
    warnings: List[str] = []

    for idx, row in df.iterrows():
        label = _str_or_none(row.get("edl_c_teacher_label"))
        if label is None:
            label = _str_or_none(row.get("hierarchical_action_type"))
        if label is None:
            n_unavailable += 1
            labels.append(None)
        else:
            label_upper = label.strip().upper()
            if label_upper not in EDL_ACTION_CLASSES_4:
                # Map CHANGE_STRATEGY to HOLD (out-of-EDL-space action)
                if label_upper == "CHANGE_STRATEGY":
                    label_upper = "HOLD"
                else:
                    w = f"Row {idx}: unknown EDL-C label '{label}' → mapped to HOLD"
                    warnings.append(w)
                    label_upper = "HOLD"
            labels.append(label_upper)

    if n_unavailable > 0:
        warnings.append(
            f"EDL-C: {n_unavailable}/{len(df)} rows have no teacher label "
            "(edl_c_teacher_label and hierarchical_action_type both missing)"
        )
    logger.info(
        "EDL-C labels: %d / %d rows labelled, %d unavailable",
        len(df) - n_unavailable,
        len(df),
        n_unavailable,
    )
    return labels, n_unavailable, warnings


# ---------------------------------------------------------------------------
# EDL-B labeler
# ---------------------------------------------------------------------------


def label_edl_b(
    df: pd.DataFrame,
    min_margin: float = EDL_B_MIN_MARGIN,
    invested_threshold: float = EDL_B_INVESTED_THRESHOLD,
    cash_threshold: float = EDL_B_CASH_THRESHOLD,
    overbought_threshold: float = EDL_B_OVERBOUGHT_THRESHOLD,
) -> Tuple[List[str], int, List[str]]:
    """
    Generate EDL-B (rules) labels from combined audit DataFrame.

    Rules (applied in priority order):
    1. If iqn_action_margin < min_margin → HOLD (low-signal step)
    2. If max-score action is REBALANCE and 0.10 < cash_weight < 0.85 → REBALANCE
    3. If max-score action is SELL and cash_weight < 0.85 → SELL
    4. If max-score action is BUY and cash_weight >= 0.15 and
       price_vs_ma50 <= 0.25 (not overbought) → BUY
    5. HOLD (default)

    Parameters
    ----------
    df : pd.DataFrame
    min_margin : float
        Minimum IQN action margin for a non-HOLD rule label.
    invested_threshold : float
        Cash weight below this → portfolio is invested (may SELL/REBALANCE).
    cash_threshold : float
        Cash weight above this → portfolio has cash available (may BUY).
    overbought_threshold : float
        price_vs_ma50 above this suppresses BUY rule.

    Returns
    -------
    labels : list[str]
    n_unavailable : int (always 0 for EDL-B)
    warnings : list[str]
    """
    labels: List[str] = []
    warnings: List[str] = []
    score_cols_missing: List[str] = []

    for col in _EDL_B_SCORE_COLS.values():
        if col not in df.columns:
            score_cols_missing.append(col)

    if score_cols_missing:
        w = (
            f"EDL-B: Missing IQN score columns: {score_cols_missing}. "
            "Falling back to HOLD for all rows."
        )
        warnings.append(w)
        logger.warning(w)
        return ["HOLD"] * len(df), 0, warnings

    for _, row in df.iterrows():
        label = _apply_edl_b_rules(
            row=row,
            min_margin=min_margin,
            invested_threshold=invested_threshold,
            cash_threshold=cash_threshold,
            overbought_threshold=overbought_threshold,
        )
        labels.append(label)

    dist = pd.Series(labels).value_counts().to_dict()
    logger.info("EDL-B label distribution: %s", dist)
    return labels, 0, warnings


def _apply_edl_b_rules(
    row: pd.Series,
    min_margin: float,
    invested_threshold: float,
    cash_threshold: float,
    overbought_threshold: float,
) -> str:
    """Apply EDL-B rules to one row and return a label string."""
    margin = _float_or(row.get("iqn_action_margin"), 0.0)
    cash_weight = _float_or(row.get("cash_weight"), 0.5)

    # Low-signal: IQN action margin too small to trust any non-HOLD label
    if margin < min_margin:
        return "HOLD"

    # Get IQN score per action
    score_hold = _float_or(row.get("iqn_score_hold"), 0.0)
    score_buy = _float_or(row.get("iqn_score_buy"), 0.0)
    score_sell = _float_or(row.get("iqn_score_sell"), 0.0)
    score_rebalance = _float_or(row.get("iqn_score_rebalance"), 0.0)

    scores = {
        "HOLD": score_hold,
        "BUY": score_buy,
        "SELL": score_sell,
        "REBALANCE": score_rebalance,
    }
    best_action = max(scores, key=lambda a: scores[a])

    # Rule 1: REBALANCE — max score is REBALANCE and portfolio is partially invested
    if best_action == "REBALANCE" and (0.10 < cash_weight < invested_threshold):
        return "REBALANCE"

    # Rule 2: SELL — max score is SELL and portfolio has holdings
    if best_action == "SELL" and cash_weight < invested_threshold:
        return "SELL"

    # Rule 3: BUY — max score is BUY and portfolio has cash; not overbought
    if best_action == "BUY" and cash_weight >= cash_threshold:
        price_vs_ma50 = _float_or(row.get("price_vs_ma50"), 0.0)
        if price_vs_ma50 <= overbought_threshold:
            return "BUY"

    return "HOLD"


# ---------------------------------------------------------------------------
# EDL-A labeler (unavailable)
# ---------------------------------------------------------------------------


def label_edl_a(df: pd.DataFrame) -> Tuple[List[None], int, List[str]]:
    """
    Attempt to generate EDL-A (hindsight) labels.

    EDL-A requires future realized return columns that are not yet
    populated in the combined audit CSV. Returns None for all rows
    and emits a summary warning.

    If edl_a_hindsight_label column already contains populated labels
    (from a future-price oracle pass), those are used instead.
    """
    if "edl_a_hindsight_label" in df.columns:
        populated = df["edl_a_hindsight_label"].dropna()
        if len(populated) > 0 and not (populated == "").all():
            logger.info(
                "EDL-A: Found %d pre-populated hindsight labels", len(populated)
            )
            labels: List[Optional[str]] = []
            n_unavail = 0
            warnings: List[str] = []
            for v in df["edl_a_hindsight_label"]:
                s = _str_or_none(v)
                if s is None:
                    n_unavail += 1
                    labels.append(None)
                else:
                    labels.append(s.upper().strip())
            return labels, n_unavail, warnings

    n = len(df)
    w = (
        "EDL-A (hindsight) labels are NOT available. "
        "edl_a_hindsight_label column is empty/missing in the combined audit CSV. "
        "To generate EDL-A labels, run a future-price oracle pass that appends "
        "realized risk-adjusted returns over the horizon window to the combined CSV, "
        "then re-run this builder."
    )
    logger.warning(w)
    return [None] * n, n, [w]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_labels(
    df: pd.DataFrame,
    label_mode: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate labels for the combined audit DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Combined audit CSV loaded as DataFrame.
    label_mode : str
        One of "hindsight" (EDL-A), "rules" (EDL-B), "iqn_teacher" (EDL-C).

    Returns
    -------
    df_out : pd.DataFrame
        Input DataFrame with added columns:
        - edl_label_mode   : label mode string
        - edl_label_name   : human-readable action label (or UNAVAILABLE)
        - edl_label_id     : integer class index (or -1 for unavailable)
    warnings : list[str]
        Non-fatal warnings.

    Raises
    ------
    ValueError if label_mode is unknown.
    """
    mode = label_mode.lower().strip()
    if mode == "iqn_teacher":
        labels_raw, n_unavail, warnings = label_edl_c(df)
    elif mode == "rules":
        labels_raw, n_unavail, warnings = label_edl_b(df)
    elif mode == "hindsight":
        labels_raw, n_unavail, warnings = label_edl_a(df)
    else:
        raise ValueError(
            f"Unknown label_mode '{label_mode}'. "
            "Valid: 'hindsight' (EDL-A), 'rules' (EDL-B), 'iqn_teacher' (EDL-C)"
        )

    # Convert to name/id columns
    label_names: List[str] = []
    label_ids: List[int] = []
    for lbl in labels_raw:
        if lbl is None:
            label_names.append(EDL_LABEL_UNAVAILABLE)
            label_ids.append(-1)
        else:
            label_names.append(lbl)
            try:
                label_ids.append(action_to_idx(lbl))
            except ValueError:
                label_names[-1] = EDL_LABEL_UNAVAILABLE
                label_ids[-1] = -1

    df_out = df.copy()
    df_out["edl_label_mode"] = label_mode
    df_out["edl_label_name"] = label_names
    df_out["edl_label_id"] = label_ids

    available_count = sum(1 for lid in label_ids if lid >= 0)
    logger.info(
        "Labels generated (%s): %d available, %d unavailable",
        label_mode,
        available_count,
        n_unavail,
    )
    return df_out, warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _str_or_none(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def _float_or(v, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default
