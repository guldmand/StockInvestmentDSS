"""
EDL-A Counterfactual Hindsight Oracle (v3.5)

Computes per-action forward scores for each decision row and assigns the
best risk-adjusted action as the EDL-A hindsight label.

Point-in-time rule: future return/drawdown columns are LABEL targets only.
They must never be included in the EDL input feature matrix.

Class space: HOLD=0, BUY=1, SELL=2 (REBALANCE excluded from EDL-A).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_HOLD = "HOLD"
LABEL_BUY = "BUY"
LABEL_SELL = "SELL"
LABEL_UNAVAILABLE = "UNAVAILABLE"

LABEL_ID = {LABEL_HOLD: 0, LABEL_BUY: 1, LABEL_SELL: 2}

# Reason strings
REASON_CF_ARGMAX = "counterfactual_argmax"
REASON_AMBIGUOUS = "ambiguous_margin"
REASON_NO_TICKER_HOLD = "no_ticker_hold_preserved"
REASON_NO_TICKER_NON_HOLD = "no_ticker_non_hold"
REASON_INSUFFICIENT_DATA = "insufficient_future_data"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CounterfactualConfig:
    horizon_days: int = 20
    drawdown_lambda: float = 0.5
    min_label_margin: float = 0.005
    class_space: List[str] = field(default_factory=lambda: ["HOLD", "BUY", "SELL"])


# ---------------------------------------------------------------------------
# Ticker Price Index
# ---------------------------------------------------------------------------


class TickerPriceIndex:
    """Pre-built O(1) date-lookup index over market close prices per ticker."""

    def __init__(self, market_df: pd.DataFrame):
        self._closes: Dict[str, np.ndarray] = {}
        self._date_pos: Dict[str, Dict[str, int]] = {}

        date_col = "date"
        tic_col = "tic"
        close_col = "close"

        for tic, grp in market_df.groupby(tic_col):
            grp = grp.sort_values(date_col).reset_index(drop=True)
            dates = grp[date_col].astype(str).tolist()
            closes = grp[close_col].to_numpy(dtype=float)
            self._closes[tic] = closes
            self._date_pos[tic] = {d: i for i, d in enumerate(dates)}

        self.tickers = set(self._closes.keys())
        logger.info("TickerPriceIndex built: %d tickers", len(self.tickers))

    def get_window(
        self, ticker: str, date_str: str, horizon: int
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Returns (window, available_bars) where window[0] = close[t].

        Returns (None, 0) if ticker/date not found.
        available_bars <= horizon if the series ends before t+horizon.
        """
        if ticker not in self._closes:
            return None, 0
        pos_map = self._date_pos[ticker]
        if date_str not in pos_map:
            return None, 0
        idx = pos_map[date_str]
        closes = self._closes[ticker]
        end_idx = min(idx + horizon + 1, len(closes))
        window = closes[idx:end_idx]
        return window, len(window) - 1  # available_bars = steps after t


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


@dataclass
class CounterfactualScores:
    buy_score: float
    sell_score: float
    hold_score: float
    future_return_pct: float
    future_max_drawdown_pct: float
    risk_adjusted_future_score: float
    available_bars: int
    label_available: bool
    label_reason: str


def _compute_scores(
    ticker: str,
    date_str: str,
    action_before_edl: str,
    price_index: TickerPriceIndex,
    config: CounterfactualConfig,
) -> CounterfactualScores:
    """Compute BUY / SELL / HOLD counterfactual scores for one decision row."""

    hold_score = 0.0
    nan = float("nan")

    # --- No-ticker cases ---------------------------------------------------
    if not ticker or str(ticker).strip().upper() in ("", "NONE", "NAN", "UNKNOWN"):
        if str(action_before_edl).upper() == "HOLD":
            return CounterfactualScores(
                buy_score=nan,
                sell_score=0.0,
                hold_score=0.0,
                future_return_pct=nan,
                future_max_drawdown_pct=nan,
                risk_adjusted_future_score=nan,
                available_bars=0,
                label_available=True,
                label_reason=REASON_NO_TICKER_HOLD,
            )
        return CounterfactualScores(
            buy_score=nan,
            sell_score=nan,
            hold_score=nan,
            future_return_pct=nan,
            future_max_drawdown_pct=nan,
            risk_adjusted_future_score=nan,
            available_bars=0,
            label_available=False,
            label_reason=REASON_NO_TICKER_NON_HOLD,
        )

    # --- Fetch price window ------------------------------------------------
    window, available_bars = price_index.get_window(
        ticker, date_str, config.horizon_days
    )

    if window is None or available_bars < 1:
        return CounterfactualScores(
            buy_score=nan,
            sell_score=nan,
            hold_score=nan,
            future_return_pct=nan,
            future_max_drawdown_pct=nan,
            risk_adjusted_future_score=nan,
            available_bars=available_bars or 0,
            label_available=False,
            label_reason=REASON_INSUFFICIENT_DATA,
        )

    price_t = window[0]
    if price_t <= 0:
        return CounterfactualScores(
            buy_score=nan,
            sell_score=nan,
            hold_score=nan,
            future_return_pct=nan,
            future_max_drawdown_pct=nan,
            risk_adjusted_future_score=nan,
            available_bars=available_bars,
            label_available=False,
            label_reason="zero_price_at_t",
        )

    # Use actual available horizon (may be less than config.horizon_days near series end)
    price_th = window[-1]
    future_return = price_th / price_t - 1.0
    max_drawdown = float(np.min(window)) / price_t - 1.0

    # BUY score: risk-adjusted forward return
    buy_score = future_return - config.drawdown_lambda * abs(min(0.0, max_drawdown))

    # SELL score: value of avoiding the position
    # Selling at t = receiving price_t, holding cash (return = 0)
    # SELL advantage = 0 - buy_future_return (no drawdown penalty on cash)
    sell_score = -future_return

    risk_adjusted = buy_score  # same as BUY score by definition

    label_available = True
    label_reason = REASON_CF_ARGMAX
    if available_bars < config.horizon_days:
        # Partial horizon — label is still computed but flagged
        label_reason = f"{REASON_CF_ARGMAX}_partial_{available_bars}d"

    return CounterfactualScores(
        buy_score=buy_score,
        sell_score=sell_score,
        hold_score=hold_score,
        future_return_pct=future_return * 100.0,
        future_max_drawdown_pct=max_drawdown * 100.0,
        risk_adjusted_future_score=risk_adjusted,
        available_bars=available_bars,
        label_available=label_available,
        label_reason=label_reason,
    )


def _assign_label(
    scores: CounterfactualScores, config: CounterfactualConfig
) -> Tuple[str, int, float, float, float]:
    """
    Returns (label, label_id, best_score, second_best_score, margin).
    """
    if not scores.label_available:
        return LABEL_UNAVAILABLE, -1, float("nan"), float("nan"), float("nan")

    # Special case: no-ticker HOLD
    if scores.label_reason == REASON_NO_TICKER_HOLD:
        return LABEL_HOLD, LABEL_ID[LABEL_HOLD], 0.0, 0.0, 0.0

    score_map = {
        LABEL_HOLD: scores.hold_score,
        LABEL_BUY: scores.buy_score,
        LABEL_SELL: scores.sell_score,
    }
    # Filter to available class space
    filtered = {
        k: v
        for k, v in score_map.items()
        if k in config.class_space and not np.isnan(v)
    }
    if not filtered:
        return LABEL_UNAVAILABLE, -1, float("nan"), float("nan"), float("nan")

    sorted_labels = sorted(filtered, key=lambda k: filtered[k], reverse=True)
    best_label = sorted_labels[0]
    best_score = filtered[best_label]
    second_best_score = (
        filtered[sorted_labels[1]] if len(sorted_labels) > 1 else float("nan")
    )
    margin = (
        best_score - second_best_score
        if not np.isnan(second_best_score)
        else float("nan")
    )

    if np.isnan(margin) or margin < config.min_label_margin:
        return (
            LABEL_HOLD,
            LABEL_ID[LABEL_HOLD],
            best_score,
            second_best_score,
            margin if not np.isnan(margin) else 0.0,
        )

    return best_label, LABEL_ID[best_label], best_score, second_best_score, margin


# ---------------------------------------------------------------------------
# Main labeling function
# ---------------------------------------------------------------------------


def label_combined_audit(
    combined_df: pd.DataFrame,
    price_index: TickerPriceIndex,
    config: CounterfactualConfig,
) -> pd.DataFrame:
    """
    Add counterfactual hindsight label columns to the combined audit DataFrame.

    Returns a new DataFrame with all original columns plus the edl_a_cf_* columns.
    Future-data columns are clearly prefixed edl_a_cf_ and must NOT be used as
    EDL input features.
    """
    out = combined_df.copy()

    cf_labels: List[str] = []
    cf_label_ids: List[int] = []
    cf_available: List[bool] = []
    cf_reasons: List[str] = []
    cf_horizon: List[int] = []
    cf_buy_scores: List[float] = []
    cf_sell_scores: List[float] = []
    cf_hold_scores: List[float] = []
    cf_best_scores: List[float] = []
    cf_second_best_scores: List[float] = []
    cf_margins: List[float] = []
    cf_future_returns: List[float] = []
    cf_future_drawdowns: List[float] = []
    cf_risk_adj: List[float] = []

    for _, row in combined_df.iterrows():
        date_str = str(row.get("date", "")).strip()
        ticker = str(row.get("selected_ticker", "")).strip()
        action = str(
            row.get(
                "final_recommendation_before_edl",
                row.get("hierarchical_action_type", ""),
            )
        ).strip()

        scores = _compute_scores(ticker, date_str, action, price_index, config)
        label, label_id, best_score, second_best, margin = _assign_label(scores, config)

        cf_labels.append(label)
        cf_label_ids.append(label_id)
        cf_available.append(scores.label_available)
        cf_reasons.append(scores.label_reason)
        cf_horizon.append(config.horizon_days)
        cf_buy_scores.append(scores.buy_score)
        cf_sell_scores.append(scores.sell_score)
        cf_hold_scores.append(scores.hold_score)
        cf_best_scores.append(best_score)
        cf_second_best_scores.append(second_best)
        cf_margins.append(margin)
        cf_future_returns.append(scores.future_return_pct)
        cf_future_drawdowns.append(scores.future_max_drawdown_pct)
        cf_risk_adj.append(scores.risk_adjusted_future_score)

    out["edl_a_cf_label"] = cf_labels
    out["edl_a_cf_label_id"] = cf_label_ids
    out["edl_a_cf_label_available"] = cf_available
    out["edl_a_cf_label_reason"] = cf_reasons
    out["edl_a_cf_horizon_days"] = cf_horizon
    out["edl_a_cf_buy_score"] = cf_buy_scores
    out["edl_a_cf_sell_score"] = cf_sell_scores
    out["edl_a_cf_hold_score"] = cf_hold_scores
    out["edl_a_cf_best_score"] = cf_best_scores
    out["edl_a_cf_second_best_score"] = cf_second_best_scores
    out["edl_a_cf_margin"] = cf_margins
    out["edl_a_cf_future_return_pct"] = cf_future_returns
    out["edl_a_cf_future_max_drawdown_pct"] = cf_future_drawdowns
    out["edl_a_cf_risk_adjusted_future_score"] = cf_risk_adj

    return out


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def build_summary(
    out_df: pd.DataFrame,
    config: CounterfactualConfig,
    source_run_id: str,
    market_data_file: str,
    output_csv_path: str,
    warnings: List[str],
) -> dict:
    labeled = out_df[out_df["edl_a_cf_label_available"] == True]  # noqa: E712
    unavailable = out_df[out_df["edl_a_cf_label_available"] == False]  # noqa: E712
    ambiguous = out_df[
        out_df["edl_a_cf_label_reason"].str.startswith("ambiguous", na=False)
    ]

    label_dist = labeled["edl_a_cf_label"].value_counts().to_dict()

    label_dist_by_ticker: dict = {}
    if "selected_ticker" in labeled.columns:
        for tic, grp in labeled.groupby("selected_ticker"):
            label_dist_by_ticker[str(tic)] = (
                grp["edl_a_cf_label"].value_counts().to_dict()
            )

    label_dist_by_action: dict = {}
    action_col = (
        "final_recommendation_before_edl"
        if "final_recommendation_before_edl" in labeled.columns
        else "hierarchical_action_type"
    )
    if action_col in labeled.columns:
        for act, grp in labeled.groupby(action_col):
            label_dist_by_action[str(act)] = (
                grp["edl_a_cf_label"].value_counts().to_dict()
            )

    # Confusion table: action_before_edl vs cf_label
    confusion: dict = {}
    if action_col in out_df.columns:
        ct = pd.crosstab(
            out_df[action_col].fillna("UNKNOWN"),
            out_df["edl_a_cf_label"].fillna("UNAVAILABLE"),
        )
        for act, row in ct.iterrows():
            confusion[str(act)] = row.to_dict()

    mean_return_by_label: dict = {}
    mean_drawdown_by_label: dict = {}
    for lbl in ["HOLD", "BUY", "SELL"]:
        subset = labeled[labeled["edl_a_cf_label"] == lbl]
        if len(subset) > 0:
            mean_return_by_label[lbl] = round(
                float(subset["edl_a_cf_future_return_pct"].mean()), 4
            )
            mean_drawdown_by_label[lbl] = round(
                float(subset["edl_a_cf_future_max_drawdown_pct"].mean()), 4
            )

    return {
        "source_combined_run_id": source_run_id,
        "market_data_file": market_data_file,
        "class_space": config.class_space,
        "horizon_days": config.horizon_days,
        "parameters": {
            "drawdown_lambda": config.drawdown_lambda,
            "min_label_margin": config.min_label_margin,
        },
        "total_rows": int(len(out_df)),
        "labeled_rows": int(len(labeled)),
        "unavailable_rows": int(len(unavailable)),
        "ambiguous_rows": int(len(ambiguous)),
        "label_distribution": label_dist,
        "label_distribution_by_ticker": label_dist_by_ticker,
        "label_distribution_by_action_before_edl": label_dist_by_action,
        "confusion_action_before_edl_vs_cf_label": confusion,
        "mean_future_return_pct_by_label": mean_return_by_label,
        "mean_future_max_drawdown_pct_by_label": mean_drawdown_by_label,
        "output_csv_path": output_csv_path,
        "warnings": warnings,
        "caveat": (
            "Do not use this labeled file for tuning if it is from the final evaluation period."
        ),
    }
