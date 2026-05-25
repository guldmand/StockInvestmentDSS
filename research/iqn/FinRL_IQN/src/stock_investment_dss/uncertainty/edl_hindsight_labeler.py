"""
edl_hindsight_labeler.py  (EDL v3.4)

Point-in-time safe hindsight labeler for EDL-A supervised learning.

For each combined IQN+HierarchicalDecisionPolicy decision row at date t with
a selected_ticker, computes the realized return and maximum drawdown over the
next h trading days using frozen historical market data, then assigns a
BUY / HOLD / SELL label based on configurable thresholds.

Crucially:
  - X_t (features) uses only information visible at or before t
  - y_t (hindsight label) uses close[t..t+h], which is future information
  - Future prices are used ONLY as the supervised target, never as features

Label logic
-----------
    future_return_h      = close[t+h] / close[t] - 1
    future_max_drawdown  = min(close[t:t+h] / close[t] - 1)   over window
    risk_adjusted_score  = future_return_h
                         - lambda_drawdown * abs(min(0, future_max_drawdown))

    BUY  if future_return_h >= buy_threshold
             AND future_max_drawdown >= max_drawdown_threshold
    SELL if future_return_h <= sell_threshold
             OR  future_max_drawdown <  max_drawdown_threshold
    HOLD otherwise

Rows with no selected_ticker:
    HOLD  if final_recommendation_before_edl is HOLD
    UNAVAILABLE otherwise (with warning)

Rows where t+h data does not exist (end of data):
    UNAVAILABLE — label left blank, NOT faked

REBALANCE is excluded from hindsight labeling in this version (K=3).
REBALANCE remains handled by the HDP/risk layer.

References
----------
Sensoy et al. (NeurIPS 2018) — EDL framework
D-IQN-DSS design plan: copilot-diagnostics/design/edl_uncertainty_poc/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

LABEL_HOLD = "HOLD"
LABEL_BUY = "BUY"
LABEL_SELL = "SELL"
LABEL_REBALANCE = "REBALANCE"
LABEL_UNAVAILABLE = ""  # blank → excluded from EDL-A training dataset

LABEL_ID: Dict[str, int] = {
    LABEL_HOLD: 0,
    LABEL_BUY: 1,
    LABEL_SELL: 2,
    LABEL_REBALANCE: 3,
    LABEL_UNAVAILABLE: -1,
}

REASON_TICKER_HOLD = "no_ticker_hold_action"
REASON_TICKER_UNAVAILABLE = "no_ticker_non_hold_action"
REASON_NO_FUTURE_DATA = "insufficient_future_data"
REASON_PRICE_LOOKUP_ERROR = "price_lookup_error"
REASON_BUY_THRESHOLD = "buy_threshold_met"
REASON_SELL_RETURN = "sell_threshold_return"
REASON_SELL_DRAWDOWN = "sell_threshold_drawdown"
REASON_HOLD_DEFAULT = "hold_default"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HindsightLabelConfig:
    """Parameters controlling hindsight label construction."""

    horizon_days: int = 20
    buy_threshold: float = 0.03
    sell_threshold: float = -0.03
    max_drawdown_threshold: float = -0.08
    drawdown_lambda: float = 0.5
    include_rebalance: bool = False
    min_future_bars: int = (
        10  # minimum bars needed to assign a label (< horizon = skip)
    )


# ---------------------------------------------------------------------------
# Price index
# ---------------------------------------------------------------------------


@dataclass
class TickerPriceIndex:
    """
    Fast per-ticker date-indexed close price lookup.

    Built once from the market data DataFrame and reused for all rows.

    Attributes
    ----------
    closes : dict[ticker -> list[float]]  in chronological order
    dates  : dict[ticker -> list[str]]    matching date strings (YYYY-MM-DD)
    date_pos: dict[ticker -> dict[date_str -> int]]  reverse index for O(1) lookup
    """

    closes: Dict[str, List[float]] = field(default_factory=dict)
    dates: Dict[str, List[str]] = field(default_factory=dict)
    date_pos: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @classmethod
    def from_dataframe(cls, df: "pd.DataFrame") -> "TickerPriceIndex":
        """
        Build the index from a long-form market data DataFrame.

        Parameters
        ----------
        df : pd.DataFrame with columns ['date', 'tic', 'close']
            date values must be strings in YYYY-MM-DD format or datetime-like.
        """
        import pandas as pd  # type: ignore

        idx = cls()
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values(["tic", "date"])

        for tic, grp in df.groupby("tic"):
            dates_list = grp["date"].tolist()
            closes_list = grp["close"].astype(float).tolist()
            idx.dates[tic] = dates_list
            idx.closes[tic] = closes_list
            idx.date_pos[tic] = {d: i for i, d in enumerate(dates_list)}

        logger.info(
            "TickerPriceIndex built: %d tickers, %d total rows",
            len(idx.closes),
            sum(len(v) for v in idx.closes.values()),
        )
        return idx

    def get_close_at(self, ticker: str, date_str: str) -> Optional[float]:
        """Return close price for ticker on date_str, or None if not found."""
        pos_map = self.date_pos.get(ticker)
        if pos_map is None:
            return None
        idx = pos_map.get(date_str)
        if idx is None:
            return None
        return self.closes[ticker][idx]

    def get_window(
        self, ticker: str, date_str: str, horizon: int
    ) -> Tuple[Optional[List[float]], int]:
        """
        Return close prices from date_str through date_str + horizon steps.

        Returns
        -------
        (window, available_bars)
            window         : list of close prices starting at date_str (inclusive),
                             length min(horizon+1, available)
                             or None if date_str not found for ticker
            available_bars : number of bars after date_str (excluding date_str itself)
        """
        pos_map = self.date_pos.get(ticker)
        if pos_map is None:
            return None, 0
        start_idx = pos_map.get(date_str)
        if start_idx is None:
            return None, 0
        closes = self.closes[ticker]
        end_idx = min(start_idx + horizon + 1, len(closes))
        window = closes[start_idx:end_idx]
        available_bars = len(window) - 1  # bars AFTER date_str
        return window, available_bars


# ---------------------------------------------------------------------------
# Core label logic
# ---------------------------------------------------------------------------


@dataclass
class HindsightLabelResult:
    """Result of hindsight label computation for one decision row."""

    label: str  # HOLD / BUY / SELL / REBALANCE or "" (unavailable)
    label_id: int
    label_available: bool
    label_reason: str
    future_return_pct: Optional[float]
    future_max_drawdown_pct: Optional[float]
    risk_adjusted_future_score: Optional[float]


def compute_hindsight_label(
    date_str: str,
    ticker: Optional[str],
    action_before_edl: str,
    price_index: TickerPriceIndex,
    cfg: HindsightLabelConfig,
) -> HindsightLabelResult:
    """
    Compute a single hindsight label for one decision row.

    Parameters
    ----------
    date_str         : Decision date in YYYY-MM-DD format.
    ticker           : Selected ticker, or None / "" for no-ticker rows.
    action_before_edl: The final recommendation before EDL (e.g., "HOLD").
    price_index      : Pre-built TickerPriceIndex.
    cfg              : HindsightLabelConfig.

    Returns
    -------
    HindsightLabelResult
    """
    # --- No ticker case ---
    if not ticker or str(ticker).strip().upper() in ("", "NONE", "NAN", "UNKNOWN"):
        action_upper = str(action_before_edl).strip().upper()
        if action_upper == "HOLD":
            return HindsightLabelResult(
                label=LABEL_HOLD,
                label_id=LABEL_ID[LABEL_HOLD],
                label_available=True,
                label_reason=REASON_TICKER_HOLD,
                future_return_pct=None,
                future_max_drawdown_pct=None,
                risk_adjusted_future_score=None,
            )
        else:
            logger.warning(
                "Row %s: no ticker but action=%s — label unavailable",
                date_str,
                action_before_edl,
            )
            return HindsightLabelResult(
                label=LABEL_UNAVAILABLE,
                label_id=-1,
                label_available=False,
                label_reason=REASON_TICKER_UNAVAILABLE,
                future_return_pct=None,
                future_max_drawdown_pct=None,
                risk_adjusted_future_score=None,
            )

    # --- Price lookup ---
    window, available_bars = price_index.get_window(ticker, date_str, cfg.horizon_days)

    if window is None:
        logger.warning(
            "Row %s/%s: no price data found — label unavailable",
            date_str,
            ticker,
        )
        return HindsightLabelResult(
            label=LABEL_UNAVAILABLE,
            label_id=-1,
            label_available=False,
            label_reason=REASON_PRICE_LOOKUP_ERROR,
            future_return_pct=None,
            future_max_drawdown_pct=None,
            risk_adjusted_future_score=None,
        )

    if available_bars < cfg.min_future_bars:
        logger.debug(
            "Row %s/%s: only %d future bars (need %d) — label unavailable",
            date_str,
            ticker,
            available_bars,
            cfg.min_future_bars,
        )
        return HindsightLabelResult(
            label=LABEL_UNAVAILABLE,
            label_id=-1,
            label_available=False,
            label_reason=REASON_NO_FUTURE_DATA,
            future_return_pct=None,
            future_max_drawdown_pct=None,
            risk_adjusted_future_score=None,
        )

    # --- Compute metrics ---
    close_t = window[0]
    if close_t <= 0:
        return HindsightLabelResult(
            label=LABEL_UNAVAILABLE,
            label_id=-1,
            label_available=False,
            label_reason=REASON_PRICE_LOOKUP_ERROR,
            future_return_pct=None,
            future_max_drawdown_pct=None,
            risk_adjusted_future_score=None,
        )

    # Use the last available bar (up to horizon)
    close_th = window[-1]
    future_return = close_th / close_t - 1.0

    # Max drawdown = worst intra-window return relative to t (all bars including t)
    future_max_drawdown = min(c / close_t - 1.0 for c in window)

    # Risk-adjusted score penalises downward path
    drawdown_penalty = cfg.drawdown_lambda * abs(min(0.0, future_max_drawdown))
    risk_adjusted_score = future_return - drawdown_penalty

    # --- Label assignment ---
    if (
        future_return >= cfg.buy_threshold
        and future_max_drawdown >= cfg.max_drawdown_threshold
    ):
        label = LABEL_BUY
        reason = REASON_BUY_THRESHOLD
    elif future_return <= cfg.sell_threshold:
        label = LABEL_SELL
        reason = REASON_SELL_RETURN
    elif future_max_drawdown < cfg.max_drawdown_threshold:
        label = LABEL_SELL
        reason = REASON_SELL_DRAWDOWN
    else:
        label = LABEL_HOLD
        reason = REASON_HOLD_DEFAULT

    return HindsightLabelResult(
        label=label,
        label_id=LABEL_ID[label],
        label_available=True,
        label_reason=reason,
        future_return_pct=round(future_return * 100, 4),
        future_max_drawdown_pct=round(future_max_drawdown * 100, 4),
        risk_adjusted_future_score=round(risk_adjusted_score, 6),
    )


# ---------------------------------------------------------------------------
# Batch labeler
# ---------------------------------------------------------------------------


def label_combined_audit(
    rows: List[dict],
    price_index: TickerPriceIndex,
    cfg: HindsightLabelConfig,
) -> List[dict]:
    """
    Apply hindsight labels to a list of combined audit row dicts.

    Parameters
    ----------
    rows        : List of dicts from combined audit CSV (all original columns preserved).
    price_index : Pre-built TickerPriceIndex.
    cfg         : HindsightLabelConfig.

    Returns
    -------
    List of dicts with hindsight label columns appended.
    """
    output = []
    unavailable_count = 0
    warnings_list = []

    for row in rows:
        date_str = str(row.get("date", "")).strip()
        ticker = str(row.get("selected_ticker", "")).strip()
        if ticker.lower() in ("nan", "none", ""):
            ticker = ""
        action = str(row.get("final_recommendation_before_edl", "HOLD")).strip()

        result = compute_hindsight_label(
            date_str=date_str,
            ticker=ticker or None,
            action_before_edl=action,
            price_index=price_index,
            cfg=cfg,
        )

        if not result.label_available:
            unavailable_count += 1
            if result.label_reason not in (REASON_TICKER_HOLD, REASON_NO_FUTURE_DATA):
                warnings_list.append(f"{date_str}/{ticker}: {result.label_reason}")

        enriched = dict(row)
        enriched["edl_a_hindsight_label"] = result.label
        enriched["edl_a_hindsight_label_id"] = (
            result.label_id if result.label_available else ""
        )
        enriched["edl_a_future_return_horizon_days"] = cfg.horizon_days
        enriched["edl_a_future_return_pct"] = (
            result.future_return_pct if result.future_return_pct is not None else ""
        )
        enriched["edl_a_future_max_drawdown_pct"] = (
            result.future_max_drawdown_pct
            if result.future_max_drawdown_pct is not None
            else ""
        )
        enriched["edl_a_risk_adjusted_future_score"] = (
            result.risk_adjusted_future_score
            if result.risk_adjusted_future_score is not None
            else ""
        )
        enriched["edl_a_label_available"] = result.label_available
        enriched["edl_a_label_reason"] = result.label_reason

        output.append(enriched)

    labeled_count = len(rows) - unavailable_count
    logger.info(
        "Labeling complete: %d rows total, %d labeled, %d unavailable",
        len(rows),
        labeled_count,
        unavailable_count,
    )
    if warnings_list:
        logger.warning(
            "%d unusual unavailable rows:\n  %s",
            len(warnings_list),
            "\n  ".join(warnings_list[:10]),
        )

    return output
