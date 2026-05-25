# src/stock_investment_dss/decision/ticker_selector.py
"""
Rule-based ticker selector for the D-IQN-DSS hierarchical policy PoC (v3.0).

Stage 2 of the hierarchical decision policy:
  Given an action type (BUY / SELL / HOLD), select the most appropriate
  ticker from the universe using transparent, auditable scoring.

BUY scoring formula:
    ticker_score = 0.25 * value_score
                 + 0.25 * quality_score
                 + 0.20 * profitability_score
                 + 0.20 * momentum_score
                 + 0.10 * risk_fit_score

SELL scoring:
    Rank existing holdings by weakness (lowest momentum, below MA200,
    high drawdown, poor quality, overconcentration).

HOLD classification (4 sub-types):
    HOLD_CASH_ONLY         — portfolio is entirely cash
    HOLD_WHILE_INVESTED    — holding equity positions intentionally
    HOLD_NO_CANDIDATE      — no ticker passed the risk/filter checks
    HOLD_IQN_SELECTED      — IQN action type was HOLD
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from stock_investment_dss.decision.investor_risk_profile import InvestorRiskProfile

logger = logging.getLogger(__name__)

# BUY score weights
_BUY_WEIGHTS = {
    "value_score": 0.25,
    "quality_score": 0.25,
    "profitability_score": 0.20,
    "momentum_score": 0.20,
    "risk_fit_score": 0.10,
}

# RSI threshold — reject overbought candidates
_RSI_OVERBOUGHT_THRESHOLD = 70.0

# Minimum momentum score for BUY candidates (soft floor)
_MIN_MOMENTUM_FOR_BUY = -0.30


@dataclass
class TickerScoreRow:
    ticker: str
    value_score: float = 0.0
    quality_score: float = 0.0
    profitability_score: float = 0.0
    valuation_score: float = 0.0
    momentum_score: float = 0.0
    risk_fit_score: float = 0.0
    strategy_fit_score: float = 0.0
    final_ticker_score: float = 0.0
    rank: int = 0
    rejected: bool = False
    rejection_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "value_score": self.value_score,
            "quality_score": self.quality_score,
            "profitability_score": self.profitability_score,
            "valuation_score": self.valuation_score,
            "momentum_score": self.momentum_score,
            "risk_fit_score": self.risk_fit_score,
            "strategy_fit_score": self.strategy_fit_score,
            "final_ticker_score": self.final_ticker_score,
            "rank": self.rank,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
        }


class TickerSelector:
    """
    Selects the best ticker for a given action type.

    Parameters
    ----------
    risk_profile : InvestorRiskProfile
        Defines position limits and risk tolerance.
    strategy_id : str
        Strategy label used in audit logs.
    allow_partial_features : bool
        If True, missing fundamental/technical scores default to 0.5.
    """

    def __init__(
        self,
        risk_profile: Optional[InvestorRiskProfile] = None,
        strategy_id: str = "balanced_v1",
        allow_partial_features: bool = True,
    ) -> None:
        self.risk_profile = risk_profile or InvestorRiskProfile.balanced()
        self.strategy_id = strategy_id
        self.allow_partial_features = allow_partial_features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_buy_ticker(
        self,
        features: pd.DataFrame,
        portfolio_weights: dict[str, float],
        cash_weight: float,
        decision_date: str,
        bear_market_penalty: float = 0.0,
    ) -> tuple[Optional[str], list[TickerScoreRow]]:
        """
        Select the best ticker to BUY.

        Parameters
        ----------
        features : pd.DataFrame
            One row per ticker with technical + fundamental scores.
            Required columns: tic (or ticker), momentum_score.
            Optional: value_score, quality_score, profitability_score,
                      risk_fit_score, rsi_30, price_vs_ma200,
                      drawdown_from_recent_high.
        portfolio_weights : dict
            Current weight per ticker (0–1).
        cash_weight : float
            Current cash fraction.
        decision_date : str
            ISO date string for audit.
        bear_market_penalty : float
            Penalty subtracted from all ticker scores (0–1). Used by
            the risk validator to dampen scores in weak-trend regimes.

        Returns
        -------
        selected_ticker : str or None
        score_rows : list of TickerScoreRow (all candidates, including rejected)
        """
        if cash_weight < self.risk_profile.min_cash_weight:
            logger.info(
                "TickerSelector: BUY blocked — insufficient cash (%.2f < %.2f)",
                cash_weight,
                self.risk_profile.min_cash_weight,
            )
            return None, []

        rows = self._score_buy_candidates(
            features, portfolio_weights, bear_market_penalty
        )

        valid = [r for r in rows if not r.rejected]
        if not valid:
            logger.info(
                "TickerSelector: no BUY candidates passed filters on %s", decision_date
            )
            return None, rows

        selected = valid[0]
        logger.debug(
            "TickerSelector: BUY selected %s (score=%.3f)",
            selected.ticker,
            selected.final_ticker_score,
        )
        return selected.ticker, rows

    def select_sell_ticker(
        self,
        features: pd.DataFrame,
        portfolio_weights: dict[str, float],
        decision_date: str,
    ) -> tuple[Optional[str], list[TickerScoreRow]]:
        """
        Select the best ticker to SELL from existing holdings.

        Returns the holding with the worst (lowest) sell_score.
        """
        holdings = {t: w for t, w in portfolio_weights.items() if w > 0.001}
        if not holdings:
            logger.info(
                "TickerSelector: SELL blocked — no holdings on %s", decision_date
            )
            return None, []

        rows = self._score_sell_candidates(features, holdings)
        if not rows:
            return None, []

        selected = rows[0]
        logger.debug(
            "TickerSelector: SELL selected %s (sell_score=%.3f)",
            selected.ticker,
            selected.final_ticker_score,
        )
        return selected.ticker, rows

    def classify_hold(
        self,
        portfolio_weights: dict[str, float],
        cash_weight: float,
        hold_reason: str = "IQN_SELECTED",
    ) -> str:
        """
        Classify the HOLD action into one of 4 audit sub-types.

        Parameters
        ----------
        hold_reason : str
            "IQN_SELECTED" | "NO_CANDIDATE" | other (maps to CASH_ONLY or INVESTED)
        """
        holdings_weight = sum(w for t, w in portfolio_weights.items() if w > 0.001)

        if hold_reason == "IQN_SELECTED":
            return "HOLD_IQN_SELECTED"

        if hold_reason == "NO_CANDIDATE":
            return "HOLD_NO_CANDIDATE"

        if holdings_weight < 0.02:
            return "HOLD_CASH_ONLY"

        return "HOLD_WHILE_INVESTED"

    # ------------------------------------------------------------------
    # Internal BUY scoring
    # ------------------------------------------------------------------

    def _score_buy_candidates(
        self,
        features: pd.DataFrame,
        portfolio_weights: dict[str, float],
        bear_market_penalty: float,
    ) -> list[TickerScoreRow]:
        tic_col = "tic" if "tic" in features.columns else "ticker"
        rows: list[TickerScoreRow] = []

        for _, feat in features.iterrows():
            ticker = feat.get(tic_col, feat.get("ticker", "UNKNOWN"))
            row = self._build_buy_score_row(
                ticker, feat, portfolio_weights, bear_market_penalty
            )
            rows.append(row)

        # Sort valid candidates by descending score, then append rejected
        valid = sorted(
            [r for r in rows if not r.rejected], key=lambda r: -r.final_ticker_score
        )
        rejected = [r for r in rows if r.rejected]
        for i, r in enumerate(valid):
            r.rank = i + 1
        for i, r in enumerate(rejected):
            r.rank = len(valid) + i + 1

        return valid + rejected

    def _build_buy_score_row(
        self,
        ticker: str,
        feat: pd.Series,
        portfolio_weights: dict[str, float],
        bear_market_penalty: float,
    ) -> TickerScoreRow:
        default = 0.5 if self.allow_partial_features else 0.0

        def _get(col: str) -> float:
            val = feat.get(col, default)
            if pd.isna(val):
                return default
            return float(val)

        row = TickerScoreRow(
            ticker=ticker,
            value_score=_get("value_score"),
            quality_score=_get("quality_score"),
            profitability_score=_get("profitability_score"),
            valuation_score=_get("valuation_score"),
            momentum_score=_get("momentum_score"),
            risk_fit_score=_get("risk_fit_score"),
            strategy_fit_score=_get("strategy_fit_score"),
        )

        # Compute weighted BUY score
        raw_score = (
            _BUY_WEIGHTS["value_score"] * row.value_score
            + _BUY_WEIGHTS["quality_score"] * row.quality_score
            + _BUY_WEIGHTS["profitability_score"] * row.profitability_score
            + _BUY_WEIGHTS["momentum_score"] * row.momentum_score
            + _BUY_WEIGHTS["risk_fit_score"] * row.risk_fit_score
        )
        row.final_ticker_score = max(0.0, raw_score - bear_market_penalty)

        # --- Rejection filters ---
        current_weight = portfolio_weights.get(ticker, 0.0)
        if current_weight >= self.risk_profile.max_position_weight:
            row.rejected = True
            row.rejection_reason = f"already at max_position_weight ({current_weight:.2%} >= {self.risk_profile.max_position_weight:.2%})"
            return row

        rsi = feat.get("rsi_30", None)
        if (
            rsi is not None
            and not pd.isna(rsi)
            and float(rsi) > _RSI_OVERBOUGHT_THRESHOLD
        ):
            row.rejected = True
            row.rejection_reason = (
                f"RSI overbought ({float(rsi):.1f} > {_RSI_OVERBOUGHT_THRESHOLD})"
            )
            return row

        if row.momentum_score < _MIN_MOMENTUM_FOR_BUY:
            row.rejected = True
            row.rejection_reason = f"momentum too weak ({row.momentum_score:.3f} < {_MIN_MOMENTUM_FOR_BUY})"
            return row

        return row

    # ------------------------------------------------------------------
    # Internal SELL scoring
    # ------------------------------------------------------------------

    def _score_sell_candidates(
        self,
        features: pd.DataFrame,
        holdings: dict[str, float],
    ) -> list[TickerScoreRow]:
        """
        Score holdings for SELL — rank by weakness (lowest score = sell first).
        """
        tic_col = "tic" if "tic" in features.columns else "ticker"
        rows: list[TickerScoreRow] = []

        holding_features = features[features[tic_col].isin(holdings.keys())]

        for _, feat in holding_features.iterrows():
            ticker = feat.get(tic_col, feat.get("ticker", "UNKNOWN"))
            row = self._build_sell_score_row(ticker, feat, holdings)
            rows.append(row)

        # Any holding not in features → add with worst score
        featured_tickers = {r.ticker for r in rows}
        for t in holdings:
            if t not in featured_tickers:
                rows.append(TickerScoreRow(ticker=t, final_ticker_score=0.0))

        # Rank: lowest score = sell first (ascending)
        rows = sorted(rows, key=lambda r: r.final_ticker_score)
        for i, r in enumerate(rows):
            r.rank = i + 1

        return rows

    def _build_sell_score_row(
        self,
        ticker: str,
        feat: pd.Series,
        holdings: dict[str, float],
    ) -> TickerScoreRow:
        default = 0.5 if self.allow_partial_features else 0.0

        def _get(col: str) -> float:
            val = feat.get(col, default)
            if pd.isna(val):
                return default
            return float(val)

        row = TickerScoreRow(
            ticker=ticker,
            value_score=_get("value_score"),
            quality_score=_get("quality_score"),
            profitability_score=_get("profitability_score"),
            valuation_score=_get("valuation_score"),
            momentum_score=_get("momentum_score"),
            risk_fit_score=_get("risk_fit_score"),
            strategy_fit_score=_get("strategy_fit_score"),
        )

        # SELL score = quality composite (high → keep, low → sell)
        sell_keep_score = (
            0.30 * row.quality_score
            + 0.25 * row.profitability_score
            + 0.25 * (row.momentum_score + 1.0) / 2.0  # normalise [-1,1] → [0,1]
            + 0.20 * row.risk_fit_score
        )

        # Overconcentration penalty: weight > max → deduct from keep score
        current_weight = holdings.get(ticker, 0.0)
        if current_weight > self.risk_profile.max_position_weight:
            overshoot = (
                current_weight - self.risk_profile.max_position_weight
            ) / self.risk_profile.max_position_weight
            sell_keep_score = max(0.0, sell_keep_score - 0.3 * overshoot)

        # Below MA200 → weaker keep score
        price_vs_ma200 = feat.get("price_vs_ma200", None)
        if (
            price_vs_ma200 is not None
            and not pd.isna(price_vs_ma200)
            and float(price_vs_ma200) < 0
        ):
            sell_keep_score = max(0.0, sell_keep_score - 0.10)

        row.final_ticker_score = sell_keep_score
        return row
