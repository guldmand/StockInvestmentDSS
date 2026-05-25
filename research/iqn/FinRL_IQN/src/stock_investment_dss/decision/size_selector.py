# src/stock_investment_dss/decision/size_selector.py
"""
Rule-based size selector for the D-IQN-DSS hierarchical policy PoC (v3.0).

Stage 3 of the hierarchical decision policy:
  Given a selected action type and ticker, determine the appropriate
  trade size using discrete buckets adjusted for risk.

Discrete size buckets:
    BUY_25  — allocate 25% of available cash
    BUY_50  — allocate 50% of available cash
    BUY_75  — allocate 75% of available cash
    BUY_100 — allocate up to 100% of available cash
    SELL_25 — sell 25% of position
    SELL_50 — sell 50% of position
    SELL_75 — sell 75% of position
    SELL_100 — sell entire position

The selector starts from a base bucket and applies downward adjustments
based on risk factors:
  - Low cash buffer → reduce
  - High concentration → reduce
  - High volatility → reduce
  - Deep drawdown → reduce
  - Low score confidence → reduce
  - Poor liquidity proxy → reduce
  - High transaction cost estimate → reduce
  - Max position limit → reduce
  - Bear-market / trend guard → reduce
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from stock_investment_dss.decision.investor_risk_profile import InvestorRiskProfile

logger = logging.getLogger(__name__)

# Discrete bucket levels for BUY (fraction of available cash to deploy)
BUY_BUCKETS: list[tuple[str, float]] = [
    ("BUY_25", 0.25),
    ("BUY_50", 0.50),
    ("BUY_75", 0.75),
    ("BUY_100", 1.00),
]

# Discrete bucket levels for SELL (fraction of position to liquidate)
SELL_BUCKETS: list[tuple[str, float]] = [
    ("SELL_25", 0.25),
    ("SELL_50", 0.50),
    ("SELL_75", 0.75),
    ("SELL_100", 1.00),
]


@dataclass
class SizeScoreRow:
    """Audit row for the size selection stage."""

    size_label: str
    fraction: float
    base_score: float = 1.0
    cash_buffer_penalty: float = 0.0
    concentration_penalty: float = 0.0
    volatility_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    confidence_penalty: float = 0.0
    trend_guard_penalty: float = 0.0
    final_size_score: float = 1.0
    selected: bool = False

    def to_dict(self) -> dict:
        return {
            "size_label": self.size_label,
            "fraction": self.fraction,
            "base_score": self.base_score,
            "cash_buffer_penalty": self.cash_buffer_penalty,
            "concentration_penalty": self.concentration_penalty,
            "volatility_penalty": self.volatility_penalty,
            "drawdown_penalty": self.drawdown_penalty,
            "confidence_penalty": self.confidence_penalty,
            "trend_guard_penalty": self.trend_guard_penalty,
            "final_size_score": self.final_size_score,
            "selected": self.selected,
        }


@dataclass
class SizeResult:
    selected_size: str
    selected_fraction: float
    risk_adjusted_allocation_fraction: float
    score_rows: list[SizeScoreRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SizeSelector:
    """
    Selects the appropriate trade size bucket given risk context.

    Starting from a base bucket (default BUY_50 / SELL_50), risk factors
    reduce the size downward. The result is the highest bucket whose
    adjusted score exceeds the minimum threshold.

    Parameters
    ----------
    risk_profile : InvestorRiskProfile
    base_buy_bucket : str
        Default starting bucket for BUY decisions.
    base_sell_bucket : str
        Default starting bucket for SELL decisions.
    min_score_threshold : float
        Minimum composite score to use a given bucket (0–1).
    """

    def __init__(
        self,
        risk_profile: Optional[InvestorRiskProfile] = None,
        base_buy_bucket: str = "BUY_50",
        base_sell_bucket: str = "SELL_50",
        min_score_threshold: float = 0.25,
    ) -> None:
        self.risk_profile = risk_profile or InvestorRiskProfile.balanced()
        self.base_buy_bucket = base_buy_bucket
        self.base_sell_bucket = base_sell_bucket
        self.min_score_threshold = min_score_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_buy_size(
        self,
        ticker: str,
        ticker_score: float,
        cash_weight: float,
        current_ticker_weight: float,
        volatility_score: float,
        drawdown_from_high: float,
        portfolio_value: float,
        bear_market_signal: bool = False,
        defensive_strategy: bool = False,
    ) -> SizeResult:
        """
        Select the BUY size bucket for *ticker*.

        Parameters
        ----------
        ticker_score : float [0, 1]
            Composite score from TickerSelector.
        cash_weight : float [0, 1]
            Current cash fraction of portfolio.
        current_ticker_weight : float [0, 1]
            Current weight of *ticker* in portfolio.
        volatility_score : float [0, 1]
            From TechnicalFeatureBuilder (annualised vol / 1.0).
        drawdown_from_high : float [≤ 0]
            From TechnicalFeatureBuilder (fractional drawdown, negative).
        portfolio_value : float
            Total portfolio value in currency units.
        bear_market_signal : bool
            If True, apply additional trend guard penalty.
        defensive_strategy : bool
            If True, cap at BUY_25.
        """
        rows = self._score_buy_buckets(
            ticker_score=ticker_score,
            cash_weight=cash_weight,
            current_ticker_weight=current_ticker_weight,
            volatility_score=volatility_score,
            drawdown_from_high=drawdown_from_high,
            bear_market_signal=bear_market_signal,
            defensive_strategy=defensive_strategy,
        )

        warnings: list[str] = []
        selected_row = self._pick_highest_viable(rows, warnings)

        # Compute effective allocation fraction (fraction of total portfolio value)
        selected_fraction = selected_row.fraction
        available_cash = cash_weight * portfolio_value
        effective_allocation = min(
            selected_fraction * available_cash / max(portfolio_value, 1.0),
            self.risk_profile.max_position_weight - current_ticker_weight,
        )
        effective_allocation = max(0.0, effective_allocation)

        selected_row.selected = True
        return SizeResult(
            selected_size=selected_row.size_label,
            selected_fraction=selected_fraction,
            risk_adjusted_allocation_fraction=effective_allocation,
            score_rows=rows,
            warnings=warnings,
        )

    def select_sell_size(
        self,
        ticker: str,
        ticker_keep_score: float,
        current_ticker_weight: float,
        volatility_score: float,
        drawdown_from_high: float,
        portfolio_value: float,
        bear_market_signal: bool = False,
    ) -> SizeResult:
        """
        Select the SELL size bucket for *ticker*.

        Lower keep_score → larger sell fraction.
        """
        rows = self._score_sell_buckets(
            ticker_keep_score=ticker_keep_score,
            current_ticker_weight=current_ticker_weight,
            volatility_score=volatility_score,
            drawdown_from_high=drawdown_from_high,
            bear_market_signal=bear_market_signal,
        )

        warnings: list[str] = []
        selected_row = self._pick_highest_viable(rows, warnings)

        position_value = current_ticker_weight * portfolio_value
        effective_allocation = (
            selected_row.fraction * position_value / max(portfolio_value, 1.0)
        )

        selected_row.selected = True
        return SizeResult(
            selected_size=selected_row.size_label,
            selected_fraction=selected_row.fraction,
            risk_adjusted_allocation_fraction=effective_allocation,
            score_rows=rows,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Internal: BUY bucket scoring
    # ------------------------------------------------------------------

    def _score_buy_buckets(
        self,
        ticker_score: float,
        cash_weight: float,
        current_ticker_weight: float,
        volatility_score: float,
        drawdown_from_high: float,
        bear_market_signal: bool,
        defensive_strategy: bool,
    ) -> list[SizeScoreRow]:
        rows = []
        max_bucket = "BUY_25" if defensive_strategy else self.base_buy_bucket

        for label, fraction in BUY_BUCKETS:
            # Cap at defensive limit
            if defensive_strategy and label in ("BUY_50", "BUY_75", "BUY_100"):
                row = SizeScoreRow(
                    size_label=label,
                    fraction=fraction,
                    base_score=0.0,
                    final_size_score=0.0,
                )
                rows.append(row)
                continue

            base = ticker_score  # higher ticker confidence → larger base

            # Cash buffer penalty: if cash is near minimum, penalise large allocations
            cash_surplus = max(0.0, cash_weight - self.risk_profile.min_cash_weight)
            cash_penalty = (
                max(0.0, (fraction - cash_surplus) * 0.5)
                if cash_surplus < fraction
                else 0.0
            )

            # Concentration penalty: if adding fraction pushes over max_position
            post_weight = current_ticker_weight + fraction * cash_weight
            conc_penalty = max(
                0.0, (post_weight - self.risk_profile.max_position_weight) * 2.0
            )

            # Volatility penalty: high vol → prefer smaller size
            vol_penalty = volatility_score * 0.3 * fraction

            # Drawdown penalty: deeper drawdown → reduce size (abs of negative value)
            dd_depth = abs(min(0.0, drawdown_from_high))
            dd_penalty = dd_depth * 0.4 * fraction

            # Confidence penalty: low ticker score → prefer smaller size
            conf_penalty = (
                max(0.0, (0.5 - ticker_score) * 0.2) if ticker_score < 0.5 else 0.0
            )

            # Trend guard penalty
            trend_penalty = 0.25 * fraction if bear_market_signal else 0.0

            final = max(
                0.0,
                base
                - cash_penalty
                - conc_penalty
                - vol_penalty
                - dd_penalty
                - conf_penalty
                - trend_penalty,
            )

            row = SizeScoreRow(
                size_label=label,
                fraction=fraction,
                base_score=base,
                cash_buffer_penalty=cash_penalty,
                concentration_penalty=conc_penalty,
                volatility_penalty=vol_penalty,
                drawdown_penalty=dd_penalty,
                confidence_penalty=conf_penalty,
                trend_guard_penalty=trend_penalty,
                final_size_score=final,
            )
            rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # Internal: SELL bucket scoring
    # ------------------------------------------------------------------

    def _score_sell_buckets(
        self,
        ticker_keep_score: float,
        current_ticker_weight: float,
        volatility_score: float,
        drawdown_from_high: float,
        bear_market_signal: bool,
    ) -> list[SizeScoreRow]:
        rows = []
        # Lower keep_score → higher base sell impulse
        sell_impulse = max(0.0, 1.0 - ticker_keep_score)

        for label, fraction in SELL_BUCKETS:
            base = sell_impulse * fraction

            # Volatility → prefer larger sells in high-vol to reduce exposure faster
            vol_bonus = volatility_score * 0.1 * fraction

            # Drawdown → larger sell fraction if already in deep drawdown
            dd_depth = abs(min(0.0, drawdown_from_high))
            dd_bonus = dd_depth * 0.3 * fraction

            # Bear market signal → larger sell fractions preferred
            trend_bonus = 0.15 * fraction if bear_market_signal else 0.0

            final = min(1.0, base + vol_bonus + dd_bonus + trend_bonus)

            row = SizeScoreRow(
                size_label=label,
                fraction=fraction,
                base_score=base,
                volatility_penalty=-vol_bonus,  # stored negative = bonus
                drawdown_penalty=-dd_bonus,
                trend_guard_penalty=-trend_bonus,
                final_size_score=final,
            )
            rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # Internal: pick the highest viable bucket
    # ------------------------------------------------------------------

    def _pick_highest_viable(
        self,
        rows: list[SizeScoreRow],
        warnings: list[str],
    ) -> SizeScoreRow:
        viable = [r for r in rows if r.final_size_score >= self.min_score_threshold]
        if not viable:
            warnings.append(
                "No size bucket passed threshold — defaulting to smallest bucket."
            )
            fallback = rows[0]
            fallback.final_size_score = self.min_score_threshold
            return fallback

        # Return the bucket with the highest final_size_score
        return max(viable, key=lambda r: r.final_size_score)
