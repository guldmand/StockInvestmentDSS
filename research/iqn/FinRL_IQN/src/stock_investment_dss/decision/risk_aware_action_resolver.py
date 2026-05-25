# src/stock_investment_dss/decision/risk_aware_action_resolver.py

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np

from stock_investment_dss.decision.decision_actions import DSSDecisionAction
from stock_investment_dss.decision.investor_risk_profile import InvestorRiskProfile


@dataclass(frozen=True)
class ResolvedDSSAction:
    decision_action: str
    selected_ticker: str | None
    requested_shares: int
    requested_cash_value: float
    submitted_shares_estimate: int
    submitted_cash_value_estimate: float
    hmax_limited: bool
    continuous_action: list[float]
    reason: str
    constraints: dict
    metadata: dict


class RiskAwareActionResolver:
    """
    Converts high-level DSS decision actions into FinRL continuous action vectors.

    Important:
    - This class does NOT update cash/holdings.
    - It only resolves DSS intent into a FinRL-compatible action vector.
    - FinRL StockTradingEnv still performs the actual trading mechanics.
    """

    def __init__(
        self,
        tickers: list[str] | tuple[str, ...],
        hmax: int,
        risk_profile: InvestorRiskProfile,
    ) -> None:
        self.tickers = [ticker.upper().strip() for ticker in tickers]
        self.hmax = int(hmax)
        self.risk_profile = risk_profile

        if self.hmax <= 0:
            raise ValueError("hmax must be positive.")

    def resolve(
        self,
        decision_action: DSSDecisionAction | int,
        state_summary: dict,
    ) -> ResolvedDSSAction:
        action = DSSDecisionAction(decision_action)

        if action == DSSDecisionAction.HOLD:
            return self._resolve_hold(state_summary)

        if action == DSSDecisionAction.BUY:
            return self._resolve_buy(state_summary)

        if action == DSSDecisionAction.SELL:
            return self._resolve_sell(state_summary)

        if action == DSSDecisionAction.REBALANCE:
            return self._resolve_rebalance(state_summary)

        if action == DSSDecisionAction.CHANGE_STRATEGY:
            return self._resolve_change_strategy(state_summary)

        raise ValueError(f"Unsupported DSS decision action: {decision_action}")

    def resolve_blocked_action(
        self,
        requested_decision_action: DSSDecisionAction | int,
        state_summary: dict,
        blocked_reason: str,
    ) -> ResolvedDSSAction:
        requested_action = DSSDecisionAction(requested_decision_action)

        return ResolvedDSSAction(
            decision_action=requested_action.name,
            selected_ticker=None,
            requested_shares=0,
            requested_cash_value=0.0,
            submitted_shares_estimate=0,
            submitted_cash_value_estimate=0.0,
            hmax_limited=False,
            continuous_action=self._zero_action().tolist(),
            reason=f"{requested_action.name} was blocked by action mask. {blocked_reason}",
            constraints=self._base_constraints(),
            metadata={
                "blocked_by_action_mask": True,
                "state_summary": state_summary,
            },
        )

    def _zero_action(self) -> np.ndarray:
        return np.zeros(len(self.tickers), dtype=float)

    def _base_constraints(self) -> dict:
        return {
            "hmax": self.hmax,
            "risk_profile": self.risk_profile.to_dict(),
        }

    def _make_action_vector(
        self,
        ticker: str | None,
        submitted_shares: int,
        direction: int,
    ) -> list[float]:
        continuous_action = self._zero_action()

        if ticker is None or submitted_shares == 0:
            return continuous_action.tolist()

        ticker_index = self.tickers.index(ticker)
        continuous_action[ticker_index] = direction * min(
            1.0,
            submitted_shares / self.hmax,
        )

        return continuous_action.tolist()

    def _resolve_hold(self, state_summary: dict) -> ResolvedDSSAction:
        return ResolvedDSSAction(
            decision_action="HOLD",
            selected_ticker=None,
            requested_shares=0,
            requested_cash_value=0.0,
            submitted_shares_estimate=0,
            submitted_cash_value_estimate=0.0,
            hmax_limited=False,
            continuous_action=self._zero_action().tolist(),
            reason="HOLD maps to a zero FinRL action vector.",
            constraints=self._base_constraints(),
            metadata={"state_summary": state_summary},
        )

    def _resolve_buy(self, state_summary: dict) -> ResolvedDSSAction:
        cash = float(state_summary["cash"])
        portfolio_value = float(state_summary["portfolio_value"])
        prices = state_summary["prices"]
        weights = state_summary["position_weights"]

        max_available_cash = max(
            0.0,
            cash - self.risk_profile.min_cash_weight * portfolio_value,
        )

        requested_cash = min(
            cash * self.risk_profile.max_trade_fraction_of_cash,
            max_available_cash,
        )

        eligible_tickers = [
            ticker
            for ticker in self.tickers
            if float(weights.get(ticker, 0.0)) < self.risk_profile.max_position_weight
        ]

        if requested_cash <= 0 or not eligible_tickers:
            return ResolvedDSSAction(
                decision_action="BUY",
                selected_ticker=None,
                requested_shares=0,
                requested_cash_value=0.0,
                submitted_shares_estimate=0,
                submitted_cash_value_estimate=0.0,
                hmax_limited=False,
                continuous_action=self._zero_action().tolist(),
                reason=(
                    "BUY blocked because available cash is too low or all positions "
                    "are at/above max_position_weight."
                ),
                constraints=self._base_constraints(),
                metadata={
                    "cash": cash,
                    "portfolio_value": portfolio_value,
                    "max_available_cash": max_available_cash,
                    "eligible_tickers": eligible_tickers,
                },
            )

        selected_ticker = min(
            eligible_tickers,
            key=lambda ticker: float(weights.get(ticker, 0.0)),
        )

        price = float(prices[selected_ticker])
        requested_shares = max(0, floor(requested_cash / price))
        submitted_shares = min(requested_shares, self.hmax)
        hmax_limited = requested_shares > submitted_shares

        return ResolvedDSSAction(
            decision_action="BUY",
            selected_ticker=selected_ticker,
            requested_shares=requested_shares,
            requested_cash_value=float(requested_shares * price),
            submitted_shares_estimate=submitted_shares,
            submitted_cash_value_estimate=float(submitted_shares * price),
            hmax_limited=hmax_limited,
            continuous_action=self._make_action_vector(
                ticker=selected_ticker,
                submitted_shares=submitted_shares,
                direction=1,
            ),
            reason=(
                "BUY resolved to the eligible ticker with the lowest current "
                "position weight."
            ),
            constraints=self._base_constraints(),
            metadata={
                "cash": cash,
                "portfolio_value": portfolio_value,
                "max_available_cash": max_available_cash,
                "requested_cash_before_rounding": requested_cash,
                "selected_price": price,
                "selected_position_weight": float(weights.get(selected_ticker, 0.0)),
            },
        )

    def _resolve_sell(self, state_summary: dict) -> ResolvedDSSAction:
        holdings = state_summary["holdings"]
        weights = state_summary["position_weights"]
        prices = state_summary["prices"]

        held_tickers = [
            ticker for ticker in self.tickers if float(holdings.get(ticker, 0.0)) > 0
        ]

        if not held_tickers:
            return ResolvedDSSAction(
                decision_action="SELL",
                selected_ticker=None,
                requested_shares=0,
                requested_cash_value=0.0,
                submitted_shares_estimate=0,
                submitted_cash_value_estimate=0.0,
                hmax_limited=False,
                continuous_action=self._zero_action().tolist(),
                reason="SELL blocked because there are no holdings to sell.",
                constraints=self._base_constraints(),
                metadata={"holdings": holdings},
            )

        selected_ticker = max(
            held_tickers,
            key=lambda ticker: float(weights.get(ticker, 0.0)),
        )

        current_shares = float(holdings[selected_ticker])
        requested_shares = max(
            1,
            floor(current_shares * self.risk_profile.max_sell_fraction_of_position),
        )
        requested_shares = min(int(current_shares), requested_shares)

        submitted_shares = min(requested_shares, self.hmax)
        hmax_limited = requested_shares > submitted_shares

        price = float(prices[selected_ticker])

        return ResolvedDSSAction(
            decision_action="SELL",
            selected_ticker=selected_ticker,
            requested_shares=requested_shares,
            requested_cash_value=float(requested_shares * price),
            submitted_shares_estimate=submitted_shares,
            submitted_cash_value_estimate=float(submitted_shares * price),
            hmax_limited=hmax_limited,
            continuous_action=self._make_action_vector(
                ticker=selected_ticker,
                submitted_shares=submitted_shares,
                direction=-1,
            ),
            reason="SELL resolved to the currently largest portfolio position.",
            constraints=self._base_constraints(),
            metadata={
                "current_shares": current_shares,
                "selected_price": price,
                "selected_position_weight": float(weights.get(selected_ticker, 0.0)),
            },
        )

    def _resolve_rebalance(self, state_summary: dict) -> ResolvedDSSAction:
        weights = state_summary["position_weights"]
        cash = float(state_summary["cash"])
        portfolio_value = float(state_summary["portfolio_value"])
        cash_weight = cash / portfolio_value if portfolio_value > 0 else 0.0

        overweight_tickers = [
            ticker
            for ticker in self.tickers
            if float(weights.get(ticker, 0.0)) > self.risk_profile.max_position_weight
        ]

        if overweight_tickers or cash_weight < self.risk_profile.min_cash_weight:
            sell_action = self._resolve_sell(state_summary)
            return ResolvedDSSAction(
                decision_action="REBALANCE",
                selected_ticker=sell_action.selected_ticker,
                requested_shares=sell_action.requested_shares,
                requested_cash_value=sell_action.requested_cash_value,
                submitted_shares_estimate=sell_action.submitted_shares_estimate,
                submitted_cash_value_estimate=sell_action.submitted_cash_value_estimate,
                hmax_limited=sell_action.hmax_limited,
                continuous_action=sell_action.continuous_action,
                reason=(
                    "REBALANCE resolved to SELL because portfolio is overweight "
                    "or cash weight is below minimum."
                ),
                constraints=self._base_constraints(),
                metadata={
                    "cash_weight": cash_weight,
                    "overweight_tickers": overweight_tickers,
                    "underlying_sell_metadata": sell_action.metadata,
                },
            )

        return ResolvedDSSAction(
            decision_action="REBALANCE",
            selected_ticker=None,
            requested_shares=0,
            requested_cash_value=0.0,
            submitted_shares_estimate=0,
            submitted_cash_value_estimate=0.0,
            hmax_limited=False,
            continuous_action=self._zero_action().tolist(),
            reason="REBALANCE resolved to HOLD because portfolio is within simple risk limits.",
            constraints=self._base_constraints(),
            metadata={
                "cash_weight": cash_weight,
                "overweight_tickers": overweight_tickers,
            },
        )

    def _resolve_change_strategy(self, state_summary: dict) -> ResolvedDSSAction:
        return ResolvedDSSAction(
            decision_action="CHANGE_STRATEGY",
            selected_ticker=None,
            requested_shares=0,
            requested_cash_value=0.0,
            submitted_shares_estimate=0,
            submitted_cash_value_estimate=0.0,
            hmax_limited=False,
            continuous_action=self._zero_action().tolist(),
            reason=(
                "CHANGE_STRATEGY is a DSS recommendation and does not execute "
                "a trade in this smoke-test implementation."
            ),
            constraints=self._base_constraints(),
            metadata={
                "state_summary": state_summary,
                "suggested_next_step": "Review investor strategy and risk profile.",
            },
        )
