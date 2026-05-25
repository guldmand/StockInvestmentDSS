# src/stock_investment_dss/decision/action_mask.py

from __future__ import annotations

from dataclasses import dataclass

from stock_investment_dss.decision.decision_actions import DSSDecisionAction
from stock_investment_dss.decision.investor_risk_profile import InvestorRiskProfile


@dataclass(frozen=True)
class DSSActionMaskResult:
    allowed_actions: dict[str, bool]
    blocked_reasons: dict[str, str]
    mask_vector: list[int]

    def is_allowed(self, action: DSSDecisionAction | int) -> bool:
        action_enum = DSSDecisionAction(action)
        return bool(self.allowed_actions[action_enum.name])


class DSSActionMaskGenerator:
    """
    Computes which high-level DSS actions are valid in the current state.

    This does not execute trades.
    It only says which DSS decision actions should be available.
    """

    def __init__(
        self,
        tickers: list[str] | tuple[str, ...],
        risk_profile: InvestorRiskProfile,
        min_trade_cash_value: float = 1.0,
        allow_change_strategy_without_signal: bool = False,
    ) -> None:
        self.tickers = [ticker.upper().strip() for ticker in tickers]
        self.risk_profile = risk_profile
        self.min_trade_cash_value = float(min_trade_cash_value)
        self.allow_change_strategy_without_signal = allow_change_strategy_without_signal

    def generate(self, state_summary: dict) -> DSSActionMaskResult:
        cash = float(state_summary["cash"])
        portfolio_value = float(state_summary["portfolio_value"])
        holdings = state_summary["holdings"]
        prices = state_summary["prices"]
        weights = state_summary["position_weights"]

        cash_weight = cash / portfolio_value if portfolio_value > 0 else 0.0

        max_available_cash = max(
            0.0,
            cash - self.risk_profile.min_cash_weight * portfolio_value,
        )

        affordable_tickers = [
            ticker
            for ticker in self.tickers
            if max_available_cash >= float(prices[ticker])
        ]

        underweight_tickers = [
            ticker
            for ticker in self.tickers
            if float(weights.get(ticker, 0.0)) < self.risk_profile.max_position_weight
        ]

        held_tickers = [
            ticker for ticker in self.tickers if float(holdings.get(ticker, 0.0)) > 0
        ]

        overweight_tickers = [
            ticker
            for ticker in self.tickers
            if float(weights.get(ticker, 0.0)) > self.risk_profile.max_position_weight
        ]

        rebalance_needed = (
            bool(overweight_tickers) or cash_weight < self.risk_profile.min_cash_weight
        )

        allowed_actions = {
            "HOLD": True,
            "BUY": (
                max_available_cash >= self.min_trade_cash_value
                and bool(affordable_tickers)
                and bool(underweight_tickers)
            ),
            "SELL": bool(held_tickers),
            "REBALANCE": rebalance_needed,
            "CHANGE_STRATEGY": self.allow_change_strategy_without_signal,
        }

        blocked_reasons = {}

        if not allowed_actions["BUY"]:
            blocked_reasons["BUY"] = (
                "BUY blocked because available cash is too low, no ticker is "
                "affordable, or all positions are at/above max_position_weight."
            )

        if not allowed_actions["SELL"]:
            blocked_reasons["SELL"] = "SELL blocked because there are no holdings."

        if not allowed_actions["REBALANCE"]:
            blocked_reasons["REBALANCE"] = (
                "REBALANCE blocked because portfolio is within simple risk limits."
            )

        if not allowed_actions["CHANGE_STRATEGY"]:
            blocked_reasons["CHANGE_STRATEGY"] = (
                "CHANGE_STRATEGY blocked because no strategy-change signal is "
                "available in this smoke-test implementation."
            )

        mask_vector = [
            int(allowed_actions["HOLD"]),
            int(allowed_actions["BUY"]),
            int(allowed_actions["SELL"]),
            int(allowed_actions["REBALANCE"]),
            int(allowed_actions["CHANGE_STRATEGY"]),
        ]

        return DSSActionMaskResult(
            allowed_actions=allowed_actions,
            blocked_reasons=blocked_reasons,
            mask_vector=mask_vector,
        )
