# src/stock_investment_dss/decision/investor_risk_profile.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InvestorRiskProfile:
    """
    Minimal investor risk profile for the DSS action resolver.

    risk_willingness:
        0.0 = defensive
        0.5 = balanced
        1.0 = aggressive
    """

    risk_willingness: float = 0.5
    max_position_weight: float = 0.25
    min_cash_weight: float = 0.10
    max_trade_fraction_of_cash: float = 0.25
    max_sell_fraction_of_position: float = 0.50
    max_drawdown_tolerance: float = 0.15
    downside_risk_weight: float = 0.60
    uncertainty_penalty_weight: float = 0.40

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_willingness <= 1.0:
            raise ValueError("risk_willingness must be in [0, 1].")

        if not 0.0 <= self.max_position_weight <= 1.0:
            raise ValueError("max_position_weight must be in [0, 1].")

        if not 0.0 <= self.min_cash_weight <= 1.0:
            raise ValueError("min_cash_weight must be in [0, 1].")

    @staticmethod
    def balanced() -> "InvestorRiskProfile":
        return InvestorRiskProfile(
            risk_willingness=0.5,
            max_position_weight=0.25,
            min_cash_weight=0.10,
            max_trade_fraction_of_cash=0.25,
            max_sell_fraction_of_position=0.50,
            max_drawdown_tolerance=0.15,
            downside_risk_weight=0.60,
            uncertainty_penalty_weight=0.40,
        )

    @staticmethod
    def defensive() -> "InvestorRiskProfile":
        return InvestorRiskProfile(
            risk_willingness=0.2,
            max_position_weight=0.15,
            min_cash_weight=0.20,
            max_trade_fraction_of_cash=0.10,
            max_sell_fraction_of_position=0.75,
            max_drawdown_tolerance=0.08,
            downside_risk_weight=0.85,
            uncertainty_penalty_weight=0.70,
        )

    @staticmethod
    def aggressive() -> "InvestorRiskProfile":
        return InvestorRiskProfile(
            risk_willingness=0.8,
            max_position_weight=0.40,
            min_cash_weight=0.02,
            max_trade_fraction_of_cash=0.50,
            max_sell_fraction_of_position=0.35,
            max_drawdown_tolerance=0.25,
            downside_risk_weight=0.35,
            uncertainty_penalty_weight=0.20,
        )

    def to_dict(self) -> dict:
        return {
            "risk_willingness": self.risk_willingness,
            "max_position_weight": self.max_position_weight,
            "min_cash_weight": self.min_cash_weight,
            "max_trade_fraction_of_cash": self.max_trade_fraction_of_cash,
            "max_sell_fraction_of_position": self.max_sell_fraction_of_position,
            "max_drawdown_tolerance": self.max_drawdown_tolerance,
            "downside_risk_weight": self.downside_risk_weight,
            "uncertainty_penalty_weight": self.uncertainty_penalty_weight,
        }
