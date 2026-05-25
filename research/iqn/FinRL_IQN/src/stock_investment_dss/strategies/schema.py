# src/stock_investment_dss/strategies/schema.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RiskProfile = Literal["low", "medium", "high"]
ScoreQuantile = Literal["q25", "q50", "q75"]


@dataclass(frozen=True)
class StrategyConfig:
    strategy_id: str
    display_name: str
    risk_profile: RiskProfile
    objective: str
    description: str
    constraints: dict[str, Any]
    risk_policy: dict[str, Any]
    allowed_actions: tuple[str, ...]
    rebalance: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyConfig":
        validate_strategy_dict(data)

        return cls(
            strategy_id=data["strategy_id"],
            display_name=data["display_name"],
            risk_profile=data["risk_profile"],
            objective=data["objective"],
            description=data["description"],
            constraints=data["constraints"],
            risk_policy=data["risk_policy"],
            allowed_actions=tuple(data["allowed_actions"]),
            rebalance=data["rebalance"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "display_name": self.display_name,
            "risk_profile": self.risk_profile,
            "objective": self.objective,
            "description": self.description,
            "constraints": self.constraints,
            "risk_policy": self.risk_policy,
            "allowed_actions": list(self.allowed_actions),
            "rebalance": self.rebalance,
        }


def validate_strategy_dict(data: dict[str, Any]) -> None:
    required_top_level_keys = {
        "strategy_id",
        "display_name",
        "risk_profile",
        "objective",
        "description",
        "constraints",
        "risk_policy",
        "allowed_actions",
        "rebalance",
    }

    missing_keys = required_top_level_keys - set(data.keys())

    if missing_keys:
        raise ValueError(f"Strategy config is missing keys: {sorted(missing_keys)}")

    if data["strategy_id"] not in {"defensive", "balanced", "aggressive"}:
        raise ValueError("strategy_id must be one of: defensive, balanced, aggressive")

    if data["risk_profile"] not in {"low", "medium", "high"}:
        raise ValueError("risk_profile must be one of: low, medium, high")

    if not isinstance(data["allowed_actions"], list):
        raise ValueError("allowed_actions must be a list")

    if not data["allowed_actions"]:
        raise ValueError("allowed_actions must not be empty")

    constraints = data["constraints"]
    required_constraint_keys = {
        "allow_shorting",
        "allow_margin",
        "max_position_weight",
        "max_sector_weight",
        "min_cash_weight",
        "target_cash_weight",
        "max_turnover_per_decision",
    }

    missing_constraint_keys = required_constraint_keys - set(constraints.keys())

    if missing_constraint_keys:
        raise ValueError(
            f"strategy.constraints is missing keys: {sorted(missing_constraint_keys)}"
        )

    if constraints["allow_shorting"] is not False:
        raise ValueError("D-IQN-DSS is long-only: allow_shorting must be false")

    if constraints["allow_margin"] is not False:
        raise ValueError("D-IQN-DSS does not allow margin: allow_margin must be false")

    _validate_weight(
        constraints["max_position_weight"],
        "constraints.max_position_weight",
    )
    _validate_weight(
        constraints["max_sector_weight"],
        "constraints.max_sector_weight",
    )
    _validate_weight(
        constraints["min_cash_weight"],
        "constraints.min_cash_weight",
    )
    _validate_weight(
        constraints["target_cash_weight"],
        "constraints.target_cash_weight",
    )
    _validate_weight(
        constraints["max_turnover_per_decision"],
        "constraints.max_turnover_per_decision",
    )

    if constraints["min_cash_weight"] > constraints["target_cash_weight"]:
        raise ValueError(
            "constraints.min_cash_weight cannot be greater than "
            "constraints.target_cash_weight"
        )

    risk_policy = data["risk_policy"]
    required_risk_policy_keys = {
        "lambda_cvar",
        "lambda_drawdown",
        "lambda_volatility",
        "lambda_transaction_cost",
        "lambda_concentration",
        "lambda_strategy_violation",
        "lambda_epistemic_uncertainty",
        "score_quantile",
        "downside_metric",
        "uncertainty_metric",
    }

    missing_risk_policy_keys = required_risk_policy_keys - set(risk_policy.keys())

    if missing_risk_policy_keys:
        raise ValueError(
            f"strategy.risk_policy is missing keys: {sorted(missing_risk_policy_keys)}"
        )

    if risk_policy["score_quantile"] not in {"q25", "q50", "q75"}:
        raise ValueError("risk_policy.score_quantile must be q25, q50, or q75")

    if risk_policy["downside_metric"] != "cvar10":
        raise ValueError("risk_policy.downside_metric must currently be cvar10")

    if risk_policy["uncertainty_metric"] != "epistemic_uncertainty":
        raise ValueError(
            "risk_policy.uncertainty_metric must currently be epistemic_uncertainty"
        )

    for key in [
        "lambda_cvar",
        "lambda_drawdown",
        "lambda_volatility",
        "lambda_transaction_cost",
        "lambda_concentration",
        "lambda_strategy_violation",
        "lambda_epistemic_uncertainty",
    ]:
        _validate_non_negative_number(risk_policy[key], f"risk_policy.{key}")

    rebalance = data["rebalance"]
    required_rebalance_keys = {
        "frequency",
        "drift_threshold",
        "force_if_position_above_max",
    }

    missing_rebalance_keys = required_rebalance_keys - set(rebalance.keys())

    if missing_rebalance_keys:
        raise ValueError(
            f"strategy.rebalance is missing keys: {sorted(missing_rebalance_keys)}"
        )

    if rebalance["frequency"] not in {"weekly", "monthly", "quarterly"}:
        raise ValueError("rebalance.frequency must be weekly, monthly, or quarterly")

    _validate_weight(rebalance["drift_threshold"], "rebalance.drift_threshold")

    if not isinstance(rebalance["force_if_position_above_max"], bool):
        raise ValueError("rebalance.force_if_position_above_max must be boolean")


def _validate_weight(value: Any, field_name: str) -> None:
    if not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric")

    if value < 0 or value > 1:
        raise ValueError(f"{field_name} must be between 0 and 1")


def _validate_non_negative_number(value: Any, field_name: str) -> None:
    if not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric")

    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
