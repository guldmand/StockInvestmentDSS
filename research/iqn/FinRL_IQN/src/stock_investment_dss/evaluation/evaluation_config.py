# src/stock_investment_dss/evaluation/evaluation_config.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVALUATION_CONFIG_DIRECTORY = Path(__file__).resolve().parent / "configs"
DEFAULT_EVALUATION_ID = "default"


@dataclass(frozen=True)
class EvaluationConfig:
    evaluation_id: str
    description: str
    portfolio_metrics: tuple[str, ...]
    distributional_metrics: tuple[str, ...]
    uncertainty_metrics: tuple[str, ...]
    decision_support_metrics: tuple[str, ...]
    comparison_groups: tuple[str, ...]
    primary_metrics: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationConfig":
        validate_evaluation_config_dict(data)

        return cls(
            evaluation_id=data["evaluation_id"],
            description=data["description"],
            portfolio_metrics=tuple(data["portfolio_metrics"]),
            distributional_metrics=tuple(data["distributional_metrics"]),
            uncertainty_metrics=tuple(data["uncertainty_metrics"]),
            decision_support_metrics=tuple(data["decision_support_metrics"]),
            comparison_groups=tuple(data["comparison_groups"]),
            primary_metrics=tuple(data["primary_metrics"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "description": self.description,
            "portfolio_metrics": list(self.portfolio_metrics),
            "distributional_metrics": list(self.distributional_metrics),
            "uncertainty_metrics": list(self.uncertainty_metrics),
            "decision_support_metrics": list(self.decision_support_metrics),
            "comparison_groups": list(self.comparison_groups),
            "primary_metrics": list(self.primary_metrics),
        }


def validate_evaluation_config_dict(data: dict[str, Any]) -> None:
    required_keys = {
        "evaluation_id",
        "description",
        "portfolio_metrics",
        "distributional_metrics",
        "uncertainty_metrics",
        "decision_support_metrics",
        "comparison_groups",
        "primary_metrics",
    }

    missing_keys = required_keys - set(data.keys())

    if missing_keys:
        raise ValueError(f"Evaluation config is missing keys: {sorted(missing_keys)}")

    list_fields = [
        "portfolio_metrics",
        "distributional_metrics",
        "uncertainty_metrics",
        "decision_support_metrics",
        "comparison_groups",
        "primary_metrics",
    ]

    for field_name in list_fields:
        if not isinstance(data[field_name], list):
            raise ValueError(f"evaluation.{field_name} must be a list")

        if len(data[field_name]) == 0:
            raise ValueError(f"evaluation.{field_name} must not be empty")

    required_primary_metrics = {"total_return", "sharpe", "max_drawdown", "cvar10"}
    primary_metrics = set(data["primary_metrics"])

    missing_primary_metrics = required_primary_metrics - primary_metrics

    if missing_primary_metrics:
        raise ValueError(
            "evaluation.primary_metrics must include: "
            f"{sorted(missing_primary_metrics)}"
        )


def load_evaluation_config_from_json(path: Path) -> EvaluationConfig:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return EvaluationConfig.from_dict(data)


def get_evaluation_config(
    evaluation_id: str | None = None,
) -> EvaluationConfig:
    selected_evaluation_id = evaluation_id or DEFAULT_EVALUATION_ID
    evaluation_path = EVALUATION_CONFIG_DIRECTORY / f"{selected_evaluation_id}.json"

    return load_evaluation_config_from_json(evaluation_path)


def list_available_evaluation_ids() -> tuple[str, ...]:
    if not EVALUATION_CONFIG_DIRECTORY.exists():
        return tuple()

    return tuple(
        sorted(path.stem for path in EVALUATION_CONFIG_DIRECTORY.glob("*.json"))
    )
