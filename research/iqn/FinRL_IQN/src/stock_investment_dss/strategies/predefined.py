# src/stock_investment_dss/strategies/predefined.py

from __future__ import annotations

import json
from pathlib import Path

from stock_investment_dss.strategies.schema import StrategyConfig

STRATEGY_DIRECTORY = Path(__file__).resolve().parent / "predefined"

DEFAULT_STRATEGY_ID = "balanced"


def load_strategy_from_json(path: Path) -> StrategyConfig:
    if not path.exists():
        raise FileNotFoundError(f"Strategy file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return StrategyConfig.from_dict(data)


def get_strategy(strategy_id: str | None = None) -> StrategyConfig:
    selected_strategy_id = strategy_id or DEFAULT_STRATEGY_ID
    strategy_path = STRATEGY_DIRECTORY / f"{selected_strategy_id}.json"

    return load_strategy_from_json(strategy_path)


def list_available_strategy_ids() -> tuple[str, ...]:
    if not STRATEGY_DIRECTORY.exists():
        return tuple()

    return tuple(sorted(path.stem for path in STRATEGY_DIRECTORY.glob("*.json")))


def load_all_predefined_strategies() -> tuple[StrategyConfig, ...]:
    return tuple(
        load_strategy_from_json(path)
        for path in sorted(STRATEGY_DIRECTORY.glob("*.json"))
    )
