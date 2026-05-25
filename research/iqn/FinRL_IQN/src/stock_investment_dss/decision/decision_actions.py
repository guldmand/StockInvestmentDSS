# src/stock_investment_dss/decision/decision_actions.py

from __future__ import annotations

from enum import IntEnum


class DSSDecisionAction(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2
    REBALANCE = 3
    CHANGE_STRATEGY = 4


ACTION_LABELS = {
    DSSDecisionAction.HOLD: "HOLD",
    DSSDecisionAction.BUY: "BUY",
    DSSDecisionAction.SELL: "SELL",
    DSSDecisionAction.REBALANCE: "REBALANCE",
    DSSDecisionAction.CHANGE_STRATEGY: "CHANGE_STRATEGY",
}


def action_to_label(action: DSSDecisionAction | int) -> str:
    return ACTION_LABELS[DSSDecisionAction(action)]


def parse_action_label(label: str) -> DSSDecisionAction:
    normalized = label.strip().upper()

    for action, action_label in ACTION_LABELS.items():
        if normalized == action_label:
            return action

    available = ", ".join(ACTION_LABELS.values())
    raise ValueError(f"Unknown DSS decision action: {label}. Available: {available}")


def parse_action_sequence(raw_value: str) -> list[DSSDecisionAction]:
    return [parse_action_label(part) for part in raw_value.split(",") if part.strip()]
