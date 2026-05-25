# src/stock_investment_dss/evaluation/portfolio_metrics.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioMetricsResult:
    summary: dict[str, Any]
    timeseries: pd.DataFrame


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def _find_portfolio_value_column(data: pd.DataFrame) -> str:
    candidates = [
        "account_value",
        "portfolio_value",
        "total_asset",
        "asset_value",
        "value",
    ]

    for column in candidates:
        if column in data.columns:
            return column

    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_columns:
        raise ValueError(
            "Could not find a portfolio value column. "
            "Expected one of: account_value, portfolio_value, total_asset, asset_value."
        )

    return numeric_columns[-1]


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass

    return float(value)


def _compute_max_drawdown_pct(values: pd.Series) -> float:
    if values.empty:
        return 0.0

    running_max = values.cummax()
    drawdown = (values / running_max) - 1.0

    return float(drawdown.min() * 100.0)


def _compute_annualized_volatility_pct(
    returns: pd.Series, periods_per_year: int
) -> float:
    clean_returns = returns.dropna()

    if len(clean_returns) < 2:
        return 0.0

    return float(clean_returns.std(ddof=1) * np.sqrt(periods_per_year) * 100.0)


def _compute_annualized_sharpe(
    returns: pd.Series,
    periods_per_year: int,
    risk_free_rate_per_period: float = 0.0,
) -> float | None:
    clean_returns = returns.dropna()

    if len(clean_returns) < 2:
        return None

    excess_returns = clean_returns - risk_free_rate_per_period
    std = excess_returns.std(ddof=1)

    if std == 0 or pd.isna(std):
        return None

    return float((excess_returns.mean() / std) * np.sqrt(periods_per_year))


def _compute_cvar_pct(returns: pd.Series, alpha: float = 0.10) -> float | None:
    clean_returns = returns.dropna()

    if clean_returns.empty:
        return None

    cutoff = clean_returns.quantile(alpha)
    tail_returns = clean_returns[clean_returns <= cutoff]

    if tail_returns.empty:
        return None

    return float(tail_returns.mean() * 100.0)


def _extract_decision_rows(decision_memory: dict) -> pd.DataFrame:
    decisions = decision_memory.get("decisions", [])

    rows: list[dict[str, Any]] = []

    for decision in decisions:
        state_after = decision.get("state_after") or {}
        state_before = decision.get("state_before") or {}
        resolved_action = decision.get("resolved_action") or {}
        execution_delta = decision.get("execution_delta") or {}

        portfolio_value_after = _safe_float(state_after.get("portfolio_value"))
        portfolio_value_before = _safe_float(state_before.get("portfolio_value"))
        cash_after = _safe_float(state_after.get("cash"))
        cash_weight_after = (
            cash_after / portfolio_value_after if portfolio_value_after > 0 else 0.0
        )

        position_weights = state_after.get("position_weights") or {}
        max_position_weight_after = (
            max([_safe_float(value) for value in position_weights.values()])
            if position_weights
            else 0.0
        )

        rows.append(
            {
                "decision_step": decision.get("decision_step"),
                "requested_decision_action": decision.get(
                    "requested_decision_action_label"
                ),
                "effective_decision_action": decision.get(
                    "effective_decision_action_label"
                ),
                "action_was_masked": decision.get("action_was_masked"),
                "selected_ticker": resolved_action.get("selected_ticker"),
                "requested_shares": resolved_action.get("requested_shares"),
                "submitted_shares_estimate": resolved_action.get(
                    "submitted_shares_estimate"
                ),
                "hmax_limited": resolved_action.get("hmax_limited"),
                "executed_shares_delta": execution_delta.get("executed_shares_delta"),
                "cash_delta": execution_delta.get("cash_delta"),
                "portfolio_value_delta": execution_delta.get("portfolio_value_delta"),
                "cost_delta": execution_delta.get("cost_delta"),
                "trades_delta": execution_delta.get("trades_delta"),
                "portfolio_value_before": portfolio_value_before,
                "portfolio_value_after": portfolio_value_after,
                "cash_after": cash_after,
                "cash_weight_after": cash_weight_after,
                "max_position_weight_after": max_position_weight_after,
                "finrl_cost": decision.get("finrl_cost"),
                "finrl_trades": decision.get("finrl_trades"),
            }
        )

    return pd.DataFrame(rows)


def compute_portfolio_metrics(
    asset_memory: pd.DataFrame,
    decision_memory: dict | None = None,
    step_table: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    cvar_alpha: float = 0.10,
) -> PortfolioMetricsResult:
    """
    Computes basic portfolio metrics from FinRL/DSS output.

    Inputs:
    - asset_memory: FinRL asset memory, usually with account_value.
    - decision_memory: DSS decision memory JSON.
    - step_table: optional flattened DSS step table.

    Output:
    - summary: aggregate metrics.
    - timeseries: portfolio value curve and derived returns/drawdowns.
    """

    if asset_memory.empty:
        raise ValueError("asset_memory is empty.")

    asset_data = asset_memory.copy()
    portfolio_value_column = _find_portfolio_value_column(asset_data)

    asset_data["portfolio_value"] = pd.to_numeric(
        asset_data[portfolio_value_column],
        errors="coerce",
    )

    asset_data = asset_data.dropna(subset=["portfolio_value"]).reset_index(drop=True)

    if asset_data.empty:
        raise ValueError("asset_memory did not contain usable portfolio values.")

    asset_data["step_return"] = asset_data["portfolio_value"].pct_change().fillna(0.0)
    asset_data["cumulative_return"] = (
        asset_data["portfolio_value"] / asset_data["portfolio_value"].iloc[0]
    ) - 1.0

    asset_data["running_max"] = asset_data["portfolio_value"].cummax()
    asset_data["drawdown"] = (
        asset_data["portfolio_value"] / asset_data["running_max"]
    ) - 1.0

    initial_value = float(asset_data["portfolio_value"].iloc[0])
    final_value = float(asset_data["portfolio_value"].iloc[-1])

    total_return_pct = (
        ((final_value - initial_value) / initial_value) * 100.0
        if initial_value != 0
        else None
    )

    max_drawdown_pct = _compute_max_drawdown_pct(asset_data["portfolio_value"])
    volatility_pct = _compute_annualized_volatility_pct(
        asset_data["step_return"],
        periods_per_year=periods_per_year,
    )
    sharpe = _compute_annualized_sharpe(
        asset_data["step_return"],
        periods_per_year=periods_per_year,
    )
    cvar_pct = _compute_cvar_pct(asset_data["step_return"], alpha=cvar_alpha)

    decision_rows = (
        _extract_decision_rows(decision_memory)
        if decision_memory is not None
        else pd.DataFrame()
    )

    total_transaction_cost = None
    total_trades = None
    turnover_estimate_pct = None
    final_cash_weight = None
    max_concentration = None

    if not decision_rows.empty:
        total_transaction_cost = float(
            pd.to_numeric(decision_rows["cost_delta"], errors="coerce")
            .fillna(0.0)
            .sum()
        )

        total_trades = int(
            pd.to_numeric(decision_rows["trades_delta"], errors="coerce")
            .fillna(0)
            .sum()
        )

        submitted_cash_values = pd.to_numeric(
            decision_rows["cash_delta"],
            errors="coerce",
        ).fillna(0.0)

        average_portfolio_value = float(asset_data["portfolio_value"].mean())

        if average_portfolio_value > 0:
            turnover_estimate_pct = float(
                submitted_cash_values.abs().sum() / average_portfolio_value * 100.0
            )

        final_cash_weight = float(decision_rows["cash_weight_after"].iloc[-1])
        max_concentration = float(decision_rows["max_position_weight_after"].max())

        decision_metrics = decision_rows[
            [
                "decision_step",
                "requested_decision_action",
                "effective_decision_action",
                "action_was_masked",
                "selected_ticker",
                "requested_shares",
                "submitted_shares_estimate",
                "hmax_limited",
                "executed_shares_delta",
                "cash_delta",
                "portfolio_value_delta",
                "cost_delta",
                "trades_delta",
                "cash_weight_after",
                "max_position_weight_after",
            ]
        ].copy()

        asset_data = asset_data.reset_index(drop=True)
        decision_metrics = decision_metrics.reset_index(drop=True)

        timeseries = pd.concat([asset_data, decision_metrics], axis=1)
    else:
        timeseries = asset_data

    summary = {
        "status": "ok",
        "portfolio_value_column_used": portfolio_value_column,
        "periods_per_year": periods_per_year,
        "cvar_alpha": cvar_alpha,
        "row_count": int(len(asset_data)),
        "initial_value": initial_value,
        "final_value": final_value,
        "profit_loss": final_value - initial_value,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "annualized_volatility_pct": volatility_pct,
        "annualized_sharpe": sharpe,
        "cvar_pct": cvar_pct,
        "total_transaction_cost": total_transaction_cost,
        "total_trades": total_trades,
        "turnover_estimate_pct": turnover_estimate_pct,
        "final_cash_weight": final_cash_weight,
        "max_concentration": max_concentration,
    }

    if step_table is not None and not step_table.empty:
        summary["step_table_rows"] = int(len(step_table))
        summary["step_table_columns"] = list(step_table.columns)

    return PortfolioMetricsResult(
        summary=summary,
        timeseries=timeseries,
    )


def compute_portfolio_metrics_from_files(
    asset_memory_path: Path,
    decision_memory_path: Path | None = None,
    step_table_path: Path | None = None,
    periods_per_year: int = 252,
    cvar_alpha: float = 0.10,
) -> PortfolioMetricsResult:
    asset_memory = pd.read_csv(asset_memory_path)

    decision_memory = (
        read_json(decision_memory_path)
        if decision_memory_path is not None and decision_memory_path.exists()
        else None
    )

    step_table = (
        pd.read_csv(step_table_path)
        if step_table_path is not None and step_table_path.exists()
        else None
    )

    return compute_portfolio_metrics(
        asset_memory=asset_memory,
        decision_memory=decision_memory,
        step_table=step_table,
        periods_per_year=periods_per_year,
        cvar_alpha=cvar_alpha,
    )
