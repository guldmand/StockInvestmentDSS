# src/stock_investment_dss/decision/hierarchical_decision_policy.py
"""
Hierarchical Decision Policy for D-IQN-DSS (v3.0 PoC).

Orchestrates the 5-stage DSS decision pipeline:

    Stage 1 — Action type (from IQN or forced via env/smoke test)
              HOLD / BUY / SELL / REBALANCE

    Stage 2 — Ticker selection (rule-based, TickerSelector)

    Stage 3 — Size selection (rule-based, SizeSelector)

    Stage 4 — Risk / strategy validation
              Constraints checked, warnings added, size may be reduced.

    Stage 5 — Audit ledger
              Full point-in-time decision record written to audit dict.

This module does NOT depend on any IQN training infrastructure.
It can be driven by a forced action type for smoke testing.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from stock_investment_dss.decision.decision_actions import (
    DSSDecisionAction,
    action_to_label,
)
from stock_investment_dss.decision.investor_risk_profile import InvestorRiskProfile
from stock_investment_dss.decision.size_selector import SizeSelector, SizeResult
from stock_investment_dss.decision.ticker_selector import TickerSelector, TickerScoreRow

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Snapshot of portfolio at decision time."""

    total_value: float = 1_000_000.0
    cash: float = 1_000_000.0
    holdings: dict[str, float] = field(default_factory=dict)  # {ticker: shares}
    holding_values: dict[str, float] = field(default_factory=dict)  # {ticker: $value}

    @property
    def cash_weight(self) -> float:
        if self.total_value <= 0:
            return 1.0
        return self.cash / self.total_value

    @property
    def position_weights(self) -> dict[str, float]:
        if self.total_value <= 0:
            return {}
        return {t: v / self.total_value for t, v in self.holding_values.items()}

    def to_dict(self) -> dict:
        return {
            "total_value": self.total_value,
            "cash": self.cash,
            "cash_weight": self.cash_weight,
            "holdings": self.holdings,
            "holding_values": self.holding_values,
            "position_weights": self.position_weights,
        }


@dataclass
class RiskCheckResult:
    max_position_ok: bool = True
    cash_buffer_ok: bool = True
    concentration_ok: bool = True
    drawdown_guard_ok: bool = True
    ma200_trend_ok: bool = True
    no_shorting_ok: bool = True
    no_sell_without_holdings_ok: bool = True
    no_buy_without_cash_ok: bool = True
    strategy_constraints_ok: bool = True
    bear_market_penalty: float = 0.0
    bear_market_signal: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(
            [
                self.max_position_ok,
                self.cash_buffer_ok,
                self.concentration_ok,
                self.drawdown_guard_ok,
                self.no_shorting_ok,
                self.no_sell_without_holdings_ok,
                self.no_buy_without_cash_ok,
                self.strategy_constraints_ok,
            ]
        )

    def to_dict(self) -> dict:
        return {
            "max_position_ok": self.max_position_ok,
            "cash_buffer_ok": self.cash_buffer_ok,
            "concentration_ok": self.concentration_ok,
            "drawdown_guard_ok": self.drawdown_guard_ok,
            "ma200_trend_ok": self.ma200_trend_ok,
            "no_shorting_ok": self.no_shorting_ok,
            "no_sell_without_holdings_ok": self.no_sell_without_holdings_ok,
            "no_buy_without_cash_ok": self.no_buy_without_cash_ok,
            "strategy_constraints_ok": self.strategy_constraints_ok,
            "bear_market_signal": self.bear_market_signal,
            "bear_market_penalty": self.bear_market_penalty,
            "warnings": self.warnings,
        }


@dataclass
class HierarchicalDecision:
    """Full output of one hierarchical policy evaluation."""

    decision_id: str
    date: str
    visible_data_cutoff: str
    portfolio_state: dict
    strategy_id: str
    risk_profile: dict

    # IQN / model context (optional — None in smoke test)
    iqn_model_run_id: Optional[str] = None
    iqn_config: Optional[dict] = None
    score_mode: Optional[str] = None

    # Stage 1
    stage_1_action_type_scores: dict = field(default_factory=dict)
    selected_action_type: str = "HOLD"

    # Stage 2
    stage_2_ticker_scores: list[dict] = field(default_factory=list)
    selected_ticker: Optional[str] = None

    # Stage 3
    stage_3_size_scores: list[dict] = field(default_factory=list)
    selected_size: Optional[str] = None
    selected_fraction: float = 0.0
    risk_adjusted_allocation_fraction: float = 0.0

    # Stage 4
    risk_checks: dict = field(default_factory=dict)

    # Stage 5 — final recommendation
    final_recommendation: dict = field(default_factory=dict)
    execution_result: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "date": self.date,
            "visible_data_cutoff": self.visible_data_cutoff,
            "portfolio_state": self.portfolio_state,
            "strategy_id": self.strategy_id,
            "risk_profile": self.risk_profile,
            "iqn_model_run_id": self.iqn_model_run_id,
            "iqn_config": self.iqn_config,
            "score_mode": self.score_mode,
            "stage_1_action_type_scores": self.stage_1_action_type_scores,
            "selected_action_type": self.selected_action_type,
            "stage_2_ticker_scores": self.stage_2_ticker_scores,
            "selected_ticker": self.selected_ticker,
            "stage_3_size_scores": self.stage_3_size_scores,
            "selected_size": self.selected_size,
            "selected_fraction": self.selected_fraction,
            "risk_adjusted_allocation_fraction": self.risk_adjusted_allocation_fraction,
            "risk_checks": self.risk_checks,
            "final_recommendation": self.final_recommendation,
            "execution_result": self.execution_result,
        }


class HierarchicalDecisionPolicy:
    """
    Orchestrates all 5 stages of the DSS decision pipeline.

    Parameters
    ----------
    risk_profile : InvestorRiskProfile
    strategy_id : str
    ticker_selector : TickerSelector (optional, created from risk_profile if None)
    size_selector : SizeSelector (optional, created from risk_profile if None)
    defensive_strategy : bool
        If True, applies conservative size caps and stronger bear-market guard.
    max_drawdown_guard_threshold : float
        If portfolio drawdown exceeds this, trigger drawdown_guard.
    bear_ma200_penalty : float
        Score penalty applied when ticker is below MA200.
    """

    def __init__(
        self,
        risk_profile: Optional[InvestorRiskProfile] = None,
        strategy_id: str = "balanced_v1",
        ticker_selector: Optional[TickerSelector] = None,
        size_selector: Optional[SizeSelector] = None,
        defensive_strategy: bool = False,
        max_drawdown_guard_threshold: float = 0.15,
        bear_ma200_penalty: float = 0.10,
    ) -> None:
        self.risk_profile = risk_profile or InvestorRiskProfile.balanced()
        self.strategy_id = strategy_id
        self.ticker_selector = ticker_selector or TickerSelector(
            risk_profile=self.risk_profile,
            strategy_id=strategy_id,
        )
        self.size_selector = size_selector or SizeSelector(
            risk_profile=self.risk_profile,
        )
        self.defensive_strategy = defensive_strategy
        self.max_drawdown_guard_threshold = max_drawdown_guard_threshold
        self.bear_ma200_penalty = bear_ma200_penalty

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        action_type: DSSDecisionAction,
        features: pd.DataFrame,
        portfolio: PortfolioState,
        decision_date: str,
        visible_data_cutoff: Optional[str] = None,
        iqn_model_run_id: Optional[str] = None,
        iqn_config: Optional[dict] = None,
        score_mode: Optional[str] = None,
        stage_1_action_type_scores: Optional[dict] = None,
    ) -> HierarchicalDecision:
        """
        Run the full 5-stage pipeline for one decision step.

        Parameters
        ----------
        action_type : DSSDecisionAction
            The action type from IQN or forced (smoke test).
        features : pd.DataFrame
            One row per ticker with technical + fundamental scores.
        portfolio : PortfolioState
            Current portfolio snapshot.
        decision_date : str
            ISO date string (YYYY-MM-DD).
        visible_data_cutoff : str, optional
            PIT cutoff for audit. Defaults to decision_date.
        stage_1_action_type_scores : dict, optional
            Scores per action from IQN (HOLD/BUY/SELL/REBALANCE). Mocked if None.
        """
        decision_id = str(uuid.uuid4())[:12]
        cutoff = visible_data_cutoff or decision_date
        action_label = action_to_label(action_type)

        dec = HierarchicalDecision(
            decision_id=decision_id,
            date=decision_date,
            visible_data_cutoff=cutoff,
            portfolio_state=portfolio.to_dict(),
            strategy_id=self.strategy_id,
            risk_profile=self.risk_profile.to_dict(),
            iqn_model_run_id=iqn_model_run_id,
            iqn_config=iqn_config,
            score_mode=score_mode,
            stage_1_action_type_scores=stage_1_action_type_scores
            or self._mock_stage1_scores(action_label),
            selected_action_type=action_label,
        )

        # --- Stage 4 (pre-check): risk/strategy validation ---
        risk = self._validate_risk(action_type, features, portfolio)
        dec.risk_checks = risk.to_dict()

        # Override action to HOLD if hard constraint violated
        effective_action = action_type
        if action_type == DSSDecisionAction.BUY and not risk.no_buy_without_cash_ok:
            effective_action = DSSDecisionAction.HOLD
            dec.selected_action_type = "HOLD"
            risk.warnings.append("BUY overridden to HOLD — insufficient cash.")
        elif (
            action_type == DSSDecisionAction.SELL
            and not risk.no_sell_without_holdings_ok
        ):
            effective_action = DSSDecisionAction.HOLD
            dec.selected_action_type = "HOLD"
            risk.warnings.append("SELL overridden to HOLD — no holdings.")

        # --- Stage 2: ticker selection ---
        selected_ticker = None
        ticker_score_rows: list[TickerScoreRow] = []

        if effective_action == DSSDecisionAction.BUY:
            selected_ticker, ticker_score_rows = self.ticker_selector.select_buy_ticker(
                features=features,
                portfolio_weights=portfolio.position_weights,
                cash_weight=portfolio.cash_weight,
                decision_date=decision_date,
                bear_market_penalty=risk.bear_market_penalty,
            )
            if selected_ticker is None:
                effective_action = DSSDecisionAction.HOLD
                dec.selected_action_type = "HOLD"

        elif effective_action == DSSDecisionAction.SELL:
            selected_ticker, ticker_score_rows = (
                self.ticker_selector.select_sell_ticker(
                    features=features,
                    portfolio_weights=portfolio.position_weights,
                    decision_date=decision_date,
                )
            )
            if selected_ticker is None:
                effective_action = DSSDecisionAction.HOLD
                dec.selected_action_type = "HOLD"

        dec.selected_ticker = selected_ticker
        dec.stage_2_ticker_scores = [r.to_dict() for r in ticker_score_rows]

        # --- Stage 3: size selection ---
        size_result: Optional[SizeResult] = None
        if selected_ticker is not None:
            size_result = self._select_size(
                action_type=effective_action,
                ticker=selected_ticker,
                features=features,
                portfolio=portfolio,
                risk=risk,
            )
            dec.selected_size = size_result.selected_size
            dec.selected_fraction = size_result.selected_fraction
            dec.risk_adjusted_allocation_fraction = (
                size_result.risk_adjusted_allocation_fraction
            )
            dec.stage_3_size_scores = [r.to_dict() for r in size_result.score_rows]

        # --- Stage 5: final recommendation and audit ---
        dec.final_recommendation = self._build_recommendation(
            effective_action, selected_ticker, size_result, risk, portfolio
        )

        dec.execution_result = self._build_execution_result(
            effective_action, selected_ticker, size_result, portfolio
        )

        return dec

    # ------------------------------------------------------------------
    # Stage 4: Risk validation
    # ------------------------------------------------------------------

    def _validate_risk(
        self,
        action_type: DSSDecisionAction,
        features: pd.DataFrame,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        r = RiskCheckResult()
        weights = portfolio.position_weights

        # --- Hard constraints ---
        if action_type == DSSDecisionAction.BUY:
            if portfolio.cash_weight < self.risk_profile.min_cash_weight:
                r.cash_buffer_ok = False
                r.no_buy_without_cash_ok = False
                r.warnings.append(
                    f"Insufficient cash for BUY: {portfolio.cash_weight:.2%} < {self.risk_profile.min_cash_weight:.2%}"
                )

        if action_type == DSSDecisionAction.SELL:
            has_holdings = any(v > 0.001 for v in portfolio.holding_values.values())
            if not has_holdings:
                r.no_sell_without_holdings_ok = False
                r.warnings.append("SELL requested but no holdings in portfolio.")

        # Concentration check — any position over max?
        for ticker, w in weights.items():
            if w > self.risk_profile.max_position_weight * 1.1:
                r.concentration_ok = False
                r.warnings.append(f"Concentration warning: {ticker} at {w:.2%}")
                break

        # --- Bear-market / trend guard ---
        tic_col = "tic" if "tic" in features.columns else "ticker"
        bear_signals = 0
        n_checked = 0

        for _, feat in features.iterrows():
            n_checked += 1
            price_vs_ma200 = feat.get("price_vs_ma200", None)
            recent_return = feat.get("recent_return", None)
            drawdown = feat.get("drawdown_from_recent_high", None)
            momentum = feat.get("momentum_score", None)

            if (
                price_vs_ma200 is not None
                and not pd.isna(price_vs_ma200)
                and float(price_vs_ma200) < -0.05
            ):
                bear_signals += 1
            if (
                recent_return is not None
                and not pd.isna(recent_return)
                and float(recent_return) < -0.10
            ):
                bear_signals += 1
            if (
                drawdown is not None
                and not pd.isna(drawdown)
                and float(drawdown) < -0.15
            ):
                bear_signals += 1

        if n_checked > 0 and bear_signals >= n_checked:
            r.bear_market_signal = True
            r.ma200_trend_ok = False
            r.bear_market_penalty = self.bear_ma200_penalty
            r.warnings.append(
                f"Bear-market guard triggered: {bear_signals} signals across {n_checked} tickers."
            )
            if self.defensive_strategy:
                r.warnings.append(
                    "Defensive strategy: HOLD may be forced by size_selector."
                )

        # --- Portfolio-level drawdown guard ---
        # (simplified: checked externally if portfolio_drawdown is passed)

        return r

    # ------------------------------------------------------------------
    # Stage 3 wrapper
    # ------------------------------------------------------------------

    def _select_size(
        self,
        action_type: DSSDecisionAction,
        ticker: str,
        features: pd.DataFrame,
        portfolio: PortfolioState,
        risk: RiskCheckResult,
    ) -> SizeResult:
        tic_col = "tic" if "tic" in features.columns else "ticker"
        ticker_feat = features[features[tic_col] == ticker]

        volatility_score = 0.3
        drawdown_from_high = 0.0
        ticker_score = 0.5
        current_weight = portfolio.position_weights.get(ticker, 0.0)

        if not ticker_feat.empty:
            row = ticker_feat.iloc[0]
            volatility_score = float(row.get("volatility_score", 0.3) or 0.3)
            drawdown_from_high = float(row.get("drawdown_from_recent_high", 0.0) or 0.0)
            if action_type == DSSDecisionAction.BUY:
                ticker_score = float(row.get("momentum_score", 0.5) or 0.5)
                ticker_score = (ticker_score + 1.0) / 2.0  # normalise [-1,1] → [0,1]

        if action_type == DSSDecisionAction.BUY:
            return self.size_selector.select_buy_size(
                ticker=ticker,
                ticker_score=ticker_score,
                cash_weight=portfolio.cash_weight,
                current_ticker_weight=current_weight,
                volatility_score=volatility_score,
                drawdown_from_high=drawdown_from_high,
                portfolio_value=portfolio.total_value,
                bear_market_signal=risk.bear_market_signal,
                defensive_strategy=self.defensive_strategy,
            )
        else:  # SELL
            return self.size_selector.select_sell_size(
                ticker=ticker,
                ticker_keep_score=ticker_score,
                current_ticker_weight=current_weight,
                volatility_score=volatility_score,
                drawdown_from_high=drawdown_from_high,
                portfolio_value=portfolio.total_value,
                bear_market_signal=risk.bear_market_signal,
            )

    # ------------------------------------------------------------------
    # Stage 5 helpers
    # ------------------------------------------------------------------

    def _build_recommendation(
        self,
        action_type: DSSDecisionAction,
        ticker: Optional[str],
        size_result: Optional[SizeResult],
        risk: RiskCheckResult,
        portfolio: PortfolioState,
    ) -> dict:
        action_label = action_to_label(action_type)

        if action_type == DSSDecisionAction.HOLD:
            hold_type = self.ticker_selector.classify_hold(
                portfolio_weights=portfolio.position_weights,
                cash_weight=portfolio.cash_weight,
                hold_reason="IQN_SELECTED" if not risk.warnings else "NO_CANDIDATE",
            )
            explanation = (
                f"Action: HOLD ({hold_type}). "
                f"Portfolio: {portfolio.cash_weight:.1%} cash, "
                f"{1 - portfolio.cash_weight:.1%} equity."
            )
        else:
            size_label = size_result.selected_size if size_result else "N/A"
            alloc_frac = (
                size_result.risk_adjusted_allocation_fraction if size_result else 0.0
            )
            explanation = (
                f"Action: {action_label} {ticker} @ {size_label} "
                f"(effective allocation: {alloc_frac:.1%} of portfolio)."
            )

        return {
            "action_type": action_label,
            "ticker": ticker,
            "size": size_result.selected_size if size_result else None,
            "risk_adjusted_allocation_fraction": (
                size_result.risk_adjusted_allocation_fraction if size_result else 0.0
            ),
            "explanation": explanation,
            "warnings": risk.warnings + (size_result.warnings if size_result else []),
        }

    def _build_execution_result(
        self,
        action_type: DSSDecisionAction,
        ticker: Optional[str],
        size_result: Optional[SizeResult],
        portfolio: PortfolioState,
    ) -> dict:
        """
        Compute estimated execution parameters.
        This is a PoC estimate; actual execution requires live price feeds.
        """
        if (
            action_type == DSSDecisionAction.HOLD
            or ticker is None
            or size_result is None
        ):
            return {
                "requested_shares": 0,
                "submitted_shares": 0,
                "executed_shares": 0,
                "transaction_cost_estimate": 0.0,
                "portfolio_before": portfolio.to_dict(),
                "portfolio_after": portfolio.to_dict(),
            }

        alloc_frac = size_result.risk_adjusted_allocation_fraction
        trade_value = alloc_frac * portfolio.total_value

        # Assume 0.1% transaction cost
        tx_cost = trade_value * 0.001

        if action_type == DSSDecisionAction.BUY:
            new_cash = portfolio.cash - trade_value - tx_cost
            new_holdings = dict(portfolio.holdings)
            new_holding_values = dict(portfolio.holding_values)
            new_holding_values[ticker] = (
                new_holding_values.get(ticker, 0.0) + trade_value
            )
        else:
            liquidation_value = trade_value
            new_cash = portfolio.cash + liquidation_value - tx_cost
            new_holdings = dict(portfolio.holdings)
            new_holding_values = dict(portfolio.holding_values)
            new_holding_values[ticker] = max(
                0.0, new_holding_values.get(ticker, 0.0) - liquidation_value
            )

        after = PortfolioState(
            total_value=portfolio.total_value - tx_cost,
            cash=new_cash,
            holdings=new_holdings,
            holding_values=new_holding_values,
        )

        return {
            "requested_shares": "n/a (PoC — no live price)",
            "submitted_shares": "n/a",
            "executed_shares": "n/a",
            "trade_value_estimate": round(trade_value, 2),
            "transaction_cost_estimate": round(tx_cost, 2),
            "portfolio_before": portfolio.to_dict(),
            "portfolio_after": after.to_dict(),
        }

    @staticmethod
    def _mock_stage1_scores(selected_label: str) -> dict:
        """Generate plausible mock Stage 1 scores when IQN is not available."""
        base = {"HOLD": 0.20, "BUY": 0.20, "SELL": 0.20, "REBALANCE": 0.10}
        base[selected_label] = 0.70
        return base
