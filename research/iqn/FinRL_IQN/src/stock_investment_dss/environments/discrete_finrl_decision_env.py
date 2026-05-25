# src/stock_investment_dss/environments/discrete_finrl_decision_env.py

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stock_investment_dss.decision.action_mask import DSSActionMaskGenerator
from stock_investment_dss.decision.decision_actions import (
    DSSDecisionAction,
    action_to_label,
)
from stock_investment_dss.decision.risk_aware_action_resolver import (
    RiskAwareActionResolver,
)
from stock_investment_dss.environments.finrl_env_factory import (
    extract_finrl_state_summary,
    unpack_reset_result,
    unpack_step_result,
)


class DiscreteFinRLDecisionEnv(gym.Env):
    """
    Smart DSS adapter around FinRL StockTradingEnv.

    The IQN/DSS sees a discrete decision action space:
        HOLD, BUY, SELL, REBALANCE, CHANGE_STRATEGY

    FinRL still receives a continuous action vector and performs:
        cash update, holdings update, transaction costs, rewards, memories.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        finrl_env,
        tickers: list[str] | tuple[str, ...],
        resolver: RiskAwareActionResolver,
        action_mask_generator: DSSActionMaskGenerator | None = None,
        enforce_action_mask: bool = True,
    ) -> None:
        super().__init__()

        self.finrl_env = finrl_env
        self.tickers = [ticker.upper().strip() for ticker in tickers]
        self.resolver = resolver
        self.action_mask_generator = action_mask_generator
        self.enforce_action_mask = enforce_action_mask

        self.action_space = spaces.Discrete(len(DSSDecisionAction))
        self.observation_space = self.finrl_env.observation_space

        self.current_observation = None
        self.last_info: dict[str, Any] = {}
        self.decision_memory: list[dict] = []

    def reset(self, *, seed=None, options=None):
        reset_result = self.finrl_env.reset()
        observation, info = unpack_reset_result(reset_result)

        self.current_observation = observation
        self.decision_memory = []

        self.last_info = {
            "source_env": "FinRL StockTradingEnv",
            "adapter": "DiscreteFinRLDecisionEnv",
            "reset_info": info,
        }

        return observation, self.last_info

    def get_action_mask(self) -> dict | None:
        if self.current_observation is None or self.action_mask_generator is None:
            return None

        state_summary = extract_finrl_state_summary(
            state=self.current_observation,
            tickers=self.tickers,
        )

        action_mask = self.action_mask_generator.generate(state_summary)

        return {
            "allowed_actions": action_mask.allowed_actions,
            "blocked_reasons": action_mask.blocked_reasons,
            "mask_vector": action_mask.mask_vector,
        }

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid DSS discrete action: {action}")

        if self.current_observation is None:
            raise RuntimeError("Environment must be reset before calling step().")

        requested_decision_action = DSSDecisionAction(action)
        effective_decision_action = requested_decision_action
        action_was_masked = False
        mask_snapshot = None

        state_before = extract_finrl_state_summary(
            state=self.current_observation,
            tickers=self.tickers,
        )

        if self.action_mask_generator is not None:
            action_mask = self.action_mask_generator.generate(state_before)
            mask_snapshot = {
                "allowed_actions": action_mask.allowed_actions,
                "blocked_reasons": action_mask.blocked_reasons,
                "mask_vector": action_mask.mask_vector,
            }

            if self.enforce_action_mask and not action_mask.is_allowed(
                requested_decision_action
            ):
                action_was_masked = True
                resolved_action = self.resolver.resolve_blocked_action(
                    requested_decision_action=requested_decision_action,
                    state_summary=state_before,
                    blocked_reason=action_mask.blocked_reasons.get(
                        requested_decision_action.name,
                        "Action is not allowed in the current state.",
                    ),
                )
                effective_decision_action = DSSDecisionAction.HOLD
            else:
                resolved_action = self.resolver.resolve(
                    decision_action=effective_decision_action,
                    state_summary=state_before,
                )
        else:
            resolved_action = self.resolver.resolve(
                decision_action=effective_decision_action,
                state_summary=state_before,
            )

        continuous_action = np.array(
            resolved_action.continuous_action,
            dtype=float,
        )

        cost_before = float(getattr(self.finrl_env, "cost", 0.0))
        trades_before = int(getattr(self.finrl_env, "trades", 0))

        step_result = self.finrl_env.step(continuous_action)
        observation, reward, done, info = unpack_step_result(step_result)

        state_after = extract_finrl_state_summary(
            state=observation,
            tickers=self.tickers,
        )

        cost_after = float(getattr(self.finrl_env, "cost", 0.0))
        trades_after = int(getattr(self.finrl_env, "trades", 0))

        self.current_observation = observation

        executed_holdings_delta = {
            ticker: float(state_after["holdings"].get(ticker, 0.0))
            - float(state_before["holdings"].get(ticker, 0.0))
            for ticker in self.tickers
        }

        selected_ticker = resolved_action.selected_ticker
        executed_shares_delta = (
            executed_holdings_delta.get(selected_ticker, 0.0)
            if selected_ticker is not None
            else 0.0
        )

        cash_delta = float(state_after["cash"]) - float(state_before["cash"])
        portfolio_value_delta = float(state_after["portfolio_value"]) - float(
            state_before["portfolio_value"]
        )

        cost_delta = cost_after - cost_before
        trades_delta = trades_after - trades_before

        decision_record = {
            "decision_step": len(self.decision_memory),
            "requested_decision_action_index": int(requested_decision_action),
            "requested_decision_action_label": action_to_label(
                requested_decision_action
            ),
            "effective_decision_action_index": int(effective_decision_action),
            "effective_decision_action_label": action_to_label(
                effective_decision_action
            ),
            "action_was_masked": action_was_masked,
            "action_mask": mask_snapshot,
            "resolved_action": {
                "decision_action": resolved_action.decision_action,
                "selected_ticker": resolved_action.selected_ticker,
                "requested_shares": resolved_action.requested_shares,
                "requested_cash_value": resolved_action.requested_cash_value,
                "submitted_shares_estimate": resolved_action.submitted_shares_estimate,
                "submitted_cash_value_estimate": resolved_action.submitted_cash_value_estimate,
                "hmax_limited": resolved_action.hmax_limited,
                "continuous_action": resolved_action.continuous_action,
                "reason": resolved_action.reason,
                "constraints": resolved_action.constraints,
                "metadata": resolved_action.metadata,
            },
            "execution_delta": {
                "executed_holdings_delta": executed_holdings_delta,
                "executed_shares_delta": executed_shares_delta,
                "cash_delta": cash_delta,
                "portfolio_value_delta": portfolio_value_delta,
                "cost_delta": cost_delta,
                "trades_delta": trades_delta,
            },
            "state_before": state_before,
            "state_after": state_after,
            "reward": float(reward),
            "done": bool(done),
            "finrl_info": info,
            "finrl_cost": cost_after,
            "finrl_trades": trades_after,
        }

        self.decision_memory.append(decision_record)

        enriched_info = {
            **info,
            "source_env": "FinRL StockTradingEnv",
            "adapter": "DiscreteFinRLDecisionEnv",
            "decision_record": decision_record,
        }

        terminated = bool(done)
        truncated = False

        return observation, reward, terminated, truncated, enriched_info

    def save_asset_memory(self):
        return self.finrl_env.save_asset_memory()

    def save_action_memory(self):
        return self.finrl_env.save_action_memory()

    def save_decision_memory(self) -> list[dict]:
        return self.decision_memory

    @property
    def cost(self):
        return getattr(self.finrl_env, "cost", 0.0)

    @property
    def trades(self):
        return getattr(self.finrl_env, "trades", 0)
