# src/stock_investment_dss/rl/agents/iqn_agent.py

"""
D-IQN-DSS IQN agent.

This agent keeps the core IQN principles:

- sample τ ~ U([0, 1])
- estimate Z_τ(s, a) for each discrete action
- use mean over quantile samples for greedy action selection
- use online network + target network
- use Double-DQN-style target action selection
- train with quantile Huber loss

The implementation is adapted from the V1 StockDSS IQN agent, but modified
for the V2 DiscreteFinRLDecisionEnv:

Actions:
    0 = HOLD
    1 = BUY
    2 = SELL
    3 = REBALANCE
    4 = CHANGE_STRATEGY
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from stock_investment_dss.rl.config.iqn_config import IQNConfig
from stock_investment_dss.rl.nets.iqn_net import IQNNetwork
from stock_investment_dss.rl.replay_buffers.replay_buffer import ReplayBuffer


class IQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: IQNConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = config.device

        self.online_net = IQNNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            cosine_embedding_dim=config.cosine_embedding_dim,
        ).to(self.device)

        self.target_net = IQNNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            cosine_embedding_dim=config.cosine_embedding_dim,
        ).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=config.lr,
        )

        self.replay_buffer = ReplayBuffer(config.replay_capacity)

    def sample_taus(self, batch_size: int, num_quantiles: int) -> torch.Tensor:
        """
        IQN samples τ from U([0, 1]).

        Shape:
            [batch_size, num_quantiles]
        """
        return torch.rand(batch_size, num_quantiles, device=self.device)

    def epsilon(self, step: int) -> float:
        cfg = self.config

        if step >= cfg.epsilon_decay_steps:
            return cfg.epsilon_final

        fraction = step / cfg.epsilon_decay_steps
        return cfg.epsilon_start + fraction * (cfg.epsilon_final - cfg.epsilon_start)

    def _mask_q_values(
        self,
        q_values: torch.Tensor,
        action_mask: list[int] | np.ndarray | None,
    ) -> torch.Tensor:
        """
        Masks invalid actions by setting their Q-values to a very negative value.

        q_values:
            shape [batch_size, action_dim]

        action_mask:
            1 = allowed
            0 = blocked
        """
        if action_mask is None:
            return q_values

        mask = torch.tensor(
            np.asarray(action_mask, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1)

        if mask.shape[1] != self.action_dim:
            raise ValueError(
                f"Action mask length {mask.shape[1]} does not match "
                f"action_dim {self.action_dim}."
            )

        masked_q_values = q_values.clone()
        masked_q_values = masked_q_values.masked_fill(mask <= 0, -1e9)

        return masked_q_values

    def _allowed_action_indices(
        self,
        action_mask: list[int] | np.ndarray | None,
    ) -> list[int]:
        if action_mask is None:
            return list(range(self.action_dim))

        mask = np.asarray(action_mask, dtype=np.int64).reshape(-1)

        allowed = [int(index) for index, value in enumerate(mask) if value == 1]

        if not allowed:
            return [0]  # safe fallback: HOLD

        return allowed

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        step: int,
        eval_mode: bool = False,
        action_mask: list[int] | np.ndarray | None = None,
    ) -> int:
        """
        Selects an action using epsilon-greedy over IQN mean quantile values.

        During exploration:
            sample uniformly among allowed actions.

        During exploitation:
            choose argmax_a mean_k Z_{τ_k}(s, a), masked by action_mask.
        """
        eps = 0.0 if eval_mode else self.epsilon(step)
        allowed_actions = self._allowed_action_indices(action_mask)

        if random.random() < eps:
            return int(random.choice(allowed_actions))

        state_t = torch.tensor(
            np.asarray(state, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        taus = self.sample_taus(
            batch_size=1,
            num_quantiles=self.config.num_action_quantiles,
        )

        quantile_values = self.online_net(state_t, taus)
        q_values = quantile_values.mean(dim=1)
        q_values = self._mask_q_values(q_values, action_mask)

        return int(q_values.argmax(dim=1).item())

    def learn(self) -> float:
        """
        One IQN training step.

        Uses:
        - current quantiles Z_τ(s, a)
        - target quantiles r + γ Z_τ'(s', argmax_a mean Z(s', a))
        - quantile Huber loss
        """
        cfg = self.config

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            cfg.batch_size,
            self.device,
        )

        batch_size = states.shape[0]

        taus = self.sample_taus(batch_size, cfg.num_tau_samples)
        tau_primes = self.sample_taus(batch_size, cfg.num_tau_prime_samples)
        action_taus = self.sample_taus(batch_size, cfg.num_action_quantiles)

        current_quantiles = self.online_net(states, taus)

        chosen_quantiles = current_quantiles.gather(
            dim=2,
            index=actions.view(batch_size, 1, 1).expand(
                batch_size,
                cfg.num_tau_samples,
                1,
            ),
        ).squeeze(2)

        with torch.no_grad():
            # Double-DQN-style action selection:
            # online network chooses next action from mean quantile values.
            next_action_quantiles = self.online_net(next_states, action_taus)
            next_q_values = next_action_quantiles.mean(dim=1)
            next_actions = next_q_values.argmax(dim=1)

            # target network evaluates selected next action distribution.
            next_target_quantiles = self.target_net(next_states, tau_primes)
            next_chosen_target_quantiles = next_target_quantiles.gather(
                dim=2,
                index=next_actions.view(batch_size, 1, 1).expand(
                    batch_size,
                    cfg.num_tau_prime_samples,
                    1,
                ),
            ).squeeze(2)

            target_quantiles = (
                rewards.unsqueeze(1)
                + (1.0 - dones.unsqueeze(1)) * cfg.gamma * next_chosen_target_quantiles
            )

        td_errors = target_quantiles.unsqueeze(2) - chosen_quantiles.unsqueeze(1)

        huber_loss = torch.where(
            td_errors.abs() <= cfg.kappa,
            0.5 * td_errors.pow(2),
            cfg.kappa * (td_errors.abs() - 0.5 * cfg.kappa),
        )

        tau = taus.unsqueeze(1)

        quantile_weights = torch.abs(tau - (td_errors.detach() < 0).float())

        quantile_huber_loss = quantile_weights * huber_loss / cfg.kappa

        loss = quantile_huber_loss.sum(dim=2).mean(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(),
            max_norm=10.0,
        )
        self.optimizer.step()

        return float(loss.item())

    @torch.no_grad()
    def estimate_action_distributions(
        self,
        state: np.ndarray,
        num_quantiles: int = 128,
        action_mask: list[int] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Estimates the learned return distribution per discrete DSS action.

        Returns q10/q25/q50/q75/q90, mean and CVaR10 per action.

        This is the first DSS-facing output of IQN:
            IQN makes future return distributions decision-visible.
        """
        self.online_net.eval()

        state_t = torch.tensor(
            np.asarray(state, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        taus = self.sample_taus(
            batch_size=1,
            num_quantiles=num_quantiles,
        )

        quantile_values = self.online_net(state_t, taus)
        quantile_values_np = quantile_values.squeeze(0).detach().cpu().numpy()

        mask = (
            np.ones(self.action_dim, dtype=np.int64)
            if action_mask is None
            else np.asarray(action_mask, dtype=np.int64).reshape(-1)
        )

        action_labels = {
            0: "HOLD",
            1: "BUY",
            2: "SELL",
            3: "REBALANCE",
            4: "CHANGE_STRATEGY",
        }

        distributions: dict[str, Any] = {}

        for action_index in range(self.action_dim):
            samples = quantile_values_np[:, action_index]
            samples_sorted = np.sort(samples)

            cvar_cutoff = max(1, int(0.10 * len(samples_sorted)))
            cvar10 = float(np.mean(samples_sorted[:cvar_cutoff]))

            label = action_labels.get(action_index, f"ACTION_{action_index}")

            distributions[label] = {
                "action_index": action_index,
                "allowed": bool(mask[action_index] == 1),
                "mean": float(np.mean(samples)),
                "q10": float(np.quantile(samples, 0.10)),
                "q25": float(np.quantile(samples, 0.25)),
                "q50": float(np.quantile(samples, 0.50)),
                "q75": float(np.quantile(samples, 0.75)),
                "q90": float(np.quantile(samples, 0.90)),
                "cvar10": cvar10,
                "num_quantiles": int(num_quantiles),
            }

        q_values = quantile_values.mean(dim=1)
        masked_q_values = self._mask_q_values(q_values, action_mask)
        selected_action = int(masked_q_values.argmax(dim=1).item())

        return {
            "num_quantiles": int(num_quantiles),
            "selected_action_index": selected_action,
            "selected_action_label": action_labels.get(
                selected_action,
                f"ACTION_{selected_action}",
            ),
            "distributions": distributions,
        }

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "config": self.config,
                "online_net_state_dict": self.online_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
