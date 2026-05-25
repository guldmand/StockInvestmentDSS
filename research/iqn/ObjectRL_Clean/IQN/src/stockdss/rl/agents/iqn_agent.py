"""
D-IQN-DSS IQN agent.

This agent implements the core IQN learning logic:
- epsilon-greedy action selection
- online and target networks
- replay buffer sampling
- Double-DQN-style target action selection
- quantile Huber loss
"""

from __future__ import annotations

import random

import numpy as np
import torch

from stockdss.rl.config.iqn_config import IQNConfig
from stockdss.rl.nets.iqn_net import IQNNetwork
from stockdss.rl.replay_buffers.replay_buffer import ReplayBuffer


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

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.lr)
        self.replay_buffer = ReplayBuffer(config.replay_capacity)

    def sample_taus(self, batch_size: int, num_quantiles: int) -> torch.Tensor:
        return torch.rand(batch_size, num_quantiles, device=self.device)

    def epsilon(self, step: int) -> float:
        cfg = self.config

        if step >= cfg.epsilon_decay_steps:
            return cfg.epsilon_final

        fraction = step / cfg.epsilon_decay_steps
        return cfg.epsilon_start + fraction * (cfg.epsilon_final - cfg.epsilon_start)

    @torch.no_grad()
    def select_action(
        self, state: np.ndarray, step: int, eval_mode: bool = False
    ) -> int:
        eps = 0.0 if eval_mode else self.epsilon(step)

        if random.random() < eps:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        taus = self.sample_taus(
            batch_size=1,
            num_quantiles=self.config.num_action_quantiles,
        )

        quantile_values = self.online_net(state_t, taus)
        q_values = quantile_values.mean(dim=1)

        return int(q_values.argmax(dim=1).item())

    def learn(self) -> float:
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
            next_action_quantiles = self.online_net(next_states, action_taus)
            next_q_values = next_action_quantiles.mean(dim=1)
            next_actions = next_q_values.argmax(dim=1)

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
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())
