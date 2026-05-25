"""
D-IQN-DSS minimal CartPole sanity check.

Purpose:
- Implement a minimal PyTorch IQN agent.
- Test it on CartPole-v1, which has a discrete action space.
- Use this as a correctness check before adapting IQN to FinRL-style stock decision support.

This is NOT the final stock trading implementation.
It is the minimal learning engine.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class IQNConfig:
    env_name: str = "CartPole-v1"
    seed: int = 42

    total_steps: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 64
    replay_capacity: int = 100_000

    gamma: float = 0.99
    lr: float = 1e-3
    target_update_interval: int = 500

    hidden_dim: int = 128
    num_tau_samples: int = 32  # N: online quantile samples
    num_tau_prime_samples: int = 32  # N': target quantile samples
    num_action_quantiles: int = 32  # K: samples for action selection
    cosine_embedding_dim: int = 64
    kappa: float = 1.0

    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 25_000

    log_interval: int = 1_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: str):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states_t = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=device
        )
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------
# IQN Network
# ---------------------------------------------------------------------


class IQNNetwork(nn.Module):
    """
    IQN network.

    Input:
        state: shape [batch_size, state_dim]
        taus:  shape [batch_size, num_quantiles]

    Output:
        quantile_values: shape [batch_size, num_quantiles, action_dim]

    Meaning:
        quantile_values[b, q, a] = Z_tau_q(state_b, action_a)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        cosine_embedding_dim: int,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.cosine_embedding_dim = cosine_embedding_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        self.tau_embedding = nn.Sequential(
            nn.Linear(cosine_embedding_dim, hidden_dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        batch_size = states.shape[0]
        num_quantiles = taus.shape[1]

        # Encode state: [B, H]
        state_features = self.state_encoder(states)

        # Cosine embedding of tau:
        # taus: [B, N]
        # i_pi: [1, 1, C]
        i_pi = (
            torch.arange(
                1,
                self.cosine_embedding_dim + 1,
                device=states.device,
                dtype=torch.float32,
            )
            * torch.pi
        ).view(1, 1, -1)

        cosines = torch.cos(taus.unsqueeze(-1) * i_pi)
        # cosines: [B, N, C]

        tau_features = self.tau_embedding(
            cosines.view(batch_size * num_quantiles, self.cosine_embedding_dim)
        )
        # tau_features: [B*N, H]

        state_features = state_features.unsqueeze(1).expand(
            batch_size, num_quantiles, self.hidden_dim
        )
        state_features = state_features.reshape(
            batch_size * num_quantiles, self.hidden_dim
        )

        # Hadamard product, as in IQN-style architecture
        combined = state_features * tau_features

        quantile_values = self.head(combined)
        quantile_values = quantile_values.view(
            batch_size, num_quantiles, self.action_dim
        )

        return quantile_values


# ---------------------------------------------------------------------
# IQN Agent
# ---------------------------------------------------------------------


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
            batch_size=1, num_quantiles=self.config.num_action_quantiles
        )

        quantile_values = self.online_net(state_t, taus)
        # [1, K, action_dim]

        q_values = quantile_values.mean(dim=1)
        # [1, action_dim]

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

        # Current quantile estimates: [B, N, A]
        current_quantiles = self.online_net(states, taus)

        # Gather quantiles for chosen actions: [B, N]
        chosen_quantiles = current_quantiles.gather(
            dim=2,
            index=actions.view(batch_size, 1, 1).expand(
                batch_size, cfg.num_tau_samples, 1
            ),
        ).squeeze(2)

        with torch.no_grad():
            # Double-DQN style action selection:
            # use online net to choose action, target net to evaluate it.
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

        # Pairwise TD-errors:
        # target: [B, N']
        # current: [B, N]
        # td_errors: [B, N', N]
        td_errors = target_quantiles.unsqueeze(2) - chosen_quantiles.unsqueeze(1)

        huber_loss = torch.where(
            td_errors.abs() <= cfg.kappa,
            0.5 * td_errors.pow(2),
            cfg.kappa * (td_errors.abs() - 0.5 * cfg.kappa),
        )

        # taus: [B, N] -> [B, 1, N]
        tau = taus.unsqueeze(1)

        quantile_weights = torch.abs(tau - (td_errors.detach() < 0).float())
        quantile_huber_loss = quantile_weights * huber_loss / cfg.kappa

        # Sum over current quantiles, mean over target quantiles, mean over batch
        loss = quantile_huber_loss.sum(dim=2).mean(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())


# ---------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------


def train_cartpole() -> None:
    config = IQNConfig()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = gym.make(config.env_name)
    state, _ = env.reset(seed=config.seed)

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    agent = IQNAgent(state_dim=state_dim, action_dim=action_dim, config=config)

    episode_reward = 0.0
    episode_rewards = []
    losses = []

    print("=" * 80)
    print("D-IQN-DSS minimal CartPole sanity check")
    print("=" * 80)
    print(f"Environment: {config.env_name}")
    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Device: {config.device}")
    print("=" * 80)

    for step in range(1, config.total_steps + 1):
        action = agent.select_action(state, step)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            episode_rewards.append(episode_reward)
            state, _ = env.reset()
            episode_reward = 0.0

        if (
            step > config.learning_starts
            and len(agent.replay_buffer) >= config.batch_size
        ):
            loss = agent.learn()
            losses.append(loss)

        if step % config.target_update_interval == 0:
            agent.update_target_network()

        if step % config.log_interval == 0:
            recent_rewards = episode_rewards[-20:]
            mean_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            recent_loss = np.mean(losses[-100:]) if losses else 0.0

            print(
                f"step={step:>6} | "
                f"episodes={len(episode_rewards):>4} | "
                f"mean_reward_20={mean_reward:>7.2f} | "
                f"epsilon={agent.epsilon(step):.3f} | "
                f"loss_100={recent_loss:>8.4f}"
            )

            if mean_reward >= 475:
                print("Solved CartPole according to mean_reward_20 >= 475.")
                break

    env.close()

    print("=" * 80)
    print("Training done")
    print(f"Episodes: {len(episode_rewards)}")
    if episode_rewards:
        print(
            f"Final mean reward over last 20 episodes: {np.mean(episode_rewards[-20:]):.2f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    train_cartpole()
