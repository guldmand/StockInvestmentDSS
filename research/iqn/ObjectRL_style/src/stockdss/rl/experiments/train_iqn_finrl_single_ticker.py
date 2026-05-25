"""
Train D-IQN-DSS on a single stock using FinRL-style CSV data.

This experiment is the bridge between:
1. CartPole IQN sanity check
2. FinRL-style stock decision support

V1 setup:
- One ticker only, e.g. AAPL
- Discrete action space:
    0 = HOLD
    1 = BUY
    2 = SELL
- Uses train_data.csv
"""

from __future__ import annotations

import random

import numpy as np
import torch

from stockdss.envs.finrl_discrete_env import (
    FinRLDiscreteEnv,
    FinRLDiscreteEnvConfig,
)
from stockdss.rl.agents.iqn_agent import IQNAgent
from stockdss.rl.config.iqn_config import IQNConfig


def train_iqn_finrl_single_ticker() -> None:
    config = IQNConfig(
        env_name="FinRLDiscreteEnv-AAPL-v1",
        total_steps=50_000,
        learning_starts=1_000,
        batch_size=64,
        replay_capacity=100_000,
        gamma=0.99,
        lr=1e-3,
        target_update_interval=500,
        hidden_dim=128,
        num_tau_samples=32,
        num_tau_prime_samples=32,
        num_action_quantiles=32,
        cosine_embedding_dim=64,
        kappa=1.0,
        epsilon_start=1.0,
        epsilon_final=0.05,
        epsilon_decay_steps=25_000,
        log_interval=1_000,
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env_config = FinRLDiscreteEnvConfig(
        csv_path="train_data.csv",
        ticker="AAPL",
        initial_amount=1_000_000.0,
        buy_cost_pct=0.01,
        sell_cost_pct=0.01,
        reward_scaling=1e-4,
    )

    env = FinRLDiscreteEnv(env_config)

    state, info = env.reset(seed=config.seed)

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    agent = IQNAgent(state_dim=state_dim, action_dim=action_dim, config=config)

    episode_reward = 0.0
    episode_rewards: list[float] = []
    losses: list[float] = []

    last_info = info

    separator = "=" * 100

    print(separator)
    print("D-IQN-DSS single-ticker FinRL-style training")
    print(separator)
    print(f"Environment: {config.env_name}")
    print(f"CSV path: {env_config.csv_path}")
    print(f"Ticker: {env_config.ticker}")
    print(f"Initial amount: {env_config.initial_amount:,.2f}")
    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Device: {config.device}")
    print(separator)

    for step in range(1, config.total_steps + 1):
        action = agent.select_action(state, step)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        last_info = info

        if done:
            episode_rewards.append(episode_reward)
            state, info = env.reset()
            episode_reward = 0.0
            last_info = info

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
            mean_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            recent_loss = float(np.mean(losses[-100:])) if losses else 0.0

            print(
                f"step={step:>6} | "
                f"episodes={len(episode_rewards):>4} | "
                f"mean_reward_20={mean_reward:>10.4f} | "
                f"epsilon={agent.epsilon(step):.3f} | "
                f"loss_100={recent_loss:>10.4f} | "
                f"date={last_info.get('date', 'n/a')} | "
                f"action={last_info.get('action_name', 'RESET'):<5} | "
                f"portfolio={last_info.get('portfolio_value', 0.0):>12.2f} | "
                f"cash={last_info.get('cash', 0.0):>12.2f} | "
                f"shares={last_info.get('shares_held', 0):>8}"
            )

    env.close()

    print(separator)
    print("Training done")
    print(f"Episodes: {len(episode_rewards)}")
    if episode_rewards:
        print(
            "Final mean reward over last 20 episodes: "
            f"{np.mean(episode_rewards[-20:]):.4f}"
        )
    print(f"Final portfolio value: {last_info.get('portfolio_value', 0.0):,.2f}")
    print(f"Final cash: {last_info.get('cash', 0.0):,.2f}")
    print(f"Final shares held: {last_info.get('shares_held', 0)}")
    print(separator)


if __name__ == "__main__":
    train_iqn_finrl_single_ticker()
