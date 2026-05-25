"""
Train D-IQN-DSS on CartPole-v1.

This is a sanity-check experiment. It verifies that the modular IQN
implementation works before adapting it to FinRL-style stock decision support.
"""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
import torch

from stockdss.rl.agents.iqn_agent import IQNAgent
from stockdss.rl.config.iqn_config import IQNConfig


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
    print("D-IQN-DSS modular CartPole sanity check")
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
            "Final mean reward over last 20 episodes: "
            f"{np.mean(episode_rewards[-20:]):.2f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    train_cartpole()
