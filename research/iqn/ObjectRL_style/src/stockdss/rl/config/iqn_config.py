"""
Configuration for the D-IQN-DSS agent.

This config is first used for the minimal CartPole sanity check,
and later extended/adapted for the FinRL-style stock decision environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
    num_tau_samples: int = 32
    num_tau_prime_samples: int = 32
    num_action_quantiles: int = 32
    cosine_embedding_dim: int = 64
    kappa: float = 1.0

    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 25_000

    log_interval: int = 1_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
