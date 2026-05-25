"""
Configuration for the D-IQN-DSS agent.

This config is first used for the minimal CartPole sanity check,
and later extended/adapted for the FinRL-style stock decision environment.

Seed handling:
- Default seed is still 42 for reproducible single-run smoke tests.
- Multi-seed runners can override the seed through environment variables.
- Preferred env vars:
    STOCK_INVESTMENT_DSS_IQN_SEED
    STOCK_INVESTMENT_DSS_RANDOM_SEED
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch


def _get_int_environment_variable(name: str, default: int) -> int:
    """Read an integer environment variable with a safe fallback."""

    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return int(value)
    except ValueError:
        return default


def _default_seed() -> int:
    """
    Resolve the default IQN seed.

    STOCK_INVESTMENT_DSS_IQN_SEED is the most specific override.
    STOCK_INVESTMENT_DSS_RANDOM_SEED is the shared project-wide fallback.
    If neither is set, use 42 for reproducible single-run experiments.
    """

    project_seed = _get_int_environment_variable(
        "STOCK_INVESTMENT_DSS_RANDOM_SEED",
        default=42,
    )

    return _get_int_environment_variable(
        "STOCK_INVESTMENT_DSS_IQN_SEED",
        default=project_seed,
    )


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class IQNConfig:
    env_name: str = "D-IQN-DSS-FinRL"
    seed: int = field(default_factory=_default_seed)

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
    device: str = field(default_factory=_default_device)
