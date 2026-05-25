"""
Configuration for the D-IQN-DSS IQN agent.

Design goals:
- Keep safe defaults for reproducible smoke tests.
- Make all important IQN/RL hyperparameters configurable through environment
  variables so long experiments are reproducible and not silently hardcoded.
- Preserve the core IQN parameters used in Dopamine-style IQN:
  quantile samples, target quantile samples, action quantiles, cosine/quantile
  embedding dimension, kappa, gamma, target network update, epsilon-greedy
  exploration, replay buffer and Adam optimizer.

Environment variable convention:
- Shared project seed:
    STOCK_INVESTMENT_DSS_RANDOM_SEED
- IQN-specific seed override:
    STOCK_INVESTMENT_DSS_IQN_SEED
- IQN hyperparameters:
    STOCK_INVESTMENT_DSS_IQN_<NAME>

Examples:
    STOCK_INVESTMENT_DSS_IQN_LR=0.0001
    STOCK_INVESTMENT_DSS_IQN_BATCH_SIZE=64
    STOCK_INVESTMENT_DSS_IQN_NUM_TAU_SAMPLES=32
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import torch

TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value if value != "" else None


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    return default


def _env_str(name: str, default: str) -> str:
    value = _env(name)
    return default if value is None else value


def _default_seed() -> int:
    """
    Resolve IQN seed.

    STOCK_INVESTMENT_DSS_IQN_SEED is the most specific override.
    STOCK_INVESTMENT_DSS_RANDOM_SEED is the shared project-wide fallback.
    If neither is set, use 42 for reproducible single-run experiments.
    """

    project_seed = _env_int("STOCK_INVESTMENT_DSS_RANDOM_SEED", default=42)
    return _env_int("STOCK_INVESTMENT_DSS_IQN_SEED", default=project_seed)


def _default_device() -> str:
    requested = _env("STOCK_INVESTMENT_DSS_IQN_DEVICE")
    if requested is not None:
        requested = requested.lower()
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class IQNConfig:
    """Config object for the D-IQN-DSS IQN implementation."""

    # Identity / reproducibility
    env_name: str = field(
        default_factory=lambda: _env_str(
            "STOCK_INVESTMENT_DSS_IQN_ENV_NAME",
            "D-IQN-DSS-FinRL",
        )
    )
    seed: int = field(default_factory=_default_seed)
    config_preset: str = field(
        default_factory=lambda: _env_str(
            "STOCK_INVESTMENT_DSS_IQN_CONFIG_PRESET",
            "stockdss_default",
        )
    )

    # Training loop
    total_steps: int = field(
        default_factory=lambda: _env_int("STOCK_INVESTMENT_DSS_IQN_TOTAL_STEPS", 50_000)
    )
    learning_starts: int = field(
        default_factory=lambda: _env_int(
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_STARTS", 1_000
        )
    )
    batch_size: int = field(
        default_factory=lambda: _env_int("STOCK_INVESTMENT_DSS_IQN_BATCH_SIZE", 64)
    )
    replay_capacity: int = field(
        default_factory=lambda: _env_int(
            "STOCK_INVESTMENT_DSS_IQN_REPLAY_CAPACITY", 100_000
        )
    )

    # RL objective / optimizer
    gamma: float = field(
        default_factory=lambda: _env_float("STOCK_INVESTMENT_DSS_IQN_GAMMA", 0.99)
    )
    lr: float = field(
        default_factory=lambda: _env_float("STOCK_INVESTMENT_DSS_IQN_LR", 1e-3)
    )
    optimizer_name: str = field(
        default_factory=lambda: _env_str("STOCK_INVESTMENT_DSS_IQN_OPTIMIZER", "adam")
    )
    target_update_interval: int = field(
        default_factory=lambda: _env_int(
            "STOCK_INVESTMENT_DSS_IQN_TARGET_UPDATE_INTERVAL",
            500,
        )
    )
    double_dqn: bool = field(
        default_factory=lambda: _env_bool("STOCK_INVESTMENT_DSS_IQN_DOUBLE_DQN", False)
    )

    # Network
    hidden_dim: int = field(
        default_factory=lambda: _env_int("STOCK_INVESTMENT_DSS_IQN_HIDDEN_DIM", 128)
    )
    activation_name: str = field(
        default_factory=lambda: _env_str("STOCK_INVESTMENT_DSS_IQN_ACTIVATION", "relu")
    )

    # IQN-specific parameters.
    # Naming maps to Dopamine approximately as:
    #   num_tau_samples       -> num_tau_samples
    #   num_tau_prime_samples -> num_tau_prime_samples
    #   num_action_quantiles  -> num_quantile_samples
    #   cosine_embedding_dim  -> quantile_embedding_dim
    num_tau_samples: int = field(
        default_factory=lambda: _env_int("STOCK_INVESTMENT_DSS_IQN_NUM_TAU_SAMPLES", 32)
    )
    num_tau_prime_samples: int = field(
        default_factory=lambda: _env_int(
            "STOCK_INVESTMENT_DSS_IQN_NUM_TAU_PRIME_SAMPLES",
            32,
        )
    )
    num_action_quantiles: int = field(
        default_factory=lambda: _env_int(
            "STOCK_INVESTMENT_DSS_IQN_NUM_ACTION_QUANTILES",
            32,
        )
    )
    cosine_embedding_dim: int = field(
        default_factory=lambda: _env_int(
            "STOCK_INVESTMENT_DSS_IQN_COSINE_EMBEDDING_DIM",
            64,
        )
    )
    kappa: float = field(
        default_factory=lambda: _env_float("STOCK_INVESTMENT_DSS_IQN_KAPPA", 1.0)
    )
    grad_clip_norm: float = field(
        default_factory=lambda: _env_float(
            "STOCK_INVESTMENT_DSS_IQN_GRAD_CLIP_NORM", 10.0
        )
    )
    state_norm_scale: float = field(
        default_factory=lambda: _env_float(
            "STOCK_INVESTMENT_DSS_IQN_STATE_NORM_SCALE", 1.0
        )
    )

    # Exploration schedule
    epsilon_start: float = field(
        default_factory=lambda: _env_float(
            "STOCK_INVESTMENT_DSS_IQN_EPSILON_START", 1.0
        )
    )
    epsilon_final: float = field(
        default_factory=lambda: _env_float(
            "STOCK_INVESTMENT_DSS_IQN_EPSILON_FINAL", 0.05
        )
    )
    epsilon_eval: float = field(
        default_factory=lambda: _env_float("STOCK_INVESTMENT_DSS_IQN_EPSILON_EVAL", 0.0)
    )
    epsilon_decay_steps: int = field(
        default_factory=lambda: _env_int(
            "STOCK_INVESTMENT_DSS_IQN_EPSILON_DECAY_STEPS",
            25_000,
        )
    )

    # Logging / runtime
    log_interval: int = field(
        default_factory=lambda: _env_int("STOCK_INVESTMENT_DSS_IQN_LOG_INTERVAL", 1_000)
    )
    device: str = field(default_factory=_default_device)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable config snapshot for audit logs."""

        return asdict(self)

    @classmethod
    def dopamine_jax_like(cls) -> "IQNConfig":
        """
        Return a Dopamine-JAX-inspired IQN config.

        This is a reference preset, not necessarily optimal for StockDSS.
        Dopamine's defaults are designed for Atari-scale discrete RL, while
        StockDSS uses smaller financial decision environments and shorter PIT
        windows.
        """

        return cls(
            config_preset="dopamine_jax_like",
            lr=_env_float("STOCK_INVESTMENT_DSS_IQN_LR", 5e-5),
            batch_size=_env_int("STOCK_INVESTMENT_DSS_IQN_BATCH_SIZE", 32),
            replay_capacity=_env_int(
                "STOCK_INVESTMENT_DSS_IQN_REPLAY_CAPACITY",
                1_000_000,
            ),
            learning_starts=_env_int(
                "STOCK_INVESTMENT_DSS_IQN_LEARNING_STARTS",
                20_000,
            ),
            target_update_interval=_env_int(
                "STOCK_INVESTMENT_DSS_IQN_TARGET_UPDATE_INTERVAL",
                8_000,
            ),
            num_tau_samples=_env_int("STOCK_INVESTMENT_DSS_IQN_NUM_TAU_SAMPLES", 32),
            num_tau_prime_samples=_env_int(
                "STOCK_INVESTMENT_DSS_IQN_NUM_TAU_PRIME_SAMPLES",
                32,
            ),
            num_action_quantiles=_env_int(
                "STOCK_INVESTMENT_DSS_IQN_NUM_ACTION_QUANTILES",
                32,
            ),
            cosine_embedding_dim=_env_int(
                "STOCK_INVESTMENT_DSS_IQN_COSINE_EMBEDDING_DIM",
                64,
            ),
            epsilon_start=_env_float("STOCK_INVESTMENT_DSS_IQN_EPSILON_START", 1.0),
            epsilon_final=_env_float("STOCK_INVESTMENT_DSS_IQN_EPSILON_FINAL", 0.01),
            epsilon_eval=_env_float("STOCK_INVESTMENT_DSS_IQN_EPSILON_EVAL", 0.001),
            epsilon_decay_steps=_env_int(
                "STOCK_INVESTMENT_DSS_IQN_EPSILON_DECAY_STEPS",
                250_000,
            ),
        )

    @classmethod
    def stockdss_long_v1(cls) -> "IQNConfig":
        """Return the current practical long-run StockDSS IQN preset."""

        return cls(
            config_preset="stockdss_long_v1",
            total_steps=_env_int("STOCK_INVESTMENT_DSS_IQN_TOTAL_STEPS", 25_000),
            learning_starts=_env_int("STOCK_INVESTMENT_DSS_IQN_LEARNING_STARTS", 2_000),
            batch_size=_env_int("STOCK_INVESTMENT_DSS_IQN_BATCH_SIZE", 64),
            replay_capacity=_env_int(
                "STOCK_INVESTMENT_DSS_IQN_REPLAY_CAPACITY",
                100_000,
            ),
            lr=_env_float("STOCK_INVESTMENT_DSS_IQN_LR", 1e-4),
            target_update_interval=_env_int(
                "STOCK_INVESTMENT_DSS_IQN_TARGET_UPDATE_INTERVAL",
                1_000,
            ),
            epsilon_decay_steps=_env_int(
                "STOCK_INVESTMENT_DSS_IQN_EPSILON_DECAY_STEPS",
                15_000,
            ),
        )


def build_iqn_config() -> IQNConfig:
    """
    Build the runtime IQN config.

    The name deliberately avoids the word ``environment`` because this project
    also uses FinRL StockTradingEnv / DiscreteFinRLDecisionEnv.

    Supported optional presets via STOCK_INVESTMENT_DSS_IQN_CONFIG_PRESET:
    - stockdss_default: dataclass defaults with env overrides
    - stockdss_long_v1: practical long-run settings used in current PoC
    - dopamine_jax_like: reference preset inspired by Dopamine JAX IQN

    Prefer this function in runner code.
    """

    preset = _env_str("STOCK_INVESTMENT_DSS_IQN_CONFIG_PRESET", "stockdss_default")
    normalized = preset.strip().lower()

    if normalized == "stockdss_long_v1":
        return IQNConfig.stockdss_long_v1()

    if normalized in {"dopamine_jax_like", "dopamine"}:
        return IQNConfig.dopamine_jax_like()

    return IQNConfig(config_preset="stockdss_default")


def build_iqn_config_from_environment() -> IQNConfig:
    """Backward-compatible alias. Prefer build_iqn_config() in new code."""

    return build_iqn_config()
