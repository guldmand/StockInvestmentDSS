"""Optional Weights & Biases tracking helpers for StockInvestmentDSS.

This module is intentionally optional: if wandb is not installed or disabled,
callers can keep running local experiments unchanged.
"""

from __future__ import annotations

import os
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def is_wandb_enabled() -> bool:
    return _env_bool("STOCK_INVESTMENT_DSS_WANDB_ENABLED", False)


def init_wandb_run(
    *,
    run_name: str,
    config: dict[str, Any] | None = None,
    group: str | None = None,
    job_type: str | None = None,
    tags: list[str] | None = None,
):
    """Initialize a W&B run if enabled; otherwise return None."""

    if not is_wandb_enabled():
        return None

    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking is enabled, but the 'wandb' package is not installed. "
            "Install it or set STOCK_INVESTMENT_DSS_WANDB_ENABLED=false."
        ) from exc

    project = os.getenv("STOCK_INVESTMENT_DSS_WANDB_PROJECT", "stock-investment-dss")
    entity = os.getenv("STOCK_INVESTMENT_DSS_WANDB_ENTITY") or None
    mode = os.getenv("STOCK_INVESTMENT_DSS_WANDB_MODE") or None

    return wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=group,
        job_type=job_type,
        tags=tags,
        config=config or {},
        mode=mode,
    )


def wandb_log(data: dict[str, Any], *, step: int | None = None) -> None:
    """Log data to the active W&B run if enabled and initialized."""

    if not is_wandb_enabled():
        return

    try:
        import wandb  # type: ignore
    except ImportError:
        return

    if wandb.run is not None:
        wandb.log(data, step=step)


def finish_wandb_run() -> None:
    if not is_wandb_enabled():
        return

    try:
        import wandb  # type: ignore
    except ImportError:
        return

    if wandb.run is not None:
        wandb.finish()
