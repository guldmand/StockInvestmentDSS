# src/stockdss/utilities/experiment_tracking.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExperimentTracker:
    enabled: bool = False
    project_name: str = "stockdss-finrl-iqn"
    run_name: str | None = None
    _wandb_run: Any = None

    def start(self, config: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return

        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "Weights & Biases is enabled, but the 'wandb' package is not installed."
            ) from exc

        self._wandb_run = wandb.init(
            project=self.project_name,
            name=self.run_name,
            config=config or {},
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._wandb_run is None:
            return

        import wandb

        wandb.log(metrics, step=step)

    def finish(self) -> None:
        if not self.enabled or self._wandb_run is None:
            return

        import wandb

        wandb.finish()
        self._wandb_run = None
