"""
Central run-folder path helper for StockDSS experiments.

Purpose:
- Keep runner outputs in one canonical folder.
- Avoid scattered outputs across outputs/, trained_models/, and results/.
- Preserve standalone scripts by only using these paths when --run-root is provided.

Canonical structure:

outputs/runs/<run_id>/
├─ run_config.json
├─ run_commands.ps1
├─ run_summary.json
│
├─ baseline_finrl/
│  ├─ models/
│  ├─ files/
│  │  ├─ train/
│  │  └─ backtest/
│  ├─ plots/
│  └─ logs/
│
└─ iqn_finrl/
   ├─ models/
   ├─ files/
   │  ├─ train/
   │  └─ backtest/
   ├─ plots/
   ├─ visualizations/
   │  └─ decision_distribution/
   └─ logs/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """Canonical folder layout for one complete StockDSS experiment run."""

    run_root: Path

    baseline_root: Path
    baseline_models: Path
    baseline_train_files: Path
    baseline_backtest_files: Path
    baseline_plots: Path
    baseline_logs: Path

    iqn_root: Path
    iqn_models: Path
    iqn_train_files: Path
    iqn_backtest_files: Path
    iqn_plots: Path
    iqn_visualizations: Path
    iqn_decision_visualizations: Path
    iqn_logs: Path


def build_run_paths(run_root: str | Path, create: bool = True) -> RunPaths:
    """
    Build canonical paths for a full StockDSS run.

    Args:
        run_root:
            Root folder for the run, e.g.
            outputs/runs/2026_05_14_0300_runner_pit500_20260101_aapl

        create:
            If True, all folders are created.

    Returns:
        RunPaths object with all important output folders.
    """
    run_root = Path(run_root)

    paths = RunPaths(
        run_root=run_root,
        baseline_root=run_root / "baseline_finrl",
        baseline_models=run_root / "baseline_finrl" / "models",
        baseline_train_files=run_root / "baseline_finrl" / "files" / "train",
        baseline_backtest_files=run_root / "baseline_finrl" / "files" / "backtest",
        baseline_plots=run_root / "baseline_finrl" / "plots",
        baseline_logs=run_root / "baseline_finrl" / "logs",
        iqn_root=run_root / "iqn_finrl",
        iqn_models=run_root / "iqn_finrl" / "models",
        iqn_train_files=run_root / "iqn_finrl" / "files" / "train",
        iqn_backtest_files=run_root / "iqn_finrl" / "files" / "backtest",
        iqn_plots=run_root / "iqn_finrl" / "plots",
        iqn_visualizations=run_root / "iqn_finrl" / "visualizations",
        iqn_decision_visualizations=(
            run_root / "iqn_finrl" / "visualizations" / "decision_distribution"
        ),
        iqn_logs=run_root / "iqn_finrl" / "logs",
    )

    if create:
        create_run_paths(paths)

    return paths


def create_run_paths(paths: RunPaths) -> None:
    """Create all folders in a RunPaths object."""
    for value in paths.__dict__.values():
        if isinstance(value, Path):
            value.mkdir(parents=True, exist_ok=True)


def print_run_paths(paths: RunPaths) -> None:
    """Print canonical run folders in a readable way."""
    print("=" * 100)
    print("StockDSS canonical run paths")
    print("=" * 100)
    print(f"Run root:                    {paths.run_root}")
    print()
    print("Baseline FinRL")
    print("-" * 100)
    print(f"Models:                      {paths.baseline_models}")
    print(f"Train files:                 {paths.baseline_train_files}")
    print(f"Backtest files:              {paths.baseline_backtest_files}")
    print(f"Plots:                       {paths.baseline_plots}")
    print(f"Logs:                        {paths.baseline_logs}")
    print()
    print("IQN FinRL")
    print("-" * 100)
    print(f"Models:                      {paths.iqn_models}")
    print(f"Train files:                 {paths.iqn_train_files}")
    print(f"Backtest files:              {paths.iqn_backtest_files}")
    print(f"Plots:                       {paths.iqn_plots}")
    print(f"Visualizations:              {paths.iqn_visualizations}")
    print(f"Decision visualizations:     {paths.iqn_decision_visualizations}")
    print(f"Logs:                        {paths.iqn_logs}")
    print("=" * 100)
