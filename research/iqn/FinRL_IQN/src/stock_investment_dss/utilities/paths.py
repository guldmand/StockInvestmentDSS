from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def find_project_root() -> Path:
    """
    Find the FinRL_IQN project root.

    Expected structure:
        FinRL_IQN/
            .env
            src/
                stockdss/
    """
    current_path = Path(__file__).resolve()

    for parent in current_path.parents:
        if (parent / "src").exists() and (parent / ".env.example").exists():
            return parent

    raise RuntimeError(
        "Could not find project root. Expected to find a folder containing "
        "'src/' and '.env.example'."
    )


PROJECT_ROOT = find_project_root()

DATA_DIRECTORY = PROJECT_ROOT / "data"
EXPERIMENTS_DIRECTORY = PROJECT_ROOT / "experiments"
LOGS_DIRECTORY = PROJECT_ROOT / "logs"
OUTPUTS_DIRECTORY = PROJECT_ROOT / "outputs"
RUNS_DIRECTORY = OUTPUTS_DIRECTORY / "runs"


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_directory: Path
    config_directory: Path
    data_directory: Path
    models_directory: Path
    logs_directory: Path
    audit_directory: Path
    metrics_directory: Path
    plots_directory: Path
    summary_directory: Path


def ensure_project_directories() -> None:
    """
    Create only the project-level directories.
    Safe to run repeatedly.
    """
    for directory in [
        DATA_DIRECTORY,
        EXPERIMENTS_DIRECTORY,
        LOGS_DIRECTORY,
        OUTPUTS_DIRECTORY,
        RUNS_DIRECTORY,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def create_run_paths(run_name: str) -> RunPaths:
    """
    Create a self-contained run directory under outputs/runs/.

    Example:
        outputs/runs/2026_05_17_0412_v2_smoke_test/
    """
    ensure_project_directories()

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    safe_run_name = run_name.strip().lower().replace(" ", "_")
    run_id = f"{timestamp}_{safe_run_name}"

    run_directory = RUNS_DIRECTORY / run_id

    config_directory = run_directory / "config"
    data_directory = run_directory / "data"
    models_directory = run_directory / "models"
    logs_directory = run_directory / "logs"
    audit_directory = run_directory / "audit"
    metrics_directory = run_directory / "metrics"
    plots_directory = run_directory / "plots"
    summary_directory = run_directory / "summary"

    for directory in [
        run_directory,
        config_directory,
        data_directory,
        models_directory,
        logs_directory,
        audit_directory,
        metrics_directory,
        plots_directory,
        summary_directory,
    ]:
        directory.mkdir(parents=True, exist_ok=False)

    return RunPaths(
        run_id=run_id,
        run_directory=run_directory,
        config_directory=config_directory,
        data_directory=data_directory,
        models_directory=models_directory,
        logs_directory=logs_directory,
        audit_directory=audit_directory,
        metrics_directory=metrics_directory,
        plots_directory=plots_directory,
        summary_directory=summary_directory,
    )
