# src/stock_investment_dss/runner/run_iqn_learning_curve_multiseed_launcher.py

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from stock_investment_dss.utilities.config import get_environment_variable
from stock_investment_dss.utilities.logging import setup_run_logger, setup_system_logger
from stock_investment_dss.utilities.paths import PROJECT_ROOT, create_run_paths


def get_int_environment_variable(name: str, default: int) -> int:
    value = get_environment_variable(name, default=str(default))
    return int(value or default)


def get_seed_list_from_environment() -> list[int]:
    raw = get_environment_variable(
        "STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST",
        default="1,2,3",
    )
    if raw is None or not raw.strip():
        return [1, 2, 3]

    seeds: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    return seeds


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def main() -> int:
    log_level = (
        get_environment_variable(
            "STOCK_INVESTMENT_DSS_LOG_LEVEL",
            default="INFO",
        )
        or "INFO"
    )

    system_logger = setup_system_logger(log_level=log_level)
    system_logger.info(
        "Starting StockInvestmentDSS IQN learning curve multiseed launcher."
    )
    system_logger.info("Project root: %s", PROJECT_ROOT)

    run_paths = None

    try:
        seeds = get_seed_list_from_environment()
        stop_on_failure = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_STOP_ON_FAILURE",
                default="true",
            )
            or "true"
        ).strip().lower() in {"1", "true", "yes", "y"}
        run_summary_after = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_RUN_SUMMARY_AFTER",
                default="true",
            )
            or "true"
        ).strip().lower() in {"1", "true", "yes", "y"}

        run_paths = create_run_paths("d_iqn_dss_iqn_learning_curve_multiseed_launcher")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Seeds: %s", seeds)
        run_logger.info("Stop on failure: %s", stop_on_failure)
        run_logger.info("Run summary after: %s", run_summary_after)

        launched_runs: list[dict[str, Any]] = []

        for seed in seeds:
            env = os.environ.copy()
            env["PYTHONPATH"] = env.get("PYTHONPATH", "src")
            env["STOCK_INVESTMENT_DSS_RANDOM_SEED"] = str(seed)
            env["STOCK_INVESTMENT_DSS_IQN_SEED"] = str(seed)

            command = [
                sys.executable,
                "-m",
                "stock_investment_dss.runner.run_iqn_learning_curve_smoke_test",
            ]

            run_logger.info("Launching IQN learning curve run for seed=%s", seed)
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                env=env,
                text=True,
                capture_output=True,
            )

            seed_log_path = run_paths.logs_directory / f"seed_{seed}_subprocess.log"
            seed_log_path.parent.mkdir(parents=True, exist_ok=True)
            seed_log_path.write_text(
                "COMMAND: "
                + " ".join(command)
                + "\n\nSTDOUT:\n"
                + completed.stdout
                + "\n\nSTDERR:\n"
                + completed.stderr,
                encoding="utf-8",
            )

            launched = {
                "seed": seed,
                "return_code": completed.returncode,
                "log_path": str(seed_log_path),
            }
            launched_runs.append(launched)

            if completed.returncode != 0:
                run_logger.error(
                    "Seed %s failed with return code %s. See %s",
                    seed,
                    completed.returncode,
                    seed_log_path,
                )
                if stop_on_failure:
                    break
            else:
                run_logger.info("Seed %s completed successfully.", seed)

        summary_run_return_code = None
        summary_stdout_path = None
        if run_summary_after:
            env = os.environ.copy()
            env["PYTHONPATH"] = env.get("PYTHONPATH", "src")
            env.setdefault(
                "STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_DEDUPLICATE_SEEDS",
                "true",
            )
            command = [
                sys.executable,
                "-m",
                "stock_investment_dss.runner.run_iqn_learning_curve_multiseed_summary",
            ]
            run_logger.info("Launching multiseed summary runner.")
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                env=env,
                text=True,
                capture_output=True,
            )
            summary_stdout_path = (
                run_paths.logs_directory / "multiseed_summary_subprocess.log"
            )
            summary_stdout_path.write_text(
                "COMMAND: "
                + " ".join(command)
                + "\n\nSTDOUT:\n"
                + completed.stdout
                + "\n\nSTDERR:\n"
                + completed.stderr,
                encoding="utf-8",
            )
            summary_run_return_code = completed.returncode
            if completed.returncode != 0:
                run_logger.error(
                    "Multiseed summary failed with return code %s. See %s",
                    completed.returncode,
                    summary_stdout_path,
                )
            else:
                run_logger.info("Multiseed summary completed successfully.")

        summary_path = (
            run_paths.summary_directory
            / "iqn_learning_curve_multiseed_launcher_summary.json"
        )
        failed_runs = [row for row in launched_runs if row["return_code"] != 0]
        payload = {
            "status": "ok" if not failed_runs else "partial_failure",
            "project_name": "StockInvestmentDSS",
            "prototype_name": "D-IQN-DSS",
            "run_id": run_paths.run_id,
            "project_root": str(PROJECT_ROOT),
            "run_directory": str(run_paths.run_directory),
            "seeds": seeds,
            "launched_runs": launched_runs,
            "failed_run_count": len(failed_runs),
            "summary_run_return_code": summary_run_return_code,
            "summary_log_path": (
                str(summary_stdout_path) if summary_stdout_path else None
            ),
            "interpretation": (
                "This launcher runs the IQN learning-curve smoke test with multiple "
                "random seeds by setting STOCK_INVESTMENT_DSS_RANDOM_SEED and "
                "STOCK_INVESTMENT_DSS_IQN_SEED for each subprocess. Afterward, it can "
                "run the multiseed summary aggregator."
            ),
        }
        write_json(summary_path, payload)

        run_logger.info("Wrote launcher summary: %s", summary_path)
        system_logger.info(
            "StockInvestmentDSS IQN learning curve multiseed launcher completed."
        )
        return 0 if not failed_runs and (summary_run_return_code in {None, 0}) else 1

    except Exception:
        system_logger.exception(
            "StockInvestmentDSS IQN learning curve multiseed launcher failed."
        )
        if run_paths is not None:
            try:
                run_logger = setup_run_logger(run_paths, log_level=log_level)
                run_logger.exception("Run failed.")
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
