# src/stock_investment_dss/runner/run_finrl_baseline_multiseed_launcher.py
"""Launch FinRL baseline suite runs across multiple random seeds.

This runner is the FinRL/SB3 counterpart to the IQN multiseed launcher.
It repeatedly calls:

    python -m stock_investment_dss.runner.run_finrl_baseline_suite_smoke_test

with different seed values. It is intended for fairer comparison against the
D-IQN-DSS IQN multiseed experiment.

Notes on MVO
------------
MVO is included by default when the existing baseline-suite runner supports it.
If the MVO implementation is deterministic for a fixed dataset, then running it
across seeds will produce identical rows and std=0 in the multiseed summary.
That is still useful: it clearly marks MVO as a deterministic baseline while
keeping the comparison table format aligned with the stochastic RL agents.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

RUN_KIND = "finrl_baseline_multiseed_launcher"
PROJECT_NAME = "StockInvestmentDSS"
PROTOTYPE_NAME = "D-IQN-DSS"

DEFAULT_SEEDS = "1,2,3,4,5"
DEFAULT_AGENTS = "a2c,ddpg,td3,ppo,sac"


def now_run_id() -> str:
    return f"{datetime.now():%Y_%m_%d_%H%M%S}_d_iqn_dss_{RUN_KIND}"


def find_project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src").exists() or (candidate / "outputs").exists():
            return candidate
    return current


def log(message: str, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} | {level} | stock_investment_dss.run | {message}", file=sys.stderr)


def get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_str_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def parse_seed_list(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        seeds.append(int(value))
    if not seeds:
        raise ValueError("Seed list is empty.")
    return seeds


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=str)


def main() -> int:
    project_root = find_project_root()
    run_id = now_run_id()
    run_directory = project_root / "outputs" / "runs" / run_id
    logs_directory = run_directory / "logs"
    summary_directory = run_directory / "summary"
    logs_directory.mkdir(parents=True, exist_ok=True)
    summary_directory.mkdir(parents=True, exist_ok=True)

    log("Starting StockInvestmentDSS FinRL baseline multiseed launcher.")
    log(f"Project root: {project_root}")
    log(f"Created run directory: {run_directory}")
    log(f"Run id: {run_id}")

    seeds = parse_seed_list(
        get_str_env(
            "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_LIST",
            get_str_env("STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST", DEFAULT_SEEDS),
        )
    )
    agents = get_str_env("STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS", DEFAULT_AGENTS)
    stop_on_failure = get_bool_env("STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_STOP_ON_FAILURE", True)
    run_summary_after = get_bool_env("STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_RUN_SUMMARY_AFTER", True)
    include_mvo = get_bool_env("STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO", True)

    log(f"Seeds: {seeds}")
    log(f"Agents: {agents}")
    log(f"Include MVO: {include_mvo}")
    log(f"Stop on failure: {stop_on_failure}")
    log(f"Run summary after: {run_summary_after}")

    launched_runs: list[dict[str, Any]] = []
    failed_run_count = 0

    for seed in seeds:
        env = os.environ.copy()
        env["PYTHONPATH"] = env.get("PYTHONPATH", "src")
        env["STOCK_INVESTMENT_DSS_RANDOM_SEED"] = str(seed)
        env["STOCK_INVESTMENT_DSS_FINRL_SEED"] = str(seed)
        env["STOCK_INVESTMENT_DSS_SB3_SEED"] = str(seed)
        env["STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS"] = agents
        env["STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO"] = "true" if include_mvo else "false"

        command = [
            sys.executable,
            "-m",
            "stock_investment_dss.runner.run_finrl_baseline_suite_smoke_test",
        ]
        log_path = logs_directory / f"seed_{seed}_subprocess.log"
        log(f"Launching FinRL baseline suite run for seed={seed}")

        completed = subprocess.run(
            command,
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
        )

        log_path.write_text(
            "COMMAND: "
            + " ".join(command)
            + "\n\nSTDOUT:\n"
            + completed.stdout
            + "\n\nSTDERR:\n"
            + completed.stderr,
            encoding="utf-8",
        )

        record = {
            "seed": seed,
            "return_code": completed.returncode,
            "log_path": str(log_path),
        }
        launched_runs.append(record)

        if completed.returncode == 0:
            log(f"Seed {seed} completed successfully.")
        else:
            failed_run_count += 1
            log(
                f"Seed {seed} failed with return code {completed.returncode}. See {log_path}",
                level="ERROR",
            )
            if stop_on_failure:
                break

    summary_return_code: int | None = None
    summary_log_path: str | None = None
    if run_summary_after:
        command = [
            sys.executable,
            "-m",
            "stock_investment_dss.runner.run_finrl_baseline_multiseed_summary",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = env.get("PYTHONPATH", "src")
        env["STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_LIST"] = ",".join(str(seed) for seed in seeds)
        env["STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS"] = agents
        env["STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO"] = "true" if include_mvo else "false"
        log("Launching FinRL baseline multiseed summary runner.")
        summary_log = logs_directory / "finrl_multiseed_summary_subprocess.log"
        completed = subprocess.run(
            command,
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
        )
        summary_log.write_text(
            "COMMAND: "
            + " ".join(command)
            + "\n\nSTDOUT:\n"
            + completed.stdout
            + "\n\nSTDERR:\n"
            + completed.stderr,
            encoding="utf-8",
        )
        summary_return_code = completed.returncode
        summary_log_path = str(summary_log)
        if completed.returncode == 0:
            log("FinRL baseline multiseed summary completed successfully.")
        else:
            log(
                f"FinRL baseline multiseed summary failed with return code {completed.returncode}. See {summary_log}",
                level="ERROR",
            )

    summary = {
        "status": "ok" if failed_run_count == 0 else "failed_or_partial",
        "project_name": PROJECT_NAME,
        "prototype_name": PROTOTYPE_NAME,
        "run_id": run_id,
        "project_root": str(project_root),
        "run_directory": str(run_directory),
        "seeds": seeds,
        "agents": [part.strip() for part in agents.split(",") if part.strip()],
        "include_mvo": include_mvo,
        "launched_runs": launched_runs,
        "failed_run_count": failed_run_count,
        "summary_run_return_code": summary_return_code,
        "summary_log_path": summary_log_path,
        "interpretation": (
            "This launcher runs the FinRL/SB3 baseline suite with multiple random seeds. "
            "MVO can be included; if deterministic, it should produce std=0 across seeds."
        ),
    }
    summary_path = summary_directory / "finrl_baseline_multiseed_launcher_summary.json"
    write_json(summary_path, summary)
    log(f"Wrote launcher summary: {summary_path}")
    log("StockInvestmentDSS FinRL baseline multiseed launcher completed.")
    return 0 if failed_run_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
