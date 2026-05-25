# src/stock_investment_dss/runner/run_finrl_baseline_learning_curve_multiseed_launcher.py
"""Launch FinRL/SB3 baseline learning-budget curves across seeds.

This runner creates a practical learning-curve approximation for the FinRL
baselines by repeatedly training from scratch with increasing training budgets:

    train_steps = 5000, 10000, 15000, 20000, 25000

for each requested seed. Each subprocess calls the existing baseline suite:

    python -m stock_investment_dss.runner.run_finrl_baseline_suite_smoke_test

Important interpretation
------------------------
This is a *training-budget curve*, not a true single-run checkpoint curve. Each
point is trained from scratch for that number of timesteps. It answers:

    "How does out-of-sample backtest performance change as training budget grows?"

It does not show the exact internal trajectory of one continuously trained SB3
model. That would require checkpoint callbacks inside the SB3 training loop.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

RUN_KIND = "finrl_baseline_learning_curve_multiseed_launcher"
PROJECT_NAME = "StockInvestmentDSS"
PROTOTYPE_NAME = "D-IQN-DSS"
DEFAULT_SEEDS = "1,2,3,4,5"
DEFAULT_AGENTS = "a2c,ddpg,td3,ppo,sac"
DEFAULT_TRAIN_STEPS = "5000,10000,15000,20000,25000"

RUN_ID_RE = re.compile(r"Run id:\s*(?P<run_id>\S+)")


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


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("Integer list is empty.")
    return values


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def parse_child_run_id(stdout: str, stderr: str) -> str | None:
    text = stdout + "\n" + stderr
    matches = RUN_ID_RE.findall(text)
    if matches:
        return matches[-1]
    return None


def main() -> int:
    project_root = find_project_root()
    run_id = now_run_id()
    run_dir = project_root / "outputs" / "runs" / run_id
    logs_dir = run_dir / "logs"
    summary_dir = run_dir / "summary"
    data_dir = run_dir / "data"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(
        get_str_env(
            "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_LIST",
            get_str_env("STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST", DEFAULT_SEEDS),
        )
    )
    train_steps = parse_int_list(
        get_str_env("STOCK_INVESTMENT_DSS_FINRL_LEARNING_CURVE_TRAIN_STEPS", DEFAULT_TRAIN_STEPS)
    )
    agents = get_str_env("STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS", DEFAULT_AGENTS)
    include_mvo = get_bool_env("STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO", True)
    stop_on_failure = get_bool_env("STOCK_INVESTMENT_DSS_FINRL_LEARNING_CURVE_STOP_ON_FAILURE", True)
    run_summary_after = get_bool_env("STOCK_INVESTMENT_DSS_FINRL_LEARNING_CURVE_RUN_SUMMARY_AFTER", True)

    log("Starting StockInvestmentDSS FinRL baseline learning-curve multiseed launcher.")
    log(f"Project root: {project_root}")
    log(f"Created run directory: {run_dir}")
    log(f"Run id: {run_id}")
    log(f"Seeds: {seeds}")
    log(f"Train steps: {train_steps}")
    log(f"Agents: {agents}")
    log(f"Include MVO: {include_mvo}")
    log("Interpretation: each checkpoint retrains from scratch with the specified training budget.")

    launched_runs: list[dict[str, Any]] = []
    failed_run_count = 0

    for train_step in train_steps:
        for seed in seeds:
            env = os.environ.copy()
            env["PYTHONPATH"] = env.get("PYTHONPATH", "src")
            env["STOCK_INVESTMENT_DSS_RANDOM_SEED"] = str(seed)
            env["STOCK_INVESTMENT_DSS_FINRL_SEED"] = str(seed)
            env["STOCK_INVESTMENT_DSS_SB3_SEED"] = str(seed)
            env["STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS"] = agents
            env["STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO"] = "true" if include_mvo else "false"
            env["STOCK_INVESTMENT_DSS_FINRL_BASELINE_TOTAL_TIMESTEPS"] = str(train_step)
            env["STOCK_INVESTMENT_DSS_FINRL_LEARNING_CURVE_TRAIN_STEP"] = str(train_step)

            command = [
                sys.executable,
                "-m",
                "stock_investment_dss.runner.run_finrl_baseline_suite_smoke_test",
            ]
            log_path = logs_dir / f"seed_{seed}_train_step_{train_step}_subprocess.log"
            log(f"Launching FinRL baseline suite run for seed={seed}, train_step={train_step}")

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
                + f"\nSEED: {seed}\nTRAIN_STEP: {train_step}\n"
                + "\nSTDOUT:\n"
                + completed.stdout
                + "\n\nSTDERR:\n"
                + completed.stderr,
                encoding="utf-8",
            )
            child_run_id = parse_child_run_id(completed.stdout, completed.stderr)
            record = {
                "seed": seed,
                "train_step": train_step,
                "return_code": completed.returncode,
                "child_run_id": child_run_id,
                "log_path": str(log_path),
            }
            launched_runs.append(record)

            if completed.returncode == 0:
                log(f"Seed {seed}, train_step {train_step} completed successfully. child_run_id={child_run_id}")
            else:
                failed_run_count += 1
                log(
                    f"Seed {seed}, train_step {train_step} failed with return code {completed.returncode}. See {log_path}",
                    level="ERROR",
                )
                if stop_on_failure:
                    break
        if failed_run_count and stop_on_failure:
            break

    launched_csv = data_dir / "finrl_baseline_learning_curve_launched_runs.csv"
    try:
        import pandas as pd

        pd.DataFrame(launched_runs).to_csv(launched_csv, index=False)
    except Exception as exc:  # pragma: no cover
        log(f"Could not write launched_runs CSV: {exc}", level="WARNING")

    summary_return_code: int | None = None
    summary_log_path: str | None = None
    if run_summary_after:
        command = [
            sys.executable,
            "-m",
            "stock_investment_dss.runner.run_finrl_baseline_learning_curve_multiseed_summary",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = env.get("PYTHONPATH", "src")
        env["STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_LIST"] = ",".join(str(seed) for seed in seeds)
        env["STOCK_INVESTMENT_DSS_FINRL_LEARNING_CURVE_TRAIN_STEPS"] = ",".join(str(step) for step in train_steps)
        env["STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS"] = agents
        env["STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO"] = "true" if include_mvo else "false"
        log("Launching FinRL baseline learning-curve multiseed summary runner.")
        summary_log = logs_dir / "finrl_learning_curve_multiseed_summary_subprocess.log"
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
            log("FinRL baseline learning-curve multiseed summary completed successfully.")
        else:
            log(
                f"FinRL baseline learning-curve multiseed summary failed with return code {completed.returncode}. See {summary_log}",
                level="ERROR",
            )

    summary = {
        "status": "ok" if failed_run_count == 0 else "failed_or_partial",
        "project_name": PROJECT_NAME,
        "prototype_name": PROTOTYPE_NAME,
        "run_id": run_id,
        "project_root": str(project_root),
        "run_directory": str(run_dir),
        "seeds": seeds,
        "train_steps": train_steps,
        "agents": agents,
        "include_mvo": include_mvo,
        "launched_runs": launched_runs,
        "failed_run_count": failed_run_count,
        "summary_run_return_code": summary_return_code,
        "summary_log_path": summary_log_path,
        "outputs": {"launched_runs_path": str(launched_csv)},
        "interpretation": (
            "This launcher builds a FinRL training-budget curve by retraining each baseline from scratch "
            "for each requested training budget and seed. It is useful for learning-diagnostics, but is "
            "not the same as checkpointing one continuous training run."
        ),
    }
    summary_path = summary_dir / "finrl_baseline_learning_curve_multiseed_launcher_summary.json"
    write_json(summary_path, summary)
    log(f"Wrote launcher summary: {summary_path}")
    log("StockInvestmentDSS FinRL baseline learning-curve multiseed launcher completed.")
    return 0 if failed_run_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
