"""D-IQN-DSS Demo Baselines and IQN — end-to-end thesis showcase orchestrator.

Runs the full thesis comparison pipeline in a single command, orchestrating all
three strategy tiers and the downstream reporting stages:

    1. Load Mode B dataset          — validate frozen import + SHA-256 hash
    2. Algorithmic baselines        — etape 3 grid (all single-ticker + portfolio)
    3. FinRL baselines              — etape 4 multiseed (or auto-discover existing)
    4. IQN                          — auto-discover existing checkpoint + summary run
    5. Decision dashboard           — skipped (integrated into IQN run output)
    6. Summary dashboard            — etape 5 (4-panel, 246 strategies)
    7. Comparison report            — etape 6 (thesis-citable ranking table)
    8. Master summary               — demo_master_summary.md

Output structure::

    outputs/runs/{timestamp}_d_iqn_dss_demo_baselines_and_iqn/
    ├── audit/pipeline_execution.json
    ├── config/demo_config.json
    ├── config/universe_config.json
    ├── data/pipeline_input_hashes.json
    ├── logs/run.log  (+ per-step logs)
    ├── summary/demo_master_summary.md
    ├── summary/demo_master_summary.json
    ├── summary/subprocess_results.csv
    └── summary/final_comparison.md

Usage::

    python -m stock_investment_dss.runner.run_demo_baselines_and_iqn [OPTIONS]

    # Smoke test (skip FinRL training, use existing IQN run):
    python -m stock_investment_dss.runner.run_demo_baselines_and_iqn \\
        --universe demo_10_new \\
        --skip-finrl-training \\
        --iqn-checkpoint auto

Pre-existing runs required for the smoke test:
    - outputs/runs/*_finrl_baseline_multiseed_summary*/  (FinRL aggregate CSV)
    - outputs/runs/*_iqn_learning_curve_multiseed_summary*/  (IQN eval records CSV)
    Algorithmic baselines are always re-run (fast, ~20-60 s).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from stock_investment_dss.utilities.paths import (  # noqa: E402
    RUNS_DIRECTORY,
    create_run_paths,
)
from stock_investment_dss.utilities.logging import setup_run_logger  # noqa: E402
from stock_investment_dss.experiment_tracking.wandb_tracking import (  # noqa: E402
    finish_wandb_run,
    init_wandb_run,
    wandb_log,
)

# ---------------------------------------------------------------------------
# Universe registry
# ---------------------------------------------------------------------------

_UNIVERSE_TICKERS: dict[str, list[str]] = {
    "demo_5": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    "demo_10_new": [
        "COST",
        "AVGO",
        "LLY",
        "ORCL",
        "CAT",
        "BA",
        "KO",
        "MCD",
        "WMT",
        "PG",
    ],
}

_UNIVERSE_DATA_FILE: dict[str, str] = {
    "demo_5": "data/market/daily/imports/market_data_demo5_2010_2026.csv",
    "demo_10_new": "data/market/daily/imports/market_data_demo10_new_2010_2026.csv",
}

# ---------------------------------------------------------------------------
# StepResult (plain dict — TypedDict for static checkers only)
# ---------------------------------------------------------------------------

try:
    from typing import TypedDict

    class StepResult(TypedDict):
        name: str
        status: str  # "ok" | "failed" | "skipped" | "found_existing"
        returncode: int
        duration_sec: float
        output_path: str
        log_path: str
        command: str

except ImportError:
    StepResult = dict  # type: ignore[misc,assignment]


def _make_result(
    name: str,
    *,
    status: str,
    returncode: int = 0,
    duration_sec: float = 0.0,
    output_path: str = "n/a",
    log_path: str = "n/a",
    command: str = "n/a",
) -> StepResult:
    return {
        "name": name,
        "status": status,
        "returncode": returncode,
        "duration_sec": duration_sec,
        "output_path": output_path,
        "log_path": log_path,
        "command": command,
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _banner(title: str, char: str = "=") -> None:
    print(char * 78, flush=True)
    print(title, flush=True)
    print(char * 78, flush=True)


def stream_subprocess(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    label: str,
    log_path: Optional[Path] = None,
) -> int:
    """Run *cmd*, stream stdout+stderr line-by-line to console and optionally to file."""
    print(f"[{label}] CMD: {' '.join(cmd)}", flush=True)
    print(f"[{label}] CWD: {cwd}", flush=True)
    print(f"[{label}] START: {datetime.now():%H:%M:%S}", flush=True)
    print("-" * 78, flush=True)

    log_fh = (
        open(log_path, "w", encoding="utf-8", errors="replace")  # noqa: WPS515
        if log_path
        else None
    )
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            if log_fh:
                log_fh.write(line)
    except KeyboardInterrupt:
        print(f"\n[{label}] INTERRUPTED — terminating subprocess", flush=True)
        if process is not None:
            process.terminate()
            process.wait()
        if log_fh:
            log_fh.close()
        return 130
    finally:
        if log_fh:
            log_fh.close()

    rc = process.wait()
    print("-" * 78, flush=True)
    print(f"[{label}] END:   {datetime.now():%H:%M:%S}  RC={rc}", flush=True)
    return rc


def _find_latest_run(runs_dir: Path, name_pattern: str) -> Path:
    """Return the most recent run whose name contains *name_pattern* (lex = chrono)."""
    matches = [d for d in runs_dir.iterdir() if d.is_dir() and name_pattern in d.name]
    if not matches:
        raise FileNotFoundError(
            f"No run matching pattern {name_pattern!r} found in {runs_dir}"
        )
    return sorted(matches)[-1]


def find_run_created_after(
    runs_dir: Path, name_pattern: str, after_ts: datetime
) -> Path:
    """Find newest run matching *name_pattern* whose timestamp is after *after_ts*.

    Directory names are expected to start with ``YYYY_MM_DD_HHMMSS``.
    """
    matches = [d for d in runs_dir.iterdir() if d.is_dir() and name_pattern in d.name]
    qualifying: list[Path] = []
    for d in matches:
        parts = d.name.split("_")
        if len(parts) >= 6:
            ts_str = "_".join(parts[:6])
            try:
                run_ts = datetime.strptime(ts_str, "%Y_%m_%d_%H%M%S")
            except ValueError:
                continue
            if run_ts > after_ts:
                qualifying.append(d)
    if not qualifying:
        raise FileNotFoundError(
            f"No run matching {name_pattern!r} created after {after_ts} in {runs_dir}"
        )
    return sorted(qualifying)[-1]


def resolve_iqn_checkpoint(arg: str, repo_root: Path) -> Path:
    """Resolve the ``--iqn-checkpoint`` argument to an actual ``.pt`` file path.

    Tier 1 (``auto``):
        Search ``*_d_iqn_dss_clean_25k_baseline_v1_seed_*`` runs (latest first).
    Tier 2 (``auto``):
        Fallback to any ``*_iqn_*`` run that contains a ``models/*.pt`` file.
    ``train``:
        Reserved — raises ``NotImplementedError``.
    Explicit path:
        Resolved relative to *repo_root* if not absolute; validated to exist as ``.pt``.
    """
    runs_dir = repo_root / "outputs" / "runs"

    if arg == "auto":
        for run_dir in sorted(
            runs_dir.glob("*_d_iqn_dss_clean_25k_baseline_v1_seed_*"), reverse=True
        ):
            pt_files = list((run_dir / "models").glob("*.pt"))
            if pt_files:
                return pt_files[0]
        for run_dir in sorted(runs_dir.glob("*_iqn_*"), reverse=True):
            models_dir = run_dir / "models"
            if models_dir.exists():
                pt_files = list(models_dir.glob("*.pt"))
                if pt_files:
                    return pt_files[0]
        raise FileNotFoundError(
            "No IQN checkpoint (.pt) found in outputs/runs/. "
            "Run an IQN training experiment first, or specify --iqn-checkpoint explicitly."
        )

    if arg == "train":
        raise NotImplementedError(
            "--iqn-checkpoint=train is reserved for a future implementation. "
            "Use --iqn-checkpoint=auto or provide an explicit path to a .pt file."
        )

    ckpt = Path(arg)
    if not ckpt.is_absolute():
        ckpt = repo_root / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"IQN checkpoint not found: {ckpt}")
    if ckpt.suffix != ".pt":
        raise ValueError(f"Expected a .pt checkpoint file, got suffix: {ckpt.suffix}")
    return ckpt


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D-IQN-DSS Demo Baselines and IQN — end-to-end thesis demo orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--universe",
        default="demo_10_new",
        choices=list(_UNIVERSE_TICKERS.keys()) + ["demo_30"],
        help="Ticker universe to use.",
    )
    p.add_argument(
        "--pit-decision-date",
        default="2024-01-01",
        metavar="YYYY-MM-DD",
        help="Point-in-time split date (train end / eval start).",
    )
    p.add_argument(
        "--iqn-checkpoint",
        default="auto",
        metavar="PATH|auto|train",
        help="IQN checkpoint .pt path, 'auto' to find latest, or 'train' (reserved).",
    )
    p.add_argument(
        "--finrl-timesteps",
        type=int,
        default=25000,
        metavar="N",
        help="FinRL training timesteps per seed (ignored with --skip-finrl-training).",
    )
    p.add_argument(
        "--skip-finrl-training",
        action="store_true",
        help="Auto-discover the latest existing FinRL multiseed summary instead of training.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="Override default output directory.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        default=5,
        metavar="N",
        help="Number of seeds for FinRL / IQN multiseed runs.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue pipeline even if a step fails.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step_load_dataset(
    args: argparse.Namespace,
    repo_root: Path,
    run_paths,
    logger: logging.Logger,
) -> StepResult:
    """Validate the frozen Mode B dataset and compute a SHA-256 hash."""
    t0 = time.monotonic()
    name = "load_dataset"
    _banner("STEP 1 / 8 — LOAD MODE B DATASET", char="-")

    if args.universe not in _UNIVERSE_DATA_FILE:
        msg = (
            f"Universe '{args.universe}' has no registered data file. "
            f"Known: {list(_UNIVERSE_DATA_FILE.keys())}"
        )
        logger.error(msg)
        print(f"[{name}] ERROR: {msg}", flush=True)
        return _make_result(
            name, status="failed", returncode=1, duration_sec=time.monotonic() - t0
        )

    data_path = repo_root / _UNIVERSE_DATA_FILE[args.universe]
    if not data_path.exists():
        msg = f"Frozen dataset not found: {data_path}"
        logger.error(msg)
        print(f"[{name}] ERROR: {msg}", flush=True)
        return _make_result(
            name, status="failed", returncode=1, duration_sec=time.monotonic() - t0
        )

    digest = _sha256(data_path)
    hashes = {
        "universe": args.universe,
        "data_file": str(data_path),
        "sha256": digest,
    }
    out_path = run_paths.data_directory / "pipeline_input_hashes.json"
    out_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    dur = time.monotonic() - t0
    logger.info("Dataset validated: %s  SHA-256: %s…", data_path.name, digest[:16])
    print(
        f"[{name}] OK — {data_path.name}  SHA-256: {digest[:16]}…  ({dur:.1f}s)",
        flush=True,
    )
    return _make_result(name, status="ok", duration_sec=dur, output_path=str(out_path))


def step_algorithmic_baselines(
    args: argparse.Namespace,
    repo_root: Path,
    run_paths,
    logger: logging.Logger,
) -> StepResult:
    """Run the full algorithmic baseline grid for the configured universe."""
    t0 = time.monotonic()
    name = "algorithmic_baselines"
    _banner("STEP 2 / 8 — ALGORITHMIC BASELINES", char="-")

    data_rel = _UNIVERSE_DATA_FILE.get(args.universe, "")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "PYTHONUNBUFFERED": "1",
            "STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME": args.pit_decision_date,
            "STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE": "2026-12-31",
        }
    )
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "stock_investment_dss.algorithmic_trading.experiments.run_all_algorithmic_experiments",
        "--trade-data",
        data_rel,
        "--dataset-tag",
        args.universe,
        "--ticker",
        "ALL",
        "--continue-on-error",
    ]
    log_path = run_paths.logs_directory / "step_02_algorithmic.log"
    before_ts = datetime.now()
    rc = stream_subprocess(cmd, repo_root, env, label="ALGORITHMIC", log_path=log_path)

    output_path = "n/a"
    try:
        out = find_run_created_after(
            RUNS_DIRECTORY, "algorithmic_baseline_grid", before_ts
        )
        output_path = str(out)
    except FileNotFoundError:
        pass

    dur = time.monotonic() - t0
    status = "ok" if rc == 0 else "failed"
    logger.info("Algorithmic baselines: %s  rc=%d  %.1fs", status, rc, dur)
    return _make_result(
        name,
        status=status,
        returncode=rc,
        duration_sec=dur,
        output_path=output_path,
        log_path=str(log_path),
        command=" ".join(cmd),
    )


def step_finrl_baselines(
    args: argparse.Namespace,
    repo_root: Path,
    run_paths,
    logger: logging.Logger,
) -> StepResult:
    """Run FinRL multiseed training or auto-discover an existing summary run."""
    t0 = time.monotonic()
    name = "finrl_baselines"
    _banner("STEP 3 / 8 — FINRL BASELINES", char="-")

    if args.skip_finrl_training:
        print(
            f"[{name}] --skip-finrl-training set — auto-discovering latest FinRL summary run",
            flush=True,
        )
        try:
            finrl_run = _find_latest_run(
                RUNS_DIRECTORY, "finrl_baseline_multiseed_summary"
            )
        except FileNotFoundError as exc:
            logger.error("FinRL auto-discover failed: %s", exc)
            print(f"[{name}] ERROR: {exc}", flush=True)
            return _make_result(
                name, status="failed", returncode=1, duration_sec=time.monotonic() - t0
            )

        csv_check = (
            finrl_run / "summary" / "finrl_baseline_multiseed_aggregate_by_agent.csv"
        )
        if not csv_check.exists():
            msg = f"Required FinRL aggregate CSV not found: {csv_check}"
            logger.error(msg)
            print(f"[{name}] ERROR: {msg}", flush=True)
            return _make_result(
                name,
                status="failed",
                returncode=1,
                duration_sec=time.monotonic() - t0,
                output_path=str(finrl_run),
            )

        dur = time.monotonic() - t0
        logger.info("FinRL: found existing run %s", finrl_run.name)
        print(f"[{name}] Found: {finrl_run.name}  ({dur:.1f}s)", flush=True)
        return _make_result(
            name, status="found_existing", duration_sec=dur, output_path=str(finrl_run)
        )

    # Train fresh
    data_rel = _UNIVERSE_DATA_FILE.get(args.universe, "")
    tickers = ",".join(_UNIVERSE_TICKERS.get(args.universe, []))
    seed_list = ",".join(str(i) for i in range(1, args.seeds + 1))
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "PYTHONUNBUFFERED": "1",
            "STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE": args.universe,
            "STOCK_INVESTMENT_DSS_DAILY_DATASET_ID": args.universe,
            "STOCK_INVESTMENT_DSS_DAILY_DATA_START": "2010-01-01",
            "STOCK_INVESTMENT_DSS_DAILY_DATA_END": "2026-12-31",
            "STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE": data_rel,
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE": "true",
            "STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD": "false",
            "STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD": "false",
            "STOCK_INVESTMENT_DSS_FINRL_TICKERS": tickers,
            "STOCK_INVESTMENT_DSS_PIT_SPLIT_ID": f"{args.universe}_pit",
            "STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME": args.pit_decision_date,
            "STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE": "2026-12-31",
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS": "a2c,ppo,ddpg,sac,td3",
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO": "true",
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_TOTAL_TIMESTEPS": str(
                args.finrl_timesteps
            ),
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_DEVICE": "auto",
            "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_LIST": seed_list,
            "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_STOP_ON_FAILURE": "false",
            "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_RUN_SUMMARY_AFTER": "true",
            "STOCK_INVESTMENT_DSS_FINRL_ENV_INITIAL_AMOUNT": "1000000",
            "STOCK_INVESTMENT_DSS_FINRL_ENV_HMAX": "10000",
            "STOCK_INVESTMENT_DSS_FINRL_ENV_BUY_COST_PCT": "0.001",
            "STOCK_INVESTMENT_DSS_FINRL_ENV_SELL_COST_PCT": "0.001",
            "STOCK_INVESTMENT_DSS_FINRL_ENV_REWARD_SCALING": "0.0001",
        }
    )
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "stock_investment_dss.runner.run_finrl_baseline_multiseed_launcher",
    ]
    log_path = run_paths.logs_directory / "step_03_finrl.log"
    before_ts = datetime.now()
    rc = stream_subprocess(cmd, repo_root, env, label="FINRL", log_path=log_path)

    output_path = "n/a"
    try:
        out = find_run_created_after(
            RUNS_DIRECTORY, "finrl_baseline_multiseed_summary", before_ts
        )
        output_path = str(out)
    except FileNotFoundError:
        pass

    dur = time.monotonic() - t0
    status = "ok" if rc == 0 else "failed"
    logger.info("FinRL: %s  rc=%d  %.1fs", status, rc, dur)
    return _make_result(
        name,
        status=status,
        returncode=rc,
        duration_sec=dur,
        output_path=output_path,
        log_path=str(log_path),
        command=" ".join(cmd),
    )


def step_iqn(
    args: argparse.Namespace,
    repo_root: Path,
    run_paths,
    logger: logging.Logger,
) -> StepResult:
    """Resolve the IQN checkpoint and validate the existing multiseed summary run."""
    t0 = time.monotonic()
    name = "iqn"
    _banner("STEP 4 / 8 — IQN INFERENCE", char="-")

    # Resolve checkpoint for reproducibility logging
    ckpt_path_str = args.iqn_checkpoint
    try:
        ckpt = resolve_iqn_checkpoint(args.iqn_checkpoint, repo_root)
        ckpt_path_str = str(ckpt)
        logger.info("IQN checkpoint: %s", ckpt)
        print(f"[{name}] Checkpoint: {ckpt}", flush=True)
    except (FileNotFoundError, NotImplementedError, ValueError) as exc:
        logger.error("IQN checkpoint resolution failed: %s", exc)
        print(f"[{name}] ERROR: {exc}", flush=True)
        return _make_result(
            name, status="failed", returncode=1, duration_sec=time.monotonic() - t0
        )

    # Validate multiseed summary run (required by build_summary_dashboard)
    try:
        iqn_run = _find_latest_run(
            RUNS_DIRECTORY, "iqn_learning_curve_multiseed_summary"
        )
    except FileNotFoundError as exc:
        logger.error("IQN multiseed summary not found: %s", exc)
        print(f"[{name}] ERROR: {exc}", flush=True)
        return _make_result(
            name, status="failed", returncode=1, duration_sec=time.monotonic() - t0
        )

    eval_csv = iqn_run / "data" / "iqn_learning_curve_multiseed_eval_records.csv"
    if not eval_csv.exists():
        msg = f"IQN eval records not found: {eval_csv}"
        logger.error(msg)
        print(f"[{name}] ERROR: {msg}", flush=True)
        return _make_result(
            name,
            status="failed",
            returncode=1,
            duration_sec=time.monotonic() - t0,
            output_path=str(iqn_run),
        )

    dur = time.monotonic() - t0
    logger.info(
        "IQN: found summary run %s  checkpoint: %s  (%.1fs)",
        iqn_run.name,
        ckpt_path_str,
        dur,
    )
    print(f"[{name}] Summary run: {iqn_run.name}  ({dur:.1f}s)", flush=True)
    return _make_result(
        name, status="found_existing", duration_sec=dur, output_path=str(iqn_run)
    )


def step_decision_dashboard(
    run_paths,
    logger: logging.Logger,
) -> StepResult:
    """Step 5: decision dashboard — skipped (output is embedded in IQN run directory)."""
    name = "decision_dashboard"
    _banner("STEP 5 / 8 — DECISION DASHBOARD (skipped)", char="-")
    msg = (
        "Decision dashboard visualizations are part of the IQN learning curve "
        "run output. No standalone rendering script is registered for this step."
    )
    logger.warning(msg)
    print(f"[{name}] SKIPPED — {msg}", flush=True)
    return _make_result(name, status="skipped")


def step_summary_dashboard(
    repo_root: Path,
    run_paths,
    logger: logging.Logger,
) -> StepResult:
    """Run build_summary_dashboard.py (etape 5) to produce the 4-panel comparison figure."""
    t0 = time.monotonic()
    name = "summary_dashboard"
    _banner("STEP 6 / 8 — SUMMARY DASHBOARD", char="-")

    env = os.environ.copy()
    env.update({"PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"})
    cmd = [
        sys.executable,
        "-u",
        str(repo_root / "scripts" / "build_summary_dashboard.py"),
    ]
    log_path = run_paths.logs_directory / "step_06_summary_dashboard.log"
    before_ts = datetime.now()
    rc = stream_subprocess(
        cmd, repo_root, env, label="SUMMARY_DASHBOARD", log_path=log_path
    )

    output_path = "n/a"
    if rc == 0:
        try:
            out = find_run_created_after(
                RUNS_DIRECTORY, "d_iqn_dss_summary_dashboard", before_ts
            )
            output_path = str(out)
        except FileNotFoundError:
            pass

    dur = time.monotonic() - t0
    status = "ok" if rc == 0 else "failed"
    logger.info("Summary dashboard: %s  rc=%d  %.1fs", status, rc, dur)
    return _make_result(
        name,
        status=status,
        returncode=rc,
        duration_sec=dur,
        output_path=output_path,
        log_path=str(log_path),
        command=" ".join(cmd),
    )


def step_comparison_report(
    repo_root: Path,
    run_paths,
    logger: logging.Logger,
) -> StepResult:
    """Run build_comparison_report.py (etape 6) to produce the strategy ranking report."""
    t0 = time.monotonic()
    name = "comparison_report"
    _banner("STEP 7 / 8 — COMPARISON REPORT", char="-")

    env = os.environ.copy()
    env.update({"PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"})
    cmd = [
        sys.executable,
        "-u",
        str(repo_root / "scripts" / "build_comparison_report.py"),
    ]
    log_path = run_paths.logs_directory / "step_07_comparison_report.log"
    before_ts = datetime.now()
    rc = stream_subprocess(
        cmd, repo_root, env, label="COMPARISON_REPORT", log_path=log_path
    )

    output_path = "n/a"
    if rc == 0:
        try:
            out = find_run_created_after(
                RUNS_DIRECTORY, "d_iqn_dss_comparison_report", before_ts
            )
            output_path = str(out)
        except FileNotFoundError:
            pass

    dur = time.monotonic() - t0
    status = "ok" if rc == 0 else "failed"
    logger.info("Comparison report: %s  rc=%d  %.1fs", status, rc, dur)
    return _make_result(
        name,
        status=status,
        returncode=rc,
        duration_sec=dur,
        output_path=output_path,
        log_path=str(log_path),
        command=" ".join(cmd),
    )


def step_master_summary(
    args: argparse.Namespace,
    results: list,
    repo_root: Path,
    run_paths,
    logger: logging.Logger,
    comparison_report_path: Optional[Path],
    dataset_sha256: str,
    iqn_checkpoint_path: str,
    pipeline_start: datetime,
) -> StepResult:
    """Write demo_master_summary.md, .json, subprocess_results.csv, final_comparison.md."""
    t0 = time.monotonic()
    name = "master_summary"
    _banner("STEP 8 / 8 — MASTER SUMMARY", char="-")

    # subprocess_results.csv
    csv_path = run_paths.summary_directory / "subprocess_results.csv"
    fields = [
        "name",
        "status",
        "returncode",
        "duration_sec",
        "output_path",
        "log_path",
        "command",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fields})

    # Extract IQN ranking from comparison.csv if available
    iqn_info: dict = {}
    if comparison_report_path is not None:
        comp_csv = comparison_report_path / "summary" / "comparison.csv"
        if comp_csv.exists():
            with open(comp_csv, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    src = row.get("source", "").lower()
                    if "d-iqn" in src or "iqn" in src:
                        iqn_info = {
                            k: row.get(k, "")
                            for k in (
                                "strategy",
                                "combined_rank",
                                "total_return_pct",
                                "annualized_sharpe",
                                "max_drawdown_pct",
                                "rank_return",
                                "rank_sharpe",
                                "rank_drawdown",
                            )
                        }
                        break

    def _icon(s: str) -> str:
        return {"ok": "✅", "found_existing": "✅", "skipped": "⏭", "failed": "❌"}.get(
            s, "?"
        )

    total_dur = sum(float(r.get("duration_sec", 0)) for r in results)
    table_rows = "\n".join(
        f"| {i + 1}. {r['name']} | {_icon(r['status'])} {r['status']} "
        f"| {float(r.get('duration_sec', 0)):.1f}s "
        f"| {Path(r['output_path']).name if r.get('output_path', 'n/a') != 'n/a' else 'n/a'} |"
        for i, r in enumerate(results)
    )

    iqn_block = ""
    if iqn_info:
        iqn_block = (
            "## D-IQN-DSS Performance\n\n"
            f"- Combined rank: {iqn_info.get('combined_rank', 'n/a')}\n"
            f"- Total return: {iqn_info.get('total_return_pct', 'n/a')}%\n"
            f"- Annualized Sharpe: {iqn_info.get('annualized_sharpe', 'n/a')}\n"
            f"- Max drawdown: {iqn_info.get('max_drawdown_pct', 'n/a')}%\n"
            f"- Rank by return: {iqn_info.get('rank_return', 'n/a')}\n"
            f"- Rank by Sharpe: {iqn_info.get('rank_sharpe', 'n/a')}\n"
            f"- Rank by drawdown: {iqn_info.get('rank_drawdown', 'n/a')}\n\n"
        )

    md_content = (
        "# D-IQN-DSS Demo Master Summary\n\n"
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Universe: {args.universe}\n"
        f"PIT split: {args.pit_decision_date}\n"
        f"Total pipeline duration: {int(total_dur // 60)}m {int(total_dur % 60)}s\n\n"
        "## Pipeline Execution\n\n"
        "| Step | Status | Duration | Output |\n"
        "|------|--------|----------|--------|\n"
        f"{table_rows}\n\n"
        f"{iqn_block}"
        "## Reproducibility\n\n"
        f"- Dataset file: {_UNIVERSE_DATA_FILE.get(args.universe, 'n/a')}\n"
        f"- Dataset SHA-256: {dataset_sha256[:32]}…\n"
        f"- IQN checkpoint: {iqn_checkpoint_path}\n"
        f"- Run directory: {run_paths.run_directory}\n"
        f"- Pipeline start: {pipeline_start:%Y-%m-%d %H:%M:%S}\n"
    )
    (run_paths.summary_directory / "demo_master_summary.md").write_text(
        md_content, encoding="utf-8"
    )

    summary_json = {
        "generated": datetime.now().isoformat(),
        "universe": args.universe,
        "pit_decision_date": args.pit_decision_date,
        "total_duration_sec": total_dur,
        "pipeline_start": pipeline_start.isoformat(),
        "dataset_sha256": dataset_sha256,
        "iqn_checkpoint": iqn_checkpoint_path,
        "run_directory": str(run_paths.run_directory),
        "steps": results,
        "iqn_performance": iqn_info,
    }
    (run_paths.summary_directory / "demo_master_summary.json").write_text(
        json.dumps(summary_json, indent=2, default=str), encoding="utf-8"
    )

    if comparison_report_path is not None:
        src_md = comparison_report_path / "summary" / "comparison.md"
        if src_md.exists():
            shutil.copy2(src_md, run_paths.summary_directory / "final_comparison.md")
            logger.info("Copied comparison.md → final_comparison.md")

    dur = time.monotonic() - t0
    logger.info("Master summary written (%.1fs)", dur)
    md_out = run_paths.summary_directory / "demo_master_summary.md"
    print(f"[{name}] Summary: {md_out}  ({dur:.1f}s)", flush=True)
    return _make_result(name, status="ok", duration_sec=dur, output_path=str(md_out))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main() -> int:  # noqa: C901 — orchestrator is intentionally long
    args = parse_args()
    repo_root = _PROJECT_ROOT

    run_paths = create_run_paths("d_iqn_dss_demo_baselines_and_iqn")
    logger = setup_run_logger(run_paths)
    pipeline_start = datetime.now()

    _banner("D-IQN-DSS DEMO BASELINES AND IQN — FULL PIPELINE", char="=")
    print(f"  Universe:         {args.universe}", flush=True)
    print(f"  PIT split:        {args.pit_decision_date}", flush=True)
    print(f"  IQN checkpoint:   {args.iqn_checkpoint}", flush=True)
    print(f"  Skip FinRL train: {args.skip_finrl_training}", flush=True)
    print(f"  FinRL timesteps:  {args.finrl_timesteps}", flush=True)
    print(f"  Seeds:            {args.seeds}", flush=True)
    print(f"  Continue-on-err:  {args.continue_on_error}", flush=True)
    print(f"  Run directory:    {run_paths.run_directory}", flush=True)
    print(flush=True)

    # Write config files
    config: dict = {
        "universe": args.universe,
        "pit_decision_date": args.pit_decision_date,
        "iqn_checkpoint": args.iqn_checkpoint,
        "finrl_timesteps": args.finrl_timesteps,
        "skip_finrl_training": args.skip_finrl_training,
        "seeds": args.seeds,
        "continue_on_error": args.continue_on_error,
        "run_directory": str(run_paths.run_directory),
        "pipeline_start": pipeline_start.isoformat(),
    }
    (run_paths.config_directory / "demo_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (run_paths.config_directory / "universe_config.json").write_text(
        json.dumps(
            {
                "universe": args.universe,
                "tickers": _UNIVERSE_TICKERS.get(args.universe, []),
                "pit_split": {
                    "train_end": args.pit_decision_date,
                    "eval_start": args.pit_decision_date,
                },
                "data_file": _UNIVERSE_DATA_FILE.get(args.universe, ""),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # W&B (no-op if disabled)
    init_wandb_run(
        run_name=run_paths.run_id,
        config=config,
        group="thesis-demo",
        job_type="pipeline_orchestrator",
        tags=[args.universe, "v4.0", "demo"],
        run_directory=str(run_paths.run_directory),
    )

    results: list = []
    dataset_sha256 = ""
    iqn_checkpoint_resolved = args.iqn_checkpoint
    comparison_report_path: Optional[Path] = None
    should_continue = True

    try:
        # Step 1: load dataset
        r = step_load_dataset(args, repo_root, run_paths, logger)
        results.append(r)
        if r["status"] == "failed":
            should_continue = args.continue_on_error
        elif r["status"] == "ok":
            hashes_file = Path(r["output_path"])
            if hashes_file.exists():
                dataset_sha256 = json.loads(hashes_file.read_text())["sha256"]

        # Step 2: algorithmic baselines
        if should_continue:
            r = step_algorithmic_baselines(args, repo_root, run_paths, logger)
            results.append(r)
            if r["status"] == "failed":
                should_continue = args.continue_on_error

        # Step 3: FinRL baselines
        if should_continue:
            r = step_finrl_baselines(args, repo_root, run_paths, logger)
            results.append(r)
            if r["status"] == "failed":
                should_continue = args.continue_on_error

        # Step 4: IQN
        if should_continue:
            r = step_iqn(args, repo_root, run_paths, logger)
            results.append(r)
            if r["status"] == "failed":
                should_continue = args.continue_on_error
            elif r["status"] == "found_existing":
                try:
                    iqn_checkpoint_resolved = str(
                        resolve_iqn_checkpoint(args.iqn_checkpoint, repo_root)
                    )
                except Exception:  # noqa: BLE001
                    pass

        # Step 5: decision dashboard (always skipped in v1)
        results.append(step_decision_dashboard(run_paths, logger))

        # Step 6: summary dashboard
        if should_continue:
            r = step_summary_dashboard(repo_root, run_paths, logger)
            results.append(r)
            if r["status"] == "failed":
                should_continue = args.continue_on_error

        # Step 7: comparison report
        if should_continue:
            r = step_comparison_report(repo_root, run_paths, logger)
            results.append(r)
            if r["output_path"] != "n/a":
                comparison_report_path = Path(r["output_path"])
            if r["status"] == "failed":
                should_continue = args.continue_on_error

    finally:
        # Step 8: master summary always runs
        r8 = step_master_summary(
            args,
            results,
            repo_root,
            run_paths,
            logger,
            comparison_report_path,
            dataset_sha256,
            iqn_checkpoint_resolved,
            pipeline_start,
        )
        results.append(r8)

    return _finalize(
        results,
        run_paths,
        logger,
        pipeline_start,
        comparison_report_path,
        dataset_sha256,
        args,
    )


def _finalize(
    results: list,
    run_paths,
    logger: logging.Logger,
    pipeline_start: datetime,
    comparison_report_path: Optional[Path],
    dataset_sha256: str,
    args: argparse.Namespace,
) -> int:
    pipeline_end = datetime.now()
    total_sec = (pipeline_end - pipeline_start).total_seconds()

    # audit/pipeline_execution.json
    execution = {
        "pipeline_start": pipeline_start.isoformat(),
        "pipeline_end": pipeline_end.isoformat(),
        "total_duration_sec": total_sec,
        "steps": results,
    }
    (run_paths.audit_directory / "pipeline_execution.json").write_text(
        json.dumps(execution, indent=2, default=str), encoding="utf-8"
    )

    # W&B metrics
    failed = [r for r in results if r.get("status") == "failed"]
    wb_data: dict = {
        "pipeline_total_sec": total_sec,
        "steps_total": len(results),
        "steps_ok": sum(
            1 for r in results if r.get("status") in ("ok", "found_existing")
        ),
        "steps_failed": len(failed),
        "steps_skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "dataset_sha256_prefix": dataset_sha256[:16] if dataset_sha256 else "",
    }
    for r in results:
        wb_data[f"step_{r['name']}_status"] = r.get("status", "")
        wb_data[f"step_{r['name']}_duration_sec"] = float(r.get("duration_sec", 0))

    if comparison_report_path is not None:
        comp_csv = comparison_report_path / "summary" / "comparison.csv"
        if comp_csv.exists():
            with open(comp_csv, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if (
                        "d-iqn" in row.get("source", "").lower()
                        or "iqn" in row.get("strategy", "").lower()
                    ):
                        for key in (
                            "combined_rank",
                            "rank_return",
                            "rank_sharpe",
                            "rank_drawdown",
                        ):
                            try:
                                wb_data[f"iqn_{key}"] = float(row.get(key, ""))
                            except (ValueError, TypeError):
                                pass
                        break

    wandb_log(wb_data)

    # Final banner
    def _icon(s: str) -> str:
        return {"ok": "✅", "found_existing": "✅", "skipped": "⏭", "failed": "❌"}.get(
            s, "?"
        )

    _banner("FINAL SUMMARY — D-IQN-DSS DEMO PIPELINE", char="=")
    print(f"  Run directory: {run_paths.run_directory}", flush=True)
    print(
        f"  Total runtime: {int(total_sec // 60)}m {int(total_sec % 60)}s", flush=True
    )
    print(flush=True)
    print(f"  {'Step':<28} {'Status':<18} {'Duration':>10}", flush=True)
    print(f"  {'-' * 28} {'-' * 18} {'-' * 10}", flush=True)
    for r in results:
        icon = _icon(r.get("status", ""))
        status_str = f"{icon} {r.get('status', '')}"
        dur = float(r.get("duration_sec", 0))
        print(f"  {r['name']:<28} {status_str:<18} {dur:>9.1f}s", flush=True)
    print(flush=True)

    md_path = run_paths.summary_directory / "demo_master_summary.md"
    if md_path.exists():
        print(f"  Master summary: {md_path}", flush=True)
    print(flush=True)

    try:
        finish_wandb_run()
    except Exception:  # noqa: BLE001
        pass

    if failed:
        _banner(f"{len(failed)} STEP(S) FAILED — review logs above", char="!")
        return 1

    _banner("ALL STEPS PASSED", char="=")
    return 0


if __name__ == "__main__":
    sys.exit(main())
