# src/stock_investment_dss/runner/run_finrl_baseline_seed_diagnostics.py
"""Diagnose whether FinRL/SB3 baseline seeds affect final outcomes.

This runner is intentionally lightweight. It reads the latest
``finrl_baseline_multiseed_summary`` output and checks whether each strategy has
variation across seeds in return, Sharpe, drawdown, trades, costs, and, when
available, action magnitude / trading status columns.

Why this exists
---------------
In early experiments some agents produced identical final returns across seeds.
That can happen for several reasons:

1. the seed is not passed deeply enough into SB3/FinRL,
2. the two-asset environment leads to identical learned policies,
3. policies collapse to identical action patterns,
4. deterministic components dominate the result,
5. MVO is deterministic by construction in this setup.

This diagnostic does not prove which reason is true, but it makes the issue
explicit and audit-friendly.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

RUN_KIND = "finrl_baseline_seed_diagnostics"
PROJECT_NAME = "StockInvestmentDSS"
PROTOTYPE_NAME = "D-IQN-DSS"
DEFAULT_RECENT_RUN_LIMIT = 120

METRICS_TO_CHECK = [
    "total_return_pct",
    "annualized_sharpe",
    "max_drawdown_pct",
    "cvar_pct",
    "final_value",
    "total_transaction_cost",
    "total_trades",
    "turnover_estimate_pct",
    "action_mean_abs",
    "action_max_abs",
    "non_zero_action_steps",
]


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


def get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def find_latest_multiseed_summary(project_root: Path, recent_limit: int) -> Path | None:
    runs_dir = project_root / "outputs" / "runs"
    candidates = sorted(
        [p for p in runs_dir.glob("*_finrl_baseline_multiseed_summary") if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )[:recent_limit]
    for run_dir in candidates:
        member_path = run_dir / "data" / "finrl_baseline_multiseed_member_records.csv"
        if member_path.exists():
            return run_dir
    return None


def is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    try:
        if isinstance(value, float) and math.isnan(value):
            return False
    except TypeError:
        pass
    text = str(value).strip()
    return text != "" and text.lower() not in {"nan", "none", "null"}


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def main() -> int:
    project_root = find_project_root()
    run_id = now_run_id()
    run_dir = project_root / "outputs" / "runs" / run_id
    data_dir = run_dir / "data"
    summary_dir = run_dir / "summary"
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    recent_limit = get_int_env("STOCK_INVESTMENT_DSS_RECENT_RUN_LIMIT", DEFAULT_RECENT_RUN_LIMIT)

    log("Starting StockInvestmentDSS FinRL baseline seed diagnostics.")
    log(f"Project root: {project_root}")
    log(f"Created run directory: {run_dir}")
    log(f"Run id: {run_id}")

    source_run = find_latest_multiseed_summary(project_root, recent_limit)
    if source_run is None:
        raise FileNotFoundError("No finrl_baseline_multiseed_summary run with member records was found.")

    member_path = source_run / "data" / "finrl_baseline_multiseed_member_records.csv"
    member_records = pd.read_csv(member_path)

    if "strategy" not in member_records.columns and "agent_name" in member_records.columns:
        member_records["strategy"] = member_records["agent_name"]
    if "agent_name" not in member_records.columns and "strategy" in member_records.columns:
        member_records["agent_name"] = member_records["strategy"]

    diagnostics: list[dict[str, Any]] = []
    for strategy, group in member_records.groupby("strategy", dropna=False):
        row: dict[str, Any] = {
            "strategy": strategy,
            "row_count": int(len(group)),
            "seed_count": int(group["seed"].nunique()) if "seed" in group.columns else None,
            "seeds": ",".join(str(int(s)) for s in sorted(pd.to_numeric(group.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().unique())),
        }
        zero_std_metrics: list[str] = []
        nonzero_std_metrics: list[str] = []
        missing_metrics: list[str] = []
        for metric in METRICS_TO_CHECK:
            if metric not in group.columns:
                missing_metrics.append(metric)
                continue
            values = to_numeric(group[metric]).dropna()
            if values.empty:
                missing_metrics.append(metric)
                continue
            row[f"{metric}_count"] = int(values.count())
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0)) if values.count() > 1 else 0.0
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
            if values.count() > 1 and abs(float(values.std(ddof=0))) <= 1e-12:
                zero_std_metrics.append(metric)
            elif values.count() > 1:
                nonzero_std_metrics.append(metric)
        if "trading_status" in group.columns:
            row["trading_status_values"] = ",".join(sorted(str(v) for v in group["trading_status"].dropna().unique()))
        row["zero_std_metrics"] = ",".join(zero_std_metrics)
        row["nonzero_std_metrics"] = ",".join(nonzero_std_metrics)
        row["missing_metrics"] = ",".join(missing_metrics)
        if str(strategy).lower() == "mvo":
            row["interpretation"] = "MVO is expected to be deterministic under this setup; zero std is acceptable."
        elif zero_std_metrics and not nonzero_std_metrics:
            row["interpretation"] = "All available checked metrics are identical across seeds; inspect seed propagation or action patterns."
        elif zero_std_metrics:
            row["interpretation"] = "Some metrics are identical across seeds, but other metrics vary; inspect action patterns and seed propagation if needed."
        else:
            row["interpretation"] = "Available checked metrics vary across seeds."
        diagnostics.append(row)

    diagnostic_table = pd.DataFrame(diagnostics).sort_values("strategy")
    out_csv = data_dir / "finrl_baseline_seed_diagnostics.csv"
    out_md = summary_dir / "finrl_baseline_seed_diagnostics.md"
    summary_path = summary_dir / "finrl_baseline_seed_diagnostics_summary.json"
    copied_member_path = data_dir / "finrl_baseline_multiseed_member_records_copy.csv"

    diagnostic_table.to_csv(out_csv, index=False)
    member_records.to_csv(copied_member_path, index=False)

    lines = [
        "# FinRL baseline seed diagnostics",
        "",
        f"Source run: `{source_run.name}`",
        "",
        "This report checks whether final metrics vary across seeds. Zero std is expected for deterministic MVO, but suspicious for stochastic RL agents unless the environment/policy collapses to identical behavior.",
        "",
        diagnostic_table[["strategy", "seed_count", "zero_std_metrics", "nonzero_std_metrics", "interpretation"]].to_markdown(index=False),
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "status": "ok",
        "project_name": PROJECT_NAME,
        "prototype_name": PROTOTYPE_NAME,
        "run_id": run_id,
        "source_run_id": source_run.name,
        "source_member_records_path": str(member_path),
        "strategy_count": int(diagnostic_table["strategy"].nunique()) if not diagnostic_table.empty else 0,
        "outputs": {
            "diagnostic_table_path": str(out_csv),
            "markdown_report_path": str(out_md),
            "member_records_copy_path": str(copied_member_path),
            "summary_path": str(summary_path),
        },
        "interpretation": "Use this diagnostic to decide whether zero standard deviation across seeds is expected/deterministic or requires deeper seed/action-pattern inspection.",
    }
    write_json(summary_path, summary)

    log("FinRL baseline seed diagnostics completed.")
    log(f"Source run: {source_run.name}")
    log(f"Wrote diagnostics: {out_csv}")
    log(f"Wrote report: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
