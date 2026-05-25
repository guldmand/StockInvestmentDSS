# src/stock_investment_dss/runner/run_finrl_baseline_learning_curve_multiseed_summary.py
"""Aggregate FinRL/SB3 baseline learning-budget curves across seeds.

This summary runner reads outputs from
``run_finrl_baseline_learning_curve_multiseed_launcher.py`` and creates
learning-curve style diagnostics for FinRL baselines.

Important interpretation
------------------------
These are training-budget curves, not true checkpoint curves. Each point is a
separately trained run with a fixed training budget. This is still useful for
answering whether performance tends to improve as the training budget increases.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUN_KIND = "finrl_baseline_learning_curve_multiseed_summary"
PROJECT_NAME = "StockInvestmentDSS"
PROTOTYPE_NAME = "D-IQN-DSS"
DEFAULT_RECENT_RUN_LIMIT = 200
DEFAULT_SEEDS = "1,2,3,4,5"
DEFAULT_AGENTS = "a2c,ddpg,td3,ppo,sac"
DEFAULT_TRAIN_STEPS = "5000,10000,15000,20000,25000"

AGENT_ORDER = ["a2c", "ddpg", "td3", "ppo", "sac", "mvo"]
METRICS = [
    "final_value",
    "total_return_pct",
    "annualized_sharpe",
    "max_drawdown_pct",
    "cvar_pct",
    "annualized_volatility_pct",
    "total_transaction_cost",
    "total_trades",
    "turnover_estimate_pct",
]

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


def get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


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


def parse_agent_list(raw: str, include_mvo: bool) -> list[str]:
    agents = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if include_mvo and "mvo" not in agents:
        agents.append("mvo")
    return agents


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def safe_read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def normalize_agent(value: Any) -> str:
    return str(value).strip().lower()


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def read_suite_comparison(run_dir: Path) -> pd.DataFrame | None:
    path = first_existing(
        [
            run_dir / "data" / "finrl_baseline_suite" / "finrl_baseline_suite_comparison.csv",
            run_dir / "summary" / "finrl_baseline_suite_comparison_snapshot.csv",
        ]
    )
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    return df


def find_latest_learning_curve_launcher(project_root: Path, recent_limit: int) -> Path | None:
    runs_dir = project_root / "outputs" / "runs"
    candidates = sorted(
        [p for p in runs_dir.glob("*_finrl_baseline_learning_curve_multiseed_launcher") if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )[:recent_limit]
    for run_dir in candidates:
        summary_path = run_dir / "summary" / "finrl_baseline_learning_curve_multiseed_launcher_summary.json"
        launched_csv = run_dir / "data" / "finrl_baseline_learning_curve_launched_runs.csv"
        if summary_path.exists() or launched_csv.exists():
            return run_dir
    return None


def collect_mapping_from_launcher(launcher_dir: Path) -> pd.DataFrame:
    launched_csv = launcher_dir / "data" / "finrl_baseline_learning_curve_launched_runs.csv"
    if launched_csv.exists():
        df = pd.read_csv(launched_csv)
    else:
        summary = safe_read_json(launcher_dir / "summary" / "finrl_baseline_learning_curve_multiseed_launcher_summary.json")
        df = pd.DataFrame(summary.get("launched_runs", []))
    if df.empty:
        return df
    if "child_run_id" not in df.columns:
        df["child_run_id"] = None
    # Fallback parse log files when child_run_id is missing.
    for idx, row in df.iterrows():
        if str(row.get("child_run_id", "")).strip() and str(row.get("child_run_id", "")).lower() != "nan":
            continue
        log_path = Path(str(row.get("log_path", "")))
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            matches = RUN_ID_RE.findall(text)
            if matches:
                df.at[idx, "child_run_id"] = matches[-1]
    return df


def add_context_from_summary(row: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    summary = safe_read_json(run_dir / "summary" / "finrl_baseline_suite_smoke_summary.json")
    for key in [
        "dataset_id",
        "pit_split_id",
        "universe_id",
        "point_in_time",
        "trade_end_date",
    ]:
        if not row.get(key):
            value = summary.get(key)
            if value is not None:
                row[key] = value
    # common alternative nesting/fallbacks
    if not row.get("dataset_id"):
        row["dataset_id"] = os.getenv("STOCK_INVESTMENT_DSS_COMPARISON_DATASET_ID")
    if not row.get("pit_split_id"):
        row["pit_split_id"] = os.getenv("STOCK_INVESTMENT_DSS_COMPARISON_PIT_SPLIT_ID")
    if not row.get("point_in_time"):
        row["point_in_time"] = os.getenv("STOCK_INVESTMENT_DSS_COMPARISON_POINT_IN_TIME")
    if not row.get("trade_end_date"):
        row["trade_end_date"] = os.getenv("STOCK_INVESTMENT_DSS_COMPARISON_TRADE_END_DATE")
    return row


def build_member_records(project_root: Path, mapping: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    runs_dir = project_root / "outputs" / "runs"
    for _, map_row in mapping.iterrows():
        child_run_id = str(map_row.get("child_run_id", "")).strip()
        if not child_run_id or child_run_id.lower() == "nan":
            continue
        run_dir = runs_dir / child_run_id
        if not run_dir.exists():
            continue
        comp = read_suite_comparison(run_dir)
        if comp is None:
            continue
        seed = map_row.get("seed")
        train_step = map_row.get("train_step")
        for _, comp_row in comp.iterrows():
            row: dict[str, Any] = comp_row.to_dict()
            strategy = row.get("agent_name", row.get("agent", row.get("strategy")))
            row["strategy"] = normalize_agent(strategy)
            row["agent_name"] = row["strategy"]
            row["seed"] = int(seed) if pd.notna(seed) else None
            row["train_step"] = int(train_step) if pd.notna(train_step) else None
            row["configured_total_steps"] = row["train_step"]
            row["source_run_id"] = child_run_id
            row["source_run_directory"] = str(run_dir)
            row["source"] = "FinRL / SB3 baseline learning curve"
            row["model_family"] = (
                "classical_portfolio_optimization" if row["strategy"] == "mvo" else "parametric_rl_expected_return"
            )
            row["variant"] = "learning_curve_budget_member"
            row = add_context_from_summary(row, run_dir)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def filter_records(df: pd.DataFrame, seeds: list[int], agents: list[str], train_steps: list[int]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["strategy"] = out["strategy"].map(normalize_agent)
    out = out[out["strategy"].isin(agents)]
    out = out[pd.to_numeric(out["seed"], errors="coerce").isin(seeds)]
    out = out[pd.to_numeric(out["train_step"], errors="coerce").isin(train_steps)]
    # Known context mismatches rejected; missing context tolerated.
    context_filters = {
        "dataset_id": os.getenv("STOCK_INVESTMENT_DSS_COMPARISON_DATASET_ID"),
        "pit_split_id": os.getenv("STOCK_INVESTMENT_DSS_COMPARISON_PIT_SPLIT_ID"),
        "point_in_time": os.getenv("STOCK_INVESTMENT_DSS_COMPARISON_POINT_IN_TIME"),
        "trade_end_date": os.getenv("STOCK_INVESTMENT_DSS_COMPARISON_TRADE_END_DATE"),
    }
    for col, expected in context_filters.items():
        if not expected or col not in out.columns:
            continue
        values = out[col].fillna("").astype(str).str.strip()
        keep = (values == "") | (values == expected)
        out = out[keep]
    out = out.sort_values(["strategy", "seed", "train_step", "source_run_id"])
    out = out.drop_duplicates(subset=["strategy", "seed", "train_step"], keep="last")
    return out.reset_index(drop=True)


def aggregate_records(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    working = df.copy()
    for metric in METRICS:
        if metric in working.columns:
            working[metric] = pd.to_numeric(working[metric], errors="coerce")
    grouped = working.groupby(["strategy", "train_step"], dropna=False)
    rows: list[dict[str, Any]] = []
    for (strategy, train_step), group in grouped:
        row: dict[str, Any] = {
            "strategy": strategy,
            "agent_name": strategy,
            "train_step": int(train_step),
            "seed_count": int(group["seed"].nunique()) if "seed" in group.columns else None,
            "seeds": ",".join(str(int(s)) for s in sorted(pd.to_numeric(group["seed"], errors="coerce").dropna().unique())),
            "source": "FinRL / SB3 baseline learning curve",
            "model_family": "classical_portfolio_optimization" if strategy == "mvo" else "parametric_rl_expected_return",
            "variant": "learning_curve_budget_mean",
        }
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_count"] = int(values.count())
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else None
            row[f"{metric}_std"] = float(values.std(ddof=0)) if values.count() > 1 else 0.0 if values.count() == 1 else None
            row[f"{metric}_min"] = float(values.min()) if not values.empty else None
            row[f"{metric}_max"] = float(values.max()) if not values.empty else None
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {agent: idx for idx, agent in enumerate(AGENT_ORDER)}
    out["_order"] = out["strategy"].map(lambda x: order.get(str(x), 999))
    out = out.sort_values(["_order", "strategy", "train_step"]).drop(columns=["_order"])
    return out.reset_index(drop=True)


def final_by_strategy(aggregate: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        return aggregate
    rows = []
    for strategy, group in aggregate.groupby("strategy", dropna=False):
        max_step = pd.to_numeric(group["train_step"], errors="coerce").max()
        rows.append(group[pd.to_numeric(group["train_step"], errors="coerce") == max_step].tail(1))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_metric(aggregate: pd.DataFrame, metric: str, ylabel: str, title: str, output_path: Path) -> bool:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if aggregate.empty or mean_col not in aggregate.columns:
        return False
    plot_df = aggregate.dropna(subset=[mean_col]).copy()
    if plot_df.empty:
        return False
    fig, ax = plt.subplots(figsize=(12, 7))
    for strategy in [a for a in AGENT_ORDER if a in set(plot_df["strategy"])] + [
        a for a in sorted(set(plot_df["strategy"])) if a not in AGENT_ORDER
    ]:
        group = plot_df[plot_df["strategy"] == strategy].sort_values("train_step")
        x = pd.to_numeric(group["train_step"], errors="coerce")
        y = pd.to_numeric(group[mean_col], errors="coerce")
        ax.plot(x, y, marker="o", linewidth=2, label=strategy.upper())
        if std_col in group.columns:
            std = pd.to_numeric(group[std_col], errors="coerce").fillna(0.0)
            if (std > 0).any():
                ax.fill_between(x, y - std, y + std, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Training budget / timesteps")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def main() -> int:
    project_root = find_project_root()
    run_id = now_run_id()
    run_dir = project_root / "outputs" / "runs" / run_id
    data_dir = run_dir / "data"
    summary_dir = run_dir / "summary"
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    recent_limit = get_int_env("STOCK_INVESTMENT_DSS_RECENT_RUN_LIMIT", DEFAULT_RECENT_RUN_LIMIT)
    seeds = parse_int_list(
        get_str_env(
            "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_LIST",
            get_str_env("STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST", DEFAULT_SEEDS),
        )
    )
    train_steps = parse_int_list(get_str_env("STOCK_INVESTMENT_DSS_FINRL_LEARNING_CURVE_TRAIN_STEPS", DEFAULT_TRAIN_STEPS))
    include_mvo = get_bool_env("STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO", True)
    agents = parse_agent_list(get_str_env("STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS", DEFAULT_AGENTS), include_mvo)

    log("Starting StockInvestmentDSS FinRL baseline learning-curve multiseed summary.")
    log(f"Project root: {project_root}")
    log(f"Created run directory: {run_dir}")
    log(f"Run id: {run_id}")
    log(f"Seeds: {seeds}")
    log(f"Train steps: {train_steps}")
    log(f"Agents: {agents}")

    launcher_dir = find_latest_learning_curve_launcher(project_root, recent_limit)
    if launcher_dir is None:
        raise FileNotFoundError("No finrl_baseline_learning_curve_multiseed_launcher output was found.")
    mapping = collect_mapping_from_launcher(launcher_dir)
    member_records = build_member_records(project_root, mapping)
    rows_before_filter = int(len(member_records))
    member_records = filter_records(member_records, seeds, agents, train_steps)
    rows_after_filter = int(len(member_records))
    aggregate = aggregate_records(member_records)
    final_records = final_by_strategy(aggregate)

    member_path = data_dir / "finrl_baseline_learning_curve_member_records.csv"
    aggregate_path = summary_dir / "finrl_baseline_learning_curve_aggregate_by_agent_step.csv"
    final_path = summary_dir / "finrl_baseline_learning_curve_final_by_agent.csv"
    member_records.to_csv(member_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    final_records.to_csv(final_path, index=False)

    plots: list[str] = []
    plot_specs = [
        ("total_return_pct", "Total return (%)", "FinRL baseline learning curve: total return", "finrl_baseline_learning_curve_total_return_mean_std.png"),
        ("annualized_sharpe", "Annualized Sharpe", "FinRL baseline learning curve: Sharpe", "finrl_baseline_learning_curve_sharpe_mean_std.png"),
        ("max_drawdown_pct", "Max drawdown (%)", "FinRL baseline learning curve: max drawdown", "finrl_baseline_learning_curve_max_drawdown_mean_std.png"),
        ("cvar_pct", "CVaR (%)", "FinRL baseline learning curve: CVaR", "finrl_baseline_learning_curve_cvar_mean_std.png"),
        ("final_value", "Final portfolio value", "FinRL baseline learning curve: final value", "finrl_baseline_learning_curve_final_value_mean_std.png"),
    ]
    for metric, ylabel, title, filename in plot_specs:
        path = summary_dir / filename
        if plot_metric(aggregate, metric, ylabel, title, path):
            plots.append(str(path))

    # Markdown report
    md_path = summary_dir / "finrl_baseline_learning_curve_multiseed_summary.md"
    lines = [
        "# FinRL baseline learning-curve multiseed summary",
        "",
        f"Source launcher: `{launcher_dir.name}`",
        "",
        "These are training-budget curves: each checkpoint is trained from scratch with the specified number of timesteps.",
        "",
    ]
    if not final_records.empty:
        display_cols = [
            "strategy",
            "seed_count",
            "train_step",
            "total_return_pct_mean",
            "total_return_pct_std",
            "annualized_sharpe_mean",
            "annualized_sharpe_std",
            "max_drawdown_pct_mean",
            "max_drawdown_pct_std",
            "cvar_pct_mean",
            "cvar_pct_std",
        ]
        existing = [c for c in display_cols if c in final_records.columns]
        lines.append(final_records[existing].to_markdown(index=False))
    md_path.write_text("\n".join(lines), encoding="utf-8")

    final_summary: dict[str, Any] = {}
    for _, row in final_records.iterrows():
        strategy = row.get("strategy")
        final_summary[str(strategy)] = {
            "seed_count": row.get("seed_count"),
            "train_step": row.get("train_step"),
            "total_return_pct_mean": row.get("total_return_pct_mean"),
            "total_return_pct_std": row.get("total_return_pct_std"),
            "annualized_sharpe_mean": row.get("annualized_sharpe_mean"),
            "annualized_sharpe_std": row.get("annualized_sharpe_std"),
            "max_drawdown_pct_mean": row.get("max_drawdown_pct_mean"),
            "max_drawdown_pct_std": row.get("max_drawdown_pct_std"),
            "cvar_pct_mean": row.get("cvar_pct_mean"),
            "cvar_pct_std": row.get("cvar_pct_std"),
        }

    summary_path = summary_dir / "finrl_baseline_learning_curve_multiseed_summary.json"
    summary = {
        "status": "ok",
        "project_name": PROJECT_NAME,
        "prototype_name": PROTOTYPE_NAME,
        "run_id": run_id,
        "project_root": str(project_root),
        "run_directory": str(run_dir),
        "source_launcher_run_id": launcher_dir.name,
        "seeds": seeds,
        "train_steps": train_steps,
        "agents": agents,
        "rows_before_filter": rows_before_filter,
        "rows_after_filter": rows_after_filter,
        "aggregate_row_count": int(len(aggregate)),
        "final_row_count": int(len(final_records)),
        "final_summary": final_summary,
        "outputs": {
            "member_records_path": str(member_path),
            "aggregate_by_agent_step_path": str(aggregate_path),
            "final_by_agent_path": str(final_path),
            "markdown_report_path": str(md_path),
            "plots": plots,
            "summary_path": str(summary_path),
        },
        "interpretation": (
            "This summary approximates learning curves by retraining each FinRL baseline from scratch "
            "for increasing training budgets. For true within-run learning curves, SB3 checkpoint callbacks "
            "would need to be added to the baseline training loop."
        ),
    }
    write_json(summary_path, summary)

    log("FinRL baseline learning-curve multiseed summary completed.")
    log(f"Source launcher: {launcher_dir.name}")
    log(f"Rows before filter: {rows_before_filter}")
    log(f"Rows after filter: {rows_after_filter}")
    log(f"Aggregate rows: {len(aggregate)}")
    log(f"Final rows: {len(final_records)}")
    log(f"Plots: {len(plots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
