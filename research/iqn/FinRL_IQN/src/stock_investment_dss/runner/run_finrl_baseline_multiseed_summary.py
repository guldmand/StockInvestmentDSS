# src/stock_investment_dss/runner/run_finrl_baseline_multiseed_summary.py
"""Aggregate FinRL/SB3 baseline suite runs across multiple seeds.

This runner scans ``outputs/runs`` for ``finrl_baseline_suite_smoke_test`` runs,
reads each run's baseline comparison CSV, filters by seed/context when possible,
and writes mean/std tables per strategy.

The output is intentionally shaped like the IQN multiseed summary so the final
thesis comparison can later use:

- FinRL baseline multiseed mean rows
- MVO deterministic/multiseed row
- D-IQN-DSS IQN multiseed mean row

MVO note
--------
MVO may be included in this summary. If the underlying MVO implementation is
deterministic, the seed rows will be identical and std will be 0. That is not a
problem; it documents that MVO is deterministic under the chosen setup.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUN_KIND = "finrl_baseline_multiseed_summary"
PROJECT_NAME = "StockInvestmentDSS"
PROTOTYPE_NAME = "D-IQN-DSS"
DEFAULT_RECENT_RUN_LIMIT = 160
DEFAULT_SEEDS = "1,2,3,4,5"
DEFAULT_AGENTS = "a2c,ddpg,td3,ppo,sac"

SUMMARY_COLUMNS = [
    "strategy",
    "source",
    "model_family",
    "variant",
    "seed_count",
    "seeds",
    "dataset_id",
    "pit_split_id",
    "universe_id",
    "point_in_time",
    "trade_end_date",
    "configured_total_steps",
    "final_value_mean",
    "final_value_std",
    "total_return_pct_mean",
    "total_return_pct_std",
    "annualized_sharpe_mean",
    "annualized_sharpe_std",
    "max_drawdown_pct_mean",
    "max_drawdown_pct_std",
    "cvar_pct_mean",
    "cvar_pct_std",
    "annualized_volatility_pct_mean",
    "annualized_volatility_pct_std",
    "total_transaction_cost_mean",
    "total_transaction_cost_std",
    "total_trades_mean",
    "total_trades_std",
    "turnover_estimate_pct_mean",
    "turnover_estimate_pct_std",
]

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


def now_run_id() -> str:
    return f"{datetime.now():%Y_%m_%d_%H%M%S}_d_iqn_dss_{RUN_KIND}"


def find_project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src").exists() or (candidate / "outputs").exists():
            return candidate
    return current


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} | INFO | stock_investment_dss.run | {message}")


def get_str_env(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def get_int_env(name: str, default: int) -> int:
    raw = get_str_env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_bool_env(name: str, default: bool) -> bool:
    raw = get_str_env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def parse_seed_list(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    seeds: list[int] = []
    for part in raw.split(","):
        value = part.strip()
        if value:
            seeds.append(int(value))
    return seeds or None


def parse_agent_list(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == "":
        return None
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def recursive_first_present(value: Any, keys: list[str]) -> Any:
    if isinstance(value, dict):
        for key in keys:
            found = value.get(key)
            if found is not None and found != "":
                return found
        for nested in value.values():
            found = recursive_first_present(nested, keys)
            if found is not None and found != "":
                return found
    elif isinstance(value, list):
        for nested in value:
            found = recursive_first_present(nested, keys)
            if found is not None and found != "":
                return found
    return None


def build_launcher_child_run_index(
    project_root: Path, recent_limit: int = 40
) -> dict[str, dict[str, Any]]:
    """Map baseline-suite child run ids to launcher metadata.

    Older/newer baseline-suite summaries may not persist the random seed. The
    multiseed launcher does know which subprocess log belongs to which seed, so
    we recover the child run id from each subprocess log line:

        Run id: <...finrl_baseline_suite_smoke_test>

    This keeps the summary robust without requiring every historical baseline
    suite summary to have identical metadata fields.
    """
    runs_root = project_root / "outputs" / "runs"
    if not runs_root.exists():
        return {}

    launcher_dirs = [
        path
        for path in runs_root.iterdir()
        if path.is_dir() and "finrl_baseline_multiseed_launcher" in path.name
    ]
    launcher_dirs.sort(key=lambda path: path.name, reverse=True)

    index: dict[str, dict[str, Any]] = {}
    run_id_pattern = re.compile(
        r"Run id:\s*([0-9_]+_d_iqn_dss_finrl_baseline_suite_smoke_test)"
    )

    for launcher_dir in launcher_dirs[:recent_limit]:
        summary_path = (
            launcher_dir / "summary" / "finrl_baseline_multiseed_launcher_summary.json"
        )
        summary = read_json(summary_path)
        launched_runs = (
            summary.get("launched_runs")
            if isinstance(summary.get("launched_runs"), list)
            else []
        )
        for launched in launched_runs:
            if not isinstance(launched, dict):
                continue
            seed = launched.get("seed")
            log_path_raw = launched.get("log_path")
            if seed is None or not log_path_raw:
                continue
            log_path = Path(str(log_path_raw))
            if not log_path.exists():
                continue
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            match = run_id_pattern.search(text)
            if not match:
                continue
            child_run_id = match.group(1)
            index[child_run_id] = {
                "seed": int(seed),
                "launcher_run_id": summary.get("run_id") or launcher_dir.name,
                "launcher_run_directory": str(launcher_dir),
                "launcher_summary_path": str(summary_path),
                "launcher_log_path": str(log_path),
            }
    return index


def find_recent_run_dirs(project_root: Path, limit: int) -> list[Path]:
    runs_root = project_root / "outputs" / "runs"
    if not runs_root.exists():
        return []
    candidates = [path for path in runs_root.iterdir() if path.is_dir()]
    candidates.sort(key=lambda path: path.name, reverse=True)
    return candidates[:limit]


def run_id_to_datetime(run_id: str) -> datetime | None:
    try:
        return datetime.strptime("_".join(run_id.split("_")[:6]), "%Y_%m_%d_%H%M%S")
    except Exception:
        return None


def extract_metadata(summary: dict[str, Any], run_directory: Path) -> dict[str, Any]:
    point_in_time_split = (
        summary.get("point_in_time_split")
        if isinstance(summary.get("point_in_time_split"), dict)
        else {}
    )
    config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    seed = (
        summary.get("random_seed")
        or summary.get("seed")
        or summary.get("finrl_seed")
        or config.get("seed")
        or recursive_first_present(
            summary, ["random_seed", "seed", "finrl_seed", "sb3_seed"]
        )
    )
    return {
        "run_id": summary.get("run_id") or run_directory.name,
        "run_datetime": run_id_to_datetime(summary.get("run_id") or run_directory.name),
        "dataset_id": summary.get("dataset_id")
        or recursive_first_present(summary, ["dataset_id"]),
        "pit_split_id": summary.get("pit_split_id")
        or recursive_first_present(summary, ["pit_split_id", "split_id"]),
        "universe_id": summary.get("universe_id")
        or recursive_first_present(summary, ["universe_id"]),
        "point_in_time": summary.get("point_in_time")
        or point_in_time_split.get("point_in_time")
        or recursive_first_present(summary, ["point_in_time"]),
        "trade_end_date": summary.get("trade_end_date")
        or point_in_time_split.get("trade_end_date")
        or recursive_first_present(summary, ["trade_end_date"]),
        "configured_total_steps": summary.get("total_timesteps")
        or summary.get("total_timesteps_per_agent")
        or recursive_first_present(
            summary, ["total_timesteps", "total_timesteps_per_agent"]
        ),
        "seed": int(seed) if str(seed).strip().isdigit() else None,
        "source_run_directory": str(run_directory),
        "summary_path": "",
    }


def read_baseline_suite_run(
    run_directory: Path,
    child_run_index: dict[str, dict[str, Any]] | None = None,
    context_fill: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    summary_path = run_directory / "summary" / "finrl_baseline_suite_smoke_summary.json"
    if not summary_path.exists():
        return []
    summary = read_json(summary_path)
    metadata = extract_metadata(summary, run_directory)
    metadata["summary_path"] = str(summary_path)

    child_info = (child_run_index or {}).get(str(metadata.get("run_id")))
    if child_info:
        metadata["seed"] = child_info.get("seed", metadata.get("seed"))
        metadata["launcher_run_id"] = child_info.get("launcher_run_id")
        metadata["launcher_run_directory"] = child_info.get("launcher_run_directory")
        metadata["launcher_log_path"] = child_info.get("launcher_log_path")
        # If older baseline-suite summaries did not store context metadata,
        # fill missing fields from the active requested comparison context.
        # Known metadata mismatches are still preserved and later filtered out.
        for key, value in (context_fill or {}).items():
            if (
                value is not None
                and value != ""
                and (metadata.get(key) is None or metadata.get(key) == "")
            ):
                metadata[key] = value

    comparison_paths = [
        run_directory
        / "data"
        / "finrl_baseline_suite"
        / "finrl_baseline_suite_comparison.csv",
        run_directory / "summary" / "finrl_baseline_suite_comparison_snapshot.csv",
    ]
    comparison_path = next((path for path in comparison_paths if path.exists()), None)
    if comparison_path is None:
        return []

    try:
        frame = pd.read_csv(comparison_path)
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for _, record in frame.iterrows():
        item = record.to_dict()
        strategy = str(
            first_present(item, ["agent", "agent_name", "strategy"]) or "unknown"
        ).lower()
        row = {
            **metadata,
            "strategy": strategy,
            "source": (
                "MVO baseline" if strategy == "mvo" else "FinRL / SB3 baseline suite"
            ),
            "model_family": (
                "classical_portfolio_optimization"
                if strategy == "mvo"
                else "parametric_rl_expected_return"
            ),
            "variant": "multiseed_member",
            "source_run_id": metadata["run_id"],
            "source_comparison_path": str(comparison_path),
            "start_value": first_present(item, ["initial_value", "start_value"]),
            "final_value": first_present(item, ["final_value", "end_value"]),
            "profit_loss": item.get("profit_loss"),
            "total_return_pct": item.get("total_return_pct")
            or item.get("cumulative_return_pct"),
            "cumulative_return_pct": item.get("cumulative_return_pct")
            or item.get("total_return_pct"),
            "annualized_sharpe": first_present(item, ["annualized_sharpe", "sharpe"]),
            "annualized_volatility_pct": item.get("annualized_volatility_pct"),
            "max_drawdown_pct": item.get("max_drawdown_pct"),
            "cvar_pct": item.get("cvar_pct"),
            "total_transaction_cost": first_present(
                item, ["total_transaction_cost", "finrl_cost"]
            ),
            "total_trades": first_present(item, ["total_trades", "finrl_trades"]),
            "turnover_estimate_pct": item.get("turnover_estimate_pct"),
            "status": item.get("status") or item.get("trading_status"),
        }
        rows.append(row)
    return rows


def metadata_matches(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if expected is None or expected == "":
            continue
        value = row.get(key)
        if value is None or value == "":
            continue
        if str(value) != str(expected):
            return False
    return True


def deduplicate_strategy_seed(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    temp = frame.copy()
    temp["run_datetime_sort"] = pd.to_datetime(temp["run_datetime"], errors="coerce")
    temp = temp.sort_values(["strategy", "seed", "run_datetime_sort", "run_id"])
    temp = temp.drop_duplicates(subset=["strategy", "seed"], keep="last")
    return temp.drop(columns=["run_datetime_sort"], errors="ignore")


def aggregate_by_strategy(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    for strategy, group in frame.groupby("strategy", dropna=False):
        seeds = sorted({int(value) for value in group["seed"].dropna().tolist()})
        first = group.iloc[0].to_dict()
        row: dict[str, Any] = {
            "strategy": strategy,
            "source": first.get("source"),
            "model_family": first.get("model_family"),
            "variant": (
                "multiseed_mean"
                if strategy != "mvo"
                else "multiseed_or_deterministic_mean"
            ),
            "seed_count": len(seeds),
            "seeds": ",".join(str(seed) for seed in seeds),
            "dataset_id": first.get("dataset_id"),
            "pit_split_id": first.get("pit_split_id"),
            "universe_id": first.get("universe_id"),
            "point_in_time": first.get("point_in_time"),
            "trade_end_date": first.get("trade_end_date"),
            "configured_total_steps": first.get("configured_total_steps"),
        }
        for metric in METRICS:
            values = pd.to_numeric(group.get(metric), errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            row[f"{metric}_std"] = (
                float(values.std(ddof=1))
                if len(values) > 1
                else 0.0 if len(values) == 1 else None
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    for column in SUMMARY_COLUMNS:
        if column not in result.columns:
            result[column] = None
    return result[SUMMARY_COLUMNS]


def plot_metric(
    summary: pd.DataFrame, metric: str, output_path: Path, title: str, ylabel: str
) -> bool:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in summary.columns:
        return False
    plot_df = summary[["strategy", mean_col, std_col]].copy()
    plot_df[mean_col] = pd.to_numeric(plot_df[mean_col], errors="coerce")
    plot_df[std_col] = pd.to_numeric(plot_df[std_col], errors="coerce").fillna(0.0)
    plot_df = plot_df.dropna(subset=[mean_col])
    if plot_df.empty:
        return False
    plot_df = plot_df.sort_values(mean_col, ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(plot_df))))
    ax.barh(plot_df["strategy"], plot_df[mean_col], xerr=plot_df[std_col])
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def main() -> int:
    project_root = find_project_root()
    run_id = now_run_id()
    run_directory = project_root / "outputs" / "runs" / run_id
    data_directory = run_directory / "data"
    summary_directory = run_directory / "summary"
    data_directory.mkdir(parents=True, exist_ok=True)
    summary_directory.mkdir(parents=True, exist_ok=True)

    log("Starting StockInvestmentDSS FinRL baseline multiseed summary.")
    log(f"Project root: {project_root}")
    log(f"Created run directory: {run_directory}")
    log(f"Run id: {run_id}")

    recent_run_limit = get_int_env(
        "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_RECENT_RUN_LIMIT",
        DEFAULT_RECENT_RUN_LIMIT,
    )
    seed_filter = parse_seed_list(
        get_str_env(
            "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_LIST",
            get_str_env("STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST", DEFAULT_SEEDS),
        )
    )
    agent_filter = parse_agent_list(
        get_str_env("STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS", DEFAULT_AGENTS)
    )
    include_mvo = get_bool_env("STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO", True)
    deduplicate = get_bool_env(
        "STOCK_INVESTMENT_DSS_FINRL_MULTI_SEED_DEDUPLICATE", True
    )

    context_filters = {
        "dataset_id": get_str_env("STOCK_INVESTMENT_DSS_COMPARISON_DATASET_ID"),
        "pit_split_id": get_str_env("STOCK_INVESTMENT_DSS_COMPARISON_PIT_SPLIT_ID"),
        "point_in_time": get_str_env("STOCK_INVESTMENT_DSS_COMPARISON_POINT_IN_TIME"),
        "trade_end_date": get_str_env("STOCK_INVESTMENT_DSS_COMPARISON_TRADE_END_DATE"),
    }

    log(f"Recent run limit: {recent_run_limit}")
    log(f"Seed filter: {seed_filter}")
    log(f"Agent filter: {agent_filter}")
    log(f"Include MVO: {include_mvo}")
    log(f"Context filters: {context_filters}")

    all_rows: list[dict[str, Any]] = []
    scanned = find_recent_run_dirs(project_root, recent_run_limit)
    child_run_index = build_launcher_child_run_index(project_root)
    baseline_run_dirs = [
        path for path in scanned if "finrl_baseline_suite_smoke_test" in path.name
    ]

    for run_dir in baseline_run_dirs:
        all_rows.extend(
            read_baseline_suite_run(
                run_dir, child_run_index=child_run_index, context_fill=context_filters
            )
        )

    frame = pd.DataFrame(all_rows)
    rows_before_filter = int(len(frame))
    if not frame.empty:
        frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce")
        if seed_filter is not None:
            frame = frame[frame["seed"].isin(seed_filter)]
        if agent_filter is not None:
            allowed = set(agent_filter)
            if include_mvo:
                allowed.add("mvo")
            frame = frame[frame["strategy"].str.lower().isin(allowed)]
        elif not include_mvo:
            frame = frame[frame["strategy"].str.lower() != "mvo"]
        for key, expected in context_filters.items():
            if expected:
                # Keep rows with missing legacy metadata, but reject known mismatches.
                mask = (
                    frame[key].isna()
                    | (frame[key].astype(str) == "")
                    | (frame[key].astype(str) == str(expected))
                )
                frame = frame[mask]

    rows_after_filter = int(len(frame))
    if deduplicate and not frame.empty:
        frame = deduplicate_strategy_seed(frame)

    member_path = data_directory / "finrl_baseline_multiseed_member_records.csv"
    frame.to_csv(member_path, index=False)

    aggregate = aggregate_by_strategy(frame)
    if not aggregate.empty and "agent_name" not in aggregate.columns:
        aggregate.insert(0, "agent_name", aggregate["strategy"])
    aggregate_path = (
        summary_directory / "finrl_baseline_multiseed_aggregate_by_strategy.csv"
    )
    aggregate.to_csv(aggregate_path, index=False)
    # Backward-compatible alias for commands that refer to agents rather than strategies.
    aggregate_by_agent_path = (
        summary_directory / "finrl_baseline_multiseed_aggregate_by_agent.csv"
    )
    aggregate.to_csv(aggregate_by_agent_path, index=False)

    successful_plots: list[str] = []
    plot_specs = [
        (
            "total_return_pct",
            "FinRL Baseline Multiseed: Total Return",
            "Total return (%)",
        ),
        (
            "annualized_sharpe",
            "FinRL Baseline Multiseed: Sharpe Ratio",
            "Annualized Sharpe",
        ),
        (
            "max_drawdown_pct",
            "FinRL Baseline Multiseed: Maximum Drawdown",
            "Max drawdown (%)",
        ),
        ("cvar_pct", "FinRL Baseline Multiseed: CVaR / Downside Risk", "CVaR (%)"),
        (
            "final_value",
            "FinRL Baseline Multiseed: Final Portfolio Value",
            "Final value",
        ),
    ]
    for metric, title, ylabel in plot_specs:
        path = summary_directory / f"finrl_baseline_multiseed_{metric}_mean_std.png"
        if plot_metric(aggregate, metric, path, title, ylabel):
            successful_plots.append(str(path))

    final_summary: dict[str, Any] = {}
    for _, row in aggregate.iterrows():
        strategy = str(row.get("strategy"))
        final_summary[strategy] = {
            "seed_count": int(row.get("seed_count") or 0),
            "seeds": row.get("seeds"),
            "total_return_pct_mean": row.get("total_return_pct_mean"),
            "total_return_pct_std": row.get("total_return_pct_std"),
            "annualized_sharpe_mean": row.get("annualized_sharpe_mean"),
            "annualized_sharpe_std": row.get("annualized_sharpe_std"),
            "max_drawdown_pct_mean": row.get("max_drawdown_pct_mean"),
            "max_drawdown_pct_std": row.get("max_drawdown_pct_std"),
            "cvar_pct_mean": row.get("cvar_pct_mean"),
            "cvar_pct_std": row.get("cvar_pct_std"),
        }

    summary = {
        "status": "ok",
        "project_name": PROJECT_NAME,
        "prototype_name": PROTOTYPE_NAME,
        "run_id": run_id,
        "project_root": str(project_root),
        "run_directory": str(run_directory),
        "runs_scanned": len(scanned),
        "baseline_suite_runs_found": len(baseline_run_dirs),
        "launcher_child_run_index_count": len(child_run_index),
        "rows_before_filter": rows_before_filter,
        "rows_after_filter": rows_after_filter,
        "rows_after_deduplication": int(len(frame)),
        "seed_filter": seed_filter,
        "agent_filter": agent_filter,
        "include_mvo": include_mvo,
        "context_filters": context_filters,
        "strategies": (
            sorted(aggregate["strategy"].dropna().astype(str).tolist())
            if not aggregate.empty
            else []
        ),
        "final_summary": final_summary,
        "outputs": {
            "member_records_path": str(member_path),
            "aggregate_by_strategy_path": str(aggregate_path),
            "aggregate_by_agent_path": str(aggregate_by_agent_path),
            "plots": successful_plots,
            "summary_path": str(
                summary_directory / "finrl_baseline_multiseed_summary.json"
            ),
        },
        "interpretation": (
            "This summary aggregates FinRL/SB3 baseline suite runs across seeds. "
            "MVO is included when requested; if deterministic, it should have std=0."
        ),
    }

    summary_path = summary_directory / "finrl_baseline_multiseed_summary.json"
    write_json(summary_path, summary)

    log("FinRL baseline multiseed summary completed.")
    log(f"Rows before filter: {rows_before_filter}")
    log(f"Rows after filter: {rows_after_filter}")
    log(f"Rows after deduplication: {len(frame)}")
    log(f"Strategies: {summary['strategies']}")
    log(f"Wrote member records: {member_path}")
    log(f"Wrote aggregate table: {aggregate_path}")
    log(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
