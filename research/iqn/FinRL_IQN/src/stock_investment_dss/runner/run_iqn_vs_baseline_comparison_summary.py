"""Build a thesis-oriented comparison summary for FinRL baselines and D-IQN-DSS.

This runner scans ``outputs/runs`` and creates a single comparison table across:

- FinRL/SB3 baseline suite runs: A2C, DDPG, TD3, PPO, SAC, MVO
- Single D-IQN-DSS / IQN backtest runs
- IQN learning-curve multiseed summaries
- Optional V1/generic summary CSV files, when present

The important V2 addition is that IQN multiseed learning-curve outputs are now
included explicitly. This means the comparison table can compare a long-period
FinRL baseline suite against the long-period D-IQN-DSS IQN multiseed mean/std
results, instead of only the older single IQN smoke backtests.

Expected outputs:

- data/iqn_vs_baseline_comparison_table.csv
- summary/iqn_vs_baseline_comparison_summary.csv
- summary/iqn_vs_baseline_family_summary.csv
- summary/iqn_vs_baseline_comparison_summary.md
- summary/iqn_vs_baseline_comparison_summary.json
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

RUN_KIND = "iqn_vs_baseline_comparison_summary"
PROJECT_NAME = "StockInvestmentDSS"
PROTOTYPE_NAME = "D-IQN-DSS"

DEFAULT_RECENT_RUN_LIMIT = 120

OUTPUT_COLUMNS = [
    "rank",
    "strategy",
    "source",
    "model_family",
    "variant",
    "run_id",
    "source_run_id",
    "dataset_id",
    "pit_split_id",
    "universe_id",
    "point_in_time",
    "trade_end_date",
    "score_mode",
    "risk_lambda",
    "seed",
    "seed_count",
    "configured_total_steps",
    "eval_interval",
    "start_value",
    "end_value",
    "final_value",
    "profit_loss",
    "total_return_pct",
    "cumulative_return_pct",
    "annualized_sharpe",
    "annualized_volatility_pct",
    "max_drawdown_pct",
    "cvar_pct",
    "downside_risk_pct",
    "total_transaction_cost",
    "transaction_cost",
    "total_trades",
    "trades",
    "turnover_estimate_pct",
    "portfolio_value_changed",
    "action_counts",
    "status",
    "row_type",
    "source_run_directory",
    "summary_path",
]

NUMERIC_COLUMNS = [
    "rank",
    "risk_lambda",
    "seed",
    "seed_count",
    "configured_total_steps",
    "eval_interval",
    "start_value",
    "end_value",
    "final_value",
    "profit_loss",
    "total_return_pct",
    "cumulative_return_pct",
    "annualized_sharpe",
    "annualized_volatility_pct",
    "max_drawdown_pct",
    "cvar_pct",
    "downside_risk_pct",
    "total_transaction_cost",
    "transaction_cost",
    "total_trades",
    "trades",
    "turnover_estimate_pct",
]


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_directory: Path
    data_directory: Path
    summary_directory: Path
    logs_directory: Path


def now_run_id() -> str:
    return f"{datetime.now():%Y_%m_%d_%H%M%S}_d_iqn_dss_{RUN_KIND}"


def find_project_root() -> Path:
    """Return the project root.

    The runners are executed from the repository root in the current workflow.
    This fallback makes the script a bit more robust if called from a subfolder.
    """
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "outputs").exists() or (candidate / "src").exists():
            return candidate
    return current


def create_run_directories(project_root: Path) -> RunPaths:
    run_id = now_run_id()
    run_directory = project_root / "outputs" / "runs" / run_id
    data_directory = run_directory / "data"
    summary_directory = run_directory / "summary"
    logs_directory = run_directory / "logs"

    data_directory.mkdir(parents=True, exist_ok=True)
    summary_directory.mkdir(parents=True, exist_ok=True)
    logs_directory.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_id=run_id,
        run_directory=run_directory,
        data_directory=data_directory,
        summary_directory=summary_directory,
        logs_directory=logs_directory,
    )


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} | INFO | {message}")


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
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_str_env(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def parse_int_list_env(name: str) -> list[int] | None:
    raw = get_str_env(name)
    if raw is None:
        return None
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            continue
    return values or None


def recursive_first_present(value: Any, keys: Iterable[str]) -> Any:
    """Find the first non-empty value for any key in a nested JSON-like object."""
    key_set = set(keys)
    if isinstance(value, dict):
        for key in key_set:
            if key in value and value[key] not in [None, ""]:
                return value[key]
        for child in value.values():
            found = recursive_first_present(child, key_set)
            if found not in [None, ""]:
                return found
    elif isinstance(value, list):
        for child in value:
            found = recursive_first_present(child, key_set)
            if found not in [None, ""]:
                return found
    return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def first_present(mapping: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] not in [None, ""]:
            return mapping[name]
    return None


def normalize_percent(value: Any) -> float | None:
    """Return a percentage value as stored by project metrics.

    The project generally stores return/drawdown as percent values, not decimal
    fractions. This function intentionally does not multiply by 100.
    """
    return as_float(value)


def bool_from_value(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    number = as_float(value)
    if number is not None:
        return bool(number)
    return None


def to_python_nan(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[OUTPUT_COLUMNS]


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)

    # Alias final/end values.
    end_value = first_present(normalized, ["end_value", "final_value"])
    final_value = first_present(normalized, ["final_value", "end_value"])
    normalized["end_value"] = as_float(end_value)
    normalized["final_value"] = as_float(final_value)

    start_value = first_present(normalized, ["start_value", "initial_value"])
    normalized["start_value"] = as_float(start_value)

    if normalized.get("profit_loss") in [None, ""]:
        if (
            normalized.get("final_value") is not None
            and normalized.get("start_value") is not None
        ):
            normalized["profit_loss"] = (
                normalized["final_value"] - normalized["start_value"]
            )

    total_return = first_present(
        normalized,
        ["total_return_pct", "cumulative_return_pct", "return_pct"],
    )
    normalized["total_return_pct"] = normalize_percent(total_return)
    normalized["cumulative_return_pct"] = normalize_percent(
        first_present(normalized, ["cumulative_return_pct", "total_return_pct"])
    )

    normalized["annualized_sharpe"] = as_float(
        first_present(normalized, ["annualized_sharpe", "sharpe", "sharpe_ratio"])
    )
    normalized["annualized_volatility_pct"] = normalize_percent(
        first_present(
            normalized,
            ["annualized_volatility_pct", "volatility_pct", "annualized_volatility"],
        )
    )
    normalized["max_drawdown_pct"] = normalize_percent(
        first_present(normalized, ["max_drawdown_pct", "max_drawdown"])
    )
    normalized["cvar_pct"] = normalize_percent(
        first_present(normalized, ["cvar_pct", "cvar10_pct", "cvar"])
    )
    normalized["downside_risk_pct"] = normalize_percent(
        first_present(normalized, ["downside_risk_pct", "downside_risk"])
    )

    total_cost = first_present(
        normalized,
        [
            "total_transaction_cost",
            "transaction_cost",
            "finrl_cost",
            "total_cost_delta",
        ],
    )
    normalized["total_transaction_cost"] = as_float(total_cost)
    normalized["transaction_cost"] = as_float(
        first_present(normalized, ["transaction_cost", "total_transaction_cost"])
    )

    total_trades = first_present(
        normalized,
        ["total_trades", "trades", "finrl_trades", "total_trades_delta"],
    )
    normalized["total_trades"] = as_int(total_trades)
    normalized["trades"] = as_int(first_present(normalized, ["trades", "total_trades"]))

    normalized["turnover_estimate_pct"] = normalize_percent(
        first_present(normalized, ["turnover_estimate_pct", "turnover_pct", "turnover"])
    )

    if normalized.get("portfolio_value_changed") in [None, ""]:
        if (
            normalized.get("final_value") is not None
            and normalized.get("start_value") is not None
        ):
            normalized["portfolio_value_changed"] = (
                abs(normalized["final_value"] - normalized["start_value"]) > 1e-8
            )
    else:
        normalized["portfolio_value_changed"] = bool_from_value(
            normalized["portfolio_value_changed"]
        )

    for column in NUMERIC_COLUMNS:
        if column in normalized and column not in {"portfolio_value_changed"}:
            if column in {
                "rank",
                "seed",
                "seed_count",
                "configured_total_steps",
                "eval_interval",
                "total_trades",
                "trades",
            }:
                normalized[column] = as_int(normalized[column])
            else:
                normalized[column] = as_float(normalized[column])

    for key, value in list(normalized.items()):
        normalized[key] = to_python_nan(value)

    return normalized


def recent_run_directories(project_root: Path, limit: int) -> list[Path]:
    runs_root = project_root / "outputs" / "runs"
    if not runs_root.exists():
        return []
    directories = [path for path in runs_root.iterdir() if path.is_dir()]
    directories.sort(key=lambda path: path.name, reverse=True)
    return directories[:limit]


def extract_metadata(summary: dict[str, Any], run_directory: Path) -> dict[str, Any]:
    iqn = summary.get("iqn") if isinstance(summary.get("iqn"), dict) else {}
    final_eval = (
        iqn.get("final_eval") if isinstance(iqn.get("final_eval"), dict) else {}
    )
    config = iqn.get("config") if isinstance(iqn.get("config"), dict) else {}

    point_in_time_split = (
        summary.get("point_in_time_split")
        if isinstance(summary.get("point_in_time_split"), dict)
        else {}
    )

    dataset_id = first_present(
        summary,
        ["dataset_id", "daily_dataset_id", "data_set_id"],
    ) or recursive_first_present(
        summary,
        ["dataset_id", "daily_dataset_id", "data_set_id"],
    )

    pit_split_id = (
        first_present(
            summary,
            ["pit_split_id", "point_in_time_split_id", "split_id"],
        )
        or first_present(
            point_in_time_split,
            ["split_id", "pit_split_id", "point_in_time_split_id"],
        )
        or recursive_first_present(
            summary,
            ["pit_split_id", "point_in_time_split_id", "split_id"],
        )
    )

    return {
        "run_id": summary.get("run_id") or run_directory.name,
        "dataset_id": dataset_id,
        "pit_split_id": pit_split_id,
        "universe_id": summary.get("universe_id")
        or recursive_first_present(summary, ["universe_id"]),
        "point_in_time": summary.get("point_in_time")
        or point_in_time_split.get("point_in_time")
        or recursive_first_present(summary, ["point_in_time"]),
        "trade_end_date": summary.get("trade_end_date")
        or point_in_time_split.get("trade_end_date")
        or recursive_first_present(summary, ["trade_end_date"]),
        "score_mode": iqn.get("eval_score_mode") or final_eval.get("score_mode"),
        "risk_lambda": iqn.get("risk_lambda") or final_eval.get("risk_lambda"),
        "seed": summary.get("random_seed") or config.get("seed"),
        "configured_total_steps": iqn.get("total_steps") or config.get("total_steps"),
        "eval_interval": iqn.get("eval_interval"),
        "source_run_directory": str(run_directory),
    }


def read_finrl_baseline_suite(run_directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary_path = run_directory / "summary" / "finrl_baseline_suite_smoke_summary.json"
    if not summary_path.exists():
        return rows

    summary = read_json(summary_path)
    metadata = extract_metadata(summary, run_directory)

    candidate_csvs = [
        run_directory
        / "data"
        / "finrl_baseline_suite"
        / "finrl_baseline_suite_comparison.csv",
        run_directory / "summary" / "finrl_baseline_suite_comparison_snapshot.csv",
    ]
    comparison_path = next((path for path in candidate_csvs if path.exists()), None)
    if comparison_path is None:
        return rows

    try:
        frame = pd.read_csv(comparison_path)
    except Exception:
        return rows

    for index, record in frame.iterrows():
        item = record.to_dict()
        strategy = str(
            first_present(item, ["agent", "agent_name", "strategy"]) or "unknown"
        )
        row = {
            **metadata,
            "rank": item.get("rank") or index + 1,
            "strategy": strategy,
            "source": (
                "FinRL / SB3 baseline suite"
                if strategy.lower() != "mvo"
                else "MVO baseline"
            ),
            "model_family": (
                "parametric_rl_expected_return"
                if strategy.lower() != "mvo"
                else "classical_portfolio_optimization"
            ),
            "variant": "long_or_smoke_baseline_suite",
            "source_run_id": metadata["run_id"],
            "start_value": first_present(item, ["initial_value", "start_value"]),
            "end_value": first_present(item, ["final_value", "end_value"]),
            "final_value": first_present(item, ["final_value", "end_value"]),
            "profit_loss": item.get("profit_loss"),
            "total_return_pct": item.get("total_return_pct"),
            "cumulative_return_pct": item.get("cumulative_return_pct")
            or item.get("total_return_pct"),
            "annualized_sharpe": first_present(item, ["annualized_sharpe", "sharpe"]),
            "annualized_volatility_pct": item.get("annualized_volatility_pct"),
            "max_drawdown_pct": item.get("max_drawdown_pct"),
            "cvar_pct": item.get("cvar_pct"),
            "downside_risk_pct": item.get("downside_risk_pct"),
            "total_transaction_cost": first_present(
                item, ["total_transaction_cost", "finrl_cost"]
            ),
            "total_trades": first_present(item, ["total_trades", "finrl_trades"]),
            "turnover_estimate_pct": item.get("turnover_estimate_pct"),
            "portfolio_value_changed": item.get("portfolio_value_changed"),
            "status": item.get("status") or item.get("trading_status"),
            "row_type": "baseline_agent",
            "summary_path": str(summary_path),
        }
        rows.append(normalize_row(row))

    return rows


def read_single_finrl_baseline(run_directory: Path) -> list[dict[str, Any]]:
    """Read older/single baseline summaries, if present."""
    rows: list[dict[str, Any]] = []
    summary_candidates = list(
        (run_directory / "summary").glob("*baseline*summary.json")
    )
    for summary_path in summary_candidates:
        if summary_path.name == "finrl_baseline_suite_smoke_summary.json":
            continue
        summary = read_json(summary_path)
        if not summary:
            continue
        agent_name = summary.get("agent_name") or summary.get("strategy")
        if not agent_name:
            continue
        metadata = extract_metadata(summary, run_directory)
        row = {
            **metadata,
            "strategy": str(agent_name),
            "source": "FinRL / SB3 baseline",
            "model_family": "parametric_rl_expected_return",
            "variant": "single_baseline_run",
            "source_run_id": metadata["run_id"],
            "start_value": summary.get("initial_value") or summary.get("start_value"),
            "end_value": summary.get("final_value") or summary.get("end_value"),
            "final_value": summary.get("final_value") or summary.get("end_value"),
            "profit_loss": summary.get("profit_loss"),
            "total_return_pct": summary.get("total_return_pct"),
            "cumulative_return_pct": summary.get("cumulative_return_pct")
            or summary.get("total_return_pct"),
            "annualized_sharpe": summary.get("annualized_sharpe"),
            "annualized_volatility_pct": summary.get("annualized_volatility_pct"),
            "max_drawdown_pct": summary.get("max_drawdown_pct"),
            "cvar_pct": summary.get("cvar_pct"),
            "downside_risk_pct": summary.get("downside_risk_pct"),
            "total_transaction_cost": summary.get("total_transaction_cost"),
            "total_trades": summary.get("total_trades"),
            "turnover_estimate_pct": summary.get("turnover_estimate_pct"),
            "portfolio_value_changed": summary.get("portfolio_value_changed"),
            "status": summary.get("status") or summary.get("trading_status"),
            "row_type": "baseline_agent",
            "summary_path": str(summary_path),
        }
        rows.append(normalize_row(row))
    return rows


def read_iqn_backtest(run_directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary_path = run_directory / "summary" / "iqn_backtest_smoke_summary.json"
    if not summary_path.exists():
        return rows

    summary = read_json(summary_path)
    metadata = extract_metadata(summary, run_directory)
    outputs = summary.get("outputs") if isinstance(summary.get("outputs"), dict) else {}
    policy_path_raw = outputs.get("policy_summary_path")
    policy_summary = read_json(Path(policy_path_raw)) if policy_path_raw else {}

    metrics = dict(summary)
    if isinstance(summary.get("metrics"), dict):
        metrics.update(summary["metrics"])

    action_counts = None
    if isinstance(policy_summary.get("iqn_selected_action_counts"), list):
        action_counts = policy_summary.get("iqn_selected_action_counts")
    elif isinstance(policy_summary.get("action_counts"), list):
        action_counts = policy_summary.get("action_counts")
    elif isinstance(summary.get("action_counts"), dict):
        action_counts = summary.get("action_counts")

    score_mode = (
        metadata.get("score_mode")
        or summary.get("score_mode")
        or policy_summary.get("score_mode")
    )
    strategy = f"D-IQN-DSS ({score_mode or 'unknown'})"

    row = {
        **metadata,
        "strategy": strategy,
        "source": "D-IQN-DSS / distributional RL",
        "model_family": "distributional_rl_iqn",
        "variant": "single_backtest",
        "source_run_id": metadata["run_id"],
        "score_mode": score_mode,
        "risk_lambda": metadata.get("risk_lambda")
        or summary.get("risk_lambda")
        or policy_summary.get("risk_lambda"),
        "start_value": metrics.get("initial_value") or metrics.get("start_value"),
        "end_value": metrics.get("final_value") or metrics.get("end_value"),
        "final_value": metrics.get("final_value") or metrics.get("end_value"),
        "profit_loss": metrics.get("profit_loss"),
        "total_return_pct": metrics.get("total_return_pct"),
        "cumulative_return_pct": metrics.get("cumulative_return_pct")
        or metrics.get("total_return_pct"),
        "annualized_sharpe": metrics.get("annualized_sharpe"),
        "annualized_volatility_pct": metrics.get("annualized_volatility_pct"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "cvar_pct": metrics.get("cvar_pct"),
        "downside_risk_pct": metrics.get("downside_risk_pct"),
        "total_transaction_cost": metrics.get("total_transaction_cost"),
        "total_trades": metrics.get("total_trades"),
        "turnover_estimate_pct": metrics.get("turnover_estimate_pct"),
        "portfolio_value_changed": metrics.get("portfolio_value_changed"),
        "action_counts": (
            json.dumps(action_counts, ensure_ascii=False)
            if action_counts is not None
            else None
        ),
        "status": summary.get("status") or metrics.get("status"),
        "row_type": "iqn_single_backtest",
        "summary_path": str(summary_path),
    }
    rows.append(normalize_row(row))
    return rows


def read_iqn_learning_curve_multiseed(run_directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary_path = (
        run_directory / "summary" / "iqn_learning_curve_multiseed_summary.json"
    )
    if not summary_path.exists():
        return rows

    summary = read_json(summary_path)
    if not summary:
        return rows

    metadata = extract_metadata(summary, run_directory)
    outputs = summary.get("outputs") if isinstance(summary.get("outputs"), dict) else {}
    final_records_path = (
        Path(outputs.get("final_records_path"))
        if outputs.get("final_records_path")
        else None
    )
    source_run_count = summary.get("source_run_count")
    unique_seed_count = summary.get("unique_seed_count")
    seeds = summary.get("seeds") if isinstance(summary.get("seeds"), list) else []

    final_summary = (
        summary.get("final_summary")
        if isinstance(summary.get("final_summary"), dict)
        else {}
    )

    # Load final_records early as well, because older multiseed summary JSON files
    # may not store dataset/PIT metadata at the aggregate level. The per-seed
    # final_records rows usually do contain that context. We use it to keep the
    # multiseed mean row in context-filtered thesis comparisons.
    final_records = pd.DataFrame()
    if final_records_path and final_records_path.exists():
        try:
            final_records = pd.read_csv(final_records_path)
        except Exception:
            final_records = pd.DataFrame()

    def first_non_empty_column(frame: pd.DataFrame, names: Iterable[str]) -> Any:
        if frame.empty:
            return None
        for name in names:
            if name not in frame.columns:
                continue
            series = frame[name].dropna()
            for value in series:
                if value not in [None, ""] and str(value).strip().lower() != "nan":
                    return value
        return None

    inferred_dataset_id = first_non_empty_column(
        final_records, ["dataset_id", "daily_dataset_id"]
    )
    inferred_pit_split_id = first_non_empty_column(
        final_records, ["pit_split_id", "split_id", "point_in_time_split_id"]
    )
    inferred_universe_id = first_non_empty_column(final_records, ["universe_id"])
    inferred_point_in_time = first_non_empty_column(final_records, ["point_in_time"])
    inferred_trade_end_date = first_non_empty_column(final_records, ["trade_end_date"])
    inferred_score_mode = first_non_empty_column(final_records, ["score_mode"])
    inferred_risk_lambda = first_non_empty_column(final_records, ["risk_lambda"])
    inferred_total_steps = first_non_empty_column(
        final_records, ["configured_total_steps", "total_steps"]
    )
    inferred_eval_interval = first_non_empty_column(final_records, ["eval_interval"])

    def metric_mean(metric: str) -> Any:
        value = final_summary.get(metric)
        return value.get("mean") if isinstance(value, dict) else None

    def metric_std(metric: str) -> Any:
        value = final_summary.get(metric)
        return value.get("std") if isinstance(value, dict) else None

    def metric_count(metric: str) -> Any:
        value = final_summary.get(metric)
        return value.get("count") if isinstance(value, dict) else None

    # Add one aggregate row for the multiseed mean. This is the main row used
    # when comparing D-IQN-DSS to FinRL baselines on the same long split.
    aggregate_row = {
        "run_id": metadata.get("run_id") or summary.get("run_id") or run_directory.name,
        "source_run_id": metadata.get("run_id")
        or summary.get("run_id")
        or run_directory.name,
        "dataset_id": metadata.get("dataset_id")
        or summary.get("dataset_id")
        or inferred_dataset_id,
        "pit_split_id": metadata.get("pit_split_id") or inferred_pit_split_id,
        "universe_id": metadata.get("universe_id")
        or summary.get("universe_id")
        or inferred_universe_id,
        "point_in_time": metadata.get("point_in_time")
        or summary.get("point_in_time")
        or inferred_point_in_time,
        "trade_end_date": metadata.get("trade_end_date")
        or summary.get("trade_end_date")
        or inferred_trade_end_date,
        "strategy": "D-IQN-DSS IQN risk-aware / multiseed mean",
        "source": "D-IQN-DSS / distributional RL / multiseed",
        "model_family": "distributional_rl_iqn",
        "variant": "multiseed_mean",
        "score_mode": summary.get("eval_score_mode")
        or inferred_score_mode
        or "q50_minus_cvar_penalty",
        "risk_lambda": summary.get("risk_lambda") or inferred_risk_lambda,
        "seed_count": unique_seed_count,
        "seed": None,
        "configured_total_steps": inferred_total_steps,
        "eval_interval": inferred_eval_interval,
        "start_value": 1000000.0,
        "end_value": metric_mean("final_value"),
        "final_value": metric_mean("final_value"),
        "profit_loss": None,
        "total_return_pct": metric_mean("total_return_pct"),
        "cumulative_return_pct": metric_mean("total_return_pct"),
        "annualized_sharpe": metric_mean("annualized_sharpe"),
        "annualized_volatility_pct": metric_mean("annualized_volatility_pct"),
        "max_drawdown_pct": metric_mean("max_drawdown_pct"),
        "cvar_pct": metric_mean("cvar_pct"),
        "turnover_estimate_pct": metric_mean("turnover_estimate_pct"),
        "portfolio_value_changed": None,
        "action_counts": None,
        "status": summary.get("status"),
        "row_type": "iqn_multiseed_mean",
        "source_run_directory": str(run_directory),
        "summary_path": str(summary_path),
        # Extra values are useful for JSON/markdown interpretation, even if not
        # part of the standardized table columns.
        "source_run_count": source_run_count,
        "seeds": json.dumps(seeds, ensure_ascii=False),
        "total_return_pct_std": metric_std("total_return_pct"),
        "annualized_sharpe_count": metric_count("annualized_sharpe"),
    }
    rows.append(normalize_row(aggregate_row))

    # Add one row per seed where available. These rows let us inspect seed-level
    # variability without hiding it inside only the mean/std aggregate.
    if not final_records.empty:
        for _, record in final_records.iterrows():
            item = record.to_dict()
            seed = as_int(item.get("seed"))
            row = {
                "run_id": metadata.get("run_id")
                or summary.get("run_id")
                or run_directory.name,
                "source_run_id": item.get("source_run_id")
                or metadata.get("run_id")
                or summary.get("run_id")
                or run_directory.name,
                "dataset_id": item.get("dataset_id")
                or metadata.get("dataset_id")
                or summary.get("dataset_id"),
                "pit_split_id": item.get("pit_split_id")
                or metadata.get("pit_split_id"),
                "universe_id": item.get("universe_id")
                or metadata.get("universe_id")
                or summary.get("universe_id"),
                "point_in_time": item.get("point_in_time")
                or metadata.get("point_in_time")
                or summary.get("point_in_time"),
                "trade_end_date": item.get("trade_end_date")
                or metadata.get("trade_end_date")
                or summary.get("trade_end_date"),
                "strategy": f"D-IQN-DSS IQN risk-aware / seed {seed}",
                "source": "D-IQN-DSS / distributional RL / multiseed",
                "model_family": "distributional_rl_iqn",
                "variant": "multiseed_member",
                "score_mode": item.get("score_mode"),
                "risk_lambda": item.get("risk_lambda"),
                "seed": seed,
                "seed_count": unique_seed_count,
                "configured_total_steps": item.get("configured_total_steps"),
                "eval_interval": item.get("eval_interval"),
                "start_value": item.get("initial_value"),
                "end_value": item.get("final_value"),
                "final_value": item.get("final_value"),
                "profit_loss": item.get("profit_loss"),
                "total_return_pct": item.get("total_return_pct"),
                "cumulative_return_pct": item.get("total_return_pct"),
                "annualized_sharpe": item.get("annualized_sharpe"),
                "annualized_volatility_pct": item.get("annualized_volatility_pct"),
                "max_drawdown_pct": item.get("max_drawdown_pct"),
                "cvar_pct": item.get("cvar_pct"),
                "total_transaction_cost": item.get("total_transaction_cost"),
                "total_trades": item.get("total_trades"),
                "turnover_estimate_pct": item.get("turnover_estimate_pct"),
                "portfolio_value_changed": None,
                "action_counts": item.get("action_counts"),
                "status": item.get("status"),
                "row_type": "iqn_multiseed_member",
                "source_run_directory": item.get("source_run_directory")
                or str(run_directory),
                "summary_path": str(summary_path),
            }
            rows.append(normalize_row(row))

    return rows


def read_generic_v1_summary(run_directory: Path) -> list[dict[str, Any]]:
    """Read optional V1-style summary_report.csv files if present."""
    rows: list[dict[str, Any]] = []
    for csv_path in [
        run_directory / "summary" / "summary_report.csv",
        run_directory / "summary" / "comparison_summary.csv",
    ]:
        if not csv_path.exists():
            continue
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        for _, record in frame.iterrows():
            item = record.to_dict()
            strategy = str(
                first_present(item, ["strategy", "agent", "model", "name"]) or "generic"
            )
            source = str(
                first_present(item, ["source", "baseline_type"]) or "Generic/V1 summary"
            )
            row = {
                "run_id": run_directory.name,
                "source_run_id": run_directory.name,
                "strategy": strategy,
                "source": source,
                "model_family": "generic_or_algorithmic_baseline",
                "variant": "v1_generic_summary",
                "start_value": item.get("initial_value") or item.get("start_value"),
                "end_value": item.get("final_value") or item.get("end_value"),
                "final_value": item.get("final_value") or item.get("end_value"),
                "profit_loss": item.get("profit_loss"),
                "total_return_pct": item.get("total_return_pct"),
                "cumulative_return_pct": item.get("cumulative_return_pct")
                or item.get("total_return_pct"),
                "annualized_sharpe": item.get("annualized_sharpe"),
                "annualized_volatility_pct": item.get("annualized_volatility_pct"),
                "max_drawdown_pct": item.get("max_drawdown_pct"),
                "cvar_pct": item.get("cvar_pct"),
                "total_transaction_cost": item.get("total_transaction_cost"),
                "total_trades": item.get("total_trades"),
                "status": item.get("status"),
                "row_type": "generic_v1_summary",
                "source_run_directory": str(run_directory),
                "summary_path": str(csv_path),
            }
            rows.append(normalize_row(row))
    return rows


def collect_rows(project_root: Path, recent_run_limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_directory in recent_run_directories(project_root, recent_run_limit):
        # Skip this runner's own output directories if rerun repeatedly.
        if RUN_KIND in run_directory.name:
            continue
        rows.extend(read_finrl_baseline_suite(run_directory))
        rows.extend(read_single_finrl_baseline(run_directory))
        rows.extend(read_iqn_backtest(run_directory))
        rows.extend(read_iqn_learning_curve_multiseed(run_directory))
        rows.extend(read_generic_v1_summary(run_directory))
    return rows


def value_equals(value: Any, expected: str | None) -> bool:
    if expected is None:
        return True
    if value is None:
        return False
    return str(value).strip() == str(expected).strip()


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def value_matches(value: Any, expected: str | None) -> bool | None:
    """Return True/False for known values, None for unknown/missing values.

    Some valid rows store date context but not dataset_id/pit_split_id, and
    some baseline rows store dataset context but not full split context. For a
    fair thesis comparison, reject known contradictions, but do not drop a row
    only because an older runner omitted one metadata field.
    """
    if expected is None:
        return True
    if is_missing(value):
        return None
    return str(value) == str(expected)


def filter_rows_by_context(
    rows: list[dict[str, Any]],
    dataset_id_filter: str | None,
    pit_split_id_filter: str | None,
    point_in_time_filter: str | None,
    trade_end_date_filter: str | None,
    seed_filter: list[int] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not any(
        [
            dataset_id_filter,
            pit_split_id_filter,
            point_in_time_filter,
            trade_end_date_filter,
            seed_filter,
        ]
    ):
        return rows, {
            "enabled": False,
            "rows_before_filter": len(rows),
            "rows_after_filter": len(rows),
            "rows_removed_by_filter": 0,
        }

    filtered: list[dict[str, Any]] = []
    removed_reasons = {
        "known_context_mismatch": 0,
        "insufficient_context": 0,
        "seed": 0,
    }

    requested_context_count = sum(
        value is not None
        for value in [
            dataset_id_filter,
            pit_split_id_filter,
            point_in_time_filter,
            trade_end_date_filter,
        ]
    )

    for row in rows:
        dataset_id = row.get("dataset_id")
        pit_split_id = row.get("pit_split_id")
        point_in_time = row.get("point_in_time")
        trade_end_date = row.get("trade_end_date")
        seed = as_int(row.get("seed"))

        matches = {
            "dataset_id": value_matches(dataset_id, dataset_id_filter),
            "pit_split_id": value_matches(pit_split_id, pit_split_id_filter),
            "point_in_time": value_matches(point_in_time, point_in_time_filter),
            "trade_end_date": value_matches(trade_end_date, trade_end_date_filter),
        }

        # Reject known contradictions. Example: an old short smoke run with
        # point_in_time=2024-01-16 should not pass a 2023-01-01 long-split filter.
        if any(value is False for value in matches.values()):
            removed_reasons["known_context_mismatch"] += 1
            continue

        known_positive_matches = sum(value is True for value in matches.values())

        # Avoid keeping totally contextless legacy rows, while still allowing
        # valid rows where only some metadata fields were written by older runners.
        if requested_context_count > 0 and known_positive_matches == 0:
            removed_reasons["insufficient_context"] += 1
            continue

        # Seed filtering applies only to seed-level IQN rows. Aggregate IQN rows,
        # FinRL baselines and MVO rows have seed=None and should remain in the
        # same comparison table.
        seed_ok = seed_filter is None or seed is None or seed in seed_filter
        if not seed_ok:
            removed_reasons["seed"] += 1
            continue

        filtered.append(row)

    return filtered, {
        "enabled": True,
        "dataset_id_filter": dataset_id_filter,
        "pit_split_id_filter": pit_split_id_filter,
        "point_in_time_filter": point_in_time_filter,
        "trade_end_date_filter": trade_end_date_filter,
        "seed_filter": seed_filter,
        "rows_before_filter": len(rows),
        "rows_after_filter": len(filtered),
        "rows_removed_by_filter": len(rows) - len(filtered),
        "removed_reasons": removed_reasons,
        "filter_policy": (
            "Known metadata mismatches are rejected. Missing metadata is tolerated "
            "when at least one requested context field matches, so valid legacy "
            "FinRL/IQN rows are not dropped solely because older summaries did not "
            "store every context field."
        ),
    }


def filter_latest_iqn_multiseed_summary(
    rows: list[dict[str, Any]],
    enabled: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep only the newest IQN multiseed summary run.

    The project may contain several multiseed-summary runs for the same dataset,
    PIT split and seed list because we often rerun the aggregator after small
    code fixes. For the fair thesis comparison we want one multiseed block:
    the latest relevant aggregate row plus its per-seed member rows.

    This filter is intentionally applied *after* context filtering, so the
    newest run is selected among rows that already match the requested
    dataset/PIT/seed context.
    """
    multiseed_source = "D-IQN-DSS / distributional RL / multiseed"
    multiseed_rows = [row for row in rows if row.get("source") == multiseed_source]

    if not enabled or not multiseed_rows:
        return rows, {
            "enabled": bool(enabled),
            "rows_before_filter": len(rows),
            "rows_after_filter": len(rows),
            "multiseed_rows_before_filter": len(multiseed_rows),
            "multiseed_rows_after_filter": len(multiseed_rows),
            "selected_multiseed_run_id": None,
            "removed_multiseed_rows": 0,
        }

    run_ids = sorted(
        {str(row.get("run_id")) for row in multiseed_rows if row.get("run_id")},
        reverse=True,
    )
    selected_run_id = run_ids[0] if run_ids else None

    if selected_run_id is None:
        return rows, {
            "enabled": bool(enabled),
            "rows_before_filter": len(rows),
            "rows_after_filter": len(rows),
            "multiseed_rows_before_filter": len(multiseed_rows),
            "multiseed_rows_after_filter": len(multiseed_rows),
            "selected_multiseed_run_id": None,
            "removed_multiseed_rows": 0,
            "warning": "No run_id found on IQN multiseed rows; no latest-only filtering applied.",
        }

    filtered = [
        row
        for row in rows
        if row.get("source") != multiseed_source
        or str(row.get("run_id")) == selected_run_id
    ]
    selected_count = sum(
        1
        for row in filtered
        if row.get("source") == multiseed_source
        and str(row.get("run_id")) == selected_run_id
    )

    return filtered, {
        "enabled": bool(enabled),
        "rows_before_filter": len(rows),
        "rows_after_filter": len(filtered),
        "multiseed_rows_before_filter": len(multiseed_rows),
        "multiseed_rows_after_filter": selected_count,
        "available_multiseed_run_ids": run_ids,
        "selected_multiseed_run_id": selected_run_id,
        "removed_multiseed_rows": len(rows) - len(filtered),
        "filter_policy": (
            "After dataset/PIT/seed context filtering, only the latest IQN "
            "multiseed summary run is kept. This prevents multiple reruns of "
            "the same multiseed experiment from being counted twice in the "
            "fair comparison table."
        ),
    }


def rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return ensure_columns(frame)

    # Sort by return first, then Sharpe, then drawdown. For drawdown, higher is
    # better because values are usually negative and 0 is no drawdown.
    sort_columns = ["total_return_pct", "annualized_sharpe", "max_drawdown_pct"]
    for column in sort_columns:
        if column not in frame.columns:
            frame[column] = None

    sorted_frame = frame.sort_values(
        by=sort_columns,
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    sorted_frame["rank"] = range(1, len(sorted_frame) + 1)
    return ensure_columns(sorted_frame)


def family_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    numeric_metrics = [
        "total_return_pct",
        "annualized_sharpe",
        "annualized_volatility_pct",
        "max_drawdown_pct",
        "cvar_pct",
        "final_value",
    ]

    available_metrics = [
        column for column in numeric_metrics if column in frame.columns
    ]
    if not available_metrics:
        return pd.DataFrame()

    grouped = (
        frame.groupby(["model_family", "source"], dropna=False)[available_metrics]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    grouped.columns = [
        (
            "_".join([str(part) for part in column if str(part) != ""])
            if isinstance(column, tuple)
            else str(column)
        )
        for column in grouped.columns
    ]
    return grouped


def best_row(
    frame: pd.DataFrame, metric: str, ascending: bool = False
) -> dict[str, Any] | None:
    if frame.empty or metric not in frame.columns:
        return None
    subset = frame.dropna(subset=[metric]).copy()
    if subset.empty:
        return None
    subset = subset.sort_values(metric, ascending=ascending)
    row = subset.iloc[0].to_dict()
    keys = [
        "strategy",
        "source",
        "model_family",
        "variant",
        "run_id",
        "source_run_id",
        "dataset_id",
        "pit_split_id",
        "point_in_time",
        "trade_end_date",
        "total_return_pct",
        "annualized_sharpe",
        "max_drawdown_pct",
        "cvar_pct",
        "seed_count",
    ]
    return {key: to_python_nan(row.get(key)) for key in keys if key in row}



FAIR_COMPACT_COMPARISON_FILENAME = "iqn_vs_baseline_fair_compact_comparison_summary.csv"
FAIR_SEED_DIAGNOSTIC_FILENAME = "iqn_vs_baseline_fair_seed_diagnostic_comparison_summary.csv"

FAIR_COMPACT_STRATEGY_ORDER = [
    "a2c",
    "ddpg",
    "td3",
    "ppo",
    "sac",
    "mvo",
    "D-IQN-DSS IQN risk-aware / multiseed mean",
    "D-IQN-DSS IQN risk-aware / seed 1",
    "D-IQN-DSS IQN risk-aware / seed 2",
    "D-IQN-DSS IQN risk-aware / seed 3",
    "D-IQN-DSS IQN risk-aware / seed 4",
    "D-IQN-DSS IQN risk-aware / seed 5",
]

FAIR_COMPACT_NUMERIC_METRICS = [
    "start_value",
    "end_value",
    "final_value",
    "profit_loss",
    "total_return_pct",
    "cumulative_return_pct",
    "annualized_sharpe",
    "annualized_volatility_pct",
    "max_drawdown_pct",
    "cvar_pct",
    "downside_risk_pct",
    "total_transaction_cost",
    "transaction_cost",
    "total_trades",
    "trades",
    "turnover_estimate_pct",
]


def _fair_sort_key(strategy: Any) -> tuple[int, str]:
    label = str(strategy or "").strip()
    if label in FAIR_COMPACT_STRATEGY_ORDER:
        return FAIR_COMPACT_STRATEGY_ORDER.index(label), label

    lower = label.lower()
    for index, expected in enumerate(FAIR_COMPACT_STRATEGY_ORDER):
        if lower == expected.lower():
            return index, label

    return len(FAIR_COMPACT_STRATEGY_ORDER), label


def _first_non_empty_value(series: pd.Series) -> Any:
    for value in series:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return value
    return None


def _safe_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def _unique_non_empty_values(series: pd.Series) -> list[Any]:
    values: list[Any] = []
    seen: set[str] = set()
    for value in series:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        if text not in seen:
            seen.add(text)
            values.append(value)
    return values


def _mean_row_from_group(
    group: pd.DataFrame,
    *,
    strategy: str,
    source: str,
    model_family: str,
    variant: str,
    row_type: str,
) -> dict[str, Any]:
    """Build one fair compact mean row from seed/member rows.

    This is used for FinRL/SB3 agent means and the MVO row. It deliberately
    preserves the shared dataset/PIT context from the first non-empty values and
    averages only numeric performance metrics.
    """
    base: dict[str, Any] = {}
    for column in OUTPUT_COLUMNS:
        if column in group.columns:
            base[column] = _first_non_empty_value(group[column])
        else:
            base[column] = None

    base.update(
        {
            "strategy": strategy,
            "source": source,
            "model_family": model_family,
            "variant": variant,
            "seed": None,
            "row_type": row_type,
        }
    )

    if "seed" in group.columns:
        seeds = _unique_non_empty_values(group["seed"])
        base["seed_count"] = len(seeds) if seeds else len(group)
    else:
        base["seed_count"] = len(group)

    for metric in FAIR_COMPACT_NUMERIC_METRICS:
        values = _safe_numeric_series(group, metric)
        if values.empty:
            continue
        base[metric] = float(values.mean())
        base[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        base[f"{metric}_min"] = float(values.min())
        base[f"{metric}_max"] = float(values.max())

    if base.get("final_value") is None and base.get("end_value") is not None:
        base["final_value"] = base.get("end_value")
    if base.get("end_value") is None and base.get("final_value") is not None:
        base["end_value"] = base.get("final_value")
    if base.get("cumulative_return_pct") is None:
        base["cumulative_return_pct"] = base.get("total_return_pct")

    return normalize_row(base)


def build_fair_compact_comparison_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the thesis-facing compact comparison table.

    The full comparison table intentionally preserves all seed/member rows for
    auditability. For thesis plots and tables we also need one deterministic
    compact table:
    - one multiseed mean row per FinRL/SB3 agent,
    - one MVO mean/deterministic row,
    - the IQN multiseed mean row,
    - the IQN seed-level rows.
    """
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    table = frame.copy()
    rows: list[dict[str, Any]] = []

    source_col = table["source"].astype(str) if "source" in table.columns else ""
    variant_col = table["variant"].astype(str) if "variant" in table.columns else ""
    row_type_col = table["row_type"].astype(str) if "row_type" in table.columns else ""

    finrl = table[source_col == "FinRL / SB3 baseline suite"].copy()
    if not finrl.empty and "strategy" in finrl.columns:
        for strategy, group in finrl.groupby("strategy", dropna=False, sort=False):
            strategy_label = str(strategy).strip()
            rows.append(
                _mean_row_from_group(
                    group,
                    strategy=strategy_label,
                    source="FinRL / SB3 baseline suite",
                    model_family="parametric_rl_expected_return",
                    variant="baseline_multiseed_mean",
                    row_type="fair_compact_baseline_mean",
                )
            )

    mvo = table[source_col == "MVO baseline"].copy()
    if not mvo.empty:
        strategy_label = "mvo"
        if "strategy" in mvo.columns:
            first_strategy = _first_non_empty_value(mvo["strategy"])
            strategy_label = str(first_strategy or "mvo").strip()
        rows.append(
            _mean_row_from_group(
                mvo,
                strategy=strategy_label,
                source="MVO baseline",
                model_family="classical_portfolio_optimization",
                variant="mvo_mean",
                row_type="fair_compact_mvo_mean",
            )
        )

    iqn_mean_mask = (
        source_col.eq("D-IQN-DSS / distributional RL / multiseed")
        & (
            variant_col.eq("multiseed_mean")
            | row_type_col.eq("iqn_multiseed_mean")
            | table.get("strategy", pd.Series(index=table.index, dtype=str))
            .astype(str)
            .eq("D-IQN-DSS IQN risk-aware / multiseed mean")
        )
    )
    iqn_mean = table[iqn_mean_mask].copy()
    if not iqn_mean.empty:
        iqn_mean = iqn_mean.sort_values("run_id", ascending=False, kind="stable")
        rows.append(normalize_row(iqn_mean.iloc[0].to_dict()))

    iqn_seed_mask = (
        source_col.eq("D-IQN-DSS / distributional RL / multiseed")
        & variant_col.eq("multiseed_member")
    )
    iqn_seeds = table[iqn_seed_mask].copy()
    if not iqn_seeds.empty:
        if "seed" in iqn_seeds.columns:
            iqn_seeds["seed_numeric_sort"] = pd.to_numeric(
                iqn_seeds["seed"], errors="coerce"
            )
            iqn_seeds = iqn_seeds.sort_values(
                ["seed_numeric_sort", "strategy"], kind="stable"
            ).drop(columns=["seed_numeric_sort"], errors="ignore")
        for _, row in iqn_seeds.iterrows():
            rows.append(normalize_row(row.to_dict()))

    compact = pd.DataFrame(rows)
    if compact.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    compact["fair_sort_key"] = compact["strategy"].map(_fair_sort_key)
    compact = compact.sort_values("fair_sort_key", kind="stable").drop(
        columns=["fair_sort_key"], errors="ignore"
    )
    compact = compact.reset_index(drop=True)
    compact["rank"] = range(1, len(compact) + 1)

    return compact


def build_fair_seed_diagnostic_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a seed/member diagnostics table without duplicate aggregate rows."""
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    table = frame.copy()
    variant_col = table["variant"].astype(str) if "variant" in table.columns else ""
    row_type_col = table["row_type"].astype(str) if "row_type" in table.columns else ""

    diagnostic = table[
        variant_col.isin(
            [
                "long_or_smoke_baseline_suite",
                "multiseed_member",
                "mvo_mean",
                "baseline_multiseed_mean",
            ]
        )
        | row_type_col.isin(
            [
                "baseline_agent",
                "iqn_multiseed_member",
                "fair_compact_baseline_mean",
                "fair_compact_mvo_mean",
            ]
        )
    ].copy()

    if diagnostic.empty:
        diagnostic = table.copy()

    return rank_frame(diagnostic)


def write_fair_comparison_artifacts(
    *,
    frame: pd.DataFrame,
    run_paths: RunPaths,
) -> dict[str, Any]:
    compact_frame = build_fair_compact_comparison_table(frame)
    seed_diagnostic_frame = build_fair_seed_diagnostic_table(frame)

    compact_path = run_paths.summary_directory / FAIR_COMPACT_COMPARISON_FILENAME
    compact_data_path = run_paths.data_directory / FAIR_COMPACT_COMPARISON_FILENAME
    seed_diagnostic_path = run_paths.summary_directory / FAIR_SEED_DIAGNOSTIC_FILENAME
    seed_diagnostic_data_path = run_paths.data_directory / FAIR_SEED_DIAGNOSTIC_FILENAME

    compact_frame.to_csv(compact_path, index=False)
    compact_frame.to_csv(compact_data_path, index=False)
    seed_diagnostic_frame.to_csv(seed_diagnostic_path, index=False)
    seed_diagnostic_frame.to_csv(seed_diagnostic_data_path, index=False)

    compact_source_counts = (
        compact_frame["source"].value_counts(dropna=False).to_dict()
        if not compact_frame.empty and "source" in compact_frame.columns
        else {}
    )
    compact_variant_counts = (
        compact_frame["variant"].value_counts(dropna=False).to_dict()
        if not compact_frame.empty and "variant" in compact_frame.columns
        else {}
    )

    return {
        "compact_row_count": int(len(compact_frame)),
        "seed_diagnostic_row_count": int(len(seed_diagnostic_frame)),
        "compact_source_counts": compact_source_counts,
        "compact_variant_counts": compact_variant_counts,
        "compact_summary_path": str(compact_path),
        "compact_data_path": str(compact_data_path),
        "seed_diagnostic_summary_path": str(seed_diagnostic_path),
        "seed_diagnostic_data_path": str(seed_diagnostic_data_path),
    }



def markdown_report(frame: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# IQN vs Baseline Comparison Summary")
    lines.append("")
    lines.append(f"Run id: `{summary['run_id']}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This report compares FinRL/SB3 baselines, MVO, single IQN backtests, "
        "and IQN multiseed learning-curve summaries when available."
    )
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Rows found: {summary['row_count']}")
    lines.append(f"- Runs scanned: {summary['runs_scanned']}")
    if summary.get("context_filter", {}).get("enabled"):
        context_filter = summary["context_filter"]
        lines.append(f"- Context filter: `{context_filter}`")
    lines.append("")
    lines.append("### Source counts")
    lines.append("")
    for source, count in summary.get("source_counts", {}).items():
        lines.append(f"- {source}: {count}")
    lines.append("")
    fair_outputs = summary.get("fair_outputs", {})
    if fair_outputs:
        lines.append("### Fair compact comparison")
        lines.append("")
        lines.append(f"- Compact rows: {fair_outputs.get('compact_row_count')}")
        lines.append(f"- Seed diagnostic rows: {fair_outputs.get('seed_diagnostic_row_count')}")
        lines.append(f"- Compact CSV: `{fair_outputs.get('compact_summary_path')}`")
        lines.append(f"- Seed diagnostic CSV: `{fair_outputs.get('seed_diagnostic_summary_path')}`")
        lines.append("")

    lines.append("### Model family counts")
    lines.append("")
    for family, count in summary.get("model_family_counts", {}).items():
        lines.append(f"- {family}: {count}")
    lines.append("")
    lines.append("## Best rows")
    lines.append("")
    for name, item in summary.get("best_rows", {}).items():
        lines.append(f"### {name}")
        lines.append("")
        if item is None:
            lines.append("No valid row.")
        else:
            for key, value in item.items():
                lines.append(f"- {key}: {value}")
        lines.append("")

    if not frame.empty:
        display_columns = [
            "rank",
            "strategy",
            "source",
            "variant",
            "dataset_id",
            "pit_split_id",
            "point_in_time",
            "trade_end_date",
            "seed_count",
            "total_return_pct",
            "annualized_sharpe",
            "max_drawdown_pct",
            "cvar_pct",
            "final_value",
        ]
        display_columns = [
            column for column in display_columns if column in frame.columns
        ]
        lines.append("## Top rows by total return")
        lines.append("")
        lines.append(frame[display_columns].head(20).to_markdown(index=False))
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "The IQN multiseed mean row is the main row for comparing the "
        "distributional D-IQN-DSS result against FinRL baselines on the same "
        "long point-in-time split. Per-seed IQN rows are included to preserve "
        "variation and avoid hiding instability."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    project_root = find_project_root()
    run_paths = create_run_directories(project_root)
    recent_run_limit = get_int_env(
        "STOCK_INVESTMENT_DSS_COMPARISON_RECENT_RUN_LIMIT",
        DEFAULT_RECENT_RUN_LIMIT,
    )
    include_generic = get_bool_env(
        "STOCK_INVESTMENT_DSS_COMPARISON_INCLUDE_GENERIC", True
    )
    dataset_id_filter = get_str_env("STOCK_INVESTMENT_DSS_COMPARISON_DATASET_ID")
    pit_split_id_filter = get_str_env("STOCK_INVESTMENT_DSS_COMPARISON_PIT_SPLIT_ID")
    point_in_time_filter = get_str_env(
        "STOCK_INVESTMENT_DSS_COMPARISON_POINT_IN_TIME"
    ) or get_str_env("STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME")
    trade_end_date_filter = get_str_env(
        "STOCK_INVESTMENT_DSS_COMPARISON_TRADE_END_DATE"
    ) or get_str_env("STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE")
    seed_filter = parse_int_list_env("STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST")
    latest_multiseed_only = get_bool_env(
        "STOCK_INVESTMENT_DSS_COMPARISON_LATEST_MULTI_SEED_ONLY",
        True,
    )

    log(
        "stock_investment_dss.system | Starting StockInvestmentDSS IQN vs baseline comparison summary."
    )
    log(f"stock_investment_dss.system | Project root: {project_root}")
    log(
        f"stock_investment_dss.run.{run_paths.run_id} | Created run directory: {run_paths.run_directory}"
    )
    log(f"stock_investment_dss.run.{run_paths.run_id} | Run id: {run_paths.run_id}")
    log(
        f"stock_investment_dss.run.{run_paths.run_id} | Recent run limit: {recent_run_limit}"
    )
    log(
        f"stock_investment_dss.run.{run_paths.run_id} | Dataset filter: {dataset_id_filter}"
    )
    log(
        f"stock_investment_dss.run.{run_paths.run_id} | PIT split filter: {pit_split_id_filter}"
    )
    log(
        f"stock_investment_dss.run.{run_paths.run_id} | Point-in-time filter: {point_in_time_filter}"
    )
    log(
        f"stock_investment_dss.run.{run_paths.run_id} | Trade-end filter: {trade_end_date_filter}"
    )
    log(f"stock_investment_dss.run.{run_paths.run_id} | Seed filter: {seed_filter}")
    log(
        f"stock_investment_dss.run.{run_paths.run_id} | Latest IQN multiseed only: {latest_multiseed_only}"
    )

    rows = collect_rows(project_root, recent_run_limit)
    rows_before_context_filter = len(rows)
    if not include_generic:
        rows = [row for row in rows if row.get("row_type") != "generic_v1_summary"]

    rows, context_filter_summary = filter_rows_by_context(
        rows=rows,
        dataset_id_filter=dataset_id_filter,
        pit_split_id_filter=pit_split_id_filter,
        point_in_time_filter=point_in_time_filter,
        trade_end_date_filter=trade_end_date_filter,
        seed_filter=seed_filter,
    )

    rows, latest_multiseed_filter_summary = filter_latest_iqn_multiseed_summary(
        rows=rows,
        enabled=latest_multiseed_only,
    )

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        frame = rank_frame(frame)

    family_frame = family_summary(frame)

    comparison_csv_path = (
        run_paths.summary_directory / "iqn_vs_baseline_comparison_summary.csv"
    )
    comparison_data_path = (
        run_paths.data_directory / "iqn_vs_baseline_comparison_table.csv"
    )
    family_summary_path = (
        run_paths.summary_directory / "iqn_vs_baseline_family_summary.csv"
    )
    markdown_path = (
        run_paths.summary_directory / "iqn_vs_baseline_comparison_summary.md"
    )
    summary_path = (
        run_paths.summary_directory / "iqn_vs_baseline_comparison_summary.json"
    )

    frame.to_csv(comparison_csv_path, index=False)
    frame.to_csv(comparison_data_path, index=False)
    family_frame.to_csv(family_summary_path, index=False)
    fair_outputs = write_fair_comparison_artifacts(frame=frame, run_paths=run_paths)

    source_counts = (
        frame["source"].value_counts(dropna=False).to_dict() if not frame.empty else {}
    )
    model_family_counts = (
        frame["model_family"].value_counts(dropna=False).to_dict()
        if not frame.empty
        else {}
    )

    best_rows = {
        "best_by_total_return_pct": best_row(
            frame, "total_return_pct", ascending=False
        ),
        "best_by_annualized_sharpe": best_row(
            frame, "annualized_sharpe", ascending=False
        ),
        # Higher max_drawdown_pct is better because drawdown is negative.
        "best_by_max_drawdown_pct": best_row(
            frame, "max_drawdown_pct", ascending=False
        ),
        # Higher CVaR is less bad because it is usually negative.
        "best_by_cvar_pct": best_row(frame, "cvar_pct", ascending=False),
    }

    summary = {
        "status": "ok",
        "project_name": PROJECT_NAME,
        "prototype_name": PROTOTYPE_NAME,
        "run_id": run_paths.run_id,
        "project_root": str(project_root),
        "run_directory": str(run_paths.run_directory),
        "runs_scanned": len(recent_run_directories(project_root, recent_run_limit)),
        "recent_run_limit": recent_run_limit,
        "row_count_before_context_filter": rows_before_context_filter,
        "row_count": int(len(frame)),
        "context_filter": context_filter_summary,
        "latest_multiseed_filter": latest_multiseed_filter_summary,
        "source_counts": source_counts,
        "model_family_counts": model_family_counts,
        "fair_outputs": fair_outputs,
        "best_rows": best_rows,
        "outputs": {
            "comparison_csv_path": str(comparison_csv_path),
            "comparison_data_path": str(comparison_data_path),
            "family_summary_path": str(family_summary_path),
            "fair_compact_summary_path": fair_outputs["compact_summary_path"],
            "fair_compact_data_path": fair_outputs["compact_data_path"],
            "fair_seed_diagnostic_summary_path": fair_outputs["seed_diagnostic_summary_path"],
            "fair_seed_diagnostic_data_path": fair_outputs["seed_diagnostic_data_path"],
            "markdown_path": str(markdown_path),
            "summary_path": str(summary_path),
        },
        "interpretation": (
            "This comparison includes FinRL/SB3 baseline suite rows and D-IQN-DSS "
            "distributional IQN rows. V2.6 additionally supports dataset/PIT/seed "
            "context filtering and latest-IQN-multiseed filtering, so the thesis "
            "table can be restricted to the fair long-period split without "
            "mixing older smoke-test runs or duplicate multiseed summary reruns."
        ),
        "next_step": (
            "Use the comparison CSV for thesis tables and rerun the comparison plot. "
            "If the plot should show the IQN multiseed mean as a curve, update the "
            "plot runner to synthesize or aggregate a multiseed portfolio trajectory."
        ),
    }

    markdown_path.write_text(markdown_report(frame, summary), encoding="utf-8")
    write_json(summary_path, summary)

    log(
        f"stock_investment_dss.run.{run_paths.run_id} | IQN vs baseline comparison summary completed."
    )
    log(
        f"stock_investment_dss.run.{run_paths.run_id} | Context filter summary: {context_filter_summary}"
    )
    log(
        f"stock_investment_dss.run.{run_paths.run_id} | Latest multiseed filter summary: {latest_multiseed_filter_summary}"
    )
    log(f"stock_investment_dss.run.{run_paths.run_id} | Rows found: {len(frame)}")
    log(f"stock_investment_dss.run.{run_paths.run_id} | Source counts: {source_counts}")
    log(
        f"stock_investment_dss.run.{run_paths.run_id} | Model family counts: {model_family_counts}"
    )
    log(f"stock_investment_dss.run.{run_paths.run_id} | Fair outputs: {fair_outputs}")
    log(f"stock_investment_dss.run.{run_paths.run_id} | Best rows: {best_rows}")
    log(
        f"stock_investment_dss.run.{run_paths.run_id} | Wrote comparison CSV: {comparison_csv_path}"
    )
    log(
        f"stock_investment_dss.run.{run_paths.run_id} | Wrote markdown report: {markdown_path}"
    )
    log(
        f"stock_investment_dss.run.{run_paths.run_id} | Wrote summary JSON: {summary_path}"
    )
    log(
        "stock_investment_dss.system | StockInvestmentDSS IQN vs baseline comparison summary completed successfully."
    )


if __name__ == "__main__":
    main()
