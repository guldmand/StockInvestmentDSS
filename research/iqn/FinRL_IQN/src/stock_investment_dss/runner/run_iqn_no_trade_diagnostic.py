# src/stock_investment_dss/runner/run_iqn_no_trade_diagnostic.py
"""Diagnose IQN no-trade seeds.

Read-only runner. It does not train models. It inspects the latest IQN multiseed
summary and the child IQN learning-curve runs, focusing on seeds that ended with
0% return and 0 trades.

v2.8b update:
- distinguishes best non-HOLD action from best ALLOWED non-HOLD action
- reports whether HOLD beat BUY among allowed actions
- avoids misleading conclusions when disallowed REBALANCE has a higher score
"""

from __future__ import annotations

import ast
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from stock_investment_dss.utilities.config import get_environment_variable
from stock_investment_dss.utilities.logging import setup_run_logger, setup_system_logger
from stock_investment_dss.utilities.paths import PROJECT_ROOT, create_run_paths


RUN_KIND = "d_iqn_dss_iqn_no_trade_diagnostic"
PROJECT_NAME = "StockInvestmentDSS"
PROTOTYPE_NAME = "D-IQN-DSS"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def normalize(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int | None = None) -> int | None:
    value_float = to_float(value, None)
    if value_float is None:
        return default
    return int(value_float)


def parse_seed_list(value: str | None) -> list[int]:
    if value is None or not str(value).strip():
        return []
    return sorted({int(part.strip()) for part in str(value).split(",") if part.strip()})


def count_values(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df.empty or column not in df.columns:
        return {}
    counts = Counter()
    for value in df[column].tolist():
        text = normalize(value) or "UNKNOWN"
        counts[text] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def count_true(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(sum(1 for value in df[column].tolist() if parse_bool(value) is True))


def find_latest_multiseed_summary_run() -> Path:
    explicit_dir = get_environment_variable(
        "STOCK_INVESTMENT_DSS_IQN_NO_TRADE_SOURCE_SUMMARY_RUN_DIR",
        default="",
    )
    explicit_id = get_environment_variable(
        "STOCK_INVESTMENT_DSS_IQN_NO_TRADE_SOURCE_SUMMARY_RUN_ID",
        default="",
    )

    if explicit_dir:
        path = Path(explicit_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Source summary run directory not found: {path}")
        return path.resolve()

    if explicit_id:
        path = PROJECT_ROOT / "outputs" / "runs" / explicit_id
        if not path.exists():
            raise FileNotFoundError(f"Source summary run id not found: {path}")
        return path.resolve()

    runs_root = PROJECT_ROOT / "outputs" / "runs"
    candidates = sorted(
        [
            path
            for path in runs_root.iterdir()
            if path.is_dir()
            and path.name.endswith("d_iqn_dss_iqn_learning_curve_multiseed_summary")
            and (path / "summary" / "iqn_learning_curve_multiseed_final_records.csv").exists()
            and (path / "data" / "iqn_learning_curve_multiseed_run_index.csv").exists()
        ],
        key=lambda path: path.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No IQN multiseed summary run found.")
    return candidates[0].resolve()


def load_summary_inputs(source_summary_run: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    final_records = pd.read_csv(
        source_summary_run / "summary" / "iqn_learning_curve_multiseed_final_records.csv"
    )
    run_index = pd.read_csv(
        source_summary_run / "data" / "iqn_learning_curve_multiseed_run_index.csv"
    )
    summary = read_json(source_summary_run / "summary" / "iqn_learning_curve_multiseed_summary.json")

    if "seed" in final_records.columns:
        final_records["seed"] = pd.to_numeric(final_records["seed"], errors="coerce")
    if "seed" in run_index.columns:
        run_index["seed"] = pd.to_numeric(run_index["seed"], errors="coerce")

    return final_records, run_index, summary


def select_target_seeds(final_records: pd.DataFrame) -> list[int]:
    explicit = parse_seed_list(
        get_environment_variable("STOCK_INVESTMENT_DSS_IQN_NO_TRADE_SEEDS", default="")
    )
    if explicit:
        return explicit

    if final_records.empty or "seed" not in final_records.columns:
        return []

    trades = pd.to_numeric(final_records.get("total_trades", 0), errors="coerce").fillna(0)
    returns = pd.to_numeric(final_records.get("total_return_pct", 0), errors="coerce").fillna(0)
    final_value = pd.to_numeric(final_records.get("final_value", 0), errors="coerce").fillna(0)
    initial_value = pd.to_numeric(final_records.get("initial_value", 1_000_000), errors="coerce").fillna(1_000_000)

    mask = (trades <= 0) & (returns.abs() < 1e-9) & ((final_value - initial_value).abs() < 1e-6)
    return sorted(int(seed) for seed in final_records.loc[mask, "seed"].dropna().tolist())


def find_source_run_for_seed(seed: int, run_index: pd.DataFrame) -> Path | None:
    if run_index.empty or "seed" not in run_index.columns:
        return None

    matches = run_index[run_index["seed"].astype("Int64") == int(seed)]
    if matches.empty:
        return None

    row = matches.iloc[-1].to_dict()
    candidates: list[Any] = [
        row.get("run_directory"),
        row.get("source_run_directory"),
        row.get("source_run_dir"),
    ]

    run_id = row.get("run_id") or row.get("source_run_id")
    if run_id:
        candidates.append(PROJECT_ROOT / "outputs" / "runs" / str(run_id))

    for candidate in candidates:
        if candidate is None or str(candidate).strip() == "":
            continue
        path = Path(str(candidate))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            return path.resolve()

    return None


def latest_train_step(df: pd.DataFrame, fallback: int | None = None) -> int | None:
    if df.empty or "train_step" not in df.columns:
        return fallback
    values = pd.to_numeric(df["train_step"], errors="coerce").dropna()
    if values.empty:
        return fallback
    return int(values.max())


def summarize_step_records(seed: int, source_run_dir: Path, final_step: int | None) -> tuple[dict[str, Any], pd.DataFrame]:
    step_path = source_run_dir / "data" / "iqn_learning_curve_eval_step_records.csv"
    step_df = safe_read_csv(step_path)
    if step_df.empty:
        return {
            "seed": seed,
            "source_run_id": source_run_dir.name,
            "step_records_found": False,
            "inferred_no_trade_cause": "missing_step_records",
        }, pd.DataFrame()

    step = latest_train_step(step_df, fallback=final_step)
    if step is not None and "train_step" in step_df.columns:
        step_df["train_step"] = pd.to_numeric(step_df["train_step"], errors="coerce")
        selected = step_df[step_df["train_step"] == step].copy()
    else:
        selected = step_df.copy()

    chosen_counts = count_values(selected, "chosen_action_label")
    effective_counts = count_values(selected, "effective_action")
    ticker_counts = count_values(selected, "selected_ticker")
    reason_counts = count_values(selected, "resolved_reason")

    trades_delta = (
        pd.to_numeric(selected.get("trades_delta", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        if "trades_delta" in selected.columns
        else pd.Series(dtype=float)
    )
    cost_delta = (
        pd.to_numeric(selected.get("cost_delta", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        if "cost_delta" in selected.columns
        else pd.Series(dtype=float)
    )
    executed_shares_delta = (
        pd.to_numeric(selected.get("executed_shares_delta", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        if "executed_shares_delta" in selected.columns
        else pd.Series(dtype=float)
    )

    non_hold_chosen = int(
        sum(count for action, count in chosen_counts.items() if str(action).upper() not in {"HOLD", "UNKNOWN", "NONE", "NAN"})
    )
    non_hold_effective = int(
        sum(count for action, count in effective_counts.items() if str(action).upper() not in {"HOLD", "UNKNOWN", "NONE", "NAN"})
    )

    selected["seed"] = seed
    selected["source_run_id"] = source_run_dir.name

    summary = {
        "seed": seed,
        "source_run_id": source_run_dir.name,
        "source_run_directory": str(source_run_dir),
        "step_records_found": True,
        "diagnostic_scope": "final_eval_checkpoint",
        "train_step": step,
        "decision_rows": int(len(selected)),
        "chosen_action_counts": json.dumps(chosen_counts, sort_keys=True),
        "effective_action_counts": json.dumps(effective_counts, sort_keys=True),
        "selected_ticker_counts": json.dumps(ticker_counts, sort_keys=True),
        "resolved_reason_counts": json.dumps(reason_counts, sort_keys=True),
        "action_was_masked_count": count_true(selected, "action_was_masked"),
        "non_hold_chosen_count": non_hold_chosen,
        "non_hold_effective_count": non_hold_effective,
        "trade_rows": int((trades_delta > 0).sum()) if len(trades_delta) else 0,
        "total_trades_delta": float(trades_delta.sum()) if len(trades_delta) else 0.0,
        "total_cost_delta": float(cost_delta.sum()) if len(cost_delta) else 0.0,
        "total_executed_shares_delta_abs": float(executed_shares_delta.abs().sum()) if len(executed_shares_delta) else None,
    }

    return summary, selected


def choose_best_action(rows: pd.DataFrame, score_column: str = "score_mean") -> dict[str, Any]:
    if rows.empty or score_column not in rows.columns:
        return {
            "action": None,
            "score": None,
            "q50": None,
            "cvar10": None,
            "allowed_count": None,
            "row_count": None,
        }

    working = rows.copy()
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
    working = working.dropna(subset=[score_column])
    if working.empty:
        return {
            "action": None,
            "score": None,
            "q50": None,
            "cvar10": None,
            "allowed_count": None,
            "row_count": None,
        }

    row = working.sort_values(score_column, ascending=False).iloc[0]
    return {
        "action": normalize(row.get("action")),
        "score": to_float(row.get(score_column)),
        "q50": to_float(row.get("q50_mean")),
        "cvar10": to_float(row.get("cvar10_mean")),
        "allowed_count": to_int(row.get("allowed_count")),
        "row_count": to_int(row.get("row_count")),
    }


def summarize_distribution_records(seed: int, source_run_dir: Path, final_step: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    distribution_path = source_run_dir / "data" / "iqn_learning_curve_eval_distributions.csv"
    distribution_df = safe_read_csv(distribution_path)
    if distribution_df.empty:
        return pd.DataFrame(), {
            "seed": seed,
            "source_run_id": source_run_dir.name,
            "distribution_records_found": False,
        }

    step = latest_train_step(distribution_df, fallback=final_step)
    if step is not None and "train_step" in distribution_df.columns:
        distribution_df["train_step"] = pd.to_numeric(distribution_df["train_step"], errors="coerce")
        selected = distribution_df[distribution_df["train_step"] == step].copy()
    else:
        selected = distribution_df.copy()

    for column in ["score", "mean", "q10", "q25", "q50", "q75", "q90", "cvar10"]:
        if column in selected.columns:
            selected[column] = pd.to_numeric(selected[column], errors="coerce")

    if "allowed" in selected.columns:
        selected["allowed_bool"] = selected["allowed"].map(parse_bool)
    else:
        selected["allowed_bool"] = True

    action_column = "action" if "action" in selected.columns else None
    if action_column is None:
        return pd.DataFrame(), {
            "seed": seed,
            "source_run_id": source_run_dir.name,
            "distribution_records_found": True,
            "distribution_summary_available": False,
        }

    rows: list[dict[str, Any]] = []
    for action, group in selected.groupby(action_column, dropna=False):
        row = {
            "seed": seed,
            "source_run_id": source_run_dir.name,
            "train_step": step,
            "action": str(action),
            "row_count": int(len(group)),
            "allowed_count": int((group["allowed_bool"] == True).sum()),
        }
        for metric in ["score", "mean", "q10", "q25", "q50", "q75", "q90", "cvar10"]:
            if metric in group.columns:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                row[f"{metric}_mean"] = float(values.mean()) if not values.empty else None
                row[f"{metric}_min"] = float(values.min()) if not values.empty else None
                row[f"{metric}_max"] = float(values.max()) if not values.empty else None
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_min"] = None
                row[f"{metric}_max"] = None
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df, {
            "seed": seed,
            "source_run_id": source_run_dir.name,
            "distribution_records_found": True,
            "distribution_summary_available": False,
        }

    summary_df["action_upper"] = summary_df["action"].astype(str).str.upper()
    summary_df["allowed_count_numeric"] = pd.to_numeric(summary_df["allowed_count"], errors="coerce").fillna(0)

    hold_rows = summary_df[summary_df["action_upper"] == "HOLD"]
    buy_rows = summary_df[summary_df["action_upper"] == "BUY"]
    all_non_hold = summary_df[summary_df["action_upper"] != "HOLD"].copy()
    allowed_non_hold = all_non_hold[all_non_hold["allowed_count_numeric"] > 0].copy()

    hold = choose_best_action(hold_rows)
    buy = choose_best_action(buy_rows)
    best_non_hold = choose_best_action(all_non_hold)
    best_allowed_non_hold = choose_best_action(allowed_non_hold)

    hold_score = hold["score"]
    buy_score = buy["score"]
    allowed_non_hold_score = best_allowed_non_hold["score"]

    overall = {
        "seed": seed,
        "source_run_id": source_run_dir.name,
        "distribution_records_found": True,
        "distribution_summary_available": True,
        "train_step": step,
        "distribution_rows": int(len(selected)),

        "hold_score_mean": hold_score,
        "hold_q50_mean": hold["q50"],
        "hold_cvar10_mean": hold["cvar10"],
        "hold_allowed_count": hold["allowed_count"],

        "buy_score_mean": buy_score,
        "buy_q50_mean": buy["q50"],
        "buy_cvar10_mean": buy["cvar10"],
        "buy_allowed_count": buy["allowed_count"],

        "best_non_hold_action_by_score": best_non_hold["action"],
        "best_non_hold_score_mean": best_non_hold["score"],
        "best_non_hold_allowed_count": best_non_hold["allowed_count"],

        "best_allowed_non_hold_action_by_score": best_allowed_non_hold["action"],
        "best_allowed_non_hold_score_mean": allowed_non_hold_score,
        "best_allowed_non_hold_q50_mean": best_allowed_non_hold["q50"],
        "best_allowed_non_hold_cvar10_mean": best_allowed_non_hold["cvar10"],
        "best_allowed_non_hold_allowed_count": best_allowed_non_hold["allowed_count"],

        "hold_score_minus_buy_score": (
            float(hold_score - buy_score)
            if hold_score is not None and buy_score is not None
            else None
        ),
        "hold_score_minus_best_allowed_non_hold_score": (
            float(hold_score - allowed_non_hold_score)
            if hold_score is not None and allowed_non_hold_score is not None
            else None
        ),
        "best_non_hold_was_disallowed": (
            best_non_hold["action"] is not None
            and best_non_hold["allowed_count"] is not None
            and int(best_non_hold["allowed_count"]) <= 0
        ),
    }

    return summary_df.drop(columns=["action_upper", "allowed_count_numeric"], errors="ignore"), overall


def infer_cause(step_summary: dict[str, Any], distribution_summary: dict[str, Any]) -> str:
    if not step_summary.get("step_records_found"):
        return "unknown_missing_step_records"

    decision_rows = int(step_summary.get("decision_rows") or 0)
    non_hold_chosen = int(step_summary.get("non_hold_chosen_count") or 0)
    non_hold_effective = int(step_summary.get("non_hold_effective_count") or 0)
    masked = int(step_summary.get("action_was_masked_count") or 0)
    trade_rows = int(step_summary.get("trade_rows") or 0)

    if decision_rows > 0 and non_hold_chosen == 0:
        allowed_margin = distribution_summary.get("hold_score_minus_best_allowed_non_hold_score")
        buy_margin = distribution_summary.get("hold_score_minus_buy_score")
        best_non_hold_disallowed = bool(distribution_summary.get("best_non_hold_was_disallowed"))

        if allowed_margin is not None and float(allowed_margin) >= 0:
            if buy_margin is not None and float(buy_margin) >= 0:
                if best_non_hold_disallowed:
                    return "policy_selected_hold_over_allowed_buy_disallowed_rebalance_scored_higher"
                return "policy_selected_hold_over_allowed_non_hold"
            return "policy_selected_hold_over_allowed_non_hold"
        return "policy_selected_hold"

    if non_hold_chosen > 0 and masked > 0 and non_hold_effective == 0:
        return "non_hold_chosen_but_masked_or_resolved_to_hold"

    if non_hold_chosen > 0 and non_hold_effective > 0 and trade_rows == 0:
        return "non_hold_effective_but_no_execution"

    if non_hold_chosen > 0 and trade_rows == 0:
        return "non_hold_chosen_but_no_trades"

    return "no_trade_reason_unclear"


def write_markdown(path: Path, summary: dict[str, Any], seed_df: pd.DataFrame) -> None:
    lines = [
        "# IQN No-Trade Diagnostic",
        "",
        "## Source",
        "",
        f"- Source summary run: {summary.get('source_summary_run_id')}",
        f"- Target seeds: {summary.get('target_seeds')}",
        f"- Cause counts: {summary.get('cause_counts')}",
        "",
        "## Main conclusion",
        "",
        "- This diagnostic is read-only.",
        "- It checks whether no-trade seeds selected HOLD, were masked/resolved into no-op actions, or had scores favoring HOLD over allowed non-HOLD actions.",
        "- The v2.8b diagnostic separates best non-HOLD action from best allowed non-HOLD action.",
        "",
        "## Seed-level summary",
        "",
    ]

    if seed_df.empty:
        lines.append("- No seed rows produced.")
    else:
        columns = [
            "seed",
            "final_total_return_pct",
            "final_total_trades",
            "decision_rows",
            "non_hold_chosen_count",
            "non_hold_effective_count",
            "action_was_masked_count",
            "trade_rows",
            "hold_score_mean",
            "buy_score_mean",
            "hold_score_minus_buy_score",
            "best_non_hold_action_by_score",
            "best_non_hold_allowed_count",
            "best_allowed_non_hold_action_by_score",
            "best_allowed_non_hold_score_mean",
            "hold_score_minus_best_allowed_non_hold_score",
            "inferred_no_trade_cause",
        ]
        available = [column for column in columns if column in seed_df.columns]
        lines.append(seed_df[available].to_markdown(index=False))

    lines.extend(
        [
            "",
            "## Reading guide",
            "",
            "- non_hold_chosen_count = 0 means the policy selected HOLD.",
            "- action_was_masked_count > 0 means action masking may explain no execution.",
            "- best_non_hold_action_by_score can be misleading if that action is not allowed.",
            "- best_allowed_non_hold_action_by_score is the relevant comparison against HOLD.",
            "- hold_score_minus_buy_score > 0 means HOLD was scored above BUY.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_margin_plot(seed_df: pd.DataFrame, output_path: Path) -> None:
    if seed_df.empty:
        return
    plot_df = seed_df.copy()
    for column in ["hold_score_minus_buy_score", "hold_score_minus_best_allowed_non_hold_score"]:
        if column in plot_df.columns:
            plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    if "hold_score_minus_buy_score" not in plot_df.columns:
        return

    plot_df["seed_label"] = plot_df["seed"].map(lambda seed: f"seed {int(seed)}")
    x = range(len(plot_df))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(
        [i - width / 2 for i in x],
        plot_df["hold_score_minus_buy_score"],
        width=width,
        label="HOLD - BUY score",
    )
    if "hold_score_minus_best_allowed_non_hold_score" in plot_df.columns:
        plt.bar(
            [i + width / 2 for i in x],
            plot_df["hold_score_minus_best_allowed_non_hold_score"],
            width=width,
            label="HOLD - best allowed non-HOLD score",
        )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(list(x), plot_df["seed_label"])
    plt.title("IQN no-trade diagnostic: HOLD score margins")
    plt.ylabel("Score margin")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_action_count_plot(seed_df: pd.DataFrame, output_path: Path) -> None:
    if seed_df.empty:
        return
    plot_df = seed_df.copy()
    for column in ["decision_rows", "non_hold_chosen_count", "trade_rows"]:
        if column in plot_df.columns:
            plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce").fillna(0)
    plot_df["seed_label"] = plot_df["seed"].map(lambda seed: f"seed {int(seed)}")
    x = range(len(plot_df))
    width = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar([i - width for i in x], plot_df["decision_rows"], width=width, label="decision rows")
    plt.bar(x, plot_df["non_hold_chosen_count"], width=width, label="non-HOLD chosen")
    plt.bar([i + width for i in x], plot_df["trade_rows"], width=width, label="trade rows")
    plt.xticks(list(x), plot_df["seed_label"])
    plt.title("IQN no-trade diagnostic: action/execution counts")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    log_level = get_environment_variable("STOCK_INVESTMENT_DSS_LOG_LEVEL", default="INFO") or "INFO"
    system_logger = setup_system_logger(log_level=log_level)
    system_logger.info("Starting StockInvestmentDSS IQN no-trade diagnostic.")
    system_logger.info("Project root: %s", PROJECT_ROOT)

    run_paths = None

    try:
        source_summary_run = find_latest_multiseed_summary_run()
        final_records, run_index, source_summary = load_summary_inputs(source_summary_run)
        target_seeds = select_target_seeds(final_records)

        run_paths = create_run_paths(RUN_KIND)
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Source summary run: %s", source_summary_run)
        run_logger.info("Target seeds: %s", target_seeds)

        seed_rows: list[dict[str, Any]] = []
        all_step_rows: list[pd.DataFrame] = []
        all_distribution_rows: list[pd.DataFrame] = []

        for seed in target_seeds:
            source_run_dir = find_source_run_for_seed(seed, run_index)
            final_match = final_records[final_records["seed"].astype("Int64") == int(seed)]
            final_row = final_match.iloc[-1].to_dict() if not final_match.empty else {}

            if source_run_dir is None:
                seed_rows.append(
                    {
                        "seed": seed,
                        "source_run_found": False,
                        "inferred_no_trade_cause": "source_run_not_found",
                    }
                )
                continue

            final_step = to_int(final_row.get("train_step"), None)
            step_summary, step_df = summarize_step_records(seed, source_run_dir, final_step)
            distribution_df, distribution_summary = summarize_distribution_records(seed, source_run_dir, final_step)

            cause = infer_cause(step_summary, distribution_summary)

            combined = {
                **{f"final_{key}": value for key, value in final_row.items()},
                **step_summary,
                **distribution_summary,
                "source_run_found": True,
                "inferred_no_trade_cause": cause,
            }
            seed_rows.append(combined)

            if not step_df.empty:
                all_step_rows.append(step_df)
            if not distribution_df.empty:
                all_distribution_rows.append(distribution_df)

        seed_df = pd.DataFrame(seed_rows)
        if not seed_df.empty and "seed" in seed_df.columns:
            seed_df = seed_df.sort_values("seed")

        step_rows_df = pd.concat(all_step_rows, ignore_index=True) if all_step_rows else pd.DataFrame()
        distribution_rows_df = (
            pd.concat(all_distribution_rows, ignore_index=True)
            if all_distribution_rows
            else pd.DataFrame()
        )

        cause_counts = (
            seed_df["inferred_no_trade_cause"].value_counts().to_dict()
            if not seed_df.empty and "inferred_no_trade_cause" in seed_df.columns
            else {}
        )

        seed_summary_path = run_paths.summary_directory / "iqn_no_trade_diagnostic_by_seed.csv"
        summary_json_path = run_paths.summary_directory / "iqn_no_trade_diagnostic_summary.json"
        summary_md_path = run_paths.summary_directory / "iqn_no_trade_diagnostic_summary.md"
        step_rows_path = run_paths.data_directory / "iqn_no_trade_diagnostic_eval_step_rows.csv"
        distribution_rows_path = run_paths.data_directory / "iqn_no_trade_diagnostic_distribution_by_action.csv"
        action_plot_path = run_paths.plots_directory / "iqn_no_trade_action_execution_counts.png"
        margin_plot_path = run_paths.plots_directory / "iqn_no_trade_hold_score_margins.png"

        seed_df.to_csv(seed_summary_path, index=False)
        step_rows_df.to_csv(step_rows_path, index=False)
        distribution_rows_df.to_csv(distribution_rows_path, index=False)
        save_action_count_plot(seed_df, action_plot_path)
        save_margin_plot(seed_df, margin_plot_path)

        summary_payload = {
            "status": "ok",
            "project_name": PROJECT_NAME,
            "prototype_name": PROTOTYPE_NAME,
            "run_id": run_paths.run_id,
            "run_directory": str(run_paths.run_directory),
            "source_summary_run_id": source_summary_run.name,
            "source_summary_run_directory": str(source_summary_run),
            "source_multiseed_summary_run_id": source_summary.get("run_id"),
            "target_seeds": target_seeds,
            "target_seed_count": len(target_seeds),
            "cause_counts": cause_counts,
            "outputs": {
                "seed_summary_path": str(seed_summary_path),
                "summary_json_path": str(summary_json_path),
                "summary_md_path": str(summary_md_path),
                "step_rows_path": str(step_rows_path),
                "distribution_rows_path": str(distribution_rows_path),
                "action_plot_path": str(action_plot_path) if action_plot_path.exists() else None,
                "margin_plot_path": str(margin_plot_path) if margin_plot_path.exists() else None,
            },
            "interpretation": (
                "v2.8b distinguishes best non-HOLD from best allowed non-HOLD action. "
                "This avoids blaming HOLD when a higher-scoring action was disallowed."
            ),
        }
        write_json(summary_json_path, summary_payload)
        write_markdown(summary_md_path, summary_payload, seed_df)

        run_logger.info("IQN no-trade diagnostic completed.")
        run_logger.info("Target seeds: %s", target_seeds)
        run_logger.info("Cause counts: %s", cause_counts)
        run_logger.info("Wrote seed summary: %s", seed_summary_path)
        run_logger.info("Wrote summary: %s", summary_json_path)

        system_logger.info("StockInvestmentDSS IQN no-trade diagnostic completed successfully.")
        return 0

    except Exception:
        system_logger.exception("StockInvestmentDSS IQN no-trade diagnostic failed.")
        if run_paths is not None:
            try:
                run_logger = setup_run_logger(run_paths, log_level=log_level)
                run_logger.exception("Run failed.")
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
