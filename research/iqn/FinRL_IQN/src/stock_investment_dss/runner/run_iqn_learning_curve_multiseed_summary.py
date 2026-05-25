from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stock_investment_dss.utilities.config import get_environment_variable
from stock_investment_dss.utilities.logging import setup_run_logger, setup_system_logger
from stock_investment_dss.utilities.paths import PROJECT_ROOT, create_run_paths

RUN_SUFFIX = "d_iqn_dss_iqn_learning_curve_smoke_test"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def get_int_environment_variable(name: str, default: int) -> int:
    value = get_environment_variable(name, default=str(default))
    return int(value or default)


def parse_seed_list(value: str | None) -> list[int]:
    if value is None:
        return []

    seeds: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        seeds.append(int(part))

    return sorted(set(seeds))


def find_learning_curve_runs(runs_root: Path, recent_limit: int) -> list[Path]:
    if not runs_root.exists():
        return []

    candidates = [
        path
        for path in runs_root.iterdir()
        if path.is_dir()
        and path.name.endswith(RUN_SUFFIX)
        and (path / "summary" / "iqn_learning_curve_summary.json").exists()
        and (path / "data" / "iqn_learning_curve_eval_records.csv").exists()
    ]

    candidates = sorted(candidates, key=lambda path: path.name, reverse=True)
    return candidates[:recent_limit]


def flatten_action_counts(action_counts: Any) -> str:
    if isinstance(action_counts, str):
        return action_counts
    if isinstance(action_counts, dict):
        return json.dumps(action_counts, sort_keys=True)
    return ""


def load_run(run_directory: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary_path = run_directory / "summary" / "iqn_learning_curve_summary.json"
    eval_path = run_directory / "data" / "iqn_learning_curve_eval_records.csv"

    summary = read_json(summary_path)
    iqn = summary.get("iqn", {})
    config = iqn.get("config", {})
    final_eval = iqn.get("final_eval", {})

    seed = config.get("seed")
    total_steps = iqn.get("total_steps")
    learning_starts = iqn.get("learning_starts")
    eval_interval = iqn.get("eval_interval")
    score_mode = iqn.get("eval_score_mode")
    risk_lambda = iqn.get("risk_lambda")

    # Load hold diagnostic summary if present
    hold_diag: dict[str, Any] = {}
    hold_diag_path = run_directory / "summary" / "hold_diagnostic_summary.json"
    if hold_diag_path.exists():
        try:
            hold_diag = read_json(hold_diag_path)
        except Exception:
            pass

    eval_records = pd.read_csv(eval_path)

    eval_records["source_run_id"] = summary.get("run_id", run_directory.name)
    eval_records["source_run_directory"] = str(run_directory)
    eval_records["seed"] = seed
    eval_records["dataset_id"] = summary.get("dataset_id")
    eval_records["universe_id"] = summary.get("universe_id")
    eval_records["score_mode"] = score_mode
    eval_records["risk_lambda"] = risk_lambda
    eval_records["configured_total_steps"] = total_steps
    eval_records["learning_starts"] = learning_starts
    eval_records["eval_interval"] = eval_interval

    if "action_counts" in eval_records.columns:
        eval_records["action_counts"] = eval_records["action_counts"].apply(
            flatten_action_counts
        )

    metadata = {
        "run_id": summary.get("run_id", run_directory.name),
        "run_directory": str(run_directory),
        "seed": seed,
        "dataset_id": summary.get("dataset_id"),
        "universe_id": summary.get("universe_id"),
        "point_in_time": summary.get("point_in_time"),
        "trade_end_date": summary.get("trade_end_date"),
        "total_steps": total_steps,
        "learning_starts": learning_starts,
        "eval_interval": eval_interval,
        "score_mode": score_mode,
        "risk_lambda": risk_lambda,
        "learn_steps": iqn.get("learn_steps"),
        "final_buffer_size": iqn.get("final_buffer_size"),
        "loss_initial": iqn.get("loss_initial"),
        "loss_final": iqn.get("loss_final"),
        "loss_min": iqn.get("loss_min"),
        "loss_max": iqn.get("loss_max"),
        "loss_mean": iqn.get("loss_mean"),
        "training_action_counts": json.dumps(
            iqn.get("training_action_counts", {}), sort_keys=True
        ),
        "final_eval_total_return_pct": final_eval.get("total_return_pct"),
        "final_eval_annualized_sharpe": final_eval.get("annualized_sharpe"),
        "final_eval_max_drawdown_pct": final_eval.get("max_drawdown_pct"),
        "final_eval_cvar_pct": final_eval.get("cvar_pct"),
        "final_eval_total_trades": final_eval.get("total_trades"),
        "final_eval_turnover_estimate_pct": final_eval.get("turnover_estimate_pct"),
        "final_eval_action_counts": json.dumps(
            final_eval.get("action_counts", {}), sort_keys=True
        ),
        "masked_to_hold_count": hold_diag.get("masked_to_hold_count"),
        "masked_action_rate": hold_diag.get("masked_action_rate"),
        "hold_pct_requested": hold_diag.get("hold_pct_requested"),
        "hold_pct_effective": hold_diag.get("hold_pct_effective"),
    }

    return eval_records, metadata


def aggregate_metric(eval_records: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in eval_records.columns:
        return pd.DataFrame()

    grouped = (
        eval_records.groupby("train_step", dropna=False)[metric]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    grouped["metric"] = metric
    grouped = grouped[["train_step", "metric", "count", "mean", "std", "min", "max"]]
    return grouped


def plot_mean_std(
    aggregate: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    include_zero_line: bool = False,
) -> None:
    rows = aggregate[aggregate["metric"] == metric].copy()
    rows = rows.sort_values("train_step")

    if rows.empty:
        return

    x = rows["train_step"].astype(float).to_numpy()
    y = rows["mean"].astype(float).to_numpy()
    std = rows["std"].fillna(0.0).astype(float).to_numpy()

    plt.figure(figsize=(14, 5))
    plt.plot(x, y, marker="o", linewidth=2, label="mean")

    if np.any(std > 0):
        plt.fill_between(x, y - std, y + std, alpha=0.2, label="±1 std")

    if include_zero_line:
        plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.title(title)
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    log_level = (
        get_environment_variable("STOCK_INVESTMENT_DSS_LOG_LEVEL", default="INFO")
        or "INFO"
    )

    system_logger = setup_system_logger(log_level=log_level)
    system_logger.info(
        "Starting StockInvestmentDSS IQN learning curve multiseed summary."
    )
    system_logger.info("Project root: %s", PROJECT_ROOT)

    run_paths = None

    try:
        recent_limit = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_RECENT_RUN_LIMIT",
            default=100,
        )
        min_runs = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_MIN_RUNS",
            default=1,
        )
        deduplicate_seeds = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_DEDUPLICATE_SEEDS",
                default="true",
            )
            or "true"
        ).strip().lower() in {"1", "true", "yes", "y"}

        seed_filter = parse_seed_list(
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST",
                default="",
            )
        )

        runs_root = PROJECT_ROOT / "outputs" / "runs"
        source_runs = find_learning_curve_runs(
            runs_root=runs_root, recent_limit=recent_limit
        )

        run_paths = create_run_paths("d_iqn_dss_iqn_learning_curve_multiseed_summary")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Runs scanned: %s", recent_limit)
        run_logger.info("Learning curve runs found: %s", len(source_runs))
        run_logger.info("Deduplicate seeds: %s", deduplicate_seeds)
        run_logger.info("Seed filter: %s", seed_filter if seed_filter else "none")

        if len(source_runs) < min_runs:
            raise RuntimeError(
                f"Found {len(source_runs)} IQN learning curve runs, but min_runs={min_runs}."
            )

        eval_frames: list[pd.DataFrame] = []
        metadata_rows: list[dict[str, Any]] = []

        for run_directory in source_runs:
            try:
                eval_records, metadata = load_run(run_directory)
                eval_frames.append(eval_records)
                metadata_rows.append(metadata)
            except Exception as exc:
                run_logger.warning(
                    "Skipping run %s because it could not be loaded: %s",
                    run_directory,
                    exc,
                )

        if not eval_frames:
            raise RuntimeError("No loadable IQN learning curve runs were found.")

        all_eval_records = pd.concat(eval_frames, ignore_index=True)
        run_index = pd.DataFrame(metadata_rows)

        loaded_run_count_before_filtering = int(len(run_index))
        seed_filter_runs_dropped = 0

        if seed_filter and "seed" in run_index.columns:
            run_index["seed"] = pd.to_numeric(run_index["seed"], errors="coerce")
            allowed_seeds = set(seed_filter)
            selected_by_seed = run_index["seed"].isin(allowed_seeds)
            selected_run_ids_for_seed_filter = set(
                run_index.loc[selected_by_seed, "run_id"].astype(str).tolist()
            )
            seed_filter_runs_dropped = int((~selected_by_seed).sum())
            run_index = run_index.loc[selected_by_seed].reset_index(drop=True)
            all_eval_records = all_eval_records[
                all_eval_records["source_run_id"]
                .astype(str)
                .isin(selected_run_ids_for_seed_filter)
            ].reset_index(drop=True)

            if run_index.empty:
                raise RuntimeError(
                    "No IQN learning-curve runs matched "
                    f"STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST={seed_filter}."
                )

        loaded_run_count_before_deduplication = int(len(run_index))
        duplicate_seed_runs_dropped = 0

        if deduplicate_seeds and "seed" in run_index.columns:
            run_index["_run_sort_key"] = run_index["run_id"].astype(str)
            run_index = (
                run_index.sort_values("_run_sort_key")
                .drop_duplicates(subset=["seed"], keep="last")
                .drop(columns=["_run_sort_key"])
                .reset_index(drop=True)
            )
            selected_run_ids = set(run_index["run_id"].astype(str).tolist())
            all_eval_records = all_eval_records[
                all_eval_records["source_run_id"].astype(str).isin(selected_run_ids)
            ].reset_index(drop=True)
            duplicate_seed_runs_dropped = loaded_run_count_before_deduplication - int(
                len(run_index)
            )

        numeric_columns = [
            "final_value",
            "profit_loss",
            "total_return_pct",
            "annualized_sharpe",
            "max_drawdown_pct",
            "annualized_volatility_pct",
            "cvar_pct",
            "total_transaction_cost",
            "total_trades",
            "turnover_estimate_pct",
        ]

        for column in numeric_columns:
            if column in all_eval_records.columns:
                all_eval_records[column] = pd.to_numeric(
                    all_eval_records[column], errors="coerce"
                )

        aggregate_frames = [
            aggregate_metric(all_eval_records, metric)
            for metric in [
                "total_return_pct",
                "annualized_sharpe",
                "max_drawdown_pct",
                "final_value",
                "cvar_pct",
                "turnover_estimate_pct",
            ]
        ]
        aggregate_frames = [frame for frame in aggregate_frames if not frame.empty]
        aggregate_table = (
            pd.concat(aggregate_frames, ignore_index=True)
            if aggregate_frames
            else pd.DataFrame()
        )

        final_records = (
            all_eval_records.sort_values(["source_run_id", "train_step"])
            .groupby("source_run_id", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )

        final_summary = {}
        for metric in [
            "total_return_pct",
            "annualized_sharpe",
            "max_drawdown_pct",
            "final_value",
            "cvar_pct",
        ]:
            if metric in final_records.columns:
                values = pd.to_numeric(final_records[metric], errors="coerce").dropna()
                final_summary[metric] = {
                    "count": int(values.count()),
                    "mean": float(values.mean()) if not values.empty else None,
                    "std": float(values.std()) if len(values) > 1 else 0.0,
                    "min": float(values.min()) if not values.empty else None,
                    "max": float(values.max()) if not values.empty else None,
                }

        all_eval_records_path = (
            run_paths.data_directory / "iqn_learning_curve_multiseed_eval_records.csv"
        )
        run_index_path = (
            run_paths.data_directory / "iqn_learning_curve_multiseed_run_index.csv"
        )
        aggregate_path = (
            run_paths.summary_directory
            / "iqn_learning_curve_multiseed_aggregate_by_step.csv"
        )
        final_records_path = (
            run_paths.summary_directory
            / "iqn_learning_curve_multiseed_final_records.csv"
        )

        all_eval_records.to_csv(all_eval_records_path, index=False)
        run_index.to_csv(run_index_path, index=False)
        aggregate_table.to_csv(aggregate_path, index=False)
        final_records.to_csv(final_records_path, index=False)

        total_return_plot_path = (
            run_paths.summary_directory
            / "iqn_learning_curve_multiseed_total_return_mean_std.png"
        )
        sharpe_plot_path = (
            run_paths.summary_directory
            / "iqn_learning_curve_multiseed_sharpe_mean_std.png"
        )
        drawdown_plot_path = (
            run_paths.summary_directory
            / "iqn_learning_curve_multiseed_max_drawdown_mean_std.png"
        )
        final_value_plot_path = (
            run_paths.summary_directory
            / "iqn_learning_curve_multiseed_final_value_mean_std.png"
        )

        plot_mean_std(
            aggregate=aggregate_table,
            metric="total_return_pct",
            title="IQN Multi-Seed Learning Curve: Evaluation Total Return",
            ylabel="Evaluation total return (%)",
            output_path=total_return_plot_path,
            include_zero_line=True,
        )
        plot_mean_std(
            aggregate=aggregate_table,
            metric="annualized_sharpe",
            title="IQN Multi-Seed Learning Curve: Evaluation Sharpe",
            ylabel="Annualized Sharpe",
            output_path=sharpe_plot_path,
            include_zero_line=True,
        )
        plot_mean_std(
            aggregate=aggregate_table,
            metric="max_drawdown_pct",
            title="IQN Multi-Seed Learning Curve: Maximum Drawdown",
            ylabel="Max drawdown (%)",
            output_path=drawdown_plot_path,
            include_zero_line=True,
        )
        plot_mean_std(
            aggregate=aggregate_table,
            metric="final_value",
            title="IQN Multi-Seed Learning Curve: Final Portfolio Value",
            ylabel="Final portfolio value",
            output_path=final_value_plot_path,
            include_zero_line=False,
        )

        seeds = sorted(
            [
                int(seed)
                for seed in run_index["seed"].dropna().unique().tolist()
                if str(seed).strip() != ""
            ]
        )

        # Hold diagnostic summary aggregated across seeds
        hold_summary: dict[str, Any] = {}
        for _col in ["masked_action_rate", "hold_pct_requested", "hold_pct_effective"]:
            if _col in run_index.columns:
                _vals = pd.to_numeric(run_index[_col], errors="coerce").dropna()
                hold_summary[_col] = {
                    "count": int(_vals.count()),
                    "mean": float(_vals.mean()) if not _vals.empty else None,
                    "std": float(_vals.std()) if len(_vals) > 1 else 0.0,
                    "min": float(_vals.min()) if not _vals.empty else None,
                    "max": float(_vals.max()) if not _vals.empty else None,
                }
        if "masked_to_hold_count" in run_index.columns:
            _vals = pd.to_numeric(run_index["masked_to_hold_count"], errors="coerce").dropna()
            hold_summary["masked_to_hold_count"] = {
                "count": int(_vals.count()),
                "total": int(_vals.sum()) if not _vals.empty else None,
                "mean": float(_vals.mean()) if not _vals.empty else None,
                "max": float(_vals.max()) if not _vals.empty else None,
            }

        summary_path = (
            run_paths.summary_directory / "iqn_learning_curve_multiseed_summary.json"
        )
        summary = {
            "status": "ok",
            "project_name": "StockInvestmentDSS",
            "prototype_name": "D-IQN-DSS",
            "run_id": run_paths.run_id,
            "project_root": str(PROJECT_ROOT),
            "run_directory": str(run_paths.run_directory),
            "source_run_count": int(len(metadata_rows)),
            "loaded_run_count_before_filtering": loaded_run_count_before_filtering,
            "seed_filter": seed_filter,
            "seed_filter_enabled": bool(seed_filter),
            "seed_filter_runs_dropped": int(seed_filter_runs_dropped),
            "loaded_run_count_before_deduplication": loaded_run_count_before_deduplication,
            "duplicate_seed_runs_dropped": int(duplicate_seed_runs_dropped),
            "deduplicate_seeds": bool(deduplicate_seeds),
            "unique_seed_count": int(len(seeds)),
            "seeds": seeds,
            "source_runs": [str(path) for path in source_runs],
            "selected_run_ids": (
                run_index["run_id"].astype(str).tolist()
                if "run_id" in run_index.columns
                else []
            ),
            "final_summary": final_summary,
            "hold_summary": hold_summary,
            "outputs": {
                "all_eval_records_path": str(all_eval_records_path),
                "run_index_path": str(run_index_path),
                "aggregate_by_step_path": str(aggregate_path),
                "final_records_path": str(final_records_path),
                "total_return_plot_path": str(total_return_plot_path),
                "sharpe_plot_path": str(sharpe_plot_path),
                "max_drawdown_plot_path": str(drawdown_plot_path),
                "final_value_plot_path": str(final_value_plot_path),
                "summary_path": str(summary_path),
            },
            "interpretation": (
                "This runner aggregates multiple IQN learning-curve smoke-test runs. "
                "When those runs use different random seeds, the aggregate plots show "
                "mean performance and ±1 standard deviation across seeds. By default, "
                "duplicate runs with the same seed are deduplicated by keeping the latest "
                "run for each seed. If STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST is set, "
                "only those seeds are included, which prevents older exploratory runs from "
                "contaminating a clean multi-seed experiment. With only one unique seed, "
                "the plots are still valid as a single-seed summary but should not be "
                "interpreted as robust evidence."
            ),
            "next_step": (
                "Run the IQN learning-curve smoke test several times with different "
                "STOCK_INVESTMENT_DSS_RANDOM_SEED / IQN seed settings, then rerun this summary."
            ),
        }

        write_json(summary_path, summary)

        run_logger.info("IQN learning curve multiseed summary completed.")
        run_logger.info("Source runs loaded: %s", len(metadata_rows))
        run_logger.info("Unique seeds: %s", seeds)
        run_logger.info("Final summary: %s", final_summary)
        run_logger.info("Wrote aggregate table: %s", aggregate_path)
        run_logger.info("Wrote summary: %s", summary_path)

        system_logger.info(
            "StockInvestmentDSS IQN learning curve multiseed summary completed successfully."
        )
        return 0

    except Exception:
        system_logger.exception(
            "StockInvestmentDSS IQN learning curve multiseed summary failed."
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
