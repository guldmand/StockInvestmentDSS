# src/stock_investment_dss/runner/run_iqn_decision_distribution_smoke_test.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from stock_investment_dss.evaluation.portfolio_metrics import write_json
from stock_investment_dss.utilities.config import get_environment_variable
from stock_investment_dss.utilities.logging import (
    setup_run_logger,
    setup_system_logger,
)
from stock_investment_dss.utilities.paths import PROJECT_ROOT, create_run_paths


def find_latest_iqn_train_smoke_run() -> Path:
    runs_root = PROJECT_ROOT / "outputs" / "runs"

    if not runs_root.exists():
        raise FileNotFoundError(f"Run directory does not exist: {runs_root}")

    candidates = [
        path
        for path in runs_root.iterdir()
        if path.is_dir()
        and path.name.endswith("d_iqn_dss_iqn_train_smoke_test")
        and (path / "data" / "iqn_action_distributions.json").exists()
    ]

    if not candidates:
        raise FileNotFoundError(
            "Could not find any IQN train smoke-test run with "
            "data/iqn_action_distributions.json."
        )

    return sorted(candidates, key=lambda path: path.name)[-1]


def resolve_source_run_directory() -> Path:
    source_run_id = get_environment_variable(
        "STOCK_INVESTMENT_DSS_IQN_DISTRIBUTION_SOURCE_RUN_ID",
        default=None,
    )

    source_run_directory = get_environment_variable(
        "STOCK_INVESTMENT_DSS_IQN_DISTRIBUTION_SOURCE_RUN_DIRECTORY",
        default=None,
    )

    if source_run_directory:
        path = Path(source_run_directory)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path

    if source_run_id:
        return PROJECT_ROOT / "outputs" / "runs" / source_run_id

    return find_latest_iqn_train_smoke_run()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def flatten_iqn_action_distributions(distribution_data: dict[str, Any]) -> pd.DataFrame:
    distributions = distribution_data.get("distributions", {})

    rows: list[dict[str, Any]] = []

    selected_action_label = distribution_data.get("selected_action_label")
    selected_action_index = distribution_data.get("selected_action_index")
    num_quantiles = distribution_data.get("num_quantiles")

    for action_label, values in distributions.items():
        mean = values.get("mean")
        cvar10 = values.get("cvar10")
        q10 = values.get("q10")
        q50 = values.get("q50")
        q90 = values.get("q90")

        downside_spread = None
        upside_spread = None
        interquantile_range = None

        if q50 is not None and q10 is not None:
            downside_spread = float(q50) - float(q10)

        if q90 is not None and q50 is not None:
            upside_spread = float(q90) - float(q50)

        if q90 is not None and q10 is not None:
            interquantile_range = float(q90) - float(q10)

        simple_risk_adjusted_score = None

        if mean is not None and cvar10 is not None:
            # Very simple smoke-test score:
            # mean return minus downside-tail penalty.
            # Later this will be replaced by the full risk policy layer.
            simple_risk_adjusted_score = float(mean) + float(cvar10)

        rows.append(
            {
                "action": action_label,
                "action_index": values.get("action_index"),
                "allowed": bool(values.get("allowed")),
                "selected_by_iqn_mean_policy": action_label == selected_action_label,
                "selected_action_index": selected_action_index,
                "num_quantiles": num_quantiles,
                "mean": mean,
                "q10": q10,
                "q25": values.get("q25"),
                "q50": q50,
                "q75": values.get("q75"),
                "q90": q90,
                "cvar10": cvar10,
                "downside_spread_q50_minus_q10": downside_spread,
                "upside_spread_q90_minus_q50": upside_spread,
                "interquantile_range_q90_minus_q10": interquantile_range,
                "simple_risk_adjusted_score": simple_risk_adjusted_score,
            }
        )

    table = pd.DataFrame(rows)

    if table.empty:
        return table

    table["allowed_rank_by_mean"] = None
    table["allowed_rank_by_simple_risk_adjusted_score"] = None

    allowed_mask = table["allowed"] == True

    if allowed_mask.any():
        allowed_by_mean = (
            table[allowed_mask]
            .sort_values("mean", ascending=False, na_position="last")
            .index.tolist()
        )

        for rank, index in enumerate(allowed_by_mean, start=1):
            table.loc[index, "allowed_rank_by_mean"] = rank

        allowed_by_risk_score = (
            table[allowed_mask]
            .sort_values(
                "simple_risk_adjusted_score",
                ascending=False,
                na_position="last",
            )
            .index.tolist()
        )

        for rank, index in enumerate(allowed_by_risk_score, start=1):
            table.loc[index, "allowed_rank_by_simple_risk_adjusted_score"] = rank

    return table.sort_values(
        by=["allowed", "mean"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def build_distribution_summary(
    source_run_directory: Path,
    distribution_data: dict[str, Any],
    distribution_table: pd.DataFrame,
    output_files: dict[str, str],
) -> dict[str, Any]:
    allowed_table = distribution_table[distribution_table["allowed"] == True].copy()

    best_allowed_by_mean = None
    best_allowed_by_simple_risk_adjusted_score = None

    if not allowed_table.empty:
        best_mean_row = allowed_table.sort_values(
            "mean",
            ascending=False,
            na_position="last",
        ).iloc[0]

        best_allowed_by_mean = {
            "action": best_mean_row["action"],
            "mean": best_mean_row["mean"],
            "q10": best_mean_row["q10"],
            "q50": best_mean_row["q50"],
            "q90": best_mean_row["q90"],
            "cvar10": best_mean_row["cvar10"],
        }

        best_score_row = allowed_table.sort_values(
            "simple_risk_adjusted_score",
            ascending=False,
            na_position="last",
        ).iloc[0]

        best_allowed_by_simple_risk_adjusted_score = {
            "action": best_score_row["action"],
            "simple_risk_adjusted_score": best_score_row["simple_risk_adjusted_score"],
            "mean": best_score_row["mean"],
            "q10": best_score_row["q10"],
            "q50": best_score_row["q50"],
            "q90": best_score_row["q90"],
            "cvar10": best_score_row["cvar10"],
        }

    return {
        "status": "ok",
        "source_run_directory": str(source_run_directory),
        "selected_action_from_iqn_mean_policy": {
            "selected_action_index": distribution_data.get("selected_action_index"),
            "selected_action_label": distribution_data.get("selected_action_label"),
        },
        "num_quantiles": distribution_data.get("num_quantiles"),
        "action_count": int(len(distribution_table)),
        "allowed_action_count": (
            int(distribution_table["allowed"].sum())
            if not distribution_table.empty
            else 0
        ),
        "best_allowed_by_mean": best_allowed_by_mean,
        "best_allowed_by_simple_risk_adjusted_score": (
            best_allowed_by_simple_risk_adjusted_score
        ),
        "note": (
            "This is still an IQN smoke-test distribution table. "
            "The simple_risk_adjusted_score is only a temporary diagnostic score. "
            "The full DSS risk policy will later combine IQN downside risk, CVaR, "
            "EDL uncertainty, transaction costs, concentration, and strategy penalties."
        ),
        "output_files": output_files,
        "next_step": (
            "Build an IQN backtest/evaluation loop that loads the trained model, "
            "uses action masks, selects DSS actions from IQN return distributions, "
            "and compares portfolio metrics against the FinRL baseline suite."
        ),
    }


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
        "Starting StockInvestmentDSS IQN decision distribution smoke test."
    )
    system_logger.info("Project root: %s", PROJECT_ROOT)

    run_paths = None

    try:
        source_run_directory = resolve_source_run_directory()
        distribution_path = (
            source_run_directory / "data" / "iqn_action_distributions.json"
        )

        if not distribution_path.exists():
            raise FileNotFoundError(
                f"Missing IQN action distributions file: {distribution_path}"
            )

        run_paths = create_run_paths("d_iqn_dss_iqn_decision_distribution_smoke_test")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Source run directory: %s", source_run_directory)
        run_logger.info("Distribution path: %s", distribution_path)

        distribution_data = read_json(distribution_path)
        distribution_table = flatten_iqn_action_distributions(distribution_data)

        table_path = run_paths.data_directory / "iqn_decision_distribution_table.csv"
        markdown_path = (
            run_paths.summary_directory / "iqn_decision_distribution_table.md"
        )
        summary_path = (
            run_paths.summary_directory / "iqn_decision_distribution_summary.json"
        )
        source_snapshot_path = (
            run_paths.summary_directory
            / "iqn_decision_distribution_source_snapshot.json"
        )

        distribution_table.to_csv(table_path, index=False)
        distribution_table.to_markdown(markdown_path, index=False)

        output_files = {
            "distribution_table_csv": str(table_path),
            "distribution_table_markdown": str(markdown_path),
            "distribution_summary": str(summary_path),
            "source_snapshot": str(source_snapshot_path),
        }

        summary = build_distribution_summary(
            source_run_directory=source_run_directory,
            distribution_data=distribution_data,
            distribution_table=distribution_table,
            output_files=output_files,
        )

        write_json(summary_path, summary)

        source_snapshot = {
            "source_run_directory": str(source_run_directory),
            "source_distribution_path": str(distribution_path),
            "distribution_run_id": run_paths.run_id,
            "selected_action_label": distribution_data.get("selected_action_label"),
            "selected_action_index": distribution_data.get("selected_action_index"),
        }

        write_json(source_snapshot_path, source_snapshot)

        run_logger.info("IQN decision distribution smoke test completed.")
        run_logger.info(
            "Selected action from IQN mean policy: %s",
            distribution_data.get("selected_action_label"),
        )
        run_logger.info(
            "Best allowed action by mean: %s",
            summary.get("best_allowed_by_mean"),
        )
        run_logger.info(
            "Best allowed action by simple risk-adjusted score: %s",
            summary.get("best_allowed_by_simple_risk_adjusted_score"),
        )
        run_logger.info("Wrote distribution table: %s", table_path)
        run_logger.info("Wrote markdown table: %s", markdown_path)
        run_logger.info("Wrote summary: %s", summary_path)

        if not distribution_table.empty:
            run_logger.info("Decision distribution table:")
            for row in distribution_table.to_dict(orient="records"):
                run_logger.info(
                    (
                        "action=%s allowed=%s mean=%s q10=%s q50=%s q90=%s "
                        "cvar10=%s rank_mean=%s rank_risk=%s"
                    ),
                    row.get("action"),
                    row.get("allowed"),
                    row.get("mean"),
                    row.get("q10"),
                    row.get("q50"),
                    row.get("q90"),
                    row.get("cvar10"),
                    row.get("allowed_rank_by_mean"),
                    row.get("allowed_rank_by_simple_risk_adjusted_score"),
                )

        system_logger.info(
            "StockInvestmentDSS IQN decision distribution smoke test completed successfully."
        )

        return 0

    except Exception:
        system_logger.exception(
            "StockInvestmentDSS IQN decision distribution smoke test failed."
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
