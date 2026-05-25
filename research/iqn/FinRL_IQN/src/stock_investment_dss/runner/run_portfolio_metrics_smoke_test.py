# src/stock_investment_dss/runner/run_portfolio_metrics_smoke_test.py

from __future__ import annotations

import json
from pathlib import Path

from stock_investment_dss.evaluation.portfolio_metrics import (
    compute_portfolio_metrics_from_files,
    write_json,
)
from stock_investment_dss.utilities.config import get_environment_variable
from stock_investment_dss.utilities.logging import (
    setup_run_logger,
    setup_system_logger,
)
from stock_investment_dss.utilities.paths import PROJECT_ROOT, create_run_paths


def get_int_environment_variable(name: str, default: int) -> int:
    value = get_environment_variable(name, default=str(default))
    return int(value or default)


def get_float_environment_variable(name: str, default: float) -> float:
    value = get_environment_variable(name, default=str(default))
    return float(value or default)


def find_latest_discrete_dss_run() -> Path:
    runs_root = PROJECT_ROOT / "outputs" / "runs"

    if not runs_root.exists():
        raise FileNotFoundError(f"Run directory does not exist: {runs_root}")

    candidates = [
        path
        for path in runs_root.iterdir()
        if path.is_dir()
        and path.name.endswith("d_iqn_dss_discrete_finrl_decision_env_smoke_test")
        and (path / "data" / "discrete_dss_asset_memory.csv").exists()
    ]

    if not candidates:
        raise FileNotFoundError(
            "Could not find any discrete DSS smoke-test run with "
            "data/discrete_dss_asset_memory.csv."
        )

    return sorted(candidates, key=lambda path: path.name)[-1]


def resolve_source_run_directory() -> Path:
    source_run_id = get_environment_variable(
        "STOCK_INVESTMENT_DSS_PORTFOLIO_METRICS_SOURCE_RUN_ID",
        default=None,
    )

    source_run_directory = get_environment_variable(
        "STOCK_INVESTMENT_DSS_PORTFOLIO_METRICS_SOURCE_RUN_DIRECTORY",
        default=None,
    )

    if source_run_directory:
        path = Path(source_run_directory)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path

    if source_run_id:
        return PROJECT_ROOT / "outputs" / "runs" / source_run_id

    return find_latest_discrete_dss_run()


def validate_source_files(source_run_directory: Path) -> dict[str, Path]:
    data_directory = source_run_directory / "data"

    asset_memory_path = data_directory / "discrete_dss_asset_memory.csv"
    decision_memory_path = data_directory / "discrete_dss_decision_memory.json"
    step_table_path = data_directory / "discrete_dss_step_table.csv"
    action_memory_path = data_directory / "discrete_dss_action_memory.csv"

    if not asset_memory_path.exists():
        raise FileNotFoundError(f"Missing asset memory: {asset_memory_path}")

    if not decision_memory_path.exists():
        raise FileNotFoundError(f"Missing decision memory: {decision_memory_path}")

    if not step_table_path.exists():
        raise FileNotFoundError(f"Missing step table: {step_table_path}")

    return {
        "asset_memory_path": asset_memory_path,
        "decision_memory_path": decision_memory_path,
        "step_table_path": step_table_path,
        "action_memory_path": action_memory_path,
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
    system_logger.info("Starting StockInvestmentDSS portfolio metrics smoke test.")
    system_logger.info("Project root: %s", PROJECT_ROOT)

    run_paths = None

    try:
        source_run_directory = resolve_source_run_directory()
        source_files = validate_source_files(source_run_directory)

        periods_per_year = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_PORTFOLIO_METRICS_PERIODS_PER_YEAR",
            default=252,
        )

        cvar_alpha = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_PORTFOLIO_METRICS_CVAR_ALPHA",
            default=0.10,
        )

        run_paths = create_run_paths("d_iqn_dss_portfolio_metrics_smoke_test")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Source run directory: %s", source_run_directory)
        run_logger.info("Asset memory: %s", source_files["asset_memory_path"])
        run_logger.info("Decision memory: %s", source_files["decision_memory_path"])
        run_logger.info("Step table: %s", source_files["step_table_path"])
        run_logger.info("Periods per year: %s", periods_per_year)
        run_logger.info("CVaR alpha: %s", cvar_alpha)

        metrics_result = compute_portfolio_metrics_from_files(
            asset_memory_path=source_files["asset_memory_path"],
            decision_memory_path=source_files["decision_memory_path"],
            step_table_path=source_files["step_table_path"],
            periods_per_year=periods_per_year,
            cvar_alpha=cvar_alpha,
        )

        metrics_timeseries_path = (
            run_paths.data_directory / "portfolio_metrics_timeseries.csv"
        )

        metrics_summary_path = (
            run_paths.summary_directory / "portfolio_metrics_summary.json"
        )

        source_snapshot_path = (
            run_paths.summary_directory / "portfolio_metrics_source_snapshot.json"
        )

        metrics_result.timeseries.to_csv(metrics_timeseries_path, index=False)

        summary = {
            **metrics_result.summary,
            "source_run_directory": str(source_run_directory),
            "source_files": {
                key: str(value) for key, value in source_files.items() if value.exists()
            },
            "output_files": {
                "portfolio_metrics_timeseries": str(metrics_timeseries_path),
                "portfolio_metrics_summary": str(metrics_summary_path),
            },
            "next_step": (
                "Use portfolio metrics as the common evaluation format for "
                "continuous FinRL baselines, discrete DSS runs, and future IQN/EDL runs."
            ),
        }

        write_json(metrics_summary_path, summary)

        source_snapshot = {
            "source_run_directory": str(source_run_directory),
            "source_files": {
                key: str(value) for key, value in source_files.items() if value.exists()
            },
            "metrics_run_id": run_paths.run_id,
        }

        write_json(source_snapshot_path, source_snapshot)

        run_logger.info("Portfolio metrics smoke test completed.")
        run_logger.info("Initial value: %s", summary["initial_value"])
        run_logger.info("Final value: %s", summary["final_value"])
        run_logger.info("Total return pct: %s", summary["total_return_pct"])
        run_logger.info("Max drawdown pct: %s", summary["max_drawdown_pct"])
        run_logger.info(
            "Annualized volatility pct: %s", summary["annualized_volatility_pct"]
        )
        run_logger.info("Annualized Sharpe: %s", summary["annualized_sharpe"])
        run_logger.info("CVaR pct: %s", summary["cvar_pct"])
        run_logger.info("Total transaction cost: %s", summary["total_transaction_cost"])
        run_logger.info("Total trades: %s", summary["total_trades"])
        run_logger.info("Turnover estimate pct: %s", summary["turnover_estimate_pct"])
        run_logger.info("Wrote metrics timeseries: %s", metrics_timeseries_path)
        run_logger.info("Wrote metrics summary: %s", metrics_summary_path)

        system_logger.info(
            "StockInvestmentDSS portfolio metrics smoke test completed successfully."
        )

        return 0

    except Exception:
        system_logger.exception(
            "StockInvestmentDSS portfolio metrics smoke test failed."
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
