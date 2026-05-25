# src/stock_investment_dss/runner/run_smoke_test.py

from __future__ import annotations

import json

from stock_investment_dss.evaluation.evaluation_config import get_evaluation_config
from stock_investment_dss.strategies.predefined import get_strategy
from stock_investment_dss.utilities.config import (
    get_boolean_environment_variable,
    get_environment_variable,
)
from stock_investment_dss.utilities.logging import (
    setup_run_logger,
    setup_system_logger,
)
from stock_investment_dss.utilities.paths import PROJECT_ROOT, create_run_paths
from stock_investment_dss.utilities.seed import set_global_seed


def write_json(path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def main() -> int:
    log_level = (
        get_environment_variable(
            "STOCK_INVESTMENT_DSS_LOG_LEVEL",
            default="INFO",
        )
        or "INFO"
    )

    system_logger = setup_system_logger(log_level=log_level)
    system_logger.info("Starting StockInvestmentDSS smoke test.")
    system_logger.info("Project root: %s", PROJECT_ROOT)

    run_paths = None

    try:
        set_global_seed(42)

        default_strategy_id = get_environment_variable(
            "STOCK_INVESTMENT_DSS_DEFAULT_STRATEGY",
            default="balanced",
        )

        default_evaluation_id = get_environment_variable(
            "STOCK_INVESTMENT_DSS_DEFAULT_EVALUATION",
            default="default",
        )

        use_wandb = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_USE_WANDB",
            default=False,
        )

        run_paths = create_run_paths("d_iqn_dss_smoke_test")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Weights & Biases enabled: %s", use_wandb)

        strategy = get_strategy(default_strategy_id)
        evaluation_config = get_evaluation_config(default_evaluation_id)

        strategy_snapshot_path = run_paths.config_directory / "strategy.json"
        evaluation_snapshot_path = run_paths.config_directory / "evaluation.json"

        write_json(strategy_snapshot_path, strategy.to_dict())
        write_json(evaluation_snapshot_path, evaluation_config.to_dict())

        run_logger.info(
            "Loaded strategy: %s (%s)",
            strategy.strategy_id,
            strategy.display_name,
        )
        run_logger.info("Wrote strategy snapshot: %s", strategy_snapshot_path)

        run_logger.info(
            "Loaded evaluation config: %s",
            evaluation_config.evaluation_id,
        )
        run_logger.info("Wrote evaluation snapshot: %s", evaluation_snapshot_path)

        smoke_summary = {
            "status": "ok",
            "project_name": "StockInvestmentDSS",
            "prototype_name": "D-IQN-DSS",
            "prototype_full_name": (
                "Discrete Implicit Quantile Network for "
                "Stock Investment Decision Support"
            ),
            "run_id": run_paths.run_id,
            "project_root": str(PROJECT_ROOT),
            "run_directory": str(run_paths.run_directory),
            "wandb_enabled": use_wandb,
            "strategy": {
                "strategy_id": strategy.strategy_id,
                "display_name": strategy.display_name,
                "risk_profile": strategy.risk_profile,
                "objective": strategy.objective,
                "allowed_actions": list(strategy.allowed_actions),
            },
            "evaluation": {
                "evaluation_id": evaluation_config.evaluation_id,
                "primary_metrics": list(evaluation_config.primary_metrics),
                "portfolio_metrics": list(evaluation_config.portfolio_metrics),
                "distributional_metrics": list(
                    evaluation_config.distributional_metrics
                ),
                "uncertainty_metrics": list(evaluation_config.uncertainty_metrics),
                "decision_support_metrics": list(
                    evaluation_config.decision_support_metrics
                ),
            },
            "next_step": "Build FinRL data pipeline.",
        }

        summary_path = run_paths.summary_directory / "smoke_test_summary.json"
        write_json(summary_path, smoke_summary)

        run_logger.info("Wrote smoke test summary: %s", summary_path)
        system_logger.info("StockInvestmentDSS smoke test completed successfully.")

        return 0

    except Exception:
        system_logger.exception("StockInvestmentDSS smoke test failed.")

        if run_paths is not None:
            try:
                run_logger = setup_run_logger(run_paths, log_level=log_level)
                run_logger.exception("Run failed.")
            except Exception:
                pass

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
