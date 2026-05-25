# src/stock_investment_dss/runner/run_finrl_baseline_suite_smoke_test.py

from __future__ import annotations

from pathlib import Path

from stock_investment_dss.baselines.finrl_baseline_suite import (
    FinRLBaselineSuiteConfig,
    normalize_agents,
    run_finrl_baseline_suite,
)
from stock_investment_dss.data.finrl_data_pipeline import (
    load_or_create_finrl_daily_dataset,
)
from stock_investment_dss.data.point_in_time_split import (
    create_point_in_time_split,
)
from stock_investment_dss.evaluation.portfolio_metrics import write_json
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


def get_int_environment_variable(name: str, default: int) -> int:
    value = get_environment_variable(name, default=str(default))
    return int(value or default)


def get_float_environment_variable(name: str, default: float) -> float:
    value = get_environment_variable(name, default=str(default))
    return float(value or default)


def main() -> int:
    log_level = (
        get_environment_variable(
            "STOCK_INVESTMENT_DSS_LOG_LEVEL",
            default="INFO",
        )
        or "INFO"
    )

    system_logger = setup_system_logger(log_level=log_level)
    system_logger.info("Starting StockInvestmentDSS FinRL baseline suite smoke test.")
    system_logger.info("Project root: %s", PROJECT_ROOT)

    run_paths = None

    try:
        seed_value = int(
            get_environment_variable("STOCK_INVESTMENT_DSS_RANDOM_SEED", default=None)
            or get_environment_variable("STOCK_INVESTMENT_DSS_FINRL_SEED", default=None)
            or get_environment_variable("STOCK_INVESTMENT_DSS_SB3_SEED", default=None)
            or "42"
        )
        set_global_seed(seed_value)
        system_logger.info("Random seed: %d", seed_value)

        universe_id = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE",
                default="demo_2",
            )
            or "demo_2"
        )

        dataset_id = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_DAILY_DATASET_ID",
                default=universe_id,
            )
            or universe_id
        )

        start_date = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_DAILY_DATA_START",
                default="2023-10-01",
            )
            or "2023-10-01"
        )

        end_date = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_DAILY_DATA_END",
                default="2024-02-01",
            )
            or "2024-02-01"
        )

        use_cache = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE",
            default=True,
        )

        allow_download = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD",
            default=True,
        )

        force_download = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD",
            default=False,
        )

        import_file = get_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE",
            default=None,
        )

        chunk_size = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_CHUNK_SIZE",
            default=1,
        )

        sleep_seconds = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_SLEEP_SECONDS",
            default=5.0,
        )

        use_technical_indicators = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_TECHNICAL_INDICATORS",
            default=True,
        )

        use_vix = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_VIX",
            default=False,
        )

        use_turbulence = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_DAILY_DATA_USE_TURBULENCE",
            default=False,
        )

        yfinance_impersonate = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_YFINANCE_IMPERSONATE",
                default="firefox135",
            )
            or "firefox135"
        )

        yfinance_timeout_seconds = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_YFINANCE_TIMEOUT_SECONDS",
            default=30,
        )

        split_id = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_PIT_SPLIT_ID",
                default=f"{dataset_id}_pit",
            )
            or f"{dataset_id}_pit"
        )

        point_in_time = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME",
                default="2024-01-16",
            )
            or "2024-01-16"
        )

        trade_end_date = get_environment_variable(
            "STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE",
            default=end_date,
        )

        min_tickers_per_date = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_PIT_MIN_TICKERS_PER_DATE",
            default=0,
        )

        raw_agents = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_FINRL_BASELINE_AGENTS",
                default="a2c,ddpg,td3,ppo,sac,mvo",
            )
            or "a2c,ddpg,td3,ppo,sac,mvo"
        )

        include_mvo = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_INCLUDE_MVO",
            default=True,
        )

        agents = normalize_agents(
            raw_agents=raw_agents,
            include_mvo=include_mvo,
        )

        total_timesteps = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_TOTAL_TIMESTEPS",
            default=500,
        )

        initial_amount = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_INITIAL_AMOUNT",
            default=1_000_000.0,
        )

        hmax = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_HMAX",
            default=10000,
        )

        buy_cost_pct = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_BUY_COST_PCT",
            default=0.001,
        )

        sell_cost_pct = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_SELL_COST_PCT",
            default=0.001,
        )

        reward_scaling = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_REWARD_SCALING",
            default=0.0001,
        )

        device = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_FINRL_BASELINE_DEVICE",
                default="auto",
            )
            or "auto"
        )

        deterministic_backtest = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_DETERMINISTIC_BACKTEST",
            default=True,
        )

        run_paths = create_run_paths("d_iqn_dss_finrl_baseline_suite_smoke_test")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Dataset id: %s", dataset_id)
        run_logger.info("Universe id: %s", universe_id)
        run_logger.info("Date range: %s -> %s", start_date, end_date)
        run_logger.info("Point in time: %s", point_in_time)
        run_logger.info("Trade end date: %s", trade_end_date)
        run_logger.info("Baseline agents: %s", agents)
        run_logger.info("Total timesteps per RL agent: %s", total_timesteps)
        run_logger.info("Initial amount: %s", initial_amount)
        run_logger.info("hmax: %s", hmax)
        run_logger.info("Device: %s", device)
        run_logger.info("Include MVO: %s", include_mvo)

        daily_data_result = load_or_create_finrl_daily_dataset(
            universe_id=universe_id,
            dataset_id=dataset_id,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
            allow_download=allow_download,
            force_download=force_download,
            import_file=import_file,
            chunk_size=chunk_size,
            sleep_seconds=sleep_seconds,
            use_technical_indicators=use_technical_indicators,
            use_vix=use_vix,
            use_turbulence=use_turbulence,
            yfinance_impersonate=yfinance_impersonate,
            yfinance_timeout_seconds=yfinance_timeout_seconds,
        )

        split_result = create_point_in_time_split(
            data=daily_data_result.processed_data,
            split_id=split_id,
            point_in_time=point_in_time,
            trade_end_date=trade_end_date,
            expected_tickers=daily_data_result.tickers,
            min_tickers_per_date=(
                min_tickers_per_date if min_tickers_per_date > 0 else None
            ),
            source_metadata=daily_data_result.metadata,
        )

        suite_output_directory = run_paths.data_directory / "finrl_baseline_suite"
        suite_model_directory = run_paths.models_directory / "finrl_baseline_suite"

        suite_config = FinRLBaselineSuiteConfig(
            agents=agents,
            total_timesteps=total_timesteps,
            initial_amount=initial_amount,
            hmax=hmax,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            device=device,
            deterministic_backtest=deterministic_backtest,
            include_mvo=include_mvo,
        )

        suite_result = run_finrl_baseline_suite(
            train_data=split_result.train_data,
            trade_data=split_result.trade_data,
            tickers=list(daily_data_result.tickers),
            output_directory=suite_output_directory,
            model_directory=suite_model_directory,
            config=suite_config,
        )

        comparison = suite_result["comparison"]
        comparison_path = suite_result["comparison_path"]

        summary = {
            "status": "ok",
            "project_name": "StockInvestmentDSS",
            "prototype_name": "D-IQN-DSS",
            "run_id": run_paths.run_id,
            "project_root": str(PROJECT_ROOT),
            "run_directory": str(run_paths.run_directory),
            "source_dataset": {
                "dataset_id": daily_data_result.dataset_id,
                "universe_id": daily_data_result.universe_id,
                "tickers": list(daily_data_result.tickers),
                "actual_data_method": daily_data_result.metadata.get(
                    "actual_data_method"
                ),
                "cache_origin_data_method": daily_data_result.metadata.get(
                    "cache_origin_data_method"
                ),
                "fallback_used": daily_data_result.metadata.get("fallback_used"),
            },
            "point_in_time_split": {
                "split_id": split_result.split_id,
                "point_in_time": split_result.point_in_time,
                "train_row_count": int(len(split_result.train_data)),
                "trade_row_count": int(len(split_result.trade_data)),
            },
            "baseline_suite": {
                "agents": agents,
                "rl_agents": [agent for agent in agents if agent != "mvo"],
                "include_mvo": include_mvo,
                "total_timesteps": total_timesteps,
                "initial_amount": initial_amount,
                "hmax": hmax,
                "buy_cost_pct": buy_cost_pct,
                "sell_cost_pct": sell_cost_pct,
                "reward_scaling": reward_scaling,
                "device": device,
                "deterministic_backtest": deterministic_backtest,
                "suite_output_directory": str(suite_output_directory),
                "suite_model_directory": str(suite_model_directory),
                "comparison_path": str(comparison_path),
            },
            "results": suite_result["summary"].get("results", []),
            "next_step": (
                "Use the FinRL baseline suite comparison as the blue baseline layer "
                "against discrete DSS, IQN, and later IQN+EDL variants."
            ),
        }

        summary_path = (
            run_paths.summary_directory / "finrl_baseline_suite_smoke_summary.json"
        )

        write_json(summary_path, summary)

        comparison_snapshot_path = (
            run_paths.summary_directory / "finrl_baseline_suite_comparison_snapshot.csv"
        )
        comparison.to_csv(comparison_snapshot_path, index=False)

        run_logger.info("FinRL baseline suite smoke test completed.")
        run_logger.info("Agents: %s", agents)
        run_logger.info("Comparison path: %s", comparison_path)
        run_logger.info("Comparison snapshot: %s", comparison_snapshot_path)
        run_logger.info("Summary path: %s", summary_path)

        if not comparison.empty:
            run_logger.info("Comparison table:")
            for row in comparison.to_dict(orient="records"):
                run_logger.info(
                    "rank=%s agent=%s final_value=%s total_return_pct=%s max_drawdown_pct=%s sharpe=%s",
                    row.get("rank"),
                    row.get("agent_name"),
                    row.get("final_value"),
                    row.get("total_return_pct"),
                    row.get("max_drawdown_pct"),
                    row.get("annualized_sharpe"),
                )

        system_logger.info(
            "StockInvestmentDSS FinRL baseline suite smoke test completed successfully."
        )

        return 0

    except Exception:
        system_logger.exception(
            "StockInvestmentDSS FinRL baseline suite smoke test failed."
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
