# src/stock_investment_dss/runner/run_finrl_environment_smoke_test.py

from __future__ import annotations

import json

import pandas as pd

from stock_investment_dss.data.finrl_data_pipeline import (
    load_or_create_finrl_daily_dataset,
)
from stock_investment_dss.data.point_in_time_split import (
    create_point_in_time_split,
)
from stock_investment_dss.environments.finrl_env_factory import (
    FinRLStockTradingEnvConfig,
    create_finrl_stock_trading_env,
    extract_finrl_state_summary,
    make_zero_action,
    unpack_reset_result,
    unpack_step_result,
)
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
        json.dump(data, file, indent=2, default=str)


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
    system_logger.info("Starting StockInvestmentDSS FinRL environment smoke test.")
    system_logger.info("Project root: %s", PROJECT_ROOT)

    run_paths = None

    try:
        set_global_seed(42)

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
                default="2024-01-01",
            )
            or "2024-01-01"
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
            default=False,
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

        initial_amount = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_INITIAL_AMOUNT",
            default=1_000_000,
        )

        hmax = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_HMAX",
            default=100,
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

        smoke_steps = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_ENV_SMOKE_STEPS",
            default=5,
        )

        run_paths = create_run_paths("d_iqn_dss_finrl_environment_smoke_test")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Dataset id: %s", dataset_id)
        run_logger.info("Universe id: %s", universe_id)
        run_logger.info("Point in time: %s", point_in_time)
        run_logger.info("Trade end date: %s", trade_end_date)
        run_logger.info("Initial amount: %s", initial_amount)
        run_logger.info("hmax: %s", hmax)
        run_logger.info("Smoke steps: %s", smoke_steps)

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

        env, prepared_trade_data, env_metadata = create_finrl_stock_trading_env(
            market_data=split_result.trade_data,
            tickers=daily_data_result.tickers,
            config=FinRLStockTradingEnvConfig(
                initial_amount=initial_amount,
                hmax=hmax,
                buy_cost_pct=buy_cost_pct,
                sell_cost_pct=sell_cost_pct,
                reward_scaling=reward_scaling,
                print_verbosity=10_000,
            ),
            technical_indicators=[],
        )

        reset_result = env.reset()
        initial_state, reset_info = unpack_reset_result(reset_result)

        initial_state_summary = extract_finrl_state_summary(
            state=initial_state,
            tickers=daily_data_result.tickers,
        )

        step_records = []
        done = False

        for step_index in range(smoke_steps):
            if done:
                break

            action = make_zero_action(stock_dim=env_metadata["stock_dim"])
            step_result = env.step(action)
            state, reward, done, info = unpack_step_result(step_result)

            state_summary = extract_finrl_state_summary(
                state=state,
                tickers=daily_data_result.tickers,
            )

            step_records.append(
                {
                    "step_index": step_index,
                    "action": action.tolist(),
                    "reward": float(reward),
                    "done": bool(done),
                    "info": info,
                    "state_summary": state_summary,
                }
            )

        asset_memory = env.save_asset_memory()
        action_memory = env.save_action_memory()

        prepared_trade_data_path = (
            run_paths.data_directory / "finrl_env_prepared_trade_data.csv"
        )
        asset_memory_path = run_paths.data_directory / "finrl_asset_memory.csv"
        action_memory_path = run_paths.data_directory / "finrl_action_memory.csv"
        step_records_path = run_paths.data_directory / "finrl_env_step_records.json"

        prepared_trade_data.to_csv(prepared_trade_data_path)
        asset_memory.to_csv(asset_memory_path, index=False)
        action_memory.to_csv(action_memory_path)
        write_json(step_records_path, {"steps": step_records})

        final_state_summary = (
            step_records[-1]["state_summary"] if step_records else initial_state_summary
        )

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
            "finrl_environment": env_metadata,
            "reset_info": reset_info,
            "smoke_test": {
                "requested_steps": smoke_steps,
                "executed_steps": len(step_records),
                "initial_state_summary": initial_state_summary,
                "final_state_summary": final_state_summary,
                "env_cost": float(getattr(env, "cost", 0.0)),
                "env_trades": int(getattr(env, "trades", 0)),
                "asset_memory_rows": int(len(asset_memory)),
                "action_memory_rows": int(len(action_memory)),
                "prepared_trade_data_path": str(prepared_trade_data_path),
                "asset_memory_path": str(asset_memory_path),
                "action_memory_path": str(action_memory_path),
                "step_records_path": str(step_records_path),
            },
            "next_step": "Build continuous FinRL-compatible baseline track and discrete IQN-compatible adapter.",
        }

        summary_path = (
            run_paths.summary_directory / "finrl_environment_smoke_summary.json"
        )
        write_json(summary_path, summary)

        run_logger.info("FinRL environment instantiated successfully.")
        run_logger.info("Executed environment steps: %s", len(step_records))
        run_logger.info(
            "Initial portfolio value: %s",
            initial_state_summary["portfolio_value"],
        )
        run_logger.info(
            "Final portfolio value: %s",
            final_state_summary["portfolio_value"],
        )
        run_logger.info("FinRL env cost: %s", getattr(env, "cost", 0.0))
        run_logger.info("FinRL env trades: %s", getattr(env, "trades", 0))
        run_logger.info("Wrote asset memory: %s", asset_memory_path)
        run_logger.info("Wrote action memory: %s", action_memory_path)
        run_logger.info("Wrote environment summary: %s", summary_path)

        system_logger.info(
            "StockInvestmentDSS FinRL environment smoke test completed successfully."
        )

        return 0

    except Exception:
        system_logger.exception(
            "StockInvestmentDSS FinRL environment smoke test failed."
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
