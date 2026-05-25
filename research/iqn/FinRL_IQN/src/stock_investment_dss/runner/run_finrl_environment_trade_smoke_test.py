# src/stock_investment_dss/runner/run_finrl_environment_trade_smoke_test.py

from __future__ import annotations

import json

import numpy as np
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


def parse_continuous_action_sequence(
    raw_value: str,
    stock_dim: int,
) -> list[np.ndarray]:
    """
    Parse continuous FinRL action vectors.

    Example for two tickers:
        1,0;0,1;0,0;-1,0;0,-1

    Meaning:
        [ 1,  0] -> buy first ticker
        [ 0,  1] -> buy second ticker
        [ 0,  0] -> hold
        [-1,  0] -> sell first ticker
        [ 0, -1] -> sell second ticker

    FinRL StockTradingEnv typically scales actions internally by hmax.
    """
    actions: list[np.ndarray] = []

    for raw_action in raw_value.split(";"):
        raw_action = raw_action.strip()

        if not raw_action:
            continue

        values = [
            float(value.strip()) for value in raw_action.split(",") if value.strip()
        ]

        if len(values) != stock_dim:
            raise ValueError(
                f"Action vector has length {len(values)}, "
                f"but stock_dim={stock_dim}. Raw action: {raw_action}"
            )

        actions.append(np.array(values, dtype=float))

    if not actions:
        raise ValueError("No valid continuous actions were provided.")

    return actions


def main() -> int:
    log_level = (
        get_environment_variable(
            "STOCK_INVESTMENT_DSS_LOG_LEVEL",
            default="INFO",
        )
        or "INFO"
    )

    system_logger = setup_system_logger(log_level=log_level)
    system_logger.info("Starting StockInvestmentDSS FinRL trade smoke test.")
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

        action_sequence_raw = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_FINRL_ENV_TRADE_ACTIONS",
                default="1,0;0,1;0,0;-1,0;0,-1",
            )
            or "1,0;0,1;0,0;-1,0;0,-1"
        )

        run_paths = create_run_paths("d_iqn_dss_finrl_environment_trade_smoke_test")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Dataset id: %s", dataset_id)
        run_logger.info("Universe id: %s", universe_id)
        run_logger.info("Point in time: %s", point_in_time)
        run_logger.info("Trade end date: %s", trade_end_date)
        run_logger.info("Initial amount: %s", initial_amount)
        run_logger.info("hmax: %s", hmax)
        run_logger.info("Action sequence: %s", action_sequence_raw)

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
            technical_indicators=None,
        )

        actions = parse_continuous_action_sequence(
            raw_value=action_sequence_raw,
            stock_dim=env_metadata["stock_dim"],
        )

        reset_result = env.reset()
        initial_state, reset_info = unpack_reset_result(reset_result)

        initial_state_summary = extract_finrl_state_summary(
            state=initial_state,
            tickers=daily_data_result.tickers,
        )

        step_records = []
        done = False

        previous_state_summary = initial_state_summary

        for step_index, action in enumerate(actions):
            if done:
                break

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
                    "state_before": previous_state_summary,
                    "state_after": state_summary,
                    "portfolio_value_change": (
                        state_summary["portfolio_value"]
                        - previous_state_summary["portfolio_value"]
                    ),
                    "cash_change": (
                        state_summary["cash"] - previous_state_summary["cash"]
                    ),
                    "holdings_change": {
                        ticker: (
                            state_summary["holdings"][ticker]
                            - previous_state_summary["holdings"][ticker]
                        )
                        for ticker in daily_data_result.tickers
                    },
                }
            )

            previous_state_summary = state_summary

        asset_memory = env.save_asset_memory()
        action_memory = env.save_action_memory()

        prepared_trade_data_path = (
            run_paths.data_directory / "finrl_trade_env_prepared_trade_data.csv"
        )
        asset_memory_path = run_paths.data_directory / "finrl_trade_asset_memory.csv"
        action_memory_path = run_paths.data_directory / "finrl_trade_action_memory.csv"
        step_records_path = run_paths.data_directory / "finrl_trade_step_records.json"

        prepared_trade_data.to_csv(prepared_trade_data_path)
        asset_memory.to_csv(asset_memory_path, index=False)
        action_memory.to_csv(action_memory_path)
        write_json(step_records_path, {"steps": step_records})

        final_state_summary = (
            step_records[-1]["state_after"] if step_records else initial_state_summary
        )

        step_table = pd.DataFrame(
            [
                {
                    "step_index": record["step_index"],
                    "action": record["action"],
                    "reward": record["reward"],
                    "done": record["done"],
                    "portfolio_value_before": record["state_before"]["portfolio_value"],
                    "portfolio_value_after": record["state_after"]["portfolio_value"],
                    "portfolio_value_change": record["portfolio_value_change"],
                    "cash_before": record["state_before"]["cash"],
                    "cash_after": record["state_after"]["cash"],
                    "cash_change": record["cash_change"],
                    "holdings_before": record["state_before"]["holdings"],
                    "holdings_after": record["state_after"]["holdings"],
                    "holdings_change": record["holdings_change"],
                }
                for record in step_records
            ]
        )

        step_table_path = run_paths.data_directory / "finrl_trade_step_table.csv"
        step_table.to_csv(step_table_path, index=False)

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
            "trade_smoke_test": {
                "action_sequence_raw": action_sequence_raw,
                "action_sequence": [action.tolist() for action in actions],
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
                "step_table_path": str(step_table_path),
            },
            "next_step": "Build continuous FinRL-compatible baseline track.",
        }

        summary_path = (
            run_paths.summary_directory / "finrl_environment_trade_smoke_summary.json"
        )
        write_json(summary_path, summary)

        run_logger.info("FinRL trade smoke test completed.")
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
        run_logger.info("Wrote trade asset memory: %s", asset_memory_path)
        run_logger.info("Wrote trade action memory: %s", action_memory_path)
        run_logger.info("Wrote trade step table: %s", step_table_path)
        run_logger.info("Wrote trade environment summary: %s", summary_path)

        system_logger.info(
            "StockInvestmentDSS FinRL environment trade smoke test completed successfully."
        )

        return 0

    except Exception:
        system_logger.exception(
            "StockInvestmentDSS FinRL environment trade smoke test failed."
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
