# src/stock_investment_dss/runner/run_finrl_baseline_train_smoke_test.py

from __future__ import annotations

import json

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from stock_investment_dss.data.finrl_data_pipeline import (
    load_or_create_finrl_daily_dataset,
)
from stock_investment_dss.data.point_in_time_split import (
    create_point_in_time_split,
)
from stock_investment_dss.environments.finrl_env_factory import (
    FinRLStockTradingEnvConfig,
    create_finrl_stock_trading_env,
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

MODEL_CLASSES = {
    "A2C": A2C,
    "PPO": PPO,
    "DDPG": DDPG,
    "TD3": TD3,
    "SAC": SAC,
}


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


def create_model(algorithm: str, env):
    normalized = algorithm.strip().upper()

    if normalized not in MODEL_CLASSES:
        available = ", ".join(sorted(MODEL_CLASSES.keys()))
        raise ValueError(
            f"Unsupported FinRL baseline algorithm: {algorithm}. "
            f"Available: {available}"
        )

    model_class = MODEL_CLASSES[normalized]

    # Keep hyperparameters deliberately minimal for smoke testing.
    if normalized == "PPO":
        return model_class(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            n_steps=16,
            batch_size=16,
            seed=42,
        )

    if normalized == "A2C":
        return model_class(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            n_steps=8,
            seed=42,
        )

    return model_class(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        seed=42,
    )


def main() -> int:
    log_level = (
        get_environment_variable(
            "STOCK_INVESTMENT_DSS_LOG_LEVEL",
            default="INFO",
        )
        or "INFO"
    )

    system_logger = setup_system_logger(log_level=log_level)
    system_logger.info("Starting StockInvestmentDSS FinRL baseline train smoke test.")
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

        algorithm = (
            (
                get_environment_variable(
                    "STOCK_INVESTMENT_DSS_FINRL_BASELINE_ALGORITHM",
                    default="A2C",
                )
                or "A2C"
            )
            .strip()
            .upper()
        )

        total_timesteps = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_TIMESTEPS",
            default=500,
        )

        run_paths = create_run_paths("d_iqn_dss_finrl_baseline_train_smoke_test")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Dataset id: %s", dataset_id)
        run_logger.info("Universe id: %s", universe_id)
        run_logger.info("Point in time: %s", point_in_time)
        run_logger.info("Algorithm: %s", algorithm)
        run_logger.info("Total timesteps: %s", total_timesteps)

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

        env, prepared_train_data, env_metadata = create_finrl_stock_trading_env(
            market_data=split_result.train_data,
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

        model = create_model(algorithm=algorithm, env=env)
        model.learn(total_timesteps=total_timesteps)

        model_filename = (
            f"{algorithm.lower()}_{dataset_id}_{split_id}_{total_timesteps}_steps"
        )
        model_path = run_paths.models_directory / model_filename
        model.save(str(model_path))

        prepared_train_data_path = (
            run_paths.data_directory / "finrl_baseline_prepared_train_data.csv"
        )
        prepared_train_data.to_csv(prepared_train_data_path)

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
            "training": {
                "algorithm": algorithm,
                "total_timesteps": total_timesteps,
                "model_path": str(model_path) + ".zip",
                "prepared_train_data_path": str(prepared_train_data_path),
            },
            "next_step": "Backtest saved FinRL baseline model on PIT trade data.",
        }

        summary_path = (
            run_paths.summary_directory / "finrl_baseline_train_smoke_summary.json"
        )
        write_json(summary_path, summary)

        run_logger.info("FinRL baseline model trained successfully.")
        run_logger.info("Algorithm: %s", algorithm)
        run_logger.info("Total timesteps: %s", total_timesteps)
        run_logger.info("Saved model: %s.zip", model_path)
        run_logger.info("Wrote summary: %s", summary_path)

        system_logger.info(
            "StockInvestmentDSS FinRL baseline train smoke test completed successfully."
        )

        return 0

    except Exception:
        system_logger.exception(
            "StockInvestmentDSS FinRL baseline train smoke test failed."
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
