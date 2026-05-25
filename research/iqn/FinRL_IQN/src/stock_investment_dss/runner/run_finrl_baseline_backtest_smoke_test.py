# src/stock_investment_dss/runner/run_finrl_baseline_backtest_smoke_test.py

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
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

MODEL_CLASSES = {
    "A2C": A2C,
    "PPO": PPO,
    "DDPG": DDPG,
    "TD3": TD3,
    "SAC": SAC,
}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def get_int_environment_variable(name: str, default: int) -> int:
    value = get_environment_variable(name, default=str(default))
    return int(value or default)


def get_float_environment_variable(name: str, default: float) -> float:
    value = get_environment_variable(name, default=str(default))
    return float(value or default)


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_model(algorithm: str, model_path: Path):
    normalized = algorithm.strip().upper()

    if normalized not in MODEL_CLASSES:
        available = ", ".join(sorted(MODEL_CLASSES.keys()))
        raise ValueError(
            f"Unsupported FinRL baseline algorithm: {algorithm}. "
            f"Available: {available}"
        )

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    return MODEL_CLASSES[normalized].load(str(model_path))


def compute_basic_asset_metrics(asset_memory: pd.DataFrame) -> dict:
    if asset_memory.empty:
        return {
            "initial_asset": None,
            "final_asset": None,
            "absolute_return": None,
            "total_return_pct": None,
            "max_drawdown_pct": None,
        }

    value_column = None

    for candidate in ["account_value", "total_asset", "asset"]:
        if candidate in asset_memory.columns:
            value_column = candidate
            break

    if value_column is None:
        numeric_columns = asset_memory.select_dtypes(include="number").columns.tolist()
        if not numeric_columns:
            return {
                "value_column": None,
                "initial_asset": None,
                "final_asset": None,
                "absolute_return": None,
                "total_return_pct": None,
                "max_drawdown_pct": None,
            }
        value_column = numeric_columns[-1]

    values = pd.to_numeric(asset_memory[value_column], errors="coerce").dropna()

    if values.empty:
        return {
            "value_column": value_column,
            "initial_asset": None,
            "final_asset": None,
            "absolute_return": None,
            "total_return_pct": None,
            "max_drawdown_pct": None,
        }

    initial_asset = float(values.iloc[0])
    final_asset = float(values.iloc[-1])
    absolute_return = final_asset - initial_asset

    total_return_pct = (
        (absolute_return / initial_asset) * 100.0 if initial_asset != 0 else None
    )

    running_max = values.cummax()
    drawdown = (values / running_max) - 1.0
    max_drawdown_pct = float(drawdown.min() * 100.0)

    return {
        "value_column": value_column,
        "initial_asset": initial_asset,
        "final_asset": final_asset,
        "absolute_return": absolute_return,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
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
        "Starting StockInvestmentDSS FinRL baseline backtest smoke test."
    )
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

        model_path_raw = get_environment_variable(
            "STOCK_INVESTMENT_DSS_FINRL_BASELINE_MODEL_PATH",
            default="",
        )

        if not model_path_raw:
            raise ValueError(
                "Missing STOCK_INVESTMENT_DSS_FINRL_BASELINE_MODEL_PATH in .env"
            )

        model_path = resolve_project_path(model_path_raw)

        run_paths = create_run_paths("d_iqn_dss_finrl_baseline_backtest_smoke_test")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Dataset id: %s", dataset_id)
        run_logger.info("Universe id: %s", universe_id)
        run_logger.info("Point in time: %s", point_in_time)
        run_logger.info("Algorithm: %s", algorithm)
        run_logger.info("Model path: %s", model_path)

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

        model = load_model(algorithm=algorithm, model_path=model_path)

        reset_result = env.reset()
        state, reset_info = unpack_reset_result(reset_result)

        initial_state_summary = extract_finrl_state_summary(
            state=state,
            tickers=daily_data_result.tickers,
        )

        step_records = []
        done = False
        step_index = 0

        while not done:
            action, _model_state = model.predict(state, deterministic=True)
            step_result = env.step(action)
            next_state, reward, done, info = unpack_step_result(step_result)

            state_summary = extract_finrl_state_summary(
                state=next_state,
                tickers=daily_data_result.tickers,
            )

            step_records.append(
                {
                    "step_index": step_index,
                    "action": (
                        action.tolist() if hasattr(action, "tolist") else list(action)
                    ),
                    "reward": float(reward),
                    "done": bool(done),
                    "info": info,
                    "state_summary": state_summary,
                }
            )

            state = next_state
            step_index += 1

        asset_memory = env.save_asset_memory()
        action_memory = env.save_action_memory()
        asset_metrics = compute_basic_asset_metrics(asset_memory)

        prepared_trade_data_path = (
            run_paths.data_directory / "finrl_baseline_prepared_trade_data.csv"
        )
        asset_memory_path = run_paths.data_directory / "finrl_baseline_asset_memory.csv"
        action_memory_path = (
            run_paths.data_directory / "finrl_baseline_action_memory.csv"
        )
        step_records_path = (
            run_paths.data_directory / "finrl_baseline_step_records.json"
        )

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
            "backtest": {
                "algorithm": algorithm,
                "model_path": str(model_path),
                "executed_steps": len(step_records),
                "reset_info": reset_info,
                "initial_state_summary": initial_state_summary,
                "final_state_summary": final_state_summary,
                "env_cost": float(getattr(env, "cost", 0.0)),
                "env_trades": int(getattr(env, "trades", 0)),
                "asset_memory_rows": int(len(asset_memory)),
                "action_memory_rows": int(len(action_memory)),
                "asset_metrics": asset_metrics,
                "prepared_trade_data_path": str(prepared_trade_data_path),
                "asset_memory_path": str(asset_memory_path),
                "action_memory_path": str(action_memory_path),
                "step_records_path": str(step_records_path),
            },
            "next_step": "Promote continuous FinRL baseline train/backtest from smoke test to reusable experiment runner.",
        }

        summary_path = (
            run_paths.summary_directory / "finrl_baseline_backtest_smoke_summary.json"
        )
        write_json(summary_path, summary)

        run_logger.info("FinRL baseline backtest completed successfully.")
        run_logger.info("Algorithm: %s", algorithm)
        run_logger.info("Executed steps: %s", len(step_records))
        run_logger.info("Initial portfolio value: %s", asset_metrics["initial_asset"])
        run_logger.info("Final portfolio value: %s", asset_metrics["final_asset"])
        run_logger.info("Total return pct: %s", asset_metrics["total_return_pct"])
        run_logger.info("Max drawdown pct: %s", asset_metrics["max_drawdown_pct"])
        run_logger.info("FinRL env cost: %s", getattr(env, "cost", 0.0))
        run_logger.info("FinRL env trades: %s", getattr(env, "trades", 0))
        run_logger.info("Wrote asset memory: %s", asset_memory_path)
        run_logger.info("Wrote action memory: %s", action_memory_path)
        run_logger.info("Wrote summary: %s", summary_path)

        system_logger.info(
            "StockInvestmentDSS FinRL baseline backtest smoke test completed successfully."
        )

        return 0

    except Exception:
        system_logger.exception(
            "StockInvestmentDSS FinRL baseline backtest smoke test failed."
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
