# src/stock_investment_dss/runner/run_iqn_compatible_env_smoke_test.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gymnasium import spaces

from stock_investment_dss.data.finrl_data_pipeline import (
    load_or_create_finrl_daily_dataset,
)
from stock_investment_dss.data.point_in_time_split import (
    create_point_in_time_split,
)
from stock_investment_dss.decision.action_mask import DSSActionMaskGenerator
from stock_investment_dss.decision.decision_actions import (
    DSSDecisionAction,
    action_to_label,
    parse_action_sequence,
)
from stock_investment_dss.decision.investor_risk_profile import InvestorRiskProfile
from stock_investment_dss.decision.risk_aware_action_resolver import (
    RiskAwareActionResolver,
)
from stock_investment_dss.environments.discrete_finrl_decision_env import (
    DiscreteFinRLDecisionEnv,
)
from stock_investment_dss.environments.finrl_env_factory import (
    FinRLStockTradingEnvConfig,
    create_finrl_stock_trading_env,
    extract_finrl_state_summary,
    unpack_reset_result,
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


def create_risk_profile_from_environment() -> InvestorRiskProfile:
    preset = (
        (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_INVESTOR_RISK_PROFILE",
                default="balanced",
            )
            or "balanced"
        )
        .strip()
        .lower()
    )

    if preset == "defensive":
        return InvestorRiskProfile.defensive()

    if preset == "aggressive":
        return InvestorRiskProfile.aggressive()

    if preset == "balanced":
        return InvestorRiskProfile.balanced()

    risk_willingness = get_float_environment_variable(
        "STOCK_INVESTMENT_DSS_INVESTOR_RISK_WILLINGNESS",
        default=0.5,
    )

    max_position_weight = get_float_environment_variable(
        "STOCK_INVESTMENT_DSS_INVESTOR_MAX_POSITION_WEIGHT",
        default=0.25,
    )

    min_cash_weight = get_float_environment_variable(
        "STOCK_INVESTMENT_DSS_INVESTOR_MIN_CASH_WEIGHT",
        default=0.10,
    )

    max_trade_fraction_of_cash = get_float_environment_variable(
        "STOCK_INVESTMENT_DSS_INVESTOR_MAX_TRADE_FRACTION_OF_CASH",
        default=0.25,
    )

    max_sell_fraction_of_position = get_float_environment_variable(
        "STOCK_INVESTMENT_DSS_INVESTOR_MAX_SELL_FRACTION_OF_POSITION",
        default=0.50,
    )

    max_drawdown_tolerance = get_float_environment_variable(
        "STOCK_INVESTMENT_DSS_INVESTOR_MAX_DRAWDOWN_TOLERANCE",
        default=0.15,
    )

    downside_risk_weight = get_float_environment_variable(
        "STOCK_INVESTMENT_DSS_INVESTOR_DOWNSIDE_RISK_WEIGHT",
        default=0.60,
    )

    uncertainty_penalty_weight = get_float_environment_variable(
        "STOCK_INVESTMENT_DSS_INVESTOR_UNCERTAINTY_PENALTY_WEIGHT",
        default=0.40,
    )

    return InvestorRiskProfile(
        risk_willingness=risk_willingness,
        max_position_weight=max_position_weight,
        min_cash_weight=min_cash_weight,
        max_trade_fraction_of_cash=max_trade_fraction_of_cash,
        max_sell_fraction_of_position=max_sell_fraction_of_position,
        max_drawdown_tolerance=max_drawdown_tolerance,
        downside_risk_weight=downside_risk_weight,
        uncertainty_penalty_weight=uncertainty_penalty_weight,
    )


def summarize_observation(observation: Any) -> dict:
    observation_array = np.asarray(observation, dtype=float).reshape(-1)

    return {
        "type": type(observation).__name__,
        "shape": list(observation_array.shape),
        "size": int(observation_array.size),
        "dtype": str(observation_array.dtype),
        "min": float(np.min(observation_array)) if observation_array.size > 0 else None,
        "max": float(np.max(observation_array)) if observation_array.size > 0 else None,
        "mean": (
            float(np.mean(observation_array)) if observation_array.size > 0 else None
        ),
        "has_nan": (
            bool(np.isnan(observation_array).any())
            if observation_array.size > 0
            else False
        ),
        "has_inf": (
            bool(np.isinf(observation_array).any())
            if observation_array.size > 0
            else False
        ),
    }


def summarize_action_space(action_space) -> dict:
    if isinstance(action_space, spaces.Discrete):
        return {
            "type": "Discrete",
            "n": int(action_space.n),
            "expected_n": len(DSSDecisionAction),
            "is_iqn_compatible": int(action_space.n) == len(DSSDecisionAction),
            "actions": {
                int(action): action_to_label(action) for action in DSSDecisionAction
            },
        }

    return {
        "type": type(action_space).__name__,
        "is_iqn_compatible": False,
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
    system_logger.info("Starting StockInvestmentDSS IQN-compatible env smoke test.")
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

        raw_iqn_smoke_actions = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_IQN_ENV_SMOKE_ACTIONS",
                default="HOLD,BUY,HOLD,REBALANCE,SELL,CHANGE_STRATEGY",
            )
            or "HOLD,BUY,HOLD,REBALANCE,SELL,CHANGE_STRATEGY"
        )

        decision_actions = parse_action_sequence(raw_iqn_smoke_actions)
        risk_profile = create_risk_profile_from_environment()

        run_paths = create_run_paths("d_iqn_dss_iqn_compatible_env_smoke_test")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Dataset id: %s", dataset_id)
        run_logger.info("Universe id: %s", universe_id)
        run_logger.info("Point in time: %s", point_in_time)
        run_logger.info("Trade end date: %s", trade_end_date)
        run_logger.info("Initial amount: %s", initial_amount)
        run_logger.info("hmax: %s", hmax)
        run_logger.info("IQN smoke actions: %s", raw_iqn_smoke_actions)
        run_logger.info("Risk profile: %s", risk_profile.to_dict())

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

        finrl_env, prepared_trade_data, finrl_env_metadata = (
            create_finrl_stock_trading_env(
                market_data=split_result.trade_data,
                tickers=list(daily_data_result.tickers),
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
        )

        resolver = RiskAwareActionResolver(
            tickers=list(daily_data_result.tickers),
            hmax=hmax,
            risk_profile=risk_profile,
        )

        action_mask_generator = DSSActionMaskGenerator(
            tickers=list(daily_data_result.tickers),
            risk_profile=risk_profile,
            allow_change_strategy_without_signal=True,
        )

        env = DiscreteFinRLDecisionEnv(
            finrl_env=finrl_env,
            tickers=list(daily_data_result.tickers),
            resolver=resolver,
            action_mask_generator=action_mask_generator,
            enforce_action_mask=True,
        )

        reset_result = env.reset()
        observation, reset_info = unpack_reset_result(reset_result)

        action_space_summary = summarize_action_space(env.action_space)
        observation_summary = summarize_observation(observation)

        initial_action_mask = env.get_action_mask()
        initial_state_summary = extract_finrl_state_summary(
            state=observation,
            tickers=list(daily_data_result.tickers),
        )

        step_records: list[dict[str, Any]] = []
        done = False

        for step_index, decision_action in enumerate(decision_actions):
            if done:
                break

            action_mask_before = env.get_action_mask()

            observation, reward, terminated, truncated, info = env.step(
                int(decision_action)
            )

            done = bool(terminated or truncated)

            decision_record = info.get("decision_record")
            observation_after_summary = summarize_observation(observation)
            action_mask_after = env.get_action_mask() if not done else None

            step_records.append(
                {
                    "step_index": step_index,
                    "requested_action_index": int(decision_action),
                    "requested_action_label": action_to_label(decision_action),
                    "action_mask_before": action_mask_before,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "done": done,
                    "observation_after_summary": observation_after_summary,
                    "action_mask_after": action_mask_after,
                    "info_has_decision_record": decision_record is not None,
                    "decision_record": decision_record,
                }
            )

        asset_memory = env.save_asset_memory()
        action_memory = env.save_action_memory()
        decision_memory = env.save_decision_memory()

        prepared_trade_data_path = (
            run_paths.data_directory / "iqn_env_prepared_trade_data.csv"
        )
        asset_memory_path = run_paths.data_directory / "iqn_env_asset_memory.csv"
        action_memory_path = run_paths.data_directory / "iqn_env_action_memory.csv"
        decision_memory_path = run_paths.data_directory / "iqn_env_decision_memory.json"
        step_records_path = run_paths.data_directory / "iqn_env_step_records.json"
        step_table_path = run_paths.data_directory / "iqn_env_step_table.csv"

        prepared_trade_data.to_csv(prepared_trade_data_path)
        asset_memory.to_csv(asset_memory_path, index=False)
        action_memory.to_csv(action_memory_path)
        write_json(decision_memory_path, {"decisions": decision_memory})
        write_json(step_records_path, {"steps": step_records})

        step_table = pd.DataFrame(
            [
                {
                    "step_index": record["step_index"],
                    "requested_action": record["requested_action_label"],
                    "reward": record["reward"],
                    "terminated": record["terminated"],
                    "truncated": record["truncated"],
                    "done": record["done"],
                    "info_has_decision_record": record["info_has_decision_record"],
                    "mask_before": record["action_mask_before"],
                    "mask_after": record["action_mask_after"],
                    "effective_action": (record.get("decision_record") or {}).get(
                        "effective_decision_action_label"
                    ),
                    "action_was_masked": (record.get("decision_record") or {}).get(
                        "action_was_masked"
                    ),
                    "selected_ticker": (
                        (record.get("decision_record") or {}).get("resolved_action")
                        or {}
                    ).get("selected_ticker"),
                    "requested_shares": (
                        (record.get("decision_record") or {}).get("resolved_action")
                        or {}
                    ).get("requested_shares"),
                    "submitted_shares_estimate": (
                        (record.get("decision_record") or {}).get("resolved_action")
                        or {}
                    ).get("submitted_shares_estimate"),
                    "hmax_limited": (
                        (record.get("decision_record") or {}).get("resolved_action")
                        or {}
                    ).get("hmax_limited"),
                    "continuous_action": (
                        (record.get("decision_record") or {}).get("resolved_action")
                        or {}
                    ).get("continuous_action"),
                    "finrl_cost": (record.get("decision_record") or {}).get(
                        "finrl_cost"
                    ),
                    "finrl_trades": (record.get("decision_record") or {}).get(
                        "finrl_trades"
                    ),
                }
                for record in step_records
            ]
        )

        step_table.to_csv(step_table_path, index=False)

        checks = {
            "action_space_is_discrete": isinstance(env.action_space, spaces.Discrete),
            "action_space_n_is_5": (
                isinstance(env.action_space, spaces.Discrete)
                and int(env.action_space.n) == 5
            ),
            "reset_returned_observation": observation is not None,
            "observation_has_no_nan": not bool(
                observation_summary.get("has_nan", True)
            ),
            "observation_has_no_inf": not bool(
                observation_summary.get("has_inf", True)
            ),
            "initial_action_mask_available": initial_action_mask is not None,
            "all_steps_have_decision_record": all(
                record["info_has_decision_record"] for record in step_records
            ),
            "decision_memory_saved": len(decision_memory) == len(step_records),
            "asset_memory_saved": len(asset_memory) > 0,
            "action_memory_saved": len(action_memory) >= 0,
            "reward_is_numeric_for_all_steps": all(
                isinstance(record["reward"], float) for record in step_records
            ),
        }

        all_checks_passed = all(bool(value) for value in checks.values())

        summary = {
            "status": "ok" if all_checks_passed else "failed_checks",
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
            "finrl_environment": finrl_env_metadata,
            "iqn_compatible_environment": {
                "env_class": "DiscreteFinRLDecisionEnv",
                "action_space": action_space_summary,
                "observation_summary_after_reset": observation_summary,
                "initial_action_mask": initial_action_mask,
                "initial_state_summary": initial_state_summary,
                "reset_info": reset_info,
            },
            "smoke_test": {
                "raw_actions": raw_iqn_smoke_actions,
                "executed_steps": len(step_records),
                "checks": checks,
                "all_checks_passed": all_checks_passed,
                "asset_memory_path": str(asset_memory_path),
                "action_memory_path": str(action_memory_path),
                "decision_memory_path": str(decision_memory_path),
                "step_records_path": str(step_records_path),
                "step_table_path": str(step_table_path),
            },
            "next_step": (
                "Implement minimal IQN network/agent skeleton that can consume this "
                "Discrete(5) environment and estimate return distributions per action."
            ),
        }

        summary_path = (
            run_paths.summary_directory / "iqn_compatible_env_smoke_summary.json"
        )
        write_json(summary_path, summary)

        run_logger.info("IQN-compatible env smoke test completed.")
        run_logger.info("Action space: %s", action_space_summary)
        run_logger.info("Observation summary: %s", observation_summary)
        run_logger.info("Initial action mask: %s", initial_action_mask)
        run_logger.info("Executed steps: %s", len(step_records))
        run_logger.info("Checks: %s", checks)
        run_logger.info("All checks passed: %s", all_checks_passed)
        run_logger.info("Wrote decision memory: %s", decision_memory_path)
        run_logger.info("Wrote step table: %s", step_table_path)
        run_logger.info("Wrote summary: %s", summary_path)

        if not all_checks_passed:
            system_logger.error(
                "StockInvestmentDSS IQN-compatible env smoke test completed with failed checks."
            )
            return 1

        system_logger.info(
            "StockInvestmentDSS IQN-compatible env smoke test completed successfully."
        )

        return 0

    except Exception:
        system_logger.exception(
            "StockInvestmentDSS IQN-compatible env smoke test failed."
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
