# src/stock_investment_dss/runner/run_discrete_finrl_decision_env_smoke_test.py

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stock_investment_dss.data.finrl_data_pipeline import (
    load_or_create_finrl_daily_dataset,
)
from stock_investment_dss.data.point_in_time_split import (
    create_point_in_time_split,
)
from stock_investment_dss.decision.decision_actions import (
    parse_action_sequence,
)
from stock_investment_dss.decision.investor_risk_profile import (
    InvestorRiskProfile,
)
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
from stock_investment_dss.decision.action_mask import DSSActionMaskGenerator


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
        "Starting StockInvestmentDSS discrete FinRL decision environment smoke test."
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

        # Somehow firefox135 is the best working impersonation
        # more info: docs\debug_yfinance_data_download_ssl_error_issues.txt
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

        raw_decision_actions = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_DISCRETE_DSS_SMOKE_ACTIONS",
                default="HOLD,BUY,HOLD,REBALANCE,SELL,CHANGE_STRATEGY",
            )
            or "HOLD,BUY,HOLD,REBALANCE,SELL,CHANGE_STRATEGY"
        )

        decision_actions = parse_action_sequence(raw_decision_actions)
        risk_profile = create_risk_profile_from_environment()

        run_paths = create_run_paths("d_iqn_dss_discrete_finrl_decision_env_smoke_test")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Dataset id: %s", dataset_id)
        run_logger.info("Universe id: %s", universe_id)
        run_logger.info("Point in time: %s", point_in_time)
        run_logger.info("Trade end date: %s", trade_end_date)
        run_logger.info("Initial amount: %s", initial_amount)
        run_logger.info("hmax: %s", hmax)
        run_logger.info("Risk profile: %s", risk_profile.to_dict())
        run_logger.info("Decision action sequence: %s", raw_decision_actions)

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
        )

        resolver = RiskAwareActionResolver(
            tickers=daily_data_result.tickers,
            hmax=hmax,
            risk_profile=risk_profile,
        )

        action_mask_generator = DSSActionMaskGenerator(
            tickers=daily_data_result.tickers,
            risk_profile=risk_profile,
            allow_change_strategy_without_signal=True,
        )

        env = DiscreteFinRLDecisionEnv(
            finrl_env=finrl_env,
            tickers=daily_data_result.tickers,
            resolver=resolver,
            action_mask_generator=action_mask_generator,
            enforce_action_mask=True,
        )

        reset_result = env.reset()
        initial_observation, reset_info = unpack_reset_result(reset_result)

        initial_state_summary = extract_finrl_state_summary(
            state=initial_observation,
            tickers=daily_data_result.tickers,
        )

        step_records = []
        done = False

        for step_index, decision_action in enumerate(decision_actions):
            if done:
                break

            observation, reward, terminated, truncated, info = env.step(
                int(decision_action)
            )

            done = bool(terminated or truncated)

            decision_record = info.get("decision_record", {})

            step_records.append(
                {
                    "step_index": step_index,
                    "requested_decision_action": decision_record.get(
                        "requested_decision_action_label"
                    ),
                    "effective_decision_action": decision_record.get(
                        "effective_decision_action_label"
                    ),
                    "action_was_masked": decision_record.get("action_was_masked"),
                    "action_mask": decision_record.get("action_mask"),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "done": done,
                    "resolved_action": decision_record.get("resolved_action"),
                    "execution_delta": decision_record.get("execution_delta"),
                    "state_before": decision_record.get("state_before"),
                    "state_after": decision_record.get("state_after"),
                    "finrl_cost": decision_record.get("finrl_cost"),
                    "finrl_trades": decision_record.get("finrl_trades"),
                }
            )

        asset_memory = env.save_asset_memory()
        action_memory = env.save_action_memory()
        decision_memory = env.save_decision_memory()

        prepared_trade_data_path = (
            run_paths.data_directory / "discrete_dss_prepared_trade_data.csv"
        )
        asset_memory_path = run_paths.data_directory / "discrete_dss_asset_memory.csv"
        action_memory_path = run_paths.data_directory / "discrete_dss_action_memory.csv"
        decision_memory_path = (
            run_paths.data_directory / "discrete_dss_decision_memory.json"
        )
        step_records_path = run_paths.data_directory / "discrete_dss_step_records.json"
        step_table_path = run_paths.data_directory / "discrete_dss_step_table.csv"

        prepared_trade_data.to_csv(prepared_trade_data_path)
        asset_memory.to_csv(asset_memory_path, index=False)
        action_memory.to_csv(action_memory_path)
        write_json(decision_memory_path, {"decisions": decision_memory})
        write_json(step_records_path, {"steps": step_records})

        step_table = pd.DataFrame(
            [
                {
                    "step_index": record["step_index"],
                    "requested_decision_action": record.get(
                        "requested_decision_action"
                    ),
                    "effective_decision_action": record.get(
                        "effective_decision_action"
                    ),
                    "action_was_masked": record.get("action_was_masked"),
                    "reward": record["reward"],
                    "done": record["done"],
                    "selected_ticker": (record["resolved_action"] or {}).get(
                        "selected_ticker"
                    ),
                    "requested_shares": (record["resolved_action"] or {}).get(
                        "requested_shares"
                    ),
                    "submitted_shares_estimate": (record["resolved_action"] or {}).get(
                        "submitted_shares_estimate"
                    ),
                    "hmax_limited": (record["resolved_action"] or {}).get(
                        "hmax_limited"
                    ),
                    "requested_cash_value": (record["resolved_action"] or {}).get(
                        "requested_cash_value"
                    ),
                    "submitted_cash_value_estimate": (
                        record["resolved_action"] or {}
                    ).get("submitted_cash_value_estimate"),
                    "continuous_action": (record["resolved_action"] or {}).get(
                        "continuous_action"
                    ),
                    "reason": (record["resolved_action"] or {}).get("reason"),
                    "executed_shares_delta": (record.get("execution_delta") or {}).get(
                        "executed_shares_delta"
                    ),
                    "cash_delta": (record.get("execution_delta") or {}).get(
                        "cash_delta"
                    ),
                    "portfolio_value_delta": (record.get("execution_delta") or {}).get(
                        "portfolio_value_delta"
                    ),
                    "cost_delta": (record.get("execution_delta") or {}).get(
                        "cost_delta"
                    ),
                    "trades_delta": (record.get("execution_delta") or {}).get(
                        "trades_delta"
                    ),
                    "portfolio_value_before": (record["state_before"] or {}).get(
                        "portfolio_value"
                    ),
                    "portfolio_value_after": (record["state_after"] or {}).get(
                        "portfolio_value"
                    ),
                    "cash_before": (record["state_before"] or {}).get("cash"),
                    "cash_after": (record["state_after"] or {}).get("cash"),
                    "finrl_cost": record["finrl_cost"],
                    "finrl_trades": record["finrl_trades"],
                    "action_mask": record.get("action_mask"),
                }
                for record in step_records
            ]
        )

        step_table.to_csv(step_table_path, index=False)

        final_state_summary = (
            step_records[-1]["state_after"] if step_records else initial_state_summary
        )

        initial_portfolio_value = float(initial_state_summary["portfolio_value"])
        final_portfolio_value = float(final_state_summary["portfolio_value"])

        total_return_pct = (
            (
                (final_portfolio_value - initial_portfolio_value)
                / initial_portfolio_value
            )
            * 100.0
            if initial_portfolio_value != 0
            else None
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
            "finrl_environment": finrl_env_metadata,
            "smart_adapter": {
                "name": "DiscreteFinRLDecisionEnv",
                "purpose": (
                    "Expose a discrete DSS decision action space while delegating "
                    "trading mechanics to FinRL StockTradingEnv."
                ),
                "action_space": [
                    "HOLD",
                    "BUY",
                    "SELL",
                    "REBALANCE",
                    "CHANGE_STRATEGY",
                ],
            },
            "risk_profile": risk_profile.to_dict(),
            "smoke_test": {
                "decision_action_sequence_raw": raw_decision_actions,
                "executed_steps": len(step_records),
                "reset_info": reset_info,
                "initial_state_summary": initial_state_summary,
                "final_state_summary": final_state_summary,
                "initial_portfolio_value": initial_portfolio_value,
                "final_portfolio_value": final_portfolio_value,
                "total_return_pct": total_return_pct,
                "finrl_cost": float(env.cost),
                "finrl_trades": int(env.trades),
                "asset_memory_rows": int(len(asset_memory)),
                "action_memory_rows": int(len(action_memory)),
                "decision_memory_rows": int(len(decision_memory)),
                "prepared_trade_data_path": str(prepared_trade_data_path),
                "asset_memory_path": str(asset_memory_path),
                "action_memory_path": str(action_memory_path),
                "decision_memory_path": str(decision_memory_path),
                "step_records_path": str(step_records_path),
                "step_table_path": str(step_table_path),
            },
            "next_step": (
                "Promote the discrete DSS decision layer into an IQN-compatible "
                "environment interface and add action masks."
            ),
        }

        summary_path = (
            run_paths.summary_directory
            / "discrete_finrl_decision_env_smoke_summary.json"
        )
        write_json(summary_path, summary)

        run_logger.info("Discrete FinRL DSS decision environment smoke test completed.")
        run_logger.info("Executed steps: %s", len(step_records))
        run_logger.info("Initial portfolio value: %s", initial_portfolio_value)
        run_logger.info("Final portfolio value: %s", final_portfolio_value)
        run_logger.info("Total return pct: %s", total_return_pct)
        run_logger.info("FinRL env cost: %s", env.cost)
        run_logger.info("FinRL env trades: %s", env.trades)
        run_logger.info("Wrote asset memory: %s", asset_memory_path)
        run_logger.info("Wrote action memory: %s", action_memory_path)
        run_logger.info("Wrote decision memory: %s", decision_memory_path)
        run_logger.info("Wrote step table: %s", step_table_path)
        run_logger.info("Wrote summary: %s", summary_path)

        system_logger.info(
            "StockInvestmentDSS discrete FinRL decision environment smoke test completed successfully."
        )

        return 0

    except Exception:
        system_logger.exception(
            "StockInvestmentDSS discrete FinRL decision environment smoke test failed."
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
