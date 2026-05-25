# src/stock_investment_dss/runner/run_iqn_train_smoke_test.py

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from stock_investment_dss.data.finrl_data_pipeline import (
    load_or_create_finrl_daily_dataset,
)
from stock_investment_dss.data.point_in_time_split import (
    create_point_in_time_split,
)
from stock_investment_dss.decision.action_mask import DSSActionMaskGenerator
from stock_investment_dss.decision.decision_actions import action_to_label
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
    unpack_reset_result,
)
from stock_investment_dss.rl.agents.iqn_agent import IQNAgent
from stock_investment_dss.rl.config.iqn_config import IQNConfig, build_iqn_config
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

    return InvestorRiskProfile.balanced()


def create_iqn_config_from_environment() -> IQNConfig:
    """
    Builds IQNConfig in a robust way.

    This supports the current V1-derived IQNConfig, even if its dataclass
    signature changes slightly, because we only pass supported fields.
    """

    requested_device = (
        get_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_DEVICE",
            default="cuda" if torch.cuda.is_available() else "cpu",
        )
        or "cpu"
    )

    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"

    candidate_values = {
        "env_name": "D-IQN-DSS-FinRL-DiscreteDecisionEnv",
        "device": torch.device(requested_device),
        "hidden_dim": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_HIDDEN_DIM",
            default=128,
        ),
        "cosine_embedding_dim": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_COSINE_EMBEDDING_DIM",
            default=64,
        ),
        "lr": get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_RATE",
            default=1e-4,
        ),
        "gamma": get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_GAMMA",
            default=0.99,
        ),
        "kappa": get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_KAPPA",
            default=1.0,
        ),
        "batch_size": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_BATCH_SIZE",
            default=64,
        ),
        "replay_capacity": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_REPLAY_CAPACITY",
            default=100_000,
        ),
        "num_tau_samples": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_NUM_TAU_SAMPLES",
            default=32,
        ),
        "num_tau_prime_samples": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_NUM_TAU_PRIME_SAMPLES",
            default=32,
        ),
        "num_action_quantiles": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_NUM_ACTION_QUANTILES",
            default=32,
        ),
        "epsilon_start": get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_EPSILON_START",
            default=1.0,
        ),
        "epsilon_final": get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_EPSILON_FINAL",
            default=0.05,
        ),
        "epsilon_decay_steps": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_EPSILON_DECAY_STEPS",
            default=2_500,
        ),
        "target_update_interval": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_TARGET_UPDATE_INTERVAL",
            default=500,
        ),
        "learning_starts": get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_STARTS",
            default=1_000,
        ),
    }

    config = build_iqn_config()

    # Runner-specific values are applied after the base config has been built.
    # This preserves StockDSS defaults, optional JSON/env presets, and still
    # lets this smoke-test runner define its concrete experiment settings.
    for key, value in candidate_values.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def config_to_dict(config: IQNConfig) -> dict[str, Any]:
    data: dict[str, Any] = {}

    for key, value in vars(config).items():
        if isinstance(value, torch.device):
            data[key] = str(value)
        else:
            data[key] = value

    return data


def get_buffer_length(agent: IQNAgent) -> int:
    try:
        return len(agent.replay_buffer)
    except TypeError:
        pass

    for attr in ["size", "length", "count"]:
        if hasattr(agent.replay_buffer, attr):
            value = getattr(agent.replay_buffer, attr)
            return int(value() if callable(value) else value)

    if hasattr(agent.replay_buffer, "buffer"):
        return len(agent.replay_buffer.buffer)

    if hasattr(agent.replay_buffer, "memory"):
        return len(agent.replay_buffer.memory)

    return 0


def add_transition_to_replay_buffer(
    agent: IQNAgent,
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray,
    done: bool,
) -> None:
    """
    Robust adapter for V1-derived replay buffers.

    Supports common names:
    - add(...)
    - push(...)
    - store(...)
    - append(...)
    """

    transition = (state, action, reward, next_state, done)

    for method_name in ["add", "push", "store", "append"]:
        method = getattr(agent.replay_buffer, method_name, None)

        if method is None:
            continue

        try:
            method(state, action, reward, next_state, done)
            return
        except TypeError:
            pass

        try:
            method(*transition)
            return
        except TypeError:
            pass

        try:
            method(transition)
            return
        except TypeError:
            pass

    raise AttributeError(
        "Could not add transition to replay buffer. "
        "Expected replay buffer to expose add/push/store/append."
    )


def create_training_environment(
    tickers: list[str],
    training_data: pd.DataFrame,
    initial_amount: float,
    hmax: int,
    buy_cost_pct: float,
    sell_cost_pct: float,
    reward_scaling: float,
    risk_profile: InvestorRiskProfile,
):
    finrl_env, prepared_training_data, finrl_env_metadata = (
        create_finrl_stock_trading_env(
            market_data=training_data,
            tickers=tickers,
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
        tickers=tickers,
        hmax=hmax,
        risk_profile=risk_profile,
    )

    action_mask_generator = DSSActionMaskGenerator(
        tickers=tickers,
        risk_profile=risk_profile,
        allow_change_strategy_without_signal=True,
    )

    env = DiscreteFinRLDecisionEnv(
        finrl_env=finrl_env,
        tickers=tickers,
        resolver=resolver,
        action_mask_generator=action_mask_generator,
        enforce_action_mask=True,
    )

    return env, prepared_training_data, finrl_env_metadata


def main() -> int:
    log_level = (
        get_environment_variable(
            "STOCK_INVESTMENT_DSS_LOG_LEVEL",
            default="INFO",
        )
        or "INFO"
    )

    system_logger = setup_system_logger(log_level=log_level)
    system_logger.info("Starting StockInvestmentDSS IQN train smoke test.")
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

        split_id = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_PIT_SPLIT_ID",
                default=f"{dataset_id}_pit",
            )
            or f"{dataset_id}_pit"
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

        total_steps = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_TRAIN_SMOKE_STEPS",
            default=5_000,
        )

        learning_starts = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_LEARNING_STARTS",
            default=1_000,
        )

        disable_change_strategy = get_boolean_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_DISABLE_CHANGE_STRATEGY",
            default=True,
        )

        eval_quantiles = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_EVAL_QUANTILES",
            default=128,
        )

        risk_profile = create_risk_profile_from_environment()
        iqn_config = create_iqn_config_from_environment()

        # Keep config metadata aligned with this V2 environment.
        # Some V1-derived IQNConfig versions expose these fields; guard with hasattr
        # so this runner stays compatible with small config changes.
        if hasattr(iqn_config, "env_name"):
            iqn_config.env_name = "D-IQN-DSS-FinRL-DiscreteDecisionEnv"
        if hasattr(iqn_config, "total_steps"):
            iqn_config.total_steps = total_steps
        if hasattr(iqn_config, "learning_starts"):
            iqn_config.learning_starts = learning_starts

        run_paths = create_run_paths("d_iqn_dss_iqn_train_smoke_test")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Dataset id: %s", dataset_id)
        run_logger.info("Universe id: %s", universe_id)
        run_logger.info("Point in time: %s", point_in_time)
        run_logger.info("Trade end date: %s", trade_end_date)
        run_logger.info("IQN train smoke steps: %s", total_steps)
        run_logger.info("IQN learning starts: %s", learning_starts)
        run_logger.info(
            "Disable CHANGE_STRATEGY during IQN train: %s", disable_change_strategy
        )
        run_logger.info("IQN config: %s", config_to_dict(iqn_config))
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

        env, prepared_training_data, finrl_env_metadata = create_training_environment(
            tickers=list(daily_data_result.tickers),
            training_data=split_result.train_data,
            initial_amount=initial_amount,
            hmax=hmax,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            risk_profile=risk_profile,
        )

        reset_result = env.reset()
        observation, reset_info = unpack_reset_result(reset_result)
        state = np.asarray(observation, dtype=np.float32).reshape(-1)

        state_dim = int(state.shape[0])
        action_dim = int(env.action_space.n)

        agent = IQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=iqn_config,
        )

        training_records: list[dict[str, Any]] = []
        losses: list[float] = []
        episode_count = 1
        done = False

        for step in range(total_steps):
            if done:
                reset_result = env.reset()
                observation, reset_info = unpack_reset_result(reset_result)
                state = np.asarray(observation, dtype=np.float32).reshape(-1)
                done = False
                episode_count += 1

            action_mask_info = env.get_action_mask()
            action_mask = action_mask_info.get("mask_vector")

            # CHANGE_STRATEGY is intentionally disabled for the current IQN v1
            # training/backtest loop until real strategy-switch execution logic exists.
            # It must be reactivated later when CHANGE_STRATEGY has real semantics.
            if (
                disable_change_strategy
                and action_mask is not None
                and len(action_mask) >= 5
            ):
                action_mask[4] = 0

            action = agent.select_action(
                state=state,
                step=step,
                eval_mode=False,
                action_mask=action_mask,
            )

            next_observation, reward, terminated, truncated, info = env.step(action)

            next_state = np.asarray(next_observation, dtype=np.float32).reshape(-1)
            done = bool(terminated or truncated)

            add_transition_to_replay_buffer(
                agent=agent,
                state=state,
                action=int(action),
                reward=float(reward),
                next_state=next_state,
                done=done,
            )

            buffer_size = get_buffer_length(agent)
            loss_value = None

            if step >= learning_starts and buffer_size >= iqn_config.batch_size:
                loss_value = agent.learn()
                losses.append(loss_value)

            target_update_interval = getattr(
                iqn_config,
                "target_update_interval",
                25,
            )

            if step > 0 and step % int(target_update_interval) == 0:
                agent.update_target_network()

            decision_record = info.get("decision_record", {})

            training_records.append(
                {
                    "step": step,
                    "episode": episode_count,
                    "state_dim": state_dim,
                    "action": int(action),
                    "action_label": action_to_label(action),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "done": done,
                    "buffer_size": buffer_size,
                    "loss": loss_value,
                    "epsilon": agent.epsilon(step),
                    "action_mask": action_mask,
                    "effective_action": decision_record.get(
                        "effective_decision_action_label"
                    ),
                    "action_was_masked": decision_record.get("action_was_masked"),
                    "selected_ticker": (
                        decision_record.get("resolved_action") or {}
                    ).get("selected_ticker"),
                    "finrl_cost": decision_record.get("finrl_cost"),
                    "finrl_trades": decision_record.get("finrl_trades"),
                }
            )

            state = next_state

        final_action_mask_info = env.get_action_mask()
        final_action_mask = final_action_mask_info.get("mask_vector")

        if (
            disable_change_strategy
            and final_action_mask is not None
            and len(final_action_mask) >= 5
        ):
            final_action_mask[4] = 0

        action_distributions = agent.estimate_action_distributions(
            state=state,
            num_quantiles=eval_quantiles,
            action_mask=final_action_mask,
        )

        model_path = run_paths.models_directory / "iqn_train_smoke_model.pt"
        training_records_path = run_paths.data_directory / "iqn_training_records.csv"
        action_distributions_path = (
            run_paths.data_directory / "iqn_action_distributions.json"
        )
        decision_memory_path = (
            run_paths.data_directory / "iqn_train_decision_memory.json"
        )
        asset_memory_path = run_paths.data_directory / "iqn_train_asset_memory.csv"
        action_memory_path = run_paths.data_directory / "iqn_train_action_memory.csv"
        prepared_training_data_path = (
            run_paths.data_directory / "iqn_train_prepared_train_data.csv"
        )

        agent.save(str(model_path))
        pd.DataFrame(training_records).to_csv(training_records_path, index=False)
        write_json(action_distributions_path, action_distributions)
        write_json(decision_memory_path, {"decisions": env.save_decision_memory()})

        env.save_asset_memory().to_csv(asset_memory_path, index=False)
        env.save_action_memory().to_csv(action_memory_path)
        prepared_training_data.to_csv(prepared_training_data_path)

        training_table = pd.DataFrame(training_records)
        action_counts_path = run_paths.data_directory / "iqn_train_action_counts.csv"
        effective_action_counts_path = (
            run_paths.data_directory / "iqn_train_effective_action_counts.csv"
        )

        if not training_table.empty:
            action_counts = (
                training_table["action_label"]
                .value_counts()
                .rename_axis("action")
                .reset_index(name="count")
            )
            action_counts["pct"] = (
                action_counts["count"] / action_counts["count"].sum() * 100
            )

            effective_action_counts = (
                training_table["effective_action"]
                .fillna("UNKNOWN")
                .value_counts()
                .rename_axis("effective_action")
                .reset_index(name="count")
            )
            effective_action_counts["pct"] = (
                effective_action_counts["count"]
                / effective_action_counts["count"].sum()
                * 100
            )
        else:
            action_counts = pd.DataFrame(columns=["action", "count", "pct"])
            effective_action_counts = pd.DataFrame(
                columns=["effective_action", "count", "pct"]
            )

        action_counts.to_csv(action_counts_path, index=False)
        effective_action_counts.to_csv(effective_action_counts_path, index=False)

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
            "iqn": {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "action_labels": {
                    str(index): action_to_label(index) for index in range(action_dim)
                },
                "config": config_to_dict(iqn_config),
                "total_steps": total_steps,
                "learning_starts": learning_starts,
                "trained_on_split": "train_data",
                "change_strategy_disabled_during_training": disable_change_strategy,
                "episodes_seen": episode_count,
                "final_buffer_size": get_buffer_length(agent),
                "learn_steps": len(losses),
                "loss_initial": losses[0] if losses else None,
                "loss_final": losses[-1] if losses else None,
                "loss_min": min(losses) if losses else None,
                "loss_max": max(losses) if losses else None,
                "loss_mean": float(np.mean(losses)) if losses else None,
                "eval_quantiles": eval_quantiles,
                "final_action_mask": final_action_mask_info,
                "action_counts": action_counts.to_dict(orient="records"),
                "effective_action_counts": effective_action_counts.to_dict(
                    orient="records"
                ),
                "selected_action_from_distribution": action_distributions.get(
                    "selected_action_label"
                ),
            },
            "outputs": {
                "model_path": str(model_path),
                "training_records_path": str(training_records_path),
                "action_distributions_path": str(action_distributions_path),
                "decision_memory_path": str(decision_memory_path),
                "asset_memory_path": str(asset_memory_path),
                "action_memory_path": str(action_memory_path),
                "prepared_training_data_path": str(prepared_training_data_path),
                "action_counts_path": str(action_counts_path),
                "effective_action_counts_path": str(effective_action_counts_path),
            },
            "next_step": (
                "Run IQN backtest on PIT trade_data with the newly trained model. "
                "If the policy still only chooses HOLD, inspect reward scaling and "
                "risk/action scoring before moving to full comparison."
            ),
        }

        summary_path = run_paths.summary_directory / "iqn_train_smoke_summary.json"
        write_json(summary_path, summary)

        run_logger.info("IQN train smoke test completed.")
        run_logger.info("State dim: %s", state_dim)
        run_logger.info("Action dim: %s", action_dim)
        run_logger.info("Total steps: %s", total_steps)
        run_logger.info("Replay buffer size: %s", get_buffer_length(agent))
        run_logger.info("Learn steps: %s", len(losses))
        run_logger.info("Loss final: %s", losses[-1] if losses else None)
        run_logger.info(
            "Selected action from distribution: %s",
            action_distributions.get("selected_action_label"),
        )
        run_logger.info("Wrote model: %s", model_path)
        run_logger.info("Wrote action distributions: %s", action_distributions_path)
        run_logger.info("Wrote summary: %s", summary_path)

        system_logger.info(
            "StockInvestmentDSS IQN train smoke test completed successfully."
        )

        return 0

    except Exception:
        system_logger.exception("StockInvestmentDSS IQN train smoke test failed.")

        if run_paths is not None:
            try:
                run_logger = setup_run_logger(run_paths, log_level=log_level)
                run_logger.exception("Run failed.")
            except Exception:
                pass

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
