# src/stock_investment_dss/runner/run_iqn_backtest_smoke_test.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from stock_investment_dss.data.finrl_data_pipeline import (
    load_or_create_finrl_daily_dataset,
)
from stock_investment_dss.data.point_in_time_split import create_point_in_time_split
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
from stock_investment_dss.evaluation.portfolio_metrics import (
    compute_portfolio_metrics,
    write_json,
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

ACTION_LABELS = {
    0: "HOLD",
    1: "BUY",
    2: "SELL",
    3: "REBALANCE",
    4: "CHANGE_STRATEGY",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_int_environment_variable(name: str, default: int) -> int:
    value = get_environment_variable(name, default=str(default))
    return int(value or default)


def get_float_environment_variable(name: str, default: float) -> float:
    value = get_environment_variable(name, default=str(default))
    return float(value or default)


def find_latest_iqn_train_run() -> Path:
    runs_root = PROJECT_ROOT / "outputs" / "runs"

    candidates = [
        path
        for path in runs_root.iterdir()
        if path.is_dir()
        and path.name.endswith("d_iqn_dss_iqn_train_smoke_test")
        and (path / "models" / "iqn_train_smoke_model.pt").exists()
    ]

    if not candidates:
        raise FileNotFoundError(
            "Could not find an IQN train smoke-test run with "
            "models/iqn_train_smoke_model.pt."
        )

    return sorted(candidates, key=lambda path: path.name)[-1]


def resolve_iqn_source_run_directory() -> Path:
    source_run_id = get_environment_variable(
        "STOCK_INVESTMENT_DSS_IQN_BACKTEST_SOURCE_RUN_ID",
        default=None,
    )

    source_run_directory = get_environment_variable(
        "STOCK_INVESTMENT_DSS_IQN_BACKTEST_SOURCE_RUN_DIRECTORY",
        default=None,
    )

    if source_run_directory:
        path = Path(source_run_directory)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path

    if source_run_id:
        return PROJECT_ROOT / "outputs" / "runs" / source_run_id

    return find_latest_iqn_train_run()


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


def create_iqn_config_from_checkpoint_or_default(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> IQNConfig:
    config = build_iqn_config()

    checkpoint_config = checkpoint.get("config")

    if isinstance(checkpoint_config, IQNConfig):
        config = checkpoint_config

    elif isinstance(checkpoint_config, dict):
        for key, value in checkpoint_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    config.device = device
    config.env_name = "D-IQN-DSS-FinRL-DiscreteDecisionEnv"

    return config


def create_backtest_environment(
    tickers: list[str],
    trade_data: pd.DataFrame,
    initial_amount: float,
    hmax: int,
    buy_cost_pct: float,
    sell_cost_pct: float,
    reward_scaling: float,
    risk_profile: InvestorRiskProfile,
):
    finrl_env, prepared_trade_data, finrl_env_metadata = create_finrl_stock_trading_env(
        market_data=trade_data,
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

    return env, prepared_trade_data, finrl_env_metadata


def load_iqn_agent_from_checkpoint(
    model_path: Path,
    state_dim: int,
    action_dim: int,
    device: torch.device,
) -> tuple[IQNAgent, IQNConfig, dict[str, Any]]:
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported IQN checkpoint format: {model_path}")

    config = create_iqn_config_from_checkpoint_or_default(
        checkpoint=checkpoint,
        device=device,
    )

    agent = IQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
    )

    if "online_net_state_dict" in checkpoint:
        agent.online_net.load_state_dict(checkpoint["online_net_state_dict"])
        agent.target_net.load_state_dict(
            checkpoint.get("target_net_state_dict", checkpoint["online_net_state_dict"])
        )

    elif "model_state_dict" in checkpoint:
        agent.online_net.load_state_dict(checkpoint["model_state_dict"])
        agent.target_net.load_state_dict(checkpoint["model_state_dict"])

    elif "state_dict" in checkpoint:
        agent.online_net.load_state_dict(checkpoint["state_dict"])
        agent.target_net.load_state_dict(checkpoint["state_dict"])

    else:
        raise ValueError(
            "Could not load IQN checkpoint. Expected online_net_state_dict, "
            "target_net_state_dict, model_state_dict, or state_dict."
        )

    agent.online_net.eval()
    agent.target_net.eval()

    return agent, config, checkpoint


def score_action_distribution(
    values: dict[str, Any],
    score_mode: str,
    risk_lambda: float,
) -> float:
    """Score an IQN action return distribution for evaluation-time action choice.

    This function is the DSS/evaluation-time action-selection policy on top of
    the IQN distribution estimates. It is not the IQN training loss.

    Supported ablation modes:
    - mean
    - q50 / median
    - q25, q75, q90
    - cvar10
    - q50_minus_cvar_penalty
    - mean_minus_cvar_penalty
    - mean_plus_cvar10
    """

    normalized_mode = str(score_mode or "q50_minus_cvar_penalty").strip().lower()

    mean = float(values.get("mean", 0.0))
    q25 = float(values.get("q25", 0.0))
    q50 = float(values.get("q50", 0.0))
    q75 = float(values.get("q75", 0.0))
    q90 = float(values.get("q90", 0.0))
    cvar10 = float(values.get("cvar10", 0.0))

    if normalized_mode == "mean":
        return mean

    if normalized_mode in {"q50", "median"}:
        return q50

    if normalized_mode == "q25":
        return q25

    if normalized_mode == "q75":
        return q75

    if normalized_mode == "q90":
        return q90

    if normalized_mode == "cvar10":
        return cvar10

    if normalized_mode == "q50_minus_cvar_penalty":
        return q50 - risk_lambda * abs(cvar10)

    if normalized_mode == "mean_minus_cvar_penalty":
        return mean - risk_lambda * abs(cvar10)

    if normalized_mode == "mean_plus_cvar10":
        return mean + cvar10

    raise ValueError(
        f"Unsupported IQN score mode: {score_mode}. "
        "Use mean, q50, median, q25, q75, q90, cvar10, "
        "q50_minus_cvar_penalty, mean_minus_cvar_penalty, or mean_plus_cvar10."
    )


def choose_action_from_distribution(
    distribution_output: dict[str, Any],
    score_mode: str,
    risk_lambda: float,
) -> tuple[int, str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []

    for action_label, values in distribution_output.get("distributions", {}).items():
        allowed = bool(values.get("allowed"))

        score = score_action_distribution(
            values=values,
            score_mode=score_mode,
            risk_lambda=risk_lambda,
        )

        rows.append(
            {
                "action": action_label,
                "action_index": int(values.get("action_index")),
                "allowed": allowed,
                "score_mode": score_mode,
                "score": score,
                "mean": values.get("mean"),
                "q10": values.get("q10"),
                "q25": values.get("q25"),
                "q50": values.get("q50"),
                "q75": values.get("q75"),
                "q90": values.get("q90"),
                "cvar10": values.get("cvar10"),
            }
        )

    allowed_rows = [row for row in rows if row["allowed"]]

    if not allowed_rows:
        return 0, "HOLD", rows

    selected = sorted(
        allowed_rows,
        key=lambda row: row["score"],
        reverse=True,
    )[0]

    return int(selected["action_index"]), str(selected["action"]), rows


def main() -> int:
    log_level = (
        get_environment_variable(
            "STOCK_INVESTMENT_DSS_LOG_LEVEL",
            default="INFO",
        )
        or "INFO"
    )

    system_logger = setup_system_logger(log_level=log_level)
    system_logger.info("Starting StockInvestmentDSS IQN backtest smoke test.")
    system_logger.info("Project root: %s", PROJECT_ROOT)

    run_paths = None

    try:
        set_global_seed(42)

        source_run_directory = resolve_iqn_source_run_directory()
        model_path = source_run_directory / "models" / "iqn_train_smoke_model.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing IQN model: {model_path}")

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

        max_steps = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_BACKTEST_MAX_STEPS",
            default=10_000,
        )

        num_quantiles = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_BACKTEST_NUM_QUANTILES",
            default=128,
        )

        risk_lambda = get_float_environment_variable(
            "STOCK_INVESTMENT_DSS_IQN_BACKTEST_RISK_LAMBDA",
            default=0.75,
        )

        score_mode = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_IQN_BACKTEST_SCORE_MODE",
                default="mean",
            )
            or "mean"
        ).strip()

        requested_device = (
            get_environment_variable(
                "STOCK_INVESTMENT_DSS_IQN_DEVICE",
                default="cuda" if torch.cuda.is_available() else "cpu",
            )
            or "cpu"
        )

        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"

        device = torch.device(requested_device)

        risk_profile = create_risk_profile_from_environment()

        run_paths = create_run_paths("d_iqn_dss_iqn_backtest_smoke_test")
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Source IQN run: %s", source_run_directory)
        run_logger.info("Model path: %s", model_path)
        run_logger.info("Dataset id: %s", dataset_id)
        run_logger.info("Universe id: %s", universe_id)
        run_logger.info("Point in time: %s", point_in_time)
        run_logger.info("Trade end date: %s", trade_end_date)
        run_logger.info("Score mode: %s", score_mode)
        run_logger.info("Risk lambda: %s", risk_lambda)
        run_logger.info("Num quantiles: %s", num_quantiles)

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

        env, prepared_trade_data, finrl_env_metadata = create_backtest_environment(
            tickers=list(daily_data_result.tickers),
            trade_data=split_result.trade_data,
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

        agent, iqn_config, checkpoint = load_iqn_agent_from_checkpoint(
            model_path=model_path,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )

        done = False
        step_index = 0

        decision_rows: list[dict[str, Any]] = []
        distribution_rows: list[dict[str, Any]] = []

        while not done and step_index < max_steps:
            action_mask_info = env.get_action_mask()
            action_mask = action_mask_info.get("mask_vector")

            # Disable CHANGE_STRATEGY during IQN backtest smoke test until
            # real strategy-switch execution logic exists.
            if action_mask is not None and len(action_mask) >= 5:
                action_mask[4] = 0

            distribution_output = agent.estimate_action_distributions(
                state=state,
                num_quantiles=num_quantiles,
                action_mask=action_mask,
            )

            action, action_label, estimate_rows = choose_action_from_distribution(
                distribution_output=distribution_output,
                score_mode=score_mode,
                risk_lambda=risk_lambda,
            )

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            decision_record = info.get("decision_record", {})

            for row in estimate_rows:
                distribution_rows.append(
                    {
                        "step_index": step_index,
                        "chosen_action": action_label,
                        "score_mode": score_mode,
                        **row,
                    }
                )

            state_after = decision_record.get("state_after") or {}
            state_before = decision_record.get("state_before") or {}
            resolved_action = decision_record.get("resolved_action") or {}
            execution_delta = decision_record.get("execution_delta") or {}

            decision_rows.append(
                {
                    "step_index": step_index,
                    "chosen_action_index": action,
                    "chosen_action_label": action_label,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "done": done,
                    "action_mask": action_mask,
                    "effective_action": decision_record.get(
                        "effective_decision_action_label"
                    ),
                    "action_was_masked": decision_record.get("action_was_masked"),
                    "selected_ticker": resolved_action.get("selected_ticker"),
                    "requested_shares": resolved_action.get("requested_shares"),
                    "submitted_shares_estimate": resolved_action.get(
                        "submitted_shares_estimate"
                    ),
                    "hmax_limited": resolved_action.get("hmax_limited"),
                    "cash_before": state_before.get("cash"),
                    "cash_after": state_after.get("cash"),
                    "portfolio_value_before": state_before.get("portfolio_value"),
                    "portfolio_value_after": state_after.get("portfolio_value"),
                    "cash_delta": execution_delta.get("cash_delta"),
                    "portfolio_value_delta": execution_delta.get(
                        "portfolio_value_delta"
                    ),
                    "cost_delta": execution_delta.get("cost_delta"),
                    "trades_delta": execution_delta.get("trades_delta"),
                    "finrl_cost": decision_record.get("finrl_cost"),
                    "finrl_trades": decision_record.get("finrl_trades"),
                }
            )

            state = np.asarray(next_observation, dtype=np.float32).reshape(-1)
            step_index += 1

        asset_memory = env.save_asset_memory()
        action_memory = env.save_action_memory()
        decision_memory = env.save_decision_memory()

        metrics_result = compute_portfolio_metrics(
            asset_memory=asset_memory,
            decision_memory={"decisions": decision_memory},
            step_table=pd.DataFrame(decision_rows),
        )

        prepared_trade_data_path = (
            run_paths.data_directory / "iqn_backtest_prepared_trade_data.csv"
        )
        asset_memory_path = run_paths.data_directory / "iqn_backtest_asset_memory.csv"
        action_memory_path = run_paths.data_directory / "iqn_backtest_action_memory.csv"
        decision_memory_path = (
            run_paths.data_directory / "iqn_backtest_decision_memory.json"
        )
        decision_table_path = (
            run_paths.data_directory / "iqn_backtest_decision_table.csv"
        )
        distribution_table_path = (
            run_paths.data_directory / "iqn_backtest_distribution_table.csv"
        )
        metrics_timeseries_path = (
            run_paths.data_directory / "iqn_backtest_metrics_timeseries.csv"
        )
        metrics_summary_path = run_paths.summary_directory / "iqn_backtest_metrics.json"
        summary_path = run_paths.summary_directory / "iqn_backtest_smoke_summary.json"

        prepared_trade_data.to_csv(prepared_trade_data_path)
        asset_memory.to_csv(asset_memory_path, index=False)
        action_memory.to_csv(action_memory_path)
        write_json(decision_memory_path, {"decisions": decision_memory})

        decision_table = pd.DataFrame(decision_rows)
        distribution_table = pd.DataFrame(distribution_rows)

        policy_summary_path = (
            run_paths.summary_directory / "iqn_backtest_policy_summary.json"
        )
        action_counts_path = run_paths.data_directory / "iqn_backtest_action_counts.csv"
        effective_action_counts_path = (
            run_paths.data_directory / "iqn_backtest_effective_action_counts.csv"
        )
        iqn_vs_effective_action_summary_path = (
            run_paths.data_directory
            / "iqn_backtest_iqn_vs_effective_action_summary.csv"
        )
        iqn_determined_decision_summary_path = (
            run_paths.data_directory
            / "iqn_backtest_iqn_determined_decision_summary.csv"
        )
        chosen_action_distribution_summary_path = (
            run_paths.data_directory
            / "iqn_backtest_chosen_action_distribution_summary.csv"
        )

        action_counts = (
            decision_table["chosen_action_label"]
            .value_counts()
            .rename_axis("action")
            .reset_index(name="count")
        )

        action_counts["pct"] = (
            action_counts["count"] / action_counts["count"].sum() * 100
        )

        chosen_distributions = distribution_table[
            distribution_table["action"] == distribution_table["chosen_action"]
        ].copy()

        chosen_action_distribution_summary = (
            chosen_distributions.groupby("action")
            .agg(
                count=("action", "count"),
                mean_score=("score", "mean"),
                mean_return_estimate=("mean", "mean"),
                mean_q10=("q10", "mean"),
                mean_q50=("q50", "mean"),
                mean_q90=("q90", "mean"),
                mean_cvar10=("cvar10", "mean"),
            )
            .reset_index()
        )

        effective_action_counts = (
            decision_table["effective_action"]
            .fillna("UNKNOWN")
            .value_counts()
            .rename_axis("effective_action")
            .reset_index(name="count")
        )

        if not effective_action_counts.empty:
            effective_action_counts["pct"] = (
                effective_action_counts["count"]
                / effective_action_counts["count"].sum()
                * 100
            )

        iqn_vs_effective_action_summary = (
            decision_table.assign(
                effective_action=decision_table["effective_action"].fillna("UNKNOWN"),
                action_was_masked=decision_table["action_was_masked"].fillna(False),
            )
            .groupby(
                [
                    "chosen_action_label",
                    "effective_action",
                    "action_was_masked",
                ],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
        )

        iqn_determined_decisions = distribution_table[
            distribution_table["action"] == distribution_table["chosen_action"]
        ].copy()

        iqn_determined_decision_summary = (
            iqn_determined_decisions.groupby("chosen_action")
            .agg(
                count=("chosen_action", "count"),
                mean_iqn_score=("score", "mean"),
                mean_iqn_expected_return=("mean", "mean"),
                mean_iqn_q10=("q10", "mean"),
                mean_iqn_q50=("q50", "mean"),
                mean_iqn_q90=("q90", "mean"),
                mean_iqn_cvar10=("cvar10", "mean"),
            )
            .reset_index()
        )

        mask_summary = {
            "masked_step_count": (
                int(decision_table["action_was_masked"].fillna(False).sum())
                if "action_was_masked" in decision_table.columns
                else 0
            ),
            "masked_step_pct": (
                float(decision_table["action_was_masked"].fillna(False).mean() * 100)
                if "action_was_masked" in decision_table.columns
                and not decision_table.empty
                else 0.0
            ),
        }

        transaction_summary = {
            "total_cost_delta": (
                float(decision_table["cost_delta"].fillna(0).sum())
                if "cost_delta" in decision_table.columns
                else 0.0
            ),
            "total_trades_delta": (
                int(decision_table["trades_delta"].fillna(0).sum())
                if "trades_delta" in decision_table.columns
                else 0
            ),
            "buy_like_steps": (
                int((decision_table["chosen_action_label"] == "BUY").sum())
                if "chosen_action_label" in decision_table.columns
                else 0
            ),
            "sell_like_steps": (
                int((decision_table["chosen_action_label"] == "SELL").sum())
                if "chosen_action_label" in decision_table.columns
                else 0
            ),
        }

        action_counts.to_csv(action_counts_path, index=False)
        effective_action_counts.to_csv(effective_action_counts_path, index=False)
        iqn_vs_effective_action_summary.to_csv(
            iqn_vs_effective_action_summary_path,
            index=False,
        )
        iqn_determined_decision_summary.to_csv(
            iqn_determined_decision_summary_path,
            index=False,
        )
        chosen_action_distribution_summary.to_csv(
            chosen_action_distribution_summary_path,
            index=False,
        )

        policy_summary = {
            "executed_steps": int(len(decision_table)),
            "score_mode": score_mode,
            "risk_lambda": risk_lambda,
            "iqn_selected_action_counts": action_counts.to_dict(orient="records"),
            "effective_action_counts": effective_action_counts.to_dict(
                orient="records"
            ),
            "iqn_vs_effective_action_summary": (
                iqn_vs_effective_action_summary.to_dict(orient="records")
            ),
            "iqn_determined_decision_summary": (
                iqn_determined_decision_summary.to_dict(orient="records")
            ),
            "chosen_action_distribution_summary": (
                chosen_action_distribution_summary.to_dict(orient="records")
            ),
            "mask_summary": mask_summary,
            "transaction_summary": transaction_summary,
            "interpretation": (
                "This summary describes what IQN selected across the full backtest, "
                "what the DSS/FinRL adapter effectively executed, and how often "
                "actions were masked or translated. This is the primary policy "
                "summary. A last-step snapshot is only useful as a diagnostic "
                "visualization."
            ),
            "output_files": {
                "policy_summary_path": str(policy_summary_path),
                "iqn_selected_action_counts_path": str(action_counts_path),
                "effective_action_counts_path": str(effective_action_counts_path),
                "iqn_vs_effective_action_summary_path": str(
                    iqn_vs_effective_action_summary_path
                ),
                "iqn_determined_decision_summary_path": str(
                    iqn_determined_decision_summary_path
                ),
                "chosen_action_distribution_summary_path": str(
                    chosen_action_distribution_summary_path
                ),
            },
        }

        write_json(policy_summary_path, policy_summary)

        decision_table.to_csv(decision_table_path, index=False)
        distribution_table.to_csv(distribution_table_path, index=False)
        metrics_result.timeseries.to_csv(metrics_timeseries_path, index=False)
        write_json(metrics_summary_path, metrics_result.summary)

        action_counts = (
            decision_table["chosen_action_label"].value_counts().to_dict()
            if not decision_table.empty
            else {}
        )

        summary = {
            "status": "ok",
            "project_name": "StockInvestmentDSS",
            "prototype_name": "D-IQN-DSS",
            "run_id": run_paths.run_id,
            "project_root": str(PROJECT_ROOT),
            "run_directory": str(run_paths.run_directory),
            "source_iqn_run_directory": str(source_run_directory),
            "model_path": str(model_path),
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
            "iqn_backtest": {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "score_mode": score_mode,
                "risk_lambda": risk_lambda,
                "num_quantiles": num_quantiles,
                "executed_steps": step_index,
                "action_counts": action_counts,
                "reset_info": reset_info,
                "config": {
                    key: str(value) if isinstance(value, torch.device) else value
                    for key, value in vars(iqn_config).items()
                },
            },
            "finrl_environment": finrl_env_metadata,
            "portfolio_metrics": metrics_result.summary,
            "outputs": {
                "prepared_trade_data_path": str(prepared_trade_data_path),
                "asset_memory_path": str(asset_memory_path),
                "action_memory_path": str(action_memory_path),
                "decision_memory_path": str(decision_memory_path),
                "decision_table_path": str(decision_table_path),
                "distribution_table_path": str(distribution_table_path),
                "policy_summary_path": str(policy_summary_path),
                "action_counts_path": str(action_counts_path),
                "effective_action_counts_path": str(effective_action_counts_path),
                "iqn_vs_effective_action_summary_path": str(
                    iqn_vs_effective_action_summary_path
                ),
                "iqn_determined_decision_summary_path": str(
                    iqn_determined_decision_summary_path
                ),
                "chosen_action_distribution_summary_path": str(
                    chosen_action_distribution_summary_path
                ),
                "metrics_timeseries_path": str(metrics_timeseries_path),
                "metrics_summary_path": str(metrics_summary_path),
            },
            "next_step": (
                "Compare IQN backtest metrics against the FinRL baseline suite "
                "and then move toward the full risk policy scoring layer."
            ),
        }

        write_json(summary_path, summary)

        run_logger.info("IQN backtest smoke test completed.")
        run_logger.info("Executed steps: %s", step_index)
        run_logger.info("Action counts: %s", action_counts)
        run_logger.info(
            "Initial value: %s", metrics_result.summary.get("initial_value")
        )
        run_logger.info("Final value: %s", metrics_result.summary.get("final_value"))
        run_logger.info(
            "Total return pct: %s",
            metrics_result.summary.get("total_return_pct"),
        )
        run_logger.info(
            "Max drawdown pct: %s",
            metrics_result.summary.get("max_drawdown_pct"),
        )
        run_logger.info("Wrote decision table: %s", decision_table_path)
        run_logger.info("Wrote distribution table: %s", distribution_table_path)
        run_logger.info("Wrote policy summary: %s", policy_summary_path)
        run_logger.info("Wrote summary: %s", summary_path)

        system_logger.info(
            "StockInvestmentDSS IQN backtest smoke test completed successfully."
        )

        return 0

    except Exception:
        system_logger.exception("StockInvestmentDSS IQN backtest smoke test failed.")

        if run_paths is not None:
            try:
                run_logger = setup_run_logger(run_paths, log_level=log_level)
                run_logger.exception("Run failed.")
            except Exception:
                pass

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
