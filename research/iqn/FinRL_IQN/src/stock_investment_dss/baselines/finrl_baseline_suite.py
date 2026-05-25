# src/stock_investment_dss/baselines/finrl_baseline_suite.py

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

from stock_investment_dss.environments.finrl_env_factory import (
    FinRLStockTradingEnvConfig,
    create_finrl_stock_trading_env,
    unpack_reset_result,
    unpack_step_result,
)
from stock_investment_dss.evaluation.portfolio_metrics import (
    compute_portfolio_metrics,
    write_json,
)

SUPPORTED_RL_BASELINES = ["a2c", "ddpg", "td3", "ppo", "sac"]
SUPPORTED_BASELINES = SUPPORTED_RL_BASELINES + ["mvo"]


@dataclass(frozen=True)
class FinRLBaselineSuiteConfig:
    agents: list[str]
    total_timesteps: int = 500
    initial_amount: float = 1_000_000.0
    hmax: int = 100
    buy_cost_pct: float = 0.001
    sell_cost_pct: float = 0.001
    reward_scaling: float = 0.0001
    device: str = "auto"
    deterministic_backtest: bool = True
    include_mvo: bool = True


@dataclass(frozen=True)
class BaselineRunResult:
    agent_name: str
    status: str
    model_path: str | None
    asset_memory_path: str | None
    action_memory_path: str | None
    metrics_summary_path: str | None
    metrics_timeseries_path: str | None
    summary: dict[str, Any]


def normalize_agents(
    raw_agents: list[str] | str, include_mvo: bool = True
) -> list[str]:
    if isinstance(raw_agents, str):
        agents = [
            agent.strip().lower() for agent in raw_agents.split(",") if agent.strip()
        ]
    else:
        agents = [agent.strip().lower() for agent in raw_agents if agent.strip()]

    if include_mvo and "mvo" not in agents:
        agents.append("mvo")

    unknown = sorted(set(agents) - set(SUPPORTED_BASELINES))

    if unknown:
        raise ValueError(
            f"Unsupported baseline agents: {unknown}. "
            f"Supported: {SUPPORTED_BASELINES}"
        )

    return agents


def clean_market_data_for_finrl(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    unnamed_columns = [
        column for column in df.columns if str(column).lower().startswith("unnamed")
    ]

    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    required_columns = {"date", "tic", "close"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Market data missing required columns: {sorted(missing_columns)}"
        )

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["tic"] = df["tic"].astype(str).str.upper().str.strip()
    df = df.sort_values(["date", "tic"]).copy()

    return df


def infer_technical_indicators(data: pd.DataFrame) -> list[str]:
    preferred = [
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "cci_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
        "vix",
    ]

    indicators = [indicator for indicator in preferred if indicator in data.columns]

    if indicators:
        return indicators

    excluded = {
        "date",
        "tic",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "day",
    }

    indicators = (
        data.drop(columns=list(excluded), errors="ignore")
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )

    if not indicators:
        raise ValueError(
            "Could not infer technical indicators. "
            "Enable technical indicators or use a dataset with numeric features."
        )

    return indicators


def create_sb3_model(
    agent_name: str,
    env,
    stock_dim: int,
    device: str,
):
    agent = agent_name.lower()

    if agent == "a2c":
        return A2C(
            "MlpPolicy",
            env,
            n_steps=5,
            ent_coef=0.01,
            learning_rate=0.0007,
            verbose=0,
            device=device,
        )

    if agent == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            n_steps=128,
            batch_size=64,
            ent_coef=0.01,
            learning_rate=0.00025,
            verbose=0,
            device=device,
        )

    if agent == "ddpg":
        action_noise = NormalActionNoise(
            mean=np.zeros(stock_dim),
            sigma=0.1 * np.ones(stock_dim),
        )

        return DDPG(
            "MlpPolicy",
            env,
            batch_size=128,
            buffer_size=50_000,
            learning_rate=0.001,
            action_noise=action_noise,
            verbose=0,
            device=device,
        )

    if agent == "td3":
        action_noise = NormalActionNoise(
            mean=np.zeros(stock_dim),
            sigma=0.1 * np.ones(stock_dim),
        )

        return TD3(
            "MlpPolicy",
            env,
            batch_size=100,
            buffer_size=100_000,
            learning_rate=0.001,
            action_noise=action_noise,
            verbose=0,
            device=device,
        )

    if agent == "sac":
        return SAC(
            "MlpPolicy",
            env,
            batch_size=64,
            buffer_size=10_000,
            learning_rate=0.0001,
            verbose=0,
            device=device,
        )

    raise ValueError(f"Unsupported RL baseline agent: {agent_name}")


def train_single_rl_agent(
    agent_name: str,
    train_data: pd.DataFrame,
    tickers: list[str],
    technical_indicators: list[str],
    config: FinRLBaselineSuiteConfig,
    model_directory: Path,
) -> tuple[Any, Path, dict]:
    env, prepared_train_data, env_metadata = create_finrl_stock_trading_env(
        market_data=train_data,
        tickers=tickers,
        config=FinRLStockTradingEnvConfig(
            initial_amount=config.initial_amount,
            hmax=config.hmax,
            buy_cost_pct=config.buy_cost_pct,
            sell_cost_pct=config.sell_cost_pct,
            reward_scaling=config.reward_scaling,
            print_verbosity=10_000,
        ),
        technical_indicators=technical_indicators,
    )

    stock_dim = len(tickers)
    model = create_sb3_model(
        agent_name=agent_name,
        env=env,
        stock_dim=stock_dim,
        device=config.device,
    )

    started_at = time.time()
    model.learn(total_timesteps=config.total_timesteps)
    training_seconds = time.time() - started_at

    model_directory.mkdir(parents=True, exist_ok=True)
    model_path = model_directory / f"{agent_name}.zip"
    model.save(model_path)

    metadata = {
        "agent_name": agent_name,
        "model_path": str(model_path),
        "total_timesteps": config.total_timesteps,
        "training_seconds": training_seconds,
        "env_metadata": env_metadata,
        "prepared_train_rows": int(len(prepared_train_data)),
    }

    return model, model_path, metadata


def backtest_single_rl_agent(
    agent_name: str,
    model,
    trade_data: pd.DataFrame,
    tickers: list[str],
    technical_indicators: list[str],
    config: FinRLBaselineSuiteConfig,
    output_directory: Path,
) -> BaselineRunResult:
    env, prepared_trade_data, env_metadata = create_finrl_stock_trading_env(
        market_data=trade_data,
        tickers=tickers,
        config=FinRLStockTradingEnvConfig(
            initial_amount=config.initial_amount,
            hmax=config.hmax,
            buy_cost_pct=config.buy_cost_pct,
            sell_cost_pct=config.sell_cost_pct,
            reward_scaling=config.reward_scaling,
            print_verbosity=10_000,
        ),
        technical_indicators=technical_indicators,
    )

    reset_result = env.reset()
    observation, reset_info = unpack_reset_result(reset_result)

    done = False
    executed_steps = 0
    action_records: list[np.ndarray] = []

    while not done:
        action, _states = model.predict(
            observation,
            deterministic=config.deterministic_backtest,
        )

        action_array = np.asarray(action, dtype=float).reshape(-1)
        action_records.append(action_array)

        step_result = env.step(action)
        observation, reward, terminated, info = unpack_step_result(step_result)

        done = bool(terminated)
        executed_steps += 1

        if executed_steps > len(prepared_trade_data["date"].unique()) + 5:
            break

    asset_memory = env.save_asset_memory()
    action_memory = env.save_action_memory()

    action_matrix = (
        np.vstack(action_records)
        if action_records
        else np.zeros((0, len(tickers)), dtype=float)
    )

    action_mean_abs = (
        float(np.mean(np.abs(action_matrix))) if action_matrix.size > 0 else 0.0
    )

    action_max_abs = (
        float(np.max(np.abs(action_matrix))) if action_matrix.size > 0 else 0.0
    )

    non_zero_action_steps = (
        int(np.sum(np.any(np.abs(action_matrix) > 1e-8, axis=1)))
        if action_matrix.size > 0
        else 0
    )

    agent_output_directory = output_directory / agent_name
    agent_output_directory.mkdir(parents=True, exist_ok=True)

    asset_memory_path = agent_output_directory / f"{agent_name}_asset_memory.csv"
    action_memory_path = agent_output_directory / f"{agent_name}_action_memory.csv"
    metrics_timeseries_path = (
        agent_output_directory / f"{agent_name}_metrics_timeseries.csv"
    )
    metrics_summary_path = agent_output_directory / f"{agent_name}_metrics_summary.json"

    asset_memory.to_csv(asset_memory_path, index=False)
    action_memory.to_csv(action_memory_path)

    metrics = compute_portfolio_metrics(
        asset_memory=asset_memory,
        decision_memory=None,
        step_table=None,
    )

    metrics.timeseries.to_csv(metrics_timeseries_path, index=False)

    initial_value = float(metrics.summary["initial_value"])
    final_value = float(metrics.summary["final_value"])
    portfolio_value_changed = abs(final_value - initial_value) > 1e-8
    finrl_trades = int(getattr(env, "trades", 0))
    finrl_cost = float(getattr(env, "cost", 0.0))

    trading_status = (
        "ok_trading"
        if finrl_trades > 0 or non_zero_action_steps > 0 or portfolio_value_changed
        else "no_trade"
    )

    summary = {
        **metrics.summary,
        "agent_name": agent_name,
        "baseline_type": "FinRL/SB3 RL",
        "executed_steps": executed_steps,
        "reset_info": reset_info,
        "finrl_environment": env_metadata,
        "finrl_cost": finrl_cost,
        "finrl_trades": finrl_trades,
        "action_mean_abs": action_mean_abs,
        "action_max_abs": action_max_abs,
        "non_zero_action_steps": non_zero_action_steps,
        "portfolio_value_changed": portfolio_value_changed,
        "trading_status": trading_status,
        "asset_memory_path": str(asset_memory_path),
        "action_memory_path": str(action_memory_path),
        "metrics_timeseries_path": str(metrics_timeseries_path),
    }

    write_json(metrics_summary_path, summary)

    return BaselineRunResult(
        agent_name=agent_name,
        status="ok",
        model_path=None,
        asset_memory_path=str(asset_memory_path),
        action_memory_path=str(action_memory_path),
        metrics_summary_path=str(metrics_summary_path),
        metrics_timeseries_path=str(metrics_timeseries_path),
        summary=summary,
    )


def pivot_close_prices(data: pd.DataFrame) -> pd.DataFrame:
    df = clean_market_data_for_finrl(data)

    close_prices = (
        df.pivot_table(
            index="date",
            columns="tic",
            values="close",
            aggfunc="last",
        )
        .sort_index()
        .ffill()
        .dropna(axis=1, how="any")
    )

    if close_prices.empty:
        raise ValueError("Could not build close-price pivot table for MVO.")

    return close_prices


def compute_equal_weight_mvo(
    train_data: pd.DataFrame,
    trade_data: pd.DataFrame,
    initial_amount: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_prices = pivot_close_prices(train_data)
    trade_prices = pivot_close_prices(trade_data)

    common_tickers = [
        ticker for ticker in train_prices.columns if ticker in trade_prices.columns
    ]

    train_prices = train_prices[common_tickers]
    trade_prices = trade_prices[common_tickers]

    if not common_tickers:
        raise ValueError("No common tickers between train and trade prices for MVO.")

    returns = train_prices.pct_change().dropna()

    if returns.empty:
        weights = np.ones(len(common_tickers)) / len(common_tickers)
        mvo_status = "equal_weight_fallback_no_returns"
    else:
        expected_returns = returns.mean().to_numpy()
        covariance = returns.cov().to_numpy()

        try:
            from pypfopt.efficient_frontier import EfficientFrontier

            ef = EfficientFrontier(
                expected_returns,
                covariance,
                weight_bounds=(0, 0.5),
                solver="SCS",
            )
            ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            weights = np.array(
                [
                    float(cleaned_weights.get(index, 0.0))
                    for index in range(len(common_tickers))
                ],
                dtype=float,
            )

            if weights.sum() <= 0:
                raise ValueError("Optimized weights sum to zero.")

            weights = weights / weights.sum()
            mvo_status = "optimized_scs"

        except Exception as exc:
            weights = np.ones(len(common_tickers)) / len(common_tickers)
            mvo_status = f"equal_weight_fallback: {exc}"

    first_trade_prices = trade_prices.iloc[0].to_numpy(dtype=float)
    shares = (initial_amount * weights) / first_trade_prices

    portfolio_values = trade_prices.to_numpy(dtype=float) @ shares

    asset_memory = pd.DataFrame(
        {
            "date": trade_prices.index,
            "account_value": portfolio_values,
        }
    )

    weights_df = pd.DataFrame(
        {
            "ticker": common_tickers,
            "weight": weights,
            "shares": shares,
        }
    )

    metadata = {
        "mvo_status": mvo_status,
        "tickers": common_tickers,
        "initial_amount": initial_amount,
        "weights_sum": float(weights.sum()),
    }

    return asset_memory, weights_df, metadata


def run_mvo_baseline(
    train_data: pd.DataFrame,
    trade_data: pd.DataFrame,
    config: FinRLBaselineSuiteConfig,
    output_directory: Path,
) -> BaselineRunResult:
    agent_name = "mvo"
    agent_output_directory = output_directory / agent_name
    agent_output_directory.mkdir(parents=True, exist_ok=True)

    asset_memory, weights_df, metadata = compute_equal_weight_mvo(
        train_data=train_data,
        trade_data=trade_data,
        initial_amount=config.initial_amount,
    )

    asset_memory_path = agent_output_directory / "mvo_asset_memory.csv"
    weights_path = agent_output_directory / "mvo_weights.csv"
    metrics_timeseries_path = agent_output_directory / "mvo_metrics_timeseries.csv"
    metrics_summary_path = agent_output_directory / "mvo_metrics_summary.json"

    asset_memory.to_csv(asset_memory_path, index=False)
    weights_df.to_csv(weights_path, index=False)

    metrics = compute_portfolio_metrics(
        asset_memory=asset_memory,
        decision_memory=None,
        step_table=None,
    )

    metrics.timeseries.to_csv(metrics_timeseries_path, index=False)

    initial_value = float(metrics.summary["initial_value"])
    final_value = float(metrics.summary["final_value"])

    summary = {
        **metrics.summary,
        "agent_name": agent_name,
        "baseline_type": "MVO / portfolio optimization baseline",
        "trading_status": "benchmark",
        "portfolio_value_changed": bool(abs(final_value - initial_value) > 1e-8),
        "action_mean_abs": None,
        "action_max_abs": None,
        "non_zero_action_steps": None,
        "mvo_metadata": metadata,
        "weights_path": str(weights_path),
        "asset_memory_path": str(asset_memory_path),
        "metrics_timeseries_path": str(metrics_timeseries_path),
    }

    write_json(metrics_summary_path, summary)

    return BaselineRunResult(
        agent_name=agent_name,
        status="ok",
        model_path=None,
        asset_memory_path=str(asset_memory_path),
        action_memory_path=None,
        metrics_summary_path=str(metrics_summary_path),
        metrics_timeseries_path=str(metrics_timeseries_path),
        summary=summary,
    )


def run_finrl_baseline_suite(
    train_data: pd.DataFrame,
    trade_data: pd.DataFrame,
    tickers: list[str],
    output_directory: Path,
    model_directory: Path,
    config: FinRLBaselineSuiteConfig,
) -> dict[str, Any]:
    train_data = clean_market_data_for_finrl(train_data)
    trade_data = clean_market_data_for_finrl(trade_data)

    technical_indicators = infer_technical_indicators(train_data)

    results: list[BaselineRunResult] = []
    training_metadata: dict[str, Any] = {}

    for agent_name in config.agents:
        if agent_name == "mvo":
            continue

        model, model_path, train_metadata = train_single_rl_agent(
            agent_name=agent_name,
            train_data=train_data,
            tickers=tickers,
            technical_indicators=technical_indicators,
            config=config,
            model_directory=model_directory,
        )

        training_metadata[agent_name] = train_metadata

        result = backtest_single_rl_agent(
            agent_name=agent_name,
            model=model,
            trade_data=trade_data,
            tickers=tickers,
            technical_indicators=technical_indicators,
            config=config,
            output_directory=output_directory,
        )

        result = BaselineRunResult(
            agent_name=result.agent_name,
            status=result.status,
            model_path=str(model_path),
            asset_memory_path=result.asset_memory_path,
            action_memory_path=result.action_memory_path,
            metrics_summary_path=result.metrics_summary_path,
            metrics_timeseries_path=result.metrics_timeseries_path,
            summary=result.summary,
        )

        results.append(result)

    if "mvo" in config.agents:
        results.append(
            run_mvo_baseline(
                train_data=train_data,
                trade_data=trade_data,
                config=config,
                output_directory=output_directory,
            )
        )

    comparison_rows = []

    for result in results:
        row = {
            "agent_name": result.agent_name,
            "status": result.status,
            "model_path": result.model_path,
            "asset_memory_path": result.asset_memory_path,
            "action_memory_path": result.action_memory_path,
            "metrics_summary_path": result.metrics_summary_path,
            "metrics_timeseries_path": result.metrics_timeseries_path,
        }

        row.update(
            {
                "initial_value": result.summary.get("initial_value"),
                "final_value": result.summary.get("final_value"),
                "profit_loss": result.summary.get("profit_loss"),
                "total_return_pct": result.summary.get("total_return_pct"),
                "max_drawdown_pct": result.summary.get("max_drawdown_pct"),
                "annualized_volatility_pct": result.summary.get(
                    "annualized_volatility_pct"
                ),
                "annualized_sharpe": result.summary.get("annualized_sharpe"),
                "cvar_pct": result.summary.get("cvar_pct"),
                "total_transaction_cost": result.summary.get("total_transaction_cost"),
                "total_trades": result.summary.get("total_trades"),
                "turnover_estimate_pct": result.summary.get("turnover_estimate_pct"),
                "finrl_trades": result.summary.get("finrl_trades"),
                "finrl_cost": result.summary.get("finrl_cost"),
                "trading_status": result.summary.get("trading_status"),
                "portfolio_value_changed": result.summary.get(
                    "portfolio_value_changed"
                ),
                "action_mean_abs": result.summary.get("action_mean_abs"),
                "action_max_abs": result.summary.get("action_max_abs"),
                "non_zero_action_steps": result.summary.get("non_zero_action_steps"),
            }
        )

        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)

    if not comparison.empty:
        comparison = comparison.sort_values(
            by=["final_value", "total_return_pct"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

        comparison.insert(0, "rank", range(1, len(comparison) + 1))

    output_directory.mkdir(parents=True, exist_ok=True)

    comparison_path = output_directory / "finrl_baseline_suite_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    suite_summary = {
        "status": "ok",
        "agents": config.agents,
        "rl_agents": [agent for agent in config.agents if agent != "mvo"],
        "include_mvo": "mvo" in config.agents,
        "total_timesteps": config.total_timesteps,
        "initial_amount": config.initial_amount,
        "hmax": config.hmax,
        "technical_indicators": technical_indicators,
        "training_metadata": training_metadata,
        "comparison_path": str(comparison_path),
        "results": [result.summary for result in results],
    }

    write_json(output_directory / "finrl_baseline_suite_summary.json", suite_summary)

    return {
        "summary": suite_summary,
        "comparison": comparison,
        "comparison_path": comparison_path,
    }
