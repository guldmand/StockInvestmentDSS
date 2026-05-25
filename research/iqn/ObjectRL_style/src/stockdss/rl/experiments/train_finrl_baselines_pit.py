"""
Train FinRL baseline agents on a point-in-time split.

This script belongs to the StockDSS codebase.

Purpose:
- Use PIT train data generated from market_data_full_500.csv.
- Train FinRL/SB3 baseline agents:
    A2C, DDPG, PPO, TD3, SAC
- Save trained models and training metadata using the PIT dataset tag.

Example:

PowerShell:
    $env:PYTHONPATH="src"

    python -m stockdss.rl.experiments.train_finrl_baselines_pit `
        --train-data data/train_data_pit_500_20260101.csv `
        --dataset-tag pit_500_20260101 `
        --total-timesteps 20000
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from stockdss.runner.run_paths import build_run_paths

DEFAULT_INITIAL_AMOUNT = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FinRL baseline agents on PIT train data."
    )

    parser.add_argument(
        "--train-data",
        required=True,
        help="Path to PIT train CSV, e.g. data/train_data_pit_500_20260101.csv",
    )

    parser.add_argument(
        "--dataset-tag",
        required=True,
        help="Dataset tag used for outputs, e.g. pit_500_20260101",
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=20_000,
        help="Training timesteps per agent. Default: 20000",
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Optional readable run name. If omitted, one is generated from "
            "timestamp, experiment name, agents, and timesteps."
        ),
    )

    parser.add_argument(
        "--agents",
        default="a2c,ddpg,ppo,td3,sac",
        help="Comma-separated agents to train. Default: a2c,ddpg,ppo,td3,sac",
    )

    parser.add_argument(
        "--initial-amount",
        type=float,
        default=DEFAULT_INITIAL_AMOUNT,
        help="Initial portfolio amount. Default: 1000000",
    )

    parser.add_argument(
        "--hmax",
        type=int,
        default=100,
        help="Maximum shares to trade per action dimension. Default: 100",
    )

    parser.add_argument(
        "--buy-cost-pct",
        type=float,
        default=0.001,
        help="Buy transaction cost used by FinRL env. Default: 0.001",
    )

    parser.add_argument(
        "--sell-cost-pct",
        type=float,
        default=0.001,
        help="Sell transaction cost used by FinRL env. Default: 0.001",
    )

    parser.add_argument(
        "--reward-scaling",
        type=float,
        default=1e-4,
        help="Reward scaling used by FinRL env. Default: 1e-4",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for SB3 models. Default: auto",
    )

    parser.add_argument(
        "--run-root",
        default=None,
        help=(
            "Optional central runner output folder. "
            "If provided, outputs are written to the canonical outputs/runs/<run_id>/ structure. "
            "Standalone behavior is preserved when omitted."
        ),
    )

    return parser.parse_args()


def load_finrl_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Train data not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Drop CSV index columns if present.
    unnamed_columns = [
        col for col in df.columns if str(col).lower().startswith("unnamed")
    ]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    required = {"date", "tic", "close"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Train data missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["tic"] = df["tic"].astype(str)

    df = df.sort_values(["date", "tic"]).copy()

    # Critical FinRL requirement:
    # StockTradingEnv expects all tickers for the same trading day to share
    # the same integer index. Then df.loc[day] returns a dataframe, not one row.
    df.index = pd.factorize(df["date"])[0]

    return df


def infer_technical_indicators(df: pd.DataFrame) -> list[str]:
    """
    Prefer FinRL's default INDICATORS if they exist in the data.

    Falls back to numeric non-OHLCV columns if needed.
    """
    indicators = [indicator for indicator in INDICATORS if indicator in df.columns]

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
    }

    indicators = (
        df.drop(columns=list(excluded), errors="ignore")
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )

    if not indicators:
        raise ValueError(
            "Could not infer technical indicators. "
            "Expected FinRL indicators or numeric feature columns."
        )

    return indicators


def build_env_kwargs(
    train_df: pd.DataFrame,
    indicators: list[str],
    initial_amount: float,
    hmax: int,
    buy_cost_pct: float,
    sell_cost_pct: float,
    reward_scaling: float,
) -> dict[str, Any]:
    stock_dim = int(train_df["tic"].nunique())
    state_space = 1 + 2 * stock_dim + len(indicators) * stock_dim

    env_kwargs = {
        "hmax": hmax,
        "initial_amount": initial_amount,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [buy_cost_pct] * stock_dim,
        "sell_cost_pct": [sell_cost_pct] * stock_dim,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": indicators,
        "action_space": stock_dim,
        "reward_scaling": reward_scaling,
    }

    return env_kwargs


def get_model(
    agent_name: str,
    env,
    stock_dim: int,
    device: str,
):
    agent_name = agent_name.lower()

    if agent_name == "a2c":
        return A2C(
            "MlpPolicy",
            env,
            n_steps=5,
            ent_coef=0.01,
            learning_rate=0.0007,
            verbose=1,
            device=device,
        )

    if agent_name == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            n_steps=2048,
            batch_size=128,
            ent_coef=0.01,
            learning_rate=0.00025,
            verbose=1,
            device=device,
        )

    if agent_name == "ddpg":
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
            verbose=1,
            device=device,
        )

    if agent_name == "td3":
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
            verbose=1,
            device=device,
        )

    if agent_name == "sac":
        return SAC(
            "MlpPolicy",
            env,
            batch_size=128,
            buffer_size=100_000,
            learning_rate=0.0001,
            verbose=1,
            device=device,
        )

    raise ValueError(f"Unsupported agent: {agent_name}")


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def build_run_name(
    provided_run_name: str | None,
    agents: list[str],
    total_timesteps: int,
) -> str:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")

    if provided_run_name:
        clean_run_name = provided_run_name.strip().replace(" ", "_")
        return f"{timestamp}_{clean_run_name}"

    agent_part = "all_agents" if len(agents) > 1 else agents[0]

    return (
        f"{timestamp}_run_train_finrl_baselines_"
        f"{agent_part}_timesteps_{total_timesteps}"
    )


def main() -> None:
    args = parse_args()

    dataset_tag = args.dataset_tag.strip()
    selected_agents = [
        agent.strip().lower() for agent in args.agents.split(",") if agent.strip()
    ]

    run_name = build_run_name(
        provided_run_name=args.run_name,
        agents=selected_agents,
        total_timesteps=args.total_timesteps,
    )

    if args.run_root:
        run_paths = build_run_paths(args.run_root)
        output_dir = run_paths.baseline_train_files
        results_root = run_paths.baseline_logs
        trained_models_dir = run_paths.baseline_models
    else:
        output_dir = Path(f"outputs/train_{dataset_tag}") / run_name
        results_root = Path("results") / dataset_tag / run_name
        trained_models_dir = Path("trained_models") / dataset_tag / run_name

    output_dir.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    trained_models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("StockDSS - Train FinRL baselines on PIT data")
    print("=" * 100)
    print(f"Train data:       {args.train_data}")
    print(f"Dataset tag:      {dataset_tag}")
    print(f"Run name:         {run_name}")
    print(f"Agents:           {selected_agents}")
    print(f"Total timesteps:  {args.total_timesteps}")
    print("=" * 100)

    train_df = load_finrl_csv(args.train_data)
    indicators = infer_technical_indicators(train_df)

    env_kwargs = build_env_kwargs(
        train_df=train_df,
        indicators=indicators,
        initial_amount=args.initial_amount,
        hmax=args.hmax,
        buy_cost_pct=args.buy_cost_pct,
        sell_cost_pct=args.sell_cost_pct,
        reward_scaling=args.reward_scaling,
    )

    stock_dim = env_kwargs["stock_dim"]
    state_space = env_kwargs["state_space"]

    print()
    print("Dataset summary")
    print("-" * 100)
    print(f"Rows:             {len(train_df):,}")
    print(f"Date range:       {train_df['date'].min()} -> {train_df['date'].max()}")
    print(f"Stock dimension:  {stock_dim}")
    print(f"State space:      {state_space}")
    print(f"Indicators:       {indicators}")
    print("-" * 100)

    # Save basic metadata before training starts.
    pd.DataFrame({"indicator": indicators}).to_csv(
        output_dir / f"indicators_{dataset_tag}.csv",
        index=False,
    )

    pd.DataFrame({"ticker": sorted(train_df["tic"].unique())}).to_csv(
        output_dir / f"tickers_{dataset_tag}.csv",
        index=False,
    )

    save_json(
        {
            "dataset_tag": dataset_tag,
            "run_name": run_name,
            "train_data": args.train_data,
            "agents": selected_agents,
            "total_timesteps": args.total_timesteps,
            "env_kwargs": env_kwargs,
            "indicators": indicators,
            "rows": int(len(train_df)),
            "date_min": train_df["date"].min(),
            "date_max": train_df["date"].max(),
            "stock_dim": stock_dim,
            "state_space": state_space,
        },
        output_dir / f"train_config_{dataset_tag}.json",
    )

    train_env = StockTradingEnv(df=train_df, **env_kwargs)
    vec_env, _ = train_env.get_sb_env()

    training_rows: list[dict[str, Any]] = []

    for agent_name in selected_agents:
        print()
        print("=" * 100)
        print(f"Training {agent_name.upper()} on PIT dataset {dataset_tag}")
        print("=" * 100)

        start_time = time.time()
        model_path = trained_models_dir / f"agent_{agent_name}.zip"
        log_dir = results_root / agent_name

        row: dict[str, Any] = {
            "dataset_tag": dataset_tag,
            "run_name": run_name,
            "agent": agent_name,
            "enabled": True,
            "total_timesteps": args.total_timesteps,
            "model_path": str(model_path),
            "log_dir": str(log_dir),
            "error": None,
            "duration_seconds": None,
        }

        try:
            model = get_model(
                agent_name=agent_name,
                env=vec_env,
                stock_dim=stock_dim,
                device=args.device,
            )

            model.learn(
                total_timesteps=args.total_timesteps,
                tb_log_name=f"{agent_name}_{dataset_tag}",
                log_interval=100,
                progress_bar=False,
            )

            model.save(model_path)

            duration = time.time() - start_time
            row["duration_seconds"] = duration

            print(f"Finished {agent_name.upper()} in {duration:.2f} seconds")
            print(f"Model saved to: {model_path}")

        except Exception as exc:
            duration = time.time() - start_time
            row["duration_seconds"] = duration
            row["error"] = str(exc)

            print(f"ERROR while training {agent_name.upper()}: {exc}")

        training_rows.append(row)

        pd.DataFrame(training_rows).to_csv(
            output_dir / f"training_summary_{dataset_tag}.csv",
            index=False,
        )

    summary_df = pd.DataFrame(training_rows)

    model_paths_df = summary_df[
        ["dataset_tag", "agent", "model_path", "log_dir", "error"]
    ].copy()

    model_paths_df.to_csv(
        output_dir / f"model_paths_{dataset_tag}.csv",
        index=False,
    )

    save_json(
        {
            "dataset_tag": dataset_tag,
            "run_name": run_name,
            "train_data": args.train_data,
            "output_dir": str(output_dir),
            "trained_models_dir": str(trained_models_dir),
            "results_root": str(results_root),
            "agents": selected_agents,
            "total_timesteps": args.total_timesteps,
            "summary": training_rows,
        },
        output_dir / f"run_summary_{dataset_tag}.json",
    )

    print()
    print("=" * 100)
    print("Training finished")
    print("=" * 100)
    print(summary_df)
    print()
    print(f"Outputs saved in:       {output_dir}")
    print(f"Trained models saved in:{trained_models_dir}")
    print(f"Logs saved in:          {results_root}")


if __name__ == "__main__":
    main()
