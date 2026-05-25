"""
Train D-IQN-DSS on FinRL-style PIT data for a single ticker.

Purpose:
- Use point-in-time train data.
- Use FinRLDiscreteEnv.
- Train the custom PyTorch IQN agent.
- Save trained IQN checkpoint.
- Save training log, episode summary, and config.

Example:

PowerShell:
    $env:PYTHONPATH="src"

    python -m stockdss.rl.experiments.train_iqn_finrl_pit_single_ticker `
      --train-data data/train_data_pit_500_2026_01_01.csv `
      --dataset-tag pit_500_2026_01_01 `
      --run-name 2026_05_14_0205_run_train_iqn_finrl_pit_single_ticker_aapl_timesteps_5000 `
      --ticker AAPL `
      --total-steps 5000
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from stockdss.envs.finrl_discrete_env import (
    FinRLDiscreteEnv,
    FinRLDiscreteEnvConfig,
)
from stockdss.rl.agents.iqn_agent import IQNAgent
from stockdss.rl.config.iqn_config import IQNConfig
from stockdss.runner.run_paths import build_run_paths

# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train custom IQN on FinRL-style PIT single-ticker data."
    )

    parser.add_argument(
        "--train-data",
        required=True,
        help="Path to PIT train CSV, e.g. data/train_data_pit_500_2026_01_01.csv",
    )

    parser.add_argument(
        "--dataset-tag",
        required=True,
        help="Dataset tag, e.g. pit_500_2026_01_01",
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help="Readable run name. If omitted, one is generated.",
    )

    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker to train on. Default: AAPL",
    )

    parser.add_argument(
        "--total-steps",
        type=int,
        default=50_000,
        help="Total environment steps. Default: 50000",
    )

    parser.add_argument(
        "--learning-starts",
        type=int,
        default=1_000,
        help="Steps before learning starts. Default: 1000",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Replay batch size. Default: 64",
    )

    parser.add_argument(
        "--initial-amount",
        type=float,
        default=1_000_000.0,
        help="Initial portfolio amount. Default: 1000000",
    )

    parser.add_argument(
        "--buy-cost-pct",
        type=float,
        default=0.01,
        help="Buy transaction cost. Default: 0.01",
    )

    parser.add_argument(
        "--sell-cost-pct",
        type=float,
        default=0.01,
        help="Sell transaction cost. Default: 0.01",
    )

    parser.add_argument(
        "--reward-scaling",
        type=float,
        default=1e-4,
        help=(
            "Reward scaling in FinRLDiscreteEnv config. "
            "Currently only used if env uses scaled raw rewards."
        ),
    )

    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=500,
        help="Target network update interval. Default: 500",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=1_000,
        help="Console log interval. Default: 1000",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Override device, e.g. cpu or cuda.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Optional checkpoint interval. 0 disables intermediate checkpoints.",
    )

    parser.add_argument(
        "--run-root",
        default=None,
        help=(
            "Optional central runner output folder. "
            "If provided, training files are written to iqn_finrl/files/train and "
            "the model is written to iqn_finrl/models."
        ),
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def build_run_name(
    provided_run_name: str | None,
    ticker: str,
    total_steps: int,
) -> str:
    if provided_run_name:
        return provided_run_name.strip().replace(" ", "_")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")

    return (
        f"{timestamp}_run_train_iqn_finrl_pit_single_ticker_"
        f"{ticker.lower()}_timesteps_{total_steps}"
    )


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def save_checkpoint(
    agent: IQNAgent,
    config: IQNConfig,
    output_path: Path,
    step: int,
    episode_count: int,
    dataset_tag: str,
    run_name: str,
    ticker: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "online_net_state_dict": agent.online_net.state_dict(),
        "target_net_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "config": config.__dict__,
        "step": step,
        "episode_count": episode_count,
        "dataset_tag": dataset_tag,
        "run_name": run_name,
        "ticker": ticker,
        "model_type": "D-IQN-DSS",
        "framework": "PyTorch",
    }

    torch.save(checkpoint, output_path)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_training_metrics(
    training_log: pd.DataFrame,
    episode_log: pd.DataFrame,
) -> pd.DataFrame:
    if training_log.empty:
        return pd.DataFrame()

    final_row = training_log.iloc[-1]

    if episode_log.empty:
        final_episode_reward = np.nan
        mean_episode_reward_20 = np.nan
        episode_count = 0
    else:
        final_episode_reward = float(episode_log["episode_reward"].iloc[-1])
        mean_episode_reward_20 = float(episode_log["episode_reward"].tail(20).mean())
        episode_count = int(len(episode_log))

    return pd.DataFrame(
        [
            {
                "final_step": int(final_row["step"]),
                "episodes": episode_count,
                "final_portfolio_value": float(final_row["portfolio_value"]),
                "final_cash": float(final_row["cash"]),
                "final_shares_held": int(final_row["shares_held"]),
                "final_epsilon": float(final_row["epsilon"]),
                "final_loss_100": float(final_row["loss_100"]),
                "final_episode_reward": final_episode_reward,
                "mean_episode_reward_20": mean_episode_reward_20,
            }
        ]
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    dataset_tag = args.dataset_tag.strip()
    ticker = args.ticker.strip().upper()
    run_name = build_run_name(
        provided_run_name=args.run_name,
        ticker=ticker,
        total_steps=args.total_steps,
    )

    if args.run_root:
        run_paths = build_run_paths(args.run_root)
        output_dir = run_paths.iqn_train_files
        model_dir = run_paths.iqn_models
    else:
        output_dir = Path(f"outputs/train_iqn_{dataset_tag}") / run_name
        model_dir = Path("trained_models") / "iqn" / dataset_tag / run_name

    model_path = model_dir / "iqn_agent.pt"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    set_seeds(args.seed)

    env_config = FinRLDiscreteEnvConfig(
        csv_path=args.train_data,
        ticker=ticker,
        initial_amount=args.initial_amount,
        buy_cost_pct=args.buy_cost_pct,
        sell_cost_pct=args.sell_cost_pct,
        reward_scaling=args.reward_scaling,
    )

    env = FinRLDiscreteEnv(env_config)
    state, info = env.reset(seed=args.seed)

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    iqn_config = IQNConfig()
    iqn_config.env_name = f"FinRLDiscreteEnv-{ticker}-PIT-v1"
    iqn_config.seed = args.seed
    iqn_config.total_steps = args.total_steps
    iqn_config.learning_starts = args.learning_starts
    iqn_config.batch_size = args.batch_size
    iqn_config.target_update_interval = args.target_update_interval
    iqn_config.log_interval = args.log_interval

    if args.device:
        iqn_config.device = args.device

    agent = IQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=iqn_config,
    )

    print("=" * 100)
    print("D-IQN-DSS - Train IQN on FinRL-style PIT single-ticker data")
    print("=" * 100)
    print(f"Train data:       {args.train_data}")
    print(f"Dataset tag:      {dataset_tag}")
    print(f"Run name:         {run_name}")
    print(f"Ticker:           {ticker}")
    print(f"Output dir:       {output_dir}")
    print(f"Model dir:        {model_dir}")
    print(f"Model path:       {model_path}")
    print(f"Run root:         {args.run_root}")
    print(f"Initial amount:   {args.initial_amount:,.2f}")
    print(f"Buy cost pct:     {args.buy_cost_pct}")
    print(f"Sell cost pct:    {args.sell_cost_pct}")
    print(f"State dim:        {state_dim}")
    print(f"Action dim:       {action_dim}")
    print(f"Action space:     {env.action_space}")
    print(f"Observation space:{env.observation_space}")
    print(f"Device:           {agent.device}")
    print(f"Total steps:      {args.total_steps}")
    print("=" * 100)

    training_rows = []
    episode_rows = []
    losses = []

    episode_reward = 0.0
    episode_start_value = float(info["portfolio_value"])
    episode_count = 0

    for step in range(1, args.total_steps + 1):
        action = agent.select_action(state, step)

        next_state, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.add(
            state=state,
            action=int(action),
            reward=float(reward),
            next_state=next_state,
            done=bool(done),
        )

        state = next_state
        episode_reward += float(reward)

        loss = np.nan

        if (
            step > iqn_config.learning_starts
            and len(agent.replay_buffer) >= iqn_config.batch_size
        ):
            loss = agent.learn()
            losses.append(loss)

        if step % iqn_config.target_update_interval == 0:
            agent.update_target_network()

        if done:
            episode_count += 1

            episode_rows.append(
                {
                    "episode": episode_count,
                    "step": step,
                    "episode_reward": episode_reward,
                    "episode_start_value": episode_start_value,
                    "episode_end_value": float(next_info["portfolio_value"]),
                    "episode_return_pct": (
                        float(next_info["portfolio_value"]) / episode_start_value - 1.0
                    )
                    * 100,
                    "date": next_info["date"],
                }
            )

            state, info = env.reset()
            episode_reward = 0.0
            episode_start_value = float(info["portfolio_value"])

        else:
            info = next_info

        if step % iqn_config.log_interval == 0 or step == args.total_steps:
            recent_episode_rewards = [
                row["episode_reward"] for row in episode_rows[-20:]
            ]

            mean_reward_20 = (
                float(np.mean(recent_episode_rewards))
                if recent_episode_rewards
                else 0.0
            )

            loss_100 = float(np.mean(losses[-100:])) if losses else 0.0

            action_name = next_info.get("action_name", str(action))

            row = {
                "step": step,
                "episodes": episode_count,
                "date": next_info.get("date"),
                "ticker": ticker,
                "action": int(action),
                "action_name": action_name,
                "reward": float(reward),
                "raw_reward": next_info.get("raw_reward"),
                "transaction_cost": next_info.get("transaction_cost"),
                "portfolio_value": float(next_info["portfolio_value"]),
                "cash": float(next_info["cash"]),
                "shares_held": int(next_info["shares_held"]),
                "epsilon": float(agent.epsilon(step)),
                "loss": float(loss) if not np.isnan(loss) else np.nan,
                "loss_100": loss_100,
                "mean_reward_20": mean_reward_20,
            }

            training_rows.append(row)

            print(
                f"step={step:>7} | "
                f"episodes={episode_count:>4} | "
                f"mean_reward_20={mean_reward_20:>10.4f} | "
                f"epsilon={agent.epsilon(step):.3f} | "
                f"loss_100={loss_100:>10.6f} | "
                f"date={next_info.get('date')} | "
                f"action={action_name:<8} | "
                f"portfolio={float(next_info['portfolio_value']):>12.2f} | "
                f"cash={float(next_info['cash']):>12.2f} | "
                f"shares={int(next_info['shares_held']):>8}"
            )

        if args.save_every and step % args.save_every == 0:
            checkpoint_path = model_dir / f"iqn_agent_step_{step}.pt"
            save_checkpoint(
                agent=agent,
                config=iqn_config,
                output_path=checkpoint_path,
                step=step,
                episode_count=episode_count,
                dataset_tag=dataset_tag,
                run_name=run_name,
                ticker=ticker,
            )

    save_checkpoint(
        agent=agent,
        config=iqn_config,
        output_path=model_path,
        step=args.total_steps,
        episode_count=episode_count,
        dataset_tag=dataset_tag,
        run_name=run_name,
        ticker=ticker,
    )

    training_log = pd.DataFrame(training_rows)
    episode_log = pd.DataFrame(episode_rows)
    metrics = build_training_metrics(
        training_log=training_log,
        episode_log=episode_log,
    )

    training_log.to_csv(output_dir / "iqn_training_log.csv", index=False)
    episode_log.to_csv(output_dir / "iqn_episode_log.csv", index=False)
    metrics.to_csv(output_dir / "iqn_training_metrics.csv", index=False)

    save_json(
        {
            "dataset_tag": dataset_tag,
            "run_name": run_name,
            "train_data": args.train_data,
            "ticker": ticker,
            "model_path": str(model_path),
            "output_dir": str(output_dir),
            "model_dir": str(model_dir),
            "run_root": args.run_root,
            "initial_amount": args.initial_amount,
            "buy_cost_pct": args.buy_cost_pct,
            "sell_cost_pct": args.sell_cost_pct,
            "reward_scaling": args.reward_scaling,
            "total_steps": args.total_steps,
            "learning_starts": args.learning_starts,
            "batch_size": args.batch_size,
            "target_update_interval": args.target_update_interval,
            "log_interval": args.log_interval,
            "seed": args.seed,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "device": agent.device,
            "iqn_config": iqn_config.__dict__,
        },
        output_dir / "iqn_training_config.json",
    )

    print()
    print("=" * 100)
    print("Training finished")
    print("=" * 100)

    if not metrics.empty:
        print(metrics.to_string(index=False))

    print()
    print(f"Saved model:       {model_path}")
    print(f"Saved outputs to:  {output_dir.resolve()}")
    print("Key files:")
    print("- iqn_agent.pt")
    print("- iqn_training_log.csv")
    print("- iqn_episode_log.csv")
    print("- iqn_training_metrics.csv")
    print("- iqn_training_config.json")


if __name__ == "__main__":
    main()
