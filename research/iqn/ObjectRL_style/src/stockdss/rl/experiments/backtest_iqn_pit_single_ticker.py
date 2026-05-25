"""
Backtest D-IQN-DSS on a PIT trade split for a single ticker.

Purpose:
- Load a trained IQN model.
- Load PIT trade data.
- Run FinRLDiscreteEnv on one ticker.
- At each decision step, extract IQN quantile estimates per action.
- Compute q10/q25/q50/q75/q90/CVaR10 per action.
- Choose action using a risk-adjusted score.
- Save decision log, account values, action estimates, metrics, and plots.

Example:

PowerShell:
    $env:PYTHONPATH="src"

    python -m stockdss.rl.experiments.backtest_iqn_pit_single_ticker `
      --trade-data data/trade_data_pit_500_2026_01_01.csv `
      --dataset-tag pit_500_2026_01_01 `
      --run-name 2026_05_14_0145_run_backtest_iqn_pit_single_ticker_aapl `
      --ticker AAPL `
      --model-path trained_models/iqn/iqn_agent.pt `
      --risk-lambda 0.75 `
      --show

If you do not yet have a saved IQN model, use:

    --allow-random-model

This is only for pipeline testing. It is NOT a valid trained result.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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

ACTION_NAMES = [
    "HOLD",
    "BUY_25",
    "BUY_50",
    "BUY_100",
    "SELL_25",
    "SELL_50",
    "SELL_100",
]


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest IQN on PIT trade data for a single ticker."
    )

    parser.add_argument(
        "--trade-data",
        required=True,
        help="Path to PIT trade CSV, e.g. data/trade_data_pit_500_2026_01_01.csv",
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
        help="Ticker to backtest. Default: AAPL",
    )

    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to trained IQN checkpoint/model .pt/.pth file.",
    )

    parser.add_argument(
        "--allow-random-model",
        action="store_true",
        help=(
            "Allow running with randomly initialized IQN model if no model path is found. "
            "Only for pipeline testing."
        ),
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
        "--risk-lambda",
        type=float,
        default=0.75,
        help="Risk penalty. Score = q50 - risk_lambda * abs(CVaR10). Default: 0.75",
    )

    parser.add_argument(
        "--num-quantiles",
        type=int,
        default=128,
        help="Number of tau samples used for backtest decision extraction. Default: 128",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max number of backtest steps for quick testing.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Override device, e.g. cpu or cuda. Default: use IQNConfig default.",
    )

    parser.add_argument(
        "--run-root",
        default=None,
        help=(
            "Optional central runner output folder. "
            "If provided, files are written to iqn_finrl/files/backtest "
            "and plots are written to iqn_finrl/plots."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show matplotlib pop-up windows.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def build_run_name(provided_run_name: str | None, ticker: str) -> str:
    if provided_run_name:
        return provided_run_name.strip().replace(" ", "_")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")

    return f"{timestamp}_run_backtest_iqn_pit_single_ticker_" f"{ticker.lower()}"


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def load_checkpoint_into_agent(
    agent: IQNAgent,
    model_path: Path,
    device: str,
) -> None:
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "online_net_state_dict" in checkpoint:
            agent.online_net.load_state_dict(checkpoint["online_net_state_dict"])
            agent.target_net.load_state_dict(checkpoint["online_net_state_dict"])
            return

        if "model_state_dict" in checkpoint:
            agent.online_net.load_state_dict(checkpoint["model_state_dict"])
            agent.target_net.load_state_dict(checkpoint["model_state_dict"])
            return

        if "state_dict" in checkpoint:
            agent.online_net.load_state_dict(checkpoint["state_dict"])
            agent.target_net.load_state_dict(checkpoint["state_dict"])
            return

        # If the dict itself looks like a raw PyTorch state_dict.
        if all(isinstance(key, str) for key in checkpoint.keys()):
            try:
                agent.online_net.load_state_dict(checkpoint)
                agent.target_net.load_state_dict(checkpoint)
                return
            except RuntimeError:
                pass

    raise ValueError(
        f"Could not load IQN checkpoint from {model_path}. "
        "Expected keys: online_net_state_dict, model_state_dict, state_dict, "
        "or a raw PyTorch state_dict."
    )


@torch.no_grad()
def get_action_quantile_values(
    agent: IQNAgent,
    observation: np.ndarray,
    num_quantiles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        taus: shape [num_quantiles]
        quantile_values: shape [num_quantiles, action_dim]
    """
    device = agent.device

    state_t = torch.tensor(
        observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    taus_t = torch.linspace(
        0.01,
        0.99,
        steps=num_quantiles,
        device=device,
    ).unsqueeze(0)

    quantile_values_t = agent.online_net(state_t, taus_t)
    # shape: [1, num_quantiles, action_dim]

    quantile_values = quantile_values_t.squeeze(0).detach().cpu().numpy()
    taus = taus_t.squeeze(0).detach().cpu().numpy()

    return taus, quantile_values


def compute_action_estimates(
    quantile_values: np.ndarray,
    risk_lambda: float,
) -> pd.DataFrame:
    """
    quantile_values:
        shape [num_quantiles, action_dim]

    Assumption:
        IQN outputs are in reward units.
        If the environment reward is percentage return, these values are interpreted
        as percentage-return-like decision estimates.
    """
    rows = []

    action_dim = quantile_values.shape[1]

    for action_index in range(action_dim):
        values = quantile_values[:, action_index].astype(float)

        q10 = float(np.quantile(values, 0.10))
        q25 = float(np.quantile(values, 0.25))
        q50 = float(np.quantile(values, 0.50))
        q75 = float(np.quantile(values, 0.75))
        q90 = float(np.quantile(values, 0.90))

        downside = values[values <= q10]
        cvar10 = float(downside.mean()) if len(downside) else q10

        expected_value = float(values.mean())
        risk_adjusted_score = q50 - risk_lambda * abs(cvar10)

        action_name = (
            ACTION_NAMES[action_index]
            if action_index < len(ACTION_NAMES)
            else f"ACTION_{action_index}"
        )

        rows.append(
            {
                "action_index": action_index,
                "action": action_name,
                "expected_value": expected_value,
                "q10": q10,
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "q90": q90,
                "cvar10": cvar10,
                "risk_adjusted_score": risk_adjusted_score,
            }
        )

    estimates = pd.DataFrame(rows)

    estimates["selected_risk_adjusted"] = False
    estimates["selected_risk_neutral"] = False

    risk_adjusted_idx = estimates["risk_adjusted_score"].idxmax()
    risk_neutral_idx = estimates["expected_value"].idxmax()

    estimates.loc[risk_adjusted_idx, "selected_risk_adjusted"] = True
    estimates.loc[risk_neutral_idx, "selected_risk_neutral"] = True

    return estimates


def choose_action_from_estimates(estimates: pd.DataFrame) -> int:
    selected = estimates[estimates["selected_risk_adjusted"]].iloc[0]
    return int(selected["action_index"])


def flatten_estimates_for_log(estimates: pd.DataFrame) -> dict[str, float]:
    values: dict[str, float] = {}

    for _, row in estimates.iterrows():
        prefix = str(row["action"]).lower()

        values[f"{prefix}_expected_value"] = float(row["expected_value"])
        values[f"{prefix}_q10"] = float(row["q10"])
        values[f"{prefix}_q25"] = float(row["q25"])
        values[f"{prefix}_q50"] = float(row["q50"])
        values[f"{prefix}_q75"] = float(row["q75"])
        values[f"{prefix}_q90"] = float(row["q90"])
        values[f"{prefix}_cvar10"] = float(row["cvar10"])
        values[f"{prefix}_risk_adjusted_score"] = float(row["risk_adjusted_score"])

    return values


def build_metrics(account_values: pd.DataFrame, initial_amount: float) -> pd.DataFrame:
    series = account_values["portfolio_value"].dropna().astype(float)

    if series.empty:
        return pd.DataFrame()

    start_value = float(series.iloc[0])
    end_value = float(series.iloc[-1])

    daily_returns = series.pct_change().dropna()

    running_max = series.cummax()
    drawdown = series / running_max - 1.0

    sharpe = (
        float(np.sqrt(252) * daily_returns.mean() / daily_returns.std())
        if len(daily_returns) and daily_returns.std() != 0
        else np.nan
    )

    return pd.DataFrame(
        [
            {
                "strategy": "d_iqn_dss",
                "start_value": start_value,
                "end_value": end_value,
                "profit_loss": end_value - start_value,
                "total_return_pct": (end_value / start_value - 1.0) * 100,
                "max_drawdown_pct": float(drawdown.min()) * 100,
                "annualized_sharpe": sharpe,
                "days": int(len(series)),
                "ended_above_initial": bool(end_value > initial_amount),
            }
        ]
    )


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_account_values(account_values: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(
        pd.to_datetime(account_values["date"]),
        account_values["portfolio_value"],
        label="D-IQN-DSS",
    )
    plt.axhline(
        float(account_values["portfolio_value"].iloc[0]),
        linestyle="--",
        linewidth=1,
        label="initial value",
    )
    plt.title("D-IQN-DSS PIT Backtest - Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "iqn_account_values.png", dpi=150)


def plot_actions(decision_log: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(14, 5))

    action_to_index = {name: index for index, name in enumerate(ACTION_NAMES)}
    y = decision_log["chosen_action"].map(action_to_index)

    plt.scatter(
        pd.to_datetime(decision_log["date"]),
        y,
        s=18,
    )

    plt.yticks(list(action_to_index.values()), list(action_to_index.keys()))
    plt.title("D-IQN-DSS PIT Backtest - Chosen Actions")
    plt.xlabel("Date")
    plt.ylabel("Action")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "iqn_chosen_actions.png", dpi=150)


def plot_last_day_quantile_functions(
    last_estimates: pd.DataFrame,
    output_dir: Path,
) -> None:
    taus = np.array([0.10, 0.25, 0.50, 0.75, 0.90])

    plt.figure(figsize=(14, 6))

    for _, row in last_estimates.iterrows():
        values = np.array(
            [
                row["q10"],
                row["q25"],
                row["q50"],
                row["q75"],
                row["q90"],
            ],
            dtype=float,
        )

        linewidth = 3 if bool(row["selected_risk_adjusted"]) else 1.5
        alpha = 1.0 if bool(row["selected_risk_adjusted"]) else 0.45

        plt.plot(
            taus,
            values,
            marker="o",
            label=row["action"],
            linewidth=linewidth,
            alpha=alpha,
        )

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.title("D-IQN-DSS - Last Decision Quantile Function per Action")
    plt.xlabel("τ")
    plt.ylabel("Estimated return / value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "iqn_last_decision_quantile_functions.png", dpi=150)


def plot_last_day_scores(last_estimates: pd.DataFrame, output_dir: Path) -> None:
    sorted_df = last_estimates.sort_values("risk_adjusted_score", ascending=True)

    plt.figure(figsize=(12, 6))
    plt.barh(sorted_df["action"], sorted_df["risk_adjusted_score"])

    for index, (_, row) in enumerate(sorted_df.iterrows()):
        marker = " selected" if row["selected_risk_adjusted"] else ""
        plt.text(
            row["risk_adjusted_score"],
            index,
            f" {row['risk_adjusted_score']:.4f}{marker}",
            va="center",
        )

    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.title("D-IQN-DSS - Last Decision Risk-adjusted Action Score")
    plt.xlabel("Risk-adjusted score")
    plt.ylabel("Action")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "iqn_last_decision_scores.png", dpi=150)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    dataset_tag = args.dataset_tag.strip()
    ticker = args.ticker.strip().upper()
    run_name = build_run_name(args.run_name, ticker=ticker)

    if args.run_root:
        run_paths = build_run_paths(args.run_root)
        output_dir = run_paths.iqn_backtest_files
        plots_dir = run_paths.iqn_plots
    else:
        output_dir = Path(f"outputs/backtest_iqn_{dataset_tag}") / run_name
        plots_dir = output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    env_config = FinRLDiscreteEnvConfig(
        csv_path=args.trade_data,
        ticker=ticker,
        initial_amount=args.initial_amount,
        buy_cost_pct=args.buy_cost_pct,
        sell_cost_pct=args.sell_cost_pct,
    )

    env = FinRLDiscreteEnv(env_config)
    observation, info = env.reset(seed=42)

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    iqn_config = IQNConfig()

    if args.device:
        iqn_config.device = args.device

    agent = IQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=iqn_config,
    )

    model_path = Path(args.model_path) if args.model_path else None

    if model_path and model_path.exists():
        load_checkpoint_into_agent(
            agent=agent,
            model_path=model_path,
            device=agent.device,
        )
        model_status = "loaded_trained_model"

    elif args.allow_random_model:
        print(
            "WARNING: Running with randomly initialized IQN model. "
            "This is only for pipeline testing."
        )
        model_status = "random_model_pipeline_test"

    else:
        raise FileNotFoundError(
            "No trained IQN model found. Provide --model-path or use "
            "--allow-random-model for pipeline testing only."
        )

    print("=" * 100)
    print("D-IQN-DSS PIT single-ticker backtest")
    print("=" * 100)
    print(f"Trade data:       {args.trade_data}")
    print(f"Dataset tag:      {dataset_tag}")
    print(f"Run name:         {run_name}")
    print(f"Ticker:           {ticker}")
    print(f"Model status:     {model_status}")
    print(f"Model path:       {model_path}")
    print(f"Output dir:       {output_dir}")
    print(f"Plots dir:        {plots_dir}")
    print(f"Run root:         {args.run_root}")
    print(f"State dim:        {state_dim}")
    print(f"Action dim:       {action_dim}")
    print(f"Action space:     {env.action_space}")
    print(f"Observation space:{env.observation_space}")
    print(f"Device:           {agent.device}")
    print(f"Risk lambda:      {args.risk_lambda}")
    print("=" * 100)

    decision_rows = []
    account_rows = []
    all_estimate_rows = []

    terminated = False
    truncated = False
    step = 0

    while not terminated and not truncated:
        taus, quantile_values = get_action_quantile_values(
            agent=agent,
            observation=observation,
            num_quantiles=args.num_quantiles,
        )

        estimates = compute_action_estimates(
            quantile_values=quantile_values,
            risk_lambda=args.risk_lambda,
        )

        action = choose_action_from_estimates(estimates)
        chosen_action_name = str(
            estimates.loc[
                estimates["action_index"] == action,
                "action",
            ].iloc[0]
        )

        risk_neutral_action = str(
            estimates[estimates["selected_risk_neutral"]].iloc[0]["action"]
        )

        next_observation, reward, terminated, truncated, next_info = env.step(action)

        flat_estimates = flatten_estimates_for_log(estimates)

        decision_row = {
            "step": step,
            "date": next_info["date"],
            "ticker": ticker,
            "price": next_info.get("price"),
            "chosen_action_index": action,
            "chosen_action": chosen_action_name,
            "risk_neutral_action": risk_neutral_action,
            "reward": reward,
            "raw_reward": next_info.get("raw_reward"),
            "transaction_cost": next_info.get("transaction_cost"),
            "portfolio_value": next_info.get("portfolio_value"),
            "cash": next_info.get("cash"),
            "shares_held": next_info.get("shares_held"),
            "risk_lambda": args.risk_lambda,
            **flat_estimates,
        }

        decision_rows.append(decision_row)

        account_rows.append(
            {
                "step": step,
                "date": next_info["date"],
                "portfolio_value": next_info.get("portfolio_value"),
                "cash": next_info.get("cash"),
                "shares_held": next_info.get("shares_held"),
                "price": next_info.get("price"),
                "chosen_action": chosen_action_name,
            }
        )

        estimates_for_step = estimates.copy()
        estimates_for_step["step"] = step
        estimates_for_step["date"] = next_info["date"]
        estimates_for_step["ticker"] = ticker
        estimates_for_step["price"] = next_info.get("price")
        estimates_for_step["chosen_action"] = chosen_action_name
        estimates_for_step["risk_neutral_action"] = risk_neutral_action
        all_estimate_rows.append(estimates_for_step)

        observation = next_observation
        step += 1

        if args.max_steps is not None and step >= args.max_steps:
            break

    env.close()

    decision_log = pd.DataFrame(decision_rows)
    account_values = pd.DataFrame(account_rows)

    if all_estimate_rows:
        all_estimates = pd.concat(all_estimate_rows, ignore_index=True)
        last_step = int(all_estimates["step"].max())
        last_estimates = all_estimates[all_estimates["step"] == last_step].copy()
    else:
        all_estimates = pd.DataFrame()
        last_estimates = pd.DataFrame()

    metrics = build_metrics(
        account_values=account_values,
        initial_amount=args.initial_amount,
    )

    decision_log.to_csv(output_dir / "iqn_decision_log.csv", index=False)
    account_values.to_csv(output_dir / "iqn_account_values.csv", index=False)
    all_estimates.to_csv(output_dir / "iqn_action_estimates_all_steps.csv", index=False)
    last_estimates.to_csv(output_dir / "iqn_action_estimates_last_day.csv", index=False)
    metrics.to_csv(output_dir / "iqn_backtest_metrics.csv", index=False)

    save_json(
        {
            "dataset_tag": dataset_tag,
            "run_name": run_name,
            "trade_data": args.trade_data,
            "ticker": ticker,
            "model_path": str(model_path) if model_path else None,
            "model_status": model_status,
            "output_dir": str(output_dir),
            "plots_dir": str(plots_dir),
            "run_root": args.run_root,
            "initial_amount": args.initial_amount,
            "buy_cost_pct": args.buy_cost_pct,
            "sell_cost_pct": args.sell_cost_pct,
            "risk_lambda": args.risk_lambda,
            "num_quantiles": args.num_quantiles,
            "max_steps": args.max_steps,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "device": agent.device,
        },
        output_dir / "iqn_backtest_config.json",
    )

    if not account_values.empty and not last_estimates.empty:
        plot_account_values(account_values, plots_dir)
        plot_actions(decision_log, plots_dir)
        plot_last_day_quantile_functions(last_estimates, plots_dir)
        plot_last_day_scores(last_estimates, plots_dir)

    print()
    print("=" * 100)
    print("Backtest finished")
    print("=" * 100)

    if not metrics.empty:
        print(metrics.to_string(index=False))

    print()
    print(f"Saved outputs to: {output_dir.resolve()}")
    print(f"Saved plots to:   {plots_dir.resolve()}")
    print("Key files:")
    print("- iqn_decision_log.csv")
    print("- iqn_account_values.csv")
    print("- iqn_action_estimates_all_steps.csv")
    print("- iqn_action_estimates_last_day.csv")
    print("- iqn_backtest_metrics.csv")
    print("- iqn_backtest_config.json")
    print("- iqn_account_values.png")
    print("- iqn_chosen_actions.png")
    print("- iqn_last_decision_quantile_functions.png")
    print("- iqn_last_decision_scores.png")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
