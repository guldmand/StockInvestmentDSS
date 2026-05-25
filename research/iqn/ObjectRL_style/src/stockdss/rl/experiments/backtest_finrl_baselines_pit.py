"""
Backtest FinRL baseline agents on a point-in-time trade split.

This script belongs to the StockDSS codebase.

Purpose:
- Load PIT train/trade data.
- Load trained FinRL/SB3 baseline models from a specific run folder.
- Backtest A2C, DDPG, PPO, TD3, SAC.
- Save account values, actions, metrics, MVO baseline, optional DJI baseline, and plot.

Example:

PowerShell:
    $env:PYTHONPATH="src"

    python -m stockdss.rl.experiments.backtest_finrl_baselines_pit `
        --train-data data/train_data_pit_500_2026_01_01.csv `
        --trade-data data/trade_data_pit_500_2026_01_01.csv `
        --dataset-tag pit_500_2026_01_01 `
        --run-name 2026_05_14_0040_run_train_finrl_baselines_smoketest_all_agents_timesteps_500 `
        --agents a2c,ddpg,ppo,td3,sac
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from stockdss.runner.run_paths import build_run_paths

DEFAULT_INITIAL_AMOUNT = 1_000_000


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest FinRL baseline agents on PIT trade data."
    )

    parser.add_argument(
        "--train-data",
        required=True,
        help="Path to PIT train CSV, e.g. data/train_data_pit_500_2026_01_01.csv",
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
        required=True,
        help="Run folder name containing trained models.",
    )

    parser.add_argument(
        "--agents",
        default="a2c,ddpg,ppo,td3,sac",
        help="Comma-separated agents to backtest. Default: a2c,ddpg,ppo,td3,sac",
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
        "--use-mvo",
        action="store_true",
        help="Enable MVO baseline.",
    )

    parser.add_argument(
        "--use-dji",
        action="store_true",
        help="Enable DJI baseline download from Yahoo Finance.",
    )

    parser.add_argument(
        "--run-root",
        default=None,
        help=(
            "Optional central runner output folder. "
            "If provided, models are read from baseline_finrl/models and "
            "backtest outputs are written to baseline_finrl/files/backtest."
        ),
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------


def load_finrl_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    unnamed_columns = [
        col for col in df.columns if str(col).lower().startswith("unnamed")
    ]

    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    required = {"date", "tic", "close"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"CSV file missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["tic"] = df["tic"].astype(str)

    df = df.sort_values(["date", "tic"]).copy()

    # Critical FinRL requirement:
    # StockTradingEnv expects all tickers for the same trading day to share
    # the same integer index. Then df.loc[day] returns all stocks for that day.
    df.index = pd.factorize(df["date"])[0]
    df.index.names = [""]

    return df


def infer_technical_indicators(df: pd.DataFrame) -> list[str]:
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


def process_df_for_mvo(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot(index="date", columns="tic", values="close")


def stock_returns_computing(
    stock_price: np.ndarray,
    rows: int,
    columns: int,
) -> np.ndarray:
    stock_return = np.zeros([rows - 1, columns])

    for col_index in range(columns):
        for row_index in range(rows - 1):
            stock_return[row_index, col_index] = (
                (
                    stock_price[row_index + 1, col_index]
                    - stock_price[row_index, col_index]
                )
                / stock_price[row_index, col_index]
            ) * 100

    return stock_return


def build_env_kwargs(
    trade_df: pd.DataFrame,
    indicators: list[str],
    initial_amount: float,
    hmax: int,
    buy_cost_pct: float,
    sell_cost_pct: float,
    reward_scaling: float,
) -> dict[str, Any]:
    stock_dim = int(trade_df["tic"].nunique())
    state_space = 1 + 2 * stock_dim + len(indicators) * stock_dim

    return {
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


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------


MODEL_CLASSES = {
    "a2c": A2C,
    "ddpg": DDPG,
    "ppo": PPO,
    "td3": TD3,
    "sac": SAC,
}


def load_trained_models(
    agents: list[str],
    model_dir: Path,
) -> dict[str, Any]:
    trained_models: dict[str, Any] = {}

    for agent_name in agents:
        agent_name = agent_name.lower()

        if agent_name not in MODEL_CLASSES:
            raise ValueError(f"Unsupported agent: {agent_name}")

        model_path = model_dir / f"agent_{agent_name}.zip"

        if not model_path.exists():
            print(f"WARNING: Model not found, skipping {agent_name}: {model_path}")
            trained_models[agent_name] = None
            continue

        trained_models[agent_name] = MODEL_CLASSES[agent_name].load(model_path)

    return trained_models


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def save_agent_outputs(
    output_dir: Path,
    agent_name: str,
    account_value: pd.DataFrame | None,
    actions: pd.DataFrame | None,
) -> None:
    if account_value is not None:
        account_value.to_csv(
            output_dir / f"account_values_{agent_name}.csv",
            index=False,
        )

    if actions is not None:
        actions.to_csv(
            output_dir / f"actions_{agent_name}.csv",
            index=True,
        )


def build_metrics(
    result: pd.DataFrame,
    initial_amount: float = DEFAULT_INITIAL_AMOUNT,
) -> pd.DataFrame:
    rows = []

    for name in result.columns:
        series = result[name].dropna().astype(float)

        if series.empty:
            continue

        start_value = float(series.iloc[0])
        end_value = float(series.iloc[-1])
        total_return = end_value / start_value - 1
        daily_returns = series.pct_change().dropna()
        volatility = float(daily_returns.std()) if len(daily_returns) else np.nan

        sharpe = (
            float(np.sqrt(252) * daily_returns.mean() / daily_returns.std())
            if len(daily_returns) and daily_returns.std() != 0
            else np.nan
        )

        running_max = series.cummax()
        drawdown = series / running_max - 1
        max_drawdown = float(drawdown.min())

        best_day = float(daily_returns.max()) if len(daily_returns) else np.nan
        worst_day = float(daily_returns.min()) if len(daily_returns) else np.nan

        rows.append(
            {
                "strategy": name,
                "start_value": start_value,
                "end_value": end_value,
                "profit_loss": end_value - start_value,
                "total_return_pct": total_return * 100,
                "max_drawdown_pct": max_drawdown * 100,
                "daily_volatility_pct": (
                    volatility * 100 if not np.isnan(volatility) else np.nan
                ),
                "annualized_sharpe": sharpe,
                "best_day_pct": best_day * 100 if not np.isnan(best_day) else np.nan,
                "worst_day_pct": worst_day * 100 if not np.isnan(worst_day) else np.nan,
                "days": int(len(series)),
                "ended_above_initial": bool(end_value > initial_amount),
            }
        )

    metrics = pd.DataFrame(rows)

    if metrics.empty:
        return metrics

    return metrics.sort_values("end_value", ascending=False)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def print_explanation(metrics: pd.DataFrame) -> None:
    print("\n=== Human-readable summary ===")

    if metrics.empty:
        print("No metrics available.")
        return

    print(
        "Each curve starts near the initial amount. "
        "Above initial amount means profit; below means loss."
    )
    print(
        "The best strategy in this specific trade period is the row with highest end_value.\n"
    )

    display_cols = [
        "strategy",
        "end_value",
        "profit_loss",
        "total_return_pct",
        "max_drawdown_pct",
        "annualized_sharpe",
        "ended_above_initial",
    ]

    print(metrics[display_cols].to_string(index=False))


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------


def run_mvo_baseline(
    train_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    initial_amount: float,
    output_dir: Path,
) -> pd.DataFrame:
    stock_data = process_df_for_mvo(train_df)
    trade_data = process_df_for_mvo(trade_df)

    ar_stock_prices = np.asarray(stock_data)
    rows, cols = ar_stock_prices.shape
    ar_returns = stock_returns_computing(ar_stock_prices, rows, cols)

    mean_returns = np.mean(ar_returns, axis=0)
    cov_returns = np.cov(ar_returns, rowvar=False)

    np.set_printoptions(precision=3, suppress=True)
    print("\nMean returns of assets in portfolio\n", mean_returns)

    from pypfopt.efficient_frontier import EfficientFrontier

    try:
        ef_mean = EfficientFrontier(
            mean_returns,
            cov_returns,
            weight_bounds=(0, 0.5),
            solver="SCS",
        )
        ef_mean.max_sharpe()
        cleaned_weights_mean = ef_mean.clean_weights()
        mvo_status = "optimized_scs"

    except Exception as exc:
        print(f"WARNING: MVO optimization failed: {exc}")
        print("Falling back to equal-weight MVO baseline.")

        n_assets = len(mean_returns)
        cleaned_weights_mean = {i: 1.0 / n_assets for i in range(n_assets)}
        mvo_status = "equal_weight_fallback"

    mvo_weights = np.array(
        [
            initial_amount * cleaned_weights_mean[i]
            for i in range(len(cleaned_weights_mean))
        ]
    )

    last_price_inverse = np.array([1 / p for p in stock_data.tail(1).to_numpy()[0]])
    initial_portfolio = np.multiply(mvo_weights, last_price_inverse)

    portfolio_assets = trade_data @ initial_portfolio

    mvo_series = pd.Series(
        portfolio_assets,
        index=trade_data.index,
        name="mvo",
    ).astype(float)

    if mvo_series.empty:
        raise ValueError("MVO portfolio series is empty.")

    first_mvo_value = float(mvo_series.iloc[0])

    if first_mvo_value == 0:
        raise ValueError("MVO first portfolio value is zero; cannot normalize.")

    # Normalize MVO so it starts at the same initial amount as all DRL agents.
    mvo_series = mvo_series / first_mvo_value * initial_amount

    mvo_result = pd.DataFrame(
        {"mvo": mvo_series},
        index=trade_data.index,
    )

    mvo_result.to_csv(output_dir / "account_values_mvo.csv", index=True)

    mvo_weights_df = pd.Series(cleaned_weights_mean, name="weight").to_frame()
    mvo_weights_df["mvo_status"] = mvo_status
    mvo_weights_df.to_csv(output_dir / "mvo_weights.csv")

    return mvo_result


def run_dji_baseline(
    trade_df: pd.DataFrame,
    initial_amount: float,
) -> pd.DataFrame:
    try:
        trade_start_date = pd.to_datetime(trade_df["date"].min())
        trade_end_date = pd.to_datetime(trade_df["date"].max()) + pd.Timedelta(days=1)

        df_dji = yf.download(
            "^DJI",
            start=trade_start_date.strftime("%Y-%m-%d"),
            end=trade_end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        if df_dji.empty:
            raise ValueError("Downloaded DJI dataframe is empty.")

        close = df_dji["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        dji = close.reset_index()
        dji.columns = ["date", "close"]
        dji["date"] = pd.to_datetime(dji["date"]).dt.strftime("%Y-%m-%d")

        first_day = float(dji["close"].iloc[0])
        dji["dji"] = dji["close"].div(first_day).mul(initial_amount)

        return dji[["date", "dji"]].set_index("date")

    except Exception as exc:
        print(f"WARNING: Could not download DJI baseline: {exc}")
        return pd.DataFrame(columns=["dji"])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    dataset_tag = args.dataset_tag.strip()
    run_name = args.run_name.strip()

    selected_agents = [
        agent.strip().lower() for agent in args.agents.split(",") if agent.strip()
    ]

    if args.run_root:
        run_paths = build_run_paths(args.run_root)
        model_dir = run_paths.baseline_models
        output_dir = run_paths.baseline_backtest_files
    else:
        model_dir = Path("trained_models") / dataset_tag / run_name
        output_dir = Path(f"outputs/backtest_{dataset_tag}") / run_name

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("StockDSS - Backtest FinRL baselines on PIT data")
    print("=" * 100)
    print(f"Train data:       {args.train_data}")
    print(f"Trade data:       {args.trade_data}")
    print(f"Dataset tag:      {dataset_tag}")
    print(f"Run name:         {run_name}")
    print(f"Model dir:        {model_dir}")
    print(f"Output dir:       {output_dir}")
    print(f"Run root:         {args.run_root}")
    print(f"Agents:           {selected_agents}")
    print(f"Use MVO:          {args.use_mvo}")
    print(f"Use DJI:          {args.use_dji}")
    print("=" * 100)

    train_df = load_finrl_csv(args.train_data)
    trade_df = load_finrl_csv(args.trade_data)
    indicators = infer_technical_indicators(trade_df)

    env_kwargs = build_env_kwargs(
        trade_df=trade_df,
        indicators=indicators,
        initial_amount=args.initial_amount,
        hmax=args.hmax,
        buy_cost_pct=args.buy_cost_pct,
        sell_cost_pct=args.sell_cost_pct,
        reward_scaling=args.reward_scaling,
    )

    stock_dimension = env_kwargs["stock_dim"]
    state_space = env_kwargs["state_space"]

    print()
    print("Dataset summary")
    print("-" * 100)
    print(f"Train rows:       {len(train_df):,}")
    print(f"Trade rows:       {len(trade_df):,}")
    print(f"Trade date range: {trade_df['date'].min()} -> {trade_df['date'].max()}")
    print(f"Stock dimension:  {stock_dimension}")
    print(f"State space:      {state_space}")
    print(f"Indicators:       {indicators}")
    print("-" * 100)

    trained_models = load_trained_models(
        agents=selected_agents,
        model_dir=model_dir,
    )

    account_values: dict[str, pd.DataFrame] = {}
    actions: dict[str, pd.DataFrame] = {}

    for agent_name, model in trained_models.items():
        if model is None:
            continue

        print()
        print("=" * 100)
        print(f"Backtesting {agent_name.upper()}")
        print("=" * 100)

        trade_env = StockTradingEnv(
            df=trade_df,
            turbulence_threshold=70,
            risk_indicator_col="vix",
            **env_kwargs,
        )

        df_account_value, df_actions = DRLAgent.DRL_prediction(
            model=model,
            environment=trade_env,
        )

        account_values[agent_name] = df_account_value
        actions[agent_name] = df_actions

        save_agent_outputs(
            output_dir=output_dir,
            agent_name=agent_name,
            account_value=df_account_value,
            actions=df_actions,
        )

    mvo_result = pd.DataFrame()

    if args.use_mvo:
        print()
        print("=" * 100)
        print("Running MVO baseline")
        print("=" * 100)
        mvo_result = run_mvo_baseline(
            train_df=train_df,
            trade_df=trade_df,
            initial_amount=args.initial_amount,
            output_dir=output_dir,
        )

    dji = pd.DataFrame()

    if args.use_dji:
        print()
        print("=" * 100)
        print("Running DJI baseline")
        print("=" * 100)
        dji = run_dji_baseline(
            trade_df=trade_df,
            initial_amount=args.initial_amount,
        )

    result_dict = {}

    for agent_name, df in account_values.items():
        df_result = df.set_index(df.columns[0])
        result_dict[agent_name] = df_result["account_value"]

    result = pd.DataFrame(result_dict)

    if args.use_mvo and not mvo_result.empty:
        result["mvo"] = mvo_result["mvo"]

    if args.use_dji and not dji.empty:
        result["dji"] = dji["dji"]

    result.to_csv(output_dir / "backtest_result.csv", index=True)

    metrics = build_metrics(
        result,
        initial_amount=args.initial_amount,
    )

    metrics.to_csv(output_dir / "backtest_metrics.csv", index=False)

    save_json(
        {
            "dataset_tag": dataset_tag,
            "run_name": run_name,
            "train_data": args.train_data,
            "trade_data": args.trade_data,
            "model_dir": str(model_dir),
            "output_dir": str(output_dir),
            "agents": selected_agents,
            "use_mvo": args.use_mvo,
            "use_dji": args.use_dji,
            "initial_amount": args.initial_amount,
            "env_kwargs": env_kwargs,
            "indicators": indicators,
            "stock_dimension": stock_dimension,
            "state_space": state_space,
        },
        output_dir / "backtest_config.json",
    )

    print()
    print("=== Backtest result head ===")
    print(result.head())

    print()
    print("=== Backtest result tail ===")
    print(result.tail())

    print_explanation(metrics)

    if not result.empty:
        plt.rcParams["figure.figsize"] = (15, 5)
        plt.figure()
        result.plot()
        plt.axhline(args.initial_amount, linestyle="--", linewidth=1)
        plt.title("PIT Backtest - Portfolio Value Over Time")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.savefig(
            output_dir / "backtest_result.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    print()
    print("=" * 100)
    print("Backtest finished")
    print("=" * 100)
    print(f"Saved outputs to: {output_dir.resolve()}")
    print("Key files:")
    print("- backtest_result.csv")
    print("- backtest_metrics.csv")
    print("- backtest_config.json")
    print("- account_values_<agent>.csv")
    print("- actions_<agent>.csv")
    print("- backtest_result.png")
    if args.use_mvo:
        print("- account_values_mvo.csv")
        print("- mvo_weights.csv")


if __name__ == "__main__":
    main()
