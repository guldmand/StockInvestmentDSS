"""
Visualize FinRL PIT backtest outputs.

Purpose:
- Load backtest_result.csv and backtest_metrics.csv from a specific run.
- Save useful plots.
- Optionally show matplotlib pop-up windows locally.

Example:

PowerShell:
    $env:PYTHONPATH="src"

    python -m stockdss.rl.experiments.visualize_finrl_backtest_pit `
        --dataset-tag pit_500_2026_01_01 `
        --run-name 2026_05_14_0040_run_train_finrl_baselines_smoketest_all_agents_timesteps_500 `
        --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockdss.runner.run_paths import build_run_paths

DEFAULT_INITIAL_AMOUNT = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize FinRL PIT backtest results."
    )

    parser.add_argument(
        "--dataset-tag",
        required=True,
        help="Dataset tag, e.g. pit_500_2026_01_01",
    )

    parser.add_argument(
        "--run-name",
        required=True,
        help="Run name folder.",
    )

    parser.add_argument(
        "--initial-amount",
        type=float,
        default=DEFAULT_INITIAL_AMOUNT,
        help="Initial portfolio amount. Default: 1000000",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show matplotlib pop-up windows.",
    )

    parser.add_argument(
        "--run-root",
        default=None,
        help=(
            "Optional central runner output folder. "
            "If provided, reads from baseline_finrl/files/backtest and "
            "writes plots to baseline_finrl/plots."
        ),
    )

    return parser.parse_args()


def load_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_path = output_dir / "backtest_result.csv"
    metrics_path = output_dir / "backtest_metrics.csv"

    if not result_path.exists():
        raise FileNotFoundError(f"Missing backtest result file: {result_path}")

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    result = pd.read_csv(result_path)
    result = result.rename(columns={result.columns[0]: "date"})
    result["date"] = pd.to_datetime(result["date"])
    result = result.set_index("date")

    metrics = pd.read_csv(metrics_path)

    return result, metrics


def compute_daily_returns(result: pd.DataFrame) -> pd.DataFrame:
    return result.pct_change().dropna()


def compute_drawdowns(result: pd.DataFrame) -> pd.DataFrame:
    running_max = result.cummax()
    drawdown = result / running_max - 1.0
    return drawdown


def plot_portfolio_values(
    result: pd.DataFrame,
    output_dir: Path,
    initial_amount: float,
) -> None:
    plt.figure(figsize=(15, 6))

    for column in result.columns:
        plt.plot(result.index, result[column], label=column)

    plt.axhline(initial_amount, linestyle="--", linewidth=1, label="initial amount")
    plt.title("PIT Backtest - Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_01_portfolio_values.png", dpi=150)


def plot_normalized_portfolio_values(
    result: pd.DataFrame,
    output_dir: Path,
) -> None:
    normalized = result / result.iloc[0]

    plt.figure(figsize=(15, 6))

    for column in normalized.columns:
        plt.plot(normalized.index, normalized[column], label=column)

    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.title("PIT Backtest - Normalized Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Growth of 1.0")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_02_normalized_portfolio_values.png", dpi=150)


def plot_drawdowns(
    result: pd.DataFrame,
    output_dir: Path,
) -> None:
    drawdowns = compute_drawdowns(result)

    plt.figure(figsize=(15, 6))

    for column in drawdowns.columns:
        plt.plot(drawdowns.index, drawdowns[column] * 100, label=column)

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.title("PIT Backtest - Drawdown Over Time")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_03_drawdowns.png", dpi=150)


def plot_final_values(
    metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    metrics_sorted = metrics.sort_values("end_value", ascending=True)

    plt.figure(figsize=(12, 6))
    plt.barh(metrics_sorted["strategy"], metrics_sorted["end_value"])
    plt.title("PIT Backtest - Final Portfolio Value")
    plt.xlabel("Final portfolio value")
    plt.ylabel("Strategy")

    for index, row in metrics_sorted.reset_index(drop=True).iterrows():
        plt.text(
            row["end_value"],
            index,
            f" {row['total_return_pct']:.2f}%",
            va="center",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "plot_04_final_values.png", dpi=150)


def plot_return_distributions(
    daily_returns: pd.DataFrame,
    output_dir: Path,
) -> None:
    for column in daily_returns.columns:
        series = daily_returns[column].dropna()

        if series.empty:
            continue

        mean_return = series.mean() * 100
        std_return = series.std() * 100
        min_return = series.min() * 100
        max_return = series.max() * 100

        plt.figure(figsize=(10, 6))
        plt.hist(series * 100, bins=25, edgecolor="black")
        plt.axvline(mean_return, linestyle="--", linewidth=1, label="mean")

        plt.title(f"Daily Return Distribution - {column}")
        plt.xlabel("Daily return (%)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)

        text = (
            f"mean: {mean_return:.3f}%\n"
            f"std:  {std_return:.3f}%\n"
            f"min:  {min_return:.3f}%\n"
            f"max:  {max_return:.3f}%"
        )

        plt.text(
            0.98,
            0.95,
            text,
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round", "alpha": 0.15},
        )

        plt.tight_layout()
        plt.savefig(
            output_dir / f"plot_05_daily_return_distribution_{column}.png", dpi=150
        )


def plot_return_boxplot(
    daily_returns: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.boxplot(
        [daily_returns[column].dropna() * 100 for column in daily_returns.columns],
        labels=daily_returns.columns,
        showmeans=True,
    )
    plt.title("PIT Backtest - Daily Return Distribution Comparison")
    plt.xlabel("Strategy")
    plt.ylabel("Daily return (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_06_daily_return_boxplot.png", dpi=150)


def save_return_summary(
    daily_returns: pd.DataFrame,
    output_dir: Path,
) -> None:
    rows = []

    for column in daily_returns.columns:
        series = daily_returns[column].dropna()

        if series.empty:
            continue

        rows.append(
            {
                "strategy": column,
                "mean_daily_return_pct": series.mean() * 100,
                "median_daily_return_pct": series.median() * 100,
                "std_daily_return_pct": series.std() * 100,
                "min_daily_return_pct": series.min() * 100,
                "max_daily_return_pct": series.max() * 100,
                "q05_daily_return_pct": series.quantile(0.05) * 100,
                "q10_daily_return_pct": series.quantile(0.10) * 100,
                "q25_daily_return_pct": series.quantile(0.25) * 100,
                "q50_daily_return_pct": series.quantile(0.50) * 100,
                "q75_daily_return_pct": series.quantile(0.75) * 100,
                "q90_daily_return_pct": series.quantile(0.90) * 100,
                "q95_daily_return_pct": series.quantile(0.95) * 100,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "daily_return_distribution_summary.csv", index=False)


def print_console_summary(metrics: pd.DataFrame) -> None:
    display_cols = [
        "strategy",
        "end_value",
        "profit_loss",
        "total_return_pct",
        "max_drawdown_pct",
        "annualized_sharpe",
    ]

    print()
    print("=" * 100)
    print("Backtest metrics")
    print("=" * 100)
    print(metrics[display_cols].to_string(index=False))


def main() -> None:
    args = parse_args()

    if args.run_root:
        run_paths = build_run_paths(args.run_root)
        output_dir = run_paths.baseline_backtest_files
        plots_dir = run_paths.baseline_plots
    else:
        output_dir = Path(f"outputs/backtest_{args.dataset_tag}") / args.run_name
        plots_dir = output_dir / "plots"

    plots_dir.mkdir(parents=True, exist_ok=True)

    result, metrics = load_outputs(output_dir)
    daily_returns = compute_daily_returns(result)

    print_console_summary(metrics)

    plot_portfolio_values(
        result=result,
        output_dir=plots_dir,
        initial_amount=args.initial_amount,
    )

    plot_normalized_portfolio_values(
        result=result,
        output_dir=plots_dir,
    )

    plot_drawdowns(
        result=result,
        output_dir=plots_dir,
    )

    plot_final_values(
        metrics=metrics,
        output_dir=plots_dir,
    )

    plot_return_distributions(
        daily_returns=daily_returns,
        output_dir=plots_dir,
    )

    plot_return_boxplot(
        daily_returns=daily_returns,
        output_dir=plots_dir,
    )

    save_return_summary(
        daily_returns=daily_returns,
        output_dir=plots_dir,
    )

    print()
    print("=" * 100)
    print("Visualization finished")
    print("=" * 100)
    print(f"Plots saved to: {plots_dir.resolve()}")
    print("Key files:")
    print("- plot_01_portfolio_values.png")
    print("- plot_02_normalized_portfolio_values.png")
    print("- plot_03_drawdowns.png")
    print("- plot_04_final_values.png")
    print("- plot_05_daily_return_distribution_<strategy>.png")
    print("- plot_06_daily_return_boxplot.png")
    print("- daily_return_distribution_summary.csv")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
