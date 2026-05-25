from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def print_banner(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def find_metric_files(run_root: Path) -> list[Path]:
    metrics_root = run_root / "algorithmic_trading" / "results"

    if not metrics_root.exists():
        raise FileNotFoundError(
            f"Could not find algorithmic trading results directory: {metrics_root}"
        )

    metric_files = sorted(metrics_root.rglob("*_metrics.csv"))

    if not metric_files:
        raise FileNotFoundError(f"No *_metrics.csv files found under: {metrics_root}")

    return metric_files


def load_metrics(metric_files: list[Path]) -> pd.DataFrame:
    frames = []

    for path in metric_files:
        df = pd.read_csv(path)
        df["metrics_file"] = str(path)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    expected_cols = [
        "strategy",
        "source",
        "start_value",
        "end_value",
        "profit_loss",
        "total_return_pct",
        "max_drawdown_pct",
        "annualized_sharpe",
        "days",
        "ended_above_initial",
    ]

    missing = [col for col in expected_cols if col not in result.columns]
    if missing:
        raise ValueError(f"Missing expected columns in metrics files: {missing}")

    return result[expected_cols + ["metrics_file"]].copy()


def _safe_rank(series: pd.Series, *, ascending: bool) -> pd.Series:
    """Return integer ranks even when a metric contains NaN/inf values."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Missing performance values should rank last.
    fill_value = np.inf if ascending else -np.inf
    return values.fillna(fill_value).rank(ascending=ascending, method="min").astype(int)


def add_rankings(metrics: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics.copy()

    numeric_columns = [
        "start_value",
        "end_value",
        "profit_loss",
        "total_return_pct",
        "max_drawdown_pct",
        "annualized_sharpe",
        "days",
    ]

    for column in numeric_columns:
        if column in ranked.columns:
            ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
            ranked[column] = ranked[column].replace([np.inf, -np.inf], np.nan)

    # Higher is better.
    ranked["rank_end_value"] = _safe_rank(ranked["end_value"], ascending=False)
    ranked["rank_return"] = _safe_rank(ranked["total_return_pct"], ascending=False)
    ranked["rank_sharpe"] = _safe_rank(ranked["annualized_sharpe"], ascending=False)

    # Drawdown is negative. Higher/closer to zero is better.
    ranked["rank_drawdown"] = _safe_rank(ranked["max_drawdown_pct"], ascending=False)

    ranked["combined_rank_score"] = (
        ranked["rank_end_value"]
        + ranked["rank_return"]
        + ranked["rank_sharpe"]
        + ranked["rank_drawdown"]
    )

    ranked["rank_combined"] = (
        ranked["combined_rank_score"].rank(ascending=True, method="min").astype(int)
    )

    ranked = ranked.sort_values(
        ["rank_combined", "combined_rank_score", "end_value"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return ranked


def make_insights(df: pd.DataFrame) -> list[str]:
    best_return = df.sort_values("total_return_pct", ascending=False).iloc[0]
    best_end_value = df.sort_values("end_value", ascending=False).iloc[0]
    best_sharpe = df.sort_values("annualized_sharpe", ascending=False).iloc[0]
    best_drawdown = df.sort_values("max_drawdown_pct", ascending=False).iloc[0]
    best_combined = df.sort_values("rank_combined", ascending=True).iloc[0]

    insights = [
        f"Best final portfolio value: {best_end_value['strategy']} "
        f"ended at {best_end_value['end_value']:,.2f} "
        f"with total return {best_end_value['total_return_pct']:.2f}%.",
        f"Best total return: {best_return['strategy']} "
        f"returned {best_return['total_return_pct']:.2f}%.",
        f"Best annualized Sharpe ratio: {best_sharpe['strategy']} "
        f"with Sharpe {best_sharpe['annualized_sharpe']:.4f}.",
        f"Lowest drawdown: {best_drawdown['strategy']} "
        f"with max drawdown {best_drawdown['max_drawdown_pct']:.2f}%.",
        f"Best combined rank: {best_combined['strategy']} "
        f"with combined rank score {best_combined['combined_rank_score']}.",
    ]

    return insights


def save_markdown_report(
    df: pd.DataFrame, insights: list[str], output_path: Path
) -> None:
    report_df = df[
        [
            "rank_combined",
            "strategy",
            "end_value",
            "profit_loss",
            "total_return_pct",
            "max_drawdown_pct",
            "annualized_sharpe",
            "rank_return",
            "rank_sharpe",
            "rank_drawdown",
            "combined_rank_score",
        ]
    ].copy()

    lines = [
        "# Algorithmic Trading Baseline Comparison",
        "",
        "This report compares classical non-RL algorithmic trading baselines.",
        "",
        "## Key insights",
        "",
    ]

    for insight in insights:
        lines.append(f"- {insight}")

    lines.extend(
        [
            "",
            "## Ranked comparison",
            "",
            report_df.to_markdown(index=False),
            "",
            "## Notes",
            "",
            "- `total_return_pct` is profit/loss relative to the initial capital.",
            "- `end_value` is the final portfolio value including the original capital.",
            "- `max_drawdown_pct` is negative; values closer to zero are better.",
            "- `annualized_sharpe` is the annualized Sharpe ratio.",
            "- Missing/undefined metric values are ranked last for that metric.",
            "- `rank_combined` is based on end-value rank + return rank + Sharpe rank + drawdown rank.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_summary_plot(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.sort_values("total_return_pct", ascending=True)

    plt.figure(figsize=(12, 7))
    plt.barh(plot_df["strategy"], plot_df["total_return_pct"])
    plt.xlabel("Total return (%)")
    plt.ylabel("Strategy")
    plt.title("Algorithmic Trading Baselines - Total Return")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare algorithmic trading baseline results from one run root."
    )
    parser.add_argument(
        "--run-root",
        required=True,
        help="Run root containing algorithmic_trading/results",
    )
    parser.add_argument(
        "--show", action="store_true", help="Print comparison and insights"
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    summary_dir = run_root / "algorithmic_trading" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    metric_files = find_metric_files(run_root)
    metrics = load_metrics(metric_files)
    ranked = add_rankings(metrics)
    insights = make_insights(ranked)

    csv_path = summary_dir / "algorithmic_trading_summary.csv"
    md_path = summary_dir / "algorithmic_trading_summary.md"
    plot_path = summary_dir / "algorithmic_trading_summary_returns.png"

    ranked.to_csv(csv_path, index=False)
    save_markdown_report(ranked, insights, md_path)
    save_summary_plot(ranked, plot_path)

    print_banner("Algorithmic Trading baseline comparison")

    display_cols = [
        "rank_combined",
        "strategy",
        "end_value",
        "profit_loss",
        "total_return_pct",
        "max_drawdown_pct",
        "annualized_sharpe",
    ]

    print(ranked[display_cols].to_string(index=False))

    print_banner("Key insights")
    for insight in insights:
        print(f"- {insight}")

    print_banner("Summary finished")
    print("Saved outputs:")
    print(f"- summary csv: {csv_path}")
    print(f"- summary md:  {md_path}")
    print(f"- summary png: {plot_path}")


if __name__ == "__main__":
    main()
