"""
Visualize IQN-style decision support output.

Purpose:
- Show the kind of distributional RL / IQN plots we want for the thesis.
- Visualize return-distribution per action.
- Visualize quantile function F^-1(tau) per action.
- Visualize risk-adjusted action score.
- This script can run now with demo data.
- Later, backtest_iqn_pit.py should output real q10/q25/q50/q75/q90/CVaR values,
  and this script can read those instead.

Example:

PowerShell:
    $env:PYTHONPATH="src"

    python -m stockdss.rl.experiments.visualize_iqn_decision_distribution `
        --demo `
        --show

Save only:
    python -m stockdss.rl.experiments.visualize_iqn_decision_distribution `
        --demo

Visualize real backtest decision output:
    python -m stockdss.rl.experiments.visualize_iqn_decision_distribution `
        --decision-csv outputs/backtest_iqn_pit_500_2026_01_01/<run_name>/iqn_action_estimates_last_day.csv `
        --date 2026-03-18 `
        --ticker AAPL `
        --price 0 `
        --risk-lambda 0.75 `
        --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACTIONS = [
    "HOLD",
    "BUY_25",
    "BUY_50",
    "BUY_100",
    "SELL_25",
    "SELL_50",
    "SELL_100",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize IQN decision distributions."
    )

    parser.add_argument(
        "--decision-csv",
        default=None,
        help=(
            "Optional path to a decision CSV with columns: "
            "action,q10,q25,q50,q75,q90,cvar10. "
            "If omitted with --demo, demo values are used."
        ),
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use hardcoded demo IQN-style quantile values.",
    )

    parser.add_argument(
        "--date",
        default="2026-01-12",
        help="Decision date shown in plot title.",
    )

    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker shown in plot title.",
    )

    parser.add_argument(
        "--price",
        type=float,
        default=187.42,
        help="Current price shown in plot title.",
    )

    parser.add_argument(
        "--risk-lambda",
        type=float,
        default=0.75,
        help="Risk penalty. Score = q50 - risk_lambda * abs(CVaR10).",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. "
            "If omitted and --decision-csv is provided, outputs are saved inside "
            "the same backtest run folder under plots/iqn_decision_distribution. "
            "If omitted with --demo, outputs are saved under outputs/iqn_visualizations/demo."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show matplotlib pop-up window.",
    )

    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """
    Resolve output location.

    Rules:
    1. If --output-dir is given, use it.
    2. If --decision-csv is given, save inside the same backtest run folder:
       <run-folder>/plots/iqn_decision_distribution/
    3. If --demo is used, save in:
       outputs/iqn_visualizations/demo/
    """
    if args.output_dir:
        output_dir = Path(args.output_dir)

    elif args.decision_csv:
        decision_csv_path = Path(args.decision_csv)
        output_dir = decision_csv_path.parent / "plots" / "iqn_decision_distribution"

    else:
        output_dir = Path("outputs") / "iqn_visualizations" / "demo"

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def demo_decision_data() -> pd.DataFrame:
    """
    Demo values are returns in percent.

    Interpretation:
    - q10 is pessimistic estimate.
    - q50 is median estimate.
    - q90 is optimistic estimate.
    - CVaR10 is average downside in the worst 10 percent region.
    """
    rows = [
        {
            "action": "HOLD",
            "q10": -0.80,
            "q25": -0.30,
            "q50": 0.20,
            "q75": 0.70,
            "q90": 1.10,
            "cvar10": -1.20,
        },
        {
            "action": "BUY_25",
            "q10": -0.50,
            "q25": 0.00,
            "q50": 0.50,
            "q75": 1.10,
            "q90": 1.80,
            "cvar10": -0.90,
        },
        {
            "action": "BUY_50",
            "q10": -0.90,
            "q25": -0.10,
            "q50": 0.60,
            "q75": 1.40,
            "q90": 2.20,
            "cvar10": -1.50,
        },
        {
            "action": "BUY_100",
            "q10": -1.80,
            "q25": -0.60,
            "q50": 0.90,
            "q75": 2.10,
            "q90": 3.40,
            "cvar10": -2.90,
        },
        {
            "action": "SELL_25",
            "q10": -0.40,
            "q25": -0.10,
            "q50": 0.10,
            "q75": 0.30,
            "q90": 0.50,
            "cvar10": -0.60,
        },
        {
            "action": "SELL_50",
            "q10": -0.30,
            "q25": -0.10,
            "q50": 0.00,
            "q75": 0.20,
            "q90": 0.40,
            "cvar10": -0.50,
        },
        {
            "action": "SELL_100",
            "q10": -0.20,
            "q25": 0.00,
            "q50": 0.00,
            "q75": 0.10,
            "q90": 0.20,
            "cvar10": -0.30,
        },
    ]

    return pd.DataFrame(rows)


def load_decision_data(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Decision CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"action", "q10", "q25", "q50", "q75", "q90", "cvar10"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Decision CSV missing columns: {sorted(missing)}")

    return df


def add_scores(df: pd.DataFrame, risk_lambda: float) -> pd.DataFrame:
    df = df.copy()

    # Risk-aware score:
    # high median return is good, large negative CVaR is bad.
    df["risk_adjusted_score"] = df["q50"] - risk_lambda * df["cvar10"].abs()

    # Pure risk-neutral baseline:
    df["risk_neutral_score"] = df["q50"]

    df["selected_risk_adjusted"] = False
    df["selected_risk_neutral"] = False

    risk_adjusted_index = df["risk_adjusted_score"].idxmax()
    risk_neutral_index = df["risk_neutral_score"].idxmax()

    df.loc[risk_adjusted_index, "selected_risk_adjusted"] = True
    df.loc[risk_neutral_index, "selected_risk_neutral"] = True

    return df


def interpolate_quantile_curve(row: pd.Series, taus: np.ndarray) -> np.ndarray:
    known_taus = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    known_values = np.array(
        [
            row["q10"],
            row["q25"],
            row["q50"],
            row["q75"],
            row["q90"],
        ],
        dtype=float,
    )

    return np.interp(taus, known_taus, known_values)


def sample_distribution_from_quantiles(
    row: pd.Series,
    num_samples: int = 2_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Approximate a return distribution from quantile points.

    This is a visualization approximation.

    Later, when the real IQN model is used, this should be replaced by:
        taus = torch.rand(...)
        quantile_values = iqn_network(state, taus)
    """
    rng = np.random.default_rng(seed)
    taus = rng.uniform(0.10, 0.90, size=num_samples)

    values = interpolate_quantile_curve(row, taus)

    # Add tiny noise so histogram/density is not too artificial.
    values = values + rng.normal(0.0, 0.04, size=num_samples)

    return values


def plot_iqn_dashboard(
    df: pd.DataFrame,
    date: str,
    ticker: str,
    price: float,
    risk_lambda: float,
    output_dir: Path,
) -> None:
    selected_row = df[df["selected_risk_adjusted"]].iloc[0]
    selected_action = selected_row["action"]

    taus = np.linspace(0.10, 0.90, 100)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        (
            f"IQN Decision Support Dashboard | {ticker} | {date} | "
            f"price=${price:,.2f} | selected={selected_action}"
        ),
        fontsize=14,
        fontweight="bold",
    )

    grid = fig.add_gridspec(2, 2)

    ax_quantile = fig.add_subplot(grid[0, 0])
    ax_distribution = fig.add_subplot(grid[0, 1])
    ax_score = fig.add_subplot(grid[1, 0])
    ax_table = fig.add_subplot(grid[1, 1])

    # ------------------------------------------------------------------
    # Plot 1: Quantile function per action
    # ------------------------------------------------------------------
    for _, row in df.iterrows():
        action = row["action"]
        values = interpolate_quantile_curve(row, taus)

        linewidth = 3 if action == selected_action else 1.5
        alpha = 1.0 if action == selected_action else 0.45

        ax_quantile.plot(
            taus,
            values,
            label=action,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax_quantile.axhline(0.0, linestyle="--", linewidth=1)
    ax_quantile.set_title("Quantile function per action")
    ax_quantile.set_xlabel("τ")
    ax_quantile.set_ylabel("Estimated future return (%)")
    ax_quantile.grid(True, alpha=0.3)
    ax_quantile.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Plot 2: Approximate return distribution per action
    # ------------------------------------------------------------------
    for index, row in df.iterrows():
        action = row["action"]
        samples = sample_distribution_from_quantiles(
            row,
            num_samples=2_000,
            seed=42 + index,
        )

        linewidth = 2.5 if action == selected_action else 1.2
        alpha = 0.95 if action == selected_action else 0.35

        counts, bin_edges = np.histogram(samples, bins=40, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ax_distribution.plot(
            bin_centers,
            counts,
            label=action,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax_distribution.axvline(0.0, linestyle="--", linewidth=1)
    ax_distribution.set_title("Estimated return distribution per action")
    ax_distribution.set_xlabel("Estimated future return (%)")
    ax_distribution.set_ylabel("Density")
    ax_distribution.grid(True, alpha=0.3)
    ax_distribution.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Plot 3: Risk-adjusted action score
    # ------------------------------------------------------------------
    score_df = df.sort_values("risk_adjusted_score", ascending=True)

    labels = score_df["action"].tolist()
    scores = score_df["risk_adjusted_score"].to_numpy()

    ax_score.barh(labels, scores)

    for y_index, (_, row) in enumerate(score_df.iterrows()):
        marker = "  selected" if row["selected_risk_adjusted"] else ""
        ax_score.text(
            row["risk_adjusted_score"],
            y_index,
            f" {row['risk_adjusted_score']:.3f}{marker}",
            va="center",
        )

    ax_score.axvline(0.0, linestyle="--", linewidth=1)
    ax_score.set_title(f"Risk-adjusted score = q50 - {risk_lambda:.2f} * abs(CVaR10)")
    ax_score.set_xlabel("Score")
    ax_score.set_ylabel("Action")
    ax_score.grid(True, axis="x", alpha=0.3)

    # ------------------------------------------------------------------
    # Plot 4: Decision table
    # ------------------------------------------------------------------
    table_df = df[
        [
            "action",
            "q10",
            "q25",
            "q50",
            "q75",
            "q90",
            "cvar10",
            "risk_adjusted_score",
        ]
    ].copy()

    table_df = table_df.round(3)

    ax_table.axis("off")
    ax_table.set_title("Action return estimates", pad=12)

    table = ax_table.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    for row_index, action in enumerate(table_df["action"], start=1):
        if action == selected_action:
            for col_index in range(len(table_df.columns)):
                table[(row_index, col_index)].set_linewidth(2.0)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    output_path = output_dir / "iqn_decision_dashboard.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")


def plot_quantile_only(
    df: pd.DataFrame,
    date: str,
    ticker: str,
    output_dir: Path,
) -> None:
    selected_action = df[df["selected_risk_adjusted"]].iloc[0]["action"]
    taus = np.linspace(0.10, 0.90, 100)

    plt.figure(figsize=(14, 7))

    for _, row in df.iterrows():
        action = row["action"]
        values = interpolate_quantile_curve(row, taus)

        linewidth = 3 if action == selected_action else 1.5
        alpha = 1.0 if action == selected_action else 0.45

        plt.plot(
            taus,
            values,
            label=action,
            linewidth=linewidth,
            alpha=alpha,
        )

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.title(f"IQN Quantile Function per Action | {ticker} | {date}")
    plt.xlabel("τ")
    plt.ylabel("Estimated future return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "iqn_quantile_functions.png", dpi=150)


def plot_distribution_only(
    df: pd.DataFrame,
    date: str,
    ticker: str,
    output_dir: Path,
) -> None:
    selected_action = df[df["selected_risk_adjusted"]].iloc[0]["action"]

    plt.figure(figsize=(14, 7))

    for index, row in df.iterrows():
        action = row["action"]
        samples = sample_distribution_from_quantiles(
            row,
            num_samples=2_000,
            seed=100 + index,
        )

        counts, bin_edges = np.histogram(samples, bins=45, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        linewidth = 3 if action == selected_action else 1.3
        alpha = 1.0 if action == selected_action else 0.35

        plt.plot(
            bin_centers,
            counts,
            label=action,
            linewidth=linewidth,
            alpha=alpha,
        )

    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.title(f"IQN Estimated Return Distribution per Action | {ticker} | {date}")
    plt.xlabel("Estimated future return (%)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "iqn_return_distributions.png", dpi=150)


def print_decision_summary(
    df: pd.DataFrame,
    date: str,
    ticker: str,
    price: float,
) -> None:
    selected = df[df["selected_risk_adjusted"]].iloc[0]
    risk_neutral = df[df["selected_risk_neutral"]].iloc[0]

    display_cols = [
        "action",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "cvar10",
        "risk_adjusted_score",
    ]

    print()
    print("=" * 100)
    print("IQN decision support output")
    print("=" * 100)
    print(f"Date:                 {date}")
    print(f"Ticker:               {ticker}")
    print(f"Current price:         ${price:,.2f}")
    print(f"Risk-adjusted action:  {selected['action']}")
    print(f"Risk-neutral action:   {risk_neutral['action']}")
    print("-" * 100)
    print(df[display_cols].round(3).to_string(index=False))
    print("=" * 100)


def main() -> None:
    args = parse_args()

    if args.decision_csv:
        decision_df = load_decision_data(args.decision_csv)
    elif args.demo:
        decision_df = demo_decision_data()
    else:
        raise ValueError("Use --demo or provide --decision-csv.")

    decision_df = add_scores(
        decision_df,
        risk_lambda=args.risk_lambda,
    )

    output_dir = resolve_output_dir(args)

    decision_df.to_csv(output_dir / "iqn_decision_estimates.csv", index=False)

    print_decision_summary(
        df=decision_df,
        date=args.date,
        ticker=args.ticker,
        price=args.price,
    )

    plot_iqn_dashboard(
        df=decision_df,
        date=args.date,
        ticker=args.ticker,
        price=args.price,
        risk_lambda=args.risk_lambda,
        output_dir=output_dir,
    )

    plot_quantile_only(
        df=decision_df,
        date=args.date,
        ticker=args.ticker,
        output_dir=output_dir,
    )

    plot_distribution_only(
        df=decision_df,
        date=args.date,
        ticker=args.ticker,
        output_dir=output_dir,
    )

    print()
    print("Saved:")
    print(f"- {output_dir / 'iqn_decision_estimates.csv'}")
    print(f"- {output_dir / 'iqn_decision_dashboard.png'}")
    print(f"- {output_dir / 'iqn_quantile_functions.png'}")
    print(f"- {output_dir / 'iqn_return_distributions.png'}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
