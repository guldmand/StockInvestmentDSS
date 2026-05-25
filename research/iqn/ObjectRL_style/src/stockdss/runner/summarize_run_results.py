"""Summarize StockDSS run results.

This runner-level summary merges:
- FinRL baseline metrics
- D-IQN-DSS metrics
- Algorithmic Trading / non-RL baseline metrics
- Last IQN decision estimates, when available

It writes:
- summary_report.csv
- summary_report.md
- summary_dashboard.png
- summary_iqn_last_decision.csv, when IQN decision data exists
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


GROUP_COLORS = {
    "Algorithmic Trading / non-RL": "tab:green",
    "FinRL / baseline": "tab:blue",
    "D-IQN-DSS": "tab:orange",
}

REQUIRED_METRIC_COLUMNS = [
    "strategy",
    "source",
    "start_value",
    "end_value",
    "profit_loss",
    "total_return_pct",
    "max_drawdown_pct",
    "annualized_sharpe",
]


def print_section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def ensure_summary_dir(run_root: Path) -> Path:
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir


def normalise_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a metric dataframe with the expected runner-summary schema."""

    out = df.copy()

    rename_map = {
        "account_start_value": "start_value",
        "account_end_value": "end_value",
        "return_pct": "total_return_pct",
        "sharpe": "annualized_sharpe",
        "max_drawdown": "max_drawdown_pct",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    for column in REQUIRED_METRIC_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA

    for column in [
        "start_value",
        "end_value",
        "profit_loss",
        "total_return_pct",
        "max_drawdown_pct",
        "annualized_sharpe",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if out["profit_loss"].isna().any():
        out["profit_loss"] = out["end_value"] - out["start_value"]

    if out["total_return_pct"].isna().any():
        out["total_return_pct"] = (out["profit_loss"] / out["start_value"]) * 100.0

    return out[REQUIRED_METRIC_COLUMNS]


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def find_metric_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*metrics.csv"))


def load_finrl_metrics(run_root: Path) -> pd.DataFrame:
    """Load FinRL baseline metrics from common output locations."""

    candidate_dirs = [
        run_root / "finrl" / "results",
        run_root / "finrl_baselines" / "results",
        run_root / "baselines" / "finrl",
        run_root / "results" / "finrl",
    ]

    frames: list[pd.DataFrame] = []

    for directory in candidate_dirs:
        for metric_file in find_metric_files(directory):
            df = read_csv_if_exists(metric_file)
            if df is None or df.empty:
                continue

            if "strategy" not in df.columns:
                strategy = metric_file.stem.replace("_metrics", "")
                df["strategy"] = strategy

            if "source" not in df.columns:
                df["source"] = "FinRL / baseline"

            frames.append(normalise_metric_columns(df))

    fallback_files = [
        run_root / "finrl" / "summary" / "finrl_summary.csv",
        run_root / "finrl_baselines" / "summary" / "finrl_summary.csv",
        run_root / "baselines" / "finrl_summary.csv",
    ]

    for path in fallback_files:
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            continue

        if "source" not in df.columns:
            df["source"] = "FinRL / baseline"

        frames.append(normalise_metric_columns(df))

    if not frames:
        return pd.DataFrame(columns=REQUIRED_METRIC_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["strategy", "source"], keep="last")
    return combined


def load_iqn_metrics(run_root: Path) -> pd.DataFrame:
    """Load D-IQN-DSS metrics from common output locations."""

    candidate_files = [
        run_root / "d_iqn_dss" / "results" / "d_iqn_dss_metrics.csv",
        run_root / "d_iqn_dss" / "d_iqn_dss_metrics.csv",
        run_root / "iqn" / "results" / "d_iqn_dss_metrics.csv",
        run_root / "results" / "d_iqn_dss_metrics.csv",
    ]

    frames: list[pd.DataFrame] = []

    for path in candidate_files:
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            continue

        if "strategy" not in df.columns:
            df["strategy"] = "d_iqn_dss"

        if "source" not in df.columns:
            df["source"] = "D-IQN-DSS"

        frames.append(normalise_metric_columns(df))

    for path in sorted(run_root.rglob("*iqn*metrics*.csv")):
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            continue

        if "strategy" not in df.columns:
            df["strategy"] = "d_iqn_dss"

        if "source" not in df.columns:
            df["source"] = "D-IQN-DSS"

        frames.append(normalise_metric_columns(df))

    if not frames:
        return pd.DataFrame(columns=REQUIRED_METRIC_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["strategy", "source"], keep="last")
    return combined


def load_algorithmic_trading_metrics(run_root: Path) -> pd.DataFrame:
    """Load algorithmic-trading baseline metrics.

    Prefer the already-built algorithmic summary if it exists; otherwise gather
    all individual *_metrics.csv files under algorithmic_trading/results.
    """

    summary_file = (
        run_root
        / "algorithmic_trading"
        / "summary"
        / "algorithmic_trading_summary.csv"
    )

    summary_df = read_csv_if_exists(summary_file)
    if summary_df is not None and not summary_df.empty:
        if "source" not in summary_df.columns:
            summary_df["source"] = "Algorithmic Trading / non-RL"
        return normalise_metric_columns(summary_df)

    results_root = run_root / "algorithmic_trading" / "results"
    frames: list[pd.DataFrame] = []

    for metric_file in find_metric_files(results_root):
        df = read_csv_if_exists(metric_file)
        if df is None or df.empty:
            continue

        if "strategy" not in df.columns:
            df["strategy"] = metric_file.stem.replace("_metrics", "")

        if "source" not in df.columns:
            df["source"] = "Algorithmic Trading / non-RL"

        frames.append(normalise_metric_columns(df))

    if not frames:
        return pd.DataFrame(columns=REQUIRED_METRIC_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["strategy", "source"], keep="last")
    return combined


def load_all_metrics(run_root: Path) -> pd.DataFrame:
    frames = [
        load_finrl_metrics(run_root),
        load_algorithmic_trading_metrics(run_root),
        load_iqn_metrics(run_root),
    ]

    frames = [df for df in frames if df is not None and not df.empty]

    if not frames:
        return pd.DataFrame(columns=REQUIRED_METRIC_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)

    for column in [
        "start_value",
        "end_value",
        "profit_loss",
        "total_return_pct",
        "max_drawdown_pct",
        "annualized_sharpe",
    ]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")

    combined = combined.drop_duplicates(subset=["strategy", "source"], keep="last")
    combined = combined.sort_values(
        by=["end_value", "total_return_pct"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    combined.insert(0, "rank", range(1, len(combined) + 1))
    return combined


def find_last_iqn_decision_file(run_root: Path) -> Optional[Path]:
    candidate_files = [
        run_root / "d_iqn_dss" / "results" / "iqn_last_decision.csv",
        run_root / "d_iqn_dss" / "iqn_last_decision.csv",
        run_root / "iqn" / "results" / "iqn_last_decision.csv",
        run_root / "results" / "iqn_last_decision.csv",
        run_root / "summary" / "summary_iqn_last_decision.csv",
    ]

    for path in candidate_files:
        if path.exists():
            return path

    matches = sorted(run_root.rglob("*last*decision*.csv"))
    return matches[-1] if matches else None


def load_last_iqn_decision(run_root: Path) -> Optional[pd.DataFrame]:
    path = find_last_iqn_decision_file(run_root)
    if path is None:
        return None

    df = read_csv_if_exists(path)
    if df is None or df.empty:
        return None

    if "action" not in df.columns:
        return None

    for column in ["q10", "q25", "q50", "q75", "q90", "cvar10", "risk_adjusted_score"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "risk_adjusted_score" not in df.columns and {"q50", "cvar10"}.issubset(df.columns):
        df["risk_adjusted_score"] = df["q50"] - 0.75 * df["cvar10"].abs()

    return df


def write_markdown_summary(
    metrics_df: pd.DataFrame,
    decision_df: Optional[pd.DataFrame],
    output_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# StockDSS result comparison")
    lines.append("")

    if metrics_df.empty:
        lines.append("No metrics found.")
    else:
        lines.append(metrics_df.to_markdown(index=False))

    if decision_df is not None and not decision_df.empty:
        lines.append("")
        lines.append("## Last IQN decision estimates")
        lines.append("")
        lines.append(decision_df.to_markdown(index=False))

    output_path.write_text("\n".join(lines), encoding="utf-8")


def get_group_colors(df: pd.DataFrame) -> list[str]:
    return [GROUP_COLORS.get(str(source), "tab:gray") for source in df["source"]]


def add_group_legend(fig) -> None:
    handles = [
        Patch(
            color=GROUP_COLORS["FinRL / baseline"],
            label="FinRL / baseline",
        ),
        Patch(
            color=GROUP_COLORS["Algorithmic Trading / non-RL"],
            label="Algorithmic Trading / non-RL",
        ),
        Patch(
            color=GROUP_COLORS["D-IQN-DSS"],
            label="D-IQN-DSS",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.925),
    )


def plot_summary_dashboard(
    metrics_df: pd.DataFrame,
    decision_df: Optional[pd.DataFrame],
    output_path: Path,
    run_root: Path,
) -> None:
    max_display_strategies = 18

    if metrics_df.empty:
        fig = plt.figure(figsize=(14, 6))
        fig.text(
            0.5,
            0.5,
            "No runner metrics found",
            ha="center",
            va="center",
            fontsize=16,
        )
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    # Use top-ranked strategies, but always include D-IQN-DSS if it exists.
    plot_metrics_df = metrics_df.head(max_display_strategies).copy()

    iqn_rows = metrics_df[metrics_df["source"] == "D-IQN-DSS"].copy()
    if not iqn_rows.empty:
        missing_iqn = ~iqn_rows["strategy"].isin(plot_metrics_df["strategy"])
        if missing_iqn.any():
            plot_metrics_df = pd.concat(
                [plot_metrics_df, iqn_rows[missing_iqn]],
                ignore_index=True,
            )

    fig = plt.figure(figsize=(20, 12))

    fig.text(
        0.5,
        0.985,
        "StockDSS Runner Summary",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.958,
        run_root.name,
        ha="center",
        va="top",
        fontsize=11,
        family="monospace",
    )

    add_group_legend(fig)

    grid = fig.add_gridspec(
        2,
        2,
        left=0.10,
        right=0.98,
        bottom=0.08,
        top=0.86,
        wspace=0.34,
        hspace=0.30,
    )

    ax_return = fig.add_subplot(grid[0, 0])
    ax_drawdown = fig.add_subplot(grid[0, 1])
    ax_sharpe = fig.add_subplot(grid[1, 0])
    ax_iqn = fig.add_subplot(grid[1, 1])

    plot_df = plot_metrics_df.sort_values("total_return_pct", ascending=True)
    ax_return.barh(
        plot_df["strategy"],
        plot_df["total_return_pct"],
        color=get_group_colors(plot_df),
    )
    ax_return.axvline(0.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax_return.set_title("Total return by strategy", fontsize=12)
    ax_return.set_xlabel("Total return (%)")
    ax_return.tick_params(axis="y", labelsize=8)
    ax_return.grid(True, axis="x", alpha=0.25)

    drawdown_df = plot_metrics_df.sort_values("max_drawdown_pct", ascending=True)
    ax_drawdown.barh(
        drawdown_df["strategy"],
        drawdown_df["max_drawdown_pct"],
        color=get_group_colors(drawdown_df),
    )
    ax_drawdown.axvline(0.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax_drawdown.set_title("Maximum drawdown by strategy", fontsize=12)
    ax_drawdown.set_xlabel("Max drawdown (%)")
    ax_drawdown.tick_params(axis="y", labelsize=8)
    ax_drawdown.grid(True, axis="x", alpha=0.25)

    sharpe_df = plot_metrics_df.sort_values("annualized_sharpe", ascending=True)
    ax_sharpe.barh(
        sharpe_df["strategy"],
        sharpe_df["annualized_sharpe"],
        color=get_group_colors(sharpe_df),
    )
    ax_sharpe.axvline(0.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax_sharpe.set_title("Annualized Sharpe by strategy", fontsize=12)
    ax_sharpe.set_xlabel("Sharpe")
    ax_sharpe.tick_params(axis="y", labelsize=8)
    ax_sharpe.grid(True, axis="x", alpha=0.25)

    if decision_df is not None and not decision_df.empty:
        decision_plot_df = decision_df.copy()

        if "risk_adjusted_score" not in decision_plot_df.columns:
            if {"q50", "cvar10"}.issubset(decision_plot_df.columns):
                decision_plot_df["risk_adjusted_score"] = (
                    pd.to_numeric(decision_plot_df["q50"], errors="coerce")
                    - 0.75
                    * pd.to_numeric(decision_plot_df["cvar10"], errors="coerce").abs()
                )
            else:
                decision_plot_df["risk_adjusted_score"] = pd.NA

        decision_plot_df = decision_plot_df.dropna(subset=["risk_adjusted_score"])
        decision_plot_df = decision_plot_df.sort_values(
            "risk_adjusted_score",
            ascending=True,
        )

        ax_iqn.barh(
            decision_plot_df["action"],
            decision_plot_df["risk_adjusted_score"],
            color=GROUP_COLORS["D-IQN-DSS"],
        )
        ax_iqn.axvline(0.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
        ax_iqn.set_title("Last IQN decision: risk-adjusted action score", fontsize=12)
        ax_iqn.set_xlabel("Score")
        ax_iqn.tick_params(axis="y", labelsize=9)
        ax_iqn.grid(True, axis="x", alpha=0.25)
    else:
        ax_iqn.axis("off")
        ax_iqn.set_title("No IQN decision estimates found", fontsize=12)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_metrics(metrics_df: pd.DataFrame) -> None:
    print_section("StockDSS result comparison")

    if metrics_df.empty:
        print("No metrics found.")
        return

    display_columns = [
        "rank",
        "strategy",
        "source",
        "end_value",
        "profit_loss",
        "total_return_pct",
        "max_drawdown_pct",
        "annualized_sharpe",
    ]

    print(
        metrics_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def print_iqn_decisions(decision_df: Optional[pd.DataFrame]) -> None:
    if decision_df is None or decision_df.empty:
        return

    print_section("Last IQN decision estimates")

    preferred_columns = [
        "action",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "cvar10",
        "risk_adjusted_score",
    ]
    display_columns = [col for col in preferred_columns if col in decision_df.columns]

    print(
        decision_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize StockDSS run results into CSV, Markdown, and dashboard PNG."
    )
    parser.add_argument(
        "--run-root",
        required=True,
        type=Path,
        help="Run root containing FinRL, D-IQN-DSS, and/or algorithmic trading outputs.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the generated dashboard interactively after saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root
    summary_dir = ensure_summary_dir(run_root)

    metrics_df = load_all_metrics(run_root)
    decision_df = load_last_iqn_decision(run_root)

    summary_csv = summary_dir / "summary_report.csv"
    summary_md = summary_dir / "summary_report.md"
    summary_png = summary_dir / "summary_dashboard.png"
    decision_csv = summary_dir / "summary_iqn_last_decision.csv"

    metrics_df.to_csv(summary_csv, index=False)
    write_markdown_summary(metrics_df, decision_df, summary_md)
    plot_summary_dashboard(metrics_df, decision_df, summary_png, run_root)

    if decision_df is not None and not decision_df.empty:
        decision_df.to_csv(decision_csv, index=False)

    print_metrics(metrics_df)
    print_iqn_decisions(decision_df)

    print_section("Summary finished")
    print(f"Saved to: {summary_dir.resolve()}")
    print("Key files:")
    print(f"- {summary_csv}")
    print(f"- {summary_md}")
    print(f"- {summary_png}")
    if decision_df is not None and not decision_df.empty:
        print(f"- {decision_csv}")

    if args.show:
        image = plt.imread(summary_png)
        plt.figure(figsize=(16, 9))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
