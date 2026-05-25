# src/stock_investment_dss/runner/run_iqn_vs_baseline_metrics_plots.py
"""Create thesis-oriented metric comparison plots for IQN vs FinRL baselines.

This runner reads the latest context-filtered
``iqn_vs_baseline_comparison_summary.csv`` and creates bar plots for the
metrics that matter in the thesis evaluation:

- total return / cumulative return
- annualized Sharpe ratio
- maximum drawdown
- CVaR / downside risk
- final portfolio value
- transaction costs / trades / turnover when available

It is intentionally separate from ``run_iqn_vs_baseline_comparison_plot.py``.
That runner plots real portfolio trajectories. This runner plots aggregate
metrics, so it can include the IQN multiseed mean row honestly and directly.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from stock_investment_dss.utilities.config import get_environment_variable
from stock_investment_dss.utilities.logging import (
    setup_run_logger,
    setup_system_logger,
)
from stock_investment_dss.utilities.paths import PROJECT_ROOT, create_run_paths

COMPARISON_SUMMARY_FILENAME = "iqn_vs_baseline_comparison_summary.csv"
FAIR_COMPACT_COMPARISON_FILENAME = "iqn_vs_baseline_fair_compact_comparison_summary.csv"
FAIR_SEED_DIAGNOSTIC_FILENAME = "iqn_vs_baseline_fair_seed_diagnostic_comparison_summary.csv"
RUN_KIND = "d_iqn_dss_iqn_vs_baseline_metrics_plots"

METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "total_return_pct": {
        "title": "Total Return: IQN vs FinRL Baselines",
        "ylabel": "Total return (%)",
        "filename": "iqn_vs_baseline_total_return_bar.png",
        "higher_is_better": True,
        "zero_line": True,
    },
    "annualized_sharpe": {
        "title": "Annualized Sharpe Ratio: IQN vs FinRL Baselines",
        "ylabel": "Annualized Sharpe",
        "filename": "iqn_vs_baseline_sharpe_bar.png",
        "higher_is_better": True,
        "zero_line": True,
    },
    "max_drawdown_pct": {
        "title": "Maximum Drawdown: IQN vs FinRL Baselines",
        "ylabel": "Max drawdown (%)",
        "filename": "iqn_vs_baseline_max_drawdown_bar.png",
        "higher_is_better": True,
        "zero_line": True,
    },
    "cvar_pct": {
        "title": "CVaR / Downside Risk: IQN vs FinRL Baselines",
        "ylabel": "CVaR (%)",
        "filename": "iqn_vs_baseline_cvar_bar.png",
        "higher_is_better": True,
        "zero_line": True,
    },
    "final_value": {
        "title": "Final Portfolio Value: IQN vs FinRL Baselines",
        "ylabel": "Final portfolio value",
        "filename": "iqn_vs_baseline_final_value_bar.png",
        "higher_is_better": True,
        "zero_line": False,
    },
    "total_transaction_cost": {
        "title": "Transaction Costs: IQN vs FinRL Baselines",
        "ylabel": "Transaction cost",
        "filename": "iqn_vs_baseline_transaction_cost_bar.png",
        "higher_is_better": False,
        "zero_line": True,
    },
    "total_trades": {
        "title": "Number of Trades: IQN vs FinRL Baselines",
        "ylabel": "Trades",
        "filename": "iqn_vs_baseline_trades_bar.png",
        "higher_is_better": None,
        "zero_line": True,
    },
    "turnover_estimate_pct": {
        "title": "Turnover Estimate: IQN vs FinRL Baselines",
        "ylabel": "Turnover estimate (%)",
        "filename": "iqn_vs_baseline_turnover_bar.png",
        "higher_is_better": None,
        "zero_line": True,
    },
}

SOURCE_COLOR_MAP = {
    "FinRL / SB3 baseline suite": "tab:blue",
    "MVO baseline": "tab:brown",
    "D-IQN-DSS / distributional RL / multiseed": "tab:orange",
    "D-IQN-DSS / distributional RL": "tab:orange",
}

STRATEGY_ORDER = [
    "a2c",
    "ddpg",
    "td3",
    "ppo",
    "sac",
    "mvo",
    "D-IQN-DSS IQN risk-aware / multiseed mean",
    "D-IQN-DSS IQN risk-aware / seed 1",
    "D-IQN-DSS IQN risk-aware / seed 2",
    "D-IQN-DSS IQN risk-aware / seed 3",
    "D-IQN-DSS IQN risk-aware / seed 4",
    "D-IQN-DSS IQN risk-aware / seed 5",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def get_int_environment_variable(name: str, default: int) -> int:
    value = get_environment_variable(name, default=str(default))
    try:
        return int(value or default)
    except Exception:
        return default


def get_bool_environment_variable(name: str, default: bool) -> bool:
    value = get_environment_variable(name, default=str(default).lower())
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def find_latest_comparison_summary_run() -> Path:
    runs_root = PROJECT_ROOT / "outputs" / "runs"

    if not runs_root.exists():
        raise FileNotFoundError(f"Run directory does not exist: {runs_root}")

    candidates = [
        path
        for path in runs_root.iterdir()
        if path.is_dir()
        and path.name.endswith("d_iqn_dss_iqn_vs_baseline_comparison_summary")
        and (path / "summary" / COMPARISON_SUMMARY_FILENAME).exists()
    ]

    if not candidates:
        raise FileNotFoundError(
            "Could not find an IQN vs baseline comparison summary run with "
            f"summary/{COMPARISON_SUMMARY_FILENAME}."
        )

    return sorted(candidates, key=lambda path: path.name)[-1]


def resolve_source_comparison_run_directory() -> Path:
    source_run_id = get_environment_variable(
        "STOCK_INVESTMENT_DSS_METRICS_PLOT_SOURCE_RUN_ID",
        default=None,
    ) or get_environment_variable(
        "STOCK_INVESTMENT_DSS_COMPARISON_PLOT_SOURCE_RUN_ID",
        default=None,
    )
    source_run_directory = get_environment_variable(
        "STOCK_INVESTMENT_DSS_METRICS_PLOT_SOURCE_RUN_DIRECTORY",
        default=None,
    ) or get_environment_variable(
        "STOCK_INVESTMENT_DSS_COMPARISON_PLOT_SOURCE_RUN_DIRECTORY",
        default=None,
    )

    if source_run_directory:
        path = Path(source_run_directory)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    if source_run_id:
        return PROJECT_ROOT / "outputs" / "runs" / source_run_id

    return find_latest_comparison_summary_run()



def resolve_comparison_table_path(source_run_directory: Path) -> tuple[Path, str]:
    """Resolve the comparison table used for metrics plots.

    By default, prefer the fair compact table produced by
    run_iqn_vs_baseline_comparison_summary.py. This prevents the old top-N logic
    from accidentally selecting only FinRL rows and dropping IQN/MVO.
    """
    explicit_filename = get_environment_variable(
        "STOCK_INVESTMENT_DSS_METRICS_PLOT_COMPARISON_FILENAME",
        default=None,
    )
    if explicit_filename:
        explicit_path = source_run_directory / "summary" / explicit_filename
        return explicit_path, "explicit_filename"

    use_compact = get_bool_environment_variable(
        "STOCK_INVESTMENT_DSS_METRICS_PLOT_USE_FAIR_COMPACT",
        default=True,
    )
    if use_compact:
        compact_path = (
            source_run_directory / "summary" / FAIR_COMPACT_COMPARISON_FILENAME
        )
        if compact_path.exists():
            return compact_path, "fair_compact"

    full_path = source_run_directory / "summary" / COMPARISON_SUMMARY_FILENAME
    return full_path, "full_comparison"


def make_plot_label(row: pd.Series) -> str:
    strategy = normalize_label(row.get("strategy"))
    if strategy.startswith("D-IQN-DSS IQN risk-aware / multiseed mean"):
        return "D-IQN-DSS mean"
    if strategy.startswith("D-IQN-DSS IQN risk-aware / seed"):
        return strategy.replace("D-IQN-DSS IQN risk-aware / ", "IQN ")
    return strategy


def strategy_sort_key(strategy: str) -> tuple[int, str]:
    if strategy in STRATEGY_ORDER:
        return STRATEGY_ORDER.index(strategy), strategy
    return len(STRATEGY_ORDER), strategy


def prepare_comparison_table(
    comparison_table: pd.DataFrame, top_n: int | None
) -> pd.DataFrame:
    table = comparison_table.copy()

    for metric_name in METRIC_DEFINITIONS:
        if metric_name in table.columns:
            table[metric_name] = pd.to_numeric(table[metric_name], errors="coerce")

    # Harmonize costs/trades across older summary formats.
    if (
        "total_transaction_cost" in table.columns
        and "transaction_cost" in table.columns
    ):
        table["total_transaction_cost"] = table["total_transaction_cost"].fillna(
            pd.to_numeric(table["transaction_cost"], errors="coerce")
        )
    if "total_trades" in table.columns and "trades" in table.columns:
        table["total_trades"] = table["total_trades"].fillna(
            pd.to_numeric(table["trades"], errors="coerce")
        )

    table["plot_label"] = table.apply(make_plot_label, axis=1)
    table["plot_sort_key"] = table["strategy"].map(
        lambda value: strategy_sort_key(normalize_label(value))[0]
    )
    table = table.sort_values(["plot_sort_key", "strategy", "source"], kind="stable")

    if top_n is not None and top_n > 0:
        # Keep deterministic thesis order rather than top-by-return order. The summary
        # runner has already done the fair filtering; here top_n only limits clutter.
        table = table.head(top_n).copy()

    return table.drop(columns=["plot_sort_key"], errors="ignore")




def copy_png_outputs_to_plots_folder(summary_directory: Path, plots_directory: Path) -> list[str]:
    """Copy generated PNG plots from summary/ to plots/.

    Older runners saved plots directly in summary/. We keep that behavior for
    backwards compatibility, but also mirror PNG files to plots/ so the run
    directory structure is easier to inspect.
    """

    plots_directory.mkdir(parents=True, exist_ok=True)
    copied_paths: list[str] = []

    for source_path in sorted(summary_directory.glob("*.png")):
        target_path = plots_directory / source_path.name
        shutil.copy2(source_path, target_path)
        copied_paths.append(str(target_path))

    return copied_paths


# ---------------------------------------------------------------------------
# Dashboard plotting
# ---------------------------------------------------------------------------


def _metric_table_for_axis(table: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    if metric_name not in table.columns:
        return pd.DataFrame()

    metric_table = table.copy()
    metric_table[metric_name] = pd.to_numeric(
        metric_table[metric_name], errors="coerce"
    )
    metric_table = metric_table[metric_table[metric_name].notna()].copy()
    if metric_table.empty:
        return metric_table

    metric_table = metric_table.sort_values(metric_name, ascending=True, kind="stable")
    return metric_table


def _add_horizontal_metric_panel(
    ax: Any,
    table: pd.DataFrame,
    metric_name: str,
    title: str,
    xlabel: str,
    zero_line: bool = True,
) -> None:
    metric_table = _metric_table_for_axis(table, metric_name)
    if metric_table.empty:
        ax.text(0.5, 0.5, f"Missing metric: {metric_name}", ha="center", va="center")
        ax.set_axis_off()
        return

    labels = metric_table["plot_label"].astype(str).tolist()
    values = metric_table[metric_name].astype(float).tolist()
    colors = colors_for_rows(metric_table)

    bars = ax.barh(labels, values, color=colors, alpha=0.9)
    if zero_line:
        ax.axvline(0, linestyle="--", linewidth=1.0, color="black", alpha=0.5)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    ax.tick_params(axis="y", labelsize=8)

    finite_values = [value for value in values if not math.isnan(value)]
    if not finite_values:
        return
    span = max(finite_values) - min(finite_values)
    offset = 0.015 * (span if span else 1.0)

    for bar, value in zip(bars, values, strict=False):
        if math.isnan(value):
            continue
        if value >= 0:
            x = value + offset
            ha = "left"
        else:
            x = value - offset
            ha = "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", ha=ha, fontsize=7)


def _add_risk_return_panel(ax: Any, table: pd.DataFrame) -> None:
    required_columns = {"total_return_pct", "max_drawdown_pct"}
    if not required_columns.issubset(table.columns):
        ax.text(0.5, 0.5, "Missing risk/return columns", ha="center", va="center")
        ax.set_axis_off()
        return

    scatter_table = table.copy()
    scatter_table["total_return_pct"] = pd.to_numeric(
        scatter_table["total_return_pct"], errors="coerce"
    )
    scatter_table["max_drawdown_pct"] = pd.to_numeric(
        scatter_table["max_drawdown_pct"], errors="coerce"
    )
    scatter_table = scatter_table[
        scatter_table["total_return_pct"].notna()
        & scatter_table["max_drawdown_pct"].notna()
    ].copy()

    if scatter_table.empty:
        ax.text(0.5, 0.5, "No risk/return values", ha="center", va="center")
        ax.set_axis_off()
        return

    for _, row in scatter_table.iterrows():
        source = normalize_label(row.get("source"))
        variant = normalize_label(row.get("variant"))
        strategy = normalize_label(row.get("strategy"))
        color = (
            "tab:red"
            if "multiseed_mean" in variant or "multiseed mean" in strategy
            else SOURCE_COLOR_MAP.get(source, "tab:gray")
        )
        x = float(row["max_drawdown_pct"])
        y = float(row["total_return_pct"])
        ax.scatter(x, y, color=color, s=55, alpha=0.85)
        ax.annotate(
            normalize_label(row.get("plot_label")),
            (x, y),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=7,
        )

    ax.axhline(0, linestyle="--", linewidth=1.0, color="black", alpha=0.5)
    ax.axvline(0, linestyle="--", linewidth=1.0, color="black", alpha=0.5)
    ax.set_title("Risk/Return trade-off", fontsize=11, fontweight="bold")
    ax.set_xlabel("Max drawdown (%) — closer to 0 is lower drawdown", fontsize=9)
    ax.set_ylabel("Total return (%)", fontsize=9)
    ax.grid(alpha=0.2)


def plot_summary_dashboard(table: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Create a v1-style overview dashboard from the fair compact table."""

    if table.empty:
        return {
            "metric": "summary_dashboard",
            "status": "skipped",
            "reason": "empty_table",
            "output_path": str(output_path),
        }

    fig, axes = plt.subplots(2, 2, figsize=(20, 13))
    fig.suptitle("StockDSS Fair Comparison Dashboard", fontsize=16, fontweight="bold")

    _add_horizontal_metric_panel(
        axes[0, 0],
        table,
        metric_name="total_return_pct",
        title="Total return by strategy",
        xlabel="Total return (%)",
        zero_line=True,
    )
    _add_horizontal_metric_panel(
        axes[0, 1],
        table,
        metric_name="max_drawdown_pct",
        title="Maximum drawdown by strategy",
        xlabel="Max drawdown (%)",
        zero_line=True,
    )
    _add_horizontal_metric_panel(
        axes[1, 0],
        table,
        metric_name="annualized_sharpe",
        title="Annualized Sharpe by strategy",
        xlabel="Annualized Sharpe",
        zero_line=True,
    )
    _add_risk_return_panel(axes[1, 1], table)

    legend_handles = []
    legend_seen: set[str] = set()
    for source, color in SOURCE_COLOR_MAP.items():
        if source in set(table["source"].astype(str)) and source not in legend_seen:
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=source))
            legend_seen.add(source)
    if any("multiseed mean" in str(strategy) for strategy in table["strategy"]):
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, color="tab:red", label="D-IQN-DSS multiseed mean")
        )
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.965))

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "metric": "summary_dashboard",
        "status": "ok",
        "rows_plotted": int(len(table)),
        "output_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def colors_for_rows(table: pd.DataFrame) -> list[str]:
    colors: list[str] = []
    for _, row in table.iterrows():
        source = normalize_label(row.get("source"))
        variant = normalize_label(row.get("variant"))
        strategy = normalize_label(row.get("strategy"))

        if "multiseed_mean" in variant or "multiseed mean" in strategy:
            colors.append("tab:red")
        elif source in SOURCE_COLOR_MAP:
            colors.append(SOURCE_COLOR_MAP[source])
        elif source.startswith("D-IQN-DSS"):
            colors.append("tab:orange")
        else:
            colors.append("tab:gray")
    return colors


def plot_metric_bar(
    table: pd.DataFrame,
    metric_name: str,
    output_path: Path,
    title: str,
    ylabel: str,
    zero_line: bool = True,
) -> dict[str, Any]:
    if metric_name not in table.columns:
        return {
            "metric": metric_name,
            "status": "skipped",
            "reason": "missing_column",
            "output_path": str(output_path),
        }

    metric_table = table[
        ["strategy", "source", "variant", "plot_label", metric_name]
    ].copy()
    metric_table[metric_name] = pd.to_numeric(
        metric_table[metric_name], errors="coerce"
    )
    metric_table = metric_table[metric_table[metric_name].notna()].copy()

    if metric_table.empty:
        return {
            "metric": metric_name,
            "status": "skipped",
            "reason": "no_non_missing_values",
            "output_path": str(output_path),
        }

    labels = metric_table["plot_label"].astype(str).tolist()
    values = metric_table[metric_name].astype(float).tolist()
    colors = colors_for_rows(metric_table)

    fig_width = max(12, min(24, 0.9 * len(labels) + 6))
    fig, ax = plt.subplots(figsize=(fig_width, 7))
    bars = ax.bar(labels, values, color=colors, alpha=0.9)

    if zero_line:
        ax.axhline(0, linestyle="--", linewidth=1.0, color="black", alpha=0.6)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Strategy / agent")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35, labelsize=9)

    for bar, value in zip(bars, values, strict=False):
        if math.isnan(value):
            continue
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        y_offset = 0.01 * (
            max(values) - min(values) if max(values) != min(values) else 1.0
        )
        label_y = height + y_offset if height >= 0 else height - y_offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{value:.2f}",
            ha="center",
            va=va,
            fontsize=8,
            rotation=0,
        )

    # Add a compact source legend.
    legend_handles = []
    legend_seen: set[str] = set()
    for source, color in SOURCE_COLOR_MAP.items():
        if (
            source in set(metric_table["source"].astype(str))
            and source not in legend_seen
        ):
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, color=color, label=source)
            )
            legend_seen.add(source)
    if any("multiseed mean" in str(strategy) for strategy in metric_table["strategy"]):
        legend_handles.append(
            plt.Rectangle(
                (0, 0), 1, 1, color="tab:red", label="D-IQN-DSS multiseed mean"
            )
        )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="best", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "metric": metric_name,
        "status": "ok",
        "rows_plotted": int(len(metric_table)),
        "output_path": str(output_path),
        "min": safe_float(metric_table[metric_name].min()),
        "max": safe_float(metric_table[metric_name].max()),
        "mean": safe_float(metric_table[metric_name].mean()),
    }


def plot_risk_return_scatter(table: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    required = {"total_return_pct", "max_drawdown_pct"}
    if not required.issubset(table.columns):
        return {
            "metric": "risk_return_scatter",
            "status": "skipped",
            "reason": "missing_required_columns",
            "output_path": str(output_path),
        }

    scatter_table = table.copy()
    scatter_table["total_return_pct"] = pd.to_numeric(
        scatter_table["total_return_pct"], errors="coerce"
    )
    scatter_table["max_drawdown_pct"] = pd.to_numeric(
        scatter_table["max_drawdown_pct"], errors="coerce"
    )
    scatter_table = scatter_table[
        scatter_table["total_return_pct"].notna()
        & scatter_table["max_drawdown_pct"].notna()
    ].copy()

    if scatter_table.empty:
        return {
            "metric": "risk_return_scatter",
            "status": "skipped",
            "reason": "no_non_missing_values",
            "output_path": str(output_path),
        }

    fig, ax = plt.subplots(figsize=(12, 7))

    for _, row in scatter_table.iterrows():
        source = normalize_label(row.get("source"))
        variant = normalize_label(row.get("variant"))
        strategy = normalize_label(row.get("strategy"))
        color = (
            "tab:red"
            if "multiseed_mean" in variant or "multiseed mean" in strategy
            else SOURCE_COLOR_MAP.get(source, "tab:gray")
        )
        ax.scatter(
            float(row["max_drawdown_pct"]),
            float(row["total_return_pct"]),
            color=color,
            s=75,
            alpha=0.85,
        )
        ax.annotate(
            normalize_label(row.get("plot_label")),
            (float(row["max_drawdown_pct"]), float(row["total_return_pct"])),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )

    ax.axhline(0, linestyle="--", linewidth=1.0, color="black", alpha=0.6)
    ax.axvline(0, linestyle="--", linewidth=1.0, color="black", alpha=0.6)
    ax.set_title("Risk/Return Trade-off: IQN vs FinRL Baselines")
    ax.set_xlabel("Max drawdown (%) — closer to 0 is lower drawdown")
    ax.set_ylabel("Total return (%)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "metric": "risk_return_scatter",
        "status": "ok",
        "rows_plotted": int(len(scatter_table)),
        "output_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build_metric_rankings(table: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for metric_name, definition in METRIC_DEFINITIONS.items():
        if metric_name not in table.columns:
            continue
        metric_table = table.copy()
        metric_table[metric_name] = pd.to_numeric(
            metric_table[metric_name], errors="coerce"
        )
        metric_table = metric_table[metric_table[metric_name].notna()].copy()
        if metric_table.empty:
            continue

        higher_is_better = definition.get("higher_is_better")
        if higher_is_better is None:
            continue

        metric_table = metric_table.sort_values(
            metric_name,
            ascending=not bool(higher_is_better),
            kind="stable",
        ).copy()
        metric_table["metric_rank"] = range(1, len(metric_table) + 1)

        for _, row in metric_table.iterrows():
            records.append(
                {
                    "metric": metric_name,
                    "metric_rank": int(row["metric_rank"]),
                    "strategy": normalize_label(row.get("strategy")),
                    "source": normalize_label(row.get("source")),
                    "variant": normalize_label(row.get("variant")),
                    "value": safe_float(row.get(metric_name)),
                    "higher_is_better": higher_is_better,
                }
            )

    return pd.DataFrame(records)


def build_markdown_report(
    table: pd.DataFrame,
    ranking_table: pd.DataFrame,
    plot_records: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# IQN vs FinRL Baseline Metrics Plots")
    lines.append("")
    lines.append(
        "This report visualizes the fair comparison summary as metric-level bar plots."
    )
    lines.append(
        "Unlike the trajectory plot, these charts can include the IQN multiseed mean row because it is an aggregate metrics row, not a single portfolio trajectory."
    )
    lines.append("")
    lines.append("## Rows included")
    lines.append("")
    lines.append(f"- Rows: {len(table)}")
    if "source" in table.columns:
        for source, count in table["source"].value_counts(dropna=False).items():
            lines.append(f"- {source}: {count}")
    lines.append("")

    if not ranking_table.empty:
        lines.append("## Best rows by metric")
        lines.append("")
        for metric in [
            "total_return_pct",
            "annualized_sharpe",
            "max_drawdown_pct",
            "cvar_pct",
        ]:
            subset = ranking_table[
                (ranking_table["metric"] == metric)
                & (ranking_table["metric_rank"] == 1)
            ]
            if not subset.empty:
                row = subset.iloc[0]
                lines.append(
                    f"- **{metric}**: {row['strategy']} ({row['source']}) = {row['value']:.4f}"
                )
        lines.append("")

    lines.append("## Generated plots")
    lines.append("")
    for record in plot_records:
        status = record.get("status")
        metric = record.get("metric")
        output = record.get("output_path")
        if status == "ok":
            lines.append(f"- {metric}: `{output}`")
        else:
            lines.append(f"- {metric}: skipped ({record.get('reason')})")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    log_level = (
        get_environment_variable(
            "STOCK_INVESTMENT_DSS_LOG_LEVEL",
            default="INFO",
        )
        or "INFO"
    )

    system_logger = setup_system_logger(log_level=log_level)
    system_logger.info("Starting StockInvestmentDSS IQN vs baseline metrics plots.")
    system_logger.info("Project root: %s", PROJECT_ROOT)

    run_paths = None

    try:
        source_comparison_run_directory = resolve_source_comparison_run_directory()
        comparison_path, comparison_table_kind = resolve_comparison_table_path(
            source_comparison_run_directory
        )

        if not comparison_path.exists():
            raise FileNotFoundError(f"Missing comparison table: {comparison_path}")

        top_n = get_int_environment_variable(
            "STOCK_INVESTMENT_DSS_METRICS_PLOT_TOP_N",
            default=0,
        )
        if top_n <= 0:
            top_n = None
        include_optional_trade_metrics = get_bool_environment_variable(
            "STOCK_INVESTMENT_DSS_METRICS_PLOT_INCLUDE_TRADE_METRICS",
            default=True,
        )

        run_paths = create_run_paths(RUN_KIND)
        run_logger = setup_run_logger(run_paths, log_level=log_level)

        run_logger.info("Created run directory: %s", run_paths.run_directory)
        run_logger.info("Run id: %s", run_paths.run_id)
        run_logger.info("Source comparison run: %s", source_comparison_run_directory)
        run_logger.info("Comparison table: %s", comparison_path)
        run_logger.info("Comparison table kind: %s", comparison_table_kind)
        run_logger.info("Top N rows: %s", top_n)
        run_logger.info(
            "Include optional trade metrics: %s", include_optional_trade_metrics
        )

        comparison_table = pd.read_csv(comparison_path)
        selected_table = prepare_comparison_table(comparison_table, top_n=top_n)
        ranking_table = build_metric_rankings(selected_table)

        selected_rows_path = (
            run_paths.data_directory / "iqn_vs_baseline_metrics_plot_selected_rows.csv"
        )
        ranking_table_path = (
            run_paths.data_directory / "iqn_vs_baseline_metric_rankings.csv"
        )
        selected_table.to_csv(selected_rows_path, index=False)
        ranking_table.to_csv(ranking_table_path, index=False)

        plot_records: list[dict[str, Any]] = []
        plot_records.append(
            plot_summary_dashboard(
                table=selected_table,
                output_path=run_paths.summary_directory
                / "iqn_vs_baseline_summary_dashboard.png",
            )
        )

        metric_names = [
            "total_return_pct",
            "annualized_sharpe",
            "max_drawdown_pct",
            "cvar_pct",
            "final_value",
        ]
        if include_optional_trade_metrics:
            metric_names.extend(
                ["total_transaction_cost", "total_trades", "turnover_estimate_pct"]
            )

        for metric_name in metric_names:
            definition = METRIC_DEFINITIONS[metric_name]
            plot_records.append(
                plot_metric_bar(
                    table=selected_table,
                    metric_name=metric_name,
                    output_path=run_paths.summary_directory / definition["filename"],
                    title=definition["title"],
                    ylabel=definition["ylabel"],
                    zero_line=bool(definition.get("zero_line", True)),
                )
            )

        plot_records.append(
            plot_risk_return_scatter(
                table=selected_table,
                output_path=run_paths.summary_directory
                / "iqn_vs_baseline_risk_return_scatter.png",
            )
        )

        copied_plot_paths = copy_png_outputs_to_plots_folder(
            summary_directory=run_paths.summary_directory,
            plots_directory=run_paths.plots_directory,
        )

        markdown_report_path = (
            run_paths.summary_directory / "iqn_vs_baseline_metrics_plots.md"
        )
        summary_path = (
            run_paths.summary_directory / "iqn_vs_baseline_metrics_plots_summary.json"
        )

        markdown_report = build_markdown_report(
            table=selected_table,
            ranking_table=ranking_table,
            plot_records=plot_records,
        )
        markdown_report_path.write_text(markdown_report, encoding="utf-8")

        source_counts = (
            selected_table["source"].value_counts(dropna=False).to_dict()
            if "source" in selected_table.columns
            else {}
        )
        model_family_counts = (
            selected_table["model_family"].value_counts(dropna=False).to_dict()
            if "model_family" in selected_table.columns
            else {}
        )
        successful_plots = [
            record for record in plot_records if record.get("status") == "ok"
        ]
        skipped_plots = [
            record for record in plot_records if record.get("status") != "ok"
        ]

        summary = {
            "status": "ok",
            "project_name": "StockInvestmentDSS",
            "prototype_name": "D-IQN-DSS",
            "run_id": run_paths.run_id,
            "project_root": str(PROJECT_ROOT),
            "run_directory": str(run_paths.run_directory),
            "source_comparison_run_directory": str(source_comparison_run_directory),
            "comparison_summary_path": str(comparison_path),
            "comparison_table_kind": comparison_table_kind,
            "rows_selected": int(len(selected_table)),
            "source_counts": source_counts,
            "model_family_counts": model_family_counts,
            "successful_plot_count": int(len(successful_plots)),
            "skipped_plot_count": int(len(skipped_plots)),
            "plots": plot_records,
            "copied_plot_paths": copied_plot_paths,
            "outputs": {
                "comparison_table_kind": comparison_table_kind,
                "selected_rows_path": str(selected_rows_path),
                "ranking_table_path": str(ranking_table_path),
                "plots_directory": str(run_paths.plots_directory),
                "copied_plot_paths": copied_plot_paths,
                "markdown_report_path": str(markdown_report_path),
                "summary_path": str(summary_path),
            },
            "interpretation": (
                "This runner creates metric-level comparison plots from the fair "
                "IQN-vs-baseline compact table by default. This prevents top-N "
                "selection from accidentally dropping IQN or MVO rows and keeps "
                "aggregate metrics separate from trajectory plots."
            ),
        }
        write_json(summary_path, summary)

        run_logger.info("IQN vs baseline metrics plots completed.")
        run_logger.info("Rows selected: %s", len(selected_table))
        run_logger.info("Source counts: %s", source_counts)
        run_logger.info("Successful plots: %s", len(successful_plots))
        run_logger.info("Skipped plots: %s", len(skipped_plots))
        run_logger.info("Wrote selected rows: %s", selected_rows_path)
        run_logger.info("Wrote ranking table: %s", ranking_table_path)
        run_logger.info("Copied plot PNGs to plots directory: %s", len(copied_plot_paths))
        run_logger.info("Wrote markdown report: %s", markdown_report_path)
        run_logger.info("Wrote summary: %s", summary_path)

        system_logger.info(
            "StockInvestmentDSS IQN vs baseline metrics plots completed successfully."
        )
        return 0

    except Exception:
        system_logger.exception(
            "StockInvestmentDSS IQN vs baseline metrics plots failed."
        )
        if run_paths is not None:
            run_logger = setup_run_logger(run_paths, log_level=log_level)
            run_logger.exception("Run failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
