"""Summary dashboard: 4-panel strategy comparison across all tiers.

Produces a matplotlib 2x2 figure ("StockDSS Runner Summary") comparing
algorithmic trading baselines, FinRL parametric RL baselines, and the
D-IQN-DSS distributional RL system across three portfolio performance
metrics: total return, maximum drawdown, and annualised Sharpe ratio.

Panel 4 shows the aggregate IQN action-distribution across the full
evaluation window (BUY / HOLD / SELL / REBALANCE / CHANGE_STRATEGY),
replacing the V1 single-timestep risk-adjusted score panel.

Public entry point
------------------
build_summary_dashboard(
    algorithmic_summary_csv,
    finrl_aggregate_csv,
    iqn_metrics_csv,
    iqn_eval_records_csv,
    *,
    run_name="d_iqn_dss_summary_dashboard",
    output_dir=None,
    max_display_strategies=18,
) -> Path
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from stock_investment_dss.utilities.paths import create_run_paths

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

GROUP_COLORS: dict[str, str] = {
    "Algorithmic Trading / non-RL": "tab:green",
    "FinRL / baseline": "tab:blue",
    "D-IQN-DSS": "tab:orange",
}

REQUIRED_METRIC_COLUMNS: list[str] = [
    "strategy",
    "source",
    "start_value",
    "end_value",
    "profit_loss",
    "total_return_pct",
    "max_drawdown_pct",
    "annualized_sharpe",
]

_KNOWN_ACTIONS: list[str] = [
    "BUY",
    "HOLD",
    "SELL",
    "REBALANCE",
    "CHANGE_STRATEGY",
]

_INITIAL_VALUE: float = 1_000_000.0
_DEFAULT_MAX_STRATEGIES: int = 999

# ---------------------------------------------------------------------------
# Shared helpers (adopted from V1 sources)
# ---------------------------------------------------------------------------


def print_section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def get_group_colors(df: pd.DataFrame) -> list[str]:
    return [GROUP_COLORS.get(str(source), "tab:gray") for source in df["source"]]


def add_group_legend(fig) -> None:
    handles = [
        Patch(color=GROUP_COLORS["FinRL / baseline"], label="FinRL / baseline"),
        Patch(
            color=GROUP_COLORS["Algorithmic Trading / non-RL"],
            label="Algorithmic Trading / non-RL",
        ),
        Patch(color=GROUP_COLORS["D-IQN-DSS"], label="D-IQN-DSS"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.925),
    )


def get_strategy_label_color(strategy: str, source: Optional[str] = None) -> str:
    """Return the group color for a strategy y-axis label."""
    source_text = str(source).strip() if source is not None else ""
    strategy_lower = str(strategy).strip().lower()
    if source_text in GROUP_COLORS:
        return GROUP_COLORS[source_text]
    _FINRL_NAMES = {"a2c", "ddpg", "ppo", "sac", "td3", "mvo"}
    if strategy_lower == "d_iqn_dss" or "iqn" in strategy_lower:
        return GROUP_COLORS["D-IQN-DSS"]
    if strategy_lower in _FINRL_NAMES:
        return GROUP_COLORS["FinRL / baseline"]
    return GROUP_COLORS["Algorithmic Trading / non-RL"]


def color_y_ticklabels(ax, plot_df: pd.DataFrame) -> None:
    """Color y-axis strategy labels to match their visual group."""
    if plot_df.empty or "strategy" not in plot_df.columns:
        return
    source_by_strategy = (
        plot_df[["strategy", "source"]]
        .drop_duplicates(subset=["strategy"], keep="last")
        .set_index("strategy")["source"]
        .to_dict()
    )
    for label in ax.get_yticklabels():
        strategy = label.get_text().strip()
        label.set_color(
            get_strategy_label_color(strategy, source_by_strategy.get(strategy))
        )
        label.set_fontweight("bold")


def finalize_barh_axis(ax, plot_df: pd.DataFrame, value_column: str) -> None:
    """Apply shared styling and NA/zero annotations to a horizontal bar axis."""
    # Font size scales with number of strategies (smaller for more strategies)
    n_rows = len(plot_df)
    label_size = 6 if n_rows > 50 else (7 if n_rows > 30 else 8)
    ax.tick_params(axis="y", labelsize=label_size, pad=3)
    ax.grid(True, axis="x", alpha=0.25)
    # Tighter margins for many strategies
    ax.margins(y=0.05 if n_rows > 30 else 0.12)
    color_y_ticklabels(ax, plot_df)

    if not plot_df.empty:
        ax.set_ylim(-0.75, len(plot_df) - 0.25)

    values = pd.to_numeric(plot_df[value_column], errors="coerce")
    display_values = pd.to_numeric(plot_df["_display_value"], errors="coerce")
    finite_values = pd.concat([values, display_values], ignore_index=True).dropna()

    if finite_values.empty:
        return

    vmin = float(finite_values.min())
    vmax = float(finite_values.max())

    if vmin < 0 and vmax <= 0:
        ax.set_xlim(vmin * 1.08, 0.6)
    elif vmin >= 0 and vmax > 0:
        ax.set_xlim(min(-0.1, vmin - 0.02 * max(1.0, abs(vmax))), vmax * 1.08)
    else:
        left = vmin * 1.08 if vmin < 0 else -0.1
        right = vmax * 1.08 if vmax > 0 else 0.6
        if left == right:
            right = left + 1.0
        ax.set_xlim(left, right)

    for patch, raw_value, is_missing in zip(
        ax.patches, plot_df[value_column], plot_df["_is_missing"]
    ):
        y = patch.get_y() + patch.get_height() / 2
        if bool(is_missing):
            ax.text(0.02, y, "NA", va="center", ha="left", fontsize=7, color="dimgray")
        elif float(raw_value) == 0.0:
            ax.scatter([0.0], [y], s=18, color=patch.get_facecolor(), zorder=3)
            ax.text(0.02, y, "0.0", va="center", ha="left", fontsize=7, color="dimgray")


def prepare_metric_plot_df(
    metrics_df: pd.DataFrame,
    metric_column: str,
    ascending: bool,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Return a copy sorted for a single metric panel with display helpers."""
    plot_df = metrics_df.copy()
    plot_df[metric_column] = pd.to_numeric(plot_df[metric_column], errors="coerce")
    plot_df["_display_value"] = plot_df[metric_column].fillna(fill_value)
    plot_df["_is_missing"] = plot_df[metric_column].isna()
    plot_df = plot_df.sort_values(
        by=["_display_value", "rank"],
        ascending=[ascending, True],
        na_position="last",
    ).reset_index(drop=True)
    return plot_df


def select_plot_metrics(
    metrics_df: pd.DataFrame,
    max_display_strategies: int = _DEFAULT_MAX_STRATEGIES,
) -> pd.DataFrame:
    """Select top-N strategies while ensuring all source groups are represented."""
    selected = metrics_df.head(max_display_strategies).copy()
    for source in ["FinRL / baseline", "Algorithmic Trading / non-RL", "D-IQN-DSS"]:
        source_rows = metrics_df[metrics_df["source"].astype(str).str.strip() == source]
        if source_rows.empty:
            continue
        already_present = selected["source"].astype(str).str.strip().eq(source).any()
        if not already_present:
            selected = pd.concat([selected, source_rows.head(1)], ignore_index=True)
    selected = selected.drop_duplicates(subset=["strategy", "source"], keep="first")
    selected = selected.sort_values("rank", ascending=True).reset_index(drop=True)
    return selected


# ---------------------------------------------------------------------------
# V2 data loaders
# ---------------------------------------------------------------------------


def _load_algorithmic_v2(csv_path: Path) -> pd.DataFrame:
    """Load algorithmic baseline metrics from the V2 grid summary CSV."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        log.warning("Could not read algorithmic summary CSV %s: %s", csv_path, exc)
        return pd.DataFrame(columns=REQUIRED_METRIC_COLUMNS)

    df = df[df["status"].astype(str).str.strip() == "ok"].copy()

    rows = []
    for _, row in df.iterrows():
        for col in ["total_return_pct", "max_drawdown_pct", "annualized_sharpe"]:
            if pd.isna(row.get(col)):
                log.warning(
                    "Skipping algorithmic row with missing %s: %s",
                    col,
                    row.get("config_label"),
                )
                break
        else:
            scope = str(row.get("scope", "")).strip()
            ticker = str(row.get("ticker", "")).strip()
            config_label = str(row.get("config_label", "")).strip()
            if scope == "single_ticker" and ticker and ticker.lower() != "nan":
                strategy = f"{ticker.upper()}_{config_label}"
            else:
                strategy = config_label

            total_return_pct = float(row["total_return_pct"])
            start_value = _INITIAL_VALUE
            end_value = start_value * (1.0 + total_return_pct / 100.0)
            profit_loss = end_value - start_value
            rows.append(
                {
                    "strategy": strategy,
                    "source": "Algorithmic Trading / non-RL",
                    "start_value": start_value,
                    "end_value": end_value,
                    "profit_loss": profit_loss,
                    "total_return_pct": total_return_pct,
                    "max_drawdown_pct": float(row["max_drawdown_pct"]),
                    "annualized_sharpe": float(row["annualized_sharpe"]),
                }
            )

    if not rows:
        return pd.DataFrame(columns=REQUIRED_METRIC_COLUMNS)
    return pd.DataFrame(rows, columns=REQUIRED_METRIC_COLUMNS)


def _load_finrl_v2(csv_path: Path) -> pd.DataFrame:
    """Load FinRL multiseed aggregate metrics from the V2 summary CSV."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        log.warning("Could not read FinRL aggregate CSV %s: %s", csv_path, exc)
        return pd.DataFrame(columns=REQUIRED_METRIC_COLUMNS)

    rows = []
    for _, row in df.iterrows():
        agent_name = str(row.get("agent_name", "")).strip()
        if not agent_name:
            continue

        total_return_pct = pd.to_numeric(
            row.get("total_return_pct_mean"), errors="coerce"
        )
        max_drawdown_pct = pd.to_numeric(
            row.get("max_drawdown_pct_mean"), errors="coerce"
        )
        annualized_sharpe = pd.to_numeric(
            row.get("annualized_sharpe_mean"), errors="coerce"
        )

        end_value = pd.to_numeric(row.get("final_value_mean"), errors="coerce")
        if pd.isna(end_value) and not pd.isna(total_return_pct):
            end_value = _INITIAL_VALUE * (1.0 + float(total_return_pct) / 100.0)

        if pd.isna(end_value):
            log.warning("Skipping FinRL row with missing end_value: %s", agent_name)
            continue

        start_value = _INITIAL_VALUE
        profit_loss = float(end_value) - start_value
        rows.append(
            {
                "strategy": agent_name.upper(),
                "source": "FinRL / baseline",
                "start_value": start_value,
                "end_value": float(end_value),
                "profit_loss": profit_loss,
                "total_return_pct": (
                    float(total_return_pct)
                    if not pd.isna(total_return_pct)
                    else float("nan")
                ),
                "max_drawdown_pct": (
                    float(max_drawdown_pct)
                    if not pd.isna(max_drawdown_pct)
                    else float("nan")
                ),
                "annualized_sharpe": (
                    float(annualized_sharpe)
                    if not pd.isna(annualized_sharpe)
                    else float("nan")
                ),
            }
        )

    if not rows:
        return pd.DataFrame(columns=REQUIRED_METRIC_COLUMNS)
    return pd.DataFrame(rows, columns=REQUIRED_METRIC_COLUMNS)


def _load_iqn_v2(csv_path: Path) -> pd.DataFrame:
    """Load IQN multiseed final-step metrics and aggregate across seeds."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        log.warning("Could not read IQN metrics CSV %s: %s", csv_path, exc)
        return pd.DataFrame(columns=REQUIRED_METRIC_COLUMNS)

    for col in ["total_return_pct", "max_drawdown_pct", "annualized_sharpe"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["total_return_pct"])
    if df.empty:
        log.warning("IQN metrics CSV has no usable rows: %s", csv_path)
        return pd.DataFrame(columns=REQUIRED_METRIC_COLUMNS)

    mean_return = float(df["total_return_pct"].mean())
    mean_drawdown = (
        float(df["max_drawdown_pct"].mean())
        if "max_drawdown_pct" in df.columns
        else float("nan")
    )
    mean_sharpe = (
        float(df["annualized_sharpe"].mean())
        if "annualized_sharpe" in df.columns
        else float("nan")
    )

    start_value = _INITIAL_VALUE
    end_value = start_value * (1.0 + mean_return / 100.0)
    profit_loss = end_value - start_value

    return pd.DataFrame(
        [
            {
                "strategy": "D-IQN-DSS",
                "source": "D-IQN-DSS",
                "start_value": start_value,
                "end_value": end_value,
                "profit_loss": profit_loss,
                "total_return_pct": mean_return,
                "max_drawdown_pct": mean_drawdown,
                "annualized_sharpe": mean_sharpe,
            }
        ],
        columns=REQUIRED_METRIC_COLUMNS,
    )


def _load_all_metrics_v2(
    algo_csv: Path,
    finrl_csv: Path,
    iqn_csv: Path,
) -> pd.DataFrame:
    """Merge metrics from all three strategy tiers into a single ranked DataFrame."""
    frames = [
        _load_algorithmic_v2(algo_csv),
        _load_finrl_v2(finrl_csv),
        _load_iqn_v2(iqn_csv),
    ]
    frames = [f for f in frames if not f.empty]

    if not frames:
        return pd.DataFrame(columns=REQUIRED_METRIC_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)

    for col in [
        "start_value",
        "end_value",
        "profit_loss",
        "total_return_pct",
        "max_drawdown_pct",
        "annualized_sharpe",
    ]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined[
        pd.to_numeric(combined["end_value"], errors="coerce").notna()
    ].copy()

    combined = combined.drop_duplicates(subset=["strategy", "source"], keep="last")
    combined = combined.sort_values(
        by=["end_value", "total_return_pct"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    combined.insert(0, "rank", range(1, len(combined) + 1))
    return combined


# ---------------------------------------------------------------------------
# Action distribution parser
# ---------------------------------------------------------------------------


def _extract_action_data(
    eval_records_csv: Path,
) -> tuple[dict[str, int], dict[str, float]]:
    """Parse and aggregate IQN action counts at the final training step.

    Returns (action_counts, action_percentages) for all _KNOWN_ACTIONS.
    Missing actions default to zero.
    """
    empty_counts: dict[str, int] = {a: 0 for a in _KNOWN_ACTIONS}
    empty_pcts: dict[str, float] = {a: 0.0 for a in _KNOWN_ACTIONS}

    try:
        df = pd.read_csv(eval_records_csv)
    except Exception as exc:
        log.warning("Could not read IQN eval records CSV %s: %s", eval_records_csv, exc)
        return empty_counts, empty_pcts

    if "action_counts" not in df.columns:
        log.warning("Column 'action_counts' not found in %s", eval_records_csv)
        return empty_counts, empty_pcts

    if "seed" in df.columns:
        final_rows = df.sort_values("train_step").groupby("seed").tail(1)
    else:
        final_rows = df[df["train_step"] == df["train_step"].max()]

    totals: dict[str, int] = {a: 0 for a in _KNOWN_ACTIONS}
    for ac_str in final_rows["action_counts"]:
        try:
            counts = ast.literal_eval(str(ac_str))
            for action_label, count in counts.items():
                if action_label in totals:
                    totals[action_label] += int(count)
        except (ValueError, SyntaxError, TypeError):
            log.warning("Could not parse action_counts entry: %r", ac_str)

    grand_total = sum(totals.values())
    if grand_total == 0:
        return totals, empty_pcts

    pcts = {a: totals[a] / grand_total * 100.0 for a in _KNOWN_ACTIONS}
    return totals, pcts


# ---------------------------------------------------------------------------
# Panel 4 renderer
# ---------------------------------------------------------------------------


def _plot_action_distribution(
    ax,
    action_pcts: dict[str, float],
    action_counts: dict[str, int],
) -> None:
    """Render the IQN action distribution as a horizontal bar chart.

    Actions are displayed top-to-bottom: BUY, HOLD, SELL, REBALANCE,
    CHANGE_STRATEGY. CHANGE_STRATEGY at 0% receives a "(not yet implemented)"
    annotation.
    """
    # Reverse so that BUY appears at the top of the chart
    display_actions = list(reversed(_KNOWN_ACTIONS))
    values = [action_pcts.get(a, 0.0) for a in display_actions]
    colors = [
        (
            "tab:gray"
            if a == "CHANGE_STRATEGY" and action_pcts.get(a, 0.0) == 0.0
            else GROUP_COLORS["D-IQN-DSS"]
        )
        for a in display_actions
    ]

    ax.barh(display_actions, values, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percent of eval timesteps (%)")
    ax.set_title(
        "IQN action distribution (final eval, across all seeds)",
        fontsize=12,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, axis="x", alpha=0.25)

    for i, action in enumerate(display_actions):
        val = action_pcts.get(action, 0.0)
        if action == "CHANGE_STRATEGY" and val == 0.0:
            ax.scatter([0.0], [i], s=18, color="tab:gray", zorder=3)
            ax.text(
                0.5,
                i,
                "(not yet implemented)",
                va="center",
                ha="left",
                fontsize=7,
                color="gray",
                style="italic",
            )
        else:
            ax.text(val + 0.5, i, f"{val:.1f}%", va="center", ha="left", fontsize=8)

    for label in ax.get_yticklabels():
        action = label.get_text().strip()
        if action == "CHANGE_STRATEGY":
            label.set_color("gray")
        else:
            label.set_color(GROUP_COLORS["D-IQN-DSS"])
        label.set_fontweight("bold")


# ---------------------------------------------------------------------------
# Figure renderer
# ---------------------------------------------------------------------------


def _plot_dashboard_figure(
    plot_metrics_df: pd.DataFrame,
    action_pcts: dict[str, float],
    action_counts: dict[str, int],
    output_path: Path,
    run_dir_name: str,
) -> None:
    """Render and save the 4-panel summary dashboard PNG."""
    if plot_metrics_df.empty:
        fig = plt.figure(figsize=(14, 6))
        fig.text(
            0.5,
            0.5,
            "No strategy metrics found",
            ha="center",
            va="center",
            fontsize=16,
        )
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    n_strategies = len(plot_metrics_df)
    # Dynamic height: minimum 12, scales with strategies (0.30 per strategy)
    panel_height = max(12, int(0.30 * n_strategies))
    fig = plt.figure(figsize=(20, panel_height))

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
        run_dir_name,
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

    # Panel 1: Total return
    return_df = prepare_metric_plot_df(
        plot_metrics_df, "total_return_pct", ascending=True, fill_value=0.0
    )
    ax_return.barh(
        return_df["strategy"],
        return_df["_display_value"],
        color=get_group_colors(return_df),
    )
    ax_return.axvline(0.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax_return.set_title("Total return by strategy", fontsize=12, fontweight="bold")
    ax_return.set_xlabel("Total return (%)")
    finalize_barh_axis(ax_return, return_df, "total_return_pct")

    # Panel 2: Maximum drawdown
    drawdown_df = prepare_metric_plot_df(
        plot_metrics_df, "max_drawdown_pct", ascending=True, fill_value=0.0
    )
    ax_drawdown.barh(
        drawdown_df["strategy"],
        drawdown_df["_display_value"],
        color=get_group_colors(drawdown_df),
    )
    ax_drawdown.axvline(0.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax_drawdown.set_title(
        "Maximum drawdown by strategy", fontsize=12, fontweight="bold"
    )
    ax_drawdown.set_xlabel("Max drawdown (%)")
    finalize_barh_axis(ax_drawdown, drawdown_df, "max_drawdown_pct")

    # Panel 3: Annualized Sharpe
    sharpe_df = prepare_metric_plot_df(
        plot_metrics_df, "annualized_sharpe", ascending=True, fill_value=0.0
    )
    ax_sharpe.barh(
        sharpe_df["strategy"],
        sharpe_df["_display_value"],
        color=get_group_colors(sharpe_df),
    )
    ax_sharpe.axvline(0.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax_sharpe.set_title("Annualized Sharpe by strategy", fontsize=12, fontweight="bold")
    ax_sharpe.set_xlabel("Sharpe")
    finalize_barh_axis(ax_sharpe, sharpe_df, "annualized_sharpe")

    # Panel 4: IQN action distribution
    _plot_action_distribution(ax_iqn, action_pcts, action_counts)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_markdown_summary(
    metrics_df: pd.DataFrame,
    action_pcts: dict[str, float],
    action_counts: dict[str, int],
    output_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# StockDSS result comparison")
    lines.append("")

    if metrics_df.empty:
        lines.append("No metrics found.")
    else:
        display_cols = ["rank"] + [
            c for c in REQUIRED_METRIC_COLUMNS if c in metrics_df.columns
        ]
        display_cols = [c for c in display_cols if c in metrics_df.columns]
        lines.append(metrics_df[display_cols].to_markdown(index=False))

    lines.append("")
    lines.append("## IQN Action Distribution (final eval, across all seeds)")
    lines.append("")
    lines.append("| action | count | percent |")
    lines.append("| --- | ---: | ---: |")
    for action in _KNOWN_ACTIONS:
        count = action_counts.get(action, 0)
        pct = action_pcts.get(action, 0.0)
        note = (
            "  *(not yet implemented)*"
            if action == "CHANGE_STRATEGY" and count == 0
            else ""
        )
        lines.append(f"| {action} | {count} | {pct:.1f}%{note} |")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _print_metrics(metrics_df: pd.DataFrame) -> None:
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
    display_columns = [c for c in display_columns if c in metrics_df.columns]
    print(
        metrics_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def _print_action_distribution(
    action_pcts: dict[str, float], action_counts: dict[str, int]
) -> None:
    print_section("IQN Action Distribution (final eval, across all seeds)")
    print(f"{'action':<20}{'count':>8}{'percent':>10}")
    for action in _KNOWN_ACTIONS:
        count = action_counts.get(action, 0)
        pct = action_pcts.get(action, 0.0)
        note = (
            "  (not yet implemented)"
            if action == "CHANGE_STRATEGY" and count == 0
            else ""
        )
        print(f"{action:<20}{count:>8}{pct:>9.1f}%{note}")


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def build_summary_dashboard(
    algorithmic_summary_csv: str | Path,
    finrl_aggregate_csv: str | Path,
    iqn_metrics_csv: str | Path,
    iqn_eval_records_csv: str | Path,
    *,
    run_name: str = "d_iqn_dss_summary_dashboard",
    output_dir: str | Path | None = None,
    max_display_strategies: int = _DEFAULT_MAX_STRATEGIES,
) -> Path:
    """Build 4-panel summary dashboard comparing all strategy tiers.

    Parameters
    ----------
    algorithmic_summary_csv:
        Path to algorithmic_baselines_summary.csv from the algorithmic
        baseline grid run.
    finrl_aggregate_csv:
        Path to finrl_baseline_multiseed_aggregate_by_agent.csv from the
        FinRL multiseed summary run.
    iqn_metrics_csv:
        Path to iqn_learning_curve_multiseed_final_records.csv from the
        IQN multiseed summary run (one row per seed at final eval step).
    iqn_eval_records_csv:
        Path to iqn_learning_curve_multiseed_eval_records.csv, used to
        compute the IQN action distribution for Panel 4.
    run_name:
        Name suffix for the output run directory.
    output_dir:
        If provided, write all outputs here instead of creating a new
        timestamped run directory under outputs/runs/.
    max_display_strategies:
        Maximum number of strategies shown in panels 1-3 (top-N by
        end portfolio value).

    Returns
    -------
    Path to the saved summary_dashboard.png.
    """
    algorithmic_summary_csv = Path(algorithmic_summary_csv)
    finrl_aggregate_csv = Path(finrl_aggregate_csv)
    iqn_metrics_csv = Path(iqn_metrics_csv)
    iqn_eval_records_csv = Path(iqn_eval_records_csv)

    if output_dir is not None:
        summary_dir = Path(output_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        config_dir = summary_dir
        data_dir = summary_dir
        logs_dir = summary_dir
        run_dir_name = summary_dir.name
        run_paths = None
    else:
        run_paths = create_run_paths(run_name)
        summary_dir = run_paths.summary_directory
        config_dir = run_paths.config_directory
        data_dir = run_paths.data_directory
        logs_dir = run_paths.logs_directory
        run_dir_name = run_paths.run_directory.name

    _fh = logging.FileHandler(logs_dir / "run.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(_fh)
    log.setLevel(logging.INFO)

    try:
        log.info("Loading strategy metrics from all tiers")
        metrics_df = _load_all_metrics_v2(
            algorithmic_summary_csv, finrl_aggregate_csv, iqn_metrics_csv
        )
        log.info(
            "Loaded %d strategies (%d algorithmic, %d FinRL, %d IQN)",
            len(metrics_df),
            (metrics_df["source"] == "Algorithmic Trading / non-RL").sum(),
            (metrics_df["source"] == "FinRL / baseline").sum(),
            (metrics_df["source"] == "D-IQN-DSS").sum(),
        )

        log.info("Parsing IQN action distribution from eval records")
        action_counts, action_pcts = _extract_action_data(iqn_eval_records_csv)

        log.info("Selecting top %d strategies for display", max_display_strategies)
        plot_df = select_plot_metrics(metrics_df, max_display_strategies)

        dashboard_png = summary_dir / "summary_dashboard.png"
        log.info("Rendering dashboard figure -> %s", dashboard_png)
        _plot_dashboard_figure(
            plot_df, action_pcts, action_counts, dashboard_png, run_dir_name
        )

        summary_csv = summary_dir / "summary_report.csv"
        summary_md = summary_dir / "summary_report.md"
        metrics_df.to_csv(summary_csv, index=False)
        _write_markdown_summary(metrics_df, action_pcts, action_counts, summary_md)
        log.info("Saved summary_report.csv and summary_report.md")

        combined_csv = data_dir / "strategies_combined.csv"
        action_dist_csv = data_dir / "iqn_action_distribution.csv"
        metrics_df.to_csv(combined_csv, index=False)
        pd.DataFrame(
            [
                {
                    "action": a,
                    "count": action_counts.get(a, 0),
                    "percent": action_pcts.get(a, 0.0),
                }
                for a in _KNOWN_ACTIONS
            ]
        ).to_csv(action_dist_csv, index=False)
        log.info("Saved strategies_combined.csv and iqn_action_distribution.csv")

        config_data = {
            "inputs": {
                "algorithmic_summary_csv": str(algorithmic_summary_csv),
                "finrl_aggregate_csv": str(finrl_aggregate_csv),
                "iqn_metrics_csv": str(iqn_metrics_csv),
                "iqn_eval_records_csv": str(iqn_eval_records_csv),
            },
            "parameters": {
                "run_name": run_name,
                "max_display_strategies": max_display_strategies,
            },
            "output_run_dir": run_dir_name,
        }
        config_path = config_dir / "summary_dashboard_config.json"
        config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
        log.info("Saved config JSON")

        _print_metrics(metrics_df)
        _print_action_distribution(action_pcts, action_counts)

        print_section("Summary finished")
        print(f"Saved to: {summary_dir.resolve()}")
        print("Key files:")
        print(f"  {summary_csv.name}")
        print(f"  {summary_md.name}")
        print(f"  {dashboard_png.name}")
        print(f"  {action_dist_csv.name}")

    finally:
        log.removeHandler(_fh)
        _fh.close()

    return dashboard_png
