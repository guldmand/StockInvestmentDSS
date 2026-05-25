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

Important:
This script does NOT rely on the current central summary_report.csv as the
primary data source, because that file is overwritten by this script. It only
uses it as a last-resort fallback when it contains usable non-NaN metric rows.
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

FINRL_STRATEGIES = {"a2c", "ddpg", "ppo", "sac", "td3", "mvo"}


def print_section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def ensure_summary_dir(run_root: Path) -> Path:
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return None


def has_usable_metric_rows(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    if "end_value" not in df.columns:
        return False
    return pd.to_numeric(df["end_value"], errors="coerce").notna().any()


def normalise_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rename_map = {
        "account_start_value": "start_value",
        "account_end_value": "end_value",
        "return_pct": "total_return_pct",
        "sharpe": "annualized_sharpe",
        "max_drawdown": "max_drawdown_pct",
        "max_drawdown_percent": "max_drawdown_pct",
        "max_drawdown_ratio": "max_drawdown_pct",
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

    drawdown_non_na = out["max_drawdown_pct"].dropna()
    if (
        not drawdown_non_na.empty
        and drawdown_non_na.abs().le(1.0).all()
        and drawdown_non_na.abs().max() > 0
    ):
        out["max_drawdown_pct"] = out["max_drawdown_pct"] * 100.0

    missing_profit_loss = out["profit_loss"].isna()
    out.loc[missing_profit_loss, "profit_loss"] = (
        out.loc[missing_profit_loss, "end_value"]
        - out.loc[missing_profit_loss, "start_value"]
    )

    missing_return = out["total_return_pct"].isna()
    out.loc[missing_return, "total_return_pct"] = (
        out.loc[missing_return, "profit_loss"] / out.loc[missing_return, "start_value"]
    ) * 100.0

    out = out[REQUIRED_METRIC_COLUMNS].copy()

    # A metrics row without end_value is not comparable and must not be ranked.
    out = out[pd.to_numeric(out["end_value"], errors="coerce").notna()].copy()

    return out


def infer_strategy_from_filename(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in ["_metrics", "_account_values", "_account_value", "_summary"]:
        stem = stem.replace(suffix, "")
    return stem


def infer_source_from_path_or_strategy(path: Path, df: pd.DataFrame) -> str:
    lower_path = str(path).lower()

    if "algorithmic_trading" in lower_path:
        return "Algorithmic Trading / non-RL"

    if "d_iqn" in lower_path or "d-iqn" in lower_path or "iqn" in lower_path:
        return "D-IQN-DSS"

    if "finrl" in lower_path:
        return "FinRL / baseline"

    if "source" in df.columns and not df["source"].dropna().empty:
        return str(df["source"].dropna().iloc[0])

    if "strategy" in df.columns and not df["strategy"].dropna().empty:
        strategies = set(df["strategy"].astype(str).str.lower())
        if strategies.intersection(FINRL_STRATEGIES):
            return "FinRL / baseline"
        if any("iqn" in strategy for strategy in strategies):
            return "D-IQN-DSS"

    filename_strategy = infer_strategy_from_filename(path)
    if filename_strategy in FINRL_STRATEGIES:
        return "FinRL / baseline"

    return "FinRL / baseline"


def find_metric_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*metrics*.csv"))


def is_algorithmic_path(path: Path) -> bool:
    return "algorithmic_trading" in {part.lower() for part in path.parts}


def is_central_summary_path(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    return "summary" in parts and path.name.lower() == "summary_report.csv"


def load_summary_group_fallback(run_root: Path, source: str) -> pd.DataFrame:
    """Last-resort fallback from central summary_report.csv.

    It is only used when rows have real end_value values. This prevents a bad
    overwritten summary from poisoning future summaries with NaN metric rows.
    """

    summary_file = run_root / "summary" / "summary_report.csv"
    df = read_csv_if_exists(summary_file)

    if df is None or df.empty or "source" not in df.columns:
        return pd.DataFrame(columns=REQUIRED_METRIC_COLUMNS)

    subset = df[df["source"].astype(str).str.strip() == source].copy()
    if subset.empty:
        return pd.DataFrame(columns=REQUIRED_METRIC_COLUMNS)

    return normalise_metric_columns(subset)


def load_finrl_metrics(run_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    # Broad but safe search: include any CSV that contains known FinRL strategy
    # names and has usable portfolio metrics.
    for path in sorted(run_root.rglob("*.csv")):
        if is_algorithmic_path(path):
            continue
        if is_central_summary_path(path):
            continue
        if "summary_iqn_last_decision" in path.name.lower():
            continue
        if any(token in str(path).lower() for token in ["d_iqn", "d-iqn", "iqn"]):
            continue

        df = read_csv_if_exists(path)
        if df is None or df.empty:
            continue

        lower_path = str(path).lower()
        lower_columns = {str(col).lower() for col in df.columns}

        contains_finrl_name = False
        if "strategy" in df.columns:
            strategies = set(df["strategy"].astype(str).str.lower())
            contains_finrl_name = bool(strategies.intersection(FINRL_STRATEGIES))

        filename_strategy = infer_strategy_from_filename(path)
        filename_is_finrl = filename_strategy in FINRL_STRATEGIES
        path_looks_finrl = "finrl" in lower_path or any(
            name in lower_path for name in FINRL_STRATEGIES
        )

        metric_columns_present = bool(
            {
                "end_value",
                "account_end_value",
                "total_return_pct",
                "return_pct",
            }.intersection(lower_columns)
        )

        if not (path_looks_finrl or contains_finrl_name or filename_is_finrl):
            continue
        if not metric_columns_present:
            continue

        if "strategy" not in df.columns:
            df["strategy"] = filename_strategy

        if "source" not in df.columns:
            df["source"] = "FinRL / baseline"

        # Keep only FinRL rows if mixed. If the file itself clearly looks like
        # FinRL, recover by assigning the source explicitly.
        if "source" in df.columns:
            source_text = df["source"].astype(str).str.strip()
            df = df[source_text.isin(["FinRL / baseline", "finrl", "FinRL"])].copy()
            if df.empty and (
                contains_finrl_name or filename_is_finrl or path_looks_finrl
            ):
                df = read_csv_if_exists(path)
                if df is None or df.empty:
                    continue
                if "strategy" not in df.columns:
                    df["strategy"] = filename_strategy
                df["source"] = "FinRL / baseline"

        normalised = normalise_metric_columns(df)
        if not normalised.empty:
            normalised["source"] = "FinRL / baseline"
            frames.append(normalised)

    # Last-resort: existing central summary, if still valid.
    fallback = load_summary_group_fallback(run_root, "FinRL / baseline")
    if not fallback.empty:
        frames.append(fallback)

    if not frames:
        return pd.DataFrame(columns=REQUIRED_METRIC_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["strategy", "source"], keep="last")
    return combined


def load_iqn_metrics(run_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    # Known likely locations first.
    candidate_files = [
        run_root / "d_iqn_dss" / "results" / "d_iqn_dss_metrics.csv",
        run_root / "d_iqn_dss" / "d_iqn_dss_metrics.csv",
        run_root / "iqn" / "results" / "d_iqn_dss_metrics.csv",
        run_root / "results" / "d_iqn_dss_metrics.csv",
    ]

    for path in candidate_files:
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            continue

        if "strategy" not in df.columns:
            df["strategy"] = "d_iqn_dss"
        if "source" not in df.columns:
            df["source"] = "D-IQN-DSS"

        normalised = normalise_metric_columns(df)
        if not normalised.empty:
            normalised["source"] = "D-IQN-DSS"
            frames.append(normalised)

    # Broad safe search. Do NOT treat IQN decision CSV as metrics.
    for path in sorted(run_root.rglob("*.csv")):
        lower_path = str(path).lower()
        lower_name = path.name.lower()

        if "summary_iqn_last_decision" in lower_name:
            continue
        if "decision" in lower_name:
            continue
        if is_central_summary_path(path):
            continue

        if not any(token in lower_path for token in ["d_iqn", "d-iqn", "iqn"]):
            continue
        if "metrics" not in lower_name and "summary" not in lower_name:
            continue

        df = read_csv_if_exists(path)
        if df is None or df.empty:
            continue

        lower_columns = {str(col).lower() for col in df.columns}
        metric_columns_present = bool(
            {
                "end_value",
                "account_end_value",
                "total_return_pct",
                "return_pct",
            }.intersection(lower_columns)
        )
        if not metric_columns_present:
            continue

        if "strategy" not in df.columns:
            df["strategy"] = "d_iqn_dss"
        if "source" not in df.columns:
            df["source"] = "D-IQN-DSS"

        normalised = normalise_metric_columns(df)
        if not normalised.empty:
            normalised["source"] = "D-IQN-DSS"
            frames.append(normalised)

    fallback = load_summary_group_fallback(run_root, "D-IQN-DSS")
    if not fallback.empty:
        frames.append(fallback)

    if not frames:
        return pd.DataFrame(columns=REQUIRED_METRIC_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["strategy", "source"], keep="last")
    return combined


def load_algorithmic_trading_metrics(run_root: Path) -> pd.DataFrame:
    summary_file = (
        run_root / "algorithmic_trading" / "summary" / "algorithmic_trading_summary.csv"
    )

    summary_df = read_csv_if_exists(summary_file)
    if summary_df is not None and not summary_df.empty:
        if "source" not in summary_df.columns:
            summary_df["source"] = "Algorithmic Trading / non-RL"
        normalised = normalise_metric_columns(summary_df)
        normalised["source"] = "Algorithmic Trading / non-RL"
        return normalised

    frames: list[pd.DataFrame] = []
    results_root = run_root / "algorithmic_trading" / "results"

    for path in find_metric_files(results_root):
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            continue

        if "strategy" not in df.columns:
            df["strategy"] = infer_strategy_from_filename(path)
        if "source" not in df.columns:
            df["source"] = "Algorithmic Trading / non-RL"

        normalised = normalise_metric_columns(df)
        if not normalised.empty:
            normalised["source"] = "Algorithmic Trading / non-RL"
            frames.append(normalised)

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
    combined["strategy"] = combined["strategy"].astype(str).str.strip()
    combined["source"] = combined["source"].astype(str).str.strip()

    for column in [
        "start_value",
        "end_value",
        "profit_loss",
        "total_return_pct",
        "max_drawdown_pct",
        "annualized_sharpe",
    ]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")

    combined = combined[
        pd.to_numeric(combined["end_value"], errors="coerce").notna()
    ].copy()

    # Prevent stale/fallback summaries from duplicating D-IQN-DSS as FinRL.
    iqn_strategy = combined["strategy"].str.lower().str.contains("iqn", na=False)
    bad_iqn_source = combined["source"].ne("D-IQN-DSS")
    combined = combined[~(iqn_strategy & bad_iqn_source)].copy()

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
    if df is None or df.empty or "action" not in df.columns:
        return None

    for column in ["q10", "q25", "q50", "q75", "q90", "cvar10", "risk_adjusted_score"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "risk_adjusted_score" not in df.columns and {"q50", "cvar10"}.issubset(
        df.columns
    ):
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
    return [
        GROUP_COLORS.get(str(source).strip(), "tab:gray") for source in df["source"]
    ]


def get_strategy_label_color(strategy: str, source: str | None = None) -> str:
    """Return the same group color for a strategy label as for its bar."""
    source_text = str(source).strip() if source is not None else ""
    strategy_text = str(strategy).strip()
    strategy_lower = strategy_text.lower()

    if source_text in GROUP_COLORS:
        return GROUP_COLORS[source_text]
    if strategy_lower == "d_iqn_dss" or "iqn" in strategy_lower:
        return GROUP_COLORS["D-IQN-DSS"]
    if strategy_lower in FINRL_STRATEGIES:
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
        # label.set_fontweight("medium")

        label.set_fontweight("bold")


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


def select_plot_metrics(
    metrics_df: pd.DataFrame, max_display_strategies: int = 18
) -> pd.DataFrame:
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


def prepare_metric_plot_df(
    metrics_df: pd.DataFrame,
    metric_column: str,
    ascending: bool,
    fill_value: float = 0.0,
) -> pd.DataFrame:
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


def finalize_barh_axis(ax, plot_df: pd.DataFrame, value_column: str) -> None:
    ax.tick_params(axis="y", labelsize=8, pad=3)
    ax.grid(True, axis="x", alpha=0.25)
    ax.margins(y=0.12)
    color_y_ticklabels(ax, plot_df)

    if not plot_df.empty:
        # Give top/bottom bars and zero/NA annotations enough vertical room.
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


def plot_summary_dashboard(
    metrics_df: pd.DataFrame,
    decision_df: Optional[pd.DataFrame],
    output_path: Path,
    run_root: Path,
) -> None:
    if metrics_df.empty:
        fig = plt.figure(figsize=(14, 6))
        fig.text(
            0.5, 0.5, "No runner metrics found", ha="center", va="center", fontsize=16
        )
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    plot_metrics_df = select_plot_metrics(metrics_df, max_display_strategies=999)

    fig = plt.figure(figsize=(22, 16))
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
            "risk_adjusted_score", ascending=True
        )

        ax_iqn.barh(
            decision_plot_df["action"],
            decision_plot_df["risk_adjusted_score"],
            color=GROUP_COLORS["D-IQN-DSS"],
        )
        ax_iqn.axvline(0.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
        ax_iqn.set_title(
            "Last IQN decision: risk-adjusted action score",
            fontsize=12,
            fontweight="bold",
        )
        ax_iqn.set_xlabel("Score")
        ax_iqn.tick_params(axis="y", labelsize=9)
        for label in ax_iqn.get_yticklabels():
            label.set_color(GROUP_COLORS["D-IQN-DSS"])
            label.set_fontweight("medium")
        ax_iqn.grid(True, axis="x", alpha=0.25)
    else:
        ax_iqn.axis("off")
        ax_iqn.set_title(
            "No IQN decision estimates found", fontsize=12, fontweight="bold"
        )

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
