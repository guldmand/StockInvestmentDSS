"""Strategy comparison report: multi-tier ranking and insight generation.

Reads strategies_combined.csv produced by the Etape 5 summary dashboard,
assigns three per-metric ranks (total return, Sharpe ratio, maximum drawdown)
plus a combined composite rank, generates a thesis-citable markdown report,
and writes four output CSVs.

Ranking convention: rank 1 = best on each metric.
  rank_return  : rank 1 = highest total_return_pct
  rank_sharpe  : rank 1 = highest annualized_sharpe
  rank_drawdown: rank 1 = least-negative max_drawdown_pct (smallest drawdown)
  combined_rank: (rank_return + rank_sharpe + rank_drawdown) / 3; lower = better.

Public entry point
------------------
build_comparison_report(
    strategies_combined_csv,
    *,
    run_name="d_iqn_dss_comparison_report",
    output_dir=None,
    max_top_strategies=50,
) -> Path
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_investment_dss.experiment_tracking.wandb_tracking import (
    finish_wandb_run,
    init_wandb_run,
    wandb_log,
)
from stock_investment_dss.utilities.paths import create_run_paths

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REQUIRED_INPUT_COLUMNS: list[str] = [
    "strategy",
    "source",
    "total_return_pct",
    "max_drawdown_pct",
    "annualized_sharpe",
]

_SOURCE_ALGORITHMIC: str = "Algorithmic Trading / non-RL"
_SOURCE_FINRL: str = "FinRL / baseline"
_SOURCE_IQN: str = "D-IQN-DSS"

_RL_SOURCES: frozenset[str] = frozenset({_SOURCE_FINRL, _SOURCE_IQN})

_SOURCE_KEY_MAP: dict[str, str] = {
    _SOURCE_ALGORITHMIC: "algorithmic",
    _SOURCE_FINRL: "finrl",
    _SOURCE_IQN: "iqn",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print_section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def _fmt(value, spec: str = ".2f") -> str:
    """Format a numeric value, returning 'N/A' on missing/non-numeric."""
    if pd.isna(value):
        return "N/A"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "N/A"


def _print_comparison_summary(df: pd.DataFrame, max_top_strategies: int = 50) -> None:
    _print_section("Strategy comparison — top strategies by combined rank")
    if df.empty:
        print("No strategies found.")
        return
    display_cols = [
        c
        for c in [
            "combined_rank",
            "strategy",
            "source",
            "total_return_pct",
            "annualized_sharpe",
            "max_drawdown_pct",
        ]
        if c in df.columns
    ]
    print(
        df.head(max_top_strategies)[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ---------------------------------------------------------------------------
# Specified public functions
# ---------------------------------------------------------------------------


def find_metric_files(latest_dashboard_run: Path) -> list[Path]:
    """Discover strategies_combined.csv from an Etape 5 dashboard run directory.

    Parameters
    ----------
    latest_dashboard_run:
        Top-level run directory produced by build_summary_dashboard.
        Expected layout: {run}/data/strategies_combined.csv

    Returns
    -------
    list[Path] containing the resolved CSV path.

    Raises
    ------
    FileNotFoundError if strategies_combined.csv is absent.
    """
    csv_path = latest_dashboard_run / "data" / "strategies_combined.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"strategies_combined.csv not found at {csv_path}. "
            "Run build_summary_dashboard first to produce this file."
        )
    return [csv_path]


def load_metrics(files: list[Path]) -> pd.DataFrame:
    """Load and aggregate strategy metrics from one or more CSV files.

    Parameters
    ----------
    files:
        Paths to CSV files containing strategy metrics.

    Returns
    -------
    Concatenated, deduplicated DataFrame with numeric columns coerced.

    Raises
    ------
    ValueError if required columns are missing from any input file.
    ValueError if no files are provided.
    """
    if not files:
        raise ValueError("No metric files provided.")

    frames: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        missing = [c for c in _REQUIRED_INPUT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Required columns missing from {path.name}: {missing}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    for col in [
        "start_value",
        "end_value",
        "profit_loss",
        "total_return_pct",
        "max_drawdown_pct",
        "annualized_sharpe",
    ]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined.drop_duplicates(subset=["strategy", "source"], keep="last")
    combined = combined.reset_index(drop=True)
    return combined


def add_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank_return, rank_sharpe, rank_drawdown, and combined_rank columns.

    Ranking convention: rank 1 = best on each metric.
      rank_return  : rank 1 = highest total_return_pct
      rank_sharpe  : rank 1 = highest annualized_sharpe
      rank_drawdown: rank 1 = least-negative max_drawdown_pct (smallest drawdown)
      combined_rank: (rank_return + rank_sharpe + rank_drawdown) / 3; lower = better.

    Returns a copy of df sorted by combined_rank ascending (best first).
    """
    out = df.copy()
    out["rank_return"] = out["total_return_pct"].rank(
        ascending=False, na_option="bottom"
    )
    out["rank_sharpe"] = out["annualized_sharpe"].rank(
        ascending=False, na_option="bottom"
    )
    out["rank_drawdown"] = out["max_drawdown_pct"].rank(
        ascending=False, na_option="bottom"
    )
    out["combined_rank"] = (
        out["rank_return"] + out["rank_sharpe"] + out["rank_drawdown"]
    ) / 3.0
    out = out.sort_values("combined_rank", ascending=True, na_position="last")
    out = out.reset_index(drop=True)
    return out


def make_insights(df: pd.DataFrame) -> str:
    """Auto-generate a thesis-citable markdown findings section.

    The returned markdown string contains five named sections:
      1. Top 5 strategies by combined rank
      2. Best algorithmic baseline
      3. Best FinRL agent
      4. D-IQN-DSS ranking details
      5. Tier comparison (mean metrics per source group)

    Assumes df has been produced by add_rankings() and is sorted by
    combined_rank ascending (best first).
    """
    lines: list[str] = []
    n_total = len(df)

    # --- 1. Top 5 overall ---
    lines.append("## Top 5 strategies by combined rank")
    lines.append("")
    top5_cols = [
        c
        for c in [
            "combined_rank",
            "strategy",
            "source",
            "rank_return",
            "rank_sharpe",
            "rank_drawdown",
            "total_return_pct",
            "annualized_sharpe",
            "max_drawdown_pct",
        ]
        if c in df.columns
    ]
    lines.append(df.head(5)[top5_cols].to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    # --- 2. Best algorithmic baseline ---
    lines.append("## Best algorithmic baseline")
    lines.append("")
    algo_df = df[df["source"].astype(str).str.strip() == _SOURCE_ALGORITHMIC]
    if algo_df.empty:
        lines.append("_No algorithmic baseline strategies found._")
    else:
        row = algo_df.iloc[0]
        lines.append(
            f"**{row['strategy']}** — combined rank {_fmt(row['combined_rank'])} / {n_total} "
            f"| return {_fmt(row['total_return_pct'])}% "
            f"| Sharpe {_fmt(row['annualized_sharpe'])} "
            f"| max drawdown {_fmt(row['max_drawdown_pct'])}%"
        )
    lines.append("")

    # --- 3. Best FinRL agent ---
    lines.append("## Best FinRL agent")
    lines.append("")
    finrl_df = df[df["source"].astype(str).str.strip() == _SOURCE_FINRL]
    if finrl_df.empty:
        lines.append("_No FinRL baseline strategies found._")
    else:
        row = finrl_df.iloc[0]
        lines.append(
            f"**{row['strategy']}** — combined rank {_fmt(row['combined_rank'])} / {n_total} "
            f"| return {_fmt(row['total_return_pct'])}% "
            f"| Sharpe {_fmt(row['annualized_sharpe'])} "
            f"| max drawdown {_fmt(row['max_drawdown_pct'])}%"
        )
    lines.append("")

    # --- 4. D-IQN-DSS ranking ---
    lines.append("## D-IQN-DSS ranking")
    lines.append("")
    iqn_df = df[df["source"].astype(str).str.strip() == _SOURCE_IQN]
    if iqn_df.empty:
        lines.append("_D-IQN-DSS not found in metrics._")
    else:
        row = iqn_df.iloc[0]
        lines.append(
            f"**D-IQN-DSS** combined rank: **{_fmt(row['combined_rank'])} / {n_total}** "
            f"(return rank {_fmt(row['rank_return'], '.0f')}, "
            f"Sharpe rank {_fmt(row['rank_sharpe'], '.0f')}, "
            f"drawdown rank {_fmt(row['rank_drawdown'], '.0f')})"
        )
        lines.append("")
        lines.append(
            f"Metrics: return {_fmt(row['total_return_pct'])}% "
            f"| Sharpe {_fmt(row['annualized_sharpe'])} "
            f"| max drawdown {_fmt(row['max_drawdown_pct'])}%"
        )
    lines.append("")

    # --- 5. Tier comparison ---
    lines.append("## Tier comparison (mean metrics per source)")
    lines.append("")
    tier_cols = [
        c
        for c in ["total_return_pct", "max_drawdown_pct", "annualized_sharpe"]
        if c in df.columns
    ]
    tier_table = (
        df.groupby("source")[tier_cols]
        .mean()
        .reset_index()
        .rename(columns={"source": "tier"})
    )
    lines.append(tier_table.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    return "\n".join(lines)


def save_markdown_report(
    df: pd.DataFrame,
    insights: str,
    output_dir: Path,
    max_top_strategies: int = 50,
) -> None:
    """Write comparison report files to output_dir.

    Files written:
    - comparison.md: header + insights + top-N strategies table
    - comparison.csv: top N strategies by combined_rank
    - algorithmic_only.csv: algorithmic trading strategies sorted by combined_rank
    - rl_only.csv: FinRL and D-IQN-DSS strategies sorted by combined_rank

    Parameters
    ----------
    df:
        Full ranked DataFrame from add_rankings(), sorted by combined_rank.
    insights:
        Markdown string from make_insights().
    output_dir:
        Directory in which to write all files.
    max_top_strategies:
        Number of strategies shown in comparison.md and comparison.csv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_total = len(df)

    # --- comparison.md ---
    md_lines: list[str] = []
    md_lines.append("# Strategy Comparison Report: All Tiers")
    md_lines.append("")
    md_lines.append(
        f"Generated: {timestamp}  —  {n_total} strategies compared across all tiers."
    )
    md_lines.append("")
    md_lines.append(insights)

    top_n = df.head(max_top_strategies)
    table_cols = [
        c
        for c in [
            "combined_rank",
            "strategy",
            "source",
            "rank_return",
            "rank_sharpe",
            "rank_drawdown",
            "total_return_pct",
            "annualized_sharpe",
            "max_drawdown_pct",
            "end_value",
        ]
        if c in df.columns
    ]
    md_lines.append(f"## Top {max_top_strategies} strategies by combined rank")
    md_lines.append("")
    md_lines.append(top_n[table_cols].to_markdown(index=False, floatfmt=".2f"))
    md_lines.append("")

    (output_dir / "comparison.md").write_text("\n".join(md_lines), encoding="utf-8")

    # --- comparison.csv ---
    top_n.to_csv(output_dir / "comparison.csv", index=False)

    # --- algorithmic_only.csv ---
    algo_df = df[df["source"].astype(str).str.strip() == _SOURCE_ALGORITHMIC].copy()
    algo_df.to_csv(output_dir / "algorithmic_only.csv", index=False)

    # --- rl_only.csv ---
    rl_df = df[df["source"].astype(str).str.strip().isin(_RL_SOURCES)].copy()
    rl_df.to_csv(output_dir / "rl_only.csv", index=False)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def build_comparison_report(
    strategies_combined_csv: str | Path,
    *,
    run_name: str = "d_iqn_dss_comparison_report",
    output_dir: str | Path | None = None,
    max_top_strategies: int = 50,
) -> Path:
    """Build strategy comparison report from a strategies_combined.csv file.

    Orchestrates load_metrics -> add_rankings -> make_insights ->
    save_markdown_report, creates the full RunPaths output layout, writes
    W&B metrics if enabled, and prints a terminal summary.

    Parameters
    ----------
    strategies_combined_csv:
        Path to strategies_combined.csv from an Etape 5 dashboard run.
        Typically at {run}/data/strategies_combined.csv.
    run_name:
        Name suffix for the output run directory.
    output_dir:
        If provided, write all outputs here instead of creating a new
        timestamped run directory under outputs/runs/.
    max_top_strategies:
        Number of strategies included in comparison.csv and the markdown
        top-N table.

    Returns
    -------
    Path to the saved comparison.md.
    """
    strategies_combined_csv = Path(strategies_combined_csv)

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
        log.info("Loading metrics from %s", strategies_combined_csv)
        metrics_df = load_metrics([strategies_combined_csv])
        log.info("Loaded %d strategies", len(metrics_df))

        log.info("Computing rankings")
        ranked_df = add_rankings(metrics_df)

        log.info("Generating insights")
        insights = make_insights(ranked_df)

        log.info("Writing report files to %s", summary_dir)
        save_markdown_report(ranked_df, insights, summary_dir, max_top_strategies)

        full_csv = data_dir / "strategies_ranked_full.csv"
        ranked_df.to_csv(full_csv, index=False)
        log.info("Saved strategies_ranked_full.csv (%d rows)", len(ranked_df))

        config_data = {
            "inputs": {
                "strategies_combined_csv": str(strategies_combined_csv),
            },
            "parameters": {
                "run_name": run_name,
                "max_top_strategies": max_top_strategies,
            },
            "output_run_dir": run_dir_name,
        }
        config_path = config_dir / "comparison_report_config.json"
        config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
        log.info("Saved comparison_report_config.json")

        # --- W&B ---
        n_algo = int(
            (ranked_df["source"].astype(str).str.strip() == _SOURCE_ALGORITHMIC).sum()
        )
        n_finrl = int(
            (ranked_df["source"].astype(str).str.strip() == _SOURCE_FINRL).sum()
        )
        n_iqn = int((ranked_df["source"].astype(str).str.strip() == _SOURCE_IQN).sum())

        iqn_rows = ranked_df[ranked_df["source"].astype(str).str.strip() == _SOURCE_IQN]
        iqn_combined = (
            float(iqn_rows["combined_rank"].iloc[0])
            if not iqn_rows.empty
            else float("nan")
        )
        iqn_rank_return = (
            float(iqn_rows["rank_return"].iloc[0])
            if not iqn_rows.empty
            else float("nan")
        )
        iqn_rank_sharpe = (
            float(iqn_rows["rank_sharpe"].iloc[0])
            if not iqn_rows.empty
            else float("nan")
        )
        iqn_rank_drawdown = (
            float(iqn_rows["rank_drawdown"].iloc[0])
            if not iqn_rows.empty
            else float("nan")
        )

        tier_means = (
            ranked_df.groupby("source")["total_return_pct"]
            .mean()
            .rename(
                index=lambda s: _SOURCE_KEY_MAP.get(
                    str(s).strip(),
                    str(s)[:12].lower().replace(" ", "_"),
                )
            )
        )

        wandb_log_data: dict = {
            "total_strategies": len(ranked_df),
            "n_algorithmic": n_algo,
            "n_finrl": n_finrl,
            "n_iqn": n_iqn,
            "iqn_combined_rank": iqn_combined,
            "iqn_rank_return": iqn_rank_return,
            "iqn_rank_sharpe": iqn_rank_sharpe,
            "iqn_rank_drawdown": iqn_rank_drawdown,
            "top1_strategy": (
                str(ranked_df["strategy"].iloc[0]) if not ranked_df.empty else ""
            ),
            "top1_combined_rank": (
                float(ranked_df["combined_rank"].iloc[0])
                if not ranked_df.empty
                else float("nan")
            ),
        }
        for tier_key, mean_val in tier_means.items():
            wandb_log_data[f"tier_mean_return_{tier_key}"] = float(mean_val)

        init_wandb_run(
            run_name=run_dir_name,
            config=config_data,
            group="comparison_report",
            job_type="comparison",
            tags=["etape6", "comparison"],
            run_directory=(
                str(run_paths.run_directory)
                if run_paths is not None
                else str(summary_dir)
            ),
        )
        wandb_log(wandb_log_data)

        # --- Terminal output ---
        _print_comparison_summary(ranked_df, max_top_strategies)
        _print_section("Report finished")
        print(f"Saved to: {summary_dir.resolve()}")
        print("Key files:")
        print("  comparison.md")
        print("  comparison.csv")
        print("  algorithmic_only.csv")
        print("  rl_only.csv")
        print(f"  {full_csv.name}")

        log.info("Done")

    finally:
        finish_wandb_run()
        log.removeHandler(_fh)
        _fh.close()

    return summary_dir / "comparison.md"
