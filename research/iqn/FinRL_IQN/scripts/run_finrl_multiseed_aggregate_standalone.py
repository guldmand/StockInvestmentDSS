"""Standalone FinRL Multiseed Aggregator on demo_10_new.

Aggregates the 5 FinRL baseline suite seed runs from outputs/runs/ by reading
each run's comparison_snapshot.csv directly. Produces:
  - aggregate_by_agent.csv (mean +/- std across 5 seeds, 6 agents)
  - 4 bar plots (total return, max drawdown, Sharpe, CVaR)
  - aggregate_summary.json

This is a standalone alternative to run_finrl_baseline_multiseed_summary
which had a child_run_index seed lookup bug that dropped all rows.

The 5 seed runs are identified by:
  - Run name pattern: contains 'finrl_baseline_suite_smoke_test'
  - Run directory created within the multiseed launcher window
  - Specifically the 5 runs spawned by launcher 2026_05_24_215642_*

Seeds 1-5 are assigned based on chronological order of run creation
(matching launcher_summary.json's launched_runs list order).

Usage:
  python scripts/run_finrl_multiseed_aggregate_standalone.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless backend for plot generation
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration: the 5 seed runs from the 2026-05-24 multiseed launcher
# ---------------------------------------------------------------------
LAUNCHER_RUN_ID = "2026_05_24_215642_d_iqn_dss_finrl_baseline_multiseed_launcher"

# These are the 5 seed runs spawned by the launcher (in chronological order
# matching launcher_summary.json's launched_runs list, which is seed-ordered)
SEED_RUN_IDS = [
    ("2026_05_24_215644_d_iqn_dss_finrl_baseline_suite_smoke_test", 1),
    ("2026_05_24_220834_d_iqn_dss_finrl_baseline_suite_smoke_test", 2),
    ("2026_05_24_222014_d_iqn_dss_finrl_baseline_suite_smoke_test", 3),
    ("2026_05_24_223206_d_iqn_dss_finrl_baseline_suite_smoke_test", 4),
    ("2026_05_24_224402_d_iqn_dss_finrl_baseline_suite_smoke_test", 5),
]

# Metrics to aggregate (column names from comparison_snapshot.csv)
METRICS = [
    "final_value",
    "total_return_pct",
    "annualized_sharpe",
    "max_drawdown_pct",
    "cvar_pct",
    "annualized_volatility_pct",
    "total_transaction_cost",
    "total_trades",
    "turnover_estimate_pct",
]


def find_project_root() -> Path:
    """Locate the repository root."""
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "stock_investment_dss").exists():
            return candidate
    raise RuntimeError(f"Could not find repo root from cwd={current}")


def read_seed_run(project_root: Path, run_id: str, seed: int) -> pd.DataFrame:
    """Read comparison snapshot for a single seed run."""
    run_dir = project_root / "outputs" / "runs" / run_id
    snapshot = run_dir / "summary" / "finrl_baseline_suite_comparison_snapshot.csv"

    if not snapshot.exists():
        raise FileNotFoundError(f"Comparison snapshot not found: {snapshot}")

    frame = pd.read_csv(snapshot)
    frame["seed"] = seed
    frame["source_run_id"] = run_id
    return frame


def aggregate_by_agent(combined: pd.DataFrame) -> pd.DataFrame:
    """Compute mean +/- std for each agent across seeds."""
    rows: list[dict[str, Any]] = []
    for agent_name, group in combined.groupby("agent_name"):
        seeds_list = sorted(group["seed"].unique().tolist())
        row: dict[str, Any] = {
            "agent_name": agent_name,
            "source": (
                "MVO baseline" if agent_name == "mvo"
                else "FinRL / SB3 baseline suite"
            ),
            "model_family": (
                "classical_portfolio_optimization" if agent_name == "mvo"
                else "parametric_rl_expected_return"
            ),
            "seed_count": len(seeds_list),
            "seeds": ",".join(str(s) for s in seeds_list),
        }

        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(values) > 1:
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1))
            elif len(values) == 1:
                row[f"{metric}_mean"] = float(values.iloc[0])
                row[f"{metric}_std"] = 0.0
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
        rows.append(row)

    return pd.DataFrame(rows)


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    xlabel: str,
) -> bool:
    """Render a horizontal bar plot with error bars."""
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in summary.columns:
        return False

    plot_df = summary[["agent_name", mean_col, std_col]].copy()
    plot_df[mean_col] = pd.to_numeric(plot_df[mean_col], errors="coerce")
    plot_df[std_col] = pd.to_numeric(plot_df[std_col], errors="coerce").fillna(0.0)
    plot_df = plot_df.dropna(subset=[mean_col])
    if plot_df.empty:
        return False

    plot_df = plot_df.sort_values(mean_col, ascending=True)

    # FinRL color coding: MVO gets a distinct color (classical method)
    colors = [
        "#d62728" if a == "mvo" else "#1f77b4"
        for a in plot_df["agent_name"]
    ]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(plot_df))))
    ax.barh(
        plot_df["agent_name"],
        plot_df[mean_col],
        xerr=plot_df[std_col],
        color=colors,
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(0.0, linestyle="--", linewidth=1, color="gray")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    project_root = find_project_root()

    # Create output run directory
    run_id = datetime.now().strftime("%Y_%m_%d_%H%M%S") + \
        "_d_iqn_dss_finrl_multiseed_aggregate_standalone"
    output_dir = project_root / "outputs" / "runs" / run_id
    summary_dir = output_dir / "summary"
    data_dir = output_dir / "data"
    plots_dir = output_dir / "plots"

    for d in [summary_dir, data_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[OK] Created output directory: {output_dir.relative_to(project_root)}")
    print()
    print("=" * 70)
    print("Standalone FinRL Multiseed Aggregator")
    print("=" * 70)
    print()
    print(f"Source launcher: {LAUNCHER_RUN_ID}")
    print()
    print(f"Reading {len(SEED_RUN_IDS)} seed runs...")

    # Read all seed runs
    seed_frames: list[pd.DataFrame] = []
    for run_id_value, seed in SEED_RUN_IDS:
        print(f"  Seed {seed}: {run_id_value[:30]}...", end="")
        try:
            frame = read_seed_run(project_root, run_id_value, seed)
            seed_frames.append(frame)
            print(f" {len(frame)} rows")
        except FileNotFoundError as exc:
            print(f" SKIPPED: {exc}")

    if not seed_frames:
        print("[ABORT] No seed runs successfully read.", file=sys.stderr)
        return 1

    combined = pd.concat(seed_frames, ignore_index=True)
    print()
    print(f"[OK] Combined {len(combined)} rows from {len(seed_frames)} seeds.")

    # Save raw combined data (member records)
    members_path = data_dir / "finrl_multiseed_member_records.csv"
    combined.to_csv(members_path, index=False)
    print(f"[OK] Wrote member records: {members_path.relative_to(project_root)}")

    # Aggregate by agent
    aggregate = aggregate_by_agent(combined)
    print()
    print("=" * 70)
    print("Aggregate Results (mean +/- std across 5 seeds)")
    print("=" * 70)
    print()

    # Pretty print aggregate
    display_cols = [
        "agent_name",
        "seed_count",
        "total_return_pct_mean",
        "total_return_pct_std",
        "max_drawdown_pct_mean",
        "max_drawdown_pct_std",
        "annualized_sharpe_mean",
        "annualized_sharpe_std",
    ]
    display = aggregate[display_cols].copy()
    display = display.sort_values("total_return_pct_mean", ascending=False)
    for col in display.columns:
        if "_mean" in col or "_std" in col:
            display[col] = display[col].apply(
                lambda x: f"{x:8.2f}" if pd.notna(x) else "    N/A"
            )
    print(display.to_string(index=False))
    print()

    # Save aggregate CSV
    aggregate_path = summary_dir / "finrl_multiseed_aggregate_by_agent.csv"
    aggregate.to_csv(aggregate_path, index=False)
    print(f"[OK] Wrote aggregate: {aggregate_path.relative_to(project_root)}")

    # Generate plots
    print()
    print("Generating plots...")
    plot_specs = [
        ("total_return_pct", "Total Return by Agent (mean +/- std, 5 seeds)", "Total return (%)"),
        ("max_drawdown_pct", "Maximum Drawdown by Agent (mean +/- std, 5 seeds)", "Max drawdown (%)"),
        ("annualized_sharpe", "Annualized Sharpe by Agent (mean +/- std, 5 seeds)", "Sharpe"),
        ("cvar_pct", "CVaR by Agent (mean +/- std, 5 seeds)", "CVaR (%)"),
    ]
    plots_generated = []
    for metric, title, xlabel in plot_specs:
        plot_path = plots_dir / f"finrl_multiseed_{metric}.png"
        if plot_metric(aggregate, metric, plot_path, title, xlabel):
            plots_generated.append(plot_path.name)
            print(f"  [OK] {plot_path.name}")
        else:
            print(f"  [SKIP] {metric} (no valid data)")

    # Write summary JSON
    summary = {
        "status": "ok",
        "project_name": "StockInvestmentDSS",
        "prototype_name": "D-IQN-DSS",
        "run_id": run_id,
        "run_directory": str(output_dir),
        "source_launcher": LAUNCHER_RUN_ID,
        "seeds": [s for _, s in SEED_RUN_IDS],
        "seed_run_ids": [r for r, _ in SEED_RUN_IDS],
        "agents": sorted(combined["agent_name"].unique().tolist()),
        "total_member_rows": int(len(combined)),
        "agent_count": int(len(aggregate)),
        "outputs": {
            "member_records_path": str(members_path),
            "aggregate_by_agent_path": str(aggregate_path),
            "plots": [str(plots_dir / p) for p in plots_generated],
        },
        "interpretation": (
            "Standalone FinRL multiseed aggregation from demo_10_new. "
            "Reads comparison_snapshot.csv from each seed run directly, "
            "bypassing the broken child_run_index lookup in "
            "run_finrl_baseline_multiseed_summary.py."
        ),
    }
    summary_json_path = summary_dir / "finrl_multiseed_aggregate_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"[OK] Wrote summary JSON: {summary_json_path.relative_to(project_root)}")

    print()
    print("=" * 70)
    print(f"Aggregation completed - Output: {output_dir.relative_to(project_root)}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
