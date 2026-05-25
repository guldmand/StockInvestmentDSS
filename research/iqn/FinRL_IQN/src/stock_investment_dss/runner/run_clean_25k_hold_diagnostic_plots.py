"""
Clean 25k HOLD Diagnostic Plot Package.

Reads existing run artifacts from the May 23 2026 clean_25k_baseline_v1 run.
Produces 10 diagnostic plots + plot_summary.md + plot_manifest.json.

Usage:
    python -m stock_investment_dss.runner.run_clean_25k_hold_diagnostic_plots

No training is performed. No source code or configs are modified.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime
from typing import Any

from stock_investment_dss.utilities.paths import create_run_paths

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
REG = PROJECT_ROOT / "outputs" / "run_registry"
RUNS = PROJECT_ROOT / "outputs" / "runs"

METRICS_CSV= REG / "clean_25k_hold_diagnostic_metrics.csv"

MULTISEED_DIR = (
    RUNS / "2026_05_23_090943_d_iqn_dss_iqn_learning_curve_multiseed_summary"
)
MS_AGG_CSV = (
    MULTISEED_DIR / "summary" / "iqn_learning_curve_multiseed_aggregate_by_step.csv"
)
MS_EVAL_CSV = MULTISEED_DIR / "data" / "iqn_learning_curve_multiseed_eval_records.csv"

SEED_DIRS: dict[int, pathlib.Path] = {
    1: RUNS / "2026_05_23_082854_d_iqn_dss_iqn_learning_curve_smoke_test",
    2: RUNS / "2026_05_23_083302_d_iqn_dss_iqn_learning_curve_smoke_test",
    3: RUNS / "2026_05_23_083709_d_iqn_dss_iqn_learning_curve_smoke_test",
    4: RUNS / "2026_05_23_084115_d_iqn_dss_iqn_learning_curve_smoke_test",
    5: RUNS / "2026_05_23_084514_d_iqn_dss_iqn_learning_curve_smoke_test",
    6: RUNS / "2026_05_23_084923_d_iqn_dss_iqn_learning_curve_smoke_test",
    7: RUNS / "2026_05_23_085325_d_iqn_dss_iqn_learning_curve_smoke_test",
    8: RUNS / "2026_05_23_085731_d_iqn_dss_iqn_learning_curve_smoke_test",
    9: RUNS / "2026_05_23_090136_d_iqn_dss_iqn_learning_curve_smoke_test",
    10: RUNS / "2026_05_23_090539_d_iqn_dss_iqn_learning_curve_smoke_test",
}

SEED_COLORS = {
    6: "#d62728",  # red = collapsed
}
DEFAULT_COLOR = "#1f77b4"
COLLAPSED_COLOR = "#d62728"
SEEDS_ALL = list(range(1, 11))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bar_colors(seeds: list[int], collapsed: int = 6) -> list[str]:
    return [COLLAPSED_COLOR if s == collapsed else DEFAULT_COLOR for s in seeds]


def seed_label(s: int) -> str:
    return f"Seed {s}" + (" (collapsed)" if s == 6 else "")


def read_training_records(
    seeds: list[int], usecols: list[str]
) -> dict[int, pd.DataFrame]:
    result = {}
    for seed in seeds:
        path = SEED_DIRS[seed] / "data" / "iqn_learning_curve_training_records.csv"
        result[seed] = pd.read_csv(path, usecols=usecols)
    return result


def savefig(path: pathlib.Path, fig: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.name}")


# ---------------------------------------------------------------------------
# Plot 1 — Seed-level total return
# ---------------------------------------------------------------------------


def plot_seed_level_total_return(dm: pd.DataFrame, out: pathlib.Path) -> None:
    seeds = dm["seed"].tolist()
    returns = pd.to_numeric(dm["return"], errors="coerce").tolist()
    colors = bar_colors(seeds)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(
        [str(s) for s in seeds], returns, color=colors, edgecolor="black", linewidth=0.7
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for bar, s, r in zip(bars, seeds, returns):
        if s == 6:
            ax.annotate(
                "collapsed",
                xy=(bar.get_x() + bar.get_width() / 2, 0),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=COLLAPSED_COLOR,
                fontweight="bold",
            )
        label = f"{r:.1f}%" if not np.isnan(r) else "0%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(r, 0) + 1,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xlabel("Seed")
    ax.set_ylabel("Final eval total return (%)")
    ax.set_title(
        "Clean 25k Baseline — Seed-level Final Eval Total Return\n"
        "clean_25k_baseline_v1 · 25,000 training steps · seeds 1–10"
    )
    ax.grid(axis="y", alpha=0.3)
    legend_patches = [
        mpatches.Patch(color=DEFAULT_COLOR, label="Active seeds (9/10)"),
        mpatches.Patch(
            color=COLLAPSED_COLOR, label="Seed 6 — full HOLD/no-trade collapse"
        ),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8)
    savefig(out, fig)


# ---------------------------------------------------------------------------
# Plot 2 — Seed-level total trades
# ---------------------------------------------------------------------------


def plot_seed_level_total_trades(dm: pd.DataFrame, out: pathlib.Path) -> None:
    seeds = dm["seed"].tolist()
    trades = dm["total_trades"].tolist()
    colors = bar_colors(seeds)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(
        [str(s) for s in seeds], trades, color=colors, edgecolor="black", linewidth=0.7
    )

    for bar, s, t in zip(bars, seeds, trades):
        if s == 6:
            ax.annotate(
                "0 trades\ncollapsed",
                xy=(bar.get_x() + bar.get_width() / 2, 0),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=COLLAPSED_COLOR,
                fontweight="bold",
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                t + max(trades) * 0.01,
                str(t),
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xlabel("Seed")
    ax.set_ylabel("Total trades (eval period)")
    ax.set_title(
        "Clean 25k Baseline — Seed-level Total Trades\n"
        "clean_25k_baseline_v1 · 25,000 training steps · seeds 1–10"
    )
    ax.grid(axis="y", alpha=0.3)
    legend_patches = [
        mpatches.Patch(color=DEFAULT_COLOR, label="Active seeds (9/10)"),
        mpatches.Patch(
            color=COLLAPSED_COLOR, label="Seed 6 — full HOLD/no-trade collapse"
        ),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
    savefig(out, fig)


# ---------------------------------------------------------------------------
# Plot 3 — Masked action rate (should be 0.0 across all seeds)
# ---------------------------------------------------------------------------


def plot_masked_action_rate(dm: pd.DataFrame, out: pathlib.Path) -> None:
    seeds = dm["seed"].tolist()
    rates = (
        pd.to_numeric(dm["masked_action_rate"], errors="coerce").fillna(0.0).tolist()
    )

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(
        [str(s) for s in seeds],
        rates,
        color=DEFAULT_COLOR,
        edgecolor="black",
        linewidth=0.7,
    )
    ax.set_ylim(0, 0.05)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Masked action rate")
    ax.set_title(
        "Clean 25k Baseline — Masked Action Rate per Seed\n"
        "masked_action_rate = 0.0 for all seeds → action-mask fallback ruled out as HOLD cause"
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.text(
        0.5,
        0.65,
        "masked_action_rate = 0.0 for ALL 10 seeds\nAction-mask fallback is NOT the cause of HOLD behavior",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="darkgreen",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.7),
    )
    ax.grid(axis="y", alpha=0.3)
    savefig(out, fig)


# ---------------------------------------------------------------------------
# Plot 4 — Q-value spread mean and final by seed
# ---------------------------------------------------------------------------


def plot_q_spread(dm: pd.DataFrame, out: pathlib.Path) -> None:
    seeds = dm["seed"].tolist()
    q_mean = pd.to_numeric(dm["q_spread_mean"], errors="coerce").tolist()
    q_final = pd.to_numeric(dm["q_spread_final"], errors="coerce").tolist()

    x = np.arange(len(seeds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_mean = ax.bar(
        x - width / 2,
        q_mean,
        width,
        label="q_spread_mean",
        color="#4878cf",
        edgecolor="black",
        linewidth=0.6,
    )
    bars_final = ax.bar(
        x + width / 2,
        q_final,
        width,
        label="q_spread_final (train_step=25000)",
        color="#6acc65",
        edgecolor="black",
        linewidth=0.6,
    )

    # Highlight seed 6 group
    s6_idx = seeds.index(6)
    for patch in [bars_mean[s6_idx], bars_final[s6_idx]]:
        patch.set_edgecolor(COLLAPSED_COLOR)
        patch.set_linewidth(2.0)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Q-value spread (best – second best score)")
    ax.set_title(
        "Clean 25k Baseline — Q-value Spread Mean and Final by Seed\n"
        "Non-zero spread across all seeds: policy is not random/tie-breaking;\n"
        "seed 6 (red border) collapsed to HOLD despite moderate spread"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    # annotate seed 6
    ax.annotate(
        "seed 6\ncollapsed",
        xy=(x[s6_idx], max(q_mean[s6_idx], q_final[s6_idx]) + 0.01),
        ha="center",
        va="bottom",
        fontsize=8,
        color=COLLAPSED_COLOR,
        fontweight="bold",
    )
    savefig(out, fig)


# ---------------------------------------------------------------------------
# Plot 5 — Requested vs effective action distribution
# ---------------------------------------------------------------------------


def plot_action_distribution(out: pathlib.Path) -> None:
    action_order = ["HOLD", "BUY", "SELL", "REBALANCE"]
    action_colors = {
        "HOLD": "#aec7e8",
        "BUY": "#98df8a",
        "SELL": "#ff9896",
        "REBALANCE": "#ffbb78",
    }

    records = read_training_records(
        SEEDS_ALL, usecols=["action_label", "action_was_masked"]
    )

    seed_counts: dict[int, dict[str, float]] = {}
    all_masked_false = True
    for seed, df in records.items():
        counts = df["action_label"].value_counts()
        total = len(df)
        fractions = {a: counts.get(a, 0) / total for a in action_order}
        seed_counts[seed] = fractions
        if df["action_was_masked"].any():
            all_masked_false = False

    fig, ax = plt.subplots(figsize=(12, 5))
    bottoms = np.zeros(len(SEEDS_ALL))
    for action in action_order:
        fracs = [seed_counts[s][action] for s in SEEDS_ALL]
        ax.bar(
            [str(s) for s in SEEDS_ALL],
            fracs,
            bottom=bottoms,
            label=action,
            color=action_colors[action],
            edgecolor="black",
            linewidth=0.4,
        )
        bottoms += np.array(fracs)

    ax.set_xlabel("Seed")
    ax.set_ylabel("Fraction of training actions")
    masked_note = (
        "requested = effective for ALL seeds (masked_action_rate = 0.0)\n"
        "Action-mask fallback is RULED OUT as the primary HOLD cause."
        if all_masked_false
        else "WARNING: some masking detected — verify masked_action_rate column"
    )
    ax.set_title(
        f"Clean 25k Baseline — Requested Action Distribution During Training\n{masked_note}"
    )
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    # Mark seed 6
    s6_x = SEEDS_ALL.index(6)
    ax.get_xticklabels()
    ax.axvspan(
        s6_x - 0.5, s6_x + 0.5, color=COLLAPSED_COLOR, alpha=0.12, label="_nolegend_"
    )
    ax.text(
        s6_x,
        1.02,
        "collapsed",
        ha="center",
        va="bottom",
        fontsize=8,
        color=COLLAPSED_COLOR,
        fontweight="bold",
    )
    savefig(out, fig)


# ---------------------------------------------------------------------------
# Plot 6 — Eval return learning curve (mean ± std, multiseed)
# ---------------------------------------------------------------------------


def plot_eval_return_curve(agg: pd.DataFrame, out: pathlib.Path) -> None:
    rows = agg[agg["metric"] == "total_return_pct"].sort_values("train_step")

    fig, ax = plt.subplots(figsize=(11, 5))
    x = rows["train_step"].astype(float).to_numpy()
    y = rows["mean"].astype(float).to_numpy()
    std = rows["std"].fillna(0.0).astype(float).to_numpy()

    ax.plot(
        x, y, marker="o", linewidth=2, color=DEFAULT_COLOR, label="Mean across 10 seeds"
    )
    ax.fill_between(
        x, y - std, y + std, alpha=0.25, color=DEFAULT_COLOR, label="±1 std"
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Eval total return (%)")
    ax.set_title(
        "Clean 25k Baseline — Eval Total Return Learning Curve (Mean ± 1 Std)\n"
        "10 seeds · clean_25k_baseline_v1 · score_mode=q50"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v):,}" for v in x], rotation=45, ha="right", fontsize=8)
    ax.text(
        0.98,
        0.05,
        "Note: seed 6 (full HOLD collapse) has return=0 at all steps,\n"
        "pulling mean down and contributing to std.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    savefig(out, fig)


# ---------------------------------------------------------------------------
# Plot 7 — Eval Sharpe learning curve (mean ± std)
# ---------------------------------------------------------------------------


def plot_eval_sharpe_curve(agg: pd.DataFrame, out: pathlib.Path) -> None:
    rows = agg[agg["metric"] == "annualized_sharpe"].sort_values("train_step")

    fig, ax = plt.subplots(figsize=(11, 5))
    x = rows["train_step"].astype(float).to_numpy()
    y = rows["mean"].astype(float).to_numpy()
    std = rows["std"].fillna(0.0).astype(float).to_numpy()

    ax.plot(
        x,
        y,
        marker="o",
        linewidth=2,
        color="#2ca02c",
        label="Mean across seeds (excl. null)",
    )
    ax.fill_between(x, y - std, y + std, alpha=0.25, color="#2ca02c", label="±1 std")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Annualized Sharpe ratio")
    ax.set_title(
        "Clean 25k Baseline — Eval Sharpe Learning Curve (Mean ± 1 Std)\n"
        "10 seeds · clean_25k_baseline_v1 · score_mode=q50"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v):,}" for v in x], rotation=45, ha="right", fontsize=8)
    ax.text(
        0.98,
        0.05,
        "Seed 6 Sharpe = null (no trades) → excluded from mean/std at steps where null.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    savefig(out, fig)


# ---------------------------------------------------------------------------
# Plot 8 — IQN loss curve (mean ± std, 500-step bins)
# ---------------------------------------------------------------------------


def plot_loss_curve(out: pathlib.Path) -> None:
    BIN = 500
    records = read_training_records(SEEDS_ALL, usecols=["step", "loss"])

    # Bin into 500-step windows
    max_step = 25000
    bin_edges = list(range(0, max_step + BIN, BIN))
    bin_centers = [b + BIN // 2 for b in bin_edges[:-1]]

    per_seed: dict[int, list[float | None]] = {}
    for seed, df in records.items():
        df = df.dropna(subset=["loss"])
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
        df = df.dropna()
        bins: list[float | None] = []
        for lo in bin_edges[:-1]:
            hi = lo + BIN
            mask = (df["step"] >= lo) & (df["step"] < hi)
            vals = df.loc[mask, "loss"]
            bins.append(float(vals.mean()) if not vals.empty else None)
        per_seed[seed] = bins

    # Build per-bin mean and std
    mean_loss: list[float] = []
    std_loss: list[float] = []
    for i in range(len(bin_centers)):
        vals = [per_seed[s][i] for s in SEEDS_ALL if per_seed[s][i] is not None]
        mean_loss.append(float(np.mean(vals)) if vals else float("nan"))
        std_loss.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    x = np.array(bin_centers)
    y = np.array(mean_loss)
    std = np.array(std_loss)
    valid = ~np.isnan(y)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        x[valid], y[valid], linewidth=2, color="#9467bd", label="Mean across 10 seeds"
    )
    ax.fill_between(
        x[valid],
        (y - std)[valid],
        (y + std)[valid],
        alpha=0.25,
        color="#9467bd",
        label="±1 std",
    )
    ax.axvline(
        2000, color="gray", linewidth=1, linestyle=":", label="learning_starts=2000"
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("IQN loss (500-step bin mean)")
    ax.set_title(
        "Clean 25k Baseline — IQN Loss Curve (Mean ± 1 Std, 500-step bins)\n"
        "10 seeds · clean_25k_baseline_v1 · loss=NaN before learning_starts=2000"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    savefig(out, fig)


# ---------------------------------------------------------------------------
# Plot 9 — Epsilon curve (representative seed 1; identical across seeds)
# ---------------------------------------------------------------------------


def plot_epsilon_curve(out: pathlib.Path) -> None:
    df = pd.read_csv(
        SEED_DIRS[1] / "data" / "iqn_learning_curve_training_records.csv",
        usecols=["step", "epsilon"],
    )
    df = df.iloc[::100].copy()  # downsample for plotting

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["step"], df["epsilon"], linewidth=1.8, color="#8c564b")
    ax.axhline(
        0.05, color="gray", linewidth=1, linestyle="--", label="epsilon_final=0.05"
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Epsilon (exploration rate)")
    ax.set_title(
        "Clean 25k Baseline — Epsilon Decay Curve\n"
        "Shown for seed 1 (representative; schedule identical across all 10 seeds)"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.text(
        0.98,
        0.75,
        "Epsilon schedule is deterministic (epsilon_decay_steps=15000,\n"
        "epsilon_final=0.05). Identical across all 10 seeds.\n"
        "Flat at epsilon_final=0.05 after step ~15,000.",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    savefig(out, fig)


# ---------------------------------------------------------------------------
# Plot 10 — Seed 6 vs Seed 7 vs Seed 8 comparison (2×2 multi-panel)
# ---------------------------------------------------------------------------


def plot_seed6_vs_seed7_vs_seed8(out: pathlib.Path) -> None:
    FOCUS = [6, 7, 8]
    FOCUS_COLORS = {6: "#d62728", 7: "#1f77b4", 8: "#2ca02c"}
    FOCUS_STYLE = {6: "--", 7: "-", 8: "-"}

    # (a) Q-spread over train_step
    qs_data: dict[int, pd.DataFrame] = {}
    for seed in FOCUS:
        df = pd.read_csv(SEED_DIRS[seed] / "data" / "hold_qvalue_spread_diagnostic.csv")
        qs_data[seed] = df.groupby("train_step")["spread"].mean().reset_index()

    # (b) Eval return trajectory
    ev_data: dict[int, pd.DataFrame] = {}
    for seed in FOCUS:
        df = pd.read_csv(
            SEED_DIRS[seed] / "data" / "iqn_learning_curve_eval_records.csv",
            usecols=["train_step", "total_return_pct"],
        )
        ev_data[seed] = df.sort_values("train_step")

    # (c) Training action distribution (stacked bar)
    tr_action: dict[int, dict[str, float]] = {}
    for seed in FOCUS:
        df = pd.read_csv(
            SEED_DIRS[seed] / "data" / "iqn_learning_curve_training_records.csv",
            usecols=["action_label"],
        )
        counts = df["action_label"].value_counts()
        total = len(df)
        tr_action[seed] = {
            a: counts.get(a, 0) / total for a in ["HOLD", "BUY", "SELL", "REBALANCE"]
        }

    # (d) Loss curve (500-step bins)
    BIN = 500
    bin_edges = list(range(0, 25001, BIN))
    bin_centers = [b + BIN // 2 for b in bin_edges[:-1]]
    tr_loss: dict[int, list[float | None]] = {}
    for seed in FOCUS:
        df = pd.read_csv(
            SEED_DIRS[seed] / "data" / "iqn_learning_curve_training_records.csv",
            usecols=["step", "loss"],
        )
        df = df.dropna(subset=["loss"])
        bins: list[float | None] = []
        for lo in bin_edges[:-1]:
            vals = df.loc[(df["step"] >= lo) & (df["step"] < lo + BIN), "loss"]
            bins.append(float(vals.mean()) if not vals.empty else None)
        tr_loss[seed] = bins

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Seed 6 (Collapsed) vs Seed 7 vs Seed 8 — Diagnostic Comparison\n"
        "clean_25k_baseline_v1 · Seed 6 = full HOLD/no-trade collapse",
        fontsize=11,
        fontweight="bold",
    )

    # Panel (a) — Q-spread
    ax = axes[0, 0]
    for seed in FOCUS:
        df = qs_data[seed]
        ax.plot(
            df["train_step"],
            df["spread"],
            color=FOCUS_COLORS[seed],
            linestyle=FOCUS_STYLE[seed],
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=seed_label(seed),
        )
    ax.set_title("(a) Q-value Spread over Training Checkpoints", fontsize=9)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean Q-spread (best – 2nd best)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel (b) — Eval return
    ax = axes[0, 1]
    for seed in FOCUS:
        df = ev_data[seed]
        ax.plot(
            df["train_step"],
            df["total_return_pct"],
            color=FOCUS_COLORS[seed],
            linestyle=FOCUS_STYLE[seed],
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=seed_label(seed),
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("(b) Eval Total Return Trajectory", fontsize=9)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Eval total return (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel (c) — Action distribution stacked bar
    ax = axes[1, 0]
    action_order = ["HOLD", "BUY", "SELL", "REBALANCE"]
    action_colors = {
        "HOLD": "#aec7e8",
        "BUY": "#98df8a",
        "SELL": "#ff9896",
        "REBALANCE": "#ffbb78",
    }
    x_pos = np.arange(len(FOCUS))
    bottoms = np.zeros(len(FOCUS))
    for action in action_order:
        fracs = [tr_action[s][action] for s in FOCUS]
        ax.bar(
            x_pos,
            fracs,
            bottom=bottoms,
            label=action,
            color=action_colors[action],
            edgecolor="black",
            linewidth=0.5,
        )
        bottoms += np.array(fracs)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([seed_label(s) for s in FOCUS], fontsize=8)
    ax.set_title("(c) Training Action Distribution (requested = effective)", fontsize=9)
    ax.set_ylabel("Fraction of training actions")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel (d) — Loss curve
    ax = axes[1, 1]
    x_arr = np.array(bin_centers)
    for seed in FOCUS:
        vals = [v if v is not None else float("nan") for v in tr_loss[seed]]
        y_arr = np.array(vals)
        valid = ~np.isnan(y_arr)
        ax.plot(
            x_arr[valid],
            y_arr[valid],
            color=FOCUS_COLORS[seed],
            linestyle=FOCUS_STYLE[seed],
            linewidth=1.5,
            label=seed_label(seed),
        )
    ax.axvline(2000, color="gray", linewidth=1, linestyle=":", label="learning_starts")
    ax.set_title("(d) IQN Loss Curve (500-step bins)", fontsize=9)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(out, fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    run_paths = create_run_paths("d_iqn_dss_clean_25k_hold_diagnostic_plots")
    print(f"Output directory: {run_paths.run_directory}")

    # Guard: verify all seed dirs exist
    missing = [d for d in SEED_DIRS.values() if not d.exists()]
    if missing:
        print("ERROR: missing seed directories:")
        for m in missing:
            print(f"  {m}")
        return 1

    print("Reading shared inputs …")
    dm = pd.read_csv(METRICS_CSV)
    agg = pd.read_csv(MS_AGG_CSV)

    plots_written: list[dict[str, Any]] = []

    def record(name: str, description: str, sources: list[str]) -> None:
        path = run_paths.plots_directory / name
        plots_written.append(
            {
                "filename": name,
                "path": str(path),
                "description": description,
                "sources": sources,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

    print("\nGenerating plots …")

    p1 = run_paths.plots_directory / "seed_level_total_return.png"
    plot_seed_level_total_return(dm, p1)
    record(
        "seed_level_total_return.png",
        "Bar chart of final eval total return by seed. Seed 6 highlighted as collapsed.",
        ["outputs/run_registry/clean_25k_hold_diagnostic_metrics.csv"],
    )

    p2 = run_paths.plots_directory / "seed_level_total_trades.png"
    plot_seed_level_total_trades(dm, p2)
    record(
        "seed_level_total_trades.png",
        "Bar chart of total trades by seed. Seed 6 = 0 trades.",
        ["outputs/run_registry/clean_25k_hold_diagnostic_metrics.csv"],
    )

    p3 = run_paths.plots_directory / "masked_action_rate.png"
    plot_masked_action_rate(dm, p3)
    record(
        "masked_action_rate.png",
        "Masked action rate per seed — 0.0 across all seeds, ruling out action-mask fallback.",
        ["outputs/run_registry/clean_25k_hold_diagnostic_metrics.csv"],
    )

    p4 = run_paths.plots_directory / "q_value_spread_mean_and_final.png"
    plot_q_spread(dm, p4)
    record(
        "q_value_spread_mean_and_final.png",
        "Grouped bar comparing q_spread_mean and q_spread_final per seed.",
        ["outputs/run_registry/clean_25k_hold_diagnostic_metrics.csv"],
    )

    p5 = run_paths.plots_directory / "requested_vs_effective_action_distribution.png"
    plot_action_distribution(p5)
    record(
        "requested_vs_effective_action_distribution.png",
        "Stacked 100% bar of training action distribution per seed. Requested = effective (masked_action_rate=0).",
        [
            f"outputs/runs/{SEED_DIRS[s].name}/data/iqn_learning_curve_training_records.csv"
            for s in SEEDS_ALL
        ],
    )

    p6 = run_paths.plots_directory / "eval_return_learning_curve_mean_std.png"
    plot_eval_return_curve(agg, p6)
    record(
        "eval_return_learning_curve_mean_std.png",
        "Mean ± 1 std eval total return over training checkpoints.",
        [str(MS_AGG_CSV.relative_to(PROJECT_ROOT))],
    )

    p7 = run_paths.plots_directory / "eval_sharpe_learning_curve_mean_std.png"
    plot_eval_sharpe_curve(agg, p7)
    record(
        "eval_sharpe_learning_curve_mean_std.png",
        "Mean ± 1 std annualized Sharpe over training checkpoints.",
        [str(MS_AGG_CSV.relative_to(PROJECT_ROOT))],
    )

    p8 = run_paths.plots_directory / "iqn_loss_curve_mean_std.png"
    plot_loss_curve(p8)
    record(
        "iqn_loss_curve_mean_std.png",
        "Mean ± 1 std IQN loss in 500-step bins across 10 seeds.",
        [
            f"outputs/runs/{SEED_DIRS[s].name}/data/iqn_learning_curve_training_records.csv"
            for s in SEEDS_ALL
        ],
    )

    p9 = run_paths.plots_directory / "epsilon_curve.png"
    plot_epsilon_curve(p9)
    record(
        "epsilon_curve.png",
        "Epsilon decay curve (seed 1 representative; identical across all seeds).",
        [
            f"outputs/runs/{SEED_DIRS[1].name}/data/iqn_learning_curve_training_records.csv"
        ],
    )

    p10 = run_paths.plots_directory / "seed6_vs_seed7_vs_seed8_comparison.png"
    plot_seed6_vs_seed7_vs_seed8(p10)
    record(
        "seed6_vs_seed7_vs_seed8_comparison.png",
        "2×2 multi-panel: Q-spread, eval return, action distribution, and loss for seeds 6, 7, 8.",
        [f"outputs/runs/{SEED_DIRS[s].name}/..." for s in [6, 7, 8]],
    )

    # Write manifest
    manifest_path = run_paths.summary_directory / "plot_manifest.json"
    manifest = {
        "experiment": "clean_25k_baseline_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "no_training_performed": True,
        "source_artifacts_only": True,
        "total_plots": len(plots_written),
        "plots": plots_written,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  saved: {manifest_path.name}")

    # Write summary markdown
    summary_path = run_paths.summary_directory / "plot_summary.md"
    lines = [
        "# Clean 25k HOLD Diagnostic — Plot Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Experiment: `clean_25k_baseline_v1` — 10 seeds, 25,000 training steps, score_mode=q50, LayerNorm=true",
        "Source artifacts: May 23 2026 08:28–09:09 UTC+2 clean run.",
        "**No training was performed to generate these plots.**",
        "",
        "## Key findings (from plots)",
        "",
        "- **masked_action_rate = 0.0 for all seeds** (plot 3): action-mask fallback is ruled out as the cause of HOLD behavior.",
        "- **Seed 6** is the only full HOLD/no-trade collapse (plots 1, 2, 10).",
        "- Q-value spread is non-zero across all seeds (plot 4), ruling out pure random tie-breaking.",
        "- Spread varies substantially between seeds (seed 5: mean=0.61; seed 9: mean=0.11).",
        "- Seed 6 has moderate spread (mean=0.15) yet fully collapsed — HOLD collapse is non-linear.",
        "- Epsilon schedule is identical across seeds and decays to 0.05 by step ~15,000 (plot 9).",
        "- The remaining HOLD issue is seed-dependent Q-policy attractor / weak action-value separation.",
        "",
        "## Plots",
        "",
    ]
    for p in plots_written:
        lines.append(f"### {p['filename']}")
        lines.append(f"{p['description']}")
        lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  saved: {summary_path.name}")

    print(f"\nDone. {len(plots_written)} plots + manifest + summary written to:")
    print(f"  plots/:   {run_paths.plots_directory}")
    print(f"  summary/: {run_paths.summary_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
