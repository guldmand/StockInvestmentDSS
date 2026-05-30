#!/usr/bin/env python3
"""D-IQN-DSS thesis dashboard — 4 panels, portfolio-level comparison.

All numbers are read from source CSV files. There are NO hardcoded performance
values and NO estimated action distributions: Panel 4 is built from the
per-ablation ``actions.csv`` produced by the Layer 4 replay.

Panels:
  top-left     Total Return by Portfolio Strategy
  top-right    Maximum Drawdown by Strategy
  bottom-left  Annualized Sharpe Ratio by Strategy
  bottom-right Action Distribution per ablation (4 ablations x 5 actions)

The CONFIG block below holds only DATA-SOURCE PATHS (not values). The Layer 4
directory defaults to the newest ``*_d_iqn_dss_edl_portfolio_backtest_scenarioX``
run, or can be overridden with ``--layer4-dir``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Repo root = parent of this script's directory (scripts/ -> repo root).
ROOT = Path(__file__).resolve().parent.parent

# ============================================================================
# CONFIG — data-source paths only (no values)
# ============================================================================
LAYER1B_DIRS = [
    "outputs/runs/2026_05_29_121414_d_iqn_dss_single_ticker_portfolio_buy_and_hold_sp500",
    "outputs/runs/2026_05_29_123745_d_iqn_dss_single_ticker_portfolio_sma_crossover_sp500",
    "outputs/runs/2026_05_29_130109_d_iqn_dss_single_ticker_portfolio_ema_crossover_sp500",
    "outputs/runs/2026_05_29_132437_d_iqn_dss_single_ticker_portfolio_macd_signal_sp500",
    "outputs/runs/2026_05_29_134801_d_iqn_dss_single_ticker_portfolio_rsi_mean_reversion_sp500",
    "outputs/runs/2026_05_29_141217_d_iqn_dss_single_ticker_portfolio_bollinger_mean_reversion_sp500",
    "outputs/runs/2026_05_29_143656_d_iqn_dss_single_ticker_portfolio_momentum_sp500",
    "outputs/runs/2026_05_29_150121_d_iqn_dss_single_ticker_portfolio_breakout_sp500",
    "outputs/runs/2026_05_29_152600_d_iqn_dss_single_ticker_portfolio_volatility_filter_sp500",
]
LAYER1A_PORTFOLIO = (
    "outputs/runs/2026_05_29_114553_d_iqn_dss_algorithmic_baseline_grid_sp500/"
    "summary/algorithmic_baselines_summary.csv"
)
LAYER2_AGGREGATE = (
    "outputs/runs/2026_05_29_120651_d_iqn_dss_finrl_baseline_multiseed_summary/"
    "summary/finrl_baseline_multiseed_aggregate_by_agent.csv"
)
LAYER4_GLOB = "outputs/runs/*_d_iqn_dss_edl_portfolio_backtest_scenarioX"

OUTPUT_PNG = "outputs/portfolio_summary_dashboard.png"

# ----------------------------------------------------------------------------
# Colours and labels
# ----------------------------------------------------------------------------
COLORS = {
    "passive": "#16A085",   # buy_and_hold (467 tickers)
    "classical": "#1ABC9C",  # classical portfolio benchmark (Layer 1a, scope=portfolio)
    "rule": "#7F8C8D",      # rule-based strategies
    "finrl": "#2E86C1",     # FinRL parametric RL + MVO
    "a1": "#F39C12",        # D-IQN-DSS
    "a2": "#E67E22",        # + HDP
    "a3": "#D35400",        # + EDL
    "a4": "#A04000",        # + HDP + EDL
}
ABLATION_LABELS = {
    "a1": "D-IQN-DSS",
    "a2": "D-IQN-DSS + HDP",
    "a3": "D-IQN-DSS + EDL",
    "a4": "D-IQN-DSS + HDP + EDL",
}
ACTION_ORDER = ["BUY", "HOLD", "SELL", "REBALANCE", "CHANGE_STRATEGY"]


def _fail(msg: str) -> None:
    raise SystemExit(f"[portfolio_summary_dashboard] ERROR: {msg}")


def _require(path: Path) -> Path:
    if not path.exists():
        _fail(f"missing source file/dir: {path}")
    return path


# ============================================================================
# Loaders
# ============================================================================
def load_layer1b() -> pd.DataFrame:
    """Read each rule-based portfolio's portfolio_metrics.csv."""
    rows = []
    for rel in LAYER1B_DIRS:
        run_dir = _require(ROOT / rel)
        matches = list(
            run_dir.glob("metrics/single_ticker_portfolio/*/portfolio_metrics.csv")
        )
        if len(matches) != 1:
            _fail(
                f"expected exactly 1 portfolio_metrics.csv under {run_dir}, "
                f"found {len(matches)}"
            )
        metrics_path = matches[0]
        strategy = metrics_path.parent.name  # clean folder name, e.g. ema_crossover
        df = pd.read_csv(metrics_path)
        r = df.iloc[0]
        cat = "passive" if strategy == "buy_and_hold" else "rule"
        rows.append(
            {
                "label": f"{strategy} (467)",
                "return": float(r["total_return_pct"]),
                "sharpe": float(r["annualized_sharpe"]),
                "max_dd": float(r["max_drawdown_pct"]),
                "cat": cat,
            }
        )
    return pd.DataFrame(rows)


def load_layer1a_portfolio() -> pd.DataFrame:
    """True portfolio-level classical benchmarks (scope == 'portfolio').

    Filters out all single-ticker / multi-ticker rows. Skips
    equal_weight_buy_and_hold because it duplicates the Layer 1b buy_and_hold bar.
    """
    path = _require(ROOT / LAYER1A_PORTFOLIO)
    df = pd.read_csv(path)
    port = df[df["scope"] == "portfolio"].copy()
    if port.empty:
        _fail(f"no scope=='portfolio' rows in {path}")
    rows = []
    for _, r in port.iterrows():
        label = str(r["config_label"])
        if label == "equal_weight_buy_and_hold":
            continue  # == Layer 1b buy_and_hold; avoid duplicate bar
        rows.append(
            {
                "label": label,
                "return": float(r["total_return_pct"]),
                "sharpe": float(r["annualized_sharpe"]),
                "max_dd": float(r["max_drawdown_pct"]),
                "cat": "classical",
            }
        )
    return pd.DataFrame(rows)


def load_layer2() -> pd.DataFrame:
    """Read FinRL multiseed aggregate-by-agent."""
    path = _require(ROOT / LAYER2_AGGREGATE)
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        ret = r["total_return_pct_mean"]
        traded = pd.notna(ret)
        label = str(r["agent_name"]).upper()
        if not traded:
            label = f"{label} (never trades)"
        rows.append(
            {
                "label": label,
                "return": float(ret) if traded else 0.0,
                "sharpe": float(r["annualized_sharpe_mean"]) if traded else 0.0,
                "max_dd": float(r["max_drawdown_pct_mean"])
                if pd.notna(r["max_drawdown_pct_mean"])
                else 0.0,
                "cat": "finrl",
            }
        )
    return pd.DataFrame(rows)


def resolve_layer4_dir(cli_dir: str | None) -> Path:
    if cli_dir:
        return _require(Path(cli_dir) if Path(cli_dir).is_absolute() else ROOT / cli_dir)
    candidates = sorted(ROOT.glob(LAYER4_GLOB))
    if not candidates:
        _fail(
            f"no Layer 4 run dir matching {LAYER4_GLOB!r}; pass --layer4-dir explicitly"
        )
    return candidates[-1]  # newest by timestamped name


def load_layer4_metrics(layer4_dir: Path) -> pd.DataFrame:
    """Read per-ablation metrics.csv (return/sharpe/max_dd)."""
    rows = []
    for abl in ["a1", "a2", "a3", "a4"]:
        m = pd.read_csv(_require(layer4_dir / abl / "metrics.csv"))
        r = m.iloc[0]
        rows.append(
            {
                "label": ABLATION_LABELS[abl],
                "return": float(r["total_return_pct"]),
                "sharpe": float(r["annualized_sharpe"]),
                "max_dd": float(r["max_drawdown_pct"]),
                "cat": abl,
            }
        )
    return pd.DataFrame(rows)


def load_layer4_actions(layer4_dir: Path) -> dict[str, dict[str, float]]:
    """Read per-ablation actions.csv -> {ablation: {action: percent}}."""
    dist: dict[str, dict[str, float]] = {}
    for abl in ["a1", "a2", "a3", "a4"]:
        df = pd.read_csv(_require(layer4_dir / abl / "actions.csv"))
        if "final_action" not in df.columns:
            _fail(f"{layer4_dir / abl / 'actions.csv'} has no 'final_action' column")
        total = len(df)
        counts = df["final_action"].str.upper().value_counts()
        dist[abl] = {a: 100.0 * float(counts.get(a, 0)) / total for a in ACTION_ORDER}
    return dist


# ============================================================================
# Plot
# ============================================================================
def build_dashboard(layer4_dir: Path, out_png: Path) -> None:
    l1b = load_layer1b()
    l1a = load_layer1a_portfolio()
    l2 = load_layer2()
    l4 = load_layer4_metrics(layer4_dir)
    actions = load_layer4_actions(layer4_dir)

    perf = pd.concat([l1b, l1a, l2, l4], ignore_index=True)
    # each panel sorts independently by its OWN metric (best at top)
    perf_ret = perf.sort_values("return", ascending=False).reset_index(drop=True)
    perf_dd = perf.sort_values("max_dd", ascending=False).reset_index(drop=True)
    perf_shp = perf.sort_values("sharpe", ascending=False).reset_index(drop=True)
    colors_ret = [COLORS[c] for c in perf_ret["cat"]]
    colors_dd = [COLORS[c] for c in perf_dd["cat"]]
    colors_shp = [COLORS[c] for c in perf_shp["cat"]]

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(
        2, 2, hspace=0.30, wspace=0.22, left=0.18, right=0.97, top=0.92, bottom=0.10
    )
    fig.suptitle(
        "D-IQN-DSS Thesis Summary — Portfolio-Level Comparison\n"
        "SP500 467-Ticker Universe · 554 trading days "
        "(2024-01-01 to 2026-05-26) · $1M initial capital",
        fontsize=14,
        fontweight="bold",
        y=0.975,
    )

    # ---- Panel 1: Total Return -------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(perf_ret["label"], perf_ret["return"], color=colors_ret,
             edgecolor="black", linewidth=0.5)
    ax1.axvline(0, color="black", linewidth=0.7)
    ax1.set_xlabel("Total Return (%)")
    ax1.set_title("Total Return by Portfolio Strategy (sorted by return)",
                  fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)
    ax1.invert_yaxis()  # largest return at top
    for i, v in enumerate(perf_ret["return"]):
        ax1.text(v + (0.6 if v >= 0 else -0.6), i, f"{v:+.2f}%",
                 va="center", ha="left" if v >= 0 else "right",
                 fontsize=8, fontweight="bold")

    # ---- Panel 2: Maximum Drawdown ---------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(perf_dd["label"], perf_dd["max_dd"], color=colors_dd,
             edgecolor="black", linewidth=0.5)
    ax2.axvline(0, color="black", linewidth=0.7)
    ax2.set_xlabel("Maximum Drawdown (%)")
    ax2.set_title("Maximum Drawdown by Strategy (sorted by drawdown)\n"
                  "(lower magnitude = better)", fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    ax2.invert_yaxis()  # smallest-magnitude drawdown at top
    ax2.yaxis.tick_right()  # own labels on the right (panel has its own order)
    for i, v in enumerate(perf_dd["max_dd"]):
        ax2.text(v - 0.2, i, f"{v:.2f}%", va="center", ha="right",
                 fontsize=8, fontweight="bold")

    # ---- Panel 3: Annualized Sharpe --------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.barh(perf_shp["label"], perf_shp["sharpe"], color=colors_shp,
             edgecolor="black", linewidth=0.5)
    ax3.axvline(0, color="black", linewidth=0.7)
    ax3.set_xlabel("Annualized Sharpe Ratio")
    ax3.set_title("Annualized Sharpe Ratio by Strategy (sorted by Sharpe)",
                  fontweight="bold")
    ax3.grid(axis="x", alpha=0.3)
    ax3.invert_yaxis()  # largest Sharpe at top
    for i, v in enumerate(perf_shp["sharpe"]):
        ax3.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}",
                 va="center", ha="left" if v >= 0 else "right",
                 fontsize=8, fontweight="bold")

    # colour y-tick labels by category + bold the DSS rows (each panel its own order)
    for ax, frame in ((ax1, perf_ret), (ax2, perf_dd), (ax3, perf_shp)):
        for tick, cat in zip(ax.get_yticklabels(), frame["cat"]):
            tick.set_color(COLORS[cat])
            if cat in ("a1", "a2", "a3", "a4"):
                tick.set_fontweight("bold")

    # ---- Panel 4: Action Distribution per ablation -----------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ablations = ["a1", "a2", "a3", "a4"]
    y = np.arange(len(ACTION_ORDER))
    bar_h = 0.18
    for i, abl in enumerate(ablations):
        vals = [actions[abl][a] for a in ACTION_ORDER]
        offset = (i - (len(ablations) - 1) / 2) * bar_h
        ax4.barh(y + offset, vals, height=bar_h, color=COLORS[abl],
                 edgecolor="black", linewidth=0.4, label=ABLATION_LABELS[abl])
        for j, v in enumerate(vals):
            if v >= 1.0:
                ax4.text(v + 0.6, y[j] + offset, f"{v:.1f}%",
                         va="center", ha="left", fontsize=7)
    ax4.set_yticks(y)
    ax4.set_yticklabels(ACTION_ORDER)
    ax4.set_xlabel("Percent of eval timesteps (%)")
    ax4.set_title("Action Distribution per Ablation\n(executed actions, 554 days)",
                  fontweight="bold")
    ax4.set_xlim(0, 100)
    ax4.grid(axis="x", alpha=0.3)
    ax4.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax4.invert_yaxis()

    # data-driven footnote: confirm A1==A2 / A3==A4 and the HOLD shift from actions.csv
    def _counts(abl: str) -> pd.Series:
        d = pd.read_csv(layer4_dir / abl / "actions.csv")
        return d["final_action"].str.upper().value_counts()

    c1, c2, c3, c4 = (_counts(a) for a in ("a1", "a2", "a3", "a4"))
    foot_parts = []
    if c1.to_dict() == c2.to_dict():
        foot_parts.append("A1≡A2 (HDP changes sizing only)")
    if c3.to_dict() == c4.to_dict():
        hold_delta = int(c3.get("HOLD", 0) - c1.get("HOLD", 0))
        foot_parts.append(
            f"A3≡A4 (EDL gate forces +{hold_delta} BUY/SELL → HOLD)"
        )
    if foot_parts:
        ax4.text(
            0.985, 0.03, "\n".join(foot_parts), transform=ax4.transAxes,
            ha="right", va="bottom", fontsize=8, style="italic", clip_on=False,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
        )

    # ---- Master legend ---------------------------------------------------
    legend_patches = [
        mpatches.Patch(color=COLORS["passive"], label="Passive buy-and-hold (467 tickers)"),
        mpatches.Patch(color=COLORS["classical"], label="Classical portfolio benchmark (Layer 1a)"),
        mpatches.Patch(color=COLORS["rule"], label="Rule-based portfolio (467 tickers)"),
        mpatches.Patch(color=COLORS["finrl"], label="FinRL parametric RL + MVO"),
        mpatches.Patch(color=COLORS["a1"], label="D-IQN-DSS (A1)"),
        mpatches.Patch(color=COLORS["a2"], label="D-IQN-DSS + HDP (A2)"),
        mpatches.Patch(color=COLORS["a3"], label="D-IQN-DSS + EDL (A3)"),
        mpatches.Patch(color=COLORS["a4"], label="D-IQN-DSS + HDP + EDL (A4)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.005), fontsize=10, frameon=True)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- console summary --------------------------------------------------
    print(f"[portfolio_summary_dashboard] Layer 4 dir: {layer4_dir}")
    print(f"[portfolio_summary_dashboard] Saved: {out_png}")

    print("\nBars per category:")
    cat_order = ["passive", "classical", "rule", "finrl", "a1", "a2", "a3", "a4"]
    cat_counts = perf_ret["cat"].value_counts()
    for cat in cat_order:
        print(f"  {cat:10} {int(cat_counts.get(cat, 0))}")
    print(f"  {'TOTAL':10} {len(perf_ret)}")

    print("\nFootnote (computed from actions.csv, NOT hardcoded):")
    if foot_parts:
        print(f'  "{"  ·  ".join(foot_parts)}"')
        if c3.to_dict() == c4.to_dict():
            print(
                f"  -> HOLD delta {int(c3.get('HOLD', 0) - c1.get('HOLD', 0))} = "
                f"A3 HOLD {int(c3.get('HOLD', 0))} - A1 HOLD {int(c1.get('HOLD', 0))} "
                f"(read live from a1/a3 actions.csv)"
            )
    else:
        print("  (no footnote: A1!=A2 or A3!=A4 in this run)")

    print("\nPanel 4 — action distribution (counts) per ablation:")
    for abl in ablations:
        df = pd.read_csv(layer4_dir / abl / "actions.csv")
        vc = df["final_action"].str.upper().value_counts()
        cells = "  ".join(f"{a}:{int(vc.get(a, 0))}" for a in ACTION_ORDER)
        print(f"  {abl} {ABLATION_LABELS[abl]:24} {cells}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--layer4-dir",
        default=None,
        help="Layer 4 backtest run dir (default: newest *_scenarioX run).",
    )
    ap.add_argument("--output", default=OUTPUT_PNG, help="Output PNG path.")
    args = ap.parse_args()

    layer4_dir = resolve_layer4_dir(args.layer4_dir)
    out_png = Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output
    build_dashboard(layer4_dir, out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
