"""
Etape 1 — V1 IQN Distributional Decision Visualizations (V2-native port)

Post-run visualization script. Reads existing V2 clean 25k eval_distributions.csv
artifacts and produces IQN distributional decision-support plots inspired by the V1
visualize_iqn_decision_distribution.py reference implementation.

NO training is performed. NO existing run outputs are modified.
All outputs are written to: outputs/run_registry/v1_visualization_ports/

Snapshots:
  seed5_step25000  — seed 5, train_step=25000, eval_step=599  (final active policy)
  seed5_step10000  — seed 5, train_step=10000, eval_step=599  (early-stopping diagnostic)
  seed6_step25000  — seed 6, train_step=25000, eval_step=599  (HOLD-collapse contrast)

Per snapshot (6 artifacts × 3 snapshots = 18 files):
  1. iqn_quantile_function_per_action_<snapshot>.png
  2. iqn_return_distribution_per_action_<snapshot>.png
  3. iqn_risk_adjusted_score_per_action_<snapshot>.png
  4. iqn_decision_table_<snapshot>.png
  5. iqn_decision_dashboard_<snapshot>.png      (4-panel)
  6. iqn_decision_table_<snapshot>.csv

Plus:
  outputs/run_registry/v1_visualization_ports/README.md
  outputs/run_registry/v1_visualization_ports/plot_manifest.json

Hybrid highlighting:
  score       = q50 (actual V2 q50-greedy policy score; score_mode=q50 in clean 25k runs)
  risk_adj    = q50 - 0.75 * abs(cvar10)  (V1-style secondary analysis — NOT used by policy)
  chosen      = chosen_action column (actual policy decision)
  risk_winner = argmax(risk_adj) among allowed actions

  If chosen == risk_winner (agreement):   single combined highlight + star marker
  If chosen != risk_winner (disagreement): two distinct markers — square (policy) + diamond (risk-adj)
"""

import json
import pathlib
import datetime
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from stock_investment_dss.utilities.paths import create_run_paths

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
METRICS_CSV = (
    PROJECT_ROOT / "outputs" / "run_registry" / "clean_25k_hold_diagnostic_metrics.csv"
)

RISK_LAMBDA = 0.75

SNAPSHOTS = [
    {"label": "seed5_step25000", "seed": 5, "train_step": 25000, "eval_step": 599},
    {"label": "seed5_step10000", "seed": 5, "train_step": 10000, "eval_step": 599},
    {"label": "seed6_step25000", "seed": 6, "train_step": 25000, "eval_step": 599},
]

ACTION_COLORS = {
    "HOLD": "#2196F3",
    "BUY": "#4CAF50",
    "SELL": "#F44336",
    "REBALANCE": "#FF9800",
    "CHANGE_STRATEGY": "#9C27B0",
}
MASKED_COLOR = "#aaaaaa"


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------


def load_metrics_csv() -> pd.DataFrame:
    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {METRICS_CSV}")
    df = pd.read_csv(METRICS_CSV)
    required = {"seed", "source_run_directory"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metrics CSV missing columns: {missing}")
    return df


def resolve_run_directory(metrics_df: pd.DataFrame, seed: int) -> pathlib.Path:
    rows = metrics_df[metrics_df["seed"] == seed]
    if len(rows) == 0:
        raise ValueError(f"Seed {seed} not found in metrics CSV {METRICS_CSV}")
    run_rel = rows.iloc[0]["source_run_directory"]
    run_dir = PROJECT_ROOT / run_rel
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}\n"
            f"(from source_run_directory column: {run_rel})"
        )
    return run_dir


def load_snapshot(
    run_dir: pathlib.Path, train_step: int, eval_step: int
) -> pd.DataFrame:
    csv_path = run_dir / "data" / "iqn_learning_curve_eval_distributions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"eval_distributions.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = [
        "train_step",
        "eval_step",
        "chosen_action",
        "action",
        "action_index",
        "allowed",
        "score_mode",
        "score",
        "mean",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "cvar10",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"eval_distributions.csv at {csv_path} missing columns: {missing}"
        )

    snap = df[(df["train_step"] == train_step) & (df["eval_step"] == eval_step)].copy()
    if len(snap) == 0:
        available_ts = sorted(df["train_step"].unique())
        available_es = (
            sorted(df[df["train_step"] == train_step]["eval_step"].unique())
            if train_step in df["train_step"].values
            else []
        )
        raise ValueError(
            f"No rows for train_step={train_step}, eval_step={eval_step} in {csv_path}\n"
            f"  Available train_steps: {available_ts}\n"
            f"  Available eval_steps for train_step={train_step}: {available_es}\n"
            f"  (No silent fallback — fix train_step/eval_step or data source)"
        )

    snap = snap.reset_index(drop=True)
    snap["allowed_bool"] = snap["allowed"].astype(str).str.lower() == "true"
    snap["risk_adjusted_score"] = snap["q50"] - RISK_LAMBDA * snap["cvar10"].abs()

    chosen = snap["chosen_action"].iloc[0]
    allowed_snap = snap[snap["allowed_bool"]]
    if len(allowed_snap) == 0:
        raise ValueError(
            f"No allowed actions in snapshot {train_step}/{eval_step} from {csv_path}"
        )
    risk_adj_winner = allowed_snap.loc[
        allowed_snap["risk_adjusted_score"].idxmax(), "action"
    ]
    agreement = chosen == risk_adj_winner

    snap["is_chosen_action"] = snap["action"] == chosen
    snap["is_risk_adjusted_winner"] = snap["action"] == risk_adj_winner

    return snap, chosen, risk_adj_winner, agreement


# ---------------------------------------------------------------------------
# Action styling
# ---------------------------------------------------------------------------


def get_action_style(
    row: pd.Series, chosen: str, risk_adj_winner: str, agreement: bool
) -> dict:
    action = row["action"]
    is_allowed = bool(row["allowed_bool"])

    if not is_allowed:
        return dict(
            color=MASKED_COLOR,
            lw=1.0,
            alpha=0.4,
            ls="--",
            zorder=1,
            marker="",
            markersize=6,
            suffix=" (masked)",
        )

    if action == chosen and action == risk_adj_winner:
        return dict(
            color=ACTION_COLORS.get(action, "#555555"),
            lw=3.0,
            alpha=1.0,
            ls="-",
            zorder=3,
            marker="*",
            markersize=14,
            suffix=" ★ Policy & Risk-Adj.",
        )
    elif action == chosen:
        return dict(
            color=ACTION_COLORS.get(action, "#555555"),
            lw=3.0,
            alpha=1.0,
            ls="-",
            zorder=3,
            marker="s",
            markersize=9,
            suffix=" ■ Policy choice",
        )
    elif action == risk_adj_winner:
        return dict(
            color=ACTION_COLORS.get(action, "#555555"),
            lw=2.5,
            alpha=0.9,
            ls="--",
            zorder=2,
            marker="D",
            markersize=9,
            suffix=" ◆ Risk-adj. winner",
        )
    else:
        return dict(
            color=ACTION_COLORS.get(action, "#555555"),
            lw=1.5,
            alpha=0.65,
            ls="-",
            zorder=1,
            marker="",
            markersize=6,
            suffix="",
        )


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

TAU_POINTS = [0.10, 0.25, 0.50, 0.75, 0.90]
TAU_DENSE = np.linspace(0.10, 0.90, 100)


def _draw_quantile_function_on_ax(ax, snap_df, chosen, risk_adj_winner, agreement):
    for _, row in snap_df.iterrows():
        q_vals = [row["q10"], row["q25"], row["q50"], row["q75"], row["q90"]]
        q_interp = np.interp(TAU_DENSE, TAU_POINTS, q_vals)
        style = get_action_style(row, chosen, risk_adj_winner, agreement)
        label = row["action"] + style["suffix"]
        ax.plot(
            TAU_DENSE,
            q_interp,
            color=style["color"],
            lw=style["lw"],
            alpha=style["alpha"],
            ls=style["ls"],
            zorder=style["zorder"],
            label=label,
        )
        if style["marker"]:
            ax.plot(
                0.50,
                row["q50"],
                marker=style["marker"],
                color=style["color"],
                markersize=style["markersize"],
                zorder=style["zorder"] + 1,
            )
    ax.axhline(0, color="black", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("τ (quantile level)")
    ax.set_ylabel("Q-value estimate (portfolio units)")
    score_mode = snap_df["score_mode"].iloc[0]
    ax.set_title(f"IQN Quantile Function per Action\nscore_mode={score_mode}")
    ax.legend(loc="upper left", fontsize=8)


def plot_quantile_function(
    snap_df, snapshot_label, chosen, risk_adj_winner, agreement, output_dir
):
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_quantile_function_on_ax(ax, snap_df, chosen, risk_adj_winner, agreement)
    ax.set_title(
        f"IQN Quantile Function per Action\n{snapshot_label} | score_mode={snap_df['score_mode'].iloc[0]}"
    )
    fig.tight_layout()
    path = output_dir / f"iqn_quantile_function_per_action_{snapshot_label}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _draw_return_distribution_on_ax(ax, snap_df, chosen, risk_adj_winner, agreement):
    rng = np.random.default_rng(42)
    N = 2000
    for _, row in snap_df.iterrows():
        q_vals = [row["q10"], row["q25"], row["q50"], row["q75"], row["q90"]]
        iqr = row["q90"] - row["q10"]
        if iqr <= 0:
            iqr = max(abs(row["q50"]) * 0.1, 0.01)
        taus = rng.uniform(0.10, 0.90, size=N)
        samples = np.interp(taus, TAU_POINTS, q_vals)
        samples += rng.normal(0, iqr * 0.02, size=N)
        style = get_action_style(row, chosen, risk_adj_winner, agreement)
        label = row["action"] + style["suffix"]
        hist, edges = np.histogram(samples, bins=50, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(
            centers,
            hist,
            color=style["color"],
            lw=style["lw"],
            alpha=style["alpha"],
            ls=style["ls"],
            zorder=style["zorder"],
            label=label,
        )
    ax.set_xlabel("Q-value estimate (portfolio units)")
    ax.set_ylabel("Approximate density")
    ax.set_title("Approx. Return Distribution per Action")
    ax.text(
        0.02,
        0.97,
        "Visual approximation from quantile interpolation\n— not a formal density estimate",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        color="#666666",
        style="italic",
    )
    ax.legend(loc="upper right", fontsize=8)


def plot_return_distribution(
    snap_df, snapshot_label, chosen, risk_adj_winner, agreement, output_dir
):
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_return_distribution_on_ax(ax, snap_df, chosen, risk_adj_winner, agreement)
    ax.set_title(
        f"IQN Approx. Return Distribution per Action\n"
        f"{snapshot_label}\n"
        "Visual approximation from quantile interpolation — not a formal density estimate"
    )
    fig.tight_layout()
    path = output_dir / f"iqn_return_distribution_per_action_{snapshot_label}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _draw_score_bars_on_ax(ax, snap_df, chosen, risk_adj_winner, agreement):
    n = len(snap_df)
    y = np.arange(n)
    height = 0.35

    for i, (_, row) in enumerate(snap_df.iterrows()):
        is_allowed = bool(row["allowed_bool"])
        style = get_action_style(row, chosen, risk_adj_winner, agreement)
        c = style["color"]
        alpha_solid = 0.85 if is_allowed else 0.35
        alpha_hatch = 0.55 if is_allowed else 0.2

        ax.barh(
            y[i] + height / 2,
            row["score"],
            height=height,
            color=c,
            alpha=alpha_solid,
            edgecolor="none",
        )
        ax.barh(
            y[i] - height / 2,
            row["risk_adjusted_score"],
            height=height,
            color=c,
            alpha=alpha_hatch,
            hatch="///",
            edgecolor=c,
        )

        suffix = style["suffix"]
        if suffix.strip():
            ref_val = max(row["score"], row["risk_adjusted_score"])
            x_pos = ref_val + abs(ref_val) * 0.03 + 0.05
            ax.text(
                x_pos,
                y[i],
                suffix.strip(),
                va="center",
                fontsize=7,
                color=c,
                fontweight="bold" if "Policy" in suffix or "★" in suffix else "normal",
            )

    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_yticks(y)
    labels = [
        row["action"] + ("" if row["allowed_bool"] else " (masked)")
        for _, row in snap_df.iterrows()
    ]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Score value (portfolio units)")
    ax.set_title("Policy Score per Action")
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.85, label="score = q50  (V2 policy)"),
        Patch(
            facecolor="steelblue",
            alpha=0.55,
            hatch="///",
            label=f"risk_adj = q50 − {RISK_LAMBDA}·|CVaR10|  (V1-style)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)


def plot_risk_adjusted_score(
    snap_df, snapshot_label, chosen, risk_adj_winner, agreement, output_dir
):
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_score_bars_on_ax(ax, snap_df, chosen, risk_adj_winner, agreement)
    ax.set_title(
        f"Policy Score per Action | {snapshot_label}\n"
        f"solid = score=q50 (V2 policy)  ·  hatched = q50−{RISK_LAMBDA}·|CVaR10| (V1-style analysis)"
    )
    fig.tight_layout()
    path = output_dir / f"iqn_risk_adjusted_score_per_action_{snapshot_label}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _draw_decision_table_on_ax(ax, snap_df, snapshot_label):
    ax.axis("off")
    display_cols = [
        "action",
        "allowed",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "cvar10",
        "score",
        "risk_adjusted_score",
        "is_chosen_action",
        "is_risk_adjusted_winner",
    ]
    disp = snap_df.copy()
    float_fmt_cols = [
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "cvar10",
        "score",
        "risk_adjusted_score",
    ]
    for c in float_fmt_cols:
        disp[c] = disp[c].apply(lambda x: f"{x:.3f}")

    cell_data = disp[display_cols].values.tolist()

    row_colors = []
    for _, row in snap_df.iterrows():
        is_allowed = bool(row["allowed_bool"])
        is_chosen = bool(row["is_chosen_action"])
        is_winner = bool(row["is_risk_adjusted_winner"])
        if not is_allowed:
            row_colors.append(["#ebebeb"] * len(display_cols))
        elif is_chosen and is_winner:
            row_colors.append(["#fff3cd"] * len(display_cols))
        elif is_chosen:
            row_colors.append(["#d4edda"] * len(display_cols))
        elif is_winner:
            row_colors.append(["#cce5ff"] * len(display_cols))
        else:
            row_colors.append(["#ffffff"] * len(display_cols))

    col_labels = [
        "action",
        "allowed",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "cvar10",
        "score\n(q50)",
        "risk_adj\nscore",
        "chosen\naction",
        "risk_adj\nwinner",
    ]
    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    ax.set_title(
        f"Decision Table | {snapshot_label}\n"
        "yellow=policy & risk-adj. winner  ·  green=policy only  ·  blue=risk-adj. only",
        fontsize=9,
        pad=12,
    )


def plot_decision_table_png(
    snap_df, snapshot_label, chosen, risk_adj_winner, agreement, output_dir
):
    fig, ax = plt.subplots(figsize=(14, 4))
    _draw_decision_table_on_ax(ax, snap_df, snapshot_label)
    fig.tight_layout()
    path = output_dir / f"iqn_decision_table_{snapshot_label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_decision_table_csv(snap_df, snapshot_label, output_dir):
    out_cols = [
        "action",
        "action_index",
        "allowed",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "cvar10",
        "mean",
        "score",
        "chosen_action",
        "score_mode",
        "train_step",
        "eval_step",
        "risk_adjusted_score",
        "is_chosen_action",
        "is_risk_adjusted_winner",
    ]
    out = snap_df[out_cols].copy()
    path = output_dir / f"iqn_decision_table_{snapshot_label}.csv"
    out.to_csv(path, index=False)
    return path


def plot_decision_dashboard(
    snap_df, snapshot_label, chosen, risk_adj_winner, agreement, output_dir
):
    fig = plt.figure(figsize=(18, 13))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax_qfn = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_score = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])

    _draw_quantile_function_on_ax(ax_qfn, snap_df, chosen, risk_adj_winner, agreement)
    _draw_return_distribution_on_ax(
        ax_dist, snap_df, chosen, risk_adj_winner, agreement
    )
    _draw_score_bars_on_ax(ax_score, snap_df, chosen, risk_adj_winner, agreement)
    _draw_decision_table_on_ax(ax_table, snap_df, snapshot_label)

    agreement_str = "AGREEMENT" if agreement else "DISAGREEMENT"
    fig.suptitle(
        f"IQN Decision Support Dashboard  |  {snapshot_label}\n"
        f"chosen = {chosen}  ·  risk_adj_winner = {risk_adj_winner}  ·  {agreement_str}\n"
        f"score = q50 (V2 policy)   risk_adjusted_score = q50 − {RISK_LAMBDA}·|CVaR10| (V1-style analysis, not used by policy)",
        fontsize=11,
        y=0.99,
    )
    path = output_dir / f"iqn_decision_dashboard_{snapshot_label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# README and manifest
# ---------------------------------------------------------------------------


def write_readme(output_dir: pathlib.Path, snapshot_infos: list) -> pathlib.Path:
    lines = textwrap.dedent(f"""
    # V1 IQN Distributional Decision Visualizations
    ## Etape 1 — V1 → V2 Thesis Port Plan

    ### What this is
    This folder contains post-run IQN distributional decision-support visualizations
    inspired by the V1 reference implementation:
      external/ObjectRL_style/src/stockdss/rl/experiments/visualize_iqn_decision_distribution.py

    These plots were generated by:
      src/stock_investment_dss/runner/run_v1_visualization_ports.py

    ### No training was performed
    All visualizations are derived from existing V2 clean 25k run artifacts only.
    The script reads eval_distributions.csv files from clean May 23 runs and produces
    decision-support plots. No model weights were trained, updated, or modified.

    ### V2 is the canonical thesis pipeline
    V1 is used only as a visualization/design/reference implementation donor.
    V1 code is not integrated into V2. The V2 training, evaluation, and data pipeline
    remain unchanged.

    ### Data sources
    Only clean May 23 runs (2026_05_23_*) are used. Old/confounded May 22 runs are
    not used. The seed-to-run mapping is resolved from:
      outputs/run_registry/clean_25k_hold_diagnostic_metrics.csv

    ### Snapshots
    """).lstrip()

    for info in snapshot_infos:
        lines += (
            f"  - {info['label']}: seed={info['seed']}, train_step={info['train_step']}, "
            f"eval_step={info['eval_step']}, chosen={info['chosen']}, "
            f"risk_adj_winner={info['risk_adj_winner']}, "
            f"agreement={'YES' if info['agreement'] else 'NO'}\n"
        )

    lines += textwrap.dedent("""
    ### Score columns
    - `score` = q50 — the actual V2 q50-greedy policy score (score_mode=q50 in clean 25k runs)
    - `risk_adjusted_score` = q50 − 0.75·|CVaR10| — a secondary V1-inspired decision-support
      analysis score. This was NOT used by the clean 25k policy itself.

    ### Return distribution plots
    The approximate return distribution plots are visual approximations derived from
    quantile interpolation (q10/q25/q50/q75/q90) with small proportional noise.
    They are NOT formal density estimates and should not be interpreted as such.
    Labelled clearly on each plot.

    ### Highlighting
    - ★ star marker: Policy choice AND risk-adjusted winner (agreement)
    - ■ square marker: Policy choice only (disagreement)
    - ◆ diamond marker: Risk-adjusted winner only (disagreement)
    - Gray dashed: masked/not-allowed actions

    ### Thesis relevance
    These figures support the thesis problem formulation by showing:
    - Distributional IQN action estimates (full quantile functions per action)
    - Risk-sensitive decision support (q50 vs CVaR-adjusted scoring)
    - The HOLD-collapse phenomenon: seed 6 is trapped in a least-bad scenario
      where both HOLD and BUY have negative risk-adjusted scores
    - Contrast between active policy (seed 5 step 25000, BUY wins both metrics)
      and HOLD-collapse (seed 6 step 25000, all allowed actions have negative risk-adj scores)
    """)

    path = output_dir / "README.md"
    path.write_text(lines.strip() + "\n", encoding="utf-8")
    return path


def write_plot_manifest(
    output_dir: pathlib.Path,
    snapshot_infos: list,
    generated_files: list,
) -> pathlib.Path:
    manifest = {
        "generated_at": datetime.datetime.now().isoformat(),
        "no_training_performed": True,
        "existing_run_outputs_modified": False,
        "source_metrics_csv": str(METRICS_CSV.relative_to(PROJECT_ROOT)),
        "risk_lambda": RISK_LAMBDA,
        "snapshots": [
            {
                "label": info["label"],
                "seed": info["seed"],
                "train_step": info["train_step"],
                "eval_step": info["eval_step"],
                "chosen_action": info["chosen"],
                "risk_adj_winner": info["risk_adj_winner"],
                "agreement": info["agreement"],
                "source_run_directory": info["source_run_directory"],
                "eval_distributions_csv": info["eval_distributions_csv"],
                "output_files": info["output_files"],
            }
            for info in snapshot_infos
        ],
        "all_output_files": [
            str(pathlib.Path(f).relative_to(PROJECT_ROOT)) for f in generated_files
        ],
        "note_score_mode": (
            "score_mode=q50 in all clean 25k runs. "
            "score = q50 (V2 policy). "
            f"risk_adjusted_score = q50 - {RISK_LAMBDA} * abs(cvar10) (V1-style secondary analysis, "
            "NOT used by the clean 25k policy)."
        ),
        "note_return_distributions": (
            "Approximate return distribution plots are visual approximations from quantile "
            "interpolation with proportional noise. Not formal density estimates."
        ),
    }
    path = output_dir / "plot_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    run_paths = create_run_paths("d_iqn_dss_v1_visualization_ports")
    print(f"[run_v1_visualization_ports] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[run_v1_visualization_ports] Run dir      = {run_paths.run_directory}")

    metrics_df = load_metrics_csv()
    print(f"[run_v1_visualization_ports] Loaded metrics CSV: {len(metrics_df)} rows")

    snapshot_infos = []
    all_generated_files = []

    for spec in SNAPSHOTS:
        label = spec["label"]
        seed = spec["seed"]
        train_step = spec["train_step"]
        eval_step = spec["eval_step"]
        print(
            f"\n[{label}] Processing seed={seed}, train_step={train_step}, eval_step={eval_step}"
        )

        run_dir = resolve_run_directory(metrics_df, seed)
        print(f"  run_dir = {run_dir.relative_to(PROJECT_ROOT)}")

        snap_df, chosen, risk_adj_winner, agreement = load_snapshot(
            run_dir, train_step, eval_step
        )
        print(f"  chosen_action = {chosen}")
        print(f"  risk_adj_winner (allowed only) = {risk_adj_winner}")
        print(f"  agreement = {agreement}")
        print(f"  score_mode = {snap_df['score_mode'].iloc[0]}")

        eval_dist_csv = str(
            (
                run_dir / "data" / "iqn_learning_curve_eval_distributions.csv"
            ).relative_to(PROJECT_ROOT)
        )

        outputs_for_snapshot = []

        plots_subdir = run_paths.plots_directory / label
        data_subdir = run_paths.data_directory / label
        plots_subdir.mkdir(parents=True, exist_ok=True)
        data_subdir.mkdir(parents=True, exist_ok=True)

        p = plot_quantile_function(
            snap_df, label, chosen, risk_adj_winner, agreement, plots_subdir
        )
        outputs_for_snapshot.append(str(p.relative_to(PROJECT_ROOT)))
        all_generated_files.append(p)
        print(f"  [1/5] quantile_function → {p.name}")

        p = plot_return_distribution(
            snap_df, label, chosen, risk_adj_winner, agreement, plots_subdir
        )
        outputs_for_snapshot.append(str(p.relative_to(PROJECT_ROOT)))
        all_generated_files.append(p)
        print(f"  [2/5] return_distribution → {p.name}")

        p = plot_risk_adjusted_score(
            snap_df, label, chosen, risk_adj_winner, agreement, plots_subdir
        )
        outputs_for_snapshot.append(str(p.relative_to(PROJECT_ROOT)))
        all_generated_files.append(p)
        print(f"  [3/5] risk_adjusted_score → {p.name}")

        p = plot_decision_table_png(
            snap_df, label, chosen, risk_adj_winner, agreement, plots_subdir
        )
        outputs_for_snapshot.append(str(p.relative_to(PROJECT_ROOT)))
        all_generated_files.append(p)
        print(f"  [4a/5] decision_table_png → {p.name}")

        p = plot_decision_dashboard(
            snap_df, label, chosen, risk_adj_winner, agreement, plots_subdir
        )
        outputs_for_snapshot.append(str(p.relative_to(PROJECT_ROOT)))
        all_generated_files.append(p)
        print(f"  [4b/5] decision_dashboard → {p.name}")

        p = save_decision_table_csv(snap_df, label, data_subdir)
        outputs_for_snapshot.append(str(p.relative_to(PROJECT_ROOT)))
        all_generated_files.append(p)
        print(f"  [5/5] decision_table_csv → {p.name}")

        snapshot_infos.append(
            {
                "label": label,
                "seed": seed,
                "train_step": train_step,
                "eval_step": eval_step,
                "chosen": chosen,
                "risk_adj_winner": risk_adj_winner,
                "agreement": agreement,
                "source_run_directory": str(run_dir.relative_to(PROJECT_ROOT)),
                "eval_distributions_csv": eval_dist_csv,
                "output_files": outputs_for_snapshot,
            }
        )

    readme_path = write_readme(run_paths.summary_directory, snapshot_infos)
    print(f"\n[run_v1_visualization_ports] README → {readme_path.name}")

    manifest_path = write_plot_manifest(run_paths.summary_directory, snapshot_infos, all_generated_files)
    print(f"[run_v1_visualization_ports] manifest → {manifest_path.name}")

    print("\n[run_v1_visualization_ports] Done.")
    print(f"  Run directory: {run_paths.run_directory}")
    png_count = sum(1 for _ in run_paths.plots_directory.rglob("*.png"))
    csv_count = sum(1 for _ in run_paths.data_directory.rglob("*.csv"))
    print(f"  PNG files: {png_count}")
    print(f"  CSV files: {csv_count}")
    print(f"  README.md: {(run_paths.summary_directory / 'README.md').exists()}")
    print(f"  plot_manifest.json: {(run_paths.summary_directory / 'plot_manifest.json').exists()}")

    if png_count != 15:
        print(f"  WARNING: expected 15 PNG files, got {png_count}", file=sys.stderr)
    if csv_count != 3:
        print(f"  WARNING: expected 3 CSV files, got {csv_count}", file=sys.stderr)


if __name__ == "__main__":
    main()
