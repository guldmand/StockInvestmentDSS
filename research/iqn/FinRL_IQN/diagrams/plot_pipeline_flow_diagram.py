"""
Phase B.5 Pipeline Flow Diagram - Matplotlib version

Generates a publication-quality flow diagram showing how decisions
flow through the D-IQN-DSS pipeline for each of the 4 ablations.

Usage:
    python plot_pipeline_flow_diagram.py

Output:
    pipeline_flow_diagram.png (300 DPI)
    pipeline_flow_diagram_simplified.png (per-ablation simplified)
    gate_output_flow.png (gate decision routing)
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np


# ============================================================================
# Color palette (matching mermaid diagram)
# ============================================================================
COLORS = {
    "iqn": "#e1f5ff",
    "iqn_border": "#0277bd",
    "hdp": "#fff4e6",
    "hdp_border": "#e65100",
    "edl": "#fce4ec",
    "edl_border": "#c2185b",
    "rar": "#e0f2f1",
    "rar_border": "#00695c",
    "recommend": "#c8e6c9",
    "recommend_border": "#388e3c",
    "reduce": "#fff9c4",
    "reduce_border": "#f9a825",
    "force": "#ffcdd2",
    "force_border": "#c62828",
    "review": "#ffccbc",
    "review_border": "#e64a19",
    "strategy": "#f3e5f5",
    "strategy_border": "#6a1b9a",
    "input": "#f5f5f5",
    "input_border": "#616161",
    "output": "#e8eaf6",
    "output_border": "#283593",
    "ablation_a1": "#e3f2fd",
    "ablation_a2": "#fff3e0",
    "ablation_a3": "#fce4ec",
    "ablation_a4": "#f3e5f5",
}


def draw_box(ax, x, y, width, height, text, color="white", edge_color="black",
             text_size=9, text_weight="normal", line_width=1.5):
    """Draw a rounded rectangle box with centered text."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2),
        width, height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=color,
        edgecolor=edge_color,
        linewidth=line_width,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=text_size, fontweight=text_weight, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color="#424242", style="->", line_width=1.5,
               label=None, label_offset_x=0, label_offset_y=0, label_size=8):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=color,
        linewidth=line_width,
        mutation_scale=15,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arrow)
    if label:
        mid_x = (x1 + x2) / 2 + label_offset_x
        mid_y = (y1 + y2) / 2 + label_offset_y
        ax.text(mid_x, mid_y, label, ha="center", va="center",
                fontsize=label_size, style="italic",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


# ============================================================================
# FIGURE 1: Full pipeline flow diagram
# ============================================================================
def plot_full_pipeline_flow(output_path="pipeline_flow_diagram.png"):
    """Generate the comprehensive full pipeline flow diagram."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 13)
    ax.axis("off")

    # Title
    ax.text(8, 12.5, "D-IQN-DSS Pipeline Flow — Phase B.5 Ablation Architecture",
            ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(8, 12.0, "Test data: n=120 rows | All 4 ablations identical action class | Components operate orthogonally",
            ha="center", va="center", fontsize=10, style="italic", color="#555")

    # INPUT
    draw_box(ax, 8, 11.0, 4.5, 0.7, "Input: market state + features\nn=120 test rows",
             COLORS["input"], COLORS["input_border"], 10)

    # IQN
    draw_box(ax, 8, 10.0, 4.5, 0.7, "IQN Model\n(outputs action class only)",
             COLORS["iqn"], COLORS["iqn_border"], 11, text_weight="bold")
    draw_arrow(ax, 8, 10.65, 8, 10.35)

    # Action class outputs (4 boxes side by side)
    actions = [
        ("BUY\n73.3% n=88", 2.5, 9.0, COLORS["recommend"], COLORS["recommend_border"]),
        ("SELL\n24.2% n=29", 6.5, 9.0, COLORS["reduce"], COLORS["reduce_border"]),
        ("HOLD\n2.5% n=3", 10.5, 9.0, COLORS["input"], COLORS["input_border"]),
        ("REBALANCE\n0% n=0", 14, 9.0, COLORS["strategy"], COLORS["strategy_border"]),
    ]
    for text, x, y, color, edge in actions:
        draw_box(ax, x, y, 2.5, 0.6, text, color, edge, 9)
        draw_arrow(ax, 8, 9.65, x, y + 0.3, color="#666", line_width=1.0)

    # Size route decision diamond
    diamond_x, diamond_y = 8, 7.5
    diamond = patches.Polygon(
        [(diamond_x, diamond_y + 0.5), (diamond_x + 1.0, diamond_y),
         (diamond_x, diamond_y - 0.5), (diamond_x - 1.0, diamond_y)],
        facecolor="#fffde7", edgecolor="#f57f17", linewidth=1.5
    )
    ax.add_patch(diamond)
    ax.text(diamond_x, diamond_y, "Size determination\npath", ha="center", va="center",
            fontsize=9, fontweight="bold")

    # Arrows from actions to diamond
    for _, x, y, _, _ in actions:
        draw_arrow(ax, x, y - 0.3, diamond_x, diamond_y + 0.3, color="#888", line_width=0.8)

    # RAR path (A1, A3)
    draw_box(ax, 3.5, 6.0, 4.5, 1.0,
             "RiskAwareActionResolver\n(InvestorRiskProfile, cash, holdings,\nhmax, prices)\nStatic audit: N/A",
             COLORS["rar"], COLORS["rar_border"], 9)
    draw_arrow(ax, 7.0, 7.5, 4.5, 6.5, label="A1: IQN-only\nA3: IQN+EDL",
               label_offset_x=-1.5, label_offset_y=0.3)

    # HDP path (A2, A4)
    draw_box(ax, 12.5, 6.0, 4.5, 1.0,
             "HDP SizeSelector\n(BUY_25/50/75/100 buckets,\nrisk-adjusted)\nMean size: 0.425",
             COLORS["hdp"], COLORS["hdp_border"], 9, text_weight="bold")
    draw_arrow(ax, 9.0, 7.5, 11.5, 6.5, label="A2: IQN+HDP\nA4: IQN+HDP+EDL",
               label_offset_x=1.5, label_offset_y=0.3)

    # HDP additional stages
    draw_box(ax, 12.5, 4.7, 4.5, 0.6,
             "HDP TickerSelector\nAVGO 22%, ORCL 18%, BA 16% ...",
             COLORS["hdp"], COLORS["hdp_border"], 8)
    draw_arrow(ax, 12.5, 5.5, 12.5, 5.0)

    draw_box(ax, 12.5, 3.6, 4.5, 0.6,
             "HDP Risk Checks\n(bear market, concentration, drawdown)",
             COLORS["hdp"], COLORS["hdp_border"], 8)
    draw_arrow(ax, 12.5, 4.4, 12.5, 3.9)

    # EDL Gate (central)
    draw_box(ax, 8, 2.5, 5.5, 1.0,
             "EDL Gate (A3, A4 only)\nVacuity-based decision routing\n98.3% activation rate",
             COLORS["edl"], COLORS["edl_border"], 10, text_weight="bold")
    draw_arrow(ax, 3.5, 5.5, 7.0, 2.8, label="A3 path", label_offset_y=0.2)
    draw_arrow(ax, 12.5, 3.3, 9.0, 2.8, label="A4 path", label_offset_y=0.2)

    # Gate outputs (5 boxes)
    gate_outputs = [
        ("RECOMMEND\n_AS_IS\n1.7% n=2", 1.5, 0.9, COLORS["recommend"], COLORS["recommend_border"]),
        ("REDUCE_SIZE\n4.2% n=5", 4.5, 0.9, COLORS["reduce"], COLORS["reduce_border"]),
        ("FORCE_HOLD\n0% n=0", 7.8, 0.9, COLORS["force"], COLORS["force_border"]),
        ("HUMAN_REVIEW\n94.2% n=113", 11.0, 0.9, COLORS["review"], COLORS["review_border"]),
        ("STRATEGY\n_REVIEW\n0% n=0", 14.3, 0.9, COLORS["strategy"], COLORS["strategy_border"]),
    ]
    for text, x, y, color, edge in gate_outputs:
        draw_box(ax, x, y, 2.4, 0.85, text, color, edge, 8)
        draw_arrow(ax, 8, 2.0, x, y + 0.45, color="#666", line_width=0.8)

    # A1 and A2 direct outputs (bypass EDL)
    draw_box(ax, 3.5, 4.5, 4.5, 0.5, "A1 Final: action + runtime size",
             COLORS["ablation_a1"], COLORS["iqn_border"], 8)
    draw_arrow(ax, 3.5, 5.5, 3.5, 4.75, label="A1 (no EDL)", label_offset_x=-0.8)

    draw_box(ax, 12.5, 2.5, 4.5, 0.5, "A2 Final: action + HDP size",
             COLORS["ablation_a2"], COLORS["hdp_border"], 8)
    draw_arrow(ax, 12.5, 3.3, 12.5, 2.8, label="A2 (no EDL)", label_offset_x=0.8)

    # Legend
    legend_elements = [
        patches.Patch(facecolor=COLORS["iqn"], edgecolor=COLORS["iqn_border"], label="IQN (action class)"),
        patches.Patch(facecolor=COLORS["hdp"], edgecolor=COLORS["hdp_border"], label="HDP (ticker + size)"),
        patches.Patch(facecolor=COLORS["edl"], edgecolor=COLORS["edl_border"], label="EDL Gate (uncertainty)"),
        patches.Patch(facecolor=COLORS["rar"], edgecolor=COLORS["rar_border"], label="RiskAwareActionResolver"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.05),
              ncol=4, fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# FIGURE 2: Per-ablation simplified flow (4 panels)
# ============================================================================
def plot_per_ablation_flow(output_path="pipeline_flow_diagram_simplified.png"):
    """Generate simplified per-ablation flow diagrams (2x2 grid)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Per-Ablation Decision Flow Architecture",
                 fontsize=16, fontweight="bold", y=0.98)

    ablations = [
        {
            "title": "A1: IQN-only",
            "subtitle": "Baseline (no HDP, no EDL)",
            "color": COLORS["ablation_a1"],
            "stages": [
                ("IQN\n(action class)", COLORS["iqn"], COLORS["iqn_border"]),
                ("RiskAwareActionResolver\n(size at runtime)", COLORS["rar"], COLORS["rar_border"]),
                ("Final Decision\naction + runtime size", COLORS["output"], COLORS["output_border"]),
            ],
            "ax": axes[0, 0],
        },
        {
            "title": "A2: IQN + HDP",
            "subtitle": "HDP adds ticker + size selection",
            "color": COLORS["ablation_a2"],
            "stages": [
                ("IQN\n(action class)", COLORS["iqn"], COLORS["iqn_border"]),
                ("HDP\n(ticker + size 0.425 avg)", COLORS["hdp"], COLORS["hdp_border"]),
                ("Final Decision\naction + HDP size", COLORS["output"], COLORS["output_border"]),
            ],
            "ax": axes[0, 1],
        },
        {
            "title": "A3: IQN + EDL",
            "subtitle": "EDL gate filters by uncertainty",
            "color": COLORS["ablation_a3"],
            "stages": [
                ("IQN\n(action class)", COLORS["iqn"], COLORS["iqn_border"]),
                ("RiskAwareActionResolver", COLORS["rar"], COLORS["rar_border"]),
                ("EDL Gate\n(94.2% HUMAN_REVIEW)", COLORS["edl"], COLORS["edl_border"]),
                ("Final Decision\naction + runtime size + flags", COLORS["output"], COLORS["output_border"]),
            ],
            "ax": axes[1, 0],
        },
        {
            "title": "A4: IQN + HDP + EDL (Full Pipeline)",
            "subtitle": "All components active",
            "color": COLORS["ablation_a4"],
            "stages": [
                ("IQN\n(action class)", COLORS["iqn"], COLORS["iqn_border"]),
                ("HDP\n(ticker + size)", COLORS["hdp"], COLORS["hdp_border"]),
                ("EDL Gate\n(size modulation + flags)", COLORS["edl"], COLORS["edl_border"]),
                ("Final Decision\naction + final size", COLORS["output"], COLORS["output_border"]),
            ],
            "ax": axes[1, 1],
        },
    ]

    for ab in ablations:
        ax = ab["ax"]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_facecolor(ab["color"])

        # Title
        ax.text(5, 9.3, ab["title"], ha="center", va="center",
                fontsize=14, fontweight="bold")
        ax.text(5, 8.7, ab["subtitle"], ha="center", va="center",
                fontsize=10, style="italic", color="#555")

        # Frame
        ax.add_patch(Rectangle((0.2, 0.2), 9.6, 9.4, fill=False,
                               edgecolor=ab["color"], linewidth=4))

        # Calculate y positions for stages
        n_stages = len(ab["stages"])
        y_start = 7.8
        y_end = 1.5
        y_positions = np.linspace(y_start, y_end, n_stages)

        # Draw stages
        for i, (text, color, edge) in enumerate(ab["stages"]):
            draw_box(ax, 5, y_positions[i], 6.5, 0.9, text,
                     color, edge, 11, text_weight="bold")
            if i < n_stages - 1:
                draw_arrow(ax, 5, y_positions[i] - 0.45,
                           5, y_positions[i+1] + 0.45,
                           line_width=2.0, color="#424242")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# FIGURE 3: EDL Gate output flow
# ============================================================================
def plot_gate_output_flow(output_path="gate_output_flow.png"):
    """Generate EDL gate decision routing diagram."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(7, 9.5, "EDL Gate — Vacuity-Based Decision Routing",
            ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(7, 9.0, "Phase B.5 results (n=120 test rows, percentile thresholds)",
            ha="center", va="center", fontsize=10, style="italic", color="#555")

    # Input
    draw_box(ax, 7, 8.0, 5, 0.7, "EDL Gate Input\nn=120 decisions, vacuity range [0.27, 0.35]",
             COLORS["edl"], COLORS["edl_border"], 10, text_weight="bold")

    # Decision diamond
    diamond_x, diamond_y = 7, 6.8
    diamond = patches.Polygon(
        [(diamond_x, diamond_y + 0.6), (diamond_x + 1.5, diamond_y),
         (diamond_x, diamond_y - 0.6), (diamond_x - 1.5, diamond_y)],
        facecolor="#fffde7", edgecolor="#f57f17", linewidth=2.0
    )
    ax.add_patch(diamond)
    ax.text(diamond_x, diamond_y, "Vacuity-based\nrouting", ha="center", va="center",
            fontsize=10, fontweight="bold")
    draw_arrow(ax, 7, 7.65, 7, 7.4, line_width=2.0)

    # Threshold info
    ax.text(2, 6.8, "Thresholds:\nrec_as_is: ≤0.29 (p33)\nreduce_size: ≤0.31 (p66)\nforce_hold: ≥0.50",
            ha="left", va="center", fontsize=8, style="italic",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fff9c4", ec="#f9a825"))

    # Gate outputs (5 boxes in horizontal arrangement)
    gate_outputs_data = [
        ("RECOMMEND_AS_IS", "n=2\n1.7%", "Execute as recommended\n(low vacuity, confident)",
         1.4, 3.5, COLORS["recommend"], COLORS["recommend_border"]),
        ("REDUCE_SIZE", "n=5\n4.2%", "Reduce position size by\nλ·u (uncertainty penalty)",
         4.2, 3.5, COLORS["reduce"], COLORS["reduce_border"]),
        ("FORCE_HOLD", "n=0\n0%", "Override to HOLD\n(high vacuity ≥ 0.50)",
         7.0, 3.5, COLORS["force"], COLORS["force_border"]),
        ("HUMAN_REVIEW", "n=113\n94.2%", "Flag for human oversight\n(uncertainty threshold)",
         9.8, 3.5, COLORS["review"], COLORS["review_border"]),
        ("STRATEGY_REVIEW", "n=0\n0%", "Flag entire strategy\nfor fundamental review",
         12.6, 3.5, COLORS["strategy"], COLORS["strategy_border"]),
    ]

    for label, count, desc, x, y, color, edge in gate_outputs_data:
        # Box
        draw_box(ax, x, y, 2.4, 1.4, f"{label}\n{count}\n{desc}",
                 color, edge, 8, text_weight="bold")
        # Arrow from diamond
        draw_arrow(ax, diamond_x + (x - 7) * 0.15, diamond_y - 0.4,
                   x, y + 0.7, color="#666", line_width=1.0)

    # Outcomes
    outcomes_data = [
        ("Execute", 1.4, 1.5, COLORS["recommend"]),
        ("Execute\n(reduced)", 4.2, 1.5, COLORS["reduce"]),
        ("No trade", 7.0, 1.5, COLORS["force"]),
        ("Queued for\nhuman review", 9.8, 1.5, COLORS["review"]),
        ("Strategy\nreview", 12.6, 1.5, COLORS["strategy"]),
    ]
    for text, x, y, color in outcomes_data:
        draw_box(ax, x, y, 2.2, 0.6, text, color, "#424242", 9, text_weight="bold")
        draw_arrow(ax, x, 2.75, x, y + 0.3, line_width=1.5)

    # Key insight box
    insight_text = ("Key Finding: EDL gate activates on 98.3% of decisions (RECOMMEND_AS_IS + ALL others),\n"
                    "but only 4.2% trigger size modification (REDUCE_SIZE), and 0% trigger action override (FORCE_HOLD).\n"
                    "Primary mechanism: human review flagging (94.2%) — gate provides oversight signal, not automatic intervention.")
    ax.text(7, 0.4, insight_text, ha="center", va="center", fontsize=9,
            style="italic", bbox=dict(boxstyle="round,pad=0.4",
                                      fc="#fce4ec", ec="#c2185b", linewidth=1.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("Generating Phase B.5 pipeline flow diagrams...")
    plot_full_pipeline_flow("pipeline_flow_diagram.png")
    plot_per_ablation_flow("pipeline_flow_diagram_simplified.png")
    plot_gate_output_flow("gate_output_flow.png")
    print("\nDone! Generated 3 diagrams:")
    print("  1. pipeline_flow_diagram.png            (full architecture)")
    print("  2. pipeline_flow_diagram_simplified.png (per-ablation)")
    print("  3. gate_output_flow.png                 (gate routing)")
