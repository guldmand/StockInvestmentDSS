"""Cross-experiment comparison for the IQN HOLD-collapse diagnostic.

Reads the per-experiment artefacts copied into
``copilot-diagnostics/results/experiment_<x>/`` by each experiment script and
builds a single ``comparison_report.md`` plus a flat ``comparison_table.csv``.

This script is read-only with respect to the rest of the project. It only
writes inside ``copilot-diagnostics/results/``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Columns of interest from iqn_reward_action_diagnostic_by_seed.csv
INTERESTING_COLS = [
    "seed",
    "seed_status",
    "loss_final",
    "final_total_return_pct",
    "final_total_trades",
    "hold_score_mean",
    "buy_score_mean",
    "hold_minus_buy_score",
    "hold_q50_mean",
    "buy_q50_mean",
    "buy_reward_mean",
    "hold_invested_reward_mean",
    "cash_only_proxy_share",
    "training_sell_count",
    "training_buy_count",
    "epsilon_final",
]


def read_by_seed_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_effective_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_experiment_dirs() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        p
        for p in RESULTS_DIR.iterdir()
        if p.is_dir() and p.name.startswith("experiment_")
    )


def format_value(val: str | None) -> str:
    if val is None or val == "":
        return "—"
    try:
        f = float(val)
        if f != f:  # NaN
            return "—"
        if abs(f) >= 1000:
            return f"{f:,.0f}"
        if abs(f) >= 1:
            return f"{f:.3f}"
        return f"{f:.4f}"
    except (TypeError, ValueError):
        return str(val)


def collect() -> dict[str, dict[str, dict[str, str]]]:
    """Return {experiment_name: {seed: {col: value}}}."""
    data: dict[str, dict[str, dict[str, str]]] = {}
    for exp_dir in find_experiment_dirs():
        rows = read_by_seed_csv(exp_dir / "iqn_reward_action_diagnostic_by_seed.csv")
        seed_map: dict[str, dict[str, str]] = {}
        for row in rows:
            seed = str(row.get("seed", "")).strip()
            if seed:
                seed_map[seed] = row
        data[exp_dir.name] = seed_map
    return data


def build_markdown(data: dict[str, dict[str, dict[str, str]]]) -> str:
    if not data:
        return "# IQN HOLD-Collapse — Comparison Report\n\nNo experiment results found in `results/`.\n"

    experiments = list(data.keys())
    all_seeds = sorted(
        {s for seed_map in data.values() for s in seed_map.keys()},
        key=lambda x: int(x) if x.isdigit() else x,
    )

    out: list[str] = []
    out.append("# IQN HOLD-Collapse — Cross-Experiment Comparison Report")
    out.append("")
    out.append(
        "Generated from `copilot-diagnostics/results/experiment_*/iqn_reward_action_diagnostic_by_seed.csv`."
    )
    out.append("")
    out.append(
        "Higher `hold_share` (close to 1.0) and positive `hold_score_minus_buy_score` indicate HOLD-collapse."
    )
    out.append("Experiments that reduce these values isolate the responsible factor.")
    out.append("")

    # Effective config table
    out.append("## Effective configuration per experiment")
    out.append("")
    out.append("| Experiment | Description |")
    out.append("|------------|-------------|")
    for exp in experiments:
        cfg = read_effective_config(RESULTS_DIR / exp / "effective_config.json")
        desc = cfg.get("description", "")
        out.append(f"| `{exp}` | {desc} |")
    out.append("")

    # One table per metric
    metrics = [
        ("seed_status", "Seed status (active_trading / no_trade)"),
        ("loss_final", "Final training loss (lower = stable)"),
        ("final_total_return_pct", "Final total return %"),
        ("final_total_trades", "Number of trades"),
        ("hold_minus_buy_score", "q50(HOLD) - q50(BUY) (positive = HOLD preferred)"),
        ("hold_q50_mean", "q50 of HOLD action"),
        ("buy_q50_mean", "q50 of BUY action"),
        ("buy_reward_mean", "Mean reward when action=BUY"),
        ("cash_only_proxy_share", "Cash-only share during training"),
        ("training_sell_count", "SELL action count during training"),
        ("epsilon_final", "Final epsilon"),
    ]

    for col, label in metrics:
        out.append(f"## {label}")
        out.append("")
        header = "| Seed | " + " | ".join(experiments) + " |"
        sep = "|------|" + "|".join(["------"] * len(experiments)) + "|"
        out.append(header)
        out.append(sep)
        for seed in all_seeds:
            row_vals = []
            for exp in experiments:
                seed_map = data[exp]
                val = seed_map.get(seed, {}).get(col)
                row_vals.append(format_value(val))
            out.append(f"| {seed} | " + " | ".join(row_vals) + " |")
        out.append("")

    # Interpretation hints
    out.append("## How to read this")
    out.append("")
    out.append(
        "- **loss_final > 1,000,000** for a seed → Q-value divergence / gradient explosion (primary driver of HOLD-collapse)."
    )
    out.append(
        "- **hold_minus_buy_score >> 0** for a seed → HOLD dominates BUY in Q-space, agent never buys."
    )
    out.append(
        "- **training_sell_count > 12,000** (>48% of 25k steps) → diverged policy over-sells during training, draining holdings, leaving agent cash-only at backtest start."
    )
    out.append(
        "- If `loss_final` drops sharply in **experiment_e_lower_grad_clip** → gradient explosion was the root cause; max_norm fix resolves it."
    )
    out.append(
        "- If `loss_final` stays high in experiment_e → need reward normalization or lower LR."
    )
    out.append("")
    return "\n".join(out)


def write_flat_csv(data: dict[str, dict[str, dict[str, str]]], path: Path) -> None:
    cols = ["experiment", *INTERESTING_COLS]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for exp, seed_map in data.items():
            for seed, row in seed_map.items():
                writer.writerow([exp] + [row.get(c, "") for c in INTERESTING_COLS])


def main() -> int:
    data = collect()
    md_path = RESULTS_DIR / "comparison_report.md"
    csv_path = RESULTS_DIR / "comparison_table.csv"
    md = build_markdown(data)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    write_flat_csv(data, csv_path)
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
