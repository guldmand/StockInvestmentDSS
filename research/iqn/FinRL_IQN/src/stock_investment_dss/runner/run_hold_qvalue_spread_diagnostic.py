"""Post-run diagnostic: Q-value spread per eval step.

Reads the existing iqn_learning_curve_eval_distributions.csv produced by a
run_iqn_learning_curve_smoke_test run and computes the Q-value spread
(best_score - second_best_score) per (train_step, eval_step).

Usage:
    python -m stock_investment_dss.runner.run_hold_qvalue_spread_diagnostic \\
        --run-dir outputs/runs/<run_id>_d_iqn_dss_iqn_learning_curve_smoke_test

Output:
    <run_dir>/data/hold_qvalue_spread_diagnostic.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

EVAL_DIST_FILENAME = "iqn_learning_curve_eval_distributions.csv"
OUTPUT_FILENAME = "hold_qvalue_spread_diagnostic.csv"


def compute_spread(run_dir: Path) -> int:
    dist_path = run_dir / "data" / EVAL_DIST_FILENAME
    if not dist_path.exists():
        print(f"ERROR: eval distributions file not found: {dist_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(dist_path)

    score_col = "score"
    if score_col not in df.columns:
        print(
            f"ERROR: column '{score_col}' not found in {dist_path}.\n"
            f"Available columns: {list(df.columns)}",
            file=sys.stderr,
        )
        return 1

    # Determine grouping columns (train_step + eval_step or step index)
    group_cols: list[str] = []
    for candidate in ["train_step", "eval_step", "step"]:
        if candidate in df.columns:
            group_cols.append(candidate)

    if not group_cols:
        print(
            "ERROR: no recognisable step column found. "
            f"Available columns: {list(df.columns)}",
            file=sys.stderr,
        )
        return 1

    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    records = []
    for group_key, group in df.groupby(group_cols, dropna=False):
        scores = group[score_col].dropna().sort_values(ascending=False).tolist()
        best = scores[0] if len(scores) >= 1 else float("nan")
        second_best = scores[1] if len(scores) >= 2 else float("nan")
        spread = (
            best - second_best
            if (len(scores) >= 2 and not pd.isna(best) and not pd.isna(second_best))
            else float("nan")
        )

        row: dict = {}
        if isinstance(group_key, tuple):
            for col, val in zip(group_cols, group_key):
                row[col] = val
        else:
            row[group_cols[0]] = group_key

        row["best_score"] = best
        row["second_best_score"] = second_best
        row["spread"] = spread
        row["n_actions"] = len(scores)

        # Record which action had the best score (if action column present)
        if "action" in group.columns:
            best_mask = group[score_col] == best
            best_actions = group.loc[best_mask, "action"].dropna().tolist()
            row["best_action"] = best_actions[0] if best_actions else None
        records.append(row)

    result = pd.DataFrame(records)

    if "train_step" in result.columns:
        result = result.sort_values("train_step").reset_index(drop=True)

    output_path = run_dir / "data" / OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Written: {output_path}")

    # Print summary statistics
    spread_col = result["spread"].dropna()
    print(f"\nQ-value spread summary ({len(spread_col)} rows with valid spread):")
    if not spread_col.empty:
        print(f"  mean  : {spread_col.mean():.6f}")
        print(f"  std   : {spread_col.std():.6f}")
        print(f"  min   : {spread_col.min():.6f}")
        print(f"  max   : {spread_col.max():.6f}")
        print(f"  median: {spread_col.median():.6f}")

    # Per-training-phase trend if train_step available
    if "train_step" in result.columns and not spread_col.empty:
        print("\nSpread mean by training phase (train_step):")
        phase_stats = (
            result.groupby("train_step")["spread"]
            .mean()
            .reset_index()
            .sort_values("train_step")
        )
        for _, phase_row in phase_stats.iterrows():
            print(
                f"  train_step={int(phase_row['train_step']):>7}  mean_spread={phase_row['spread']:.6f}"
            )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Post-run Q-value spread diagnostic for IQN learning-curve runs. "
            "Reads eval_distributions.csv and writes hold_qvalue_spread_diagnostic.csv."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the run directory produced by run_iqn_learning_curve_smoke_test.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        return 1

    return compute_spread(run_dir)


if __name__ == "__main__":
    sys.exit(main())
