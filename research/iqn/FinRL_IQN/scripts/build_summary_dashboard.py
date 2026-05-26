"""Build summary dashboard from pre-computed multiseed aggregates.

Auto-discovers the latest algorithmic baseline grid run, FinRL multiseed
summary run, and IQN learning curve multiseed summary run under
outputs/runs/, then calls build_summary_dashboard() to produce a
4-panel strategy comparison figure.

Usage
-----
    python scripts/build_summary_dashboard.py

Expected run name patterns (substring match against outputs/runs/):
    algorithmic_baseline_grid        -> summary/algorithmic_baselines_summary.csv
    finrl_baseline_multiseed_summary -> summary/finrl_baseline_multiseed_aggregate_by_agent.csv
    iqn_learning_curve_multiseed_summary
        -> summary/iqn_learning_curve_multiseed_final_records.csv
        -> data/iqn_learning_curve_multiseed_eval_records.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package root is on sys.path when invoked from the project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

from stock_investment_dss.visualization.summary_dashboard import (
    build_summary_dashboard,
    print_section,
)

_RUNS_DIR = _PROJECT_ROOT / "outputs" / "runs"

# ---------------------------------------------------------------------------
# Auto-discovery helper
# ---------------------------------------------------------------------------


def _find_latest_run(runs_dir: Path, name_pattern: str) -> Path:
    """Return the most recent run directory whose name contains name_pattern.

    Directories are sorted lexicographically; because run directory names
    start with YYYY_MM_DD_HHMMSS, lexicographic order equals chronological
    order.

    Raises FileNotFoundError if no matching directory is found.
    """
    matches = [d for d in runs_dir.iterdir() if d.is_dir() and name_pattern in d.name]
    if not matches:
        raise FileNotFoundError(
            f"No run directory matching pattern {name_pattern!r} found in {runs_dir}"
        )
    return sorted(matches)[-1]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print_section("Summary Dashboard — auto-discovering input runs")

    algo_run = _find_latest_run(_RUNS_DIR, "algorithmic_baseline_grid")
    finrl_run = _find_latest_run(_RUNS_DIR, "finrl_baseline_multiseed_summary")
    iqn_run = _find_latest_run(_RUNS_DIR, "iqn_learning_curve_multiseed_summary")

    print(f"Algorithmic run : {algo_run.name}")
    print(f"FinRL run       : {finrl_run.name}")
    print(f"IQN run         : {iqn_run.name}")

    algo_csv = algo_run / "summary" / "algorithmic_baselines_summary.csv"
    finrl_csv = (
        finrl_run / "summary" / "finrl_baseline_multiseed_aggregate_by_agent.csv"
    )
    iqn_metrics_csv = (
        iqn_run / "summary" / "iqn_learning_curve_multiseed_final_records.csv"
    )
    iqn_eval_csv = iqn_run / "data" / "iqn_learning_curve_multiseed_eval_records.csv"

    missing = [
        (label, path)
        for label, path in [
            ("algorithmic CSV", algo_csv),
            ("FinRL aggregate CSV", finrl_csv),
            ("IQN metrics CSV", iqn_metrics_csv),
            ("IQN eval records CSV", iqn_eval_csv),
        ]
        if not path.exists()
    ]
    if missing:
        for label, path in missing:
            print(f"ERROR: Required {label} not found: {path}", file=sys.stderr)
        sys.exit(1)

    output_png = build_summary_dashboard(
        algorithmic_summary_csv=algo_csv,
        finrl_aggregate_csv=finrl_csv,
        iqn_metrics_csv=iqn_metrics_csv,
        iqn_eval_records_csv=iqn_eval_csv,
    )

    print_section("Done")
    print(f"Output: {output_png}")


if __name__ == "__main__":
    main()
