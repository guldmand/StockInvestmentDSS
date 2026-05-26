"""Build strategy comparison report from the latest Etape 5 dashboard output.

Auto-discovers the most recent d_iqn_dss_summary_dashboard run under
outputs/runs/, reads its data/strategies_combined.csv, and calls
build_comparison_report() to produce a multi-tier strategy ranking report.

Usage
-----
    python scripts/build_comparison_report.py

Expected run name pattern (substring match against outputs/runs/):
    d_iqn_dss_summary_dashboard -> data/strategies_combined.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from stock_investment_dss.algorithmic_trading.experiments.compare_algorithmic_results import (
    build_comparison_report,
    find_metric_files,
)
from stock_investment_dss.visualization.summary_dashboard import print_section

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
    print_section("Comparison Report — auto-discovering input runs")

    dashboard_run = _find_latest_run(_RUNS_DIR, "d_iqn_dss_summary_dashboard")
    print(f"Dashboard run: {dashboard_run.name}")

    try:
        files = find_metric_files(dashboard_run)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    strategies_csv = files[0]
    print(f"Input CSV    : {strategies_csv}")

    output_md = build_comparison_report(strategies_combined_csv=strategies_csv)

    print_section("Done")
    print(f"Report: {output_md}")


if __name__ == "__main__":
    main()
