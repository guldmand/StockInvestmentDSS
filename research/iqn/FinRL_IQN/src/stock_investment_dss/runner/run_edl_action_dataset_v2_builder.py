"""
run_edl_action_dataset_v2_builder.py

D-IQN-DSS EDL v3.3 dataset builder runner.

Reads the latest combined IQN + HierarchicalDecisionPolicy audit CSV and
produces train/eval datasets for EDL-A/B/C training.

Environment variables
---------------------
STOCK_INVESTMENT_DSS_EDL_V2_SOURCE_COMBINED_RUN_ID
    If set, use this specific combined run id instead of auto-detecting latest.

STOCK_INVESTMENT_DSS_EDL_LABEL_MODE
    Label mode: 'hindsight' (EDL-A), 'rules' (EDL-B), 'iqn_teacher' (EDL-C).
    Default: 'iqn_teacher'

STOCK_INVESTMENT_DSS_USE_HIERARCHICAL_POLICY
    Logged only (default true). Not ablated here.
STOCK_INVESTMENT_DSS_USE_EDL
    Logged only (default false).
STOCK_INVESTMENT_DSS_EDL_VARIANT
    Logged only (default 'none').
STOCK_INVESTMENT_DSS_EDL_GATE_ENABLED
    Logged only (default false).

Usage
-----
$env:PYTHONPATH = "src"
$env:STOCK_INVESTMENT_DSS_EDL_LABEL_MODE = "iqn_teacher"
python -m stock_investment_dss.runner.run_edl_action_dataset_v2_builder
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _find_runs_dir() -> Path:
    cwd = Path.cwd()
    candidate = cwd / "outputs" / "runs"
    if candidate.is_dir():
        return candidate
    try:
        from stock_investment_dss.utilities.paths import RUNS_DIRECTORY

        return Path(RUNS_DIRECTORY)
    except ImportError:
        pass
    raise FileNotFoundError(
        f"Cannot find outputs/runs directory. Expected: {candidate}"
    )


def _find_latest_combined_run(runs_dir: Path) -> Path:
    """Return the latest combined_iqn_hierarchical_smoke_test run with a valid audit CSV."""
    _audit_rel = "audit/combined_iqn_hierarchical_decision_by_step.csv"
    candidates = sorted(
        [
            d
            for d in runs_dir.iterdir()
            if d.is_dir()
            and "combined_iqn_hierarchical_smoke_test" in d.name
            and (d / _audit_rel).exists()
        ]
    )
    if not candidates:
        empty_dirs = [
            d.name
            for d in runs_dir.iterdir()
            if d.is_dir() and "combined_iqn_hierarchical_smoke_test" in d.name
        ]
        raise FileNotFoundError(
            f"No combined_iqn_hierarchical_smoke_test run with a valid audit CSV "
            f"found in {runs_dir}.\n"
            f"Dirs found (missing audit CSV): {empty_dirs}\n"
            "Run run_combined_iqn_hierarchical_smoke_test first."
        )
    latest = candidates[-1]
    logger.info("Auto-detected combined run: %s", latest.name)
    return latest


def _find_combined_run(runs_dir: Path, run_id: str) -> Path:
    """Find a combined run by partial or full ID."""
    candidates = [d for d in runs_dir.iterdir() if d.is_dir() and run_id in d.name]
    if not candidates:
        raise FileNotFoundError(
            f"No combined run matching '{run_id}' found in {runs_dir}"
        )
    return sorted(candidates)[-1]


def main() -> None:
    # -----------------------------------------------------------------------
    # Read env vars
    # -----------------------------------------------------------------------
    source_run_id_override = os.environ.get(
        "STOCK_INVESTMENT_DSS_EDL_V2_SOURCE_COMBINED_RUN_ID", ""
    ).strip()
    label_mode = (
        os.environ.get("STOCK_INVESTMENT_DSS_EDL_LABEL_MODE", "iqn_teacher")
        .strip()
        .lower()
    )

    use_hierarchical = (
        os.environ.get("STOCK_INVESTMENT_DSS_USE_HIERARCHICAL_POLICY", "true")
        .strip()
        .lower()
    )
    use_edl = os.environ.get("STOCK_INVESTMENT_DSS_USE_EDL", "false").strip().lower()
    edl_variant = os.environ.get("STOCK_INVESTMENT_DSS_EDL_VARIANT", "none").strip()
    edl_gate = (
        os.environ.get("STOCK_INVESTMENT_DSS_EDL_GATE_ENABLED", "false").strip().lower()
    )

    logger.info("=" * 70)
    logger.info("D-IQN-DSS EDL v3.3 Dataset Builder v2")
    logger.info("  label_mode                : %s", label_mode)
    logger.info("  source_run_id_override    : %s", source_run_id_override or "(auto)")
    logger.info("  USE_HIERARCHICAL_POLICY   : %s", use_hierarchical)
    logger.info("  USE_EDL                   : %s", use_edl)
    logger.info("  EDL_VARIANT               : %s", edl_variant)
    logger.info("  EDL_GATE_ENABLED          : %s", edl_gate)
    logger.info("=" * 70)

    if use_edl == "true":
        logger.warning(
            "USE_EDL=true is set but this runner does not perform EDL inference. "
            "This runner builds the dataset. Set USE_EDL=false for dataset building."
        )

    # -----------------------------------------------------------------------
    # Locate source combined run
    # -----------------------------------------------------------------------
    runs_dir = _find_runs_dir()

    if source_run_id_override:
        combined_run_dir = _find_combined_run(runs_dir, source_run_id_override)
    else:
        combined_run_dir = _find_latest_combined_run(runs_dir)

    source_combined_run_id = combined_run_dir.name
    logger.info("Source combined run: %s", source_combined_run_id)

    # -----------------------------------------------------------------------
    # Load combined audit CSV
    # -----------------------------------------------------------------------
    audit_csv_path = (
        combined_run_dir / "audit" / "combined_iqn_hierarchical_decision_by_step.csv"
    )
    if not audit_csv_path.exists():
        files_found = (
            list((combined_run_dir / "audit").glob("*"))
            if (combined_run_dir / "audit").exists()
            else []
        )
        raise FileNotFoundError(
            f"Combined audit CSV not found: {audit_csv_path}\n"
            f"Files in audit/: {[f.name for f in files_found]}"
        )

    combined_df = pd.read_csv(audit_csv_path)
    logger.info(
        "Loaded combined audit: %d rows × %d columns from %s",
        len(combined_df),
        len(combined_df.columns),
        audit_csv_path,
    )

    # -----------------------------------------------------------------------
    # Build dataset
    # -----------------------------------------------------------------------
    from stock_investment_dss.uncertainty.edl_action_dataset_v2 import (
        EDLDatasetBuilderV2,
        build_summary_json,
        build_summary_md,
    )

    builder = EDLDatasetBuilderV2(
        fill_nan_value=0.0,
        exclude_unavailable=True,
    )
    dataset = builder.build(combined_df, label_mode=label_mode)

    logger.info(
        "Dataset built: %d rows, %d features, label_mode=%s",
        dataset.n_total,
        dataset.n_features,
        dataset.label_mode,
    )
    logger.info("Label distribution (full): %s", dataset.label_distribution)
    if dataset.n_unavailable > 0:
        logger.warning(
            "Unavailable labels: %d/%d rows", dataset.n_unavailable, len(combined_df)
        )
    for w in dataset.warnings:
        logger.warning("[Dataset warning] %s", w)

    # -----------------------------------------------------------------------
    # Create output run directory
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_name = f"{timestamp}_d_iqn_dss_edl_action_dataset_v2_builder"
    run_dir = runs_dir / run_name
    data_dir = run_dir / "data"
    summary_dir = run_dir / "summary"
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output run directory: %s", run_dir)

    # -----------------------------------------------------------------------
    # Write CSVs
    # -----------------------------------------------------------------------
    full_csv = data_dir / "edl_v2_dataset_full.csv"
    train_csv = data_dir / "edl_v2_train_dataset.csv"
    eval_csv = data_dir / "edl_v2_eval_dataset.csv"

    dataset.full_df.to_csv(full_csv, index=False)
    dataset.train_df.to_csv(train_csv, index=False)
    dataset.eval_df.to_csv(eval_csv, index=False)

    logger.info(
        "Written: full=%d rows, train=%d rows, eval=%d rows",
        len(dataset.full_df),
        len(dataset.train_df),
        len(dataset.eval_df),
    )

    # -----------------------------------------------------------------------
    # Write summary
    # -----------------------------------------------------------------------
    output_files = {
        "edl_v2_dataset_full": str(full_csv),
        "edl_v2_train_dataset": str(train_csv),
        "edl_v2_eval_dataset": str(eval_csv),
    }
    summary_dict = build_summary_json(dataset, source_combined_run_id, output_files)

    summary_json_path = summary_dir / "edl_v2_dataset_summary.json"
    summary_md_path = summary_dir / "edl_v2_dataset_summary.md"

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, default=str)
    summary_dict["output_files"]["summary_json"] = str(summary_json_path)
    summary_dict["output_files"]["summary_md"] = str(summary_md_path)

    md_text = build_summary_md(summary_dict)
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    logger.info("Summary JSON: %s", summary_json_path)
    logger.info("Summary MD:   %s", summary_md_path)

    # -----------------------------------------------------------------------
    # Preview first 5 rows
    # -----------------------------------------------------------------------
    preview_cols = [
        c
        for c in [
            "decision_id",
            "date",
            "selected_iqn_action",
            "hierarchical_action_type",
            "edl_label_mode",
            "edl_label_name",
            "edl_label_id",
            "iqn_score_hold",
            "iqn_score_buy",
            "iqn_action_margin",
            "cash_weight",
        ]
        if c in dataset.full_df.columns
    ]
    preview = dataset.full_df[preview_cols].head(5)
    logger.info("First 5 rows (key columns):\n%s", preview.to_string(index=False))

    # -----------------------------------------------------------------------
    # Final status
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("EDL v2 dataset build COMPLETE")
    logger.info("  Output dir      : %s", run_dir)
    logger.info("  Total rows      : %d", dataset.n_total)
    logger.info("  Train rows      : %d", dataset.n_train)
    logger.info("  Eval rows       : %d", dataset.n_eval)
    logger.info("  Feature count   : %d", dataset.n_features)
    logger.info("  Label mode      : %s", dataset.label_mode)
    logger.info("  Label dist (full): %s", dataset.label_distribution)
    logger.info("  Unavailable     : %d", dataset.n_unavailable)
    logger.info("  Warnings        : %d", len(dataset.warnings))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
