# src/stock_investment_dss/runner/run_edl_uncertainty_smoke_test.py
"""
Standalone smoke test for the D-IQN-DSS EDL Uncertainty Layer (v3.1 PoC).

This runner does NOT require a trained IQN model or training run.
It reads from an existing hierarchical policy smoke test output and
produces EDL-inspired recommendation confidence estimates.

Usage
-----
From repository root with PYTHONPATH=src:

    $env:PYTHONPATH = "src"
    python -m stock_investment_dss.runner.run_edl_uncertainty_smoke_test

    # Specify a source run explicitly:
    $env:STOCK_INVESTMENT_DSS_EDL_SOURCE_RUN_ID = "2026_05_21_031912_d_iqn_dss_hierarchical_policy_smoke_test"
    python -m stock_investment_dss.runner.run_edl_uncertainty_smoke_test

Environment variables
---------------------
STOCK_INVESTMENT_DSS_EDL_SOURCE_RUN_ID
    Directory name (not full path) of the source hierarchical run.
    If not set, the latest *hierarchical_policy_smoke_test* run is used.
STOCK_INVESTMENT_DSS_EDL_RUNS_BASE
    Base directory for outputs/runs (default: outputs/runs).

Outputs (written to outputs/runs/<timestamp>_d_iqn_dss_edl_uncertainty_smoke_test/)
-----------
audit/edl_uncertainty_by_decision.csv
summary/edl_uncertainty_summary.json
summary/edl_uncertainty_summary.md
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Source run discovery
# ---------------------------------------------------------------------------

_RUNS_BASE_DEFAULT = "outputs/runs"
_HIERARCHICAL_PATTERN = "hierarchical_policy_smoke_test"
_EDL_RUN_SUFFIX = "d_iqn_dss_edl_uncertainty_smoke_test"


def find_source_run(runs_base: Path, source_run_id: str) -> Path:
    """
    Locate the hierarchical policy run to use as input.

    Parameters
    ----------
    runs_base : Path
        Base directory containing run subdirectories.
    source_run_id : str
        If non-empty, look for this exact subdirectory name.
        If empty, use the latest hierarchical_policy_smoke_test run.
    """
    if not runs_base.exists():
        raise FileNotFoundError(
            f"Runs base directory not found: {runs_base.resolve()}\n"
            "Please run the hierarchical policy smoke test first:\n"
            "  $env:PYTHONPATH='src'\n"
            "  python -m stock_investment_dss.runner.run_hierarchical_policy_smoke_test"
        )

    if source_run_id:
        candidate = runs_base / source_run_id
        if not candidate.exists():
            raise FileNotFoundError(
                f"Specified source run not found: {candidate.resolve()}\n"
                f"STOCK_INVESTMENT_DSS_EDL_SOURCE_RUN_ID='{source_run_id}'"
            )
        return candidate

    # Auto-discover: find the most recent hierarchical_policy_smoke_test directory
    candidates = sorted(
        [
            d
            for d in runs_base.iterdir()
            if d.is_dir() and _HIERARCHICAL_PATTERN in d.name
        ],
        key=lambda d: d.name,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No hierarchical_policy_smoke_test runs found under: {runs_base.resolve()}\n"
            "Please run the hierarchical policy smoke test first:\n"
            "  $env:PYTHONPATH='src'\n"
            "  python -m stock_investment_dss.runner.run_hierarchical_policy_smoke_test"
        )

    chosen = candidates[-1]  # latest by timestamp-prefixed name
    logger.info("Auto-selected source run: %s", chosen.name)
    return chosen


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_hierarchical_audit(
    source_run: Path,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load the three audit CSVs from a hierarchical policy run.

    Returns
    -------
    (decisions_df, ticker_df or None, size_df or None)
    """
    decision_path = source_run / "audit" / "hierarchical_decision_by_step.csv"
    ticker_path = source_run / "audit" / "ticker_score_table.csv"
    size_path = source_run / "audit" / "size_score_table.csv"

    if not decision_path.exists():
        raise FileNotFoundError(
            f"Decision audit CSV not found: {decision_path}\n"
            "The source run may be incomplete."
        )

    decisions_df = pd.read_csv(decision_path, dtype=str)
    logger.info(
        "Loaded %d decision rows from %s", len(decisions_df), decision_path.name
    )

    ticker_df: Optional[pd.DataFrame] = None
    if ticker_path.exists():
        ticker_df = pd.read_csv(ticker_path, dtype=str)
        logger.info(
            "Loaded %d ticker score rows from %s", len(ticker_df), ticker_path.name
        )
    else:
        logger.info("No ticker_score_table.csv found (HOLD run — expected)")

    size_df: Optional[pd.DataFrame] = None
    if size_path.exists():
        size_df = pd.read_csv(size_path, dtype=str)
        logger.info("Loaded %d size score rows from %s", len(size_df), size_path.name)
    else:
        logger.info("No size_score_table.csv found (HOLD run — expected)")

    return decisions_df, ticker_df, size_df


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_decisions(
    decisions_df: pd.DataFrame,
    ticker_df: Optional[pd.DataFrame],
    size_df: Optional[pd.DataFrame],
    source_run_name: str,
) -> list[dict]:
    """
    Process all decision rows through the EDL uncertainty layer.

    Returns
    -------
    List of audit row dicts, one per decision step.
    """
    from stock_investment_dss.uncertainty.edl_classifier import (
        EDLClassifier,
        build_feature_vector,
    )
    from stock_investment_dss.uncertainty.recommendation_confidence import (
        RecommendationConfidenceEvaluator,
    )

    classifier = EDLClassifier()
    evaluator = RecommendationConfidenceEvaluator()
    rows: list[dict] = []

    for _, decision_row in decisions_df.iterrows():
        decision_id = str(decision_row.get("decision_id", ""))
        date = str(decision_row.get("date", ""))
        action_type = str(decision_row.get("selected_action_type", "HOLD"))
        selected_ticker = str(decision_row.get("selected_ticker", "") or "")
        selected_size = str(decision_row.get("selected_size", "") or "")
        visible_cutoff = str(decision_row.get("visible_data_cutoff", date))

        # Filter matching rows from lookup tables
        t_rows = None
        s_rows = None
        if ticker_df is not None:
            t_rows = ticker_df[ticker_df["decision_id"].astype(str) == decision_id]
        if size_df is not None:
            s_rows = size_df[size_df["decision_id"].astype(str) == decision_id]

        # Build feature vector
        fv = build_feature_vector(
            decision_row=decision_row,
            ticker_rows=t_rows,
            size_rows=s_rows,
            action_type=action_type,
        )

        # Run EDL classifier
        dirichlet = classifier.classify(fv)

        # Evaluate confidence
        confidence = evaluator.evaluate(dirichlet, fv)

        # Assemble audit row
        audit_row = {
            # Identity
            "decision_id": decision_id,
            "date": date,
            "visible_data_cutoff": visible_cutoff,
            "selected_action_type": action_type,
            "selected_ticker": selected_ticker,
            "selected_size": selected_size,
            "source_run": source_run_name,
            # Primary confidence outputs
            "confidence_score": confidence.confidence_score,
            "uncertainty_score": confidence.uncertainty_score,
            "evidence_total": confidence.evidence_total,
            "recommendation_confidence_label": confidence.recommendation_confidence_label,
            "uncertainty_warning": confidence.uncertainty_warning,
            "should_require_human_review": confidence.should_require_human_review,
            "evidence_for_recommendation": confidence.evidence_for_recommendation,
            "evidence_against_recommendation": confidence.evidence_against_recommendation,
            # Dirichlet internals
            "evidence_high": dirichlet.evidence_high,
            "evidence_medium": dirichlet.evidence_medium,
            "evidence_low": dirichlet.evidence_low,
            "alpha_high": dirichlet.alpha_high,
            "alpha_medium": dirichlet.alpha_medium,
            "alpha_low": dirichlet.alpha_low,
            "dirichlet_strength": dirichlet.dirichlet_strength,
            "prob_high": dirichlet.prob_high,
            "prob_medium": dirichlet.prob_medium,
            "prob_low": dirichlet.prob_low,
            "vacuity": dirichlet.vacuity,
            # Input features (for traceability)
            "feat_action_score_margin": fv.action_score_margin,
            "feat_final_ticker_score": fv.final_ticker_score,
            "feat_score_variance": fv.score_variance,
            "feat_value_score": fv.value_score,
            "feat_quality_score": fv.quality_score,
            "feat_profitability_score": fv.profitability_score,
            "feat_momentum_score": fv.momentum_score,
            "feat_risk_fit_score": fv.risk_fit_score,
            "feat_q50": fv.q50,
            "feat_q_spread": fv.q_spread,
            "feat_cvar": fv.cvar,
            "feat_risk_adj_fraction": fv.risk_adj_fraction,
            "feat_size_reduction_ratio": fv.size_reduction_ratio,
            "feat_cash_weight": fv.cash_weight,
            "feat_max_concentration": fv.max_concentration,
            "feat_drawdown_norm": fv.drawdown_norm,
            "feat_price_vs_ma50_norm": fv.price_vs_ma50_norm,
            "feat_price_vs_ma200_norm": fv.price_vs_ma200_norm,
            # Metadata
            "iqn_features_available": fv.iqn_features_available,
            "edl_model_version": confidence.edl_model_version,
            "label_strategy": confidence.label_strategy,
            "source": confidence.source,
        }

        rows.append(audit_row)

        logger.info(
            "  [%s] %s → %s | confidence=%.3f | vacuity=%.3f | review=%s",
            date,
            action_type,
            confidence.recommendation_confidence_label,
            confidence.confidence_score,
            confidence.uncertainty_score,
            confidence.should_require_human_review,
        )

    return rows


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def build_json_summary(
    rows: list[dict],
    source_run_name: str,
    output_dir: Path,
) -> dict:
    n = len(rows)
    label_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    review_count = 0
    total_confidence = 0.0
    total_uncertainty = 0.0
    total_evidence = 0.0

    per_decision = []
    for r in rows:
        lbl = r.get("recommendation_confidence_label", "MEDIUM")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        if r.get("should_require_human_review"):
            review_count += 1
        total_confidence += float(r.get("confidence_score", 0))
        total_uncertainty += float(r.get("uncertainty_score", 0))
        total_evidence += float(r.get("evidence_total", 0))
        per_decision.append(
            {
                "decision_id": r.get("decision_id"),
                "date": r.get("date"),
                "selected_action_type": r.get("selected_action_type"),
                "selected_ticker": r.get("selected_ticker"),
                "recommendation_confidence_label": lbl,
                "confidence_score": round(float(r.get("confidence_score", 0)), 4),
                "uncertainty_score": round(float(r.get("uncertainty_score", 0)), 4),
                "should_require_human_review": r.get("should_require_human_review"),
                "uncertainty_warning": r.get("uncertainty_warning", ""),
            }
        )

    return {
        "run_id": output_dir.name,
        "source_run_directory": source_run_name,
        "n_decisions": n,
        "edl_model_version": "edl_poc_v3_1_placeholder_rule_based",
        "label_strategy": "placeholder_rule_based",
        "iqn_features_available": False,
        "confidence_distribution": label_counts,
        "human_review_required_count": review_count,
        "mean_confidence_score": round(total_confidence / n, 4) if n else 0.0,
        "mean_uncertainty_score": round(total_uncertainty / n, 4) if n else 0.0,
        "mean_evidence_total": round(total_evidence / n, 4) if n else 0.0,
        "per_decision": per_decision,
    }


def build_md_summary(summary: dict) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dist = summary["confidence_distribution"]
    n = summary["n_decisions"]
    per = summary["per_decision"]

    table_rows = "\n".join(
        f"| {r['date']} | {r['selected_action_type']} | {r.get('selected_ticker') or '—'} "
        f"| **{r['recommendation_confidence_label']}** "
        f"| {r['confidence_score']:.3f} | {r['uncertainty_score']:.3f} "
        f"| {'✅' if not r['should_require_human_review'] else '⚠️ Yes'} "
        f"| {r.get('uncertainty_warning','')[:60] or '—'} |"
        for r in per
    )

    return f"""# EDL Uncertainty Summary — {summary['run_id']}

Generated: {ts}
Source run: `{summary['source_run_directory']}`

## Confidence Distribution

| Label | Count | % |
|-------|-------|---|
| HIGH  | {dist.get('HIGH',0)} | {dist.get('HIGH',0)/n*100:.0f}% |
| MEDIUM| {dist.get('MEDIUM',0)} | {dist.get('MEDIUM',0)/n*100:.0f}% |
| LOW   | {dist.get('LOW',0)} | {dist.get('LOW',0)/n*100:.0f}% |

## Uncertainty Statistics

- Mean confidence score: **{summary['mean_confidence_score']:.4f}**
- Mean epistemic uncertainty (vacuity): **{summary['mean_uncertainty_score']:.4f}**
- Mean evidence total (above prior): **{summary['mean_evidence_total']:.4f}**
- Decisions requiring human review: **{summary['human_review_required_count']} / {n}**

## Per-Decision Results

| Date | Action | Ticker | Label | Confidence | Vacuity | Review? | Warning (truncated) |
|------|--------|--------|-------|------------|---------|---------|---------------------|
{table_rows}

## Notes

- IQN distribution features available: {summary['iqn_features_available']}
- EDL model version: `{summary['edl_model_version']}`
- Label strategy: `{summary['label_strategy']}`
- All outputs marked `source=edl_poc_placeholder`
- **v3.1 is a deterministic rule-based PoC. Full EDL requires training, labels, and calibration.**
- See `docs/EDL_Uncertainty_PoC_v3_1.md` for design rationale and thesis alignment.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _setup_logging()
    logger.info("=" * 60)
    logger.info("D-IQN-DSS EDL Uncertainty Smoke Test (v3.1 PoC)")
    logger.info("=" * 60)

    # ----------------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------------
    runs_base_str = _env("STOCK_INVESTMENT_DSS_EDL_RUNS_BASE", _RUNS_BASE_DEFAULT)
    source_run_id = _env("STOCK_INVESTMENT_DSS_EDL_SOURCE_RUN_ID", "")

    runs_base = Path(runs_base_str)
    logger.info("Runs base: %s", runs_base.resolve())

    # ----------------------------------------------------------------
    # Locate source run
    # ----------------------------------------------------------------
    source_run = find_source_run(runs_base, source_run_id)
    logger.info("Source run: %s", source_run.name)

    # ----------------------------------------------------------------
    # Load hierarchical audit output
    # ----------------------------------------------------------------
    decisions_df, ticker_df, size_df = load_hierarchical_audit(source_run)

    if decisions_df.empty:
        logger.error("Decision CSV is empty — nothing to process.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Create output directory
    # ----------------------------------------------------------------
    ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    out_dir = runs_base / f"{ts}_{_EDL_RUN_SUFFIX}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    # ----------------------------------------------------------------
    # Process decisions through EDL layer
    # ----------------------------------------------------------------
    logger.info(
        "Processing %d decisions through EDL uncertainty layer...", len(decisions_df)
    )
    rows = process_decisions(decisions_df, ticker_df, size_df, source_run.name)

    # ----------------------------------------------------------------
    # Write outputs
    # ----------------------------------------------------------------
    audit_path = out_dir / "audit" / "edl_uncertainty_by_decision.csv"
    _write_csv(audit_path, pd.DataFrame(rows))
    logger.info("Written: %s", audit_path)

    summary = build_json_summary(rows, source_run.name, out_dir)

    json_path = out_dir / "summary" / "edl_uncertainty_summary.json"
    _write_json(json_path, summary)
    logger.info("Written: %s", json_path)

    md_path = out_dir / "summary" / "edl_uncertainty_summary.md"
    _write_md(md_path, build_md_summary(summary))
    logger.info("Written: %s", md_path)

    # ----------------------------------------------------------------
    # Final report
    # ----------------------------------------------------------------
    dist = summary["confidence_distribution"]
    logger.info("-" * 60)
    logger.info("EDL Uncertainty Smoke Test Complete")
    logger.info("  Source run : %s", source_run.name)
    logger.info("  Decisions  : %d", summary["n_decisions"])
    logger.info("  HIGH       : %d", dist.get("HIGH", 0))
    logger.info("  MEDIUM     : %d", dist.get("MEDIUM", 0))
    logger.info("  LOW        : %d", dist.get("LOW", 0))
    logger.info(
        "  Human review required: %d / %d",
        summary["human_review_required_count"],
        summary["n_decisions"],
    )
    logger.info("  Mean confidence : %.4f", summary["mean_confidence_score"])
    logger.info("  Mean vacuity    : %.4f", summary["mean_uncertainty_score"])
    logger.info("  Output dir      : %s", out_dir)
    logger.info("-" * 60)
    logger.info("NO TRAINING WAS RUN.")
    logger.info("All outputs marked source=edl_poc_placeholder.")


if __name__ == "__main__":
    main()
