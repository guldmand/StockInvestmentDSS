#!/usr/bin/env python3
"""
run_edl_action_inference_smoke_test.py  (EDL v3.2)

Loads a trained EDL action classifier (or uses placeholder if none available),
runs inference on the eval dataset or the latest hierarchical policy run,
applies the EDL gate, and writes full audit CSV + summary.

Usage
-----
    python -m stock_investment_dss.runner.run_edl_action_inference_smoke_test

Environment variables
---------------------
    STOCK_INVESTMENT_DSS_EDL_MODEL_PATH     path to .pt checkpoint (default: auto-discover)
    STOCK_INVESTMENT_DSS_EDL_*              all standard EDL env vars supported
    EDL_EVAL_CSV                            path to eval CSV (default: auto-discover)
    EDL_HIER_AUDIT_CSV                      path to hierarchical decision CSV (optional)

Output
------
    outputs/runs/<timestamp>_d_iqn_dss_edl_action_inference_smoke_test/
        audit/edl_action_uncertainty_by_decision.csv
        summary/edl_action_inference_summary.json
        summary/edl_action_inference_summary.md
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure src is on path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stock_investment_dss.uncertainty.edl_action_classes import EDLActionConfig
from stock_investment_dss.uncertainty.edl_action_classifier import EDLActionClassifier
from stock_investment_dss.uncertainty.edl_ensemble import EDLEnsemble
from stock_investment_dss.uncertainty.edl_gate import EDLGate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("edl_action_inference_smoke_test")


def _get_env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _find_latest(glob_pattern: str, sub: str) -> Optional[Path]:
    """Find the most recently created path matching glob + sub path."""
    runs = _REPO_ROOT / "outputs" / "runs"
    if not runs.exists():
        return None
    candidates = sorted(runs.glob(glob_pattern), reverse=True)
    for c in candidates:
        p = c / sub
        if p.exists():
            return p
    return None


def _find_eval_csv() -> Optional[Path]:
    return _find_latest(
        "*_d_iqn_dss_edl_action_dataset_builder", "data/edl_action_eval_dataset.csv"
    )


def _find_model_path() -> Optional[Path]:
    return _find_latest(
        "*_d_iqn_dss_edl_action_training_smoke_test", "models/edl_action_classifier.pt"
    )


def _find_hier_audit_csv() -> Optional[Path]:
    return _find_latest(
        "*_d_iqn_dss_hierarchical_policy_smoke_test",
        "audit/hierarchical_decision_by_step.csv",
    )


def main() -> None:
    import numpy as np
    import pandas as pd

    from stock_investment_dss.uncertainty.edl_action_dataset import EDLActionDataset

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    edl_config = EDLActionConfig.from_env()
    action_classes = edl_config.action_classes
    K = edl_config.num_classes

    # ------------------------------------------------------------------
    # Discover input files
    # ------------------------------------------------------------------
    eval_csv_env = _get_env("EDL_EVAL_CSV", "")
    eval_csv_path = Path(eval_csv_env) if eval_csv_env else _find_eval_csv()

    hier_audit_env = _get_env("EDL_HIER_AUDIT_CSV", "")
    hier_audit_path = Path(hier_audit_env) if hier_audit_env else _find_hier_audit_csv()

    model_env = _get_env("STOCK_INVESTMENT_DSS_EDL_MODEL_PATH", "")
    model_path = Path(model_env) if model_env else _find_model_path()

    logger.info("=== EDL Action Inference Smoke Test v3.2 ===")
    logger.info("model_path: %s", model_path or "(placeholder — no trained model)")
    logger.info("eval_csv:   %s", eval_csv_path or "(none)")
    logger.info("hier_audit: %s", hier_audit_path or "(none)")

    # ------------------------------------------------------------------
    # Load classifier / ensemble
    # ------------------------------------------------------------------
    edl_config_with_path = EDLActionConfig(
        include_change_strategy=edl_config.include_change_strategy,
        edl_variant=edl_config.edl_variant,
        gate_enabled=edl_config.gate_enabled,
        uncertainty_lambda=edl_config.uncertainty_lambda,
        disagreement_lambda=edl_config.disagreement_lambda,
        horizon_days=edl_config.horizon_days,
        label_mode=edl_config.label_mode,
        model_path=str(model_path) if model_path else "",
        use_hierarchical_policy=edl_config.use_hierarchical_policy,
        use_edl=edl_config.use_edl,
    )
    ensemble = EDLEnsemble.from_config(
        config=edl_config_with_path,
        model_paths=(
            {"A": str(model_path), "B": str(model_path), "C": str(model_path)}
            if model_path
            else None
        ),
    )
    gate = EDLGate.from_edl_config(edl_config)

    # ------------------------------------------------------------------
    # Load eval dataset
    # ------------------------------------------------------------------
    if eval_csv_path and eval_csv_path.exists():
        logger.info("Loading eval dataset: %s", eval_csv_path)
        eval_df = pd.read_csv(eval_csv_path)
    elif hier_audit_path and hier_audit_path.exists():
        logger.info("Using hierarchical audit CSV for inference: %s", hier_audit_path)
        eval_df = pd.read_csv(hier_audit_path)
    else:
        logger.warning(
            "No eval dataset or hierarchical audit found. Using synthetic data..."
        )
        N = 20
        input_dim = K * 5
        X_synth = np.random.randn(N, input_dim).astype(np.float32)
        feature_cols = [f"feat_f{i}" for i in range(input_dim)]
        eval_df = pd.DataFrame(X_synth, columns=feature_cols)
        eval_df["date"] = "2024-01-01"
        eval_df["ticker"] = "AAPL"
        eval_df["selected_action_type"] = "HOLD"
        eval_df["risk_adjusted_allocation_fraction"] = 0.25
        eval_df["label_str"] = "HOLD"

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    feature_cols = [c for c in eval_df.columns if c.startswith("feat_")]
    if not feature_cols:
        logger.warning(
            "No feature columns found in eval data. Cannot run meaningful inference."
        )
        feature_cols = []

    audit_rows = []
    for i, row in eval_df.iterrows():
        date_str = str(row.get("date", ""))
        ticker = str(row.get("ticker", ""))
        selected_action = str(row.get("selected_action_type", "HOLD")).upper()
        original_fraction = float(
            row.get("risk_adjusted_allocation_fraction", 0.25) or 0.25
        )
        selected_size = str(row.get("selected_size", ""))

        # Feature vector
        if feature_cols:
            feat = row[feature_cols].values.astype(np.float32)
        else:
            feat = np.zeros(K * 5, dtype=np.float32)

        # Ensemble inference
        ens_result = ensemble.classify(feat, selected_action=selected_action)

        # Pick representative result (first available variant, or overall ensemble)
        members = list(ens_result.individual.keys())
        if members:
            rep = ens_result.individual[members[0]]
        else:
            # create a minimal result from ensemble
            from stock_investment_dss.uncertainty.edl_action_classifier import (
                EDLActionResult,
            )

            rep = EDLActionResult(
                action_classes=action_classes,
                evidence_by_action={a: 0.0 for a in action_classes},
                alpha_by_action={a: 1.0 for a in action_classes},
                probability_by_action=ens_result.p_ensemble,
                dirichlet_strength=float(K),
                uncertainty_vacuity=ens_result.u_ensemble,
                predicted_action=ens_result.ensemble_predicted_action,
                selected_action=selected_action,
                selected_action_probability=ens_result.ensemble_selected_probability,
                selected_action_evidence=0.0,
                edl_agrees_with_selected_action=ens_result.ensemble_agrees_with_selected,
            )

        u = rep.uncertainty_vacuity
        lambda_u = edl_config.uncertainty_lambda
        lambda_d = edl_config.disagreement_lambda
        disagree = ens_result.model_disagreement_score

        uncertainty_penalty = lambda_u * u
        disagreement_penalty = lambda_d * disagree

        # Gate
        p_rebalance = rep.probability_by_action.get("REBALANCE", 0.0)
        p_cs = rep.probability_by_action.get("CHANGE_STRATEGY", 0.0)

        if edl_config.gate_enabled:
            gate_result = gate.apply(
                selected_action=selected_action,
                selected_size=selected_size,
                original_fraction=original_fraction,
                vacuity=u,
                edl_agrees=rep.edl_agrees_with_selected_action,
                edl_predicted_action=rep.predicted_action,
                p_rebalance=p_rebalance,
                p_change_strategy=p_cs,
                disagreement_score=disagree,
                uncertainty_penalty=uncertainty_penalty,
                disagreement_penalty=disagreement_penalty,
            )
        else:
            gate_result = gate.null_gate(
                selected_action, selected_size, original_fraction
            )

        # Assemble full audit record
        audit_row: dict = {
            "decision_id": f"{date_str}_{ticker}_{i}",
            "date": date_str,
            "selected_action_type": selected_action,
            "selected_ticker": ticker,
            "selected_size": selected_size,
            "edl_enabled": edl_config.use_edl,
            "edl_variant": edl_config.edl_variant,
            "hierarchical_policy_enabled": edl_config.use_hierarchical_policy,
            "action_classes": "|".join(action_classes),
        }
        audit_row.update(rep.to_audit_dict())
        audit_row.update(ens_result.to_audit_dict())
        audit_row.update(gate_result.to_audit_dict())
        audit_row["uncertainty_penalty"] = round(uncertainty_penalty, 6)
        audit_row["disagreement_penalty"] = round(disagreement_penalty, 6)
        audit_rows.append(audit_row)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    ts = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
    run_name = f"{ts}_d_iqn_dss_edl_action_inference_smoke_test"
    out_dir = _REPO_ROOT / "outputs" / "runs" / run_name
    audit_dir = out_dir / "audit"
    summ_dir = out_dir / "summary"
    audit_dir.mkdir(parents=True, exist_ok=True)
    summ_dir.mkdir(parents=True, exist_ok=True)

    # Audit CSV
    audit_csv_path = audit_dir / "edl_action_uncertainty_by_decision.csv"
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(audit_csv_path, index=False)
    logger.info("Wrote audit CSV: %s  (%d rows)", audit_csv_path, len(audit_df))

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    gate_counts: dict = {}
    if "recommendation_gate" in audit_df.columns:
        gate_counts = audit_df["recommendation_gate"].value_counts().to_dict()

    mean_vacuity = (
        float(audit_df["uncertainty_vacuity"].mean())
        if "uncertainty_vacuity" in audit_df.columns
        else 0.0
    )
    agree_rate = (
        float(audit_df["edl_agrees_with_selected_action"].mean())
        if "edl_agrees_with_selected_action" in audit_df.columns
        else 0.0
    )

    summary = {
        "inference_version": "3.2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "edl_variant": edl_config.edl_variant,
        "gate_enabled": edl_config.gate_enabled,
        "model_path": str(model_path) if model_path else "",
        "is_placeholder": not bool(model_path and Path(str(model_path)).exists()),
        "action_classes": action_classes,
        "num_decisions": len(audit_rows),
        "mean_vacuity": round(mean_vacuity, 4),
        "edl_agree_rate": round(agree_rate, 4),
        "gate_distribution": gate_counts,
        "audit_csv": str(audit_csv_path),
    }

    json_path = summ_dir / "edl_action_inference_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    md_path = summ_dir / "edl_action_inference_summary.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# EDL Action Inference Smoke Test (v3.2)\n\n")
        fh.write(f"**Generated:** {summary['timestamp_utc']}\n\n")
        fh.write(
            f"**Model:** `{summary['model_path'] or 'placeholder (uniform Dirichlet)'}`\n\n"
        )
        fh.write(
            f"**Variant:** `{edl_config.edl_variant}`  "
            f"**Gate enabled:** `{edl_config.gate_enabled}`\n\n"
        )
        fh.write(f"**Decisions evaluated:** {len(audit_rows)}\n\n")
        fh.write(
            f"**Mean vacuity:** {mean_vacuity:.4f}  "
            f"**EDL agreement rate:** {agree_rate:.3f}\n\n"
        )
        fh.write("## Gate distribution\n\n")
        for g, cnt in gate_counts.items():
            fh.write(f"- `{g}`: {cnt}\n")
    logger.info("Wrote summary JSON: %s", json_path)
    logger.info("Wrote summary MD: %s", md_path)
    logger.info("=== Inference smoke test complete. ===")


if __name__ == "__main__":
    main()
