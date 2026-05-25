"""
run_edl_action_gate_end_to_end_smoke_test.py  (EDL v3.3)

End-to-end EDL gate smoke test.

Reads the combined IQN + HierarchicalDecisionPolicy audit CSV, feeds each
decision row through a trained EDL-C checkpoint, applies the vacuity-based
gate logic, and writes a final gated recommendation for every step.

IMPORTANT CAVEAT
----------------
EDL-C uses iqn_teacher labels. Because the teacher label is derived from
argmax(iqn_score_*) and those same scores are in the feature matrix, the
classifier learns teacher imitation, not market correctness.  This gate
demonstrates pipeline connectivity and uncertainty-flagging; it does NOT
prove that gated recommendations are more profitable.  EDL-A hindsight
labels are required for correctness/performance validation.

Usage
-----
    $env:PYTHONPATH = "src"
    $env:STOCK_INVESTMENT_DSS_EDL_GATE_COMBINED_RUN_ID = "<combined_run_id>"
    $env:STOCK_INVESTMENT_DSS_EDL_MODEL_RUN_ID          = "<training_run_id>"
    python -m stock_investment_dss.runner.run_edl_action_gate_end_to_end_smoke_test

Environment variables
---------------------
    STOCK_INVESTMENT_DSS_EDL_GATE_COMBINED_RUN_ID
        Combined IQN+HDP smoke-test run ID (partial match OK).
        Default: auto-discover latest valid *combined_iqn_hierarchical_smoke_test run.

    STOCK_INVESTMENT_DSS_EDL_MODEL_RUN_ID
        EDL training run ID whose checkpoint to load (partial match OK).
        Default: auto-discover latest valid *edl_action_training_v2_smoke_test run.

    STOCK_INVESTMENT_DSS_EDL_GATE_ENABLED               true/false  (default: true)
    STOCK_INVESTMENT_DSS_EDL_UNCERTAINTY_THRESHOLD       float       (default: 0.50)
    STOCK_INVESTMENT_DSS_EDL_HIGH_UNCERTAINTY_THRESHOLD  float       (default: 0.70)
    STOCK_INVESTMENT_DSS_EDL_MIN_RECOMMENDATION_PROB     float       (default: 0.50)
    STOCK_INVESTMENT_DSS_EDL_DISAGREEMENT_ACTION         str         (default: HUMAN_REVIEW)

Gate decision labels
--------------------
    PASS_THROUGH                 — gate disabled
    FORCE_HOLD_HIGH_UNCERTAINTY  — vacuity >= high_uncertainty_threshold
    HUMAN_REVIEW_UNCERTAIN       — vacuity >= uncertainty_threshold
    HUMAN_REVIEW_DISAGREEMENT    — EDL predicted action differs from recommendation
    HUMAN_REVIEW_LOW_PROBABILITY — recommended-action probability < min_prob
    RECOMMEND_AS_IS              — confident agreement

Output
------
    outputs/runs/<timestamp>_d_iqn_dss_edl_action_gate_end_to_end_smoke_test/
        audit/edl_gate_decision_by_step.csv
        summary/edl_gate_summary.json
        summary/edl_gate_summary.md
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_CLASSES = ["HOLD", "BUY", "SELL", "REBALANCE"]

_COMBINED_AUDIT_FILENAME = "audit/combined_iqn_hierarchical_decision_by_step.csv"
_CHECKPOINT_FILENAME = "models/edl_action_classifier_v2.pt"
_RUNS_DIR = Path("outputs/runs")

# Gate decision labels
PASS_THROUGH = "PASS_THROUGH"
FORCE_HOLD_HIGH_UNCERTAINTY = "FORCE_HOLD_HIGH_UNCERTAINTY"
HUMAN_REVIEW_UNCERTAIN = "HUMAN_REVIEW_UNCERTAIN"
HUMAN_REVIEW_DISAGREEMENT = "HUMAN_REVIEW_DISAGREEMENT"
HUMAN_REVIEW_LOW_PROBABILITY = "HUMAN_REVIEW_LOW_PROBABILITY"
RECOMMEND_AS_IS = "RECOMMEND_AS_IS"

EDL_CAVEAT = (
    "EDL-C is teacher-imitation confidence gating, not proof of market correctness. "
    "EDL-A hindsight labels are required for performance validation."
)

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _bool_env(key: str, default: bool) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    return default


def _float_env(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)).strip())
    except (ValueError, TypeError):
        return default


def _str_env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def _find_latest_combined_run() -> Path:
    candidates = sorted(
        [
            d
            for d in _RUNS_DIR.iterdir()
            if d.is_dir()
            and "combined_iqn_hierarchical_smoke_test" in d.name
            and (d / _COMBINED_AUDIT_FILENAME).exists()
        ],
        key=lambda d: d.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No valid combined_iqn_hierarchical_smoke_test run found in {_RUNS_DIR}. "
            "Run the combined IQN+HDP smoke test first."
        )
    return candidates[0]


def _find_combined_run(run_id: str) -> Path:
    candidates = [
        d
        for d in _RUNS_DIR.iterdir()
        if d.is_dir() and run_id in d.name and (d / _COMBINED_AUDIT_FILENAME).exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Combined run '{run_id}' not found or has no audit CSV in {_RUNS_DIR}."
        )
    return sorted(candidates, key=lambda d: d.name, reverse=True)[0]


def _find_latest_model_run() -> Path:
    candidates = sorted(
        [
            d
            for d in _RUNS_DIR.iterdir()
            if d.is_dir()
            and "edl_action_training_v2_smoke_test" in d.name
            and (d / _CHECKPOINT_FILENAME).exists()
        ],
        key=lambda d: d.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No valid edl_action_training_v2_smoke_test run found in {_RUNS_DIR}. "
            "Run the EDL training smoke test first."
        )
    return candidates[0]


def _find_model_run(run_id: str) -> Path:
    candidates = [
        d
        for d in _RUNS_DIR.iterdir()
        if d.is_dir() and run_id in d.name and (d / _CHECKPOINT_FILENAME).exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Model run '{run_id}' not found or has no checkpoint in {_RUNS_DIR}."
        )
    return sorted(candidates, key=lambda d: d.name, reverse=True)[0]


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------


def _load_checkpoint(ckpt_path: Path) -> dict:
    try:
        import torch  # type: ignore
    except ImportError:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    required = {
        "model_state_dict",
        "input_dim",
        "num_classes",
        "feature_columns",
        "feature_mean",
        "feature_std",
    }
    missing = required - set(ckpt.keys())
    if missing:
        raise KeyError(f"Checkpoint missing required keys: {missing}")
    return ckpt


def _build_network(ckpt: dict):
    from stock_investment_dss.uncertainty.edl_action_network import EDLActionNetwork
    import torch  # type: ignore

    net = EDLActionNetwork(
        input_dim=ckpt["input_dim"],
        num_classes=ckpt["num_classes"],
        hidden_dims=ckpt.get("hidden_dims", [128, 64]),
        evidence_activation=ckpt.get("evidence_activation", "softplus"),
    )
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Feature vector builder
# ---------------------------------------------------------------------------


def _build_feature_vector(row: dict, feature_columns: List[str]) -> List[float]:
    """
    Build a feature vector from a combined audit CSV row.

    Columns present in the combined audit CSV that are in feature_columns are
    used directly.  Missing values (NaN, empty string) are filled with 0.0.
    """
    vec = []
    for col in feature_columns:
        raw = row.get(col, "")
        if raw is None or raw == "" or raw != raw:  # None, empty, or NaN
            vec.append(0.0)
        else:
            try:
                vec.append(float(raw))
            except (ValueError, TypeError):
                vec.append(0.0)
    return vec


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------


def _apply_gate(
    recommendation: str,
    edl_predicted: str,
    vacuity: float,
    recommended_prob: float,
    gate_enabled: bool,
    uncertainty_threshold: float,
    high_uncertainty_threshold: float,
    min_recommendation_prob: float,
) -> Tuple[str, str]:
    """
    Apply EDL gate logic.

    Returns
    -------
    (final_action_after_edl, gate_decision)
    """
    if not gate_enabled:
        return recommendation, PASS_THROUGH

    if vacuity >= high_uncertainty_threshold:
        return "HOLD", FORCE_HOLD_HIGH_UNCERTAINTY

    if vacuity >= uncertainty_threshold:
        return recommendation, HUMAN_REVIEW_UNCERTAIN

    if edl_predicted != recommendation:
        return recommendation, HUMAN_REVIEW_DISAGREEMENT

    if recommended_prob < min_recommendation_prob:
        return recommendation, HUMAN_REVIEW_LOW_PROBABILITY

    return recommendation, RECOMMEND_AS_IS


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run() -> None:
    import csv

    try:
        import torch  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        sys.exit(1)

    # -- Resolve source runs ------------------------------------------------
    combined_run_id = _str_env("STOCK_INVESTMENT_DSS_EDL_GATE_COMBINED_RUN_ID", "")
    model_run_id = _str_env("STOCK_INVESTMENT_DSS_EDL_MODEL_RUN_ID", "")

    if combined_run_id:
        combined_run_dir = _find_combined_run(combined_run_id)
    else:
        combined_run_dir = _find_latest_combined_run()
    logger.info("Combined run: %s", combined_run_dir.name)

    if model_run_id:
        model_run_dir = _find_model_run(model_run_id)
    else:
        model_run_dir = _find_latest_model_run()
    logger.info("Model run: %s", model_run_dir.name)

    # -- Gate configuration -------------------------------------------------
    gate_enabled = _bool_env("STOCK_INVESTMENT_DSS_EDL_GATE_ENABLED", True)
    uncertainty_threshold = _float_env(
        "STOCK_INVESTMENT_DSS_EDL_UNCERTAINTY_THRESHOLD", 0.50
    )
    high_uncertainty_threshold = _float_env(
        "STOCK_INVESTMENT_DSS_EDL_HIGH_UNCERTAINTY_THRESHOLD", 0.70
    )
    min_recommendation_prob = _float_env(
        "STOCK_INVESTMENT_DSS_EDL_MIN_RECOMMENDATION_PROB", 0.50
    )

    logger.info(
        "Gate: enabled=%s, unc_thr=%.2f, high_unc_thr=%.2f, min_prob=%.2f",
        gate_enabled,
        uncertainty_threshold,
        high_uncertainty_threshold,
        min_recommendation_prob,
    )

    # -- Load checkpoint ----------------------------------------------------
    ckpt_path = model_run_dir / _CHECKPOINT_FILENAME
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = _load_checkpoint(ckpt_path)

    feature_columns: List[str] = ckpt["feature_columns"]
    action_classes: List[str] = ckpt.get("action_classes", ACTION_CLASSES)
    num_classes = ckpt["num_classes"]

    # Convert feature_mean/std to numpy for efficient vectorised normalisation
    feat_mean = np.array(ckpt["feature_mean"], dtype=np.float32)
    feat_std = np.array(ckpt["feature_std"], dtype=np.float32)
    # Avoid division by zero
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)

    net = _build_network(ckpt)
    logger.info(
        "Network: input_dim=%d, num_classes=%d, hidden_dims=%s",
        ckpt["input_dim"],
        num_classes,
        ckpt.get("hidden_dims", [128, 64]),
    )

    # -- Load combined audit CSV --------------------------------------------
    combined_csv = combined_run_dir / _COMBINED_AUDIT_FILENAME
    logger.info("Reading combined audit: %s", combined_csv)

    with open(combined_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.info("Loaded %d rows", len(rows))
    if not rows:
        logger.error("Combined audit CSV is empty.")
        sys.exit(1)

    # Check feature availability
    available_cols = set(rows[0].keys())
    missing_feat_cols = [c for c in feature_columns if c not in available_cols]
    if missing_feat_cols:
        logger.warning(
            "%d feature column(s) not in combined CSV (will be filled 0.0): %s",
            len(missing_feat_cols),
            missing_feat_cols,
        )

    # -- Run inference on each row -----------------------------------------
    audit_records = []
    vacuities = []

    for row in rows:
        feat_vec = _build_feature_vector(row, feature_columns)
        x = np.array(feat_vec, dtype=np.float32)
        x = (x - feat_mean) / feat_std

        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, D)
        out = net.predict(x_t)

        prob_arr = out["prob"].squeeze(0).numpy()  # (K,)
        evidence_arr = out["evidence"].squeeze(0).numpy()  # (K,)
        vacuity_val = float(out["vacuity"].squeeze())

        pred_idx = int(prob_arr.argmax())
        edl_predicted_action = action_classes[pred_idx]

        # Build prob/evidence dicts keyed by class name
        prob_by_class = {
            action_classes[i]: float(prob_arr[i]) for i in range(num_classes)
        }
        evidence_by_class = {
            action_classes[i]: float(evidence_arr[i]) for i in range(num_classes)
        }

        recommendation = (
            str(row.get("final_recommendation_before_edl", "HOLD")).strip().upper()
        )
        # Normalise recommendation to canonical class name
        if recommendation not in action_classes:
            recommendation = "HOLD"

        rec_prob = prob_by_class.get(recommendation, 0.0)
        disagrees = edl_predicted_action != recommendation

        final_action, gate_decision = _apply_gate(
            recommendation=recommendation,
            edl_predicted=edl_predicted_action,
            vacuity=vacuity_val,
            recommended_prob=rec_prob,
            gate_enabled=gate_enabled,
            uncertainty_threshold=uncertainty_threshold,
            high_uncertainty_threshold=high_uncertainty_threshold,
            min_recommendation_prob=min_recommendation_prob,
        )

        vacuities.append(vacuity_val)

        record = {
            "decision_id": row.get("decision_id", ""),
            "date": row.get("date", ""),
            "selected_iqn_action": row.get("selected_iqn_action", ""),
            "hierarchical_action_type": row.get("hierarchical_action_type", ""),
            "final_recommendation_before_edl": recommendation,
            "edl_predicted_action": edl_predicted_action,
            "edl_prob_hold": round(prob_by_class.get("HOLD", 0.0), 6),
            "edl_prob_buy": round(prob_by_class.get("BUY", 0.0), 6),
            "edl_prob_sell": round(prob_by_class.get("SELL", 0.0), 6),
            "edl_prob_rebalance": round(prob_by_class.get("REBALANCE", 0.0), 6),
            "edl_evidence_hold": round(evidence_by_class.get("HOLD", 0.0), 6),
            "edl_evidence_buy": round(evidence_by_class.get("BUY", 0.0), 6),
            "edl_evidence_sell": round(evidence_by_class.get("SELL", 0.0), 6),
            "edl_evidence_rebalance": round(evidence_by_class.get("REBALANCE", 0.0), 6),
            "edl_vacuity": round(vacuity_val, 6),
            "edl_recommended_action_probability": round(rec_prob, 6),
            "edl_disagrees_with_recommendation": disagrees,
            "edl_gate_decision": gate_decision,
            "final_action_after_edl": final_action,
            "selected_ticker": row.get("selected_ticker", ""),
            "selected_size": row.get("selected_size", ""),
            "selected_size_fraction": row.get("selected_size_fraction", ""),
            "ticker_score": row.get("ticker_score", ""),
            "size_score": row.get("size_score", ""),
            "source_combined_run_id": combined_run_dir.name,
            "source_edl_model_run_id": model_run_dir.name,
        }
        audit_records.append(record)

    # -- Compute summary stats ---------------------------------------------
    before_counts = Counter(r["final_recommendation_before_edl"] for r in audit_records)
    edl_pred_counts = Counter(r["edl_predicted_action"] for r in audit_records)
    after_counts = Counter(r["final_action_after_edl"] for r in audit_records)
    gate_counts = Counter(r["edl_gate_decision"] for r in audit_records)

    n = len(audit_records)
    disagree_count = sum(
        1 for r in audit_records if r["edl_disagrees_with_recommendation"]
    )
    disagreement_rate = disagree_count / n if n > 0 else 0.0
    force_hold_count = gate_counts.get(FORCE_HOLD_HIGH_UNCERTAINTY, 0)
    human_review_count = sum(
        gate_counts.get(k, 0)
        for k in (
            HUMAN_REVIEW_UNCERTAIN,
            HUMAN_REVIEW_DISAGREEMENT,
            HUMAN_REVIEW_LOW_PROBABILITY,
        )
    )
    recommend_as_is_count = gate_counts.get(RECOMMEND_AS_IS, 0)
    pass_through_count = gate_counts.get(PASS_THROUGH, 0)

    mean_vacuity = float(np.mean(vacuities)) if vacuities else 0.0
    min_vacuity = float(np.min(vacuities)) if vacuities else 0.0
    max_vacuity = float(np.max(vacuities)) if vacuities else 0.0

    # -- Create output directory -------------------------------------------
    ts = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
    run_dir = _RUNS_DIR / f"{ts}_d_iqn_dss_edl_action_gate_end_to_end_smoke_test"
    (run_dir / "audit").mkdir(parents=True, exist_ok=True)
    (run_dir / "summary").mkdir(parents=True, exist_ok=True)

    # -- Write audit CSV ---------------------------------------------------
    audit_csv_path = run_dir / "audit/edl_gate_decision_by_step.csv"
    fieldnames = list(audit_records[0].keys())
    with open(audit_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_records)
    logger.info("Wrote audit CSV: %s (%d rows)", audit_csv_path, n)

    # -- Write summary JSON ------------------------------------------------
    summary = {
        "source_combined_run_id": combined_run_dir.name,
        "source_edl_model_run_id": model_run_dir.name,
        "num_rows": n,
        "gate_enabled": gate_enabled,
        "uncertainty_threshold": uncertainty_threshold,
        "high_uncertainty_threshold": high_uncertainty_threshold,
        "min_recommendation_prob": min_recommendation_prob,
        "action_counts_before_edl": dict(before_counts),
        "edl_predicted_action_counts": dict(edl_pred_counts),
        "action_counts_after_edl": dict(after_counts),
        "gate_decision_counts": dict(gate_counts),
        "mean_vacuity": round(mean_vacuity, 6),
        "min_vacuity": round(min_vacuity, 6),
        "max_vacuity": round(max_vacuity, 6),
        "disagreement_count": disagree_count,
        "disagreement_rate": round(disagreement_rate, 4),
        "force_hold_count": force_hold_count,
        "human_review_count": human_review_count,
        "recommend_as_is_count": recommend_as_is_count,
        "pass_through_count": pass_through_count,
        "edl_caveat": EDL_CAVEAT,
        "run_id": run_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    summary_json_path = run_dir / "summary/edl_gate_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary JSON: %s", summary_json_path)

    # -- Write summary MD --------------------------------------------------
    def _count_table(counter: Counter, label: str) -> str:
        lines = [f"| {label} | Count |", "|---|---|"]
        for k, v in sorted(counter.items()):
            lines.append(f"| {k} | {v} |")
        return "\n".join(lines)

    md = f"""# EDL v3.3 Gate End-to-End Smoke Test Summary

**Run ID:** `{run_dir.name}`  
**Generated:** {datetime.now(timezone.utc).isoformat()}

---

## Sources

| Field | Value |
|---|---|
| Combined IQN+HDP run | `{combined_run_dir.name}` |
| EDL model run | `{model_run_dir.name}` |
| Total rows | {n} |

---

## Gate Configuration

| Parameter | Value |
|---|---|
| Gate enabled | {gate_enabled} |
| Uncertainty threshold (HUMAN_REVIEW) | {uncertainty_threshold} |
| High uncertainty threshold (FORCE_HOLD) | {high_uncertainty_threshold} |
| Min recommendation probability | {min_recommendation_prob} |

---

## Action Counts Before EDL

{_count_table(before_counts, "Action")}

---

## EDL Predicted Action Counts

{_count_table(edl_pred_counts, "EDL Predicted")}

---

## Action Counts After EDL Gate

{_count_table(after_counts, "Action")}

---

## Gate Decision Counts

{_count_table(gate_counts, "Gate Decision")}

---

## Uncertainty Statistics

| Metric | Value |
|---|---|
| Mean vacuity | {mean_vacuity:.4f} |
| Min vacuity | {min_vacuity:.4f} |
| Max vacuity | {max_vacuity:.4f} |

---

## Gate Outcomes

| Metric | Value |
|---|---|
| Disagreement rate (EDL vs recommendation) | {disagreement_rate:.1%} ({disagree_count}/{n}) |
| Force-HOLD (high uncertainty) | {force_hold_count} |
| Human review (all reasons) | {human_review_count} |
| Recommend as-is | {recommend_as_is_count} |
| Pass-through (gate disabled) | {pass_through_count} |

---

## ⚠️ Caveat

> {EDL_CAVEAT}

---

## Output Files

- `audit/edl_gate_decision_by_step.csv` — per-step gate decisions and EDL outputs
- `summary/edl_gate_summary.json` — machine-readable summary
- `summary/edl_gate_summary.md` — this file
"""

    summary_md_path = run_dir / "summary/edl_gate_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Wrote summary MD: %s", summary_md_path)

    # -- Console report ----------------------------------------------------
    logger.info("=" * 60)
    logger.info("EDL gate smoke test complete")
    logger.info("Output dir : %s", run_dir)
    logger.info("Rows       : %d", n)
    logger.info(
        "Mean vacuity: %.4f  min=%.4f  max=%.4f", mean_vacuity, min_vacuity, max_vacuity
    )
    logger.info(
        "Disagree rate: %.1f%% (%d/%d)", disagreement_rate * 100, disagree_count, n
    )
    logger.info("Gate decisions:")
    for k, v in sorted(gate_counts.items()):
        logger.info("  %-40s %d", k, v)
    logger.info("Actions before EDL: %s", dict(before_counts))
    logger.info("Actions after  EDL: %s", dict(after_counts))
    logger.info("CAVEAT: %s", EDL_CAVEAT)
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
