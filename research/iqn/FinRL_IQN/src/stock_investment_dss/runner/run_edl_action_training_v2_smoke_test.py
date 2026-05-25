"""
run_edl_action_training_v2_smoke_test.py  (EDL v3.3)

Training runner for the EDL action classifier v2.

Reads train/eval datasets from the EDL v2 dataset builder output
(produced by run_edl_action_dataset_v2_builder.py) and trains
EDLActionNetwork on the combined IQN + HierarchicalDecisionPolicy features.

DO NOT run heavy training from this script — it is a smoke test.

Usage
-----
    $env:PYTHONPATH = "src"
    $env:STOCK_INVESTMENT_DSS_EDL_V2_DATASET_RUN_ID = "<run_id>"
    python -m stock_investment_dss.runner.run_edl_action_training_v2_smoke_test

Environment variables
---------------------
    STOCK_INVESTMENT_DSS_EDL_V2_DATASET_RUN_ID
        Specific dataset run ID to use (partial match OK).
        Default: auto-discover latest *edl_action_dataset_v2_builder run.

    STOCK_INVESTMENT_DSS_EDL_USE_CLASS_WEIGHTS
        true/false — enable inverse-frequency class weighting.
        Default: true.

    EDL_TRAIN_EPOCHS    number of epochs   (default: 10)
    EDL_BATCH_SIZE      batch size         (default: 32)
    EDL_LR              learning rate      (default: 0.001)

Output
------
    outputs/runs/<timestamp>_d_iqn_dss_edl_action_training_v2_smoke_test/
        models/edl_action_classifier_v2.pt
        audit/edl_v2_eval_predictions.csv
        audit/edl_v2_confusion_matrix.csv
        summary/edl_v2_training_summary.json
        summary/edl_v2_training_summary.md
"""

from __future__ import annotations

import json
import logging
import os
import sys
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

_METADATA_COLS = {
    "decision_id",
    "date",
    "source_iqn_run_id",
    "selected_iqn_action",
    "hierarchical_action_type",
    "selected_ticker",
    "selected_size",
    "final_recommendation_before_edl",
    "edl_label_mode",
    "edl_label_name",
    "edl_label_id",
    "visible_data_cutoff",
    "eval_step",
    "pit_split_id",
}

ACTION_CLASSES = ["HOLD", "BUY", "SELL", "REBALANCE"]
NUM_CLASSES = 4

HIDDEN_DIMS = [128, 64]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


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


def _find_latest_v2_dataset_run(runs_dir: Path) -> Path:
    """Find the latest edl_action_dataset_v2_builder run that has the train CSV."""
    _train_rel = "data/edl_v2_train_dataset.csv"
    candidates = sorted(
        [
            d
            for d in runs_dir.iterdir()
            if d.is_dir()
            and "edl_action_dataset_v2_builder" in d.name
            and (d / _train_rel).exists()
        ]
    )
    if not candidates:
        empty = [
            d.name
            for d in runs_dir.iterdir()
            if d.is_dir() and "edl_action_dataset_v2_builder" in d.name
        ]
        raise FileNotFoundError(
            f"No edl_action_dataset_v2_builder run with a valid train CSV "
            f"found in {runs_dir}.\n"
            f"Dirs found (but missing train CSV): {empty}\n"
            "Run run_edl_action_dataset_v2_builder first."
        )
    latest = candidates[-1]
    logger.info("Auto-detected dataset run: %s", latest.name)
    return latest


def _find_dataset_run_by_id(runs_dir: Path, run_id: str) -> Path:
    candidates = [d for d in runs_dir.iterdir() if d.is_dir() and run_id in d.name]
    if not candidates:
        raise FileNotFoundError(
            f"No dataset run matching '{run_id}' found in {runs_dir}"
        )
    return sorted(candidates)[-1]


def _infer_feature_cols(df, label_col: str = "edl_label_id") -> List[str]:
    """Infer numeric feature columns by excluding metadata and label columns."""
    import numpy as np
    import pandas as pd

    exclude = _METADATA_COLS | {label_col}
    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _to_scalar(value, default: float = 0.0) -> float:
    import numpy as np

    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # -----------------------------------------------------------------------
    # Read env vars
    # -----------------------------------------------------------------------
    dataset_run_id_override = _env("STOCK_INVESTMENT_DSS_EDL_V2_DATASET_RUN_ID", "")
    use_class_weights = _env(
        "STOCK_INVESTMENT_DSS_EDL_USE_CLASS_WEIGHTS", "true"
    ).lower() in ("1", "true", "yes")
    epochs = int(_env("EDL_TRAIN_EPOCHS", "10"))
    batch_size = int(_env("EDL_BATCH_SIZE", "32"))
    lr = float(_env("EDL_LR", "0.001"))

    logger.info("=" * 70)
    logger.info("D-IQN-DSS EDL v3.3 Training Runner v2 (smoke test)")
    logger.info("  dataset_run_id    : %s", dataset_run_id_override or "(auto)")
    logger.info("  use_class_weights : %s", use_class_weights)
    logger.info("  epochs            : %d", epochs)
    logger.info("  batch_size        : %d", batch_size)
    logger.info("  lr                : %g", lr)
    logger.info("=" * 70)

    # -----------------------------------------------------------------------
    # PyTorch check
    # -----------------------------------------------------------------------
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        logger.info("PyTorch available: %s", torch.__version__)
    except ImportError:
        logger.error("PyTorch is NOT available. Install: https://pytorch.org/")
        sys.exit(1)

    import numpy as np
    import pandas as pd

    from stock_investment_dss.uncertainty.edl_action_network import EDLActionNetwork
    from stock_investment_dss.uncertainty.edl_losses import edl_action_loss

    # -----------------------------------------------------------------------
    # Locate and load dataset
    # -----------------------------------------------------------------------
    runs_dir = _find_runs_dir()
    if dataset_run_id_override:
        dataset_run_dir = _find_dataset_run_by_id(runs_dir, dataset_run_id_override)
    else:
        dataset_run_dir = _find_latest_v2_dataset_run(runs_dir)

    source_dataset_run_id = dataset_run_dir.name
    logger.info("Source dataset run: %s", source_dataset_run_id)

    train_csv = dataset_run_dir / "data" / "edl_v2_train_dataset.csv"
    eval_csv = dataset_run_dir / "data" / "edl_v2_eval_dataset.csv"
    summary_json_path = dataset_run_dir / "summary" / "edl_v2_dataset_summary.json"

    train_df = pd.read_csv(train_csv)
    eval_df = pd.read_csv(eval_csv) if eval_csv.exists() else train_df.head(50)
    logger.info(
        "Dataset: train=%d rows × %d cols, eval=%d rows × %d cols",
        len(train_df),
        len(train_df.columns),
        len(eval_df),
        len(eval_df.columns),
    )

    # -----------------------------------------------------------------------
    # Load summary JSON for feature columns and label mode
    # -----------------------------------------------------------------------
    label_mode = "iqn_teacher"  # default
    feature_cols_from_summary: Optional[List[str]] = None

    if summary_json_path.exists():
        with open(summary_json_path, encoding="utf-8") as f:
            ds_summary = json.load(f)
        label_mode = ds_summary.get("label_mode", label_mode)
        fc = ds_summary.get("feature_columns")
        if fc and isinstance(fc, list):
            feature_cols_from_summary = fc
            logger.info(
                "Feature columns from summary JSON: %d cols",
                len(feature_cols_from_summary),
            )
        else:
            logger.warning("No feature_columns in summary JSON — will infer.")
    else:
        logger.warning(
            "Summary JSON not found: %s — inferring feature cols.", summary_json_path
        )
        ds_summary = {}

    # -----------------------------------------------------------------------
    # Resolve feature columns
    # -----------------------------------------------------------------------
    if feature_cols_from_summary:
        # Use only cols that actually exist in train_df
        feature_cols = [c for c in feature_cols_from_summary if c in train_df.columns]
        missing = [c for c in feature_cols_from_summary if c not in train_df.columns]
        if missing:
            logger.warning(
                "Feature cols from summary not found in train CSV (%d): %s",
                len(missing),
                missing,
            )
    else:
        feature_cols = _infer_feature_cols(train_df)
        logger.info("Inferred %d numeric feature columns", len(feature_cols))

    if not feature_cols:
        logger.error("No feature columns found. Cannot train.")
        sys.exit(1)

    input_dim = len(feature_cols)
    logger.info("Feature count: %d", input_dim)
    logger.info("Label mode: %s", label_mode)

    # -----------------------------------------------------------------------
    # Extract X, y
    # -----------------------------------------------------------------------
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_eval = eval_df[feature_cols].values.astype(np.float32)

    if "edl_label_id" not in train_df.columns:
        logger.error("Column 'edl_label_id' not found in train dataset.")
        sys.exit(1)

    y_train = train_df["edl_label_id"].values.astype(np.int64)
    y_eval = (
        eval_df["edl_label_id"].values.astype(np.int64)
        if "edl_label_id" in eval_df.columns
        else np.zeros(len(eval_df), dtype=np.int64)
    )

    # Filter out any rows with label_id < 0 (unavailable labels)
    train_valid = y_train >= 0
    eval_valid = y_eval >= 0
    if not train_valid.all():
        n_drop = int((~train_valid).sum())
        logger.warning("Dropping %d train rows with unavailable label (id=-1)", n_drop)
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
    if not eval_valid.all():
        n_drop = int((~eval_valid).sum())
        logger.warning("Dropping %d eval rows with unavailable label (id=-1)", n_drop)
        X_eval = X_eval[eval_valid]
        y_eval = y_eval[eval_valid]

    logger.info(
        "After filtering: train=%d, eval=%d",
        len(X_train),
        len(X_eval),
    )

    # Sanity check: remaining NaN in features
    n_nan_train = int(np.isnan(X_train).sum())
    n_nan_eval = int(np.isnan(X_eval).sum())
    if n_nan_train > 0:
        logger.warning(
            "NaN in train features: %d cells — replacing with 0.0", n_nan_train
        )
        X_train = np.nan_to_num(X_train, nan=0.0)
    if n_nan_eval > 0:
        logger.warning(
            "NaN in eval features: %d cells — replacing with 0.0", n_nan_eval
        )
        X_eval = np.nan_to_num(X_eval, nan=0.0)

    # -----------------------------------------------------------------------
    # Label distributions
    # -----------------------------------------------------------------------
    K = NUM_CLASSES
    train_counts = np.bincount(y_train, minlength=K).astype(np.int64)
    eval_counts = np.bincount(y_eval, minlength=K).astype(np.int64)
    train_label_dist = {ACTION_CLASSES[i]: int(train_counts[i]) for i in range(K)}
    eval_label_dist = {ACTION_CLASSES[i]: int(eval_counts[i]) for i in range(K)}
    logger.info("Train label distribution: %s", train_label_dist)
    logger.info("Eval  label distribution: %s", eval_label_dist)

    # -----------------------------------------------------------------------
    # Feature standardisation — train only (no eval leakage)
    # -----------------------------------------------------------------------
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0  # protect zero-variance features
    X_train = ((X_train - feat_mean) / feat_std).astype(np.float32)
    X_eval = ((X_eval - feat_mean) / feat_std).astype(np.float32)
    logger.info(
        "Feature standardisation applied (train mean/std only, no eval leakage)."
    )

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    model = EDLActionNetwork(
        input_dim=input_dim,
        num_classes=K,
        hidden_dims=HIDDEN_DIMS,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logger.info(
        "Model: EDLActionNetwork(input=%d, K=%d, hidden=%s)", input_dim, K, HIDDEN_DIMS
    )

    # -----------------------------------------------------------------------
    # Class weights
    # -----------------------------------------------------------------------
    sample_weights_torch = None
    class_weights_log: Dict[str, float] = {}
    if use_class_weights:
        inv_freq = 1.0 / np.maximum(train_counts.astype(np.float32), 1.0)
        class_weights_arr = (inv_freq / inv_freq.mean()).astype(np.float32)
        class_weights_log = {
            ACTION_CLASSES[i]: round(float(class_weights_arr[i]), 3) for i in range(K)
        }
        class_weights_torch = torch.tensor(class_weights_arr, dtype=torch.float32)
        sample_weights_torch = class_weights_torch[
            torch.tensor(y_train, dtype=torch.long)
        ]
        logger.info("Class weights (inv-freq, norm mean=1): %s", class_weights_log)
    else:
        logger.info("Class weights disabled.")

    # -----------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    if sample_weights_torch is not None:
        loader = DataLoader(
            TensorDataset(X_t, y_t, sample_weights_torch),
            batch_size=batch_size,
            shuffle=True,
        )
    else:
        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True
        )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    logger.info("Training: %d epochs, batch_size=%d, lr=%g", epochs, batch_size, lr)
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for batch_data in loader:
            if len(batch_data) == 3:
                X_batch, y_batch, w_batch = batch_data
            else:
                X_batch, y_batch = batch_data
                w_batch = None

            optimizer.zero_grad()
            out = model(X_batch)
            alpha = out["alpha"]
            y_onehot = torch.zeros(y_batch.size(0), K, dtype=torch.float32)
            y_onehot.scatter_(1, y_batch.unsqueeze(1), 1.0)

            loss, mse, kl = edl_action_loss(
                alpha,
                y_onehot,
                epoch=epoch,
                total_epochs=epochs,
                kl_lambda=0.1,
            )
            if w_batch is not None:
                batch_loss = (loss * w_batch).mean()
            else:
                batch_loss = loss.mean()

            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": round(avg_loss, 6)})
        logger.info("  epoch %d/%d  loss=%.5f", epoch, epochs, avg_loss)

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    model.eval()
    X_ev = torch.tensor(X_eval, dtype=torch.float32)
    with torch.no_grad():
        eval_out = model(X_ev)
        eval_probs = eval_out["prob"].numpy()
        eval_vacuity = eval_out["vacuity"].squeeze(-1).numpy()
        eval_preds = eval_probs.argmax(axis=1)

    correct = int((eval_preds == y_eval).sum())
    acc = correct / max(len(y_eval), 1)
    mean_vacuity = float(eval_vacuity.mean())

    # Majority baseline
    majority_idx = int(eval_counts.argmax())
    majority_class_name = ACTION_CLASSES[majority_idx]
    majority_count = int(eval_counts[majority_idx])
    majority_baseline_acc = majority_count / max(len(y_eval), 1)
    model_vs_baseline = round(acc - majority_baseline_acc, 4)

    logger.info(
        "Eval accuracy: %.3f | majority baseline: %.3f (%s) | delta: %+.3f",
        acc,
        majority_baseline_acc,
        majority_class_name,
        model_vs_baseline,
    )
    logger.info("Mean vacuity: %.4f", mean_vacuity)

    # Prediction distribution
    pred_counts = np.bincount(eval_preds, minlength=K)
    true_dist = {ACTION_CLASSES[i]: int(eval_counts[i]) for i in range(K)}
    pred_dist = {ACTION_CLASSES[i]: int(pred_counts[i]) for i in range(K)}
    logger.info("True eval dist: %s", true_dist)
    logger.info("Pred eval dist: %s", pred_dist)

    # Per-class metrics
    per_class_metrics = []
    for idx in range(K):
        true_mask = y_eval == idx
        pred_mask = eval_preds == idx
        support = int(true_mask.sum())
        predicted_count = int(pred_mask.sum())
        tp = int((true_mask & pred_mask).sum())
        recall = tp / max(support, 1)
        precision = tp / max(predicted_count, 1)
        per_class_metrics.append(
            {
                "class_idx": idx,
                "class_name": ACTION_CLASSES[idx],
                "support": support,
                "predicted_count": predicted_count,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
            }
        )

    # Confusion matrix
    conf_matrix = []
    for true_idx in range(K):
        row = [
            int(((y_eval == true_idx) & (eval_preds == pred_idx)).sum())
            for pred_idx in range(K)
        ]
        conf_matrix.append(row)

    # Quality warnings
    eval_warnings: List[str] = []
    n_pred_classes = int((pred_counts > 0).sum())
    n_true_classes = int((eval_counts > 0).sum())
    if n_pred_classes < 2:
        eval_warnings.append(
            f"CRITICAL: model predicts only {n_pred_classes} unique class(es) — "
            "severe prediction collapse. Model cannot be used for multi-class decisions."
        )
    elif n_pred_classes < n_true_classes:
        eval_warnings.append(
            f"WARNING: model predicts {n_pred_classes} classes but eval has "
            f"{n_true_classes} true classes — partial prediction collapse."
        )
    if acc <= majority_baseline_acc:
        eval_warnings.append(
            f"WARNING: model accuracy {acc:.3f} ≤ majority-class baseline "
            f"{majority_baseline_acc:.3f}. Model is not better than a naive predictor."
        )

    for w in eval_warnings:
        if w.startswith("CRITICAL"):
            logger.error("QUALITY: %s", w)
        else:
            logger.warning("QUALITY: %s", w)

    # -----------------------------------------------------------------------
    # Output run directory
    # -----------------------------------------------------------------------
    timestamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
    run_name = f"{timestamp}_d_iqn_dss_edl_action_training_v2_smoke_test"
    run_dir = runs_dir / run_name
    models_dir = run_dir / "models"
    audit_dir = run_dir / "audit"
    summary_dir = run_dir / "summary"
    for d in (models_dir, audit_dir, summary_dir):
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Output run directory: %s", run_dir)

    # -----------------------------------------------------------------------
    # Save model checkpoint
    # -----------------------------------------------------------------------
    ckpt_path = models_dir / "edl_action_classifier_v2.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": K,
            "hidden_dims": HIDDEN_DIMS,
            "evidence_activation": "softplus",
            "action_classes": ACTION_CLASSES,
            "feature_columns": feature_cols,
            "feature_mean": feat_mean.tolist(),
            "feature_std": feat_std.tolist(),
            "source_dataset_run_id": source_dataset_run_id,
            "label_mode": label_mode,
            "edl_version": "3.3",
            "epochs_trained": epochs,
            "final_loss": history[-1]["loss"] if history else None,
        },
        ckpt_path,
    )
    logger.info("Saved model checkpoint: %s", ckpt_path)

    # -----------------------------------------------------------------------
    # Save eval predictions CSV
    # -----------------------------------------------------------------------
    eval_meta_cols = [
        c
        for c in [
            "decision_id",
            "date",
            "selected_iqn_action",
            "hierarchical_action_type",
            "edl_label_name",
            "edl_label_id",
        ]
        if c in eval_df.columns
    ]
    eval_records = []
    for i in range(len(y_eval)):
        rec: dict = {}
        # Add metadata from eval_df if available
        if eval_meta_cols:
            valid_indices = eval_df[eval_df["edl_label_id"] >= 0].index
            if i < len(valid_indices):
                src_row = eval_df.loc[valid_indices[i]]
                for col in eval_meta_cols:
                    rec[col] = src_row.get(col, "")
        rec.update(
            {
                "sample_idx": i,
                "true_label_id": int(y_eval[i]),
                "true_label_name": ACTION_CLASSES[int(y_eval[i])],
                "predicted_label_id": int(eval_preds[i]),
                "predicted_label_name": ACTION_CLASSES[int(eval_preds[i])],
                "correct": bool(eval_preds[i] == y_eval[i]),
                "vacuity": round(_to_scalar(eval_vacuity[i]), 6),
            }
        )
        for k, action in enumerate(ACTION_CLASSES):
            rec[f"p_{action.lower()}"] = round(_to_scalar(eval_probs[i, k]), 6)
        eval_records.append(rec)

    preds_csv = audit_dir / "edl_v2_eval_predictions.csv"
    import pandas as pd

    pd.DataFrame(eval_records).to_csv(preds_csv, index=False)
    logger.info("Saved eval predictions: %s", preds_csv)

    # Save confusion matrix
    conf_df = pd.DataFrame(
        conf_matrix,
        index=[f"true_{a}" for a in ACTION_CLASSES],
        columns=[f"pred_{a}" for a in ACTION_CLASSES],
    )
    conf_csv = audit_dir / "edl_v2_confusion_matrix.csv"
    conf_df.to_csv(conf_csv)
    logger.info("Saved confusion matrix: %s", conf_csv)

    # -----------------------------------------------------------------------
    # Summary JSON
    # -----------------------------------------------------------------------
    summary = {
        "smoke_test_version": "3.3",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_run_id": source_dataset_run_id,
        "label_mode": label_mode,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "use_class_weights": use_class_weights,
        "class_weights": class_weights_log,
        "input_dim": input_dim,
        "num_classes": K,
        "hidden_dims": HIDDEN_DIMS,
        "action_classes": ACTION_CLASSES,
        "feature_columns": feature_cols,
        "train_rows": len(X_train),
        "eval_rows": len(X_eval),
        "eval_accuracy": round(acc, 4),
        "majority_class": majority_class_name,
        "majority_class_count": majority_count,
        "majority_baseline_accuracy": round(majority_baseline_acc, 4),
        "model_accuracy_minus_majority_baseline": model_vs_baseline,
        "mean_vacuity": round(mean_vacuity, 4),
        "train_label_distribution": train_label_dist,
        "eval_label_distribution_true": true_dist,
        "eval_label_distribution_predicted": pred_dist,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": {
            "classes": ACTION_CLASSES,
            "matrix": conf_matrix,
        },
        "eval_warnings": eval_warnings,
        "eval_quality_ok": len(eval_warnings) == 0,
        "training_history": history,
        "model_path": str(ckpt_path),
        "eval_predictions_path": str(preds_csv),
        "confusion_matrix_path": str(conf_csv),
    }
    json_path = summary_dir / "edl_v2_training_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved summary JSON: %s", json_path)

    # -----------------------------------------------------------------------
    # Summary Markdown
    # -----------------------------------------------------------------------
    beat_sym = "✅ beats" if model_vs_baseline > 0 else "❌ does NOT beat"
    collapse_block = ""
    if eval_warnings:
        collapse_block = (
            "\n## ⚠️ Quality Warnings\n"
            + "\n".join(f"- {w}" for w in eval_warnings)
            + "\n"
        )

    pc_rows = "\n".join(
        f"| {m['class_name']} | {m['support']} | {m['predicted_count']} "
        f"| {m['recall']:.3f} | {m['precision']:.3f} |"
        for m in per_class_metrics
    )
    cm_header = "| | " + " | ".join(f"pred_{a}" for a in ACTION_CLASSES) + " |"
    cm_sep = "|---|" + "|".join(["---"] * K) + "|"
    cm_rows = "\n".join(
        f"| true_{ACTION_CLASSES[i]} | "
        + " | ".join(str(v) for v in conf_matrix[i])
        + " |"
        for i in range(K)
    )
    hist_rows = "\n".join(f"| {h['epoch']} | {h['loss']:.5f} |" for h in history)

    md_text = f"""# EDL Action Training Smoke Test v2 (v3.3)

**Generated:** {summary['timestamp_utc']}

**Source dataset:** `{source_dataset_run_id}`
**Label mode:** {label_mode}
**Epochs:** {epochs} | **LR:** {lr} | **Batch:** {batch_size} | **Class weights:** `{'yes' if use_class_weights else 'no'}`
**Classes:** `{', '.join(ACTION_CLASSES)}`

## Accuracy

| Metric | Value |
|--------|-------|
| Eval accuracy | {acc:.3f} |
| Majority-class baseline ({majority_class_name}) | {majority_baseline_acc:.3f} |
| Model − baseline | {model_vs_baseline:+.3f} |
| Mean vacuity | {mean_vacuity:.4f} |

**Model {beat_sym} majority-class baseline.**

## Label Distributions

| Class | Train | Eval (true) | Eval (predicted) |
|-------|-------|-------------|-----------------|
{chr(10).join(f"| {ACTION_CLASSES[i]} | {train_label_dist[ACTION_CLASSES[i]]} | {true_dist[ACTION_CLASSES[i]]} | {pred_dist[ACTION_CLASSES[i]]} |" for i in range(K))}

## Per-class Metrics

| Class | Support | Predicted | Recall | Precision |
|-------|---------|-----------|--------|-----------|
{pc_rows}

## Confusion Matrix

{cm_header}
{cm_sep}
{cm_rows}
{collapse_block}
## Training History

| Epoch | Loss |
|-------|------|
{hist_rows}
"""
    md_path = summary_dir / "edl_v2_training_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    logger.info("Saved summary MD: %s", md_path)

    # -----------------------------------------------------------------------
    # Final status
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("EDL v2 training COMPLETE")
    logger.info("  Output dir          : %s", run_dir)
    logger.info("  Train rows          : %d", len(X_train))
    logger.info("  Eval rows           : %d", len(X_eval))
    logger.info("  Features            : %d", input_dim)
    logger.info("  Accuracy            : %.3f", acc)
    logger.info(
        "  Majority baseline   : %.3f (%s)", majority_baseline_acc, majority_class_name
    )
    logger.info("  Model vs baseline   : %+.3f", model_vs_baseline)
    logger.info("  Mean vacuity        : %.4f", mean_vacuity)
    logger.info("  Pred dist           : %s", pred_dist)
    logger.info("  Warnings            : %d", len(eval_warnings))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
