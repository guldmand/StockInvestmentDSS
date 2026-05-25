#!/usr/bin/env python3
"""
run_edl_action_training_smoke_test.py  (EDL v3.2)

Small training smoke test for EDL action classifier.
Intended for verifying the training loop, loss, and checkpoint saving.
NOT intended for producing production models.

DO NOT run in CI automatically — only when explicitly invoked.

Usage
-----
    python -m stock_investment_dss.runner.run_edl_action_training_smoke_test

Environment variables
---------------------
    EDL_TRAIN_EPOCHS        number of epochs  (default: 5)
    EDL_TRAIN_LR            learning rate     (default: 1e-3)
    EDL_TRAIN_BATCH_SIZE    batch size        (default: 32)
    EDL_DATASET_DIR         path to directory containing edl_action_train_dataset.csv
                            (default: auto-discover latest outputs/runs/*/data/)
    STOCK_INVESTMENT_DSS_EDL_*  all standard EDL env vars supported

Output
------
    outputs/runs/<timestamp>_d_iqn_dss_edl_action_training_smoke_test/
        models/edl_action_classifier.pt
        audit/edl_eval_predictions.csv
        summary/edl_action_training_summary.json
        summary/edl_action_training_summary.md
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure src is on path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stock_investment_dss.uncertainty.edl_action_classes import EDLActionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("edl_action_training_smoke_test")


def _get_env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _find_latest_dataset_dir() -> Path | None:
    """Auto-discover the most recent dataset builder output directory."""
    runs_dir = _REPO_ROOT / "outputs" / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted(
        runs_dir.glob("*_d_iqn_dss_edl_action_dataset_builder"), reverse=True
    )
    for c in candidates:
        data_dir = c / "data"
        if (data_dir / "edl_action_train_dataset.csv").exists():
            return data_dir
    return None


def main() -> None:
    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    edl_config = EDLActionConfig.from_env()
    epochs = int(_get_env("EDL_TRAIN_EPOCHS", "5"))
    lr = float(_get_env("EDL_TRAIN_LR", "1e-3"))
    batch_size = int(_get_env("EDL_TRAIN_BATCH_SIZE", "32"))

    dataset_dir_env = _get_env("EDL_DATASET_DIR", "")
    if dataset_dir_env:
        dataset_dir = Path(dataset_dir_env)
    else:
        dataset_dir = _find_latest_dataset_dir()

    logger.info("=== EDL Action Training Smoke Test v3.2 ===")
    logger.info("epochs=%d  lr=%.1e  batch_size=%d", epochs, lr, batch_size)

    # ------------------------------------------------------------------
    # Lazy torch import
    # ------------------------------------------------------------------
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        TORCH_AVAILABLE = True
        logger.info("PyTorch available: %s", torch.__version__)
    except ImportError:
        TORCH_AVAILABLE = False
        logger.error("PyTorch is NOT available. Cannot run training smoke test.")
        logger.error("Install PyTorch: https://pytorch.org/get-started/locally/")
        sys.exit(1)

    import numpy as np
    import pandas as pd

    from stock_investment_dss.uncertainty.edl_action_dataset import EDLActionDataset
    from stock_investment_dss.uncertainty.edl_action_network import EDLActionNetwork
    from stock_investment_dss.uncertainty.edl_losses import edl_action_loss

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    if (
        dataset_dir is None
        or not (dataset_dir / "edl_action_train_dataset.csv").exists()
    ):
        logger.warning(
            "No train dataset found. Run run_edl_action_dataset_builder first. "
            "Generating tiny synthetic smoke dataset..."
        )
        N = max(batch_size * 4, 64)
        input_dim = edl_config.num_classes * 5  # synthetic
        K = edl_config.num_classes
        X_train = np.random.randn(N, input_dim).astype(np.float32)
        y_train = np.random.randint(0, K, size=(N,), dtype=np.int64)
        X_eval = np.random.randn(N // 4, input_dim).astype(np.float32)
        y_eval = np.random.randint(0, K, size=(N // 4,), dtype=np.int64)
        feat_cols = [f"feat_synthetic_{i}" for i in range(input_dim)]
        logger.info(
            "Synthetic dataset: train=%d, eval=%d, input_dim=%d, K=%d",
            N,
            N // 4,
            input_dim,
            K,
        )
    else:
        train_path = dataset_dir / "edl_action_train_dataset.csv"
        eval_path = dataset_dir / "edl_action_eval_dataset.csv"
        logger.info("Loading train dataset: %s", train_path)
        train_df = pd.read_csv(train_path)
        eval_df = pd.read_csv(eval_path) if eval_path.exists() else train_df.head(50)

        X_train, y_train = EDLActionDataset.to_numpy(train_df)
        X_eval, y_eval = EDLActionDataset.to_numpy(eval_df)
        feat_cols = EDLActionDataset.feature_columns_from_df(train_df)
        input_dim = X_train.shape[1]
        logger.info(
            "Dataset loaded: train=%d, eval=%d, features=%d",
            len(X_train),
            len(X_eval),
            input_dim,
        )

    # ------------------------------------------------------------------
    # Feature standardisation — train stats only, NO eval leakage
    # ------------------------------------------------------------------
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0  # protect zero-variance features
    X_train = ((X_train - feat_mean) / feat_std).astype(np.float32)
    X_eval = ((X_eval - feat_mean) / feat_std).astype(np.float32)
    logger.info("Feature standardisation applied (train mean/std only).")

    K = edl_config.num_classes
    action_classes = edl_config.action_classes
    model = EDLActionNetwork(
        input_dim=input_dim,
        num_classes=K,
        hidden_dims=[64],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Class-weighted loss (configurable via EDL_USE_CLASS_WEIGHTS)
    # ------------------------------------------------------------------
    use_class_weights = _get_env("EDL_USE_CLASS_WEIGHTS", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    class_counts_train = np.bincount(y_train, minlength=K).astype(np.float32)
    sample_weights_torch: "torch.Tensor | None" = None
    if use_class_weights:
        inv_freq = 1.0 / np.maximum(class_counts_train, 1.0)
        class_weights_arr = (inv_freq / inv_freq.mean()).astype(np.float32)
        class_weights_torch = torch.tensor(class_weights_arr, dtype=torch.float32)
        # Pre-compute per-sample weights for the full train set (used in training loop)
        sample_weights_torch = class_weights_torch[
            torch.tensor(y_train, dtype=torch.long)
        ]
        logger.info(
            "Class weights (inv-freq, norm mean=1): %s",
            {
                action_classes[i]: round(float(class_weights_arr[i]), 3)
                for i in range(K)
            },
        )
    else:
        logger.info("Class weights disabled (EDL_USE_CLASS_WEIGHTS=false).")

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    # Include per-sample class weights as a third tensor if weights are enabled
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

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger.info("Training smoke test: %d epochs...", epochs)
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
            alpha = out["alpha"]  # (batch, K)
            y_onehot = torch.zeros(y_batch.size(0), K, dtype=torch.float32)
            y_onehot.scatter_(1, y_batch.unsqueeze(1), 1.0)
            loss, mse, kl = edl_action_loss(
                alpha,
                y_onehot,
                epoch=epoch,
                total_epochs=epochs,
                kl_lambda=0.1,
            )
            # Apply per-sample class weights if available
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

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------
    def _to_scalar(value, default: float = 0.0) -> float:
        """Convert a numpy scalar, shape-(1,) or shape-(1,1) array to Python float."""
        import numpy as _np

        arr = _np.asarray(value)
        if arr.size == 0:
            return float(default)
        return float(arr.reshape(-1)[0])

    model.eval()
    X_ev = torch.tensor(X_eval, dtype=torch.float32)
    with torch.no_grad():
        eval_out = model(X_ev)
        eval_probs = eval_out["prob"].numpy()  # (N, K)
        # vacuity is (N, 1) from the network — squeeze to (N,) for indexing
        eval_vacuity = eval_out["vacuity"].squeeze(-1).numpy()  # (N,)
        eval_preds = eval_probs.argmax(axis=1)

    correct = int((eval_preds == y_eval).sum())
    acc = correct / max(len(y_eval), 1)
    mean_vacuity = float(eval_vacuity.mean())
    logger.info(
        "Eval: accuracy=%.3f  mean_vacuity=%.4f  n=%d", acc, mean_vacuity, len(y_eval)
    )

    # ------------------------------------------------------------------
    # Majority-class baseline
    # ------------------------------------------------------------------
    true_counts = np.bincount(y_eval, minlength=K)
    majority_idx = int(true_counts.argmax())
    majority_class_name = action_classes[majority_idx]
    majority_count = int(true_counts[majority_idx])
    majority_baseline_acc = majority_count / max(len(y_eval), 1)
    model_vs_baseline = round(acc - majority_baseline_acc, 4)
    logger.info(
        "Majority baseline: class=%s  count=%d  accuracy=%.3f  model_delta=%+.3f",
        majority_class_name,
        majority_count,
        majority_baseline_acc,
        model_vs_baseline,
    )

    # ------------------------------------------------------------------
    # Prediction distribution + per-class metrics
    # ------------------------------------------------------------------
    pred_counts = np.bincount(eval_preds, minlength=K)

    true_dist = {action_classes[i]: int(true_counts[i]) for i in range(K)}
    pred_dist = {action_classes[i]: int(pred_counts[i]) for i in range(K)}

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
                "class_name": action_classes[idx],
                "support": support,
                "predicted_count": predicted_count,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
            }
        )

    logger.info("True label distribution:  %s", true_dist)
    logger.info("Pred label distribution:  %s", pred_dist)

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    conf_matrix: list[list[int]] = []
    for true_idx in range(K):
        row = []
        for pred_idx in range(K):
            row.append(int(((y_eval == true_idx) & (eval_preds == pred_idx)).sum()))
        conf_matrix.append(row)

    # ------------------------------------------------------------------
    # Prediction collapse / quality warnings
    # ------------------------------------------------------------------
    eval_warnings: list[str] = []
    n_pred_classes = int((pred_counts > 0).sum())
    n_true_classes = int((true_counts > 0).sum())
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

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    ts = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
    run_name = f"{ts}_d_iqn_dss_edl_action_training_smoke_test"
    out_dir = _REPO_ROOT / "outputs" / "runs" / run_name
    models_dir = out_dir / "models"
    audit_dir = out_dir / "audit"
    summ_dir = out_dir / "summary"
    for d in (models_dir, audit_dir, summ_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint (includes scaler for reproducible inference)
    ckpt_path = models_dir / "edl_action_classifier.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": K,
            "evidence_activation": "softplus",
            "action_classes": action_classes,
            "edl_version": "3.2",
            "epochs_trained": epochs,
            "final_loss": history[-1]["loss"] if history else None,
            # Scaler — must be applied to raw features before model.forward()
            "feature_columns": feat_cols if "feat_cols" in dir() else [],
            "feature_mean": feat_mean.tolist(),
            "feature_std": feat_std.tolist(),
        },
        ckpt_path,
    )
    logger.info("Saved model: %s", ckpt_path)

    # Save eval predictions CSV
    eval_records = []
    for i in range(len(y_eval)):
        rec: dict = {
            "sample_idx": i,
            "true_label": int(y_eval[i]),
            "predicted_label": int(eval_preds[i]),
            "correct": bool(eval_preds[i] == y_eval[i]),
            "vacuity": round(_to_scalar(eval_vacuity[i]), 6),
        }
        for k, action in enumerate(action_classes):
            rec[f"p_{action.lower()}"] = round(_to_scalar(eval_probs[i, k]), 6)
        eval_records.append(rec)
    audit_csv_path = audit_dir / "edl_eval_predictions.csv"
    pd.DataFrame(eval_records).to_csv(audit_csv_path, index=False)
    logger.info("Saved eval predictions: %s", audit_csv_path)

    # Save confusion matrix CSV
    conf_df = pd.DataFrame(
        conf_matrix,
        index=[f"true_{a}" for a in action_classes],
        columns=[f"pred_{a}" for a in action_classes],
    )
    conf_csv_path = audit_dir / "edl_confusion_matrix.csv"
    conf_df.to_csv(conf_csv_path)
    logger.info("Saved confusion matrix: %s", conf_csv_path)

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------
    summary = {
        "smoke_test_version": "3.2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "use_class_weights": use_class_weights,
        "input_dim": input_dim,
        "num_classes": K,
        "action_classes": action_classes,
        "train_rows": len(X_train),
        "eval_rows": len(X_eval),
        # Accuracy
        "eval_accuracy": round(acc, 4),
        "majority_class": majority_class_name,
        "majority_class_count": majority_count,
        "majority_baseline_accuracy": round(majority_baseline_acc, 4),
        "model_accuracy_minus_majority_baseline": model_vs_baseline,
        # Uncertainty
        "mean_vacuity": round(mean_vacuity, 4),
        # Distributions
        "true_label_distribution": true_dist,
        "predicted_label_distribution": pred_dist,
        # Per-class
        "per_class_metrics": per_class_metrics,
        # Confusion matrix (list of lists, rows=true, cols=pred)
        "confusion_matrix": {
            "classes": action_classes,
            "matrix": conf_matrix,
        },
        # Warnings
        "eval_warnings": eval_warnings,
        "eval_quality_ok": len(eval_warnings) == 0,
        # Training
        "training_history": history,
        "model_path": str(ckpt_path),
    }
    json_path = summ_dir / "edl_action_training_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # ------------------------------------------------------------------
    # Summary Markdown
    # ------------------------------------------------------------------
    md_path = summ_dir / "edl_action_training_summary.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# EDL Action Training Smoke Test (v3.2)\n\n")
        fh.write(f"**Generated:** {summary['timestamp_utc']}\n\n")
        fh.write(
            f"**Epochs:** {epochs}  **LR:** {lr}  **Batch:** {batch_size}  "
            f"**Class weights:** `{'yes' if use_class_weights else 'no'}`\n\n"
        )
        fh.write(f"**Classes:** `{', '.join(action_classes)}`\n\n")

        fh.write("## Accuracy\n\n")
        beat = "✅ beats" if model_vs_baseline > 0 else "❌ does NOT beat"
        fh.write(f"| Metric | Value |\n|--------|-------|\n")
        fh.write(f"| Eval accuracy | {acc:.3f} |\n")
        fh.write(
            f"| Majority-class baseline ({majority_class_name}) | {majority_baseline_acc:.3f} |\n"
        )
        fh.write(f"| Model − baseline | {model_vs_baseline:+.3f} |\n")
        fh.write(f"| Mean vacuity | {mean_vacuity:.4f} |\n\n")
        fh.write(f"**Model {beat} majority-class baseline.**\n\n")

        fh.write("## Label distributions\n\n")
        fh.write("| Class | True count | Predicted count |\n")
        fh.write("|-------|-----------|----------------|\n")
        for idx in range(K):
            fh.write(
                f"| {action_classes[idx]} | {true_dist[action_classes[idx]]} "
                f"| {pred_dist[action_classes[idx]]} |\n"
            )

        fh.write("\n## Per-class metrics\n\n")
        fh.write("| Class | Support | Predicted | Recall | Precision |\n")
        fh.write("|-------|---------|-----------|--------|-----------|\n")
        for m in per_class_metrics:
            fh.write(
                f"| {m['class_name']} | {m['support']} | {m['predicted_count']} "
                f"| {m['recall']:.3f} | {m['precision']:.3f} |\n"
            )

        fh.write("\n## Confusion matrix (rows=true, cols=pred)\n\n")
        header = "| True \\ Pred | " + " | ".join(action_classes) + " |"
        sep = "|------------|" + "---|" * K
        fh.write(header + "\n" + sep + "\n")
        for true_idx, row in enumerate(conf_matrix):
            cells = " | ".join(str(v) for v in row)
            fh.write(f"| **{action_classes[true_idx]}** | {cells} |\n")

        if eval_warnings:
            fh.write("\n## ⚠️ Quality warnings\n\n")
            for w in eval_warnings:
                fh.write(f"- {w}\n")
            fh.write(
                f"\n**Eval quality OK:** `NO — review model before training further`\n"
            )
        else:
            fh.write("\n**✅ No quality warnings.**\n")

        fh.write("\n## Training history\n\n")
        fh.write("| Epoch | Loss |\n|-------|------|\n")
        for h in history:
            fh.write(f"| {h['epoch']} | {h['loss']:.5f} |\n")

    logger.info("Wrote summary JSON: %s", json_path)
    logger.info("Wrote summary MD:   %s", md_path)
    logger.info("=== Training smoke test complete. ===")


if __name__ == "__main__":
    main()
