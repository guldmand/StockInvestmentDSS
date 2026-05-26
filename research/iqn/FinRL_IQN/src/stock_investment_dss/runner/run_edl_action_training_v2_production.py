"""Phase B.3 — EDL-A Network Training Production Runner.

Trains the EDL-A action classifier (K=3: HOLD/BUY/SELL) using counterfactual
hindsight labels produced by Phase B.2 (edl_counterfactual_hindsight_oracle).

The trained model is the key input for Phase B.4 (ablation study) and Phase B.5
(comparison dashboard).  Labels are exogenous — computed from future price data
only — making them academically rigorous for evidential deep learning.

Usage::

    # All defaults (auto-discovers latest Phase B.2 run, 50 epochs):
    python src/stock_investment_dss/runner/run_edl_action_training_v2_production.py

    # Explicit source run:
    python src/stock_investment_dss/runner/run_edl_action_training_v2_production.py \\
        --source-b2-run-id 2026_05_26_103907_d_iqn_dss_edl_counterfactual_oracle_production

Output structure::

    outputs/runs/{timestamp}_d_iqn_dss_edl_action_training_v2_production/
    ├── audit/
    │   ├── edl_v2_eval_predictions.csv
    │   └── edl_v2_confusion_matrix.csv
    ├── config/
    │   └── training_config.json
    ├── data/
    │   └── train_eval_split.csv
    ├── logs/
    │   └── training.log
    ├── metrics/
    │   └── per_epoch_metrics.csv
    ├── models/
    │   └── edl_action_classifier_v2.pt   ← KEY OUTPUT
    ├── plots/
    │   ├── training_curves.png
    │   ├── confusion_matrix_heatmap.png
    │   └── uncertainty_distribution.png
    └── summary/
        ├── edl_v2_training_summary.md
        └── edl_v2_training_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from stock_investment_dss.utilities.paths import (
    RUNS_DIRECTORY,
    create_run_paths,
)  # noqa: E402
from stock_investment_dss.utilities.logging import setup_run_logger  # noqa: E402
from stock_investment_dss.experiment_tracking.wandb_tracking import (  # noqa: E402
    finish_wandb_run,
    init_wandb_run,
    wandb_log,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_CLASSES: List[str] = ["HOLD", "BUY", "SELL"]
NUM_CLASSES: int = 3
CLASS_TO_ID: Dict[str, int] = {"HOLD": 0, "BUY": 1, "SELL": 2}

HIDDEN_DIMS: List[int] = [128, 64]

# Columns that are metadata or labels — excluded from feature set.
# Includes the smoke-test _METADATA_COLS plus B.1/B.2 specific non-feature cols.
_EXCLUDE_PREFIXES = ("edl_a_cf_", "edl_c_", "edl_label_")
_EXCLUDE_EXACT = {
    "decision_id",
    "date",
    "tic",
    "source_iqn_run_id",
    "iqn_model_run_id",
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
    "selected_action_type",  # alias added by B.1
}


# ---------------------------------------------------------------------------
# Phase B.2 auto-discovery
# ---------------------------------------------------------------------------


def find_latest_b2_run(runs_dir: Path) -> Path:
    """Find the latest Phase B.2 EDL counterfactual oracle production run.

    Glob: ``*_d_iqn_dss_edl_counterfactual_oracle_production``  (sorted latest-first).
    Validates that ``audit/combined_with_counterfactual_labels.csv`` exists.

    Raises FileNotFoundError if no valid run is found.
    """
    candidates = sorted(
        runs_dir.glob("*_d_iqn_dss_edl_counterfactual_oracle_production"),
        reverse=True,
    )
    for run_dir in candidates:
        labels_csv = run_dir / "audit" / "combined_with_counterfactual_labels.csv"
        if labels_csv.exists():
            return run_dir
    raise FileNotFoundError(
        "No Phase B.2 production run found in outputs/runs/. "
        "Run Phase B.2 first:\n"
        "  python src/stock_investment_dss/runner/"
        "run_edl_counterfactual_hindsight_labeling_production.py"
    )


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------


def _infer_feature_cols_b2(df: pd.DataFrame) -> List[str]:
    """Infer numeric feature columns from a Phase B.2 combined CSV.

    Excludes:
    - Known metadata columns (dates, IDs, action strings, etc.)
    - All ``edl_a_cf_*`` oracle output columns
    - All ``edl_c_*`` EDL-C teacher label columns
    - All ``edl_label_*`` label columns
    Keeps all remaining numeric columns.
    """
    cols: List[str] = []
    for col in df.columns:
        if col in _EXCLUDE_EXACT:
            continue
        if any(col.startswith(p) for p in _EXCLUDE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _to_scalar(value: object, default: float = 0.0) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


# ---------------------------------------------------------------------------
# Stratified train/eval split
# ---------------------------------------------------------------------------


def _stratified_split(
    df: pd.DataFrame,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_indices, eval_indices) using stratified sampling.

    Falls back to random split if scikit-learn is unavailable.
    """
    try:
        from sklearn.model_selection import train_test_split as _tts

        idx = np.arange(len(df))
        train_idx, eval_idx = _tts(
            idx, test_size=test_size, stratify=y, random_state=seed
        )
        return train_idx, eval_idx
    except ImportError:
        pass

    # Manual fallback: for each class take the last `test_size` fraction of rows
    rng = np.random.default_rng(seed)
    train_idx_list: List[int] = []
    eval_idx_list: List[int] = []
    for cls_id in np.unique(y):
        cls_idx = np.where(y == cls_id)[0]
        rng.shuffle(cls_idx)
        n_eval = max(1, int(round(len(cls_idx) * test_size)))
        eval_idx_list.extend(cls_idx[:n_eval].tolist())
        train_idx_list.extend(cls_idx[n_eval:].tolist())
    return np.array(train_idx_list), np.array(eval_idx_list)


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def _generate_plots(
    history: List[Dict],
    eval_probs: np.ndarray,
    eval_vacuity: np.ndarray,
    y_eval: np.ndarray,
    conf_matrix: List[List[int]],
    plots_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate 3 diagnostic plots to plots/."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — plots skipped")
        return

    # 1. Training curves — dual-axis loss + accuracy
    try:
        epochs_list = [h["epoch"] for h in history]
        loss_list = [h["loss"] for h in history]
        acc_list = [h.get("eval_acc") for h in history]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Loss", color="steelblue")
        ax1.plot(epochs_list, loss_list, color="steelblue", label="Train Loss")
        ax1.tick_params(axis="y", labelcolor="steelblue")

        if any(a is not None for a in acc_list):
            ax2 = ax1.twinx()
            ax2.set_ylabel("Eval Accuracy", color="tomato")
            ax2.plot(
                epochs_list, acc_list, color="tomato", linestyle="--", label="Eval Acc"
            )
            ax2.tick_params(axis="y", labelcolor="tomato")
            ax2.set_ylim(0, 1)

        ax1.set_title("EDL-A Training Curves (Phase B.3)")
        fig.tight_layout()
        plt.savefig(plots_dir / "training_curves.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("training_curves plot failed: %s", exc)

    # 2. Confusion matrix heatmap
    try:
        conf_arr = np.array(conf_matrix, dtype=np.float32)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(conf_arr, cmap="Blues")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_xticklabels(
            [f"pred_{c}" for c in ACTION_CLASSES], rotation=30, ha="right"
        )
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_yticklabels([f"true_{c}" for c in ACTION_CLASSES])
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(
                    j,
                    i,
                    str(int(conf_arr[i, j])),
                    ha="center",
                    va="center",
                    color="white" if conf_arr[i, j] > conf_arr.max() * 0.5 else "black",
                )
        ax.set_title("EDL-A Confusion Matrix (K=3)")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(plots_dir / "confusion_matrix_heatmap.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("confusion_matrix_heatmap plot failed: %s", exc)

    # 3. Uncertainty (vacuity) distribution per predicted class
    try:
        eval_preds = eval_probs.argmax(axis=1)
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["steelblue", "seagreen", "tomato"]
        plotted = False
        for cls_id, (cls_name, color) in enumerate(zip(ACTION_CLASSES, colors)):
            mask = eval_preds == cls_id
            vals = eval_vacuity[mask]
            if vals.size > 0:
                ax.hist(
                    vals,
                    bins=15,
                    alpha=0.6,
                    color=color,
                    label=f"{cls_name} (n={vals.size})",
                )
                plotted = True
        if plotted:
            ax.set_title("Epistemic Uncertainty (Vacuity) by Predicted Class")
            ax.set_xlabel("Vacuity (K/S)")
            ax.set_ylabel("Frequency")
            ax.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "uncertainty_distribution.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("uncertainty_distribution plot failed: %s", exc)

    logger.info("Plots written to: %s", plots_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase B.3 — EDL-A Network Training Production Runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source-b2-run-id",
        default="auto",
        metavar="RUN_ID|auto",
        help="Phase B.2 run directory name, or 'auto' for latest.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="Training epochs.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Batch size.",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        metavar="LR",
        help="Learning rate.",
    )
    p.add_argument(
        "--hidden-dims",
        default="128,64",
        metavar="D1,D2,...",
        help="Hidden layer dimensions (comma-separated).",
    )
    p.add_argument(
        "--use-class-weights",
        action="store_true",
        default=True,
        help="Use inverse-frequency class weights.",
    )
    p.add_argument(
        "--train-eval-split",
        type=float,
        default=0.8,
        metavar="RATIO",
        help="Train fraction for stratified split.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="Override output directory (optional).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    run_paths = create_run_paths("d_iqn_dss_edl_action_training_v2_production")
    logger = setup_run_logger(run_paths)
    run_start = datetime.now()

    hidden_dims = [int(d.strip()) for d in args.hidden_dims.split(",")]
    test_size = 1.0 - args.train_eval_split

    logger.info("=== Phase B.3 EDL-A Network Training Production ===")
    logger.info("Source B.2 run:   %s", args.source_b2_run_id)
    logger.info("Epochs:           %d", args.epochs)
    logger.info("Batch size:       %d", args.batch_size)
    logger.info("Learning rate:    %.4f", args.learning_rate)
    logger.info("Hidden dims:      %s", hidden_dims)
    logger.info("Class weights:    %s", args.use_class_weights)
    logger.info(
        "Train/eval split: %.0f/%.0f", args.train_eval_split * 100, test_size * 100
    )
    logger.info("Seed:             %d", args.seed)
    logger.info("K (num classes):  %d (%s)", NUM_CLASSES, ACTION_CLASSES)
    logger.info("Run dir:          %s", run_paths.run_directory)

    # -----------------------------------------------------------------------
    # 1. PyTorch check
    # -----------------------------------------------------------------------
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        logger.info("PyTorch: %s", torch.__version__)
    except ImportError:
        logger.error("PyTorch not available. Install: https://pytorch.org/")
        return 1

    # -----------------------------------------------------------------------
    # 2. Deferred EDL imports
    # -----------------------------------------------------------------------
    try:
        from stock_investment_dss.uncertainty.edl_action_network import EDLActionNetwork
        from stock_investment_dss.uncertainty.edl_losses import edl_action_loss
    except ImportError as exc:
        logger.error("EDL import failed: %s", exc)
        return 1

    # -----------------------------------------------------------------------
    # 3. Locate Phase B.2 run
    # -----------------------------------------------------------------------
    if args.source_b2_run_id == "auto":
        try:
            b2_run_dir = find_latest_b2_run(RUNS_DIRECTORY)
            logger.info("B.2 run (auto): %s", b2_run_dir.name)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 1
    else:
        b2_run_dir = RUNS_DIRECTORY / args.source_b2_run_id
        if not b2_run_dir.is_dir():
            logger.error("B.2 run directory not found: %s", b2_run_dir)
            return 1

    labels_csv = b2_run_dir / "audit" / "combined_with_counterfactual_labels.csv"
    if not labels_csv.exists():
        logger.error("B.2 labels CSV not found: %s", labels_csv)
        return 1

    # -----------------------------------------------------------------------
    # 4. Load data
    # -----------------------------------------------------------------------
    logger.info("Loading B.2 labels CSV: %s", labels_csv.name)
    df = pd.read_csv(labels_csv)
    logger.info("Loaded: %d rows, %d cols", len(df), len(df.columns))

    # Filter to available labels only
    if "edl_a_cf_label_available" in df.columns:
        n_before = len(df)
        df = df[df["edl_a_cf_label_available"] == True].reset_index(
            drop=True
        )  # noqa: E712
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.warning(
                "Dropped %d rows with edl_a_cf_label_available != True", n_dropped
            )
    logger.info("Available-label rows: %d", len(df))

    if len(df) < 10:
        logger.error(
            "Too few labeled rows (%d) to train. Check Phase B.2 output.", len(df)
        )
        return 1

    # Map string labels → integer IDs
    if "edl_a_cf_label" not in df.columns:
        logger.error("Column 'edl_a_cf_label' not found in B.2 CSV.")
        return 1

    unknown_labels = set(df["edl_a_cf_label"].unique()) - set(CLASS_TO_ID)
    if unknown_labels:
        logger.error("Unknown labels in edl_a_cf_label: %s", unknown_labels)
        return 1

    y_all = df["edl_a_cf_label"].map(CLASS_TO_ID).values.astype(np.int64)

    # -----------------------------------------------------------------------
    # 5. Feature inference
    # -----------------------------------------------------------------------
    feature_cols = _infer_feature_cols_b2(df)
    if not feature_cols:
        logger.error("No feature columns found after exclusion. Check B.2 CSV.")
        return 1
    logger.info("Inferred %d feature columns", len(feature_cols))

    X_all = df[feature_cols].values.astype(np.float32)
    n_nan = int(np.isnan(X_all).sum())
    if n_nan:
        logger.warning("NaN in features: %d cells — replacing with 0.0", n_nan)
        X_all = np.nan_to_num(X_all, nan=0.0)

    input_dim = X_all.shape[1]

    # -----------------------------------------------------------------------
    # 6. Stratified 80/20 split
    # -----------------------------------------------------------------------
    np.random.seed(args.seed)
    train_idx, eval_idx = _stratified_split(
        df, y_all, test_size=test_size, seed=args.seed
    )
    X_train, X_eval = X_all[train_idx], X_all[eval_idx]
    y_train, y_eval = y_all[train_idx], y_all[eval_idx]
    logger.info("Split: train=%d, eval=%d", len(X_train), len(X_eval))

    # Label distributions
    train_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.int64)
    eval_counts = np.bincount(y_eval, minlength=NUM_CLASSES).astype(np.int64)
    train_label_dist = {
        ACTION_CLASSES[i]: int(train_counts[i]) for i in range(NUM_CLASSES)
    }
    eval_label_dist = {
        ACTION_CLASSES[i]: int(eval_counts[i]) for i in range(NUM_CLASSES)
    }
    logger.info("Train label dist: %s", train_label_dist)
    logger.info("Eval  label dist: %s", eval_label_dist)

    # -----------------------------------------------------------------------
    # 7. Feature standardisation — train stats only (no eval leakage)
    # -----------------------------------------------------------------------
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    X_train = ((X_train - feat_mean) / feat_std).astype(np.float32)
    X_eval = ((X_eval - feat_mean) / feat_std).astype(np.float32)
    logger.info("Feature standardisation applied (train mean/std, no eval leakage).")

    # -----------------------------------------------------------------------
    # 8. Write config + split CSV
    # -----------------------------------------------------------------------
    training_cfg: dict = {
        "source_b2_run_id": b2_run_dir.name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dims": hidden_dims,
        "use_class_weights": args.use_class_weights,
        "train_eval_split": args.train_eval_split,
        "seed": args.seed,
        "num_classes": NUM_CLASSES,
        "action_classes": ACTION_CLASSES,
        "input_dim": input_dim,
        "feature_columns": feature_cols,
        "total_rows": len(df),
        "train_rows": len(X_train),
        "eval_rows": len(X_eval),
        "train_label_distribution": train_label_dist,
        "eval_label_distribution": eval_label_dist,
        "run_start": run_start.isoformat(),
    }
    (run_paths.config_directory / "training_config.json").write_text(
        json.dumps(training_cfg, indent=2), encoding="utf-8"
    )

    # Save split membership CSV
    split_records = []
    for i in train_idx:
        split_records.append(
            {
                "row_idx": int(i),
                "split": "train",
                "label": ACTION_CLASSES[int(y_all[i])],
            }
        )
    for i in eval_idx:
        split_records.append(
            {"row_idx": int(i), "split": "eval", "label": ACTION_CLASSES[int(y_all[i])]}
        )
    pd.DataFrame(split_records).sort_values("row_idx").to_csv(
        run_paths.data_directory / "train_eval_split.csv", index=False
    )

    # -----------------------------------------------------------------------
    # 9. W&B init
    # -----------------------------------------------------------------------
    init_wandb_run(
        run_name=run_paths.run_id,
        config=training_cfg,
        group="phase-b3-edl-training",
        job_type="edl_training",
        tags=["phase-b", "edl-a", "training", "k3"],
        run_directory=str(run_paths.run_directory),
    )

    # -----------------------------------------------------------------------
    # 10. Build model
    # -----------------------------------------------------------------------
    torch.manual_seed(args.seed)
    model = EDLActionNetwork(
        input_dim=input_dim,
        num_classes=NUM_CLASSES,
        hidden_dims=hidden_dims,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    logger.info(
        "Model: EDLActionNetwork(input=%d, K=%d, hidden=%s)",
        input_dim,
        NUM_CLASSES,
        hidden_dims,
    )

    # -----------------------------------------------------------------------
    # 11. Class weights
    # -----------------------------------------------------------------------
    sample_weights_torch: Optional[torch.Tensor] = None
    class_weights_log: Dict[str, float] = {}
    if args.use_class_weights:
        inv_freq = 1.0 / np.maximum(train_counts.astype(np.float32), 1.0)
        cw_arr = (inv_freq / inv_freq.mean()).astype(np.float32)
        class_weights_log = {
            ACTION_CLASSES[i]: round(float(cw_arr[i]), 3) for i in range(NUM_CLASSES)
        }
        cw_torch = torch.tensor(cw_arr, dtype=torch.float32)
        sample_weights_torch = cw_torch[torch.tensor(y_train, dtype=torch.long)]
        logger.info("Class weights (inv-freq, mean=1): %s", class_weights_log)
    else:
        logger.info("Class weights disabled.")

    # -----------------------------------------------------------------------
    # 12. DataLoader
    # -----------------------------------------------------------------------
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    if sample_weights_torch is not None:
        loader = DataLoader(
            TensorDataset(X_t, y_t, sample_weights_torch),
            batch_size=args.batch_size,
            shuffle=True,
        )
    else:
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=args.batch_size,
            shuffle=True,
        )

    # -----------------------------------------------------------------------
    # 13. Training loop
    # -----------------------------------------------------------------------
    logger.info(
        "Training: %d epochs, batch_size=%d, lr=%.4f",
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )
    X_ev_t = torch.tensor(X_eval, dtype=torch.float32)
    history: List[Dict] = []

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = epoch_mse = epoch_kl = 0.0
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
            y_onehot = torch.zeros(y_batch.size(0), NUM_CLASSES, dtype=torch.float32)
            y_onehot.scatter_(1, y_batch.unsqueeze(1), 1.0)

            loss, mse, kl = edl_action_loss(
                alpha,
                y_onehot,
                epoch=epoch,
                total_epochs=args.epochs,
                kl_lambda=0.1,
            )
            batch_loss = (loss * w_batch).mean() if w_batch is not None else loss.mean()

            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.detach().item()
            epoch_mse += mse.mean().detach().item()
            epoch_kl += kl.mean().detach().item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Quick eval accuracy snapshot every 10 epochs + final
        eval_acc: Optional[float] = None
        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                ep = model(X_ev_t)["prob"].numpy()
            model.train()
            eval_acc = float((ep.argmax(axis=1) == y_eval).mean())
            logger.info(
                "  epoch %3d/%d  loss=%.5f  eval_acc=%.3f",
                epoch,
                args.epochs,
                avg_loss,
                eval_acc,
            )
        else:
            logger.info("  epoch %3d/%d  loss=%.5f", epoch, args.epochs, avg_loss)

        history.append(
            {
                "epoch": epoch,
                "loss": round(avg_loss, 6),
                "mse_loss": round(epoch_mse / max(n_batches, 1), 6),
                "kl_loss": round(epoch_kl / max(n_batches, 1), 6),
                "eval_acc": round(eval_acc, 4) if eval_acc is not None else None,
            }
        )

    # Write per-epoch metrics CSV
    pd.DataFrame(history).to_csv(
        run_paths.metrics_directory / "per_epoch_metrics.csv", index=False
    )

    # -----------------------------------------------------------------------
    # 14. Final evaluation
    # -----------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        eval_out = model(X_ev_t)
        eval_probs = eval_out["prob"].numpy()  # (N, K)
        eval_vacuity = eval_out["vacuity"].squeeze(-1).numpy()  # (N,)
        eval_preds = eval_probs.argmax(axis=1)

    correct = int((eval_preds == y_eval).sum())
    acc = correct / max(len(y_eval), 1)
    mean_vacuity = float(eval_vacuity.mean())

    # Brier score
    y_onehot_np = np.eye(NUM_CLASSES, dtype=np.float32)[y_eval]
    brier_score = float(np.mean(np.sum((eval_probs - y_onehot_np) ** 2, axis=1)))

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
    logger.info("Mean vacuity: %.4f | Brier score: %.4f", mean_vacuity, brier_score)

    # Prediction distribution
    pred_counts = np.bincount(eval_preds, minlength=NUM_CLASSES)
    true_dist = {ACTION_CLASSES[i]: int(eval_counts[i]) for i in range(NUM_CLASSES)}
    pred_dist = {ACTION_CLASSES[i]: int(pred_counts[i]) for i in range(NUM_CLASSES)}
    logger.info("True eval dist: %s | Pred eval dist: %s", true_dist, pred_dist)

    # Per-class metrics
    per_class_metrics: List[Dict] = []
    for idx in range(NUM_CLASSES):
        tm = y_eval == idx
        pm = eval_preds == idx
        support = int(tm.sum())
        predicted = int(pm.sum())
        tp = int((tm & pm).sum())
        recall = tp / max(support, 1)
        precision = tp / max(predicted, 1)
        per_class_metrics.append(
            {
                "class_idx": idx,
                "class_name": ACTION_CLASSES[idx],
                "support": support,
                "predicted_count": predicted,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
            }
        )

    # Confusion matrix (K=3)
    conf_matrix = [
        [int(((y_eval == ti) & (eval_preds == pi)).sum()) for pi in range(NUM_CLASSES)]
        for ti in range(NUM_CLASSES)
    ]

    # Quality warnings
    eval_warnings: List[str] = []
    n_pred_classes = int((pred_counts > 0).sum())
    n_true_classes = int((eval_counts > 0).sum())
    if n_pred_classes < 2:
        eval_warnings.append(
            f"CRITICAL: model predicts only {n_pred_classes} class(es) — "
            "severe prediction collapse."
        )
    elif n_pred_classes < n_true_classes:
        eval_warnings.append(
            f"WARNING: model predicts {n_pred_classes} classes but eval has "
            f"{n_true_classes} true classes — partial collapse."
        )
    if acc <= majority_baseline_acc:
        eval_warnings.append(
            f"WARNING: eval accuracy {acc:.3f} ≤ majority baseline "
            f"{majority_baseline_acc:.3f}."
        )
    for pcm in per_class_metrics:
        if pcm["support"] < 5:
            eval_warnings.append(
                f"NOTE: class '{pcm['class_name']}' has only {pcm['support']} eval "
                "samples — per-class metrics are noisy."
            )
    for w in eval_warnings:
        if w.startswith("CRITICAL"):
            logger.error("QUALITY: %s", w)
        else:
            logger.warning("QUALITY: %s", w)

    edl_a_ready = n_pred_classes == NUM_CLASSES and acc > majority_baseline_acc

    # -----------------------------------------------------------------------
    # 15. Save model checkpoint
    # -----------------------------------------------------------------------
    ckpt_path = run_paths.models_directory / "edl_action_classifier_v2.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": NUM_CLASSES,
            "hidden_dims": hidden_dims,
            "evidence_activation": "softplus",
            "action_classes": ACTION_CLASSES,
            "feature_columns": feature_cols,
            "feature_mean": feat_mean.tolist(),
            "feature_std": feat_std.tolist(),
            "source_b2_run_id": b2_run_dir.name,
            "label_mode": "counterfactual",
            "edl_version": "3.5",
            "epochs_trained": args.epochs,
            "final_loss": history[-1]["loss"] if history else None,
            "eval_accuracy": round(acc, 4),
            "majority_baseline_accuracy": round(majority_baseline_acc, 4),
            "mean_vacuity": round(mean_vacuity, 4),
            "brier_score": round(brier_score, 4),
        },
        ckpt_path,
    )
    logger.info("Saved model: %s", ckpt_path)

    # -----------------------------------------------------------------------
    # 16. Save eval predictions CSV
    # -----------------------------------------------------------------------
    meta_cols = [
        c
        for c in [
            "decision_id",
            "date",
            "tic",
            "hierarchical_action_type",
            "edl_a_cf_label",
            "edl_a_cf_label_id",
        ]
        if c in df.columns
    ]
    eval_df_rows = df.iloc[eval_idx].reset_index(drop=True)
    eval_records: List[Dict] = []
    for i in range(len(y_eval)):
        rec: Dict = {}
        for col in meta_cols:
            rec[col] = eval_df_rows.at[i, col] if col in eval_df_rows.columns else ""
        rec.update(
            {
                "sample_idx": int(eval_idx[i]),
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

    preds_csv = run_paths.audit_directory / "edl_v2_eval_predictions.csv"
    pd.DataFrame(eval_records).to_csv(preds_csv, index=False)
    logger.info(
        "Saved eval predictions: %s (%d rows)", preds_csv.name, len(eval_records)
    )

    # Confusion matrix CSV
    conf_df = pd.DataFrame(
        conf_matrix,
        index=[f"true_{a}" for a in ACTION_CLASSES],
        columns=[f"pred_{a}" for a in ACTION_CLASSES],
    )
    conf_csv = run_paths.audit_directory / "edl_v2_confusion_matrix.csv"
    conf_df.to_csv(conf_csv)
    logger.info("Saved confusion matrix: %s", conf_csv.name)

    # -----------------------------------------------------------------------
    # 17. Plots
    # -----------------------------------------------------------------------
    _generate_plots(
        history=history,
        eval_probs=eval_probs,
        eval_vacuity=eval_vacuity,
        y_eval=y_eval,
        conf_matrix=conf_matrix,
        plots_dir=run_paths.plots_directory,
        logger=logger,
    )

    # -----------------------------------------------------------------------
    # 18. Summary JSON
    # -----------------------------------------------------------------------
    dur_sec = (datetime.now() - run_start).total_seconds()
    summary: dict = {
        "production_version": "3.5",
        "timestamp": datetime.now().isoformat(),
        "source_b2_run_id": b2_run_dir.name,
        "label_mode": "counterfactual",
        "epochs": args.epochs,
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "hidden_dims": hidden_dims,
        "use_class_weights": args.use_class_weights,
        "class_weights": class_weights_log,
        "input_dim": input_dim,
        "num_classes": NUM_CLASSES,
        "action_classes": ACTION_CLASSES,
        "feature_columns": feature_cols,
        "total_rows": len(df),
        "train_rows": len(X_train),
        "eval_rows": len(X_eval),
        "train_label_distribution": train_label_dist,
        "eval_label_distribution_true": true_dist,
        "eval_label_distribution_predicted": pred_dist,
        "eval_accuracy": round(acc, 4),
        "majority_class": majority_class_name,
        "majority_baseline_accuracy": round(majority_baseline_acc, 4),
        "model_vs_baseline_delta": model_vs_baseline,
        "mean_vacuity": round(mean_vacuity, 4),
        "brier_score": round(brier_score, 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": {"classes": ACTION_CLASSES, "matrix": conf_matrix},
        "eval_warnings": eval_warnings,
        "eval_quality_ok": len(
            [
                w
                for w in eval_warnings
                if w.startswith("CRITICAL") or w.startswith("WARNING")
            ]
        )
        == 0,
        "edl_a_ready": edl_a_ready,
        "phase_b4_input_path": str(ckpt_path),
        "training_history": history,
        "model_path": str(ckpt_path),
        "eval_predictions_path": str(preds_csv),
        "duration_seconds": round(dur_sec, 1),
        "run_id": run_paths.run_id,
    }
    json_path = run_paths.summary_directory / "edl_v2_training_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Saved summary JSON: %s", json_path.name)

    # -----------------------------------------------------------------------
    # 19. Summary Markdown
    # -----------------------------------------------------------------------
    beat_sym = "beats" if model_vs_baseline > 0 else "does NOT beat"

    def _pcm_row(m: Dict) -> str:
        return (
            f"| {m['class_name']} | {m['support']} | {m['predicted_count']} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} |"
        )

    warn_block = (
        "\n".join(f"- {w}" for w in eval_warnings) if eval_warnings else "_None_"
    )
    md = (
        "# EDL-A Network Training Production Run (Phase B.3)\n\n"
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Source: `{b2_run_dir.name}`\n\n"
        "## Configuration\n\n"
        f"- Epochs: {args.epochs}\n"
        f"- Batch size: {args.batch_size}\n"
        f"- Learning rate: {args.learning_rate}\n"
        f"- Hidden dims: {hidden_dims}\n"
        f"- K (classes): {NUM_CLASSES} ({ACTION_CLASSES})\n"
        f"- Class weights: {args.use_class_weights}\n"
        f"- Train/eval split: {args.train_eval_split:.0%}/{test_size:.0%}\n"
        f"- Seed: {args.seed}\n\n"
        "## Results\n\n"
        f"- Eval accuracy: **{acc:.3f}** ({beat_sym} majority baseline {majority_baseline_acc:.3f})\n"
        f"- Majority baseline: `{majority_class_name}` ({majority_count}/{len(y_eval)} eval rows)\n"
        f"- Delta vs baseline: **{model_vs_baseline:+.3f}**\n"
        f"- Mean vacuity: {mean_vacuity:.4f}\n"
        f"- Brier score: {brier_score:.4f}\n\n"
        "## Per-Class Metrics (Eval)\n\n"
        "| Class | Support | Predicted | Precision | Recall |\n"
        "|-------|---------|-----------|-----------|--------|\n"
        + "\n".join(_pcm_row(m) for m in per_class_metrics)
        + "\n\n"
        "## Label Distributions\n\n"
        f"- Train: {train_label_dist}\n"
        f"- Eval true: {true_dist}\n"
        f"- Eval predicted: {pred_dist}\n\n"
        "## Quality Warnings\n\n"
        f"{warn_block}\n\n"
        "## EDL-A Readiness\n\n"
        f"- `edl_a_ready`: **{edl_a_ready}**\n"
        f"- Predicts all {NUM_CLASSES} classes: {n_pred_classes == NUM_CLASSES}\n"
        f"- Beats majority baseline: {model_vs_baseline > 0}\n\n"
        "## Output Files\n\n"
        f"- Model: `{ckpt_path}`\n"
        f"- Eval predictions: `{preds_csv}`\n"
        f"- Run dir: `{run_paths.run_directory}`\n"
        f"- Duration: {int(dur_sec // 60)}m {int(dur_sec % 60)}s\n"
    )
    (run_paths.summary_directory / "edl_v2_training_summary.md").write_text(
        md, encoding="utf-8"
    )

    # -----------------------------------------------------------------------
    # 20. W&B metrics
    # -----------------------------------------------------------------------
    wandb_log(
        {
            "eval_accuracy": acc,
            "majority_baseline_accuracy": majority_baseline_acc,
            "model_vs_baseline_delta": model_vs_baseline,
            "mean_vacuity": mean_vacuity,
            "brier_score": brier_score,
            "edl_a_ready": int(edl_a_ready),
            "final_train_loss": history[-1]["loss"] if history else None,
            **{
                f"precision_{m['class_name'].lower()}": m["precision"]
                for m in per_class_metrics
            },
            **{
                f"recall_{m['class_name'].lower()}": m["recall"]
                for m in per_class_metrics
            },
        }
    )

    logger.info("=== Phase B.3 EDL-A training complete ===")
    logger.info("Run directory:    %s", run_paths.run_directory)
    logger.info("Model checkpoint: %s", ckpt_path)
    logger.info("Eval accuracy:    %.3f (delta: %+.3f)", acc, model_vs_baseline)
    logger.info("Mean vacuity:     %.4f", mean_vacuity)
    logger.info("EDL-A ready:      %s", edl_a_ready)
    logger.info("Duration:         %.1f s", dur_sec)

    try:
        finish_wandb_run()
    except Exception:  # noqa: BLE001
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
