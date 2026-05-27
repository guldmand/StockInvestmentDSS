"""Phase B.3 v3 — EDL-A Production Training with Nested K-fold HP Search.

Methodology (Sensoy et al. 2018 + sequential HP search):
  Step 1: Load Phase B.2 counterfactual labels (599 rows, K=3)
  Step 2: Stratified 80/20 outer split → Train* (479) / Test (120)
  Step 3: Sequential HP search on Train*:
    Phase 1 — Optimizer × Activation × Architecture
    Phase 2 — LR × Weight decay × Batch size (+ Momentum if SGD)
    Phase 3 — Dropout × KL lambda
  Step 4a — K-fold ensemble training on Train*
  Step 4b — Single final model on 80/20 inner split of Train*
  Step 4c — 3-way test evaluation comparison

Usage::
    # Quick mode (~5-10 min):
    python -m stock_investment_dss.runner.run_edl_action_training_v3_production \\
        --search-mode quick --no-wandb

    # Full mode (~3 hours):
    wandb login
    python -m stock_investment_dss.runner.run_edl_action_training_v3_production
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("ERROR: PyTorch required. Install: pip install torch")
    sys.exit(1)

try:
    from sklearn.model_selection import StratifiedKFold, train_test_split
except ImportError:
    print("ERROR: scikit-learn required. Install: pip install scikit-learn")
    sys.exit(1)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from stock_investment_dss.uncertainty.edl_losses import (
        dirichlet_mse_loss,
        kl_divergence_uniform,
    )
except ImportError as _exc:
    print(f"ERROR: EDL losses import failed: {_exc}")
    sys.exit(1)

from stock_investment_dss.utilities.paths import RUNS_DIRECTORY, create_run_paths
from stock_investment_dss.utilities.logging import setup_run_logger

# =============================================================================
# Constants
# =============================================================================
ACTION_CLASSES: List[str] = ["HOLD", "BUY", "SELL"]
NUM_CLASSES: int = 3
CLASS_TO_ID: Dict[str, int] = {"HOLD": 0, "BUY": 1, "SELL": 2}

# Device detection (use GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_EXCLUDE_PREFIXES = (
    "edl_a_",  # ALL edl_a_* (hindsight + cf labels + future returns)
    "edl_b_",  # rule labels
    "edl_c_",  # teacher labels
    "edl_label_",  # generic label columns
)
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
    "selected_action_type",
}

_DEFAULTS_PHASE_1 = {
    "lr": 0.001,
    "weight_decay": 0.0001,
    "dropout": 0.2,
    "momentum": 0.9,
    "batch_size": 32,
    "kl_lambda_max": 0.1,
}

FULL_GRIDS: Dict = {
    "phase1": {
        "optimizer": ["Adam", "AdamW", "SGD"],
        "activation": ["ReLU", "GELU", "SiLU", "Mish"],
        "hidden_dims": [[64], [128, 64], [256, 128]],
    },
    "phase2_base": {
        "lr": [0.0001, 0.0005, 0.001, 0.002],
        "weight_decay": [0.0, 0.0001, 0.001, 0.01],
        "batch_size": [16, 32, 64],
    },
    "phase2_momentum": {"momentum": [0.7, 0.9, 0.99]},
    "phase3": {
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "kl_lambda_max": [0.05, 0.1, 0.2],
    },
    "cv_folds": 10,
}

QUICK_GRIDS: Dict = {
    "phase1": {
        "optimizer": ["Adam", "AdamW"],
        "activation": ["ReLU", "SiLU"],
        "hidden_dims": [[64], [128, 64]],
    },
    "phase2_base": {
        "lr": [0.0005, 0.001],
        "weight_decay": [0.0, 0.0001, 0.001],
        "batch_size": [32],
    },
    "phase2_momentum": {"momentum": [0.7, 0.9, 0.99]},
    "phase3": {
        "dropout": [0.1, 0.3],
        "kl_lambda_max": [0.05, 0.1],
    },
    "cv_folds": 3,
}


# =============================================================================
# Network
# =============================================================================
class EDLActionNetworkV3(nn.Module):
    """MLP with configurable activation + LayerNorm + F.relu evidence head.

    Returns alpha (Dirichlet concentration) tensor directly (NOT dict).
    """

    _ACTIVATION_MAP = {
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "SiLU": nn.SiLU,
        "Mish": nn.Mish,
    }

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int],
        activation: str = "SiLU",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if activation not in self._ACTIVATION_MAP:
            raise ValueError(f"Unknown activation '{activation}'")
        ActFn = self._ACTIVATION_MAP[activation]

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = list(hidden_dims)
        self.activation_name = activation
        self.dropout_rate = dropout

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(ActFn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.body(x)
        evidence = F.relu(logits)
        return evidence + 1.0


# =============================================================================
# Helpers
# =============================================================================
def build_optimizer(
    name: str, params, lr: float, weight_decay: float, momentum: float = 0.9
) -> torch.optim.Optimizer:
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "SGD":
        return torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=False
        )
    raise ValueError(f"Unknown optimizer: '{name}'")


def edl_mse_loss_v3_per_sample(
    alpha: torch.Tensor, y_onehot: torch.Tensor, epoch: int, kl_lambda_max: float = 0.1
) -> torch.Tensor:
    """Sensoy 2018 Eq. 5 per-sample. KL annealing: lambda_t = min(lambda_max, (t+1)/10)."""
    L_data = dirichlet_mse_loss(alpha, y_onehot)
    annealing = min(kl_lambda_max, (epoch + 1) / 10.0)
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    kl = kl_divergence_uniform(alpha_tilde)
    return L_data + annealing * kl


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    sample_weights: Optional[np.ndarray] = None,
) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    if sample_weights is not None:
        sw_t = torch.tensor(sample_weights, dtype=torch.float32)
        ds = TensorDataset(X_t, y_t, sw_t)
    else:
        ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _compute_sample_weights(y: np.ndarray, K: int = NUM_CLASSES) -> np.ndarray:
    counts = np.bincount(y, minlength=K).astype(np.float32)
    inv_freq = 1.0 / np.maximum(counts, 1.0)
    cw = (inv_freq / inv_freq.mean()).astype(np.float32)
    return cw[y]


def _combo_str(combo: dict) -> str:
    parts = []
    for k, v in combo.items():
        if isinstance(v, list):
            parts.append(f"{k}={'x'.join(str(d) for d in v)}")
        else:
            parts.append(f"{k}={v}")
    return "_".join(parts)


def _hidden_dims_from_row(row) -> List[int]:
    v = row["hidden_dims"]
    if isinstance(v, list):
        return [int(x) for x in v]
    if isinstance(v, str):
        import ast

        parsed = ast.literal_eval(v)
        return [int(x) for x in parsed]
    return [int(x) for x in v]


def find_latest_b2_run(runs_dir: Path) -> Path:
    candidates = sorted(
        runs_dir.glob("*_d_iqn_dss_edl_counterfactual_oracle_production"),
        reverse=True,
    )
    for run_dir in candidates:
        if (run_dir / "audit" / "combined_with_counterfactual_labels.csv").exists():
            return run_dir
    raise FileNotFoundError("No Phase B.2 production run found. Run Phase B.2 first.")


def _infer_feature_cols_b2(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        if col in _EXCLUDE_EXACT:
            continue
        if any(col.startswith(p) for p in _EXCLUDE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def check_wandb_setup(project: str) -> bool:
    try:
        import wandb

        api = wandb.Api()
        return True
    except Exception:
        print(
            "ERROR: W&B not configured.\n"
            "  Run: wandb login\n"
            "  Or pass --no-wandb to disable W&B logging."
        )
        return False


# =============================================================================
# Training
# =============================================================================
def train_with_early_stopping(
    model: EDLActionNetworkV3,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    wandb_run=None,
) -> Tuple[EDLActionNetworkV3, dict]:
    K = NUM_CLASSES
    kl_lambda_max = config.get("kl_lambda_max", 0.1)

    # Move model to device
    model = model.to(DEVICE)
    max_epochs = config.get("max_epochs", 3000)
    patience = config.get("early_stop_patience", 20)
    lr_sched_patience = config.get("lr_scheduler_patience", 20)

    optimizer = build_optimizer(
        config["optimizer"],
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        momentum=config.get("momentum", 0.9),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_sched_patience
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state: Optional[dict] = None
    patience_counter = 0
    history: dict = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_data in train_loader:
            if len(batch_data) == 3:
                X_b, y_b, w_b = batch_data
                X_b, y_b, w_b = X_b.to(DEVICE), y_b.to(DEVICE), w_b.to(DEVICE)
            else:
                X_b, y_b = batch_data
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                w_b = None

            optimizer.zero_grad()
            alpha = model(X_b)
            y_onehot = torch.zeros(y_b.size(0), K, dtype=torch.float32, device=DEVICE)
            y_onehot.scatter_(1, y_b.unsqueeze(1), 1.0)

            per_sample = edl_mse_loss_v3_per_sample(
                alpha, y_onehot, epoch, kl_lambda_max
            )
            batch_loss = (
                (per_sample * w_b).mean() if w_b is not None else per_sample.mean()
            )
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.detach().item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        correct = 0
        n_val = 0
        n_val_batches = 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                alpha_v = model(X_v)
                y_v_onehot = torch.zeros(
                    y_v.size(0), K, dtype=torch.float32, device=DEVICE
                )
                y_v_onehot.scatter_(1, y_v.unsqueeze(1), 1.0)
                per_s = edl_mse_loss_v3_per_sample(
                    alpha_v, y_v_onehot, epoch, kl_lambda_max
                )
                val_loss += per_s.mean().item()
                probs = alpha_v / alpha_v.sum(dim=-1, keepdim=True)
                correct += (probs.argmax(dim=1) == y_v).sum().item()
                n_val += y_v.size(0)
                n_val_batches += 1

        avg_val_loss = val_loss / max(n_val_batches, 1)
        val_acc = correct / max(n_val, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if wandb_run is not None:
            try:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "val_acc": val_acc,
                        "lr": current_lr,
                    }
                )
            except Exception:
                pass

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val_loss
    history["best_val_acc"] = best_val_acc
    history["epochs_trained"] = len(history["train_loss"])
    return model, history


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_model(
    model_or_models, X_tensor: torch.Tensor, y_np: np.ndarray, K: int = NUM_CLASSES
) -> dict:
    """Evaluate single model OR list (ensemble averages alphas)."""
    X_tensor = X_tensor.to(DEVICE)
    with torch.no_grad():
        if isinstance(model_or_models, list):
            alphas_list = []
            for m in model_or_models:
                m = m.to(DEVICE)
                m.eval()
                alphas_list.append(m(X_tensor))
            alpha = torch.stack(alphas_list).mean(dim=0)
        else:
            model_or_models = model_or_models.to(DEVICE)
            model_or_models.eval()
            alpha = model_or_models(X_tensor)

    alpha_np = alpha.cpu().numpy()
    S = alpha_np.sum(axis=1)
    probs = alpha_np / S[:, None]
    preds = probs.argmax(axis=1)
    vacuity = K / S

    acc = float((preds == y_np).mean())
    counts = np.bincount(y_np, minlength=K)
    majority_baseline = float(counts.max() / len(y_np))
    y_onehot = np.eye(K, dtype=np.float32)[y_np]
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    per_class = []
    for i in range(K):
        tm = y_np == i
        pm = preds == i
        support = int(tm.sum())
        predicted = int(pm.sum())
        tp = int((tm & pm).sum())
        per_class.append(
            {
                "class_name": ACTION_CLASSES[i],
                "support": support,
                "predicted": predicted,
                "precision": round(tp / max(predicted, 1), 4),
                "recall": round(tp / max(support, 1), 4),
            }
        )

    conf_matrix = [
        [int(((y_np == ti) & (preds == pi)).sum()) for pi in range(K)]
        for ti in range(K)
    ]
    n_pred_classes = int((np.bincount(preds, minlength=K) > 0).sum())

    return {
        "accuracy": round(acc, 4),
        "majority_baseline": round(majority_baseline, 4),
        "accuracy_vs_baseline": round(acc - majority_baseline, 4),
        "mean_vacuity": round(float(vacuity.mean()), 4),
        "brier_score": round(brier, 4),
        "n_classes_predicted": n_pred_classes,
        "per_class_metrics": per_class,
        "confusion_matrix": conf_matrix,
        "_probs": probs,
        "_preds": preds,
        "_vacuity": vacuity,
        "_alphas": alpha_np,
    }


def save_test_predictions(
    model_or_models,
    X_tensor: torch.Tensor,
    y_np: np.ndarray,
    out_path: Path,
    K: int = NUM_CLASSES,
) -> None:
    result = evaluate_model(model_or_models, X_tensor, y_np, K)
    probs = result["_probs"]
    preds = result["_preds"]
    vacuity = result["_vacuity"]
    alphas = result["_alphas"]
    records = []
    for i in range(len(y_np)):
        rec = {
            "sample_idx": i,
            "true_label_id": int(y_np[i]),
            "true_label": ACTION_CLASSES[int(y_np[i])],
            "pred_label_id": int(preds[i]),
            "pred_label": ACTION_CLASSES[int(preds[i])],
            "correct": bool(preds[i] == y_np[i]),
            "vacuity": round(float(vacuity[i]), 6),
        }
        for k, cls in enumerate(ACTION_CLASSES):
            rec[f"p_{cls.lower()}"] = round(float(probs[i, k]), 6)
            rec[f"alpha_{cls.lower()}"] = round(float(alphas[i, k]), 6)
        records.append(rec)
    pd.DataFrame(records).to_csv(out_path, index=False)


# =============================================================================
# HP Search Runner
# =============================================================================
def run_hp_phase(
    phase_num,
    grid: dict,
    defaults: dict,
    X: np.ndarray,
    y: np.ndarray,
    skf: StratifiedKFold,
    input_dim: int,
    args: argparse.Namespace,
    hp_search_dir: Path,
    csv_filename: str,
    logger: logging.Logger,
    wandb_urls: List[str],
) -> pd.DataFrame:
    """Generic HP search phase. For each combo in grid, runs K-fold CV."""
    keys = list(grid.keys())
    all_combos = list(itertools.product(*[grid[k] for k in keys]))
    n_combos = len(all_combos)
    n_folds = skf.n_splits

    logger.info(
        "Phase %s HP search: %d combos x %d folds = %d runs",
        phase_num,
        n_combos,
        n_folds,
        n_combos * n_folds,
    )

    results = []
    for combo_idx, combo_vals in enumerate(all_combos):
        combo = dict(zip(keys, combo_vals))
        config = {**defaults, **combo}
        if isinstance(config.get("hidden_dims"), (list, tuple)):
            config["hidden_dims"] = list(config["hidden_dims"])

        fold_accs: List[float] = []
        fold_epochs: List[int] = []

        for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(skf.split(X, y)):
            X_tr, X_vl = X[train_fold_idx], X[val_fold_idx]
            y_tr, y_vl = y[train_fold_idx], y[val_fold_idx]
            sw = _compute_sample_weights(y_tr)
            train_loader = _make_loader(
                X_tr, y_tr, config["batch_size"], shuffle=True, sample_weights=sw
            )
            val_loader = _make_loader(X_vl, y_vl, config["batch_size"], shuffle=False)

            torch.manual_seed(args.seed + fold_idx)
            model = EDLActionNetworkV3(
                input_dim=input_dim,
                num_classes=NUM_CLASSES,
                hidden_dims=config["hidden_dims"],
                activation=config["activation"],
                dropout=config["dropout"],
            )
            model, history = train_with_early_stopping(
                model, train_loader, val_loader, config, wandb_run=None
            )
            fold_accs.append(history["best_val_acc"])
            fold_epochs.append(history["epochs_trained"])

        cv_mean_acc = float(np.mean(fold_accs))
        cv_std_acc = float(np.std(fold_accs))
        mean_epochs = float(np.mean(fold_epochs))

        result_row = {
            **{k: (str(v) if isinstance(v, list) else v) for k, v in combo.items()},
            "cv_mean_acc": round(cv_mean_acc, 4),
            "cv_std_acc": round(cv_std_acc, 4),
            "cv_min_acc": round(float(np.min(fold_accs)), 4),
            "cv_max_acc": round(float(np.max(fold_accs)), 4),
            "mean_epochs_trained": round(mean_epochs, 1),
        }
        results.append(result_row)

        if not args.no_wandb:
            try:
                import wandb

                cs = _combo_str(combo)[:50]
                wrun = wandb.init(
                    project=args.wandb_project,
                    name=f"phase{phase_num}_{cs}",
                    tags=[f"phase_{phase_num}", "hp_search"],
                    config={
                        **config,
                        **{f"fold_{i}_acc": a for i, a in enumerate(fold_accs)},
                    },
                    reinit=True,
                )
                wandb.log(
                    {
                        "cv_mean_acc": cv_mean_acc,
                        "cv_std_acc": cv_std_acc,
                        "mean_epochs_trained": mean_epochs,
                    }
                )
                if wrun and hasattr(wrun, "url") and wrun.url:
                    wandb_urls.append(wrun.url)
                wandb.finish()
            except Exception as exc:
                logger.warning("W&B logging failed for combo %s: %s", combo, exc)

        logger.info(
            "  Phase %s [%d/%d] %s | cv_acc=%.3f+-%.3f | epochs=%.0f",
            phase_num,
            combo_idx + 1,
            n_combos,
            _combo_str(combo),
            cv_mean_acc,
            cv_std_acc,
            mean_epochs,
        )

    df = (
        pd.DataFrame(results)
        .sort_values("cv_mean_acc", ascending=False)
        .reset_index(drop=True)
    )
    df.to_csv(hp_search_dir / csv_filename, index=False)
    logger.info(
        "Phase %s complete. Best cv_acc=%.3f",
        phase_num,
        df.iloc[0]["cv_mean_acc"],
    )
    return df


# =============================================================================
# Plot Generation
# =============================================================================
def _generate_plots(
    df_phase1: Optional[pd.DataFrame],
    df_phase2: Optional[pd.DataFrame],
    df_phase3: Optional[pd.DataFrame],
    fold_results: List[dict],
    fold_histories: List[dict],
    test_results: Dict[str, dict],
    plots_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate diagnostic plots. Skips on errors."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — plots skipped")
        return

    # 1. Phase 1 heatmap
    if df_phase1 is not None and "activation" in df_phase1.columns:
        try:
            pivot = df_phase1.pivot_table(
                values="cv_mean_acc",
                index="hidden_dims",
                columns="activation",
                aggfunc="max",
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    v = pivot.values[i, j]
                    if not np.isnan(v):
                        ax.text(
                            j,
                            i,
                            f"{v:.3f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="black" if v > 0.7 else "white",
                        )
            plt.colorbar(im, ax=ax)
            ax.set_title("Phase 1: CV Accuracy — Activation x Architecture")
            plt.tight_layout()
            plt.savefig(plots_dir / "hp_search_phase1_heatmap.png", dpi=120)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Phase 1 plot failed: %s", exc)

    # 2. Phase 2 grid
    if df_phase2 is not None and "lr" in df_phase2.columns:
        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            colors = ["steelblue", "tomato", "seagreen", "orange"]
            if "weight_decay" in df_phase2.columns:
                for i, wd in enumerate(sorted(df_phase2["weight_decay"].unique())):
                    sub = df_phase2[df_phase2["weight_decay"] == wd]
                    ax.scatter(
                        sub["lr"],
                        sub["cv_mean_acc"],
                        label=f"wd={wd}",
                        color=colors[i % len(colors)],
                        s=50,
                    )
            else:
                ax.scatter(df_phase2["lr"], df_phase2["cv_mean_acc"], s=50)
            ax.set_xscale("log")
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("CV Mean Accuracy")
            ax.set_title("Phase 2: LR vs CV Accuracy by Weight Decay")
            ax.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "hp_search_phase2_grid.png", dpi=120)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Phase 2 plot failed: %s", exc)

    # 3. Phase 3 heatmap
    if df_phase3 is not None and "dropout" in df_phase3.columns:
        try:
            pivot3 = df_phase3.pivot_table(
                values="cv_mean_acc",
                index="dropout",
                columns="kl_lambda_max",
                aggfunc="max",
            )
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(pivot3.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(pivot3.columns)))
            ax.set_xticklabels([str(x) for x in pivot3.columns])
            ax.set_yticks(range(len(pivot3.index)))
            ax.set_yticklabels([str(x) for x in pivot3.index])
            for i in range(len(pivot3.index)):
                for j in range(len(pivot3.columns)):
                    v = pivot3.values[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9)
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("KL Lambda Max")
            ax.set_ylabel("Dropout")
            ax.set_title("Phase 3: CV Accuracy — Dropout x KL Lambda")
            plt.tight_layout()
            plt.savefig(plots_dir / "hp_search_phase3_grid.png", dpi=120)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Phase 3 plot failed: %s", exc)

    # 4. CV per-fold
    if fold_results:
        try:
            folds = [r["fold"] for r in fold_results]
            accs = [r["best_val_acc"] for r in fold_results]
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(folds, accs, color="steelblue", alpha=0.8)
            ax.axhline(
                np.mean(accs),
                color="red",
                linestyle="--",
                label=f"mean={np.mean(accs):.3f}",
            )
            ax.set_xlabel("Fold")
            ax.set_ylabel("Best Val Accuracy")
            ax.set_title("Phase 4a: Per-Fold Validation Accuracy")
            ax.set_xticks(folds)
            ax.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "cv_per_fold_metrics.png", dpi=120)
            plt.close(fig)
        except Exception as exc:
            logger.warning("CV fold plot failed: %s", exc)

    # 5. Training curves
    if fold_histories:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            for i, h in enumerate(fold_histories):
                ep = range(len(h["val_loss"]))
                ax1.plot(ep, h["val_loss"], alpha=0.6, label=f"fold {i}")
                ax2.plot(ep, h["val_acc"], alpha=0.6, label=f"fold {i}")
            ax1.set_title("Validation Loss per Fold")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Val Loss")
            ax2.set_title("Validation Accuracy per Fold")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Val Acc")
            plt.suptitle("Phase 4a: Final Training Curves")
            plt.tight_layout()
            plt.savefig(plots_dir / "final_training_curves.png", dpi=120)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Training curves plot failed: %s", exc)

    # 6. LR schedule
    if fold_histories:
        try:
            fig, ax = plt.subplots(figsize=(9, 4))
            for i, h in enumerate(fold_histories):
                ax.plot(range(len(h["lr"])), h["lr"], alpha=0.6, label=f"fold {i}")
            ax.set_title("Learning Rate Schedule per Fold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("LR")
            ax.set_yscale("log")
            plt.tight_layout()
            plt.savefig(plots_dir / "final_lr_schedule.png", dpi=120)
            plt.close(fig)
        except Exception as exc:
            logger.warning("LR schedule plot failed: %s", exc)

    # 7. Test CM (ensemble)
    if "ensemble_10fold" in test_results:
        try:
            cm = test_results["ensemble_10fold"]["confusion_matrix"]
            cm_arr = np.array(cm, dtype=np.float32)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm_arr, cmap="Blues")
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
                        str(int(cm_arr[i, j])),
                        ha="center",
                        va="center",
                        color="white" if cm_arr[i, j] > cm_arr.max() * 0.5 else "black",
                    )
            plt.colorbar(im, ax=ax)
            ax.set_title("Test Confusion Matrix (Ensemble)")
            plt.tight_layout()
            plt.savefig(plots_dir / "test_confusion_matrix.png", dpi=120)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Test CM plot failed: %s", exc)

    # 8. Uncertainty distribution
    if "ensemble_10fold" in test_results:
        try:
            vacuity = test_results["ensemble_10fold"]["_vacuity"]
            preds = test_results["ensemble_10fold"]["_preds"]
            fig, ax = plt.subplots(figsize=(9, 5))
            colors_c = ["steelblue", "seagreen", "tomato"]
            for cls_id, (cls_name, color) in enumerate(zip(ACTION_CLASSES, colors_c)):
                mask = preds == cls_id
                vals = vacuity[mask]
                if vals.size > 0:
                    ax.hist(
                        vals,
                        bins=15,
                        alpha=0.6,
                        color=color,
                        label=f"{cls_name} (n={vals.size})",
                    )
            ax.set_title("Uncertainty (Vacuity) by Predicted Class — Ensemble")
            ax.set_xlabel("Vacuity (K/S)")
            ax.set_ylabel("Frequency")
            ax.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "test_uncertainty_distribution.png", dpi=120)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Uncertainty plot failed: %s", exc)

    # 9. Approach comparison
    if test_results:
        try:
            approaches = list(test_results.keys())
            accs = [test_results[a]["accuracy"] for a in approaches]
            briers = [test_results[a]["brier_score"] for a in approaches]
            vacuities = [test_results[a]["mean_vacuity"] for a in approaches]
            x = np.arange(len(approaches))
            width = 0.25
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x - width, accs, width, label="Accuracy", color="steelblue")
            ax.bar(
                x,
                [1 - b for b in briers],
                width,
                label="1-Brier",
                color="seagreen",
                alpha=0.8,
            )
            ax.bar(
                x + width,
                [1 - v for v in vacuities],
                width,
                label="1-Vacuity",
                color="tomato",
                alpha=0.8,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(approaches, rotation=15, ha="right")
            ax.set_ylabel("Score (higher = better)")
            ax.set_title("Test Set: 3-Approach Comparison")
            ax.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "test_approach_comparison.png", dpi=120)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Approach comparison plot failed: %s", exc)

    logger.info("Plots written to: %s", plots_dir)


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase B.3 v3 — EDL-A Production Training with Nested K-fold HP Search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-b2-run-id", default="auto")
    p.add_argument(
        "--search-mode",
        choices=["full", "quick", "phase1", "phase2", "phase3", "phase4"],
        default="full",
    )
    p.add_argument("--test-split", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=3000)
    p.add_argument("--hp-search-patience", type=int, default=20)
    p.add_argument("--final-patience", type=int, default=50)
    p.add_argument("--lr-scheduler-patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--wandb-project", default="d-iqn-dss-edl-thesis")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    args = parse_args()

    run_paths = create_run_paths("d_iqn_dss_edl_action_training_v3_phase3_4_resume")
    logger = setup_run_logger(run_paths)
    run_start = datetime.now()

    if args.search_mode == "quick":
        grids = QUICK_GRIDS
    else:
        grids = FULL_GRIDS

    n_folds = args.cv_folds if args.cv_folds is not None else grids["cv_folds"]
    hp_patience = args.hp_search_patience
    final_patience = args.final_patience

    logger.info(
        "=== Phase B.3 v3 EDL-A Nested K-fold HP Search + Production Training ==="
    )
    logger.info("Search mode:    %s", args.search_mode)
    logger.info("CV folds:       %d", n_folds)
    logger.info("Max epochs:     %d (ML decides via early stopping)", args.max_epochs)
    logger.info("HP patience:    %d", hp_patience)
    logger.info("Final patience: %d", final_patience)
    logger.info("W&B:            %s", "disabled" if args.no_wandb else "enabled")
    logger.info("Run dir:        %s", run_paths.run_directory)

    # W&B check
    if not args.no_wandb:
        if not check_wandb_setup(args.wandb_project):
            logger.error("W&B not configured. Use --no-wandb to run offline.")
            return 1

    hp_search_dir = run_paths.run_directory / "hp_search"
    hp_search_dir.mkdir(parents=True, exist_ok=True)

    wandb_urls: List[str] = []

    # -------------------------------------------------------------------------
    # Locate Phase B.2 run
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Load and preprocess data
    # -------------------------------------------------------------------------
    logger.info("Loading B.2 labels: %s", labels_csv.name)
    df = pd.read_csv(labels_csv)
    logger.info("Loaded: %d rows, %d cols", len(df), len(df.columns))

    if "edl_a_cf_label_available" in df.columns:
        n_before = len(df)
        df = df[df["edl_a_cf_label_available"] == True].reset_index(drop=True)
        dropped = n_before - len(df)
        if dropped:
            logger.warning("Dropped %d rows with unavailable labels", dropped)

    if "edl_a_cf_label" not in df.columns:
        logger.error("Column 'edl_a_cf_label' not found.")
        return 1

    unknown = set(df["edl_a_cf_label"].unique()) - set(CLASS_TO_ID)
    if unknown:
        logger.error("Unknown labels: %s", unknown)
        return 1

    y_all = df["edl_a_cf_label"].map(CLASS_TO_ID).values.astype(np.int64)

    feature_cols = _infer_feature_cols_b2(df)
    if not feature_cols:
        logger.error("No feature columns found.")
        return 1
    logger.info("Inferred %d feature columns", len(feature_cols))

    X_all = df[feature_cols].values.astype(np.float32)
    n_nan = int(np.isnan(X_all).sum())
    if n_nan:
        logger.warning("NaN in features: %d cells — replacing with 0.0", n_nan)
        X_all = np.nan_to_num(X_all, nan=0.0)

    input_dim = X_all.shape[1]

    all_counts = np.bincount(y_all, minlength=NUM_CLASSES)
    all_dist = {ACTION_CLASSES[i]: int(all_counts[i]) for i in range(NUM_CLASSES)}
    logger.info("Full dataset: %d rows | Label dist: %s", len(df), all_dist)

    # -------------------------------------------------------------------------
    # Outer Train*/Test split (stratified 80/20)
    # -------------------------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    X_train_star, X_test, y_train_star, y_test = train_test_split(
        X_all,
        y_all,
        test_size=args.test_split,
        stratify=y_all,
        random_state=args.seed,
    )
    logger.info("Outer split: Train*=%d, Test=%d", len(X_train_star), len(X_test))

    # Standardize using Train* stats only
    feat_mean = X_train_star.mean(axis=0)
    feat_std = X_train_star.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    X_train_star = ((X_train_star - feat_mean) / feat_std).astype(np.float32)
    X_test = ((X_test - feat_mean) / feat_std).astype(np.float32)
    logger.info("Standardization: train* mean/std applied, no test leakage.")

    ts_counts = np.bincount(y_train_star, minlength=NUM_CLASSES)
    te_counts = np.bincount(y_test, minlength=NUM_CLASSES)
    logger.info(
        "Train* dist: %s",
        {ACTION_CLASSES[i]: int(ts_counts[i]) for i in range(NUM_CLASSES)},
    )
    logger.info(
        "Test dist:   %s",
        {ACTION_CLASSES[i]: int(te_counts[i]) for i in range(NUM_CLASSES)},
    )

    # Save config + split info
    training_cfg = {
        "source_b2_run_id": b2_run_dir.name,
        "search_mode": args.search_mode,
        "total_rows": len(df),
        "train_star_rows": len(X_train_star),
        "test_rows": len(X_test),
        "test_split": args.test_split,
        "cv_folds": n_folds,
        "max_epochs": args.max_epochs,
        "hp_search_patience": hp_patience,
        "final_patience": final_patience,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "seed": args.seed,
        "num_classes": NUM_CLASSES,
        "action_classes": ACTION_CLASSES,
        "input_dim": input_dim,
        "feature_columns": feature_cols,
        "all_label_distribution": all_dist,
        "run_start": run_start.isoformat(),
    }
    (run_paths.config_directory / "training_config.json").write_text(
        json.dumps(training_cfg, indent=2), encoding="utf-8"
    )

    split_records = [
        {"split": "train_star", "y": int(y), "label": ACTION_CLASSES[int(y)]}
        for y in y_train_star
    ] + [
        {"split": "test", "y": int(y), "label": ACTION_CLASSES[int(y)]} for y in y_test
    ]
    pd.DataFrame(split_records).to_csv(
        run_paths.data_directory / "train_test_split.csv", index=False
    )

    # CV fold assignments
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    fold_assignments = np.full(len(X_train_star), -1, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(X_train_star, y_train_star)):
        fold_assignments[val_idx] = fold_idx
    pd.DataFrame(
        {
            "train_star_idx": range(len(X_train_star)),
            "y": y_train_star,
            "label": [ACTION_CLASSES[y] for y in y_train_star],
            "fold": fold_assignments,
        }
    ).to_csv(run_paths.data_directory / "cv_fold_assignments.csv", index=False)

    # -------------------------------------------------------------------------
    # HP Search Phase 1: Optimizer × Activation × Architecture
    # -------------------------------------------------------------------------
    defaults_p1 = {
        **_DEFAULTS_PHASE_1,
        "max_epochs": args.max_epochs,
        "early_stop_patience": hp_patience,
        "lr_scheduler_patience": args.lr_scheduler_patience,
    }

    logger.info(
        "--- HP SEARCH Phase 1 SKIPPED (resume mode: using hardcoded best from prior interrupted run) ---"
    )
    df_phase1 = None
    best1_optimizer = "Adam"
    best1_activation = "ReLU"
    best1_hidden_dims = [256, 128]
    _phase1_prior = {
        "optimizer": best1_optimizer,
        "activation": best1_activation,
        "hidden_dims": best1_hidden_dims,
        "cv_mean_acc": 0.530,
        "cv_std_acc": 0.088,
        "mean_epochs_trained": 42,
        "source": "prior interrupted full-mode run (36 combos x 10 folds = 360 runs)",
    }
    logger.info(
        "Phase 1 hardcoded: optimizer=%s, activation=%s, hidden=%s (prior cv_acc=%.3f +- %.3f)",
        best1_optimizer,
        best1_activation,
        best1_hidden_dims,
        _phase1_prior["cv_mean_acc"],
        _phase1_prior["cv_std_acc"],
    )

    # -------------------------------------------------------------------------
    # HP Search Phase 2: LR × WD × Batch (+ Momentum if SGD)
    # -------------------------------------------------------------------------
    defaults_p2 = {
        **defaults_p1,
        "optimizer": best1_optimizer,
        "activation": best1_activation,
        "hidden_dims": best1_hidden_dims,
    }

    logger.info(
        "--- HP SEARCH Phase 2 SKIPPED (resume mode: using hardcoded best from prior interrupted run) ---"
    )
    best2_lr = 0.0005
    best2_wd = 0.001
    best2_bs = 32
    best2_momentum = 0.9
    df_phase2_combined = None
    _phase2_prior = {
        "lr": best2_lr,
        "weight_decay": best2_wd,
        "batch_size": best2_bs,
        "momentum": best2_momentum,
        "cv_mean_acc": 0.534,
        "cv_std_acc": 0.090,
        "mean_epochs_trained": 56,
        "source": "prior interrupted full-mode run (48 combos x 10 folds = 480 runs)",
        "phase2b_skipped_reason": "optimizer is Adam, not SGD",
    }
    logger.info(
        "Phase 2 hardcoded: lr=%.4f, wd=%.4f, bs=%d, momentum=%.2f (prior cv_acc=%.3f +- %.3f)",
        best2_lr,
        best2_wd,
        best2_bs,
        best2_momentum,
        _phase2_prior["cv_mean_acc"],
        _phase2_prior["cv_std_acc"],
    )

    # Provenance: persist the hardcoded Phase 1 + Phase 2 picks for traceability.
    (hp_search_dir / "phase1_phase2_hardcoded_from_prior_run.json").write_text(
        json.dumps({"phase1": _phase1_prior, "phase2": _phase2_prior}, indent=2),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # HP Search Phase 3: Dropout × KL lambda
    # -------------------------------------------------------------------------
    defaults_p3 = {
        **defaults_p2,
        "lr": best2_lr,
        "weight_decay": best2_wd,
        "batch_size": best2_bs,
        "momentum": best2_momentum,
    }

    logger.info("--- HP SEARCH Phase 3 (Dropout x KL Lambda) ---")
    df_phase3 = run_hp_phase(
        phase_num=3,
        grid=grids["phase3"],
        defaults=defaults_p3,
        X=X_train_star,
        y=y_train_star,
        skf=skf,
        input_dim=input_dim,
        args=args,
        hp_search_dir=hp_search_dir,
        csv_filename="phase3_regularization.csv",
        logger=logger,
        wandb_urls=wandb_urls,
    )

    best3 = df_phase3.iloc[0]
    best3_dropout = float(best3["dropout"])
    best3_kl_lambda = float(best3["kl_lambda_max"])

    logger.info(
        "Phase 3 best: dropout=%.2f, kl_lambda_max=%.3f -> cv_acc=%.3f",
        best3_dropout,
        best3_kl_lambda,
        best3["cv_mean_acc"],
    )

    if args.search_mode == "phase3":
        logger.info("search-mode=phase3: stopping after Phase 3.")
        return 0

    # -------------------------------------------------------------------------
    # Best config (combining all phases)
    # -------------------------------------------------------------------------
    best_config = {
        "optimizer": best1_optimizer,
        "activation": best1_activation,
        "hidden_dims": best1_hidden_dims,
        "lr": best2_lr,
        "weight_decay": best2_wd,
        "batch_size": best2_bs,
        "momentum": best2_momentum,
        "dropout": best3_dropout,
        "kl_lambda_max": best3_kl_lambda,
        "max_epochs": args.max_epochs,
        "early_stop_patience": final_patience,
        "lr_scheduler_patience": args.lr_scheduler_patience,
    }
    (hp_search_dir / "best_config.json").write_text(
        json.dumps(best_config, indent=2), encoding="utf-8"
    )
    logger.info("Best config saved: %s", best_config)

    # -------------------------------------------------------------------------
    # Phase 4a: K-fold ensemble training on Train*
    # -------------------------------------------------------------------------
    logger.info("--- Phase 4a: %d-fold ensemble training ---", n_folds)
    fold_models: List[EDLActionNetworkV3] = []
    fold_results: List[dict] = []
    fold_histories: List[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(X_train_star, y_train_star)
    ):
        logger.info("  Fold %d/%d training...", fold_idx + 1, n_folds)
        X_tr, X_vl = X_train_star[train_idx], X_train_star[val_idx]
        y_tr, y_vl = y_train_star[train_idx], y_train_star[val_idx]
        sw = _compute_sample_weights(y_tr)

        train_loader = _make_loader(
            X_tr, y_tr, best_config["batch_size"], shuffle=True, sample_weights=sw
        )
        val_loader = _make_loader(X_vl, y_vl, best_config["batch_size"], shuffle=False)

        torch.manual_seed(args.seed + fold_idx)
        model = EDLActionNetworkV3(
            input_dim=input_dim,
            num_classes=NUM_CLASSES,
            hidden_dims=best_config["hidden_dims"],
            activation=best_config["activation"],
            dropout=best_config["dropout"],
        )

        wandb_run = None
        if not args.no_wandb:
            try:
                import wandb

                wandb_run = wandb.init(
                    project=args.wandb_project,
                    name=f"phase4a_fold_{fold_idx}",
                    tags=["phase_4a", "ensemble_fold"],
                    config={**best_config, "fold": fold_idx},
                    reinit=True,
                )
            except Exception as exc:
                logger.warning("W&B init failed: %s", exc)

        model, history = train_with_early_stopping(
            model, train_loader, val_loader, best_config, wandb_run=wandb_run
        )

        if wandb_run is not None:
            try:
                import wandb

                wandb.log(
                    {
                        "fold_best_val_acc": history["best_val_acc"],
                        "fold_best_val_loss": history["best_val_loss"],
                        "fold_epochs_trained": history["epochs_trained"],
                    }
                )
                if hasattr(wandb_run, "url") and wandb_run.url:
                    wandb_urls.append(wandb_run.url)
                wandb.finish()
            except Exception:
                pass

        fold_models.append(model)
        fold_results.append(
            {
                "fold": fold_idx,
                "best_val_loss": history["best_val_loss"],
                "best_val_acc": history["best_val_acc"],
                "epochs_trained": history["epochs_trained"],
            }
        )
        fold_histories.append(history)

        torch.save(
            model.state_dict(),
            run_paths.models_directory / f"edl_action_classifier_v3_fold_{fold_idx}.pt",
        )
        pd.DataFrame(
            {
                "epoch": range(len(history["train_loss"])),
                "train_loss": history["train_loss"],
                "val_loss": history["val_loss"],
                "val_acc": history["val_acc"],
                "lr": history["lr"],
            }
        ).to_csv(
            run_paths.metrics_directory
            / f"phase4a_per_epoch_metrics_fold_{fold_idx}.csv",
            index=False,
        )

        logger.info(
            "  Fold %d: best_val_acc=%.3f, epochs=%d",
            fold_idx,
            history["best_val_acc"],
            history["epochs_trained"],
        )

    pd.DataFrame(fold_results).to_csv(
        run_paths.metrics_directory / "phase4a_per_fold_metrics.csv", index=False
    )

    # Best single fold
    best_fold_idx = max(
        range(len(fold_results)), key=lambda i: fold_results[i]["best_val_acc"]
    )
    torch.save(
        fold_models[best_fold_idx].state_dict(),
        run_paths.models_directory / "edl_action_classifier_v3.pt",
    )
    logger.info(
        "Best single fold: %d (val_acc=%.3f)",
        best_fold_idx,
        fold_results[best_fold_idx]["best_val_acc"],
    )

    # Ensemble artifact
    torch.save(
        {
            "fold_state_dicts": [m.state_dict() for m in fold_models],
            "config": best_config,
            "fold_results": fold_results,
        },
        run_paths.models_directory / "edl_action_classifier_v3_ensemble.pt",
    )

    cv_mean = np.mean([r["best_val_acc"] for r in fold_results])
    cv_std = np.std([r["best_val_acc"] for r in fold_results])
    logger.info("Phase 4a complete: CV val_acc = %.3f +- %.3f", cv_mean, cv_std)

    # -------------------------------------------------------------------------
    # Phase 4b: Single final model on Train* (inner 80/20 split)
    # -------------------------------------------------------------------------
    logger.info("--- Phase 4b: Single final model on Train* (80/20 inner split) ---")
    X_tr_final, X_vl_final, y_tr_final, y_vl_final = train_test_split(
        X_train_star,
        y_train_star,
        test_size=0.20,
        stratify=y_train_star,
        random_state=args.seed,
    )
    logger.info("Phase 4b split: train=%d, val=%d", len(X_tr_final), len(X_vl_final))

    sw_final = _compute_sample_weights(y_tr_final)
    train_loader_final = _make_loader(
        X_tr_final,
        y_tr_final,
        best_config["batch_size"],
        shuffle=True,
        sample_weights=sw_final,
    )
    val_loader_final = _make_loader(
        X_vl_final, y_vl_final, best_config["batch_size"], shuffle=False
    )

    torch.manual_seed(args.seed)
    final_model = EDLActionNetworkV3(
        input_dim=input_dim,
        num_classes=NUM_CLASSES,
        hidden_dims=best_config["hidden_dims"],
        activation=best_config["activation"],
        dropout=best_config["dropout"],
    )

    final_wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            final_wandb_run = wandb.init(
                project=args.wandb_project,
                name="phase4b_single_final",
                tags=["phase_4b", "single_final"],
                config={
                    **best_config,
                    "training_approach": "single_on_full_train_star",
                },
                reinit=True,
            )
        except Exception as exc:
            logger.warning("W&B init failed: %s", exc)

    final_model, final_history = train_with_early_stopping(
        final_model,
        train_loader_final,
        val_loader_final,
        best_config,
        wandb_run=final_wandb_run,
    )

    if final_wandb_run is not None:
        try:
            import wandb

            wandb.log(
                {
                    "final_best_val_acc": final_history["best_val_acc"],
                    "final_best_val_loss": final_history["best_val_loss"],
                    "final_epochs_trained": final_history["epochs_trained"],
                }
            )
            if hasattr(final_wandb_run, "url") and final_wandb_run.url:
                wandb_urls.append(final_wandb_run.url)
            wandb.finish()
        except Exception:
            pass

    torch.save(
        final_model.state_dict(),
        run_paths.models_directory / "edl_action_classifier_v3_final_single.pt",
    )

    pd.DataFrame(
        {
            "epoch": range(len(final_history["train_loss"])),
            "train_loss": final_history["train_loss"],
            "val_loss": final_history["val_loss"],
            "val_acc": final_history["val_acc"],
            "lr": final_history["lr"],
        }
    ).to_csv(
        run_paths.metrics_directory / "phase4b_single_final_per_epoch.csv", index=False
    )

    logger.info(
        "Phase 4b complete: final_val_acc=%.3f, epochs=%d",
        final_history["best_val_acc"],
        final_history["epochs_trained"],
    )

    # -------------------------------------------------------------------------
    # Phase 4c: Test evaluation — 3 approaches
    # -------------------------------------------------------------------------
    logger.info("--- Phase 4c: Test evaluation (3 approaches) ---")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    test_results: Dict[str, dict] = {}

    # 1. Single best fold (from Phase 4a)
    test_results["single_best_fold"] = evaluate_model(
        fold_models[best_fold_idx], X_test_tensor, y_test
    )

    # 2. Ensemble (10-fold)
    test_results["ensemble_10fold"] = evaluate_model(fold_models, X_test_tensor, y_test)

    # 3. Single final (from Phase 4b)
    test_results["single_final_full_trainstar"] = evaluate_model(
        final_model, X_test_tensor, y_test
    )

    # Save predictions for all 3
    save_test_predictions(
        fold_models[best_fold_idx],
        X_test_tensor,
        y_test,
        run_paths.audit_directory / "edl_v3_test_predictions_single_best_fold.csv",
    )
    save_test_predictions(
        fold_models,
        X_test_tensor,
        y_test,
        run_paths.audit_directory / "edl_v3_test_predictions_ensemble.csv",
    )
    save_test_predictions(
        final_model,
        X_test_tensor,
        y_test,
        run_paths.audit_directory / "edl_v3_test_predictions_single_final.csv",
    )

    # Save comparison + confusion matrices
    comparison_rows = []
    for approach_name, result in test_results.items():
        row = {"approach": approach_name}
        for k, v in result.items():
            if k.startswith("_") or k in ("per_class_metrics", "confusion_matrix"):
                continue
            row[k] = v
        comparison_rows.append(row)

        cm_df = pd.DataFrame(
            result["confusion_matrix"],
            index=[f"true_{c}" for c in ACTION_CLASSES],
            columns=[f"pred_{c}" for c in ACTION_CLASSES],
        )
        cm_df.to_csv(
            run_paths.audit_directory
            / f"edl_v3_test_confusion_matrix_{approach_name}.csv"
        )

    pd.DataFrame(comparison_rows).to_csv(
        run_paths.audit_directory / "edl_v3_test_approach_comparison.csv", index=False
    )

    # Log comparison summary
    logger.info("=== TEST SET COMPARISON ===")
    for approach_name, result in test_results.items():
        logger.info(
            "  %s: acc=%.3f, brier=%.3f, vacuity=%.3f, classes=%d",
            approach_name,
            result["accuracy"],
            result["brier_score"],
            result["mean_vacuity"],
            result["n_classes_predicted"],
        )

    # W&B comparison run
    if not args.no_wandb:
        try:
            import wandb

            cmp_run = wandb.init(
                project=args.wandb_project,
                name="phase4c_test_comparison",
                tags=["phase_4c", "test_evaluation"],
                config=best_config,
                reinit=True,
            )
            for approach_name, result in test_results.items():
                for metric_name, value in result.items():
                    if isinstance(value, (int, float)):
                        wandb.log({f"{approach_name}_{metric_name}": value})
            if hasattr(cmp_run, "url") and cmp_run.url:
                wandb_urls.append(cmp_run.url)
            wandb.finish()
        except Exception as exc:
            logger.warning("W&B comparison logging failed: %s", exc)

    # -------------------------------------------------------------------------
    # Plots + summary
    # -------------------------------------------------------------------------
    _generate_plots(
        df_phase1,
        df_phase2_combined,
        df_phase3,
        fold_results,
        fold_histories,
        test_results,
        run_paths.plots_directory,
        logger,
    )

    # Summary
    duration = (datetime.now() - run_start).total_seconds()
    summary = {
        "run_id": run_paths.run_directory.name,
        "search_mode": args.search_mode,
        "duration_seconds": round(duration, 1),
        "duration_minutes": round(duration / 60, 1),
        "source_b2_run_id": b2_run_dir.name,
        "best_config": best_config,
        "phase_4a_cv_mean_acc": round(float(cv_mean), 4),
        "phase_4a_cv_std_acc": round(float(cv_std), 4),
        "phase_4a_best_fold": best_fold_idx,
        "phase_4b_val_acc": round(final_history["best_val_acc"], 4),
        "phase_4b_epochs": final_history["epochs_trained"],
        "test_results": {
            name: {k: v for k, v in r.items() if not k.startswith("_")}
            for name, r in test_results.items()
        },
        "n_wandb_runs": len(wandb_urls),
    }
    (run_paths.summary_directory / "edl_v3_training_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # Markdown summary
    md_lines = [
        "# Phase B.3 v3 — EDL-A Training Summary",
        "",
        f"**Run ID:** `{run_paths.run_directory.name}`",
        f"**Search mode:** {args.search_mode}",
        f"**Duration:** {duration/60:.1f} min",
        f"**Source B.2:** `{b2_run_dir.name}`",
        "",
        "## Best Configuration",
        "```json",
        json.dumps(best_config, indent=2),
        "```",
        "",
        f"## Phase 4a — {n_folds}-fold Ensemble Results",
        f"- CV mean val_acc: **{cv_mean:.3f} ± {cv_std:.3f}**",
        f"- Best single fold: **fold {best_fold_idx}** (val_acc={fold_results[best_fold_idx]['best_val_acc']:.3f})",
        "",
        "## Phase 4b — Single Final Model",
        f"- Final val_acc: **{final_history['best_val_acc']:.3f}**",
        f"- Epochs trained: {final_history['epochs_trained']}",
        "",
        "## Phase 4c — Test Set Comparison (3 Approaches)",
        "",
        "| Approach | Accuracy | Brier | Vacuity | Classes |",
        "|----------|----------|-------|---------|---------|",
    ]
    for name, r in test_results.items():
        md_lines.append(
            f"| {name} | {r['accuracy']:.3f} | {r['brier_score']:.3f} | "
            f"{r['mean_vacuity']:.3f} | {r['n_classes_predicted']} |"
        )

    (run_paths.summary_directory / "edl_v3_training_summary.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )

    # W&B URLs
    if wandb_urls:
        (run_paths.summary_directory / "wandb_run_urls.txt").write_text(
            "\n".join(wandb_urls), encoding="utf-8"
        )
    else:
        (run_paths.summary_directory / "wandb_run_urls.txt").write_text(
            "(W&B disabled or no URLs collected)\n", encoding="utf-8"
        )

    logger.info("=== Phase B.3 v3 complete ===")
    logger.info("Run dir:     %s", run_paths.run_directory)
    logger.info("Duration:    %.1f minutes", duration / 60)
    logger.info("Best config: %s", best_config)
    logger.info("CV val_acc:  %.3f +- %.3f", cv_mean, cv_std)
    for name, r in test_results.items():
        logger.info("Test %s: acc=%.3f", name, r["accuracy"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
