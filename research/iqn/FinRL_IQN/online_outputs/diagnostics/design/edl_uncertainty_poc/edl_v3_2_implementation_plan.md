# EDL v3.2 — Implementation Plan

**Status:** Design only — Phase 2 pending user approval  
**Phase 1 complete:** Audit + design docs created  
**Phase 2:** Implement source files (requires explicit approval)  

---

## 1. Current State After Phase 1

### Existing v3.1 files (NOT changed, NOT deleted)

| File | v3.1 Status |
|------|------------|
| `src/stock_investment_dss/uncertainty/__init__.py` | Kept; package init |
| `src/stock_investment_dss/uncertainty/edl_classifier.py` | Deprecated PoC; K=3 LOW/MEDIUM/HIGH |
| `src/stock_investment_dss/uncertainty/recommendation_confidence.py` | Deprecated PoC |
| `src/stock_investment_dss/runner/run_edl_uncertainty_smoke_test.py` | Deprecated PoC runner |

**v3.1 module docstrings must be updated (in Phase 2) to mark as deprecated.**  
Update must NOT break existing behavior — only add deprecation warnings.

### Existing hierarchical policy files (NOT changed, Phase 2 does NOT touch them)

| File | Status |
|------|--------|
| `src/stock_investment_dss/decision/hierarchical_decision_policy.py` | Unchanged |
| `src/stock_investment_dss/decision/ticker_selector.py` | Unchanged |
| `src/stock_investment_dss/decision/size_selector.py` | Unchanged |
| `src/stock_investment_dss/runner/run_hierarchical_policy_smoke_test.py` | Unchanged |

---

## 2. New Files to Create in Phase 2

All new files in `src/stock_investment_dss/uncertainty/` and `src/stock_investment_dss/runner/`.  
**Constraint: Create only. Do NOT overwrite existing files.**

### 2.1 Core EDL modules

#### `src/stock_investment_dss/uncertainty/edl_action_classes.py`

Defines:
- `EDL_ACTION_CLASSES_4 = ["HOLD", "BUY", "SELL", "REBALANCE"]`
- `EDL_ACTION_CLASSES_5 = ["HOLD", "BUY", "SELL", "REBALANCE", "CHANGE_STRATEGY"]`
- `action_to_idx(action: str, include_change_strategy: bool) -> int`
- `idx_to_action(idx: int, include_change_strategy: bool) -> str`
- `get_num_classes(include_change_strategy: bool) -> int` → 4 or 5
- `EDLActionConfig` dataclass:
  ```
  include_change_strategy: bool = False
  num_classes: int = 4
  uncertainty_lambda: float = 0.5
  disagreement_lambda: float = 0.3
  gate_enabled: bool = True
  ```
- `EDLActionConfig.from_env()` classmethod: reads all `STOCK_INVESTMENT_DSS_EDL_*` env vars

#### `src/stock_investment_dss/uncertainty/edl_action_dataset.py`

Defines:
- `EDLActionSample` dataclass: `(feature_vector: np.ndarray, label: int, metadata: dict)`
- `EDLActionDataset(torch.utils.data.Dataset)`:
  - `from_hierarchical_audit_csv(audit_dir, config)` classmethod
  - `from_cached_npy(path)` classmethod
  - `save_npy(path)` method
  - Feature normalization: stored as `StandardScaler` params in dataset
  - Temporal train/val/test split (no random shuffle)
- `RuleBasedLabelPolicy`:
  - `generate_label(features: dict) -> int` for EDL-B
  - Configurable thresholds (YAML/dict)
- `HindsightOracleLabelBuilder`:
  - `build(audit_csv, market_data_csv, horizon_days) -> pd.DataFrame`
  - Requires future price returns per ticker — PIT-safe by construction
- `IQNTeacherLabelExtractor`:
  - `extract(audit_csv) -> pd.DataFrame` — reads `selected_action_type` column
  - Handles REBALANCE detection from portfolio state

#### `src/stock_investment_dss/uncertainty/edl_action_network.py`

Defines (PyTorch):
- `EDLActionNetwork(nn.Module)`:
  ```python
  def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.1):
      ...
  def forward(self, x):
      # logits = MLP(x)
      # evidence = F.softplus(logits)
      # alpha = evidence + 1
      # S = alpha.sum(dim=-1, keepdim=True)
      # prob = alpha / S
      # u = num_classes / S
      return {"evidence": evidence, "alpha": alpha, "prob": prob, "S": S, "u": u}
  ```
- Activation: `nn.SiLU()` in hidden layers (or `nn.ReLU` — configurable)
- Output: `Softplus` (ensures evidence ≥ 0 without hard zero gradient)
- No softmax on output
- Dropout for regularisation

#### `src/stock_investment_dss/uncertainty/edl_losses.py`

Defines:
- `EDLActionLoss(nn.Module)`:
  - `forward(alpha, y_onehot, epoch, total_epochs, kl_anneal_lambda=1.0)`
  - Components:
    ```
    # Expected MSE / Bayes risk
    mse_loss = sum_k [(y_k - p_k)^2 + p_k*(1-p_k)/S]
    
    # KL divergence toward uniform Dirichlet
    # alpha_tilde = y + (1-y)*alpha (remove evidence for true class)
    kl = KL_Dirichlet(alpha_tilde || Dir(1,1,...,1))
    
    # Annealing
    annealing = min(1.0, epoch / (total_epochs * anneal_fraction))
    
    # Total
    loss = mse_loss + annealing * kl_anneal_lambda * kl
    ```
  - This implements Eq. 4 from Sensoy et al. (NeurIPS 2018)
- `kl_divergence_uniform(alpha) -> Tensor` helper
- `dirichlet_mse_loss(alpha, y_onehot) -> Tensor` helper

#### `src/stock_investment_dss/uncertainty/edl_action_classifier.py`

Defines:
- `EDLActionClassifier`:
  - `__init__(config: EDLActionConfig, model_path: str or None)`
  - `load_model(path: str)` — loads PyTorch checkpoint
  - `classify(features: np.ndarray) -> EDLActionResult`
  - `classify_batch(features: np.ndarray) -> list[EDLActionResult]`
- `EDLActionResult` dataclass:
  ```
  evidence: dict[str, float]   # {action: e_k}
  alpha: dict[str, float]      # {action: alpha_k}
  prob: dict[str, float]       # {action: p_k}
  S: float
  vacuity: float
  predicted_action: str
  selected_action: str         # from input context
  selected_action_probability: float
  selected_action_evidence: float
  edl_agrees: bool
  ```
- `EDLActionResult.to_audit_row(decision_row: dict) -> dict` — formats for CSV output

#### `src/stock_investment_dss/uncertainty/edl_ensemble.py`

Defines:
- `EDLEnsemble`:
  - `__init__(classifiers: dict[str, EDLActionClassifier], weights: dict[str, float] or None)`
  - `classify(features: np.ndarray) -> EDLEnsembleResult`
- `EDLEnsembleResult` dataclass:
  ```
  individual: dict[str, EDLActionResult]   # {variant: result}
  p_ensemble: dict[str, float]             # weighted average
  u_ensemble: float                        # weighted average vacuity
  u_conservative: float                    # max(u_A, u_B, u_C)
  model_disagreement_score: float          # fraction that disagree with selected
  ensemble_predicted_action: str
  ensemble_weights: dict[str, float]
  ```
- `compute_disagreement(results: dict[str, EDLActionResult]) -> float`
- `compute_ensemble_probs(results, weights) -> dict[str, float]`

#### `src/stock_investment_dss/uncertainty/edl_gate.py`

Defines:
- `EDLGateConfig` dataclass:
  ```
  uncertainty_review_threshold: float = 0.55
  uncertainty_reduce_threshold: float = 0.35
  uncertainty_force_hold_threshold: float = 0.65
  disagreement_review_threshold: float = 0.50
  rebalance_signal_threshold: float = 0.60
  change_strategy_signal_threshold: float = 0.50
  uncertainty_lambda: float = 0.5
  ```
- `EDLGate`:
  - `__init__(config: EDLGateConfig)`
  - `apply(edl_result, selected_action, original_fraction, vacuity) -> EDLGateResult`
- `EDLGateResult` dataclass:
  ```
  recommendation_gate: str        # RECOMMEND_AS_IS / REDUCE_SIZE / FORCE_HOLD / HUMAN_REVIEW / STRATEGY_REVIEW
  should_reduce_size: bool
  should_force_hold: bool
  should_require_human_review: bool
  final_action: str
  final_fraction: float
  reason_codes: list[str]
  uncertainty_penalty: float
  disagreement_penalty: float
  ```

---

### 2.2 Runners

#### `src/stock_investment_dss/runner/run_edl_action_dataset_builder.py`

- Reads from all hierarchical policy smoke test runs under `outputs/runs/`
- Generates `(features, label)` dataset for the specified label mode
- Writes:
  ```
  outputs/edl_datasets/<timestamp>_edl_dataset_<label_mode>/
    train.npy / val.npy / test.npy
    metadata.json
    feature_names.json
    label_distribution.json
    feature_scaler.pkl
  ```
- Env: `STOCK_INVESTMENT_DSS_EDL_LABEL_MODE`, `STOCK_INVESTMENT_DSS_EDL_HORIZON_DAYS`

#### `src/stock_investment_dss/runner/run_edl_action_training_smoke_test.py`

- Loads dataset from `outputs/edl_datasets/`
- Trains `EDLActionNetwork` for N_EPOCHS_SMOKE (default 50, not heavy training)
- Logs train/val loss per epoch
- Saves checkpoint to `outputs/edl_checkpoints/<timestamp>_edl_<variant>/`
- Writes training curve CSV
- Env: `STOCK_INVESTMENT_DSS_EDL_VARIANT`, `STOCK_INVESTMENT_DSS_EDL_MODEL_PATH`

#### `src/stock_investment_dss/runner/run_edl_action_inference_smoke_test.py`

- Loads trained checkpoint or uses placeholder (rule-based v3.2 init)
- Reads from latest hierarchical policy run
- Runs EDL inference per decision step
- Applies EDL gate
- Writes to `outputs/runs/<timestamp>_d_iqn_dss_edl_action_inference_smoke_test/`
  - `audit/edl_action_uncertainty_by_decision.csv`
  - `summary/edl_action_uncertainty_summary.json`
  - `summary/edl_action_uncertainty_summary.md`

---

### 2.3 Documentation

#### `docs/EDL_Action_Uncertainty_v3_2.md`

Full production documentation (analogous to `EDL_Uncertainty_PoC_v3_1.md`) covering:
- v3.1 → v3.2 migration guide
- Architecture diagram
- Mathematical formulation (Sensoy et al. 2018)
- Configuration reference
- Training guide for A/B/C
- Inference guide
- Ablation protocol
- Thesis alignment

---

## 3. Phase 2 Implementation Order

**Recommended implementation sequence** (minimises inter-dependency):

1. `edl_action_classes.py` — no dependencies; defines all class constants
2. `edl_losses.py` — PyTorch only; no other new module dependencies
3. `edl_action_network.py` — depends on PyTorch only
4. `edl_action_dataset.py` — depends on action_classes
5. `edl_action_classifier.py` — depends on network + action_classes
6. `edl_ensemble.py` — depends on classifier
7. `edl_gate.py` — depends on action_classes; standalone
8. `run_edl_action_dataset_builder.py` — depends on dataset
9. `run_edl_action_training_smoke_test.py` — depends on network + losses + dataset
10. `run_edl_action_inference_smoke_test.py` — depends on classifier + ensemble + gate
11. `docs/EDL_Action_Uncertainty_v3_2.md` — documentation

---

## 4. Phase 2 Constraints (Reminder)

- ✅ Create new files only
- ❌ Do NOT modify IQN core
- ❌ Do NOT modify hierarchical policy source
- ❌ Do NOT overwrite v3.1 uncertainty files without explicit approval
- ✅ Run `py_compile` after each file
- ❌ No heavy training (smoke test only: 50 epochs, synthetic data)
- ❌ No `pip install` (must use already-installed packages only)

---

## 5. Required Packages Check (before Phase 2)

These must be available without installation:

| Package | Used by |
|---------|---------|
| `torch` | edl_action_network.py, edl_losses.py |
| `numpy` | edl_action_dataset.py |
| `pandas` | all runners |
| `sklearn.preprocessing.StandardScaler` | edl_action_dataset.py |
| `json`, `csv`, `pathlib`, `logging` | all (stdlib) |

If `torch` is unavailable, `edl_action_network.py` and `edl_losses.py` must degrade gracefully with a clear `ImportError` message, so that `edl_action_classes.py`, `edl_gate.py` and runners remain importable.

---

## 6. v3.1 Deprecation Strategy

**Do NOT delete v3.1 files.** Instead, in Phase 2:

Add to top of `edl_classifier.py` and `recommendation_confidence.py`:
```python
import warnings
warnings.warn(
    "EDL v3.1 (edl_classifier / recommendation_confidence) is deprecated. "
    "Use edl_action_classifier (v3.2) instead. "
    "v3.1 classified LOW/MEDIUM/HIGH confidence — this is NOT the correct "
    "EDL formulation for D-IQN-DSS. See docs/EDL_Action_Uncertainty_v3_2.md.",
    DeprecationWarning,
    stacklevel=2,
)
```

v3.1 runner `run_edl_uncertainty_smoke_test.py` continues to work without modification.

---

## 7. Smoke Test Acceptance Criteria (Phase 2 exit)

After Phase 2 implementation, the following must pass:

### py_compile (all new Python files)

```powershell
foreach ($f in @(
    ".\src\stock_investment_dss\uncertainty\edl_action_classes.py",
    ".\src\stock_investment_dss\uncertainty\edl_action_dataset.py",
    ".\src\stock_investment_dss\uncertainty\edl_action_network.py",
    ".\src\stock_investment_dss\uncertainty\edl_losses.py",
    ".\src\stock_investment_dss\uncertainty\edl_action_classifier.py",
    ".\src\stock_investment_dss\uncertainty\edl_ensemble.py",
    ".\src\stock_investment_dss\uncertainty\edl_gate.py",
    ".\src\stock_investment_dss\runner\run_edl_action_dataset_builder.py",
    ".\src\stock_investment_dss\runner\run_edl_action_training_smoke_test.py",
    ".\src\stock_investment_dss\runner\run_edl_action_inference_smoke_test.py"
)) { python -m py_compile $f }
```

### Inference smoke test (no training)

```powershell
$env:PYTHONPATH = "src"
$env:STOCK_INVESTMENT_DSS_USE_EDL = "true"
$env:STOCK_INVESTMENT_DSS_EDL_VARIANT = "none"
python -m stock_investment_dss.runner.run_edl_action_inference_smoke_test
```

Expected: runs to completion, writes 3 output files, no training.

### v3.1 still works

```powershell
python -m stock_investment_dss.runner.run_edl_uncertainty_smoke_test
```

Expected: still runs (may emit DeprecationWarning — that is acceptable).
