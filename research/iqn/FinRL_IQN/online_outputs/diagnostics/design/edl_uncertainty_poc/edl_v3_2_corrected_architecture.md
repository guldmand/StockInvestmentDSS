# EDL v3.2 — Corrected Architecture Design

**Status:** Design only — no source code changed  
**Supersedes:** EDL v3.1 (PoC placeholder) for production use  
**v3.1 status:** Deprecated as interface PoC; kept intact; not deleted  

---

## 1. What Was Wrong in v3.1

v3.1 `EDLClassifier` classifies over K=3 confidence labels: LOW / MEDIUM / HIGH.  
This is **not** the correct EDL formulation for D-IQN-DSS.

| Aspect | v3.1 (Wrong) | v3.2 (Correct) |
|--------|-------------|----------------|
| Class space K | 3: LOW/MEDIUM/HIGH | 4 or 5: HOLD/BUY/SELL/REBALANCE[/CHANGE_STRATEGY] |
| What Dirichlet models | "Is this recommendation confident?" | "Which DSS action does the model support?" |
| Primary EDL output | `recommendation_confidence_label` | `edl_predicted_action` + evidence per action |
| Gate logic | Baked into classifier | Separate `EDLGate` module derived post-classification |
| Training target | None (rule-based only) | One-hot over DSS actions from A/B/C label sources |

**The v3.1 confidence label (LOW/MEDIUM/HIGH) can be a derived secondary output** from the v3.2 evidence — e.g., if evidence for the selected action is high → HIGH confidence — but it is not the primary Dirichlet class space.

---

## 2. Correct EDL Formulation

### 2.1 Class Space

**4-class mode** (default, `STOCK_INVESTMENT_DSS_EDL_INCLUDE_CHANGE_STRATEGY=false`):
```
k = 0: HOLD
k = 1: BUY
k = 2: SELL
k = 3: REBALANCE
K = 4
```

**5-class mode** (`STOCK_INVESTMENT_DSS_EDL_INCLUDE_CHANGE_STRATEGY=true`):
```
k = 0: HOLD
k = 1: BUY
k = 2: SELL
k = 3: REBALANCE
k = 4: CHANGE_STRATEGY
K = 5
```

### 2.2 Neural Network Output

```
Input feature vector x (dimensionality d)
  ↓
EDLActionNetwork (MLP with configurable hidden layers)
  ↓
Logits z ∈ ℝ^K  (unbounded)
  ↓
evidence e = Softplus(z)   (ensures e_k >= 0 ∀k)
  ↓
alpha_k = e_k + 1          (Dirichlet concentration)
  ↓
S = Σ_k alpha_k            (Dirichlet strength)
  ↓
p_k = alpha_k / S          (expected class probability)
  ↓
u = K / S                  (vacuity / epistemic uncertainty)
  ↓
b_k = e_k / S              (belief mass per class)
```

No softmax is applied to the network output; probabilities are derived from the Dirichlet parameterisation.

### 2.3 Selected Action Quantities

For a decision where `selected_action_type = a*`:

```
selected_action_idx          = class_index(a*)
selected_action_probability  = p_{a*} = alpha_{a*} / S
selected_action_evidence     = e_{a*} = alpha_{a*} - 1
selected_action_belief       = b_{a*} = e_{a*} / S
selected_action_uncertainty  = u = K / S  (same for all actions — one Dirichlet per decision)
```

### 2.4 EDL Agrees?

```
edl_predicted_action     = argmax_k p_k
edl_agrees_with_selected = (edl_predicted_action == selected_action_type)
```

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  FinRL Data + PIT Features                                       │
│  (technical, fundamental, macro, news quantified)                │
└──────────────────────────────────────────────────────────────────┘
                       ↓ market state s_t
┌──────────────────────────────────────────────────────────────────┐
│  IQN Return Distribution (existing, unchanged)                   │
│  Models P(G | s_t, a) → q10/q25/q50/q75/q90/CVaR per action    │
│  Outputs: action scores, selected_action_type                    │
└──────────────────────────────────────────────────────────────────┘
                       ↓ action_type + IQN features
┌──────────────────────────────────────────────────────────────────┐
│  HierarchicalDecisionPolicy v3.0 (existing, unchanged)           │
│  Stage 1: action type                                            │
│  Stage 2: ticker selection                                       │
│  Stage 3: size selection                                         │
│  Stage 4: risk/strategy validation                               │
│  Stage 5: audit log                                              │
│  Outputs: selected_ticker, selected_size, audit CSVs             │
└──────────────────────────────────────────────────────────────────┘
                       ↓ feature vector from audit + IQN features
┌──────────────────────────────────────────────────────────────────┐
│  EDLActionFeatureBuilder (new)                                   │
│  Constructs input vector from:                                   │
│    Group A: IQN action-distribution features                     │
│    Group B: HierarchicalDecisionPolicy features                  │
│    Group C: Technical/fundamental features                       │
│    Group D: Portfolio/risk/strategy features                     │
└──────────────────────────────────────────────────────────────────┘
                       ↓ feature vector x ∈ ℝ^d
┌──────────────────────────────────────────────────────────────────┐
│  EDLActionNetwork (PyTorch MLP)                                  │
│  Input: x ∈ ℝ^d                                                  │
│  Output: evidence e ∈ ℝ^K (via Softplus)                        │
│  Trained with: EDLActionLoss (MSE + variance + KL_anneal)        │
│  Label sources: EDL-A (hindsight) | EDL-B (rules) | EDL-C (IQN) │
└──────────────────────────────────────────────────────────────────┘
                       ↓ e, alpha, p, u, b per class
┌──────────────────────────────────────────────────────────────────┐
│  EDLEnsemble (optional)                                          │
│  Variants: none | A | B | C | AB | AC | BC | ABC                │
│  Outputs: p_ensemble, u_ensemble, u_conservative,               │
│           model_disagreement_score                               │
└──────────────────────────────────────────────────────────────────┘
                       ↓ ensemble evidence + disagreement
┌──────────────────────────────────────────────────────────────────┐
│  Risk Policy with EDL scoring:                                   │
│                                                                  │
│  Score(a) = IQN_score(a)                                         │
│           - λ_u    * U_EDL(a)                                    │
│           - λ_d    * Disagreement_EDL(a)                         │
│           - λ_cost * Cost(a)                                     │
│           - λ_conc * ConcentrationPenalty(a)                     │
│           - λ_str  * StrategyMismatch(a)                         │
│                                                                  │
│  IQN_score ∈ {q50, mean, q50 − CVaR_penalty}  (config)          │
└──────────────────────────────────────────────────────────────────┘
                       ↓ final adjusted scores
┌──────────────────────────────────────────────────────────────────┐
│  EDLGate (new)                                                   │
│  Derived gate outputs:                                           │
│    RECOMMEND_AS_IS                                               │
│    REDUCE_SIZE                                                   │
│    FORCE_HOLD                                                     │
│    HUMAN_REVIEW                                                  │
│    STRATEGY_REVIEW                                               │
└──────────────────────────────────────────────────────────────────┘
                       ↓ final recommendation + gate decision
┌──────────────────────────────────────────────────────────────────┐
│  Audit Log                                                       │
│  Full per-decision record (see edl_v3_2_audit_schema.md)        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. EDL Gate Logic

The EDL gate is derived **after** EDL classification. It is not part of the Dirichlet class space.

### Gate thresholds (configurable, v3.2 defaults)

| Condition | Gate Decision |
|-----------|--------------|
| EDL agrees + u < 0.35 | `RECOMMEND_AS_IS` |
| EDL agrees + 0.35 ≤ u < 0.55 | `REDUCE_SIZE` |
| EDL agrees + u ≥ 0.55 | `HUMAN_REVIEW` |
| EDL disagrees + u < 0.45 | `HUMAN_REVIEW` |
| EDL disagrees + u ≥ 0.45 | `FORCE_HOLD` |
| EDL strongly predicts REBALANCE (p_rebalance > 0.6) | `STRATEGY_REVIEW` |
| EDL strongly predicts CHANGE_STRATEGY (p_cs > 0.5) | `STRATEGY_REVIEW` |
| model_disagreement_score > 0.5 | add `HUMAN_REVIEW` |

### Size reduction rule (when REDUCE_SIZE)

```
final_fraction = original_fraction * (1 − λ_u * u)
final_fraction = max(final_fraction, 0.0)
```

where `λ_u = STOCK_INVESTMENT_DSS_EDL_UNCERTAINTY_LAMBDA` (default 0.5).

---

## 5. Relationship to v3.1 Code

### What v3.1 had

| v3.1 module | v3.1 function | v3.2 relationship |
|------------|---------------|-------------------|
| `edl_classifier.py` | Rule-based Dirichlet over LOW/MEDIUM/HIGH | **Superseded by** `edl_action_network.py` + `edl_action_classifier.py` |
| `recommendation_confidence.py` | Label + warnings over v3.1 Dirichlet | **Superseded by** `edl_gate.py`; v3.1 warnings can be derived from v3.2 evidence |
| `run_edl_uncertainty_smoke_test.py` | Reads hierarchical audit → v3.1 outputs | **Superseded by** `run_edl_action_inference_smoke_test.py` |
| `__init__.py` | Package init | Shared / extended |

**v3.1 files are NOT deleted.** They remain as the deprecated PoC interface smoke test, clearly marked in their module docstrings.

### What v3.2 adds

New files (see `edl_v3_2_implementation_plan.md` for full list):
- `edl_action_classes.py` — K, action index maps, CHANGE_STRATEGY flag
- `edl_action_dataset.py` — Dataset builder for A/B/C supervised training
- `edl_action_network.py` — PyTorch MLP with Softplus evidence head
- `edl_losses.py` — EDL UCE loss + KL regulariser + annealing
- `edl_action_classifier.py` — High-level classifier wrapping network + inference
- `edl_ensemble.py` — A/B/C ensemble logic
- `edl_gate.py` — Gate logic, size reduction, FORCE_HOLD, HUMAN_REVIEW
- Runners: dataset builder, training smoke test, inference smoke test

---

## 6. Configuration Flags

All configurable via environment variables (see `edl_v3_2_ablation_switching_plan.md`):

```
STOCK_INVESTMENT_DSS_USE_HIERARCHICAL_POLICY    = true / false
STOCK_INVESTMENT_DSS_USE_EDL                    = true / false
STOCK_INVESTMENT_DSS_EDL_VARIANT                = none | A | B | C | AB | AC | BC | ABC
STOCK_INVESTMENT_DSS_EDL_GATE_ENABLED           = true / false
STOCK_INVESTMENT_DSS_EDL_INCLUDE_CHANGE_STRATEGY= true / false
STOCK_INVESTMENT_DSS_EDL_UNCERTAINTY_LAMBDA     = <float, default 0.5>
STOCK_INVESTMENT_DSS_EDL_DISAGREEMENT_LAMBDA    = <float, default 0.3>
STOCK_INVESTMENT_DSS_EDL_HORIZON_DAYS           = <int, default 20>
STOCK_INVESTMENT_DSS_EDL_LABEL_MODE             = hindsight | rules | iqn_teacher
STOCK_INVESTMENT_DSS_EDL_MODEL_PATH             = <path to .pt checkpoint>
```

---

## 7. Thesis Alignment

The corrected architecture directly implements the methodology statement:

> *"The thesis explores uncertainty-aware modeling by incorporating an epistemic uncertainty component inspired by Evidential Deep Learning. This allows the system to account not only for stochastic return variability but also for uncertainty in the model's own estimates, which is particularly relevant in financial environments characterized by limited data and regime shifts."*

| Thesis requirement | v3.2 implementation |
|--------------------|---------------------|
| EDL epistemic uncertainty | Vacuity u = K/S over action classes |
| IQN aleatoric uncertainty | q_spread, CVaR per action |
| Risk-sensitive decisions | λ_u penalty in final score |
| Regime-shift detection | u spike → HUMAN_REVIEW / FORCE_HOLD gate |
| Model disagreement | A/B/C disagreement_score |
| Transparent decision support | per-action evidence + gate reason codes |
| Point-in-time safety | input features from t only; labels from t+h |
