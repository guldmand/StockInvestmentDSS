# EDL v3.4 — Hindsight Supervised Learning Design Plan

**Version:** 3.4  
**Status:** Design only — no code created  
**Reference:** Sensoy et al. NeurIPS 2018 "Evidential Deep Learning to Quantify Classification Uncertainty"  
**Reference implementation:** `pytorch-classification-uncertainty-master/`

---

## 1. What the EDL Paper and Reference Repo Actually Do

### 1.1 Core Formulation

Evidential Deep Learning (EDL) is a framework for **supervised multi-class classification with Dirichlet-based uncertainty**. It does not require ensembles or MC Dropout; the uncertainty comes from the network's own evidence output.

The pipeline is:

```
x (input features)
  → MLP backbone
    → raw logits  ∈ ℝ^K
      → evidence activation  →  e_k ≥ 0  for k = 1..K
        → alpha_k = e_k + 1        (Dirichlet concentration parameters)
          → S = Σ_k alpha_k        (Dirichlet strength)
            → p_k = alpha_k / S   (expected class probability)
            → u   = K / S         (vacuity / epistemic uncertainty)
```

All quantities are derived from the raw logit via a non-negative activation. No softmax is applied.

### 1.2 Evidence Activations (from `losses.py`)

The reference repo provides three options:

| Activation | Code | Properties |
|---|---|---|
| **ReLU** | `relu_evidence(y) = F.relu(y)` | Zero for negative logits; sparse evidence |
| **Softplus** | `softplus_evidence(y) = F.softplus(y)` | Smooth, always positive, recommended default |
| **Exp** | `exp_evidence(y) = exp(clamp(y, -10, 10))` | Fast growth; clamped for numerical stability |

The reference train loop uses `relu_evidence` for the evidence display within training. The loss functions accept raw logits and apply their own evidence activation internally.

### 1.3 Loss Functions (from `losses.py`)

All EDL losses share the structure:

```
L_EDL = task_loss + annealing_coef * KL_divergence
```

where `annealing_coef = min(1.0, epoch / annealing_step)` — the KL penalty is ramped up linearly over the first `annealing_step` epochs, allowing the network to first learn to classify before being penalised for non-uniform Dirichlet concentration.

**KL divergence** is computed between the per-sample Dirichlet `Dir(alpha)` and the uniform prior `Dir(1, ..., 1)`, but only over the non-target classes (using `kl_alpha = (alpha - 1) * (1 - y_onehot) + 1`). This penalises false confidence on wrong classes, not on the correct class.

The three task losses differ in their measure of fit:

#### MSE Loss (`edl_mse_loss`)
Minimises squared error between the one-hot label and expected Dirichlet probability, plus a variance term:

```python
loglikelihood_err = Σ_k (y_k - alpha_k/S)²
loglikelihood_var = Σ_k  alpha_k * (S - alpha_k) / (S² * (S+1))
L_task = loglikelihood_err + loglikelihood_var
```

#### Log Loss (`edl_log_loss`)
Minimises the negative log likelihood under the Dirichlet:

```python
A = Σ_k y_k * (log(S) - log(alpha_k))
L_task = A
```

#### Digamma Loss (`edl_digamma_loss`)
Replaces `log` with `digamma` (ψ function), giving smoother gradients:

```python
A = Σ_k y_k * (ψ(S) - ψ(alpha_k))
L_task = A
```

All three are valid; digamma is typically most stable.

### 1.4 Training and Best-Checkpoint Logic (from `train.py`)

The reference `train_model()` function:

1. Runs a standard train/val loop for `num_epochs` epochs
2. In each epoch, for both `train` and `val` phases:
   - Computes `evidence = relu_evidence(logits)`
   - Computes `alpha = evidence + 1`, `u = K / S`
   - Tracks per-epoch `mean_evidence_succ` (correct predictions) and `mean_evidence_fail` (incorrect predictions)
   - These should diverge during training: correct predictions accumulate more evidence
3. After each val phase: `if val_acc > best_acc: save best_model_wts`
4. At the end: `model.load_state_dict(best_model_wts)` — the **best validation checkpoint** is returned, not the final epoch

Key insight: **best checkpoint is selected by validation accuracy, not by final epoch**. This is critical for generalisation.

The reference uses `annealing_step = 10` epochs as default — appropriate for small datasets. For financial data with limited samples, this should be tuned.

---

## 2. What Our A/B/C Variants Really Are

### 2.1 Clarification

EDL-A, EDL-B, and EDL-C are **our own label-source variants**, not concepts from the EDL paper. The paper has exactly one kind of label: the true class `y`. Our variants differ in how we construct `y`:

| Variant | Label source | Nature |
|---|---|---|
| **EDL-A** | Hindsight: future realised return over horizon h | Ground-truth outcome labels — the **correctness/performance track** |
| **EDL-B** | Rule-based: deterministic thresholds on technical indicators | Transparent baseline — reproducible but not outcome-validated |
| **EDL-C** | IQN+HDP teacher: `argmax(iqn_score_*)` from combined runner | Integration/imitation — demonstrates pipeline connectivity only |

### 2.2 EDL-C Teacher Imitation — What It Shows and Does Not Show

**EDL-C is useful as:**
- End-to-end pipeline connectivity smoke test (IQN → HDP → EDL → gate)
- Validation that the EDL architecture and training loop work correctly
- Baseline for comparing vacuity calibration against EDL-A

**EDL-C does NOT demonstrate:**
- Whether the recommendations lead to positive portfolio returns
- Whether high-confidence predictions correspond to correct market calls
- Any improvement over the IQN/HDP baseline in trading performance

This is because `teacher_label = argmax(iqn_score_*)` is a deterministic function of the same features used for training. The model learns the IQN's own decision rule — it does not learn from market outcomes.

### 2.3 EDL-A is the Main Track

EDL-A is where the thesis claim lives: if the EDL-A model assigns high confidence to actions that **actually led to positive outcomes** and high uncertainty to decisions that **turned out wrong**, then the vacuity signal is informative for decision support.

The EDL framework is only "evidential" in the meaningful sense when the evidence is trained against outcome labels.

### 2.4 EDL-B Rules as Transparent Baseline

EDL-B provides a rule-based supervised signal that is:
- Reproducible (no future data, no teacher bias)
- Interpretable (investors can audit the rules)
- A useful comparison point against EDL-A

If EDL-A performs comparably to EDL-B, the hindsight signal may not be strong enough at the chosen horizon.

---

## 3. Correct EDL-A Supervised Dataset Design

### 3.1 Point-in-Time Safety Principle

The feature vector `X_t` must contain **only information available at or before decision time t**.  
The label `y_t` may use outcomes from `t+h` because it is the **supervised target, not a feature**.

This mirrors the standard PIT backtesting setup already used by the IQN evaluation.

### 3.2 Feature Vector X_t

Source: combined IQN+HDP audit CSV (`combined_iqn_hierarchical_decision_by_step.csv`), which already aggregates all PIT-safe information at each decision step.

**Group A — IQN Action Distribution (29 columns)**

| Feature | Description |
|---|---|
| `iqn_q{10,25,50,75,90}_{hold,buy,sell,rebalance}` | Per-action quantiles from sampled IQN distribution |
| `iqn_cvar10_{hold,buy,sell,rebalance}` | CVaR@10% per action (downside risk) |
| `iqn_score_{hold,buy,sell,rebalance}` | Risk-adjusted composite score per action |
| `iqn_action_margin` | Score gap between best and second-best action |

**Group B — HierarchicalDecisionPolicy Context (2 columns)**

| Feature | Description |
|---|---|
| `selected_size_fraction` | Fraction of available capital allocated |
| `cash_weight` | Portfolio cash fraction at decision time |

**Group C — Ticker Technical Features (6 columns)**

| Feature | Description | HOLD fill |
|---|---|---|
| `momentum_score` | Trend momentum signal | 0.0 |
| `value_score` | Fundamental value signal | 0.0 |
| `quality_score` | Fundamental quality signal | 0.0 |
| `risk_score` | Fundamental risk signal | 0.0 |
| `price_vs_ma50` | (price / MA50) − 1 | 0.0 |
| `price_vs_ma200` | (price / MA200) − 1 | 0.0 |

**NaN handling:** HOLD rows have no selected ticker → all ticker features filled with 0.0, consistent with EDL-C training.

**Total: 37 features** — same as the current EDL-C feature matrix.

### 3.3 Label y_t (Hindsight)

For each decision row at date `t` with `selected_ticker`:

```
future_return_h      = close[t + h] / close[t] - 1
future_max_drawdown  = min(close[t:t+h] / close[t] - 1)   (rolling max drawdown over window)
risk_adjusted_score  = future_return_h - lambda_drawdown * abs(future_max_drawdown)
```

Label assignment:

```
if future_return_h >= buy_threshold AND abs(future_max_drawdown) <= max_drawdown_threshold:
    y_t = BUY
elif future_return_h <= sell_threshold OR abs(future_max_drawdown) > max_drawdown_threshold:
    y_t = SELL
else:
    y_t = HOLD
```

For HOLD rows (no ticker selected): `y_t = HOLD` by definition (the IQN/HDP decided not to act — the hindsight label validates whether that was correct).

### 3.4 Label Hyperparameters (First-Pass Values)

| Parameter | First-pass value | Notes |
|---|---|---|
| Horizon `h` | 20 trading days (~1 calendar month) | Common short-term evaluation horizon |
| `buy_threshold` | +0.03 (+3%) | Corresponds to meaningful return above noise |
| `sell_threshold` | −0.03 (−3%) | Symmetric to buy |
| `max_drawdown_threshold` | −0.08 (−8%) | Reject BUY if path drawdown exceeds 8% |
| `lambda_drawdown` | 0.5 | Weight of drawdown penalty in risk-adjusted score |

**These are hyperparameters to validate.** The distribution of resulting labels should be checked before training. If label distribution is severely imbalanced (e.g., >90% HOLD), thresholds need adjustment or horizon should be extended.

---

## 4. Proposed Time Split

### 4.1 Primary Split (sufficient data)

The EDL supervised model must be developed entirely within the **PIT training period** — the same period used to train the IQN. The final PIT evaluation/trading period remains untouched for final evaluation.

```
PIT Training Period
├── EDL train:      first 70%  (EDL model fitting)
├── EDL validation: next  15%  (best-checkpoint selection, early stopping)
└── EDL test:       final 15%  (final EDL model assessment)

PIT Trading Period (held-out — DO NOT USE FOR TUNING)
└── Final evaluation: IQN+HDP vs IQN+HDP+EDL-A trading performance
```

Time ordering must be strictly preserved: train < validation < test < trading. No shuffling.

### 4.2 Fallback Split (small dataset)

If the combined audit data is too small for a 3-way split (e.g., fewer than 200 rows in total), use a simpler 80/20 split:

```
PIT Training Period
├── EDL train:      first 80%
└── EDL validation: last  20%

No EDL internal test (use the final trading period as held-out test)
```

At current data sizes (271 rows), the 80/20 fallback is recommended. The 70/15/15 split becomes viable once the combined run covers a full multi-year IQN training history.

---

## 5. Hindsight Label Construction

### 5.1 Data Requirement

To construct hindsight labels, the labeler needs access to **post-decision price data** for each `selected_ticker` at each `date`. This data must come from the same price feed used by the IQN environment, using a separate look-forward pass that does not contaminate `X_t`.

The look-forward is applied **after** the audit CSV is fully written — it reads only from the historical price data store, not from any live feed.

### 5.2 Label Construction Algorithm

```python
def construct_hindsight_label(
    date: datetime,
    ticker: Optional[str],
    price_data: pd.DataFrame,   # historical OHLCV
    h: int = 20,
    buy_threshold: float = 0.03,
    sell_threshold: float = -0.03,
    max_drawdown_threshold: float = -0.08,
    lambda_drawdown: float = 0.5,
) -> str:
    if ticker is None:          # HOLD row — no ticker selected
        return "HOLD"

    t_close  = price_data.loc[date, ticker]["close"]
    window   = price_data.loc[date : date + h_trading_days, ticker]["close"]

    if len(window) < h // 2:    # insufficient future data → skip
        return None             # excluded from dataset

    future_return = window.iloc[-1] / t_close - 1
    running_drawdown = (window / t_close - 1).min()

    if future_return >= buy_threshold and running_drawdown >= max_drawdown_threshold:
        return "BUY"
    elif future_return <= sell_threshold or running_drawdown < max_drawdown_threshold:
        return "SELL"
    else:
        return "HOLD"
```

### 5.3 REBALANCE Label (see Section 6)

---

## 6. Whether to Include REBALANCE

### Option A — Keep REBALANCE as Fourth Class

Label REBALANCE based on a portfolio-level rule: if the portfolio concentration (largest position weight) exceeds a threshold (e.g., 0.5), the hindsight label is REBALANCE regardless of return.

**Pros:** Consistent with the 4-class action space.  
**Cons:** REBALANCE is a structural/risk decision, not a return-outcome decision. Mixing it with return-based labels creates a heterogeneous labelling policy that may confuse the model.

### Option B — Exclude REBALANCE from EDL-A (Recommended)

Handle REBALANCE entirely at the HDP/risk layer, not via EDL-A. The EDL-A model classifies only over {BUY, HOLD, SELL} (3 classes) using return-outcome labels. REBALANCE is applied by HDP as a portfolio constraint, downstream of the EDL gate.

**Pros:** Clean separation of concerns. EDL-A learns outcome quality; HDP handles portfolio structure.  
**Cons:** Slight mismatch between 3-class EDL-A and 4-class EDL-C. Must be handled carefully in the gate layer.

**Recommendation: Option B — use K=3 for EDL-A.**

This is the simplest and most defensible approach. The thesis can state: "EDL-A is trained over {BUY, HOLD, SELL} using hindsight return outcomes; REBALANCE is applied as a portfolio-structure rule by the HierarchicalDecisionPolicy layer, independently of the return-outcome classifier."

If data eventually supports K=4 with clean REBALANCE labels, it can be added in a later iteration.

---

## 7. Correct Model Evaluation

### 7.1 Required Baseline Comparisons

| Baseline | Description |
|---|---|
| **Majority baseline** | Always predict the most frequent class in training set |
| **Rule baseline (EDL-B labels)** | Train EDL on rule labels; compare accuracy and uncertainty quality |
| **IQN/HDP teacher baseline (EDL-C)** | Train EDL on teacher labels; compare whether EDL-A differs meaningfully |
| **Random baseline** | Uniform random class prediction (sanity check) |

If EDL-A does not outperform EDL-B on held-out accuracy, the hindsight label signal at h=20 may not be strong enough.

### 7.2 Metrics

**Classification quality:**

| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness |
| Balanced accuracy | Accounts for class imbalance |
| Macro F1 | Unweighted mean of per-class F1 |
| Per-class precision / recall / F1 / support | Class-specific performance |
| Confusion matrix | Misclassification patterns |

**Uncertainty quality:**

| Metric | Purpose |
|---|---|
| Mean vacuity (overall) | Average epistemic uncertainty |
| Mean vacuity on correct predictions | Should be lower |
| Mean vacuity on incorrect predictions | Should be higher |
| Vacuity gap (correct - incorrect) | Positive gap = uncertainty is informative |
| Entropy of predicted probabilities | Calibration proxy |

If `vacuity_incorrect > vacuity_correct` consistently, the model's uncertainty is meaningful for decision gating. This is the key thesis-relevant test for EDL-A.

**Calibration (if feasible):**  
Plot reliability diagrams (predicted probability vs empirical accuracy in bins). A well-calibrated EDL-A model should show monotone reliability.

### 7.3 Collapse Check

Flag if the model predicts fewer than K−1 unique classes on the held-out set. Collapse indicates training failure (often due to class imbalance or learning rate issues).

---

## 8. Hyperparameter Experiments

Small grid; run as smoke tests (e.g., 25/50 epochs) before full training.

| Hyperparameter | Values to try |
|---|---|
| Loss type | `mse`, `log`, `digamma` |
| Evidence activation | `relu`, `softplus`, `exp` |
| Hidden sizes | `[64]`, `[128]`, `[128, 64]` |
| Epochs | 25, 50, 100 |
| Class weights | `true`, `false` |
| Learning rate | `1e-3`, `3e-4` |
| KL annealing step | 10, 20 (tied to dataset size) |

**Recommended starting point:**  
`digamma` loss, `softplus` activation, `[128, 64]` hidden, 50 epochs, class_weights=true, lr=1e-3, annealing_step=10.

This matches the reference repo defaults closest while matching our current network architecture.

**Grid sweep script:** `run_edl_hyperparameter_sweep_smoke.py` (see Section 10).

---

## 9. End-to-End Ablation Plan

The thesis ablation table must compare these system configurations on the **held-out PIT trading period**:

| Configuration | IQN | HDP | EDL | Note |
|---|---|---|---|---|
| **Baseline: IQN only** | ✓ | ✗ | ✗ | Distributional RL baseline |
| **IQN + HDP** | ✓ | ✓ | ✗ | Hierarchical policy without uncertainty |
| **IQN + HDP + EDL-C** | ✓ | ✓ | EDL-C | Teacher imitation; pipeline connectivity |
| **IQN + HDP + EDL-A** | ✓ | ✓ | EDL-A | Hindsight-supervised uncertainty; **main track** |
| **IQN + HDP + EDL ensemble** | ✓ | ✓ | A+C or A+B+C | Optional; needs label alignment |

**Evaluation metric for ablation:** Final portfolio return over PIT trading period, Sharpe ratio, maximum drawdown, and HOLD rate.

**Critical constraint:** Do NOT claim EDL-A improves return unless:
1. EDL-A vacuity is demonstrably informative (vacuity gap > 0 on held-out data)
2. Gated decisions on the trading period differ from un-gated decisions
3. Return/Sharpe of gated policy is higher in the final evaluation

If EDL-A vacuity is uninformative, the honest thesis conclusion is that the EDL layer does not add signal at h=20 days, and alternative horizons or label constructions should be investigated.

---

## 10. Proposed Implementation Files for Next Phase

The following files are proposed. **Do not create until this plan is reviewed and approved.**

### Data / Labels

| File | Location | Purpose |
|---|---|---|
| `edl_hindsight_labeler.py` | `src/stock_investment_dss/uncertainty/` | Construct EDL-A hindsight labels from price data and audit CSV. Functions: `compute_future_return`, `compute_max_drawdown`, `assign_hindsight_label`, `build_hindsight_label_column`. |
| `edl_hindsight_dataset_builder.py` | `src/stock_investment_dss/uncertainty/` | Build train/validation/test splits with EDL-A labels. Mirrors `edl_action_dataset_v2.py` but uses hindsight labels and K=3 classes. Writes `edl_a_train_dataset.csv`, `edl_a_val_dataset.csv`, `edl_a_test_dataset.csv`. |

### Runners

| File | Location | Purpose |
|---|---|---|
| `run_edl_hindsight_dataset_builder.py` | `src/stock_investment_dss/runner/` | Runner for EDL-A dataset construction. Env vars for horizon, thresholds, label mode. Validates PIT integrity. |
| `run_edl_action_training_reference_aligned.py` | `src/stock_investment_dss/runner/` | Training runner aligned with reference repo: supports mse/log/digamma losses, relu/softplus/exp evidence activations, KL annealing, best-val checkpoint. Works with both EDL-A and EDL-C datasets. |
| `run_edl_hyperparameter_sweep_smoke.py` | `src/stock_investment_dss/runner/` | Lightweight grid sweep over loss/activation/hidden_dims combinations. Writes sweep summary CSV. Does not save all checkpoints — only best per combination. |
| `run_iqn_hdp_edl_ablation_smoke.py` | `src/stock_investment_dss/runner/` | Runs all four ablation configurations in sequence and writes a comparison table. Uses pre-trained IQN checkpoint and best EDL-A/C checkpoints. |

### Architecture Note

The existing `EDLActionNetwork` in `edl_action_network.py` is already compatible with the reference repo formulation. The main gap is in the loss function: the current `edl_action_loss` in `edl_losses.py` does not implement KL annealing. `run_edl_action_training_reference_aligned.py` should use the reference's `edl_digamma_loss` / `edl_mse_loss` / `edl_log_loss` with proper annealing, rather than the current simplified loss.

---

## 11. Conclusion

### Should the Current EDL-C Gate Be Retained?

**Yes — as integration and teacher-imitation evidence. No — not as performance evidence.**

Specifically:

| Claim | Status |
|---|---|
| EDL-C gate demonstrates end-to-end pipeline connectivity | ✅ Valid — retain as pipeline smoke test |
| EDL-C gate demonstrates that uncertainty gating improves portfolio performance | ❌ Invalid — do not claim |
| EDL-C accuracy (0.745–1.000) demonstrates model generalisation | ❌ Invalid — structural imitation of argmax(iqn_score_*) |
| EDL-C gate is a useful integration test for the thesis | ✅ Valid — document with correct caveats |

The EDL-C gate should be retained in the implementation as **Phase 1 of the EDL track** (pipeline connectivity) with clear documentation that it measures teacher imitation, not market correctness.

### EDL-A Is Required for Meaningful Performance/Correctness Validation

EDL-A trained on hindsight labels is the only EDL variant that can support the following thesis claims:

1. "The EDL uncertainty layer assigns higher vacuity to decisions that turned out incorrect"
2. "Gating on high-vacuity decisions reduces the frequency of bad outcomes"
3. "The IQN+HDP+EDL-A system outperforms the IQN+HDP baseline on risk-adjusted return"

Without EDL-A, the thesis can only claim:
- Distributional RL (IQN) produces richer value representations than point-value RL
- HierarchicalDecisionPolicy provides structured contextual action selection
- The EDL layer is architecturally integrated and produces calibrated uncertainty (on teacher labels)

This is still a valid thesis contribution. But the strongest claim — that uncertainty gating improves decision quality — requires EDL-A.

### Recommended Next Step

Proceed to EDL-A dataset construction:
1. Verify that the combined audit CSV has sufficient rows with a non-null `selected_ticker` (for hindsight labelling)
2. Confirm that the price data store has coverage for `t + 20` trading days beyond the audit period
3. Implement `edl_hindsight_labeler.py` and `run_edl_hindsight_dataset_builder.py`
4. Check label distribution — if HOLD dominates, adjust thresholds before training

---

*Plan created: Session `b9e47002`*  
*Reference: `pytorch-classification-uncertainty-master/losses.py` and `train.py`*  
*No source code was created or modified during this planning step.*
