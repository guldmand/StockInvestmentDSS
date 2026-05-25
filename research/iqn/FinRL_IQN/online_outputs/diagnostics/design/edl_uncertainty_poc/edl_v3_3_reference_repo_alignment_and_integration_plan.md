# EDL v3.3 — Reference Repo Alignment & Integration Plan

**Scope:** Analysis and planning only. No source changes in this document.  
**Reference repo:** `pytorch-classification-uncertainty` (dougbrion, GitHub)  
**Status:** Design / Audit — Implementation in next phase.

---

## 1. Reference Repo Audit

### 1.1 Reference Repo Structure

```
losses.py   — all EDL loss functions (mse, log, digamma, kl divergence, evidence activations)
train.py    — training loop with uncertainty mode, evidence tracking
helpers.py  — one_hot_embedding, get_device
lenet.py    — MNIST-specific LeNet (NOT to be copied)
```

### 1.2 Reference Repo: Evidence Activations

Reference repo supports three:

```python
def relu_evidence(y):    return F.relu(y)
def exp_evidence(y):     return torch.exp(torch.clamp(y, -10, 10))
def softplus_evidence(y): return F.softplus(y)
```

The `edl_mse_loss`, `edl_log_loss`, `edl_digamma_loss` **all use `relu_evidence`** internally (they re-apply it on raw logits):

```python
def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    evidence = relu_evidence(output)   # <-- applied internally to raw logits
    alpha = evidence + 1
    loss = torch.mean(mse_loss(target, alpha, ...))
```

### 1.3 Reference Repo: KL Divergence

Reference `kl_divergence(alpha, num_classes, device)`:

```python
ones = torch.ones([1, num_classes])
sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
first_term = lgamma(sum_alpha) - lgamma(alpha).sum(dim=1) + lgamma(ones).sum(dim=1) - lgamma(sum(ones))
second_term = (alpha - ones) * (digamma(alpha) - digamma(sum_alpha)).sum(dim=1)
kl = first_term + second_term
```

Returns shape `(batch, 1)` — **keepdim=True throughout**.

### 1.4 Reference Repo: alpha_tilde / KL annealing

```python
kl_alpha = (alpha - 1) * (1 - y) + 1    # equivalent to: y + (1-y)*alpha
```

This is mathematically identical to our `alpha_tilde = y_onehot + (1-y_onehot)*alpha`.  
The reference form `(alpha - 1)*(1-y) + 1 = alpha*(1-y) - (1-y) + 1 = alpha - alpha*y - 1 + y + 1 = alpha - y*(alpha-1)` — same result. ✅

```python
annealing_coef = torch.min(
    torch.tensor(1.0, dtype=torch.float32),
    torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
)
```

`annealing_step` is the epoch at which KL reaches full weight (≈ `total_epochs * anneal_frac`).

### 1.5 Reference Repo: MSE / Loglikelihood Loss

```python
loglikelihood_err  = sum_k (y_k - alpha_k/S)^2        # squared error
loglikelihood_var  = sum_k alpha_k*(S-alpha_k)/(S^2*(S+1))  # variance
loglikelihood      = err + var
mse_loss           = loglikelihood + annealing_coef * kl_div
```

Returns shape `(batch, 1)` (keepdim).

### 1.6 Reference Repo: EDL Log / Digamma losses

General form:
```python
A = sum_k  y_k * (func(S) - func(alpha_k))
loss = A + annealing_coef * kl_div
```
Where `func` is `torch.log` or `torch.digamma`.

These are **additional loss variants NOT implemented in our current codebase.**

### 1.7 Reference Repo: Train Loop

Key observations:
- `model.train()` / `model.eval()` per phase — ✅ we do this
- `one_hot_embedding(labels, num_classes)` via `torch.eye(K)[labels]` — ✅ equivalent to our `scatter_`
- Criterion called as `criterion(outputs, y.float(), epoch, num_classes, annealing_step, device)` — note: **raw logits passed to loss, not alpha**. Evidence activation is re-applied inside the loss.
- Tracks `mean_evidence_succ` (correct predictions) vs `mean_evidence_fail` (wrong) — **missing in our runner**
- Best-val checkpoint tracking (`best_acc`, `best_model_wts`) — **missing in our runner**
- `relu_evidence` used for evidence display during training, even when network uses softplus — minor inconsistency in reference repo

---

## 2. Current Implementation vs Reference: Side-by-Side Comparison

| Aspect | Reference Repo | Our v3.2 | Status |
|--------|---------------|----------|--------|
| Evidence activation | ReLU (internal to loss) | Softplus (in network.forward) | ⚠️ **Architectural difference** |
| Alpha = evidence + 1 | ✅ `evidence + 1` | ✅ `evidence + 1.0` | ✅ Correct |
| Vacuity u = K/S | ✅ `num_classes / sum(alpha)` | ✅ `self.num_classes / S` | ✅ Correct |
| MSE / loglikelihood | ✅ `(y-p)^2 + var` | ✅ identical formula | ✅ Correct |
| Log loss | ✅ `sum y*(log(S)-log(alpha))` | ❌ not implemented | ❌ **Missing** |
| Digamma loss | ✅ `sum y*(digamma(S)-digamma(alpha))` | ❌ not implemented | ❌ **Missing** |
| KL divergence formula | ✅ correct | ✅ mathematically equivalent | ✅ Correct |
| alpha_tilde | ✅ `(alpha-1)*(1-y)+1` | ✅ `y+(1-y)*alpha` | ✅ Same |
| KL annealing | ✅ `epoch/annealing_step` | ✅ `epoch/(total*frac)` | ✅ Equivalent |
| KL output shape | `(batch, 1)` keepdim | `(batch,)` | ⚠️ Shape differs — but final `.mean()` result is same |
| One-hot encoding | `torch.eye(K)[labels]` | `scatter_` | ✅ Equivalent |
| Loss applied to | Raw logits (alpha inside loss) | Pre-computed alpha | ⚠️ **Interface difference** |
| Best-val checkpointing | ✅ tracked | ❌ not tracked | ❌ **Missing** |
| Evidence success/fail tracking | ✅ tracked | ❌ not tracked | ❌ **Missing** |
| Configurable loss variant | MSE / Log / Digamma | MSE only | ❌ **Missing** |
| Exp evidence activation | ✅ supported | ❌ not supported | Low priority |

### 2.1 What Our Implementation Already Does Correctly

- `alpha = evidence + 1` — correct
- `u = K / S` — correct
- `p = alpha / S` — correct
- `belief = evidence / S` — correct
- MSE loss formula is mathematically correct (identical to reference `loglikelihood_loss`)
- KL divergence formula is mathematically correct
- `alpha_tilde` formulation is mathematically identical to reference `kl_alpha`
- KL annealing: same logic, different parameter naming
- One-hot encoding: equivalent result
- `model.train()` / `model.eval()` discipline: correct
- Softplus evidence activation: valid choice per Sensoy 2018 (reference uses ReLU but Softplus is theoretically cleaner)
- Feature standardisation (added in v3.2 smoke test): correct, train-only stats
- Class-weighted loss: correct inverse-frequency approach

### 2.2 What Differs from the Reference Repo

**Critical architectural difference:**  
The reference repo applies evidence activation **inside the loss function** on raw logits. Our network applies Softplus **inside `forward()`** and passes `alpha` to the loss. This means:

- Reference: `edl_mse_loss(logits, y, ...) → relu(logits) → alpha → loss`
- Ours: `network.forward(x) → softplus(logits) → alpha → edl_action_loss(alpha, y, ...)`

Both are valid architectures. The reference approach allows swapping loss functions without changing the network. Our approach is cleaner for inference (consistent evidence/alpha/vacuity from `forward()`).

**Recommendation:** Keep our architecture (evidence in network). It is more suitable for our thesis because inference needs the full `{evidence, alpha, S, prob, vacuity, belief}` dict at every step.

**KL shape:** Reference returns `(batch, 1)` from `kl_divergence`. Our `kl_divergence_uniform` returns `(batch,)`. Since we call `.mean()` outside, both are equivalent.

### 2.3 What Is Missing and Should Be Added

Priority order:

| # | Missing | Priority | File to change |
|---|---------|----------|---------------|
| 1 | **Log loss** `sum y*(log(S)-log(alpha))` | High — thesis comparison | `edl_reference_losses.py` (new) |
| 2 | **Digamma loss** `sum y*(digamma(S)-digamma(alpha))` | High — thesis comparison | `edl_reference_losses.py` (new) |
| 3 | **Loss variant selector** `EDL_LOSS_VARIANT=mse\|log\|digamma` | High | training runner |
| 4 | **Best-val checkpoint tracking** | Medium | training runner |
| 5 | **Evidence success/fail tracking per epoch** | Medium | training runner + audit |
| 6 | **Exp evidence activation** | Low | `edl_action_network.py` |

### 2.4 What Should Be Changed

1. **`edl_reference_losses.py` (new file):** Add `edl_log_loss` and `edl_digamma_loss` matching reference repo exactly (but without MNIST-specific code). Expose as selectable variant via `EDL_LOSS_VARIANT` env var.

2. **Training runner:** Add `best_val_acc` tracking, save best model not last model. Add `mean_evidence_succ` / `mean_evidence_fail` logging per epoch.

3. **`EDLActionConfig`:** Add `loss_variant: str = "mse"` field, validated against `["mse", "log", "digamma"]`.

---

## 3. Correct EDL Architecture for the Thesis

### 3.1 Thesis System Pipeline

```
Market data + portfolio state
        │
        ▼
   IQN agent
   (D-IQN with LayerNorm)
        │  raw quantile distributions per action
        ▼
HierarchicalDecisionPolicy
   Stage 1: action_type (HOLD/BUY/SELL/REBALANCE)
   Stage 2: ticker selection
   Stage 3: size selection
   Stage 4: risk/strategy validation
        │
        │  HierarchicalDecision + IQN audit dict
        ▼
EDL Action Classifier
   Input: combined feature vector X_t
   Output: Dirichlet(alpha) over {HOLD, BUY, SELL, REBALANCE}
        │
        ▼
EDL Gate
   Gate: RECOMMEND_AS_IS / REDUCE_SIZE / FORCE_HOLD / HUMAN_REVIEW / STRATEGY_REVIEW
        │
        ▼
Final recommendation + audit record
```

### 3.2 Correct Input Feature Vector X_t (v3.3)

The current v3.2 feature vector uses only raw market data (19 features). For the thesis, the correct input must include context from the full pipeline:

**Group A — IQN Action Distribution Features** (per IQN run)

| Feature | Shape | Source |
|---------|-------|--------|
| `iqn_q10_HOLD`, `iqn_q25_HOLD`, ..., `iqn_q90_HOLD` | 5 per action | IQN quantile network |
| `iqn_q10_BUY`, ..., `iqn_q90_BUY` | 5 per action | IQN quantile network |
| `iqn_q10_SELL`, ..., `iqn_q90_SELL` | 5 per action | IQN quantile network |
| `iqn_q10_REBALANCE`, ..., `iqn_q90_REBALANCE` | 5 per action | IQN quantile network |
| `iqn_CVaR_HOLD`, `iqn_CVaR_BUY`, `iqn_CVaR_SELL`, `iqn_CVaR_REBALANCE` | 4 | IQN CVaR |
| `iqn_score_HOLD`, ..., `iqn_score_REBALANCE` | 4 | IQN action scores |
| `iqn_action_margin` | 1 | score[top1] - score[top2] |
| `iqn_chosen_action_idx` | 1 | IQN argmax |

Total: 5×4 + 4 + 4 + 1 + 1 = **30 IQN features**

**Group B — HierarchicalDecisionPolicy Features** (per HDP run)

| Feature | Source |
|---------|--------|
| `hdp_selected_action_idx` | Stage 1 action (HOLD=0,BUY=1,SELL=2,REBALANCE=3) |
| `hdp_ticker_score_top1`, `hdp_ticker_score_top2` | TickerSelector scores |
| `hdp_size_score_top1`, `hdp_size_score_top2` | SizeSelector scores |
| `hdp_risk_ok` | RiskCheckResult.all_ok → 0/1 |
| `hdp_bear_market_signal` | bear_market_signal → 0/1 |
| `hdp_bear_market_penalty` | bear_market_penalty |
| `hdp_risk_adjusted_fraction` | risk_adjusted_allocation_fraction |

Total: ~9 HDP features

**Group C — Market / Technical Features** (current v3.2 feature set)

| Feature | Count |
|---------|-------|
| Technical: macd, rsi_30, cci_30, dx_30 | 4 |
| Price vs trend: price_vs_ma50, price_vs_ma200 | 2 |
| Rolling: drawdown, return_5d, return_20d, volatility_20d | 4 |
| Fundamental: revenue_growth, earnings_growth, profit_margin, pe_ratio, ps_ratio, fcf_yield, debt_ratio | 7 |
| Portfolio: cash_weight, risk_adjusted_fraction | 2 |

Total: **19 market/portfolio features**

**Group D — Derived Context**

| Feature | Derivation |
|---------|-----------|
| `iqn_hdp_agree` | iqn_chosen_action == hdp_selected_action → 0/1 |
| `iqn_hdp_action_distance` | abs(iqn_chosen_action_idx - hdp_selected_action_idx) |
| `vix` | VIX from market data |

Total: ~3 derived features

**Full v3.3 input dimension: ~61 features (30 IQN + 9 HDP + 19 market + 3 derived)**

In smoke test / early development, IQN features default to 0 and are gradually filled as the pipeline integrates.

### 3.3 Model Architecture

```
Input: X_t ∈ ℝ^61 (standardised using train-set mean/std)
Hidden: MLP [128, 128, 64] with SiLU + Dropout(0.1)
Output logits → Softplus → evidence e ∈ ℝ^4 (e_k ≥ 0)
alpha_k = e_k + 1
S = Σ alpha_k
prob_k = alpha_k / S
vacuity u = K/S ∈ (0, 1]
```

No softmax. No cross-entropy. Dirichlet parameterisation throughout.

---

## 4. Correct Label Construction: EDL-A / B / C

### 4.1 EDL-A: Hindsight Oracle

**What:** For each `(ticker, date=t)`, compute the forward realized return over `horizon_h` trading days using the same ticker's close price at `t+h`. Label based on realized risk-adjusted outcome.

**Formula:**
```
future_return = close[t+h] / close[t] - 1
risk_penalty  = drawdown_from_recent_high[t] * β_drawdown

if future_return > buy_threshold:      label = BUY
elif future_return < sell_threshold:   label = SELL
elif volatility_20d[t] > vol_thresh:   label = REBALANCE
else:                                  label = HOLD
```

**Point-in-time safety:** Feature vector X_t uses only data at t. Future data ONLY appears as the label target.

**Correctness for supervised training:** Highest ground truth signal. The model learns from actual outcomes.  
**Limitation:** Labels are noisy (future returns are stochastic). Not aligned with what the IQN/HDP system would recommend.

### 4.2 EDL-B: Rule-Based Labels

**What:** Labels generated from transparent heuristic rules applied to features at time t. The rules are fully deterministic, inspect no future data.

**Current implementation:** rsi_30, macd, drawdown_from_recent_high, recent_return_20d, volatility_20d.

**Correctness for supervised training:** Lower signal than EDL-A (rules may not match realized outcomes). High label consistency.  
**Advantage:** Labels available immediately without future data. Suitable for online/streaming scenarios.

### 4.3 EDL-C: IQN + HDP Teacher Labels

**What:** The IQN agent produces action-type scores, HierarchicalDecisionPolicy translates them into a final recommendation (HOLD/BUY/SELL/REBALANCE). This becomes the teacher label.

**Teacher label = `HierarchicalDecision.selected_action_type`**

**Correctness:** NOT ground truth (the system can be wrong). Treated as distillation, not oracle.  
**Purpose:** The EDL classifier learns the system's decision pattern under uncertainty. High agreement between EDL and IQN+HDP = low vacuity = RECOMMEND_AS_IS. Low agreement = higher vacuity = REDUCE_SIZE or HUMAN_REVIEW.  
**Requires:** Full IQN + HDP pipeline run to generate audit CSV. Cannot be bootstrapped from raw market data alone.

### 4.4 Label Mode Comparison

| | EDL-A Hindsight | EDL-B Rules | EDL-C Teacher |
|--|---|---|---|
| **Best for actual correctness** | ✅ Yes — trained on real outcomes | ❌ Heuristic only | ❌ Mimics system, not ground truth |
| **Best for mimicking the system** | ❌ No | ❌ Independent of IQN+HDP | ✅ Yes — learns system behaviour |
| **Best for thesis evidence** | ✅ Yes — shows risk-adjusted outcome | ⚠️ Useful baseline | ✅ Yes — validates EDL alignment |
| **Point-in-time safe** | ✅ Input only; label uses future | ✅ Fully safe | ✅ Input+label both at t |
| **Requires IQN+HDP run** | ❌ No | ❌ No | ✅ Yes |
| **Data requirements** | Market close prices | Market data | Full IQN+HDP audit CSV |

**Recommendation for thesis:**
- Use **EDL-A** as the primary correctness label (proves learning from realized outcomes)
- Use **EDL-B** as the interpretable baseline (transparent rules, easy to explain)
- Use **EDL-C** as the distillation mode (once IQN+HDP audit is generated; required for full pipeline validation)
- Train all three and compare vacuity distributions, accuracy, and gate outcomes

---

## 5. Integration Order: IQN + HDP Before EDL-C

### Decision: YES — IQN + HDP must be connected before EDL-C

**Reasoning:**

1. **EDL-C requires teacher labels from the real system.** Without a connected IQN + HDP run producing an audit CSV, there are no `selected_action_type` labels to distill from.

2. **EDL-C input features must include IQN action distribution features.** The 30 IQN features (quantiles, CVaR, scores) can only be populated from a real IQN run. Without them, the feature vector is impoverished and EDL-C cannot learn the relationship between IQN distribution shape and HDP decisions.

3. **The gate's usefulness depends on EDL learning from the same context the system sees.** If EDL-A/B are trained on market data alone, the gate measures uncertainty about market signals. EDL-C trained on IQN+HDP features measures uncertainty about the system's own decisions — which is what the thesis goal requires.

4. **EDL-A/B can run before IQN+HDP integration**, and should — they serve as development stepping stones.

**Integration order for thesis:**

```
Phase 1 (done):     EDL-A and EDL-B on market features only (v3.2 baseline)
Phase 2 (next):     IQN + HDP combined audit pipeline → combined_audit CSV
Phase 3:            EDL-C on combined features (IQN + HDP + market)
Phase 4:            EDL ensemble (A+B+C or subset) with gate
```

---

## 6. Required Combined Audit Dataset Schema

**File:** `outputs/runs/<run_id>/audit/combined_iqn_hierarchical_decision_by_step.csv`

### 6.1 Required Columns

**Identity**

| Column | Type | Description |
|--------|------|-------------|
| `decision_id` | str (UUID) | Unique ID per decision step |
| `date` | str (ISO) | Decision date `YYYY-MM-DD` |
| `visible_data_cutoff` | str (ISO) | Last date of visible data (= date for daily) |
| `market_state_id` | str | Hash of market observation state |
| `step_idx` | int | Episode step index |

**Portfolio State**

| Column | Type | Description |
|--------|------|-------------|
| `portfolio_total_value` | float | Total portfolio value |
| `portfolio_cash` | float | Cash in portfolio |
| `portfolio_cash_weight` | float | Cash / total |
| `portfolio_max_position_weight` | float | Largest single position weight |
| `portfolio_drawdown_from_peak` | float | Drawdown from portfolio peak |
| `portfolio_holdings_json` | str (JSON) | {ticker: shares} snapshot |

**IQN Action Distribution**

| Column | Type | Description |
|--------|------|-------------|
| `iqn_q10_HOLD`, `iqn_q25_HOLD`, `iqn_q50_HOLD`, `iqn_q75_HOLD`, `iqn_q90_HOLD` | float×5 | HOLD quantile returns |
| `iqn_q10_BUY`, ..., `iqn_q90_BUY` | float×5 | BUY quantile returns |
| `iqn_q10_SELL`, ..., `iqn_q90_SELL` | float×5 | SELL quantile returns |
| `iqn_q10_REBALANCE`, ..., `iqn_q90_REBALANCE` | float×5 | REBALANCE quantile returns |
| `iqn_CVaR_HOLD`, `iqn_CVaR_BUY`, `iqn_CVaR_SELL`, `iqn_CVaR_REBALANCE` | float×4 | CVaR per action |
| `iqn_score_HOLD`, `iqn_score_BUY`, `iqn_score_SELL`, `iqn_score_REBALANCE` | float×4 | Score (e.g. q50 or risk-adjusted) |
| `iqn_action_margin` | float | score[best] - score[2nd] |
| `iqn_selected_action` | str | IQN argmax recommendation |
| `iqn_model_id` | str | IQN checkpoint identifier |

**HierarchicalDecisionPolicy Output**

| Column | Type | Description |
|--------|------|-------------|
| `hdp_selected_action_type` | str | Stage 1 decision: HOLD/BUY/SELL/REBALANCE |
| `hdp_selected_ticker` | str | Stage 2 ticker (empty if HOLD/REBALANCE) |
| `hdp_selected_size` | str | Stage 3 size label |
| `hdp_selected_fraction` | float | Raw allocation fraction |
| `hdp_risk_adjusted_fraction` | float | After risk check |
| `hdp_ticker_score_top1` | float | TickerSelector top score |
| `hdp_ticker_score_top2` | float | TickerSelector 2nd score |
| `hdp_size_score_top1` | float | SizeSelector top score |
| `hdp_risk_ok` | bool | All risk checks passed |
| `hdp_bear_market_signal` | bool | Bear market guard fired |
| `hdp_bear_market_penalty` | float | Score penalty applied |
| `hdp_strategy_id` | str | Strategy profile used |

**Market / Technical Snapshot (per ticker)**

| Column | Type | Description |
|--------|------|-------------|
| `mkt_close` | float | Close price at t |
| `mkt_MA50` | float | 50-day MA (close_30_sma proxy) |
| `mkt_MA200` | float | 200-day MA (close_60_sma proxy) |
| `mkt_price_vs_ma50` | float | (close - MA50) / MA50 |
| `mkt_price_vs_ma200` | float | (close - MA200) / MA200 |
| `mkt_rsi_30` | float | RSI(30) |
| `mkt_macd` | float | MACD |
| `mkt_cci_30` | float | CCI(30) |
| `mkt_dx_30` | float | DX(30) |
| `mkt_boll_ub`, `mkt_boll_lb` | float | Bollinger bands |
| `mkt_vix` | float | VIX |
| `mkt_volatility_20d` | float | Rolling 20d vol (computed) |
| `mkt_drawdown_from_recent_high` | float | Drawdown (computed) |
| `mkt_recent_return_5d` | float | 5d return (computed) |
| `mkt_recent_return_20d` | float | 20d return (computed) |

**EDL Label Fields**

| Column | Type | Description |
|--------|------|-------------|
| `edl_label_A_hindsight` | str | Future realized label (HOLD/BUY/SELL/REBALANCE) |
| `edl_label_A_fwd_return_h` | float | Forward return used for label |
| `edl_label_A_horizon_days` | int | Horizon used |
| `edl_label_B_rules` | str | Rule-based label |
| `edl_label_B_reason` | str | Which rule fired |
| `edl_label_C_teacher` | str | IQN+HDP selected action (teacher label) |
| `edl_label_C_source` | str | `iqn_teacher` |

---

## 7. End-to-End Run Toggles

### 7.1 Toggle Environment Variables

| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `STOCK_INVESTMENT_DSS_USE_HIERARCHICAL_POLICY` | `true/false` | `true` | Enable/disable HDP stage |
| `STOCK_INVESTMENT_DSS_USE_EDL` | `true/false` | `false` | Enable/disable EDL layer |
| `STOCK_INVESTMENT_DSS_EDL_VARIANT` | `none\|A\|B\|C\|AB\|AC\|BC\|ABC` | `none` | Which EDL label variant(s) to use |
| `STOCK_INVESTMENT_DSS_EDL_GATE_ENABLED` | `true/false` | `false` | Enable/disable gate |
| `STOCK_INVESTMENT_DSS_EDL_MODEL_PATH` | path | `""` | Trained EDL checkpoint |
| `STOCK_INVESTMENT_DSS_EDL_LABEL_MODE` | `hindsight\|rules\|iqn_teacher` | `rules` | Training label mode |
| `STOCK_INVESTMENT_DSS_EDL_UNCERTAINTY_LAMBDA` | float | `0.5` | λ_u in gate size formula |
| `STOCK_INVESTMENT_DSS_EDL_DISAGREEMENT_LAMBDA` | float | `0.3` | λ_d ensemble disagreement |
| `EDL_LOSS_VARIANT` | `mse\|log\|digamma` | `mse` | Training loss function |
| `EDL_TRAIN_EPOCHS` | int | `5` | Training epochs |
| `EDL_USE_CLASS_WEIGHTS` | `true/false` | `true` | Class-weighted loss |

### 7.2 Run Configurations

**Config 1 — IQN only (baseline)**
```
USE_HIERARCHICAL_POLICY=false
USE_EDL=false
```
Output: IQN action scores → argmax action → direct execution. No gate, no HDP.

**Config 2 — IQN + HDP**
```
USE_HIERARCHICAL_POLICY=true
USE_EDL=false
```
Output: IQN scores → HDP 5-stage pipeline → HOLD/BUY/SELL/REBALANCE + ticker + size + risk check.

**Config 3 — IQN + HDP + EDL-A (hindsight)**
```
USE_HIERARCHICAL_POLICY=true
USE_EDL=true
EDL_VARIANT=A
EDL_LABEL_MODE=hindsight
EDL_GATE_ENABLED=true
EDL_MODEL_PATH=<trained_edl_A.pt>
```

**Config 4 — IQN + HDP + EDL-B (rules)**
```
USE_HIERARCHICAL_POLICY=true
USE_EDL=true
EDL_VARIANT=B
EDL_LABEL_MODE=rules
EDL_GATE_ENABLED=true
EDL_MODEL_PATH=<trained_edl_B.pt>
```

**Config 5 — IQN + HDP + EDL-C (teacher)**
```
USE_HIERARCHICAL_POLICY=true
USE_EDL=true
EDL_VARIANT=C
EDL_LABEL_MODE=iqn_teacher
EDL_GATE_ENABLED=true
EDL_MODEL_PATH=<trained_edl_C.pt>
```
**Requires:** `EDL_HIERARCHICAL_AUDIT_CSV` pointing to a pre-generated combined audit CSV.

**Config 6 — IQN + HDP + EDL ensemble ABC**
```
USE_HIERARCHICAL_POLICY=true
USE_EDL=true
EDL_VARIANT=ABC
EDL_GATE_ENABLED=true
```
All three EDL models vote; disagreement score drives ensemble gate.

---

## 8. Proposed Implementation Files

All paths are relative to `src/stock_investment_dss/`. Do NOT create these files yet — implementation in next phase.

### 8.1 Decision Layer

| File | Purpose |
|------|---------|
| `decision/combined_iqn_hierarchical_policy.py` | Thin orchestration wrapper connecting IQN agent → HierarchicalDecisionPolicy → combined audit dict. Exposes `run_step(obs, portfolio_state) → CombinedDecision`. Used as data source for EDL-C training and end-to-end testing. |

### 8.2 Uncertainty / EDL Layer

| File | Purpose |
|------|---------|
| `uncertainty/edl_reference_losses.py` | Add `edl_log_loss` and `edl_digamma_loss` matching reference repo. Expose unified `edl_loss_fn(variant, alpha, y_onehot, epoch, ...)`. Keep as separate file to not break existing `edl_losses.py`. |
| `uncertainty/edl_action_labeler.py` | Unified labeler: `EDLActionLabeler(mode).label(combined_audit_row, horizon_df)` → `(label_str, reason)`. Replaces scattered labeling logic in `edl_action_dataset.py`. Handles all three modes cleanly. |
| `uncertainty/edl_action_dataset_v2.py` | Dataset builder v2: accepts a combined audit CSV (with IQN + HDP features), builds full v3.3 feature vectors (61 features), handles EDL-A/B/C labeling via `EDLActionLabeler`. |
| `uncertainty/edl_decision_gate.py` | Rename/extend current `edl_gate.py` to accept the full `EDLActionResult` + `CombinedDecision` and produce `EDLGateResult`. Currently edl_gate.py is already complete; this may just be a thin wrapper or import alias. |

### 8.3 Runners

| File | Purpose |
|------|---------|
| `runner/run_combined_iqn_hierarchical_smoke_test.py` | Run IQN + HDP for N steps on demo_5 data; write `combined_iqn_hierarchical_decision_by_step.csv`. Required before EDL-C dataset can be built. |
| `runner/run_edl_action_dataset_v2_builder.py` | Build v3.3 EDL dataset from combined audit CSV. Outputs train/eval CSVs with 61-feature vectors + A/B/C labels. |
| `runner/run_edl_action_training_v2_smoke_test.py` | Train EDL v3.3 model with full feature set + configurable loss variant. Reports accuracy, vacuity, majority baseline, confusion matrix. |
| `runner/run_iqn_h_edl_end_to_end_smoke_test.py` | End-to-end: IQN → HDP → EDL → gate → audit. Reads trained EDL checkpoint, runs N steps, writes full per-step audit with gate decisions. |

---

## 9. Reference Repo Alignment: Final Verdict

### 9.1 Does Our EDL Loss Match the Reference?

**EDL-MSE:** ✅ **Yes, mathematically equivalent.**  
Our `dirichlet_mse_loss` + `kl_divergence_uniform` + `annealing_coeff` together reproduce the reference `mse_loss` precisely.

The only differences are cosmetic:
- Output shape: reference uses keepdim=True throughout; ours uses `(batch,)` — same after `.mean()`
- Activation placement: reference applies inside loss fn on raw logits; ours applies in `network.forward()` — both valid

**EDL-Log / EDL-Digamma:** ❌ **Not implemented.** Must add as `edl_reference_losses.py`.

### 9.2 Should IQN + HDP Be Integrated Before EDL-C?

**Yes — confirmed.** See Section 5 for full reasoning.

### 9.3 Exact Next Implementation Step

**Step 1 (immediate):** Implement `run_combined_iqn_hierarchical_smoke_test.py`  
This runner connects IQN (from existing `run_mode_b_repro_demo5_iqn.ps1` infrastructure) to `HierarchicalDecisionPolicy` and writes the combined audit CSV. This is the prerequisite for EDL-C and for enriching the feature vector with IQN distribution features.

**Step 2:** Implement `edl_reference_losses.py` — add log + digamma losses. Update `EDLActionConfig` with `loss_variant` field.

**Step 3:** Implement `edl_action_labeler.py` — unified, clean label generation replacing the scattered labeling in `edl_action_dataset.py`.

**Step 4:** Implement `edl_action_dataset_v2.py` — full 61-feature dataset builder from combined audit CSV.

**Step 5:** Implement `run_edl_action_training_v2_smoke_test.py` — retrain EDL with full features, compare loss variants, report evidence_succ/fail tracking.

**Step 6:** Implement `run_iqn_h_edl_end_to_end_smoke_test.py` — full pipeline from IQN to gate output.

---

*Document generated by Copilot CLI agent — analysis only, no source files modified.*
