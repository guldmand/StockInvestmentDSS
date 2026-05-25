# EDL v3.2 — Training Label Design

**Status:** Design only — no source code changed  
**Purpose:** Define A/B/C label generation strategies for supervised EDL training  

---

## 1. Overview

The EDL action classifier requires supervised training labels: for each decision step (date, state, features), a label `y ∈ {HOLD, BUY, SELL, REBALANCE[, CHANGE_STRATEGY]}`.

Three label generation strategies are planned:

| Variant | Name | Source | Horizon |
|---------|------|---------|---------|
| A | Hindsight oracle | Best realized risk-adjusted return ex-post | `HORIZON_DAYS` (default 20) |
| B | Rule / baseline policy | Transparent rule heuristic | Immediate |
| C | IQN / risk-policy teacher | Action selected by IQN + risk policy | Immediate (teacher signal) |

Each variant trains a separate EDL model checkpoint. Ensemble = weighted combination of A/B/C.

---

## 2. Point-in-Time Safety

**Critical rule for financial backtesting:**  
> Input features at time `t` must use ONLY information available at time `t`.  
> Labels may use outcomes at `t + h` (supervised training target) — this is expected and valid.  
> No feature at time `t` may encode any price, return, or announcement from after `t`.

This is guaranteed by:
- Features sourced exclusively from `hierarchical_decision_by_step.csv` (PIT-safe by construction)
- Labels generated separately from future price returns (never stored in feature vector)
- Dataset builder `EDLActionDataset` stores `(features_t, label_t)` with explicit timestamp join

---

## 3. EDL-A: Hindsight Oracle Labels

### Generation method

For each decision step at time `t`:
1. Record portfolio value `V_t`
2. Simulate each possible action `a ∈ {HOLD, BUY, SELL, REBALANCE}` for `h = HORIZON_DAYS` days
3. Compute realized return:  
   `R(a, t) = (V_{t+h}(a) - V_t) / V_t`
4. Compute risk-adjusted return:  
   `RA(a, t) = R(a, t) / σ(a, t)  [Sharpe-like, annualized]`
   or  
   `RA(a, t) = R(a, t) - λ * CVaR(a, t)  [CVaR-adjusted]`
5. Label:  
   `y_A = argmax_a RA(a, t)`

### CHANGE_STRATEGY label (when enabled)

`CHANGE_STRATEGY` gets the oracle label if a different strategy (e.g. defensive_v1 vs balanced_v1) would have produced strictly better `RA` over the horizon. This requires a multi-strategy simulation — planned for v4.0 but gated by `STOCK_INVESTMENT_DSS_EDL_INCLUDE_CHANGE_STRATEGY`.

### Strengths and limitations

| Strength | Limitation |
|----------|-----------|
| Uses actual realized outcomes | Requires future data simulation (offline training only) |
| Best-case oracle for supervised learning | May be noisy in sideways markets |
| Can validate against IQN predictions | Horizon `h` is a hyperparameter (default 20 days) |

### Configuration

```
STOCK_INVESTMENT_DSS_EDL_HORIZON_DAYS   = 20  (adjustable)
STOCK_INVESTMENT_DSS_EDL_LABEL_MODE     = hindsight
```

---

## 4. EDL-B: Rule / Baseline Policy Labels

### Generation method

For each decision step, apply a transparent rule heuristic to generate the label `y_B`:

```python
if portfolio.overweight(ticker, threshold=1.5 * strategy.max_position):
    y_B = REBALANCE
elif features.momentum_score < -0.3 and features.drawdown_from_recent_high < -0.12:
    y_B = SELL
elif portfolio.cash_weight > strategy.min_cash and features.momentum_score > 0.1:
    y_B = BUY
else:
    y_B = HOLD
```

Rules are parameterised and stored as a `RuleBasedLabelPolicy` configuration:

```yaml
rule_policy:
  overweight_threshold: 1.5     # multiple of max_position_weight
  sell_momentum_max: -0.3       # raw momentum (not normalised)
  sell_drawdown_max: -0.12      # drawdown_from_recent_high
  buy_momentum_min: 0.1
  buy_cash_min_fraction: 0.20   # relative to min_cash strategy param
  default_label: HOLD
```

### Strengths and limitations

| Strength | Limitation |
|----------|-----------|
| Fully interpretable | Learns the heuristic, not necessarily the best policy |
| No future data required | Rule quality directly limits model quality |
| Fast to generate at scale | May not generalise across strategies/regimes |

### REBALANCE detection

Overweight detection requires position weights from `PortfolioState`. These are available in the hierarchical audit's `portfolio_cash_weight` and `holding_values` fields. For v3.2, position weights are approximated from available audit data.

---

## 5. EDL-C: IQN / Risk-Policy Teacher Labels

### Generation method

`y_C = selected_action_type` as recorded in `hierarchical_decision_by_step.csv`.

The IQN model + hierarchical risk policy is the "teacher". EDL-C learns to reproduce the IQN+policy decision with calibrated uncertainty.

```
y_C = action selected by IQN risk policy at time t
    = hierarchical_decision_by_step.selected_action_type
```

This is the simplest label source: no simulation required.

### CHANGE_STRATEGY label (when enabled)

`CHANGE_STRATEGY` is not naturally produced by the current hierarchical policy (Stage 1 = HOLD/BUY/SELL/REBALANCE). For EDL-C in 5-class mode:
- A secondary rule fires if the strategy_id was changed between consecutive decisions (signal from audit log)
- Or: IQN strategy-switch trigger (out of distribution detection) → planned for v4.0

### Strengths and limitations

| Strength | Limitation |
|----------|-----------|
| No simulation required — labels from existing runs | Quality depends on IQN teacher quality |
| Primary production candidate | If IQN has HOLD bias (the original problem), EDL-C inherits it |
| Matches actual DSS behavior | Labels are deterministic given IQN — low label diversity unless IQN is well-calibrated |
| Enables offline dataset building from smoke test runs | REBALANCE and CHANGE_STRATEGY labels may be rare |

### Label imbalance note

The original D-IQN-DSS HOLD-collapse issue means EDL-C labels from early IQN training may be heavily biased toward HOLD. Mitigations:
1. Use runs where LayerNorm + q50 scoring was applied (post-fix seeds)
2. Apply weighted loss in EDL training (`class_weight` parameter)
3. Augment with synthetic minority class samples if needed (documented separately)

---

## 6. Dataset Builder Design

The `EDLActionDataset` (implemented in `edl_action_dataset.py`) loads from:

```
Source A: hierarchical_decision_by_step.csv  (features)
Source B: ticker_score_table.csv             (ticker features)
Source C: size_score_table.csv               (size features)
Source D: optional IQN action distribution   (q10/q50/q90/CVaR per action)
```

And generates labels from:
```
For EDL-A: future price return simulation (offline, requires market data)
For EDL-B: RuleBasedLabelPolicy.generate(features)
For EDL-C: hierarchical_decision_by_step.selected_action_type
```

### Output format (one row per decision step)

```csv
decision_id, date, feature_1, ..., feature_d, label_A, label_B, label_C
```

Labels are stored as integers (class index 0-3 or 0-4) for PyTorch `CrossEntropyLoss` compatibility.

### Train/val/test split

**CRITICAL: Always use temporal split for financial data — never random shuffle.**

```
train:   first 70% of timesteps (chronological)
val:     next 15%
test:    last 15%
```

Configurable via `edl_dataset_split` config dict:
```yaml
train_frac: 0.70
val_frac:   0.15
test_frac:  0.15
split_mode: temporal  # never random
```

---

## 7. Class Imbalance Strategy

### Expected label distribution (typical market data)

| Action | Typical frequency |
|--------|-----------------|
| HOLD | 60–80% (dominant — always a safe option) |
| BUY | 10–20% |
| SELL | 5–15% |
| REBALANCE | 2–10% |
| CHANGE_STRATEGY | < 5% |

### Mitigation options (in order of preference)

1. **Class-weighted loss**: `weight_k = N / (K * N_k)` — simple and standard
2. **Inverse frequency weighting**: emphasise rare classes in EDL loss
3. **Focal component**: down-weight easy HOLD examples in evidence loss
4. **Oversampling** rare classes during dataset construction
5. **Threshold adjustment**: adjust gate thresholds per-class based on calibration

Class weights are stored in the trained model checkpoint for reproducibility.

---

## 8. Horizon Sensitivity (EDL-A)

The oracle label depends on `h = HORIZON_DAYS`. Planned ablation:

| h (days) | Investment horizon | Notes |
|----------|-------------------|-------|
| 5  | Short-term (weekly) | Noisy; quick reversals |
| 20 | Medium-term (monthly) | **Default** |
| 60 | Long-term (quarterly) | Stable; may miss tactical signals |
| 252 | Annual | For strategic allocation comparison |

Results should be compared across horizons. See Phase 3 ablation plan.
