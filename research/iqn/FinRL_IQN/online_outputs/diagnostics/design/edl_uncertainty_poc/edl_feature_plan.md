# EDL Uncertainty PoC — Feature Plan

**Module:** `stockdss_patch_v3_1_edl_uncertainty_poc`
**Status:** Design phase — plan only

---

## 1. Feature Groups

The EDL classifier ingests 18 input features from the hierarchical policy audit output. They are organised into 5 groups, each contributing a distinct type of evidence signal.

### Group 1 — Action / Decision Features (from Stage 1)

| Feature | Source | Type | Notes |
|---------|--------|------|-------|
| `selected_action_type` | audit CSV Stage 1 | categorical (HOLD/BUY/SELL/REBALANCE) | Encoded as one-hot |
| `action_score_margin` | audit CSV Stage 1 | float [0, 1] | Margin between top and second action score; optional — if absent, set to 0.5 |

**Evidence signal:** A large action score margin means IQN is clearly preferring one action type → higher confidence. A near-50/50 margin means IQN is ambiguous → lower confidence.

---

### Group 2 — Ticker Score Features (from Stage 2)

| Feature | Source | Type | Notes |
|---------|--------|------|-------|
| `final_ticker_score` | ticker_score_table.csv | float [0, 1] | Composite weighted score |
| `value_score` | ticker_score_table.csv | float [0, 1] | From FundamentalFeatureStore |
| `quality_score` | ticker_score_table.csv | float [0, 1] | |
| `profitability_score` | ticker_score_table.csv | float [0, 1] | |
| `momentum_score` | ticker_score_table.csv | float [0, 1] | From TechnicalFeatureBuilder |
| `risk_fit_score` | ticker_score_table.csv | float [0, 1] | |

**Evidence signal:** If all 5 component scores are high and aligned (low variance across components), confidence is high. If scores conflict (e.g., strong value but very weak momentum), epistemic uncertainty rises because the model's signals are contradictory.

**Contradiction metric (derived):**
```
score_variance = Var(value, quality, profitability, momentum, risk_fit)
```
Low variance + high mean → HIGH confidence evidence
High variance → LOW confidence evidence (even if mean is acceptable)

---

### Group 3 — IQN Distribution Features (from Stage 1, optional)

| Feature | Source | Type | Notes |
|---------|--------|------|-------|
| `q10` | IQN quantile output | float | 10th percentile return estimate |
| `q50` | IQN quantile output | float | Median return estimate |
| `q90` | IQN quantile output | float | 90th percentile return estimate |
| `cvar` | IQN CVaR output | float | Conditional Value at Risk |

**Optional:** These are only available when IQN is connected. In v3.1 smoke test, they will be absent (set to `None` / filled with neutral placeholder values 0.0).

**Evidence signal when available:**
- `q90 - q10` = IQN return spread = aleatoric uncertainty proxy
  - Narrow spread → high conviction in return magnitude → evidence boost
  - Wide spread → high aleatoric uncertainty → evidence reduction
- `q50 > 0` = positive expected return → BUY confidence boost
- `cvar < -0.15` = tail risk high → confidence reduction

**Mapping to EDL framework:** IQN aleatoric features reduce `S` (total Dirichlet strength) when return uncertainty is high. This correctly propagates: a wide IQN distribution → lower overall confidence → higher `u` (vacuity).

---

### Group 4 — Size / Allocation Features (from Stage 3)

| Feature | Source | Type | Notes |
|---------|--------|------|-------|
| `selected_size` | size_score_table.csv | categorical (BUY_25..SELL_100) | Encoded as fraction |
| `risk_adjusted_allocation_fraction` | size_score_table.csv | float [0, 1] | Post-risk-adjustment fraction |

**Evidence signal:** A large `risk_adjusted_allocation_fraction` means the system committed to a larger position — implicitly signalling the size selector was also confident. If the fraction was severely reduced from the initial bucket (e.g., BUY_75 → 0.12 fraction after risk adjustments), this signals multiple risk flags fired → lower confidence.

**Derived feature:**
```
size_reduction_ratio = risk_adjusted_allocation_fraction / initial_bucket_fraction
```
Ratio close to 1.0 → size not significantly penalised → confidence maintained
Ratio < 0.5 → multiple risk penalties applied → confidence reduced

---

### Group 5 — Portfolio / Market / Risk Features (from Stage 4 + TechnicalFeatureBuilder)

| Feature | Source | Type | Notes |
|---------|--------|------|-------|
| `cash_weight` | portfolio_state | float [0, 1] | Current cash fraction |
| `max_concentration` | portfolio_state | float [0, 1] | Max single-ticker weight |
| `drawdown_from_recent_high` | technical features | float [−1, 0] | Negative → drawdown |
| `price_vs_ma50` | technical features | float [−1, 1] | Relative to MA50 |
| `price_vs_ma200` | technical features | float [−1, 1] | Relative to MA200 |

**Evidence signal:**
- `price_vs_ma200 < -0.05` → bear territory → reduced confidence for BUY
- `drawdown_from_recent_high < -0.15` → significant drawdown → uncertainty increase
- `max_concentration > 0.4` → concentration risk → uncertainty increase (portfolio over-exposed)
- `cash_weight < 0.15` → near min cash buffer → uncertainty increase (tight liquidity)

---

## 2. Feature Vector Construction

For each decision in the audit CSV, the feature vector `x` is assembled as:

```
x = [
    action_score_margin,        # Group 1
    final_ticker_score,         # Group 2
    score_variance,             # Group 2 (derived)
    value_score,
    quality_score,
    profitability_score,
    momentum_score,
    risk_fit_score,
    q50_or_0,                   # Group 3 (optional)
    q90_minus_q10_or_0,
    cvar_or_0,
    risk_adjusted_allocation_fraction,  # Group 4
    size_reduction_ratio,       # Group 4 (derived)
    cash_weight,                # Group 5
    max_concentration,
    drawdown_from_recent_high,
    price_vs_ma50,
    price_vs_ma200
]
```

Total dimension: 18 (base) — all floats after encoding.

---

## 3. Normalisation

All features are normalised to [0, 1] before evidence computation:
- Positive-orientation: higher = more confidence (e.g., `final_ticker_score`)
- Negative-orientation: inverted (e.g., `drawdown_from_recent_high` → `1 + drawdown`)
- Clipping: all values clipped to [0, 1] after normalisation
- Missing/None → 0.5 (neutral placeholder)

---

## 4. Evidence Accumulation (EDL-Inspired)

The v3.1 PoC uses a **rule-based Dirichlet approximation** rather than a trained network.

For a K=3 class problem (LOW / MEDIUM / HIGH confidence):

```
e_HIGH = w_scores × (final_ticker_score + action_score_margin)
       + w_iqn    × (q50_positive_signal + narrow_spread_signal)
       + w_risk   × (no_bear_market + low_drawdown + low_concentration)

e_LOW  = 1 - e_HIGH (simplified for PoC)

α_HIGH = e_HIGH + 1
α_LOW  = e_LOW  + 1
α_MED  = 1.5    (prior — medium confidence is always slightly supported)

S      = α_HIGH + α_MED + α_LOW
u      = K / S  (vacuity — epistemic uncertainty)
p̂_HIGH = α_HIGH / S
```

**Note:** In a full v4.0 EDL model, `e_i` values are the outputs of a neural network trained with the UCE loss + KL Dirichlet regulariser from NeurIPS 2018. In v3.1, `e_i` values are deterministic rule-based computations.

---

## 5. Missing Feature Handling

| Situation | Handling |
|-----------|---------|
| IQN not connected (v3.1 smoke test) | q10/q50/q90/CVaR → 0.0 (neutral contribution) |
| No ticker selected (HOLD action) | All ticker scores → 0.5; action is HOLD → confidence determined by portfolio state only |
| No size selected (HOLD action) | size_reduction_ratio → 1.0 (no penalty) |
| SELL action (no BUY ticker score) | Group 2 uses SELL ranking score instead |

---

## 6. Feature Source Files

| Feature group | Source file(s) |
|---------------|---------------|
| Group 1 | `audit/hierarchical_decision_by_step.csv` |
| Group 2 | `audit/ticker_score_table.csv` |
| Group 3 | `audit/hierarchical_decision_by_step.csv` (optional columns) |
| Group 4 | `audit/size_score_table.csv` |
| Group 5 | `audit/hierarchical_decision_by_step.csv` + `data/hierarchical_technical_features.csv` |
