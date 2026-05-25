# EDL v3.2 — Audit Output Schema

**Status:** Design only — no source code changed  
**Purpose:** Full specification of all audit output fields for EDL v3.2 decisions  

---

## 1. Primary Audit File

**File:** `audit/edl_action_uncertainty_by_decision.csv`  
One row per decision step.

---

## 2. Column Definitions

### 2.1 Decision Identity

| Column | Type | Description |
|--------|------|-------------|
| `decision_id` | string | Unique ID from hierarchical policy audit |
| `date` | ISO date | Decision date (visible data cutoff = same) |
| `visible_data_cutoff` | ISO date | PIT cutoff date (no future data used) |
| `source_run` | string | Name of source hierarchical run directory |

### 2.2 Selected Action (from hierarchical policy)

| Column | Type | Description |
|--------|------|-------------|
| `selected_action_type` | HOLD/BUY/SELL/REBALANCE[/CHANGE_STRATEGY] | Hierarchical policy decision |
| `selected_ticker` | string or "" | Selected ticker (empty for HOLD/REBALANCE) |
| `selected_size` | string or "" | Size label (e.g. BUY_25, SELL_100) |
| `original_selected_fraction` | float [0,1] | Allocation fraction before EDL gate |

### 2.3 EDL Configuration

| Column | Type | Description |
|--------|------|-------------|
| `edl_enabled` | bool | Was EDL active for this decision? |
| `edl_variant` | string | none/A/B/C/AB/AC/BC/ABC |
| `edl_gate_enabled` | bool | Was the EDL gate active? |
| `hierarchical_policy_enabled` | bool | Was hierarchical policy active? |
| `action_classes` | string (JSON list) | e.g. `["HOLD","BUY","SELL","REBALANCE"]` |
| `edl_model_version` | string | Model checkpoint identifier or "v3_2_placeholder" |
| `label_strategy` | string | hindsight / rules / iqn_teacher / placeholder |

### 2.4 EDL Primary Outputs

| Column | Type | Description |
|--------|------|-------------|
| `edl_predicted_action` | string | `argmax_k p_k` — EDL's top predicted action |
| `edl_agrees_with_selected_action` | bool | `edl_predicted_action == selected_action_type` |
| `selected_action_probability` | float [0,1] | `p_{selected}` = alpha_{selected} / S |
| `selected_action_evidence` | float ≥ 0 | `e_{selected}` = alpha_{selected} - 1 |
| `selected_action_belief` | float [0,1] | `b_{selected}` = e_{selected} / S |
| `selected_action_uncertainty` | float [0,1] | `u = K / S` (same for all classes) |

### 2.5 Per-Action Probabilities

| Column | Type | Description |
|--------|------|-------------|
| `p_hold` | float [0,1] | alpha_HOLD / S |
| `p_buy` | float [0,1] | alpha_BUY / S |
| `p_sell` | float [0,1] | alpha_SELL / S |
| `p_rebalance` | float [0,1] | alpha_REBALANCE / S |
| `p_change_strategy` | float [0,1] or null | alpha_CS / S (null if K=4 mode) |

### 2.6 Per-Action Evidence

| Column | Type | Description |
|--------|------|-------------|
| `evidence_hold` | float ≥ 0 | e_HOLD = alpha_HOLD - 1 |
| `evidence_buy` | float ≥ 0 | e_BUY |
| `evidence_sell` | float ≥ 0 | e_SELL |
| `evidence_rebalance` | float ≥ 0 | e_REBALANCE |
| `evidence_change_strategy` | float ≥ 0 or null | e_CS (null if K=4) |

### 2.7 Per-Action Alpha (Dirichlet Concentration)

| Column | Type | Description |
|--------|------|-------------|
| `alpha_hold` | float ≥ 1 | e_HOLD + 1 |
| `alpha_buy` | float ≥ 1 | e_BUY + 1 |
| `alpha_sell` | float ≥ 1 | e_SELL + 1 |
| `alpha_rebalance` | float ≥ 1 | e_REBALANCE + 1 |
| `alpha_change_strategy` | float ≥ 1 or null | e_CS + 1 (null if K=4) |

### 2.8 Dirichlet Summary

| Column | Type | Description |
|--------|------|-------------|
| `dirichlet_strength` | float > K | S = Σ alpha_k |
| `uncertainty_vacuity` | float [0,1] | u = K / S |
| `evidence_total` | float ≥ 0 | S - K (net evidence above uniform prior) |

### 2.9 Ensemble Outputs (when variant is AB/AC/BC/ABC)

| Column | Type | Description |
|--------|------|-------------|
| `ensemble_uncertainty` | float [0,1] | Weighted average: Σ w_i * u_i |
| `ensemble_uncertainty_conservative` | float [0,1] | max(u_A, u_B, u_C) |
| `model_disagreement_score` | float [0,1] | Fraction of A/B/C that disagree with selected action |
| `ensemble_weights` | string (JSON) | e.g. `{"A": 0.33, "B": 0.33, "C": 0.34}` |

### 2.10 EDL Gate Outputs

| Column | Type | Description |
|--------|------|-------------|
| `recommendation_gate` | string | RECOMMEND_AS_IS / REDUCE_SIZE / FORCE_HOLD / HUMAN_REVIEW / STRATEGY_REVIEW |
| `should_reduce_size` | bool | Gate issued size reduction |
| `should_force_hold` | bool | Gate overrode action to HOLD |
| `should_require_human_review` | bool | Gate flagged for human review |
| `final_action_after_edl_gate` | string | Action after gate override (may differ from selected) |
| `final_size_after_edl_gate` | string | Size label after gate reduction |
| `final_fraction_after_edl_gate` | float [0,1] | Allocation fraction after gate |
| `reason_codes` | string (pipe-separated) | e.g. `high_vacuity|edl_disagrees|bear_market_buy` |

### 2.11 Score Function Components (when EDL gate enabled)

| Column | Type | Description |
|--------|------|-------------|
| `uncertainty_penalty` | float | λ_u * u applied to score function |
| `disagreement_penalty` | float | λ_d * disagreement_score |
| `edl_score_adjustment_total` | float | Total adjustment to IQN score from EDL |

### 2.12 Input Features (traceability)

All features used by the EDL network are stored as `feat_*` columns:

**Group A — IQN features** (null if IQN not connected)

| Column | Type |
|--------|------|
| `feat_iqn_q10_{action}` | float for each action |
| `feat_iqn_q50_{action}` | float for each action |
| `feat_iqn_q90_{action}` | float for each action |
| `feat_iqn_cvar_{action}` | float for each action |
| `feat_iqn_selected_vs_hold_margin` | float |
| `feat_iqn_selected_vs_second_margin` | float |
| `feat_iqn_available` | bool |

**Group B — Hierarchical policy features**

| Column | Type |
|--------|------|
| `feat_final_ticker_score` | float |
| `feat_value_score` | float |
| `feat_quality_score` | float |
| `feat_profitability_score` | float |
| `feat_momentum_score` | float |
| `feat_risk_fit_score` | float |
| `feat_risk_adj_fraction` | float |
| `feat_bear_market_signal` | bool |
| `feat_bear_market_penalty` | float |
| `feat_max_position_ok` | bool |
| `feat_cash_buffer_ok` | bool |
| `feat_concentration_ok` | bool |
| `feat_drawdown_guard_ok` | bool |
| `feat_ma200_trend_ok` | bool |

**Group C — Technical / fundamental**

| Column | Type |
|--------|------|
| `feat_rsi_30` | float |
| `feat_macd` | float |
| `feat_price_vs_ma50` | float |
| `feat_price_vs_ma200` | float |
| `feat_drawdown_from_recent_high` | float |
| `feat_recent_return_5d` | float |
| `feat_recent_return_20d` | float |
| `feat_volatility_20d` | float |
| `feat_revenue_growth` | float |
| `feat_profit_margin` | float |
| `feat_pe_ratio` | float |
| `feat_debt_ratio` | float |

**Group D — Portfolio / risk**

| Column | Type |
|--------|------|
| `feat_cash_weight` | float |
| `feat_max_concentration` | float |
| `feat_current_drawdown` | float |
| `feat_risk_profile` | string (categorical encoded) |
| `feat_strategy_id` | string (categorical encoded) |

---

## 3. Summary JSON Schema

**File:** `summary/edl_action_uncertainty_summary.json`

```json
{
  "run_id": "<timestamp>_d_iqn_dss_edl_action_inference_smoke_test",
  "source_run_directory": "<hierarchical run name>",
  "n_decisions": 5,
  "edl_enabled": true,
  "edl_variant": "C",
  "edl_gate_enabled": true,
  "edl_model_version": "edl_v3_2_action_C",
  "label_strategy": "iqn_teacher",
  "iqn_features_available": false,
  "action_classes": ["HOLD","BUY","SELL","REBALANCE"],
  "action_distribution": {
    "HOLD": 0,
    "BUY": 5,
    "SELL": 0,
    "REBALANCE": 0
  },
  "edl_predicted_distribution": {
    "HOLD": 0,
    "BUY": 4,
    "SELL": 1,
    "REBALANCE": 0
  },
  "edl_agreement_rate": 0.80,
  "gate_distribution": {
    "RECOMMEND_AS_IS": 2,
    "REDUCE_SIZE": 2,
    "FORCE_HOLD": 0,
    "HUMAN_REVIEW": 1,
    "STRATEGY_REVIEW": 0
  },
  "human_review_required_count": 1,
  "force_hold_count": 0,
  "mean_vacuity": 0.42,
  "mean_selected_probability": 0.58,
  "mean_uncertainty_penalty": 0.21,
  "mean_disagreement_score": 0.00,
  "per_decision": [...]
}
```

---

## 4. Summary Markdown Template

**File:** `summary/edl_action_uncertainty_summary.md`

```markdown
# EDL Action Uncertainty Summary — <run_id>

Generated: <timestamp>
EDL variant: <variant>
Gate enabled: <true/false>

## Action Agreement

| | Hierarchical selected | EDL predicted |
|-|-----------------------|---------------|
| HOLD | 0 | 0 |
| BUY  | 5 | 4 |
| SELL | 0 | 1 |
| REBALANCE | 0 | 0 |

EDL agreement rate: 80%

## Uncertainty Statistics

- Mean vacuity: 0.42
- Mean selected action probability: 0.58
- Mean uncertainty penalty: 0.21
- Decisions requiring human review: 1 / 5

## Gate Decisions

| Gate | Count |
|------|-------|
| RECOMMEND_AS_IS | 2 |
| REDUCE_SIZE | 2 |
| FORCE_HOLD | 0 |
| HUMAN_REVIEW | 1 |
| STRATEGY_REVIEW | 0 |

## Per-Decision Results

| Date | Selected | EDL Predicted | Agree | p_selected | Vacuity | Gate | Reason |
|...   | ...      | ...           | ...   | ...        | ...     | ...  | ...    |
```

---

## 5. Backwards Compatibility Note

The v3.1 output files used:
- `recommendation_confidence_label` (LOW/MEDIUM/HIGH) — **not present in v3.2 primary output**
- `confidence_score` — replaced by `selected_action_probability`
- `uncertainty_score` — replaced by `uncertainty_vacuity`

v3.1 outputs remain in their own run directories under `*_edl_uncertainty_smoke_test/` naming.  
v3.2 outputs use `*_edl_action_inference_smoke_test/` naming.  
No file conflict exists.
