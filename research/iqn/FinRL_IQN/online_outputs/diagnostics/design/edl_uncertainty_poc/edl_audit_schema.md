# EDL Uncertainty PoC — Audit Schema

**Module:** `stockdss_patch_v3_1_edl_uncertainty_poc`
**Status:** Design phase — plan only

---

## 1. Output Files

The EDL smoke test produces three output files, appended to an existing hierarchical policy run directory or written to a new run directory:

```
audit/edl_uncertainty_by_decision.csv     ← main output, one row per decision step
summary/edl_uncertainty_summary.json      ← machine-readable aggregate summary
summary/edl_uncertainty_summary.md        ← human-readable summary
```

---

## 2. `edl_uncertainty_by_decision.csv` — Full Column Schema

### Identity columns

| Column | Type | Description |
|--------|------|-------------|
| `decision_id` | string | Matches `decision_id` from hierarchical_decision_by_step.csv |
| `date` | date string | Decision date (ISO format YYYY-MM-DD) |
| `visible_data_cutoff` | date string | PIT cutoff — same as hierarchical audit |
| `selected_action_type` | string | HOLD / BUY / SELL / REBALANCE |
| `selected_ticker` | string or empty | Ticker selected by Stage 2 (empty for HOLD) |
| `selected_size` | string or empty | Size bucket selected by Stage 3 (empty for HOLD) |

### Input feature columns (18 features — all normalised [0, 1])

| Column | Type | Description |
|--------|------|-------------|
| `feat_action_score_margin` | float | Margin between top and 2nd action score; 0.5 if absent |
| `feat_final_ticker_score` | float | Composite ticker score |
| `feat_score_variance` | float | Variance across 5 component scores; 0 = perfectly aligned |
| `feat_value_score` | float | |
| `feat_quality_score` | float | |
| `feat_profitability_score` | float | |
| `feat_momentum_score` | float | |
| `feat_risk_fit_score` | float | |
| `feat_q50` | float | Median IQN return estimate; 0.0 if IQN not connected |
| `feat_q_spread` | float | q90−q10 spread (aleatoric spread); 0.0 if absent |
| `feat_cvar` | float | CVaR; 0.0 if absent |
| `feat_risk_adj_fraction` | float | risk_adjusted_allocation_fraction |
| `feat_size_reduction_ratio` | float | ratio of final fraction to initial bucket fraction |
| `feat_cash_weight` | float | |
| `feat_max_concentration` | float | |
| `feat_drawdown` | float | 1 + drawdown_from_recent_high (inverted to [0,1]) |
| `feat_price_vs_ma50` | float | Clipped to [−1, 1], normalised to [0, 1] |
| `feat_price_vs_ma200` | float | Clipped to [−1, 1], normalised to [0, 1] |

### EDL Dirichlet quantities

| Column | Type | Description |
|--------|------|-------------|
| `evidence_high` | float ≥ 0 | Evidence for HIGH confidence class |
| `evidence_medium` | float ≥ 0 | Evidence for MEDIUM confidence class (prior = 0.5) |
| `evidence_low` | float ≥ 0 | Evidence for LOW confidence class |
| `alpha_high` | float ≥ 1 | `evidence_high + 1` (Dirichlet concentration parameter) |
| `alpha_medium` | float ≥ 1 | `evidence_medium + 1` |
| `alpha_low` | float ≥ 1 | `evidence_low + 1` |
| `dirichlet_strength` | float > 0 | `S = alpha_high + alpha_medium + alpha_low` |
| `vacuity` | float [0, 1] | `u = K / S` where K=3; high = high epistemic uncertainty |
| `prob_high` | float [0, 1] | `alpha_high / S` (expected probability of HIGH confidence) |
| `prob_medium` | float [0, 1] | `alpha_medium / S` |
| `prob_low` | float [0, 1] | `alpha_low / S` |

### Primary outputs

| Column | Type | Description |
|--------|------|-------------|
| `confidence_score` | float [0, 1] | `prob_high` — probability of HIGH confidence label |
| `uncertainty_score` | float [0, 1] | `vacuity` — epistemic uncertainty |
| `evidence_total` | float | `S - K` = total raw evidence (before prior shift) |
| `recommendation_confidence_label` | string | HIGH / MEDIUM / LOW |
| `uncertainty_warning` | string | Human-readable warning text; empty if none |
| `should_require_human_review` | bool | True if epistemic uncertainty is above threshold |

### Metadata / traceability

| Column | Type | Description |
|--------|------|-------------|
| `edl_model_version` | string | `"edl_poc_v3_1_placeholder_rule_based"` |
| `label_strategy` | string | `"placeholder_rule_based"` in v3.1 |
| `iqn_features_available` | bool | Whether q10/q50/q90/CVaR were present |
| `source` | string | `"edl_poc_placeholder"` — explicit PoC marker |

---

## 3. `recommendation_confidence_label` Thresholds

| Label | Condition | Interpretation |
|-------|-----------|---------------|
| `HIGH` | `confidence_score ≥ 0.65` AND `vacuity < 0.30` | Model is confident; scores align; low epistemic uncertainty |
| `MEDIUM` | `0.40 ≤ confidence_score < 0.65` OR `0.30 ≤ vacuity < 0.55` | Mixed signals; moderate confidence |
| `LOW` | `confidence_score < 0.40` OR `vacuity ≥ 0.55` | Conflicting signals; high epistemic uncertainty |

Note: These thresholds are placeholders for v3.1 and must be calibrated against real outcomes in v4.0.

---

## 4. `should_require_human_review` Trigger Logic

Human review is flagged (`True`) when **any** of the following:

| Trigger | Condition |
|---------|-----------|
| High vacuity | `vacuity ≥ 0.55` |
| LOW confidence label | `recommendation_confidence_label == "LOW"` |
| High score contradiction | `feat_score_variance > 0.15` (component scores diverge) |
| Bear-market + BUY | `feat_price_vs_ma200 < 0.45` AND `selected_action_type == "BUY"` |
| Extreme drawdown | `feat_drawdown < 0.75` (i.e., drawdown_from_recent_high < −0.25) |
| IQN not connected | `iqn_features_available == False` AND `selected_action_type != "HOLD"` |

---

## 5. `uncertainty_warning` Text Templates

| Condition | Warning text |
|-----------|-------------|
| `vacuity ≥ 0.55` | "High epistemic uncertainty: model evidence is insufficient to strongly support this recommendation." |
| `feat_score_variance > 0.15` | "Score contradiction detected: fundamental and technical signals are conflicting." |
| `feat_price_vs_ma200 < 0.45` + BUY | "Bear-market guard: ticker is trading below MA200. BUY recommendation has elevated uncertainty." |
| `feat_drawdown < 0.75` | "Significant drawdown detected: ticker is {drawdown_pct}% below recent high." |
| `iqn_features_available == False` | "IQN distribution features unavailable: confidence estimate is based on rule-based proxy only." |
| No triggers | `""` (empty) |

Multiple warnings are concatenated with `" | "`.

---

## 6. `edl_uncertainty_summary.json` Schema

```json
{
  "run_id": "...",
  "source_run_directory": "outputs/runs/...",
  "n_decisions": 5,
  "edl_model_version": "edl_poc_v3_1_placeholder_rule_based",
  "label_strategy": "placeholder_rule_based",
  "iqn_features_available": false,
  "confidence_distribution": {
    "HIGH": 3,
    "MEDIUM": 1,
    "LOW": 1
  },
  "human_review_required_count": 1,
  "mean_confidence_score": 0.71,
  "mean_uncertainty_score": 0.24,
  "mean_evidence_total": 4.12,
  "per_decision": [
    {
      "decision_id": "...",
      "date": "...",
      "selected_action_type": "BUY",
      "selected_ticker": "GOOGL",
      "recommendation_confidence_label": "HIGH",
      "confidence_score": 0.78,
      "uncertainty_score": 0.19,
      "should_require_human_review": false,
      "uncertainty_warning": ""
    }
  ]
}
```

---

## 7. `edl_uncertainty_summary.md` Content Template

```markdown
# EDL Uncertainty Summary — <run_id>
Date: <timestamp>
Source run: <source_run_directory>

## Confidence Distribution
- HIGH: N decisions (X%)
- MEDIUM: N decisions (X%)
- LOW: N decisions (X%)

## Uncertainty Stats
- Mean confidence score: X.XX
- Mean epistemic uncertainty (vacuity): X.XX
- Decisions requiring human review: N

## Per-Decision Results
| Date | Action | Ticker | Label | Confidence | Vacuity | Review? |
| ...  |

## Notes
- IQN distribution features available: True/False
- Label strategy: placeholder_rule_based
- All outputs marked source=edl_poc_placeholder
- v3.1 is a deterministic rule-based PoC. Full EDL requires training.
```
