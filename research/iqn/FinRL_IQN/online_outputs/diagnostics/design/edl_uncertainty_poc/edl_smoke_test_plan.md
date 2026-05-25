# EDL Uncertainty PoC — Smoke Test Plan

**Module:** `stockdss_patch_v3_1_edl_uncertainty_poc`
**Status:** Design phase — plan only

---

## 1. Purpose

The smoke test validates that the EDL uncertainty layer:
1. Reads from existing hierarchical policy audit output without errors
2. Computes EDL-inspired confidence/vacuity values for each decision
3. Produces correct output files with the expected schema
4. Handles both BUY and HOLD decisions correctly
5. Flags human review where appropriate

The smoke test does NOT require a trained IQN model or any training run.

---

## 2. Input: Existing Hierarchical Policy Run Output

The smoke test reads from one of the confirmed hierarchical policy smoke test runs.

### Recommended input source (BUY test)
```
outputs/runs/2026_05_21_031912_d_iqn_dss_hierarchical_policy_smoke_test/
```

### Recommended input source (HOLD test)
```
outputs/runs/2026_05_21_033219_d_iqn_dss_hierarchical_policy_smoke_test/
```

### Files read by the smoke test

| File | Purpose |
|------|---------|
| `audit/hierarchical_decision_by_step.csv` | Action type, HOLD sub-type, portfolio state, risk checks |
| `audit/ticker_score_table.csv` | Per-ticker component scores (may be absent for HOLD) |
| `audit/size_score_table.csv` | Size allocation scores (may be absent for HOLD) |
| `data/hierarchical_technical_features.csv` | MA50/MA200/drawdown/momentum per date×ticker |

---

## 3. Run Commands

```powershell
# Set Python path
$env:PYTHONPATH = "src"

# BUY smoke test — reads from BUY hierarchical run
$env:STOCK_INVESTMENT_DSS_EDL_SOURCE_RUN = "outputs/runs/2026_05_21_031912_d_iqn_dss_hierarchical_policy_smoke_test"
python -m stock_investment_dss.runner.run_edl_uncertainty_smoke_test

# HOLD smoke test — reads from HOLD hierarchical run
$env:STOCK_INVESTMENT_DSS_EDL_SOURCE_RUN = "outputs/runs/2026_05_21_033219_d_iqn_dss_hierarchical_policy_smoke_test"
python -m stock_investment_dss.runner.run_edl_uncertainty_smoke_test

# Date-controlled BUY smoke test (2024 window)
$env:STOCK_INVESTMENT_DSS_EDL_SOURCE_RUN = "outputs/runs/2026_05_21_033232_d_iqn_dss_hierarchical_policy_smoke_test"
python -m stock_investment_dss.runner.run_edl_uncertainty_smoke_test
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STOCK_INVESTMENT_DSS_EDL_SOURCE_RUN` | Latest hierarchical run (auto-detect) | Path to source hierarchical run directory |
| `STOCK_INVESTMENT_DSS_EDL_OUTPUT_DIR` | Same as source run | Override output directory |
| `STOCK_INVESTMENT_DSS_EDL_HUMAN_REVIEW_THRESHOLD` | `0.55` | Vacuity threshold for human review flag |
| `STOCK_INVESTMENT_DSS_EDL_CONFIDENCE_HIGH_THRESHOLD` | `0.65` | Minimum prob_high for HIGH label |

---

## 4. Expected Output Files

Written to the source run directory (or `EDL_OUTPUT_DIR` if set):

```
audit/edl_uncertainty_by_decision.csv
summary/edl_uncertainty_summary.json
summary/edl_uncertainty_summary.md
```

---

## 5. Scenario Test Cases

### Test Case 1 — BUY with GOOGL (default window)
**Input:** `2026_05_21_031912` run, action=BUY, ticker=GOOGL, size=BUY_25, 5 steps
**Expected behavior:**
- `feat_final_ticker_score` > 0.5 (GOOGL ranked top)
- `feat_momentum_score` positive
- No IQN features (IQN not connected) → `iqn_features_available=False`
- `uncertainty_warning` should include IQN not-connected note
- `recommendation_confidence_label` expected: MEDIUM (GOOGL has good scores but IQN absent)
- `should_require_human_review` expected: True (IQN absent + BUY action)

**Pass criteria:**
- [ ] `edl_uncertainty_by_decision.csv` has 5 rows
- [ ] `vacuity` column is populated and between 0 and 1
- [ ] `recommendation_confidence_label` is one of HIGH/MEDIUM/LOW for each row
- [ ] `should_require_human_review` is True for all BUY rows (IQN absent)
- [ ] `uncertainty_warning` contains IQN-not-connected text

---

### Test Case 2 — HOLD cash-only
**Input:** `2026_05_21_033219` run, action=HOLD, cash=1.0, 5 steps
**Expected behavior:**
- No ticker scores (absent for HOLD) → all `feat_*_score` at 0.5 (neutral)
- HOLD with 100% cash is a valid conservative state → should not be flagged as LOW confidence
- `selected_ticker` and `selected_size` empty
- `recommendation_confidence_label` expected: MEDIUM to HIGH (HOLD is well-supported by cash constraint)
- `should_require_human_review` expected: False (HOLD is a valid IQN decision)

**Pass criteria:**
- [ ] `edl_uncertainty_by_decision.csv` has 5 rows
- [ ] No row has `recommendation_confidence_label == "LOW"` for a clean HOLD
- [ ] `feat_final_ticker_score == 0.5` (neutral, ticker absent)
- [ ] `should_require_human_review == False` for HOLD_IQN_SELECTED

---

### Test Case 3 — SELL with AMZN
**Input:** `2026_05_21_031936` run, action=SELL, ticker=AMZN, size=SELL_100, 5 steps
**Expected behavior:**
- SELL scoring uses weakness ranking → `feat_final_ticker_score` reflects weakness (lower → higher SELL confidence)
- SELL_100 (full exit) with cash=0.2 (20% cash, 80% invested) → large allocation fraction
- `recommendation_confidence_label` expected: MEDIUM to HIGH if AMZN scored as clear weakest holding
- `should_require_human_review` expected: False if scoring is consistent

**Pass criteria:**
- [ ] `edl_uncertainty_by_decision.csv` has 5 rows
- [ ] SELL action handled without error
- [ ] `feat_size_reduction_ratio ≈ 1.0` (SELL_100 not penalised if holding is weak)

---

### Test Case 4 — BUY with date-controlled 2024 window
**Input:** `2026_05_21_033232` run, action=BUY, step 4-5 ticker switches GOOGL→MSFT
**Expected behavior:**
- Feature vector differs between GOOGL steps and MSFT steps
- `recommendation_confidence_label` may differ across steps
- Ticker switch visible in `selected_ticker` column

**Pass criteria:**
- [ ] `edl_uncertainty_by_decision.csv` has 5 rows
- [ ] Rows with MSFT have different `feat_final_ticker_score` than GOOGL rows
- [ ] No crash on ticker switch between steps

---

## 6. Pass/Fail Summary Criteria

| Criterion | Expected result |
|-----------|----------------|
| All 3 output files generated | ✅ |
| No crash for any of the 3 input scenarios | ✅ |
| Schema correct (all columns present) | ✅ |
| `vacuity ∈ [0, 1]` for all rows | ✅ |
| `confidence_score ∈ [0, 1]` for all rows | ✅ |
| `alpha_high + alpha_medium + alpha_low == dirichlet_strength` | ✅ |
| `prob_high + prob_medium + prob_low ≈ 1.0` | ✅ |
| `source == "edl_poc_placeholder"` on all rows | ✅ |
| `label_strategy == "placeholder_rule_based"` on all rows | ✅ |
| HOLD rows have neutral ticker features | ✅ |
| `should_require_human_review == True` when IQN absent + BUY | ✅ |
| No training run | ✅ |
| No modification to existing src files | ✅ |

---

## 7. What is NOT Tested in v3.1

| Not tested | Why |
|------------|-----|
| Calibration of confidence thresholds | Requires real outcome labels |
| EDL loss function (UCE + KL) | Requires training loop |
| IQN distribution features | IQN not connected in smoke test |
| Multi-step confidence drift | Would need longer simulation |
| Bear-market scenario | Would need to construct a bear-market input manually |

These are v4.0 items.

---

## 8. Implementation Checklist for v3.1

Before running smoke tests, the following src files must be implemented:

- [ ] `src/stock_investment_dss/uncertainty/__init__.py`
- [ ] `src/stock_investment_dss/uncertainty/edl_classifier.py` — reads feature groups, computes α/S/u
- [ ] `src/stock_investment_dss/uncertainty/recommendation_confidence.py` — label logic, warning text, human review flag
- [ ] `src/stock_investment_dss/runner/run_edl_uncertainty_smoke_test.py` — orchestrator
- [ ] py_compile passes on all 4 files
- [ ] All 4 test cases pass
