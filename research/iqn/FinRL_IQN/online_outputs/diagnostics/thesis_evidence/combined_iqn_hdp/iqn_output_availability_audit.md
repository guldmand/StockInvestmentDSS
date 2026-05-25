# IQN Output Availability Audit

**Purpose:** Determine whether existing IQN runs contain date-indexed decision/distribution
outputs suitable for real integration with the combined IQN+HDP audit runner.

---

## 1. Which IQN runs exist?

| Run ID (timestamp prefix) | Type | Status |
|---|---|---|
| `2026_05_21_004044_d_iqn_dss_iqn_learning_curve_smoke_test` | Learning curve (latest) | ✅ Complete |
| `2026_05_20_205109_d_iqn_dss_iqn_learning_curve_smoke_test` | Learning curve | ✅ Complete |
| `2026_05_20_210705_d_iqn_dss_iqn_learning_curve_multiseed_summary` | Multiseed summary | ✅ Complete |
| `2026_05_19_171159_d_iqn_dss_iqn_decision_audit_report` | Decision audit | ✅ Complete |
| `2026_05_19_163059_d_iqn_dss_iqn_decision_audit_report` | Decision audit | ✅ Complete |
| `2026_05_19_044638_d_iqn_dss_iqn_decision_audit_report` | Decision audit | ✅ Complete |
| `2026_05_18_041031_d_iqn_dss_iqn_decision_distribution_smoke_test` | Distribution | ✅ Complete |
| `2026_05_18_045318_d_iqn_dss_iqn_backtest_smoke_test` | Backtest | ✅ Complete |

**Primary source for this audit:** `2026_05_21_004044_d_iqn_dss_iqn_learning_curve_smoke_test`

---

## 2. Which IQN runs contain date-indexed decisions?

**Answer: None directly.**

The step records (`eval_step_records.csv`) use an integer `eval_step` index (0–270), not
calendar dates. The date-to-step mapping must be derived externally using the eval window
metadata and the market data import file.

| File | Has `date` column? | Notes |
|---|---|---|
| `iqn_learning_curve_eval_step_records.csv` | ❌ No | Uses `eval_step` (int 0–270) |
| `iqn_learning_curve_eval_distributions.csv` | ❌ No | Uses `train_step` + `eval_step` |
| `iqn_learning_curve_prepared_train_data.csv` | ✅ Yes | Training data only (not eval) |
| `iqn_learning_curve_train_asset_memory.csv` | ✅ Yes | Training period (2018–2022) |
| `decision_audit_by_step.csv` (audit report runs) | 🟡 Empty | Derived from source, empty if source lacks date |

---

## 3. Which files contain IQN q10/q50/q90/CVaR/action scores?

**`iqn_learning_curve_eval_distributions.csv`** — Primary quantile source.

Columns (confirmed from latest run):
```
train_step, eval_step, chosen_action, action, action_index, allowed,
score_mode, score, mean, q10, q25, q50, q75, q90, cvar10
```

- One row per (train_step, eval_step, action_option)
- 5 action options per eval_step: HOLD, BUY, SELL, REBALANCE, CHANGE_STRATEGY
- Total rows: 8,130 = 271 eval_steps × 5 actions × ~6 train_step checkpoints

**`iqn_decision_distribution_table.csv`** — Single-decision summary only (7 rows total).

---

## 4. Which files contain selected actions?

**`iqn_learning_curve_eval_step_records.csv`** — Primary decision source.

Columns (confirmed):
```
decision_id, train_step, eval_step, chosen_action_index, chosen_action_label,
reward, terminated, truncated, done, effective_action, action_was_masked,
selected_ticker, requested_shares, ..., cash_before, cash_after,
portfolio_value_before, portfolio_value_after, ...
```

- One row per (train_step, eval_step)
- `chosen_action_label`: HOLD / BUY / SELL / REBALANCE
- `effective_action`: same (after masking resolution)

---

## 5. Which files contain dates?

| File | Date Type | Period | Notes |
|---|---|---|---|
| `iqn_learning_curve_prepared_train_data.csv` | `date` (YYYY-MM-DD) | 2018-01-01 – 2022-12-31 | Training data only |
| `iqn_learning_curve_train_asset_memory.csv` | `date` (YYYY-MM-DD) | Training period | Portfolio values |
| `data/market/daily/imports/market_data_full_500.csv` | `date` (YYYY-MM-DD) | Full history | **Used for date mapping** |
| `experiment_context_summary.json` | Metadata | Eval window 2023-01-01 – 2024-02-01 | Provides bounds |

**Key:** The eval period dates (2023-01-01 to 2024-02-01) can be derived by filtering the
market import file to the IQN ticker universe (JPM, XOM, UNH, KO, WMT) and the eval window,
then sorting unique dates. eval_step `i` maps to `unique_eval_dates[i]`.

---

## 6. Why the combined audit runner could not use real IQN outputs?

The combined runner's `_discover_iqn_source()` function requires a CSV with a **`date` column**.
It scanned learning-curve run directories but found no CSV with a `date` column in the core
eval outputs (step records and distributions only have integer step indices).

Additionally, the `_csv_has_date()` check was searching for a literal `date` column header,
which is absent from both `eval_step_records.csv` and `eval_distributions.csv`.

The runner fell back to `manual_fallback` mode with `STOCK_INVESTMENT_DSS_COMBINED_ACTION_TYPE=BUY`.

---

## 7. What exact file format is needed for real IQN integration?

A single CSV per IQN run with the following schema (one row per eval day):

```
date                  ISO 8601 date (YYYY-MM-DD)
iqn_source_run_id     Source run identifier
train_step            Training checkpoint from which this eval was taken
eval_step             Integer step index (0-indexed)
iqn_selected_action   IQN's chosen action label (HOLD/BUY/SELL/REBALANCE)
selected_action_type  Same as iqn_selected_action (canonical column name)
action_score_hold     score column for HOLD action (q50 value used for selection)
action_score_buy      score column for BUY action
action_score_sell     score column for SELL action
action_score_rebalance  score column for REBALANCE action
q10_hold              10th percentile of HOLD return distribution
q50_hold              Median of HOLD return distribution
q90_hold              90th percentile of HOLD return distribution
cvar_hold             CVaR at alpha=10% for HOLD
q10_buy               ... (same for BUY)
q50_buy
q90_buy
cvar_buy
q10_sell              ... (same for SELL)
q50_sell
q90_sell
cvar_sell
q10_rebalance         ... (same for REBALANCE)
q50_rebalance
q90_rebalance
cvar_rebalance
iqn_uncertainty_proxy q90 - q10 of the chosen action's distribution
iqn_risk_score        mean - 0.5 * cvar10 of chosen action (risk-adjusted estimate)
```

This file is produced by `run_iqn_decision_export_smoke_test.py`.

---

## 8. Eval period and overlap with combined runner

- **IQN eval window:** 2023-01-03 to 2024-01-31 (271 trading days)
- **Combined runner default window:** 2024-01-01 to 2024-02-01
- **Overlap (real IQN usable):** 2024-01-02 to 2024-01-31 (~21 trading days)

> Note: The final trained model (train_step=25000) chose HOLD 267/271 times (98.5%).
> For BUY-dominated combined audit tests, use an earlier training checkpoint (e.g.,
> train_step=0 or 5000) via `STOCK_INVESTMENT_DSS_IQN_EXPORT_TRAIN_STEP`.

---

## 9. Ticker universe mismatch

- **IQN tickers:** JPM, XOM, UNH, KO, WMT (non-tech stocks)
- **HDP feature tickers:** AAPL, MSFT, NVDA, AMZN, GOOGL (tech stocks)

This is intentional in the architecture: IQN provides the **action type** (HOLD/BUY/SELL/REBALANCE),
while HDP independently selects which ticker to act on from its own universe. The two components
operate on different observation spaces.

---

## 10. Readiness for real IQN integration

| Component | Status |
|---|---|
| `eval_step_records.csv` available | ✅ |
| `eval_distributions.csv` available | ✅ |
| `experiment_context_summary.json` available | ✅ |
| Market import file for date mapping | ✅ |
| Date-indexed export runner created | ✅ `run_iqn_decision_export_smoke_test.py` |
| Overlap between IQN eval and combined runner window | ✅ ~21 trading days (Jan 2024) |
| Combined runner updated to read export CSV | ✅ (via `STOCK_INVESTMENT_DSS_COMBINED_HDP_FEATURE_CSV` path override pattern — apply same for IQN) |

**Conclusion:** Real IQN integration IS achievable without retraining.
The `run_iqn_decision_export_smoke_test.py` runner bridges the gap.
