# HDP + EDL Feature Coverage Audit

**Date:** 2026-05-22  
**Session:** b9e47002  
**Phase:** EDL v3.4 / EDL-A Hindsight Labeling preparation

---

## 1. Sources Audited

| Source | Path |
|--------|------|
| Combined IQN+HDP audit CSV | `outputs/runs/2026_05_21_172639_d_iqn_dss_combined_iqn_hierarchical_smoke_test/audit/combined_iqn_hierarchical_decision_by_step.csv` |
| EDL v2 dataset summary | `outputs/runs/2026_05_21_172652_d_iqn_dss_edl_action_dataset_v2_builder/summary/edl_v2_dataset_summary.json` |
| Market data (reference) | `data/market/daily/imports/market_data_full_500.csv` |

---

## 2. Combined Audit Column List (71 columns)

```
decision_id, date, visible_data_cutoff, eval_step, source_iqn_run_id, dataset_id, pit_split_id,
selected_iqn_action, selected_iqn_action_index, iqn_chosen_action, iqn_chosen_action_index,
iqn_q10_hold, iqn_q25_hold, iqn_q50_hold, iqn_q75_hold, iqn_q90_hold, iqn_cvar10_hold, iqn_score_hold, iqn_mean_hold,
iqn_q10_buy, iqn_q25_buy, iqn_q50_buy, iqn_q75_buy, iqn_q90_buy, iqn_cvar10_buy, iqn_score_buy, iqn_mean_buy,
iqn_q10_sell, iqn_q25_sell, iqn_q50_sell, iqn_q75_sell, iqn_q90_sell, iqn_cvar10_sell, iqn_score_sell, iqn_mean_sell,
iqn_q10_rebalance, iqn_q25_rebalance, iqn_q50_rebalance, iqn_q75_rebalance, iqn_q90_rebalance, iqn_cvar10_rebalance, iqn_score_rebalance, iqn_mean_rebalance,
iqn_q10_changestrategy, iqn_q25_changestrategy, iqn_q50_changestrategy, iqn_q75_changestrategy, iqn_q90_changestrategy, iqn_cvar10_changestrategy, iqn_score_changestrategy, iqn_mean_changestrategy,
iqn_action_margin,
hierarchical_action_type, selected_ticker, selected_size, selected_size_fraction, ticker_score, size_score,
price, ma50, ma200, price_vs_ma50, price_vs_ma200, momentum_score, value_score, quality_score, risk_score,
cash_weight, final_recommendation_before_edl, final_recommendation_source,
edl_a_hindsight_label, edl_a_future_return_horizon, edl_a_future_return_pct,
edl_b_rule_label, edl_c_teacher_label
```

---

## 3. EDL v2 Feature Matrix (37 columns used for training)

| Group | Features |
|-------|---------|
| IQN quantiles/stats — HOLD | `iqn_q10_hold` `iqn_q25_hold` `iqn_q50_hold` `iqn_q75_hold` `iqn_q90_hold` `iqn_cvar10_hold` `iqn_score_hold` |
| IQN quantiles/stats — BUY | `iqn_q10_buy` `iqn_q25_buy` `iqn_q50_buy` `iqn_q75_buy` `iqn_q90_buy` `iqn_cvar10_buy` `iqn_score_buy` |
| IQN quantiles/stats — SELL | `iqn_q10_sell` `iqn_q25_sell` `iqn_q50_sell` `iqn_q75_sell` `iqn_q90_sell` `iqn_cvar10_sell` `iqn_score_sell` |
| IQN quantiles/stats — REBALANCE | `iqn_q10_rebalance` `iqn_q25_rebalance` `iqn_q50_rebalance` `iqn_q75_rebalance` `iqn_q90_rebalance` `iqn_cvar10_rebalance` `iqn_score_rebalance` |
| IQN decision margin | `iqn_action_margin` |
| HDP context | `selected_size_fraction` `cash_weight` |
| HDP ticker scores | `momentum_score` `value_score` `quality_score` `risk_score` |
| Technical ratios | `price_vs_ma50` `price_vs_ma200` |

---

## 4. Feature Presence Classification

### Technical Indicators

| Feature | In Combined Audit | In EDL Features | Status |
|---------|:-----------------:|:---------------:|--------|
| ma50 / close_30_sma | ✅ (266/271 non-null) | ❌ | PRESENT_IN_COMBINED_AUDIT |
| ma200 | ✅ (266/271 non-null) | ❌ | PRESENT_IN_COMBINED_AUDIT |
| price_vs_ma50 | ✅ (266/271 non-null) | ✅ | PRESENT_IN_EDL_FEATURES |
| price_vs_ma200 | ✅ (266/271 non-null) | ✅ | PRESENT_IN_EDL_FEATURES |
| macd | ❌ | ❌ | MISSING |
| rsi_30 | ❌ | ❌ | MISSING |
| cci_30 | ❌ | ❌ | MISSING |
| dx_30 | ❌ | ❌ | MISSING |

> **Note:** `macd`, `rsi_30`, `cci_30`, `dx_30` ARE present in the raw market data file but are **not forwarded** through the combined audit pipeline. They are available as join-enrichment from market data if needed.

### HDP Composite Scores

| Feature | In Combined Audit | In EDL Features | Status |
|---------|:-----------------:|:---------------:|--------|
| momentum_score | ✅ (266/271 non-null) | ✅ | PRESENT_IN_EDL_FEATURES |
| value_score | ✅ (254/271 non-null) | ✅ | PRESENT_IN_EDL_FEATURES |
| quality_score | ✅ (254/271 non-null) | ✅ | PRESENT_IN_EDL_FEATURES |
| risk_score | ✅ (254/271 non-null) | ✅ | PRESENT_IN_EDL_FEATURES |
| valuation_score | ❌ | ❌ | MISSING |

> **Note:** `value_score`/`quality_score`/`risk_score` have 17 null rows because these correspond to HOLD steps where `selected_ticker` is None (no position taken, no ticker-level scores computed).

### Fundamental Features

| Feature | In Combined Audit | In EDL Features | Status | Notes |
|---------|:-----------------:|:---------------:|--------|-------|
| p/e (pe_ratio) | ❌ | ❌ | MISSING | Not in pipeline at all |
| p/s (ps_ratio) | ❌ | ❌ | MISSING | Not in pipeline at all |
| revenue_growth | ❌ | ❌ | MISSING | Not in pipeline at all |
| earnings_growth | ❌ | ❌ | MISSING | Not in pipeline at all |
| profit_margin | ❌ | ❌ | MISSING | Not in pipeline at all |
| free_cash_flow | ❌ | ❌ | MISSING | Not in pipeline at all |
| debt_ratio | ❌ | ❌ | MISSING | Not in pipeline at all |
| sector | ❌ | ❌ | MISSING | Not in pipeline at all |

> **Critical note:** `FundamentalFeatureStore` uses **frozen placeholder values** for 5 tickers (AAPL, MSFT, NVDA, AMZN, GOOGL) with `source="frozen_snapshot_placeholder"`. These placeholder scalars are fused into HDP composite scores (`quality_score`, `value_score`, `risk_score`) but are **not exposed as raw columns** in the audit CSV.

### Aggregated Score Status

| Feature | In Combined Audit | In EDL Features | Status |
|---------|:-----------------:|:---------------:|--------|
| ticker_score | ✅ (column present but 0/271 non-null) | ❌ | PLACEHOLDER_OR_UNKNOWN |
| size_score | ✅ (column present but 0/271 non-null) | ❌ | PLACEHOLDER_OR_UNKNOWN |

> `ticker_score` and `size_score` columns exist in combined audit but are entirely null/zero — they are allocated placeholders not yet populated by the HDP runner.

---

## 5. Summary Table

| Feature | Classification |
|---------|---------------|
| ma50 / ma200 | PRESENT_IN_COMBINED_AUDIT (not in EDL features) |
| price_vs_ma50 / price_vs_ma200 | PRESENT_IN_EDL_FEATURES |
| macd | MISSING (in market data, not forwarded) |
| rsi_30 | MISSING (in market data, not forwarded) |
| cci_30 | MISSING (in market data, not forwarded) |
| dx_30 | MISSING (in market data, not forwarded) |
| pe_ratio | MISSING |
| ps_ratio | MISSING |
| revenue_growth | MISSING |
| earnings_growth | MISSING |
| profit_margin | MISSING |
| free_cash_flow | MISSING |
| debt_ratio | MISSING |
| sector | MISSING |
| quality_score | PRESENT_IN_EDL_FEATURES (partially placeholder-sourced) |
| valuation_score | MISSING |
| momentum_score | PRESENT_IN_EDL_FEATURES |
| risk_score | PRESENT_IN_EDL_FEATURES |
| ticker_score | PLACEHOLDER_OR_UNKNOWN (null in audit) |
| size_score | PLACEHOLDER_OR_UNKNOWN (null in audit) |
| value_score | PRESENT_IN_EDL_FEATURES (partially placeholder-sourced) |

---

## 6. Maturity Assessment

### What is mature
- **29 IQN distributional features** (quantiles q10/q25/q50/q75/q90, CVaR, score for all 4 actions + margin) — fully populated, meaningful for EDL-A
- **2 HDP portfolio context features** (`selected_size_fraction`, `cash_weight`) — fully populated
- **4 HDP composite scores** (`momentum_score`, `value_score`, `quality_score`, `risk_score`) — present for all rows with a selected ticker (200/271 rows)
- **2 technical ratio features** (`price_vs_ma50`, `price_vs_ma200`) — present for all active-ticker rows
- **Market data technical indicators** (`macd`, `rsi_30`, `cci_30`, `dx_30`) are available via join-enrichment if the combined runner is extended

### What is missing / immature
- **All raw fundamental metrics** (P/E, P/S, revenue_growth, earnings_growth, profit_margin, etc.) are absent from the pipeline entirely
- **Composite scores fuse placeholder fundamentals** — `quality_score` and `value_score` partly reflect frozen snapshot values, not real PIT fundamentals
- **Technical oscillators** (MACD, RSI, CCI, ADX) are not forwarded to the combined audit

### EDL-A readiness for first training run

**Recommendation: Option A — Proceed with EDL-A training using current technical/IQN/portfolio features only**

Rationale:
1. The 37 current EDL features (IQN quantiles, HDP context, composite scores, price ratios) are sufficient for a first EDL-A proof-of-concept
2. Adding raw fundamentals or forwarding MACD/RSI/CCI to the audit is a data-pipeline task that should not block first EDL-A training
3. For the thesis, it is more important to demonstrate the full EDL-A pipeline (hindsight labels → supervised training → gate) with the current feature set than to wait for richer features
4. Feature enrichment can proceed in parallel and the EDL-A training run can be repeated once additional features are available

**Conditions for Option A:**
- Acknowledge in thesis that fundamentals are placeholder-sourced in quality/value scores
- Note that MACD/RSI/CCI/ADX enrichment remains as future work
- Ensure the EDL-A counterfactual labels use market data correctly (no look-ahead via input features)

---

## 7. Action Items

| Priority | Item | Blocking EDL-A? |
|----------|------|:---------------:|
| Low | Forward `macd`, `rsi_30`, `cci_30`, `dx_30` from market data into combined audit via join-enrichment | No |
| Low | Replace placeholder fundamentals with real PIT-safe fundamentals from external API | No |
| Low | Populate `ticker_score` and `size_score` columns in combined audit runner | No |
| None | Wait for all fundamental features before first EDL-A run | — |
