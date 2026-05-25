# EDL v3.5 — Counterfactual Hindsight Oracle Plan

**Date:** 2026-05-22  
**Session:** b9e47002  
**Status:** Design — pending implementation

---

## 1. Motivation: Why the First EDL-A Labeler is Only a Smoke Test

The v3.4 threshold labeler (`edl_hindsight_labeler.py`) assigns labels based solely on the forward outcome for `selected_ticker`:
- BUY if `future_return >= buy_threshold AND future_max_drawdown >= drawdown_threshold`
- SELL if `future_return <= sell_threshold OR future_max_drawdown < drawdown_threshold`
- HOLD otherwise

**Limitations:**
1. **Single-action perspective.** The label ignores what would have happened if the agent had chosen a *different* action. A HOLD label assigned when a stock rises 2% (below buy_threshold) might be wrong — there was a better BUY opportunity.
2. **Threshold sensitivity.** The thresholds (±3%, −8%) are arbitrary. Small changes cause large label swings.
3. **HOLD as residual.** HOLD is assigned by elimination, not by active evidence that holding was better than buying or selling.
4. **No counterfactual for SELL.** Selling means liquidating a position. The threshold labeler does not compare "if I had sold, would I have avoided a loss?" vs "if I had held, what would have happened?"

The threshold labeler is useful as a first integration smoke test but is insufficient as a training signal for a thesis-quality EDL-A classifier.

---

## 2. Counterfactual Labels: Why They Are Better

A counterfactual oracle explicitly computes a **score for each possible action** at decision time t and assigns the label that corresponds to the **best risk-adjusted action**.

This is analogous to offline policy evaluation:
- "What was the best action I could have taken at time t, given perfect hindsight?"

**Advantages:**
1. **Action-relative ranking.** The label is the action that maximally dominates alternatives — not a threshold judgment on the chosen action's outcome alone.
2. **Threshold-free primary criterion.** The label is determined by argmax of per-action scores, not by whether a single return exceeds a fixed threshold.
3. **SELL gets affirmative evidence.** SELL is labeled when selling truly avoids a loss — i.e., when the ticker's forward return is strongly negative and holding cash (score ≈ 0) would have been better.
4. **HOLD is affirmative, not residual.** HOLD is labeled when the ticker return is close to zero (no clear opportunity either way) or when the margin between BUY and SELL scores is below `min_label_margin`.

---

## 3. Class Space Decision

**EDL-A class space for first training run: K=3**

| Class | Label ID |
|-------|----------|
| HOLD | 0 |
| BUY | 1 |
| SELL | 2 |

**Why REBALANCE is excluded:**
- REBALANCE is a portfolio-structure action (cross-asset rebalancing, cash-equity ratio adjustment)
- It cannot be evaluated by forward return of a single ticker over a fixed horizon
- REBALANCE is handled by the HDP risk/strategy layer independently
- Including REBALANCE in EDL-A would introduce label noise and misalign the class space with the learnable signal
- REBALANCE remains available as a gate output from HDP; EDL-A supervises HOLD/BUY/SELL uncertainty only

---

## 4. Counterfactual Score Design

### Notation
- `t` = decision date
- `h` = forward horizon in trading days (default: 20)
- `close[t]` = closing price of selected_ticker at date t
- `close[t+h]` = closing price at t+h
- `min_close[t..t+h]` = minimum closing price over the window

### Per-action score computation

**BUY score** — "if I had bought selected_ticker at t, what risk-adjusted return would I have gotten?"
```
buy_future_return = close[t+h] / close[t] - 1
buy_max_drawdown = min(close[t..t+h]) / close[t] - 1
BUY_score = buy_future_return - lambda_drawdown * abs(min(0, buy_max_drawdown))
```
If `selected_ticker` is None/UNKNOWN: `BUY_score = NaN` → label unavailable

**SELL score** — "if I had sold (liquidated to cash) at t, what risk-adjusted value would I have preserved?"
```
# Selling at t = receiving close[t], holding cash through [t..t+h]
# Value of staying in cash vs staying in the position:
# Cash alternative: 0.0 return (risk-free proxy, ignoring interest)
# Position if NOT sold: buy_future_return
# SELL advantage = 0.0 - buy_future_return (avoided the loss, or gave up the gain)
SELL_score = -1 * buy_future_return
# No drawdown penalty on SELL: SELL avoids the drawdown
```
If `selected_ticker` is None/UNKNOWN: `SELL_score = 0.0` (cash; no position to sell)

**HOLD score** — "if I had done nothing, what is the baseline?"
```
# HOLD = no action = cash / existing exposure unchanged
# Modeled as 0.0 (neutral baseline)
HOLD_score = 0.0
```

### Risk-adjusted aggregate (for diagnostics)
```
risk_adjusted_future_score = buy_future_return - lambda_drawdown * abs(min(0, buy_max_drawdown))
```
This is the same as BUY_score and is stored as a summary column for analysis.

### Label assignment
```
scores = {HOLD: 0.0, BUY: BUY_score, SELL: SELL_score}
best_action = argmax(scores)
second_best = second highest score
margin = best_score - second_best_score

if margin < min_label_margin:
    label = HOLD  # ambiguous: no clear winner → conservative HOLD
    reason = "ambiguous_margin"
else:
    label = best_action
    reason = "counterfactual_argmax"
```

### Special cases

| Condition | Label | Reason |
|-----------|-------|--------|
| selected_ticker is None/UNKNOWN AND action is HOLD | HOLD | `no_ticker_hold_preserved` |
| selected_ticker is None/UNKNOWN AND action is not HOLD | UNAVAILABLE | `no_ticker_non_hold` |
| t+h > last available date in market data | UNAVAILABLE | `insufficient_future_data` |
| margin < min_label_margin | HOLD (ambiguous) | `ambiguous_margin` |

---

## 5. Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| horizon_days | 20 | 20 trading days ≈ 1 calendar month |
| lambda_drawdown | 0.5 | Penalty weight for intra-period drawdown |
| min_label_margin | 0.005 | 0.5% score margin required for non-HOLD label |
| class_space | HOLD,BUY,SELL | REBALANCE excluded |

---

## 6. Point-in-Time Safety Rules

1. **Future data for labels only.** `buy_future_return`, `buy_max_drawdown`, and all score columns are computed from future market data. They must **never** be included in the EDL input feature matrix.
2. **Input features are past-only.** The 37 EDL feature columns (IQN quantiles, HDP scores, portfolio context) are all computed from data available at decision time t.
3. **Label generation is offline.** The oracle runs after all market data is available. During live inference, EDL uses only pre-computed features — not the oracle scores.
4. **The label CSV is training data only.** The combined-with-counterfactual-labels CSV must not be used during the evaluation phase without careful PIT isolation.

---

## 7. Implementation Plan

### New files to create

| File | Purpose |
|------|---------|
| `src/stock_investment_dss/uncertainty/edl_counterfactual_hindsight_oracle.py` | Core oracle: TickerPriceIndex, CounterfactualConfig, score computation, label assignment |
| `src/stock_investment_dss/runner/run_edl_counterfactual_hindsight_labeling_smoke_test.py` | Runner: read combined audit + market data, run oracle, write outputs |
| `docs/EDL_Counterfactual_Hindsight_Oracle_v3_5.md` | Design documentation |

### Output schema additions (new columns on combined audit)

```
edl_a_cf_label                  — HOLD / BUY / SELL
edl_a_cf_label_id               — 0 / 1 / 2
edl_a_cf_label_available        — True / False
edl_a_cf_label_reason           — reason string
edl_a_cf_horizon_days           — int
edl_a_cf_buy_score              — float
edl_a_cf_sell_score             — float
edl_a_cf_hold_score             — float (always 0.0)
edl_a_cf_best_score             — float
edl_a_cf_second_best_score      — float
edl_a_cf_margin                 — float
edl_a_cf_future_return_pct      — float (buy_future_return * 100)
edl_a_cf_future_max_drawdown_pct — float (buy_max_drawdown * 100)
edl_a_cf_risk_adjusted_future_score — float
```

---

## 8. Expected Label Distribution

Based on the v3.4 threshold labeler (BUY=100, HOLD=114, SELL=57) and the counterfactual logic, the expected distribution is broadly similar but with:
- Fewer HOLD (margin-based ambiguous cases may shift to BUY or SELL)
- More precise BUY/SELL assignment (not threshold-gated)
- Ambiguous rows shifted to HOLD (margin < 0.005)

A balanced 3-class distribution (rough target: BUY≈30–40%, HOLD≈30–40%, SELL≈20–30%) is desirable for EDL-A training.

---

## 9. Next Steps After Oracle Implementation

1. **Run smoke test** to verify oracle output
2. **Build EDL-A training dataset** from oracle labels + existing 37-feature matrix
3. **Train EDL-A classifier** using reference-aligned digamma loss + KL annealing
4. **Compare EDL-A vs EDL-C gate behavior** — does hindsight-trained EDL disagree with IQN more meaningfully than imitation-trained EDL?
5. **Thesis evaluation:** Report EDL-A vacuity distribution, label accuracy, gate intervention rate, portfolio performance with/without gate
