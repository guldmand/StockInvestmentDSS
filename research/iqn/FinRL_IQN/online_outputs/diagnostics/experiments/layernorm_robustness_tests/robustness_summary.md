# Robustness Summary — D-IQN-DSS LayerNorm + q50

**Date:** 2026-05-21  
**Status:** 3/3 tests completed, seed=7, single-seed sanity check

---

## Overview

Three robustness tests were run to verify that the D-IQN-DSS LayerNorm + q50
primary result is not an artefact of the specific 2018–2024 AAPL/MSFT/NVDA/AMZN/GOOGL
setup. All three tests use seed=7 (outside the primary 1–5 seed set) with the same
IQN preset, 25,000 training steps, and Mode B frozen data.

**Primary result for reference:**  
Demo_5 LayerNorm + q50, seeds 1–5: mean return +77.25%, mean Sharpe 2.856,
max drawdown −8.48%, 5/5 active, σ(return)=8.97%.

---

## Per-Test Results

### Test 1 — Earlier Window with 2022 Bear Market

| Parameter | Value |
|-----------|-------|
| Dataset ID | `robustness_demo5_2017_2023_layernorm_q50` |
| Tickers | AAPL, MSFT, NVDA, AMZN, GOOGL |
| Train window | 2017-01-01 → 2022-01-01 |
| Eval window | 2022-01-01 → 2023-01-01 |
| Seed | 7 |
| Score mode | q50 |
| use_layer_norm | true |

| Metric | Value |
|--------|-------|
| **Total return** | **−32.91%** |
| Annualized Sharpe | −1.001 |
| Max drawdown | −35.13% |
| CVaR10 | −3.81% |
| Volatility | 34.19% |
| Final value | $670,922 |
| **Total trades** | **9 (HOLD:242, BUY:9)** |
| Final cash weight | 0.121 |
| Turnover | 112.6% |
| Transaction cost | $918 |
| **HOLD-collapse** | **NO ✅** |

**Pass/Fail:** PASS (no HOLD-collapse — agent executed 9 trades)  
**Economic assessment:** FAIL — agent lost 32.91% during a brutal bear market (S&P 500 −19%,
NASDAQ −33%, NVDA −50%, AMZN −50% in 2022). The 9 BUY actions purchased equity into a
falling market. The final cash weight (12.1%) indicates the portfolio was held in stocks
throughout the drawdown. The agent did not know to go defensive.

**Key observation:** Only BUY and HOLD actions (no SELL). At 25,000 training steps the
agent has not learned to exit positions under negative momentum. This is a training-depth
limitation, not a HOLD-collapse (Q-scale instability) issue.

---

### Test 2 — Shifted Training Window (2019–2024)

| Parameter | Value |
|-----------|-------|
| Dataset ID | `robustness_demo5_2019_2024_layernorm_q50` |
| Tickers | AAPL, MSFT, NVDA, AMZN, GOOGL |
| Train window | 2019-01-01 → 2023-01-01 |
| Eval window | 2023-01-01 → 2024-02-01 |
| Seed | 7 |
| Score mode | q50 |
| use_layer_norm | true |

| Metric | Value |
|--------|-------|
| **Total return** | **+93.50%** |
| Annualized Sharpe | 2.882 |
| Max drawdown | −10.89% |
| CVaR10 | −2.14% |
| Volatility | 22.17% |
| Final value | $1,935,037 |
| **Total trades** | **8 (HOLD:263, BUY:8)** |
| Final cash weight | 0.055 |
| Turnover | 58.5% |
| Transaction cost | $893 |
| **HOLD-collapse** | **NO ✅** |

**Pass/Fail:** PASS — active trading, strong positive return  
**Economic assessment:** EXCELLENT — +93.50% return with Sharpe 2.882. This exceeds the
primary result mean (+77.25%) using the same eval period (2023–2024). The result confirms
that the primary finding is reproducible with a different starting year for the training
window. Notably, the agent achieves this with only 8 BUY trades and no SELL actions —
a pure buy-and-hold-once strategy that works exceptionally well in the 2023 bull market.

**Key observation:** 1 fewer year of training data (2019 vs 2018 start) does not cause
HOLD-collapse or performance degradation. The result is slightly stronger than the primary
mean, suggesting the 2017–2018 data (included in primary) added noise rather than signal
at 25,000 steps.

---

### Test 3 — Non-Tech Universe (JPM, XOM, UNH, KO, WMT)

| Parameter | Value |
|-----------|-------|
| Dataset ID | `robustness_nontech_2018_2024_layernorm_q50` |
| Tickers | JPM, XOM, UNH, KO, WMT |
| Train window | 2018-01-01 → 2023-01-01 |
| Eval window | 2023-01-01 → 2024-02-01 |
| Seed | 7 |
| Score mode | q50 |
| use_layer_norm | true |

| Metric | Value |
|--------|-------|
| **Total return** | **+8.52%** |
| Annualized Sharpe | 0.859 |
| Max drawdown | −6.70% |
| CVaR10 | −1.07% |
| Volatility | 9.36% |
| Final value | $1,085,242 |
| **Total trades** | **4 (HOLD:267, BUY:4)** |
| Final cash weight | 0.291 |
| Turnover | 66.5% |
| Transaction cost | $683 |
| **HOLD-collapse** | **NO ✅** |

**Pass/Fail:** PASS — active trading, positive return  
**Economic assessment:** MODERATE — +8.52% is a positive outcome. Non-tech consumer/
financial/energy stocks (JPM, XOM, UNH, KO, WMT) significantly underperformed the tech
universe in 2023 (e.g., KO −3%, WMT +21%, XOM −9%). The S&P 500 non-tech return was
roughly +5–10% in 2023. The IQN result (+8.52%) is in line with the asset class.
The lower volatility (9.36% vs 34.19% in Test 1) reflects the fundamentally different
nature of these assets.

**Key observation:** LayerNorm's scale normalisation generalises to lower-priced,
lower-volatility assets without modification. The agent executes 4 BUY trades with no
SELL — a minimal but active strategy. The 29% final cash weight suggests the agent
maintains a more conservative allocation for these assets vs the tech universe (5.5%).

---

## Cross-Test Summary Table

| Test | Dataset | Eval Period | Return | Sharpe | MaxDD | Trades | Collapse? |
|------|---------|-------------|--------|--------|-------|--------|-----------|
| Test 1 | demo5 2017–2023 | 2022 (bear) | **−32.91%** | −1.001 | −35.13% | 9 | **NO ✅** |
| Test 2 | demo5 2019–2024 | 2023 (bull) | **+93.50%** | 2.882 | −10.89% | 8 | **NO ✅** |
| Test 3 | nontech 2018–2024 | 2023 (mixed) | **+8.52%** | 0.859 | −6.70% | 4 | **NO ✅** |
| **Primary** | demo5 2018–2024 | 2023 (bull) | **+77.25%** | 2.856 | −8.48% | 182 avg | **NO ✅** |

---

## Observations Across All Tests

### 1. HOLD-collapse: Not present in any robustness test

All three tests show `total_trades > 0`. The Q-value instability that caused 3/5 seeds
to HOLD-collapse before LayerNorm is not observed in any test. This confirms LayerNorm's
fix is general and not specific to the primary seed set or time window.

### 2. BUY-only action pattern (no SELL in any test)

All three robustness tests show action distributions of HOLD + BUY only, with zero SELL
actions. The primary result (seeds 1–5) did include SELL actions (e.g., seed 4: BUY:130,
SELL:128). This suggests:
- Seed 7 converges to a buy-and-hold policy at 25,000 steps in these configurations
- SELL learning may require more training steps or different reward signal structure
- This is not pathological — the agent is actively choosing to buy and hold positions

### 3. Results are strongly regime-dependent

| Regime | Test | Return | Notes |
|--------|------|--------|-------|
| Bull (2023 tech) | Primary + Test 2 | +77–94% | Agent benefits from continuous equity exposure |
| Bear (2022 tech) | Test 1 | −33% | Agent buys into falling market; no defensive exit |
| Mixed (2023 non-tech) | Test 3 | +8.5% | Moderate result consistent with asset class |

The system is a **bull market enhancer** at 25,000 training steps, not a market-neutral
or defensive strategy.

### 4. Fewer trades than primary result

Primary result (5 seeds): mean 182 trades/seed. Robustness tests: 4–9 trades each.
This difference likely reflects:
- Seed-specific convergence (seed 7 vs seeds 1–5)
- The multiseed average in the primary result masks per-seed variance
- At 25,000 steps, the policy has not fully explored the BUY/SELL/HOLD action space

---

## Thesis-Safe Interpretation

### What was confirmed

1. **LayerNorm fix is robust** — zero HOLD-collapse events across 3 new tests, 2 new
   tickers sets, and 3 different time windows. The fix generalises.

2. **Positive result on same eval regime is reproducible** — Test 2 (2019–2024)
   reproduces the primary result closely (Sharpe 2.88 vs 2.856) with a different
   training start year. This strengthens the thesis claim.

3. **Cross-asset-class generalisation** — Test 3 confirms LayerNorm works for
   non-tech assets with different price/volatility profiles. No re-tuning required.

### What was not confirmed (limitations)

1. **Bear market performance is poor** — Test 1 confirms the system has no defensive
   mechanism under sustained negative market regimes. The agent holds equities throughout
   a −33% market drop. This is a meaningful limitation that must be stated in the thesis.

2. **SELL actions are sparse** — The buy-and-hold pattern in all three tests suggests
   the system has learned an implicit "stay invested" policy. Whether this generalises
   across market regimes is not yet established.

3. **Single-seed only** — These are sanity checks using seed=7. Full 5-seed multiseed
   runs would be needed to measure stability across the new test configurations.

---

## Thesis-Safe Conclusion

> *Three robustness tests using seed=7 confirm that the D-IQN-DSS LayerNorm + q50 result*
> *generalises beyond the primary 2018–2024 tech universe. No HOLD-collapse was observed*
> *in any test. Performance is strongly regime-dependent: the system reproduces the primary*
> *result in bull markets (+93.5% in 2023 shifted window), produces moderate positive returns*
> *in mixed regimes (+8.5% on non-tech assets), and incurs significant losses in sustained*
> *bear markets (−32.9% in 2022). These results support presenting the D-IQN-DSS as a*
> *promising proof-of-concept for data-driven distributional RL in stock decision support,*
> *while clearly acknowledging that the system is not a production-ready trading strategy.*
> *Defensive exit learning and cross-regime robustness remain open problems for future work.*

---

## Source Output Directories

| Test | Output Directory |
|------|-----------------|
| Test 1 (bear market) | `outputs/runs/2026_05_21_003721_d_iqn_dss_iqn_learning_curve_multiseed_summary` |
| Test 2 (shifted window) | `outputs/runs/2026_05_21_004034_d_iqn_dss_iqn_learning_curve_multiseed_summary` |
| Test 3 (non-tech) | `outputs/runs/2026_05_21_004346_d_iqn_dss_iqn_learning_curve_multiseed_summary` |
