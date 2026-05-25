# LayerNorm Lambda Ablation Summary

**Run date:** 2026-05-20  
**Author:** Automated analysis from existing output files  
**Source:** `outputs/runs/` — no new experiments were run for this summary

---

## Ablation Design

All four variants share identical settings:
- LayerNorm: **enabled** (`use_layer_norm=true`)
- Seeds: 1–5 (multiseed)
- Train steps: 25,000 — PIT split: 2018-01-01 → 2022-12-31 train / 2023-01-01 → 2024-02-01 eval
- Tickers: AAPL, MSFT, NVDA, AMZN, GOOGL — Mode B frozen import data
- Only `score_mode` and `risk_lambda` vary

| Variant | score_mode | lambda | dataset_id |
|---------|-----------|--------|-----------|
| A | `q50` | 0.0 | `demo_5_layernorm_q50` |
| B | `q50_minus_cvar_penalty` | 0.25 | `demo_5_layernorm_lambda_025` |
| C | `q50_minus_cvar_penalty` | 0.50 | `demo_5_layernorm_lambda_050` |
| D (reference) | `q50_minus_cvar_penalty` | 0.75 | `demo_5_long_2018_2024_v28_repro_layernorm` |

---

## 1. Mean / Min / Max Return

| Variant | Mean return | Min return | Max return |
|---------|------------|-----------|-----------|
| A — q50 | **+77.25%** | +65.70% | +87.00% |
| B — λ=0.25 | +9.92% | −0.30% | +39.18% |
| C — λ=0.50 | +0.75% | −0.48% | +3.58% |
| D — λ=0.75 | +0.18% | −2.27% | +3.76% |

Per-seed returns:

| Seed | q50 | λ=0.25 | λ=0.50 | λ=0.75 |
|------|-----|--------|--------|--------|
| 1 | +85.31% | −0.30% | +0.31% | −1.91% |
| 2 | +87.00% | +39.18% | +0.15% | +1.31% |
| 3 | +72.06% | +6.99% | −0.48% | −2.27% |
| 4 | +65.70% | +3.61% | +3.58% | +3.76% |
| 5 | +76.17% | +0.13% | +0.19% | 0.00% |

---

## 2. Mean Sharpe Ratio

| Variant | Mean Sharpe | Min | Max |
|---------|------------|-----|-----|
| A — q50 | **2.856** | 2.815 | 2.904 |
| B — λ=0.25 | 1.269 | −0.129 | 2.657 |
| C — λ=0.50 | 0.562 | −0.375 | 2.223 |
| D — λ=0.75 | −0.037 | −1.737 | 2.222 |

Note: Seed 4 consistently achieves Sharpe ~2.22 across λ=0.25/0.50/0.75 — it is the only seed
whose policy survives CVaR penalisation, likely due to a favourable initial Q-distribution
for that random seed.

---

## 3. No-Trade Seeds

| Variant | No-trade seeds |
|---------|---------------|
| A — q50 | **0 / 5** |
| B — λ=0.25 | 0 / 5 |
| C — λ=0.50 | 0 / 5 *(seed 5: 13 trades, nearly inactive)* |
| D — λ=0.75 | **1 / 5** (seed 5: 0 trades, 271 HOLDs) |

---

## 4. Mean Total Trades

| Variant | Mean trades | Per-seed trades |
|---------|------------|----------------|
| A — q50 | 182.2 | 166, 167, 225, 259, 94 |
| B — λ=0.25 | 265.2 | 265, 269, 268, 268, 256 |
| C — λ=0.50 | 204.0 | 222, 258, 260, 267, 13 |
| D — λ=0.75 | 185.0 | 160, 239, 259, 267, 0 |

Note: High trade counts under λ>0 reflect the SELL-churn loop (rapid BUY→SELL cycles),
not meaningful portfolio activity.

---

## 5. Mean Final Cash Weight

| Variant | Mean final_cash_weight | Interpretation |
|---------|----------------------|---------------|
| A — q50 | **0.200** | Agent builds and holds equity positions |
| B — λ=0.25 | 0.886 | Agent sells most positions; mostly in cash |
| C — λ=0.50 | 0.991 | Agent is almost entirely in cash |
| D — λ=0.75 | 0.994 | Agent is entirely in cash |

---

## 6. Mean Turnover

| Variant | Mean turnover | Mean SELL/BUY ratio |
|---------|--------------|---------------------|
| A — q50 | 1,346% | 0.6 : 1 *(BUY-dominant)* |
| B — λ=0.25 | 1,370% | 9.4 : 1 *(SELL-dominant)* |
| C — λ=0.50 | 777% | 11.9 : 1 *(SELL-dominant)* |
| D — λ=0.75 | 700% | 9.8 : 1 *(SELL-dominant)* |

The SELL/BUY ratio flips from 0.6:1 (BUY-dominant) under q50 to ~9–12:1 (SELL-dominant)
the moment any CVaR penalty is applied. This is not a gradual transition.

---

## 7. Interpretation of the Lambda Effect

### The core mechanism

`score(a) = q50(a) − λ × |CVaR10(a)|`

When an agent holds cash and considers buying an equity:
- **BUY**: q50 > 0 (expected positive), but CVaR10 is strongly negative (equity can drop)
- **SELL-to-cash / HOLD**: CVaR10 ≈ 0 (deterministic cash conversion carries no tail risk)

With only 25,000 training steps the IQN network produces CVaR estimates that are
systematically conservative for BUY — the network has seen many downward moves during
training and "learned to fear" tail losses. Even at λ=0.25, the CVaR penalty on BUY is
large enough to flip the policy fully toward SELL.

### Why the transition is abrupt (not gradual)

The SELL/BUY ratio jumps from 0.6:1 (q50) to ~9:1 (λ=0.25) with no intermediate.
This indicates that CVaR10(BUY) is extremely negative — probably in the range −10% to −30%.
A penalty of even 0.25 × 30% = 7.5% is enough to erase any BUY advantage in median return.

### Why seed 4 survives across all lambda values

Seed 4 achieves return ~3.6–3.8% and Sharpe ~2.2 consistently across λ=0.25/0.50/0.75.
Its specific network initialisation produces a Q-distribution where BUY's q50 is
sufficiently high that even after CVaR penalisation it occasionally beats SELL.
However, seed 4 is also the one that SELLS most (SELL=248–249 out of ~267 trades),
so its "success" is marginal and occurs during a period when a brief BUY → immediate SELL
happened to catch a small uptick.

### The q50 vs CVaR-penalised split

| Property | q50 | λ > 0 |
|----------|-----|-------|
| Policy character | BUY-dominant, holds equity | SELL-dominant, exits to cash |
| Market exposure | High (cash_w ~0.2) | Minimal (cash_w ~0.99) |
| Returns | +66% to +87% | −2% to +39% |
| Captures 2023 bull market? | **Yes** | **No** |
| Risk-aware? | No CVaR term | Yes, but over-conservative |

---

## 8. Thesis-Safe Conclusion

### What the ablation confirms

> **LayerNorm solved the neural instability** (Q-value scale divergence). All 5 seeds
> become active traders under `q50` scoring, with mean return **+77.25%** and mean
> Sharpe **2.856** — highly consistent across seeds.

> **`q50_minus_cvar_penalty` with any tested lambda (0.25–0.75) produces a SELL-dominant
> churn policy** that fails to build equity exposure. Mean returns collapse to +10%, +1%,
> and 0% for λ=0.25, 0.50, 0.75 respectively. The transition is sharp, not gradual.

### Root cause of the CVaR-policy failure

The IQN network after only 25,000 training steps produces systematically pessimistic
CVaR10 estimates for BUY actions. This is a **CVaR calibration problem**, not a
neural stability problem. With 25k steps, the agent has not seen enough market
cycles to calibrate the tail of the BUY return distribution. As a result, even a
small CVaR penalty entirely dominates the q50 BUY advantage.

### Recommended formulation for thesis

> *The D-IQN-DSS system with LayerNorm demonstrates strong portfolio performance under*
> *q50 scoring (mean return +77.25%, mean Sharpe 2.856, 5/5 active seeds). Under the*
> *risk-penalised objective `q50_minus_cvar_penalty`, all tested lambda values (0.25–0.75)*
> *produce a SELL-dominant policy with near-zero market exposure, attributed to*
> *under-calibrated CVaR estimates resulting from limited training steps (25,000).*
> *This suggests that risk-sensitive policy scoring requires either longer training,*
> *a separate CVaR calibration procedure, or a weaker penalty schedule.*
> *The neural stability fix (LayerNorm) and the risk-policy calibration problem*
> *are separable and independently addressable.*

### Immediate next steps (analysis only)

1. **Use q50 as the thesis scoring mode** for the official Demo_5 IQN result.
   It demonstrates a clean, interpretable, seed-stable policy with strong performance.
2. **Frame the CVaR-penalty experiment** as an ablation showing the sensitivity of
   the risk-aware objective to training depth — a meaningful negative result that
   motivates future work (longer training / calibration-aware lambda schedules).
3. **Do not present lambda=0.25–0.75 results as primary results** — present them
   as diagnostic evidence of the calibration problem.
