# LayerNorm Lambda Ablation

## Why this ablation is run

The official Mode B thesis run (`run_mode_b_repro_demo5_iqn_multiseed_layernorm.ps1`)
used `score_mode=q50_minus_cvar_penalty` with `risk_lambda=0.75` and LayerNorm enabled.

**Observed result:**
- 4/5 seeds were active trading (seeds 1–4)
- All active seeds exhibited extreme SELL-dominance (~12:1 SELL/BUY ratio)
- All active seeds ended near 100% cash despite being in a bull market (2023)
- Returns ranged from -2.27% to +3.76%
- Seed 5 still chose HOLD for all 271 steps

**Diagnosis:**
The CVaR penalty in `q50_minus_cvar_penalty` systematically penalises BUY more than
SELL or HOLD. When cash is held, SELL-to-cash has CVaR ≈ 0 (no market risk), while
BUY always carries a negative CVaR10 (equity can drop). With lambda=0.75, this
asymmetry is strong enough to:
1. prevent seed 5 from ever choosing BUY (HOLD stays dominant)
2. cause seeds 1–4 to buy briefly then immediately sell (churn loop)

**Hypothesis to test:**
Does reducing lambda reduce SELL-dominance and improve returns?
At what lambda does the CVaR penalty start to matter without destroying active exposure?

This ablation holds all other settings fixed and varies only `score_mode` and
`risk_lambda`.

---

## Scripts

| Script | score_mode | lambda | Purpose |
|--------|-----------|--------|---------|
| `run_layernorm_q50.ps1` | `q50` | 0.0 | Baseline: no CVaR penalty. Reproduce Exp G in Mode B. |
| `run_layernorm_lambda_025.ps1` | `q50_minus_cvar_penalty` | 0.25 | Low penalty — risk-aware but mild. |
| `run_layernorm_lambda_050.ps1` | `q50_minus_cvar_penalty` | 0.50 | Medium penalty. |
| `run_layernorm_lambda_075.ps1` | `q50_minus_cvar_penalty` | 0.75 | **Reference** — reproduces official thesis run. |

All other settings are identical across scripts:
- Seeds: 1, 2, 3, 4, 5
- Total train steps: 25,000
- Eval interval: 5,000
- Learning starts: 2,000
- LayerNorm: enabled
- Dataset: Mode B frozen import, demo_5 tickers
- PIT split: 2018–2022 train / 2023–2024 eval
- W&B: enabled

---

## Expected outcomes

| Variant | Prediction |
|---------|-----------|
| q50 (A) | 5/5 active seeds, higher returns, low SELL/BUY ratio — mirrors Exp G |
| lambda=0.25 (B) | 4–5/5 active, moderate SELL bias, better returns than D |
| lambda=0.50 (C) | 4–5/5 active, SELL bias increases, returns between B and D |
| lambda=0.75 (D) | 4/5 active, strong SELL-dominance, returns -2% to +4% — matches known official run |

If prediction holds:
- Confirms CVaR-lambda is the driver of SELL-dominance
- Identifies a suitable lambda for thesis production (likely 0.25–0.50)
- Cleanly separates neural instability problem (solved by LayerNorm) from
  risk-policy calibration problem (solved by lambda tuning)

---

## What these scripts do NOT change

- No source files are modified (`src/` is untouched)
- No existing run scripts are modified
- Each script uses a unique `dataset_id` and `pit_split_id` to avoid
  cache/output collisions between variants
- Scripts only set environment variables and invoke the multiseed launcher

---

## How to run (when ready)

Run each variant separately in sequence:

```powershell
cd c:\Users\gurug\Dropbox\DataScience\Speciale\D-IQN-DSS\FinRL_IQN

# Variant A — q50 baseline
powershell -ExecutionPolicy Bypass -File .\copilot-diagnostics\experiments\layernorm_lambda_ablation\run_layernorm_q50.ps1

# Variant B — lambda=0.25
powershell -ExecutionPolicy Bypass -File .\copilot-diagnostics\experiments\layernorm_lambda_ablation\run_layernorm_lambda_025.ps1

# Variant C — lambda=0.50
powershell -ExecutionPolicy Bypass -File .\copilot-diagnostics\experiments\layernorm_lambda_ablation\run_layernorm_lambda_050.ps1

# Variant D — lambda=0.75 (reference, reproduces official run)
powershell -ExecutionPolicy Bypass -File .\copilot-diagnostics\experiments\layernorm_lambda_ablation\run_layernorm_lambda_075.ps1
```

Results will appear under `outputs/runs/` with dataset_id in the folder name.

---

## Key metrics to compare across variants

After all four runs complete, compare:

| Metric | What it tells you |
|--------|------------------|
| active_trading count (out of 5) | Whether seeds escape HOLD/no-trade |
| SELL/BUY ratio | Degree of churn-loop behaviour |
| final_cash_weight | Whether agent builds/holds positions |
| total_return_pct (mean ± std) | Actual performance |
| max_drawdown | Risk realised |
| total_trades | Activity level |
| turnover_estimate_pct | Trading friction |

---

## Thesis context

This ablation supports the conclusion:

> LayerNorm solved the Q-value scale instability (neural component).
> The remaining behavioural problem — SELL-dominance and insufficient market
> exposure — is attributable to the CVaR penalty magnitude (policy component).
> Lambda tuning is a separate, interpretable design choice with direct
> risk-profile implications for the D-IQN-DSS system.
