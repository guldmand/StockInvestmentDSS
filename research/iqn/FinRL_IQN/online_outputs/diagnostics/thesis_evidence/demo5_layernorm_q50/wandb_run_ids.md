# W&B Run IDs — D-IQN-DSS Demo_5 LayerNorm + q50

**Entity:** `guldmand-SDU`  
**Project:** `StockInvestmentDSS`  
**Base URL:** `https://wandb.ai/guldmand-SDU/StockInvestmentDSS/runs/<run_id>`

---

## Primary Result: demo_5_layernorm_q50 (LayerNorm + q50)

| Seed | W&B Run ID | Direct Link |
|------|-----------|-------------|
| 1 | `uz4n93e6` | https://wandb.ai/guldmand-SDU/StockInvestmentDSS/runs/uz4n93e6 |
| 2 | `05hd99d9` | https://wandb.ai/guldmand-SDU/StockInvestmentDSS/runs/05hd99d9 |
| 3 | `ukks5gmb` | https://wandb.ai/guldmand-SDU/StockInvestmentDSS/runs/ukks5gmb |
| 4 | `o4df6gup` | https://wandb.ai/guldmand-SDU/StockInvestmentDSS/runs/o4df6gup |
| 5 | `7y1he8wx` | https://wandb.ai/guldmand-SDU/StockInvestmentDSS/runs/7y1he8wx |

**Dataset ID:** `demo_5_layernorm_q50`  
**Score mode:** `q50`  
**Risk lambda:** `0.0`  
**use_layer_norm:** `true`  
**Total training steps:** 25,000  
**Eval interval:** 5,000  
**Learning starts:** 2,000  
**PIT cutoff:** 2023-01-01  
**Eval end:** 2024-02-01  
**Universe:** AAPL, MSFT, NVDA, AMZN, GOOGL (demo_5)

---

## Ablation Runs (logged to cloud W&B only, no local wandb/ folder)

| Run | Dataset ID | Score Mode | Lambda | Notes |
|-----|-----------|------------|--------|-------|
| LN + λ=0.25 | `demo_5_layernorm_lambda_025` | q50_minus_cvar_penalty | 0.25 | Negative finding |
| LN + λ=0.50 | `demo_5_layernorm_lambda_050` | q50_minus_cvar_penalty | 0.50 | Negative finding |
| LN + λ=0.75 | `demo_5_long_2018_2024_v28_repro_layernorm` | q50_minus_cvar_penalty | 0.75 | Official thesis reference run |

> Ablation W&B run IDs are available in:
> - `outputs/runs/2026_05_20_204924_d_iqn_dss_iqn_learning_curve_multiseed_summary/` (λ=0.25)
> - `outputs/runs/2026_05_20_210705_d_iqn_dss_iqn_learning_curve_multiseed_summary/` (λ=0.50)

---

## Source

W&B IDs extracted from:
- `outputs/runs/2026_05_20_202508_d_iqn_dss_iqn_learning_curve_multiseed_summary/run.log`

Lines matching `wandb: Run data is saved locally in wandb/run-*` and
`wandb: 🚀 View run ... at: https://wandb.ai/...`
