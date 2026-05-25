# D-IQN-DSS: Demo_5 LayerNorm + q50 — Run Summary

**Package date:** 2026-05-20  
**Primary result:** `demo_5_layernorm_q50`  
**Status:** ✅ Full 5/5 active-seed run — no HOLD-collapse

---

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Tickers | AAPL, MSFT, NVDA, AMZN, GOOGL |
| Train window | 2018-01-01 → 2022-12-31 |
| Eval window (PIT) | 2023-01-01 → 2024-02-01 |
| Dataset ID | `demo_5_layernorm_q50` |
| PIT split ID | `demo_5_long_2018_2024_v28_repro_layernorm_ablation` |
| Universe | `demo_5` |
| Initial capital | $1,000,000 |
| Seeds | 1, 2, 3, 4, 5 |
| Total train steps | 25,000 |
| Eval interval | 5,000 steps |
| Learning starts | 2,000 steps |
| IQN config preset | `stockdss_long_v1` |
| Score mode | **q50** (median quantile, no CVaR penalty) |
| Risk lambda | 0.0 |
| LayerNorm | **enabled** |
| Change-strategy | disabled |
| Data mode | Mode B — frozen import, no live download |

---

## Per-Seed Results

| Seed | Return | Sharpe | Max DD | CVaR10 | Volatility | Trades | Cash weight | Turnover | Actions (BUY / SELL / HOLD) |
|------|--------|--------|--------|--------|------------|--------|-------------|----------|-----------------------------|
| 1 | **+85.31%** | 2.854 | −10.26% | −2.02% | 20.88% | 166 | 0.124 | 932% | 125 / 43 / 103 |
| 2 | **+87.00%** | 2.904 | −9.37% | −2.02% | 20.81% | 167 | 0.153 | 858% | 131 / 39 / 101 |
| 3 | **+72.06%** | 2.885 | −7.44% | −1.66% | 18.07% | 225 | 0.462 | 1781% | 136 / 92 / 43 |
| 4 | **+65.70%** | 2.824 | −6.78% | −1.58% | 17.16% | 259 | 0.153 | 2374% | 130 / 128 / 10 |
| 5 | **+76.17%** | 2.815 | −8.56% | −1.89% | 19.39% | 94 | 0.106 | 786% | 56 / 37 / 177 |

---

## Aggregate Statistics (5 seeds)

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Return | **+77.25%** | 8.97% | +65.70% | +87.00% |
| Sharpe | **2.856** | 0.038 | 2.815 | 2.904 |
| Max drawdown | −8.48% | 1.41% | −10.26% | −6.78% |
| CVaR10 | −1.83% | 0.21% | −2.02% | −1.58% |
| Volatility (ann.) | 19.26% | 1.65% | 17.16% | 20.88% |
| Total trades | 182.2 | 63.3 | 94 | 259 |
| Final cash weight | 0.200 | 0.148 | 0.106 | 0.462 |
| Turnover | 1,346% | 702% | 786% | 2,374% |
| Transaction cost | $19,647 | $9,882 | $11,414 | $34,277 |

> **5/5 seeds active** — zero HOLD-collapse seeds.  
> Sharpe standard deviation (0.038) is exceptionally low — the policy is highly consistent across random seeds.

---

## Output Files

| File type | Location |
|-----------|----------|
| Multiseed summary JSON | `outputs/runs/2026_05_20_202508_.../summary/iqn_learning_curve_multiseed_summary.json` |
| Final records CSV | `outputs/runs/2026_05_20_202508_.../summary/iqn_learning_curve_multiseed_final_records.csv` |
| W&B runs | See `wandb_run_ids.md` |

---

## W&B Run Links

Project: `guldmand-SDU/StockInvestmentDSS`

| Seed | Run ID | W&B Link |
|------|--------|----------|
| 1 | `uz4n93e6` | https://wandb.ai/guldmand-SDU/StockInvestmentDSS/runs/uz4n93e6 |
| 2 | `05hd99d9` | https://wandb.ai/guldmand-SDU/StockInvestmentDSS/runs/05hd99d9 |
| 3 | `ukks5gmb` | https://wandb.ai/guldmand-SDU/StockInvestmentDSS/runs/ukks5gmb |
| 4 | `o4df6gup` | https://wandb.ai/guldmand-SDU/StockInvestmentDSS/runs/o4df6gup |
| 5 | `7y1he8wx` | https://wandb.ai/guldmand-SDU/StockInvestmentDSS/runs/7y1he8wx |
