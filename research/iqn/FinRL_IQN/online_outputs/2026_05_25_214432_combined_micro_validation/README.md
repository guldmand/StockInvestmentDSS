# Combined Micro Validation — 2026-05-25 21:44:32

This directory contains a consolidated, human-readable view of the
key artifacts produced by the 3-layer pipeline.

Raw manifests (not modified) live in `outputs/runs/`:

- Algorithmic: `outputs/runs/2026_05_25_214433_d_iqn_dss_algorithmic_baseline_grid_demo_10_new_micro/`
- FinRL launcher: `outputs/runs/2026_05_25_214524_d_iqn_dss_finrl_baseline_multiseed_launcher/`
- FinRL summary:  `outputs/runs/2026_05_25_214620_d_iqn_dss_finrl_baseline_multiseed_summary/`
- IQN launcher: `outputs/runs/2026_05_25_214620_d_iqn_dss_iqn_learning_curve_multiseed_launcher/`
- IQN summary:  `outputs/runs/2026_05_25_214731_d_iqn_dss_iqn_learning_curve_multiseed_summary/`

## Layout

- `algorithmic/` — algorithmic baselines summary CSVs
- `finrl/`       — FinRL multiseed aggregate CSVs and plots
- `iqn/`         — IQN multiseed aggregate CSVs and plots

## Thesis plots

- `thesis_plots/algorithmic_multi_ticker/single_ticker_strategies/` — 24 PNGs, one per single-ticker strategy variant, each with one line per ticker (PIT trade window only)
- `thesis_plots/algorithmic_multi_ticker/portfolio_strategies/` — 2 PNGs, one per portfolio-level strategy (PIT trade window only)
- `thesis_plots/finrl_multi_baseline/` — 1 PNG, FinRL agents (a2c, ppo, mvo) mean ± std across seeds, PIT trade window

## Transaction logs

- `transaction_logs/algorithmic/single_ticker/` — one .md per (strategy, ticker) pair with entry/exit event table
- `transaction_logs/algorithmic/portfolio/` — one .md per portfolio strategy with initial holdings or rebalance events
- `transaction_logs/finrl/` — one .md per (agent, seed) with daily action table and cumulative holdings
- `transaction_logs/iqn/` — one .md per seed with action distribution and non-HOLD decision table

## Dataset

- Dataset ID: `demo_10_new_micro`
- Tickers:    `COST,AVGO,LLY,ORCL,CAT`
- PIT split:  2024-01-01 -> 2026-12-31