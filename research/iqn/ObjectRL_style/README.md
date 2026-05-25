# StockDSS Solo Script Commands

This document collects the current standalone / solo script commands used in the `ObjectRL_style` PoC.

The purpose is to make it easy to run each part individually without using the combined runner.

## Basic setup

Run all commands from the project root:

```powershell
cd C:\Users\gurug\Dropbox\DataScience\Speciale\D-IQN-DSS\ObjectRL_style
conda activate stockdss
$env:PYTHONPATH="src"
```

---

# 1. Data

## 1.1 Create point-in-time train/trade split

Creates a PIT dataset from one source CSV.

Typical use:
- `train_data_...csv` contains all rows before the point-in-time date.
- `trade_data_...csv` contains all rows from the point-in-time date onward.
- Metadata is written to `outputs/pit`.

```powershell
python -m stockdss.data.point_in_time_split `
  --input data/train_data_pit_500_2026_01_01.csv `
  --point-in-time 2020-01-01 `
  --trade-end-date 2025-12-31 `
  --dataset-tag pit_500_2020_01_01_2025_12_31 `
  --output-data-dir data `
  --metadata-dir outputs/pit
```

Expected outputs:

```text
data/train_data_pit_500_2020_01_01_2025_12_31.csv
data/trade_data_pit_500_2020_01_01_2025_12_31.csv
outputs/pit/pit_metadata_pit_500_2020_01_01_2025_12_31.json
```

Important flags:

| Flag | Meaning |
|---|---|
| `--input` | Source CSV used as the master input file. |
| `--point-in-time` | Split date. Rows before this date become training data. Rows from this date onward become trade/test data. |
| `--trade-end-date` | Optional end date for trade/test data. |
| `--dataset-tag` | Name used in output filenames. |
| `--output-data-dir` | Directory where train/trade CSVs are saved. |
| `--metadata-dir` | Directory where PIT metadata JSON is saved. |
| `--min-tickers-per-date` | Optional validation threshold for minimum tickers per date. Default is `1`. |

Notes:
- Current workaround: `data/train_data_pit_500_2026_01_01.csv` is used as a master-ish source because it covers the full available period up to 2025-12-31.
- Later improvement: create a clearer canonical master file, for example `data/market_data_full_500_2016_01_04_2025_12_31.csv`.

---

# 2. Algorithmic Trading

This track is for classical non-RL trading baselines.

Current implemented baseline:

```text
C. Classical non-RL trading baselines
   - AAPL buy-and-hold single ticker
   - Later: SMA crossover
   - Later: momentum baseline
   - Later: equal-weight buy-and-hold portfolio
```

## 2.1 Buy-and-hold PIT single ticker

Runs a pure algorithmic-trading baseline for one ticker.

This does not train an RL model. It simply buys and holds the selected ticker over the PIT trade period.

### Example: AAPL, PIT 2026 trade period

```powershell
python -m stockdss.algorithmic_trading.experiments.run_buy_and_hold_pit_single_ticker `
  --trade-data data/trade_data_pit_500_2026_01_01.csv `
  --dataset-tag pit_500_2026_01_01 `
  --ticker AAPL `
  --run-name test_aapl_buy_and_hold `
  --run-root outputs/runs/test_algorithmic_trading `
  --initial-amount 1000000
```

### Example: AAPL, PIT 2025 trade period

```powershell
python -m stockdss.algorithmic_trading.experiments.run_buy_and_hold_pit_single_ticker `
  --trade-data data/trade_data_pit_500_2025_01_01.csv `
  --dataset-tag pit_500_2025_01_01 `
  --ticker AAPL `
  --run-name test_aapl_buy_and_hold_2025 `
  --run-root outputs/runs/test_algorithmic_trading_2025 `
  --initial-amount 1000000
```

### Example: AAPL, 2020-2025 PIT trade period

```powershell
python -m stockdss.algorithmic_trading.experiments.run_buy_and_hold_pit_single_ticker `
  --trade-data data/trade_data_pit_500_2020_01_01_2025_12_31.csv `
  --dataset-tag pit_500_2020_01_01_2025_12_31 `
  --ticker AAPL `
  --run-name test_aapl_buy_and_hold_2020_2025 `
  --run-root outputs/runs/test_algorithmic_trading_2020_2025 `
  --initial-amount 1000000
```

Expected outputs when `--run-root` is provided:

```text
outputs/runs/<run_id_or_test_name>/algorithmic_trading/files/buy_and_hold/
outputs/runs/<run_id_or_test_name>/algorithmic_trading/plots/buy_and_hold/
```

Important flags:

| Flag | Meaning |
|---|---|
| `--trade-data` | PIT trade/test CSV. |
| `--dataset-tag` | Dataset identifier used in metadata/output naming. |
| `--ticker` | Single ticker to buy and hold, for example `AAPL`. |
| `--run-name` | Human-readable run name. |
| `--run-root` | Optional central output folder. Use this to integrate into runner-style output layout. |
| `--initial-amount` | Starting capital. Usually `1000000`. |

---

# 3. FinRL Baselines

This track trains and backtests standard FinRL baseline agents on PIT data.

Current agents used:

```text
a2c, ddpg, ppo, td3, sac
```

## 3.1 Train FinRL baselines on PIT data

### Quick smoke test

```powershell
python -m stockdss.rl.experiments.train_finrl_baselines_pit `
  --train-data data/train_data_pit_500_2026_01_01.csv `
  --dataset-tag pit_500_2026_01_01 `
  --run-name test_base `
  --run-root outputs/runs/test_run_paths `
  --total-timesteps 10 `
  --agents a2c
```

### Longer multi-agent run

```powershell
python -m stockdss.rl.experiments.train_finrl_baselines_pit `
  --train-data data/train_data_pit_500_2025_01_01.csv `
  --dataset-tag pit_500_2025_01_01 `
  --run-name finrl_baselines_2025 `
  --run-root outputs/runs/finrl_baselines_2025 `
  --total-timesteps 20000 `
  --agents a2c,ddpg,ppo,td3,sac `
  --initial-amount 1000000
```

Expected runner-style outputs:

```text
outputs/runs/<run_root>/baseline_finrl/models/
outputs/runs/<run_root>/baseline_finrl/logs/
outputs/runs/<run_root>/baseline_finrl/files/train/
```

Important flags:

| Flag | Meaning |
|---|---|
| `--train-data` | PIT train CSV. |
| `--dataset-tag` | Dataset identifier. |
| `--run-name` | Human-readable run name. |
| `--run-root` | Optional central output folder. |
| `--total-timesteps` | Training length per agent. |
| `--agents` | Comma-separated agents, for example `a2c,ddpg,ppo,td3,sac`. |
| `--initial-amount` | Starting capital used by the FinRL environment. |
| `--hmax` | Maximum shares per transaction. |
| `--buy-cost-pct` | Buy transaction cost percentage. |
| `--sell-cost-pct` | Sell transaction cost percentage. |
| `--reward-scaling` | Reward scaling used by the environment. |
| `--device` | `auto`, `cpu`, or `cuda`. |

---

## 3.2 Backtest FinRL baselines on PIT data

Runs the trained FinRL agents on the PIT trade data.

### Quick smoke test

```powershell
python -m stockdss.rl.experiments.backtest_finrl_baselines_pit `
  --train-data data/train_data_pit_500_2026_01_01.csv `
  --trade-data data/trade_data_pit_500_2026_01_01.csv `
  --dataset-tag pit_500_2026_01_01 `
  --run-name test_base `
  --agents a2c `
  --run-root outputs/runs/test_run_paths `
  --initial-amount 1000000 `
  --use-mvo
```

### Longer multi-agent backtest

```powershell
python -m stockdss.rl.experiments.backtest_finrl_baselines_pit `
  --train-data data/train_data_pit_500_2025_01_01.csv `
  --trade-data data/trade_data_pit_500_2025_01_01.csv `
  --dataset-tag pit_500_2025_01_01 `
  --run-name finrl_baselines_2025 `
  --agents a2c,ddpg,ppo,td3,sac `
  --run-root outputs/runs/finrl_baselines_2025 `
  --initial-amount 1000000 `
  --use-mvo
```

Expected runner-style outputs:

```text
outputs/runs/<run_root>/baseline_finrl/files/backtest/
outputs/runs/<run_root>/baseline_finrl/files/backtest/backtest_result.csv
outputs/runs/<run_root>/baseline_finrl/files/backtest/backtest_metrics.csv
outputs/runs/<run_root>/baseline_finrl/files/backtest/account_values_<agent>.csv
outputs/runs/<run_root>/baseline_finrl/files/backtest/actions_<agent>.csv
```

Important flags:

| Flag | Meaning |
|---|---|
| `--train-data` | PIT train CSV, used for MVO and environment metadata. |
| `--trade-data` | PIT trade/test CSV. |
| `--dataset-tag` | Dataset identifier. |
| `--run-name` | Must match the trained baseline run when not using `--run-root`. |
| `--agents` | Agents to backtest. |
| `--run-root` | Central runner-style output folder. |
| `--initial-amount` | Starting capital. |
| `--use-mvo` | Adds MVO baseline to the backtest. |
| `--use-dji` | Optional DJI baseline if supported/configured. |
| `--device` | `auto`, `cpu`, or `cuda`. |

Note:
- MVO is now normalized so it starts at the same initial amount as the other strategies.

---

## 3.3 Visualize FinRL baseline backtest

Creates plots from the FinRL backtest result files.

```powershell
python -m stockdss.rl.experiments.visualize_finrl_backtest_pit `
  --dataset-tag pit_500_2026_01_01 `
  --run-name test_base `
  --run-root outputs/runs/test_run_paths
```

Expected runner-style outputs:

```text
outputs/runs/<run_root>/baseline_finrl/plots/
outputs/runs/<run_root>/baseline_finrl/plots/plot_01_portfolio_values.png
outputs/runs/<run_root>/baseline_finrl/plots/plot_02_normalized_portfolio_values.png
outputs/runs/<run_root>/baseline_finrl/plots/plot_03_drawdowns.png
outputs/runs/<run_root>/baseline_finrl/plots/plot_04_final_values.png
outputs/runs/<run_root>/baseline_finrl/plots/plot_05_daily_return_distribution_<strategy>.png
outputs/runs/<run_root>/baseline_finrl/plots/plot_06_daily_return_boxplot.png
outputs/runs/<run_root>/baseline_finrl/plots/daily_return_distribution_summary.csv
```

Important flags:

| Flag | Meaning |
|---|---|
| `--dataset-tag` | Dataset identifier. |
| `--run-name` | Name of the run to visualize when not using `--run-root`. |
| `--run-root` | Central runner-style output folder. |

---

# 4. IQN Implementation

This track is the custom D-IQN-DSS implementation.

The current PIT version is single-ticker.

## 4.1 Train IQN on PIT single ticker

### Quick smoke test

```powershell
python -m stockdss.rl.experiments.train_iqn_finrl_pit_single_ticker `
  --train-data data/train_data_pit_500_2026_01_01.csv `
  --dataset-tag pit_500_2026_01_01 `
  --run-name test_iqn_train `
  --ticker AAPL `
  --total-steps 100 `
  --log-interval 1000 `
  --save-every 10000 `
  --run-root outputs/runs/test_run_paths `
  --initial-amount 1000000
```

### Longer training run

```powershell
python -m stockdss.rl.experiments.train_iqn_finrl_pit_single_ticker `
  --train-data data/train_data_pit_500_2025_01_01.csv `
  --dataset-tag pit_500_2025_01_01 `
  --run-name iqn_aapl_2025_train `
  --ticker AAPL `
  --total-steps 50000 `
  --log-interval 1000 `
  --save-every 10000 `
  --run-root outputs/runs/iqn_aapl_2025 `
  --initial-amount 1000000
```

Expected runner-style outputs:

```text
outputs/runs/<run_root>/iqn_finrl/models/iqn_agent.pt
outputs/runs/<run_root>/iqn_finrl/files/train/iqn_training_log.csv
outputs/runs/<run_root>/iqn_finrl/files/train/iqn_episode_log.csv
outputs/runs/<run_root>/iqn_finrl/files/train/iqn_training_metrics.csv
outputs/runs/<run_root>/iqn_finrl/files/train/iqn_training_config.json
```

Important flags:

| Flag | Meaning |
|---|---|
| `--train-data` | PIT train CSV. |
| `--dataset-tag` | Dataset identifier. |
| `--run-name` | Human-readable IQN training run name. |
| `--ticker` | Single ticker used by the IQN environment. |
| `--total-steps` | Number of IQN training steps. |
| `--log-interval` | Console/log print frequency. |
| `--save-every` | Save frequency. |
| `--run-root` | Central runner-style output folder. |
| `--initial-amount` | Starting capital. |
| `--buy-cost-pct` | Buy transaction cost percentage. |
| `--sell-cost-pct` | Sell transaction cost percentage. |
| `--device` | `auto`, `cpu`, or `cuda`. |

---

## 4.2 Backtest IQN on PIT single ticker

### Quick smoke test

```powershell
python -m stockdss.rl.experiments.backtest_iqn_pit_single_ticker `
  --trade-data data/trade_data_pit_500_2026_01_01.csv `
  --dataset-tag pit_500_2026_01_01 `
  --run-name test_iqn_backtest `
  --ticker AAPL `
  --model-path outputs/runs/test_run_paths/iqn_finrl/models/iqn_agent.pt `
  --risk-lambda 0.75 `
  --run-root outputs/runs/test_run_paths `
  --initial-amount 1000000
```

Expected runner-style outputs:

```text
outputs/runs/<run_root>/iqn_finrl/files/backtest/iqn_decision_log.csv
outputs/runs/<run_root>/iqn_finrl/files/backtest/iqn_account_values.csv
outputs/runs/<run_root>/iqn_finrl/files/backtest/iqn_action_estimates_all_steps.csv
outputs/runs/<run_root>/iqn_finrl/files/backtest/iqn_action_estimates_last_day.csv
outputs/runs/<run_root>/iqn_finrl/files/backtest/iqn_backtest_metrics.csv
outputs/runs/<run_root>/iqn_finrl/files/backtest/iqn_backtest_config.json
outputs/runs/<run_root>/iqn_finrl/plots/
```

Important flags:

| Flag | Meaning |
|---|---|
| `--trade-data` | PIT trade/test CSV. |
| `--dataset-tag` | Dataset identifier. |
| `--run-name` | Human-readable IQN backtest run name. |
| `--ticker` | Single ticker used in the backtest. |
| `--model-path` | Path to trained `iqn_agent.pt`. |
| `--risk-lambda` | Risk sensitivity used in action scoring. Higher means more penalty for downside risk. |
| `--run-root` | Central runner-style output folder. |
| `--initial-amount` | Starting capital. |
| `--device` | `auto`, `cpu`, or `cuda`. |

Risk lambda:
- `0.0`: closer to risk-neutral median/expected scoring.
- `0.75`: current default experiment value.
- Higher values: more conservative because downside/CVaR gets more weight.

---

## 4.3 Visualize IQN decision distribution

Visualizes the last-day IQN action estimates.

```powershell
python -m stockdss.rl.experiments.visualize_iqn_decision_distribution `
  --decision-csv outputs/runs/test_run_paths/iqn_finrl/files/backtest/iqn_action_estimates_last_day.csv `
  --date last_trade_day `
  --ticker AAPL `
  --price 249.7099456787109 `
  --risk-lambda 0.75 `
  --output-dir outputs/runs/test_run_paths/iqn_finrl/visualizations/iqn_decision_distribution
```

Expected outputs:

```text
outputs/runs/<run_root>/iqn_finrl/visualizations/iqn_decision_distribution/iqn_decision_estimates.csv
outputs/runs/<run_root>/iqn_finrl/visualizations/iqn_decision_distribution/iqn_decision_dashboard.png
outputs/runs/<run_root>/iqn_finrl/visualizations/iqn_decision_distribution/iqn_quantile_functions.png
outputs/runs/<run_root>/iqn_finrl/visualizations/iqn_decision_distribution/iqn_return_distributions.png
```

Important flags:

| Flag | Meaning |
|---|---|
| `--decision-csv` | CSV with IQN action estimates, usually `iqn_action_estimates_last_day.csv`. |
| `--date` | Display label/date used in visualization. |
| `--ticker` | Ticker label. |
| `--price` | Current/last price shown in dashboard. |
| `--risk-lambda` | Same risk penalty used in decision scoring. |
| `--output-dir` | Output folder for visualization files. |

---

# 5. Summary

## 5.1 Summarize runner-style results

Creates a comparison table across available strategies in one `run-root`.

```powershell
python -m stockdss.runner.summarize_run_results `
  --run-root outputs/runs/2026_05_14_0356_runner_pit_500_2025_01_01_aapl_f20000_i50000 `
  --show
```

Expected outputs:

```text
outputs/runs/<run_root>/summary/summary_report.csv
outputs/runs/<run_root>/summary/summary_report.md
outputs/runs/<run_root>/summary/summary_dashboard.png
outputs/runs/<run_root>/summary/summary_iqn_last_decision.csv
```

Important flags:

| Flag | Meaning |
|---|---|
| `--run-root` | Central run folder to summarize. |
| `--show` | Prints the summary table to the console. |

Note:
- The combined runner now runs summary automatically as step `C1`.
- This standalone summary command is still useful when backtest files have been regenerated or manually edited.

---

# 6. Combined runner

The combined runner is not a solo script, but it is included here for completeness.

It runs:
1. FinRL baseline training
2. FinRL baseline backtest
3. FinRL baseline visualization
4. IQN training
5. IQN backtest
6. IQN decision visualization
7. Summary

## 6.1 Dry run

Prints the commands without executing them.

```powershell
python -m stockdss.runner.iqn_runner `
  --train-data data/train_data_pit_500_2026_01_01.csv `
  --trade-data data/trade_data_pit_500_2026_01_01.csv `
  --dataset-tag pit_500_2026_01_01 `
  --ticker AAPL `
  --finrl-timesteps 10 `
  --iqn-steps 100 `
  --risk-lambda 0.75 `
  --agents a2c `
  --use-mvo `
  --dry-run
```

## 6.2 Quick smoke test

```powershell
python -m stockdss.runner.iqn_runner `
  --train-data data/train_data_pit_500_2026_01_01.csv `
  --trade-data data/trade_data_pit_500_2026_01_01.csv `
  --dataset-tag pit_500_2026_01_01 `
  --ticker AAPL `
  --finrl-timesteps 10 `
  --iqn-steps 100 `
  --risk-lambda 0.75 `
  --agents a2c `
  --use-mvo
```

## 6.3 Larger experiment

```powershell
python -m stockdss.runner.iqn_runner `
  --train-data data/train_data_pit_500_2025_01_01.csv `
  --trade-data data/trade_data_pit_500_2025_01_01.csv `
  --dataset-tag pit_500_2025_01_01 `
  --ticker AAPL `
  --finrl-timesteps 20000 `
  --iqn-steps 50000 `
  --risk-lambda 0.75 `
  --agents a2c,ddpg,ppo,td3,sac `
  --use-mvo
```

Expected central output layout:

```text
outputs/runs/<run_id>/
├── baseline_finrl/
│   ├── models/
│   ├── logs/
│   ├── files/
│   │   ├── train/
│   │   └── backtest/
│   └── plots/
├── iqn_finrl/
│   ├── models/
│   ├── files/
│   │   ├── train/
│   │   └── backtest/
│   ├── plots/
│   └── visualizations/
├── summary/
├── run_config.json
├── run_commands.ps1
└── run_summary.json
```

Important flags:

| Flag | Meaning |
|---|---|
| `--train-data` | PIT train CSV. |
| `--trade-data` | PIT trade/test CSV. |
| `--dataset-tag` | Dataset identifier. |
| `--ticker` | Single ticker for IQN. |
| `--finrl-timesteps` | Training timesteps for each FinRL baseline agent. |
| `--iqn-steps` | Training steps for custom IQN. |
| `--risk-lambda` | Risk sensitivity for IQN action scoring. |
| `--agents` | FinRL agents to train/backtest. |
| `--use-mvo` | Adds MVO baseline. |
| `--dry-run` | Print commands only. |
| `--initial-amount` | Starting capital. Default is usually `1000000`. |

---

# 7. Current tracks

The current implementation can be documented as three tracks:

```text
A. FinRL RL baselines
   - A2C
   - DDPG
   - PPO
   - TD3
   - SAC
   - Optional MVO benchmark

B. D-IQN-DSS custom implementation
   - Single-ticker IQN
   - Quantile estimates
   - CVaR/downside-aware action scoring
   - Risk-lambda decision support

C. Classical non-RL algorithmic trading baselines
   - AAPL buy-and-hold PIT single ticker
   - Later: SMA crossover
   - Later: momentum baseline
   - Later: equal-weight buy-and-hold portfolio
```

---

# 8. Recommended quick validation flow

Use this when checking that the system still works after code changes.

```powershell
$env:PYTHONPATH="src"

python -m stockdss.runner.iqn_runner `
  --train-data data/train_data_pit_500_2026_01_01.csv `
  --trade-data data/trade_data_pit_500_2026_01_01.csv `
  --dataset-tag pit_500_2026_01_01 `
  --ticker AAPL `
  --finrl-timesteps 10 `
  --iqn-steps 100 `
  --risk-lambda 0.75 `
  --agents a2c `
  --use-mvo
```

Then inspect:

```text
outputs/runs/<latest_run>/summary/summary_report.md
outputs/runs/<latest_run>/summary/summary_dashboard.png
```
