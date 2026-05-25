# =============================================================================
# Robustness Test 1 — Demo_5 universe, earlier time window (2017–2023)
# =============================================================================
# Purpose: Verify that LayerNorm + q50 produces an active-trading policy
#          on a different time window than the primary result (2018–2024).
#          Trains on 2017-01-01 → 2022-01-01, evaluates 2022-01-01 → 2023-01-01.
#          Includes the 2022 bear market (S&P -19%), testing downside resilience.
#
# Pass criterion: seed=7 active trading (total_trades > 0, return != 0.0)
# Fail criterion: seed=7 HOLD-collapse (total_trades == 0)
#
# DO NOT MODIFY SOURCE FILES.
# This script only sets environment variables and runs the launcher.
# =============================================================================

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

$pythonExe = "C:\Users\gurug\miniconda3\envs\stockdss\python.exe"

# Pre-flight check
Write-Host "python exe: $pythonExe"
& $pythonExe -c "import matplotlib; print('matplotlib ok:', matplotlib.__version__)"
& $pythonExe -c "import torch; print('torch ok:', torch.__version__)"
& $pythonExe -c "
import os
os.environ['STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM'] = 'true'
from stock_investment_dss.rl.config.iqn_config import build_iqn_config
cfg = build_iqn_config()
print('use_layer_norm =', cfg.use_layer_norm)
"

# Dataset / universe
$env:STOCK_INVESTMENT_DSS_DAILY_DATASET_ID        = "robustness_demo5_2017_2023_layernorm_q50"
$env:STOCK_INVESTMENT_DSS_PIT_SPLIT_ID            = "robustness_demo5_2017_2023_layernorm_q50_pit"
$env:STOCK_INVESTMENT_DSS_UNIVERSE_ID             = "demo_5"
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE     = "demo_5"
$env:STOCK_INVESTMENT_DSS_FINRL_TICKERS           = "AAPL,MSFT,NVDA,AMZN,GOOGL"

# PIT window — earlier period, includes 2022 bear market
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_START         = "2017-01-01"
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_END           = "2023-01-01"
$env:STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME        = "2022-01-01"
$env:STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE       = "2023-01-01"
$env:STOCK_INVESTMENT_DSS_PIT_MIN_TICKERS_PER_DATE = "5"

# Mode B frozen data
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE      = "true"
$env:STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD      = "false"
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD = "false"
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE    = "data/market/daily/imports/market_data_full_500.csv"
$env:STOCK_INVESTMENT_DSS_ALLOW_IMPORT_FALLBACK     = "true"
$env:STOCK_INVESTMENT_DSS_REQUIRE_ALL_TICKERS       = "true"

# Download settings (unused in Mode B)
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_CHUNK_SIZE    = "25"
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_SLEEP_SECONDS = "2"
$env:STOCK_INVESTMENT_DSS_YFINANCE_IMPERSONATE     = "firefox135"
$env:STOCK_INVESTMENT_DSS_YFINANCE_TIMEOUT_SECONDS = "30"

# IQN config
$env:STOCK_INVESTMENT_DSS_IQN_CONFIG_PRESET                  = "stockdss_long_v1"
$env:STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST                = "7"
$env:STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_TOTAL_STEPS     = "25000"
$env:STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_EVAL_INTERVAL   = "5000"
$env:STOCK_INVESTMENT_DSS_IQN_LEARNING_STARTS                = "2000"
$env:STOCK_INVESTMENT_DSS_IQN_DISABLE_CHANGE_STRATEGY        = "true"

# Scoring: pure q50, no CVaR penalty
$env:STOCK_INVESTMENT_DSS_IQN_BACKTEST_SCORE_MODE  = "q50"
$env:STOCK_INVESTMENT_DSS_IQN_BACKTEST_RISK_LAMBDA = "0.0"

# LayerNorm
$env:STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM = "true"

# W&B
$env:STOCK_INVESTMENT_DSS_WANDB_ENABLED = "true"
$env:STOCK_INVESTMENT_DSS_WANDB_PROJECT = "StockInvestmentDSS"
$env:STOCK_INVESTMENT_DSS_WANDB_ENTITY  = "guldmand-SDU"

Write-Host ""
Write-Host "=== Robustness Test 1: LayerNorm + q50 | Demo_5 | 2017-2023 (bear market) ==="
Write-Host "Dataset:    $env:STOCK_INVESTMENT_DSS_DAILY_DATASET_ID"
Write-Host "Tickers:    $env:STOCK_INVESTMENT_DSS_FINRL_TICKERS"
Write-Host "Train:      2017-01-01 -> 2022-01-01"
Write-Host "Eval:       2022-01-01 -> 2023-01-01"
Write-Host "Score mode: $env:STOCK_INVESTMENT_DSS_IQN_BACKTEST_SCORE_MODE"
Write-Host "Seed:       $env:STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST"
Write-Host ""

& $pythonExe -m stock_investment_dss.runner.run_iqn_learning_curve_multiseed_launcher
