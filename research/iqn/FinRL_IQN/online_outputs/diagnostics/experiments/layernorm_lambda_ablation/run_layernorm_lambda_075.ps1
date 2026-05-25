# =============================================================================
# LayerNorm Lambda Ablation — Variant D: q50_minus_cvar_penalty, lambda=0.75
# =============================================================================
# This is the REFERENCE run — same settings as the official Mode B thesis run
# (run_mode_b_repro_demo5_iqn_multiseed_layernorm.ps1) but with isolated
# dataset_id for direct comparison within this ablation series.
#
# Expected: Reproduces official result — 4/5 active seeds, SELL-dominance,
# returns -2.3% to +3.8%, seed 5 = no-trade.
#
# DO NOT MODIFY SOURCE FILES.
# This script only sets environment variables and runs the launcher.
# =============================================================================

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

# Python executable
$pythonExe = "C:\Users\gurug\miniconda3\envs\stockdss\python.exe"

# Pre-flight check
Write-Host "python exe: $pythonExe"
& $pythonExe -c "import matplotlib; print('matplotlib ok:', matplotlib.__version__)"
& $pythonExe -c "import torch; print('torch ok:', torch.__version__)"
& $pythonExe -c "
import os; os.environ['STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM']='true'
from stock_investment_dss.rl.config.iqn_config import build_iqn_config
cfg = build_iqn_config()
print('use_layer_norm =', cfg.use_layer_norm)
"

# Dataset / universe
$env:STOCK_INVESTMENT_DSS_DAILY_DATASET_ID    = "demo_5_layernorm_lambda_075"
$env:STOCK_INVESTMENT_DSS_PIT_SPLIT_ID        = "demo_5_layernorm_lambda_075_pit"
$env:STOCK_INVESTMENT_DSS_UNIVERSE_ID         = "demo_5"
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE = "demo_5"
$env:STOCK_INVESTMENT_DSS_FINRL_TICKERS       = "AAPL,MSFT,NVDA,AMZN,GOOGL"

# PIT window
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_START         = "2018-01-01"
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_END           = "2024-02-01"
$env:STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME        = "2023-01-01"
$env:STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE       = "2024-02-01"
$env:STOCK_INVESTMENT_DSS_PIT_MIN_TICKERS_PER_DATE = "5"

# Mode B frozen data
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE      = "true"
$env:STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD      = "false"
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD = "false"
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE    = "data/market/daily/imports/market_data_full_500.csv"
$env:STOCK_INVESTMENT_DSS_ALLOW_IMPORT_FALLBACK     = "true"
$env:STOCK_INVESTMENT_DSS_REQUIRE_ALL_TICKERS       = "true"

# Download settings (unused in Mode B but defined for completeness)
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_CHUNK_SIZE    = "25"
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_SLEEP_SECONDS = "2"
$env:STOCK_INVESTMENT_DSS_YFINANCE_IMPERSONATE     = "firefox135"
$env:STOCK_INVESTMENT_DSS_YFINANCE_TIMEOUT_SECONDS = "30"

# IQN config
$env:STOCK_INVESTMENT_DSS_IQN_CONFIG_PRESET                = "stockdss_long_v1"
$env:STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST              = "1,2,3,4,5"
$env:STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_TOTAL_STEPS   = "25000"
$env:STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_EVAL_INTERVAL = "5000"
$env:STOCK_INVESTMENT_DSS_IQN_LEARNING_STARTS              = "2000"
$env:STOCK_INVESTMENT_DSS_IQN_DISABLE_CHANGE_STRATEGY      = "true"

# Ablation variable D: q50_minus_cvar_penalty, lambda=0.75 (reference / official)
$env:STOCK_INVESTMENT_DSS_IQN_BACKTEST_SCORE_MODE  = "q50_minus_cvar_penalty"
$env:STOCK_INVESTMENT_DSS_IQN_BACKTEST_RISK_LAMBDA = "0.75"

# LayerNorm fix
$env:STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM = "true"

# W&B
$env:STOCK_INVESTMENT_DSS_WANDB_ENABLED = "true"
$env:STOCK_INVESTMENT_DSS_WANDB_PROJECT = "StockInvestmentDSS"
$env:STOCK_INVESTMENT_DSS_WANDB_ENTITY  = "guldmand-SDU"

Write-Host ""
Write-Host "=== Ablation D: LayerNorm + q50_minus_cvar_penalty (lambda=0.75) [REFERENCE] ==="
Write-Host "Dataset:    $env:STOCK_INVESTMENT_DSS_DAILY_DATASET_ID"
Write-Host "Score mode: $env:STOCK_INVESTMENT_DSS_IQN_BACKTEST_SCORE_MODE"
Write-Host "Lambda:     $env:STOCK_INVESTMENT_DSS_IQN_BACKTEST_RISK_LAMBDA"
Write-Host "Seeds:      $env:STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST"
Write-Host ""

& $pythonExe -m stock_investment_dss.runner.run_iqn_learning_curve_multiseed_launcher
