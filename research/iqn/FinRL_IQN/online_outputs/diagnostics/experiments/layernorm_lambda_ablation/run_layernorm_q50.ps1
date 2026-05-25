# =============================================================================
# LayerNorm Lambda Ablation — Variant A: q50 (no CVaR penalty)
# =============================================================================
# Hypothesis: q50_minus_cvar_penalty with lambda=0.75 causes SELL-dominance.
# This script tests pure q50 scoring with LayerNorm enabled.
# Expected: 5/5 seeds active trading (reproduces Exp G in official Mode B setup).
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
print('score_mode = q50 (set via env below)')
"

# Dataset / universe
$env:STOCK_INVESTMENT_DSS_DAILY_DATASET_ID   = "demo_5_layernorm_q50"
$env:STOCK_INVESTMENT_DSS_PIT_SPLIT_ID       = "demo_5_layernorm_q50_pit"
$env:STOCK_INVESTMENT_DSS_UNIVERSE_ID        = "demo_5"
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE = "demo_5"
$env:STOCK_INVESTMENT_DSS_FINRL_TICKERS      = "AAPL,MSFT,NVDA,AMZN,GOOGL"

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
$env:STOCK_INVESTMENT_DSS_IQN_CONFIG_PRESET                  = "stockdss_long_v1"
$env:STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST                = "1,2,3,4,5"
$env:STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_TOTAL_STEPS     = "25000"
$env:STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_EVAL_INTERVAL   = "5000"
$env:STOCK_INVESTMENT_DSS_IQN_LEARNING_STARTS                = "2000"
$env:STOCK_INVESTMENT_DSS_IQN_DISABLE_CHANGE_STRATEGY        = "true"

# Ablation variable A: pure q50, no CVaR penalty
$env:STOCK_INVESTMENT_DSS_IQN_BACKTEST_SCORE_MODE  = "q50"
$env:STOCK_INVESTMENT_DSS_IQN_BACKTEST_RISK_LAMBDA = "0.0"

# LayerNorm fix
$env:STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM = "true"

# W&B
$env:STOCK_INVESTMENT_DSS_WANDB_ENABLED = "true"
$env:STOCK_INVESTMENT_DSS_WANDB_PROJECT = "StockInvestmentDSS"
$env:STOCK_INVESTMENT_DSS_WANDB_ENTITY  = "guldmand-SDU"

Write-Host ""
Write-Host "=== Ablation A: LayerNorm + q50 (lambda=0.0) ==="
Write-Host "Dataset:    $env:STOCK_INVESTMENT_DSS_DAILY_DATASET_ID"
Write-Host "Score mode: $env:STOCK_INVESTMENT_DSS_IQN_BACKTEST_SCORE_MODE"
Write-Host "Lambda:     $env:STOCK_INVESTMENT_DSS_IQN_BACKTEST_RISK_LAMBDA"
Write-Host "Seeds:      $env:STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST"
Write-Host ""

& $pythonExe -m stock_investment_dss.runner.run_iqn_learning_curve_multiseed_launcher
