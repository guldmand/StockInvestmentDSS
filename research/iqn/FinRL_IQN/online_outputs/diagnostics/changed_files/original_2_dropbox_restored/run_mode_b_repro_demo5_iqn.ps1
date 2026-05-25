# =============================================================================
# Mode B: Reproducible thesis experiment
# =============================================================================
# Purpose:
# - Run IQN demo_5 using reproducible frozen data/cache/import.
# - Do NOT depend on live Yahoo download.
# - Report must clearly say final source used = cache/import/master if applicable.

$ErrorActionPreference = "Stop"

$env:PYTHONPATH="src"

# Dataset / universe
$env:STOCK_INVESTMENT_DSS_DAILY_DATASET_ID="demo_5_long_2018_2024_v28_repro"
$env:STOCK_INVESTMENT_DSS_PIT_SPLIT_ID="demo_5_long_2018_2024_v28_repro_pit"
$env:STOCK_INVESTMENT_DSS_UNIVERSE_ID="demo_5"
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE="demo_5"
$env:STOCK_INVESTMENT_DSS_FINRL_TICKERS="AAPL,MSFT,NVDA,AMZN,GOOGL"

# Long PIT window
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_START="2018-01-01"
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_END="2024-02-01"
$env:STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME="2023-01-01"
$env:STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE="2024-02-01"
$env:STOCK_INVESTMENT_DSS_PIT_MIN_TICKERS_PER_DATE="5"

# Reproducible data mode: no live download dependency
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE="true"
$env:STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD="false"
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD="false"
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE="data/market/daily/imports/market_data_full_500.csv"
$env:STOCK_INVESTMENT_DSS_ALLOW_IMPORT_FALLBACK="true"
$env:STOCK_INVESTMENT_DSS_REQUIRE_ALL_TICKERS="true"

# Keep download settings defined, but they should not be used in Mode B
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_CHUNK_SIZE="25"
$env:STOCK_INVESTMENT_DSS_DAILY_DATA_SLEEP_SECONDS="2"
$env:STOCK_INVESTMENT_DSS_YFINANCE_IMPERSONATE="firefox135"
$env:STOCK_INVESTMENT_DSS_YFINANCE_TIMEOUT_SECONDS="30"

# IQN config
$env:STOCK_INVESTMENT_DSS_IQN_CONFIG_PRESET="stockdss_long_v1"
$env:STOCK_INVESTMENT_DSS_RANDOM_SEED="7"
Remove-Item Env:STOCK_INVESTMENT_DSS_IQN_SEED -ErrorAction SilentlyContinue

$env:STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_TOTAL_STEPS="25000"
$env:STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_EVAL_INTERVAL="5000"
$env:STOCK_INVESTMENT_DSS_IQN_LEARNING_STARTS="2000"
$env:STOCK_INVESTMENT_DSS_IQN_BACKTEST_SCORE_MODE="q50_minus_cvar_penalty"
$env:STOCK_INVESTMENT_DSS_IQN_BACKTEST_RISK_LAMBDA="0.75"
$env:STOCK_INVESTMENT_DSS_IQN_DISABLE_CHANGE_STRATEGY="true"

# W&B. Keep WANDB_API_KEY outside this script.
$env:STOCK_INVESTMENT_DSS_WANDB_ENABLED="true"
$env:STOCK_INVESTMENT_DSS_WANDB_PROJECT="StockInvestmentDSS"
$env:STOCK_INVESTMENT_DSS_WANDB_ENTITY="guldmand-SDU"

python -m stock_investment_dss.runner.run_iqn_learning_curve_smoke_test
python -m stock_investment_dss.runner.run_iqn_decision_audit_report

$run = Get-ChildItem .\outputs\runs -Directory |
  Where-Object { $_.Name -like "*iqn_learning_curve_smoke_test" } |
  Sort-Object Name -Descending |
  Select-Object -First 1

$audit = Get-ChildItem .\outputs\runs -Directory |
  Where-Object { $_.Name -like "*iqn_decision_audit_report" } |
  Sort-Object Name -Descending |
  Select-Object -First 1

Write-Host "`nLatest IQN run:" $run.FullName
Write-Host "`nExperiment context:"
Get-Content "$($run.FullName)\summary\experiment_context_summary.md"

Write-Host "`nData provenance:"
Get-Content "$($run.FullName)\data\data_provenance_summary.json"

Write-Host "`nDecision audit by ticker/action:"
Import-Csv "$($audit.FullName)\audit\decision_audit_by_ticker_action.csv" |
  Format-Table -AutoSize
