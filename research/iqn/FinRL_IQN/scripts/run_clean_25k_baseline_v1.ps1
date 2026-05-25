# =============================================================================
# Clean 25k baseline launcher (Windows PowerShell)
# Experiment: clean_25k_baseline_v1
# =============================================================================
#
# This script:
#   1. Verifies the effective config against configs/experiments/clean_25k_baseline_v1.json
#   2. If verification passes, prints the training command
#   3. Does NOT launch training — user must approve before executing step 4
#
# Prerequisites:
#   - data/market/daily/imports/market_data_demo10_new_2010_2026.csv must exist
#     (run run_mode_a_download_demo10_new.ps1 first if not present)
#   - W&B credentials configured if WANDB_ENABLED=true
#
# To launch training after approval, uncomment the last command block.
# =============================================================================

$ErrorActionPreference = "Stop"

$MANIFEST   = "configs/experiments/clean_25k_baseline_v1.json"
$PythonExe  = "C:\Users\gurug\miniconda3\envs\stockdss\python.exe"
$env:PYTHONPATH = "src"

# =============================================================================
# STEP 0: Guard — verify frozen data file exists before attempting verification
# =============================================================================
$DataFile = "data\market\daily\imports\market_data_demo10_new_2010_2026.csv"
if (-not (Test-Path $DataFile)) {
    Write-Error "[ABORT] Frozen data file not found: $DataFile"
    Write-Error "Run run_mode_a_download_demo10_new.ps1 first to download and freeze."
    exit 1
}
Write-Host "[OK] Frozen data file found: $DataFile"

# =============================================================================
# STEP 1: Config verification
# verify_experiment_config.py clears all IQN env vars internally before checking.
# No need to clear them here — the verifier owns that responsibility.
# =============================================================================
Write-Host ""
Write-Host "========================================================"
Write-Host " [STEP 1] Verifying config against manifest"
Write-Host " Manifest: $MANIFEST"
Write-Host "========================================================"

& $PythonExe -m stock_investment_dss.runner.verify_experiment_config --config $MANIFEST

if ($LASTEXITCODE -ne 0) {
    Write-Error "[ABORT] Config verification FAILED. Training NOT launched."
    Write-Error "Fix the manifest or .env contamination before proceeding."
    exit 1
}

Write-Host ""
Write-Host "[OK] Config verification PASSED."

# =============================================================================
# STEP 2: W&B group for this clean run
# =============================================================================
$env:STOCK_INVESTMENT_DSS_WANDB_ENABLED = "true"
$env:STOCK_INVESTMENT_DSS_WANDB_PROJECT = "StockInvestmentDSS"
$env:STOCK_INVESTMENT_DSS_WANDB_ENTITY  = "guldmand-SDU"
$env:STOCK_INVESTMENT_DSS_WANDB_GROUP   = "clean_25k_baseline_v1"

# =============================================================================
# STEP 3: Print training command — NOT yet launched
# =============================================================================
Write-Host ""
Write-Host "========================================================"
Write-Host " [STEP 2] Training command (NOT yet launched)"
Write-Host "========================================================"
Write-Host ""
Write-Host "  Recommended (manifest-driven, with inline re-verification):"
Write-Host "  & `"$PythonExe`" -m stock_investment_dss.runner.run_iqn_experiment_from_config --config $MANIFEST"
Write-Host ""
Write-Host "  Alternative (equivalent, if you verified manually):"
Write-Host "  & `"$PythonExe`" -m stock_investment_dss.runner.run_iqn_learning_curve_multiseed_launcher"
Write-Host "  (requires env vars to be set — use run_iqn_experiment_from_config instead)"
Write-Host ""
Write-Host "[WAITING] User approval required before training is launched."
Write-Host "Uncomment the training block below and re-run this script to proceed."
Write-Host ""

# =============================================================================
# STEP 4 (COMMENTED OUT — uncomment after user approval)
# =============================================================================
# Write-Host "========================================================"
# Write-Host " [STEP 4] Launching training (manifest-driven)"
# Write-Host "========================================================"
# & $PythonExe -m stock_investment_dss.runner.run_iqn_experiment_from_config --config $MANIFEST
# if ($LASTEXITCODE -ne 0) {
#     Write-Error "[ERROR] Training run failed with exit code $LASTEXITCODE"
#     exit $LASTEXITCODE
# }
# Write-Host "[OK] Training completed."
