#!/usr/bin/env bash
# =============================================================================
# Clean 25k baseline launcher (macOS / Linux bash)
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
#   - Python environment activated (conda activate stockdss or equivalent)
#
# To launch training after approval, uncomment the last command block.
# =============================================================================

set -euo pipefail

MANIFEST="configs/experiments/clean_25k_baseline_v1.json"
export PYTHONPATH="src"

# =============================================================================
# STEP 0: Guard — verify frozen data file exists
# =============================================================================
DATA_FILE="data/market/daily/imports/market_data_demo10_new_2010_2026.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "[ABORT] Frozen data file not found: $DATA_FILE" >&2
    echo "Run the download script first to download and freeze the data." >&2
    exit 1
fi
echo "[OK] Frozen data file found: $DATA_FILE"

# =============================================================================
# STEP 1: Config verification
# verify_experiment_config.py clears all IQN env vars internally before checking.
# =============================================================================
echo ""
echo "========================================================"
echo " [STEP 1] Verifying config against manifest"
echo " Manifest: $MANIFEST"
echo "========================================================"

python -m stock_investment_dss.runner.verify_experiment_config --config "$MANIFEST"

echo ""
echo "[OK] Config verification PASSED."

# =============================================================================
# STEP 2: W&B group for this clean run
# =============================================================================
export STOCK_INVESTMENT_DSS_WANDB_ENABLED="true"
export STOCK_INVESTMENT_DSS_WANDB_PROJECT="StockInvestmentDSS"
export STOCK_INVESTMENT_DSS_WANDB_GROUP="clean_25k_baseline_v1"

# =============================================================================
# STEP 3: Print training command — NOT yet launched
# =============================================================================
echo ""
echo "========================================================"
echo " [STEP 2] Training command (NOT yet launched)"
echo "========================================================"
echo ""
echo "  Recommended (manifest-driven, with inline re-verification):"
echo "  python -m stock_investment_dss.runner.run_iqn_experiment_from_config --config $MANIFEST"
echo ""
echo "[WAITING] User approval required before training is launched."
echo "Uncomment the training block below and re-run this script to proceed."
echo ""

# =============================================================================
# STEP 4 (COMMENTED OUT — uncomment after user approval)
# =============================================================================
# echo "========================================================"
# echo " [STEP 4] Launching training (manifest-driven)"
# echo "========================================================"
# python -m stock_investment_dss.runner.run_iqn_experiment_from_config --config "$MANIFEST"
# echo "[OK] Training completed."
