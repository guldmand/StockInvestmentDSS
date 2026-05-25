# =============================================================================
# copilot-diagnostics/experiments/run_experiment_g_layer_norm.ps1
# =============================================================================
# Experiment G: LayerNorm inside IQNNetwork (use_layer_norm = true).
#
# Background from Experiment F:
#   Experiment F tested state_norm_scale=1000 (uniform division of all obs
#   features by 1000). This made ALL 5 seeds collapse to no_trade, because
#   it created a new scale mismatch: cash=1000 vs prices=0.1–0.5 vs holdings
#   =0–0.1. The network became "blind" to price information, and the cash-
#   dominated representation made SELL universally preferred (~57% of steps).
#   Simple uniform scalar normalization is therefore not the answer.
#
# Hypothesis (REVISED ROOT CAUSE FIX):
#   The problem is not the input scale per se but that Kaiming-initialized
#   weights, combined with features of vastly different magnitudes, produce
#   initial Q-value std ≈ 150,000 — while true Q* ≈ 100. The ordering of
#   Q(HOLD) vs Q(BUY) vs Q(SELL) at initialization determines the policy for
#   the first 25,000 training steps, since bootstrapping from Q_init ≈ 150k
#   to Q* ≈ 100 requires ~728 target update cycles (728,000 steps).
#
#   LayerNorm after the first linear layer of the state encoder normalizes the
#   hidden activations to mean=0, std=1 regardless of input scale. This:
#     1. Eliminates the cash-dominance problem (all features contribute equally)
#     2. Reduces initial Q-value std to O(1) instead of O(150,000)
#     3. Makes all seeds start from a well-conditioned initial Q distribution
#     4. Allows the actual reward signal to shape the Q-ordering from step 1
#
# Architecture change:
#   state_encoder: Linear(state_dim → 128)
#     → [+ LayerNorm(128)]   ← NEW in Exp G
#     → ReLU
#
# Expected outcome if hypothesis is correct:
#   - Seeds 2 and 5 become active_trading (loss < 100,000)
#   - Seeds 3 and 4 remain active_trading with similar or better returns
#   - Seed variance reduced: all 5 seeds show meaningful trading
#   - training_sell_count normalizes to ~30–40% across all seeds
#   - loss_final consistent across seeds (no seed-specific divergence)
#
# Control: all other env vars are identical to the diagnosed baseline.
#   state_norm_scale is reset to 1.0 (no input scaling, only LayerNorm).
# Changed: STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM = true
# =============================================================================

. $PSScriptRoot\_common.ps1

Initialize-BaselineEnvironment

# Override: add LayerNorm to IQNNetwork state encoder
$env:STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM = "true"

# Ensure input scaling is off (isolate LayerNorm effect)
$env:STOCK_INVESTMENT_DSS_IQN_STATE_NORM_SCALE = "1.0"

Invoke-DiagnosticExperiment `
    -Name "experiment_g_layer_norm" `
    -Description "use_layer_norm=true. Tests whether LayerNorm in the IQN state encoder fixes Q-value divergence by normalizing hidden activations regardless of raw input scale."
