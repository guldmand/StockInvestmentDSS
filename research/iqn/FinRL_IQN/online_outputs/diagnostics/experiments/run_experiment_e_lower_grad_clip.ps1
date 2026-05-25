# =============================================================================
# copilot-diagnostics/experiments/run_experiment_e_lower_grad_clip.ps1
# =============================================================================
# Experiment E: Lower gradient clip norm (max_norm = 1.0, was 10.0).
#
# Hypothesis:
#   Seeds 2 and 5 suffer from Q-value divergence caused by unchecked gradient
#   updates early in training. The current max_norm=10.0 is too permissive:
#   when internal network activations are large (supporting Q-values ~7M), the
#   gradient norm at early layers exceeds 10 and still drives the weights further
#   away from convergence.
#
#   Reducing max_norm to 1.0 constrains each gradient step much more tightly
#   and should prevent the initial Q-value explosion that seeds 2/5 experience.
#
# Expected outcome if hypothesis is correct:
#   - loss_final for seeds 2 and 5 drops from ~15-18M to < 100,000
#   - hold_minus_buy_score becomes small / negative for previously dead seeds
#   - training_sell_count for seeds 2 and 5 drops from ~14,500 to ~7,000
#   - At least 4/5 seeds become active_trading
#
# This is the PRIMARY fix candidate for the HOLD-collapse problem.
#
# Control: all other env vars are identical to the diagnosed baseline.
# Changed: STOCK_INVESTMENT_DSS_IQN_GRAD_CLIP_NORM = 1.0
# =============================================================================

. $PSScriptRoot\_common.ps1

Initialize-BaselineEnvironment

# Override: tighten gradient clipping
$env:STOCK_INVESTMENT_DSS_IQN_GRAD_CLIP_NORM = "1.0"

Invoke-DiagnosticExperiment `
    -Name "experiment_e_lower_grad_clip" `
    -Description "max_norm=1.0 (was 10.0). Tests whether gradient explosion is the primary driver of Q-value divergence in seeds 2 and 5."
