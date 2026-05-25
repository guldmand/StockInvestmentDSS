# =============================================================================
# copilot-diagnostics/experiments/run_experiment_f_state_norm.ps1
# =============================================================================
# Experiment F: State normalization (state_norm_scale = 1000).
#
# Hypothesis (ROOT CAUSE):
#   The FinRL state vector contains raw unnormalized feature values:
#     - cash:     ~1,000,000  (7 orders of magnitude above unit scale)
#     - prices:   ~100–800    (2–3 orders of magnitude)
#     - holdings: ~0–100
#     - tech:     small but variable
#
#   IQNNetwork uses Kaiming initialization, which assumes unit-scale inputs.
#   With cash = 1,000,000 feeding into Linear(state_dim → 128), the initial
#   network activations are ~O(1,000,000 × 0.24) ≈ 240,000 per neuron.
#   After propagating through the network, the initial Q-value std ≈ 150,000.
#
#   True Q* ≈ 100 (reward_scaling = 0.0001 × portfolio_change).
#   Convergence from Q_init ≈ 150,000 to Q* ≈ 100 via bootstrapping requires
#   approximately ln(150000/100) / (1 − 0.99) ≈ 728 target update cycles
#   = 728,000 training steps. We only run 25,000 steps (25 target updates).
#
#   After 25 target updates: Q ≈ Q_init × 0.99^25 ≈ Q_init × 0.778.
#   The policy is therefore almost entirely determined by the RANDOM INITIALIZATION.
#   Seeds 2 and 5 happen to initialize with Q(SELL) >> Q(HOLD) >> Q(BUY),
#   causing the agent to SELL during all training → portfolio emptied → cash-only
#   at backtest start → Q(HOLD) >> Q(BUY) → always HOLD.
#
# Fix:
#   Divide the entire observation vector by state_norm_scale = 1000 before
#   feeding it to the network. This scales:
#     - cash:     1,000,000 / 1000 = 1,000  (manageable; initial Q std ≈ 75)
#     - prices:   100–800 / 1000   = 0.1–0.8 (visible to the network)
#     - holdings: 0–100 / 1000     = 0–0.1
#     - tech:     small / 1000     = ~0
#
#   With initial Q std ≈ 75 and reward signal ≈ 1, the Q-ordering after just
#   ~25 target updates is shaped more by ACTUAL REWARDS than initialization
#   noise. Seeds 2 and 5 should escape the SELL-dominated initialization.
#
# Expected outcome if hypothesis is correct:
#   - Seeds 2 and 5 become active_trading (loss < 100,000)
#   - Seeds 3 and 4 remain active_trading with similar or better returns
#   - All 5 seeds show lower loss_final
#   - training_sell_count normalizes across seeds (~28% of training steps)
#   - hold_minus_buy_score becomes small / negative for previously dead seeds
#
# Control: all other env vars are identical to the diagnosed baseline.
# Changed: STOCK_INVESTMENT_DSS_IQN_STATE_NORM_SCALE = 1000
# =============================================================================

. $PSScriptRoot\_common.ps1

Initialize-BaselineEnvironment

# Override: divide the observation vector by 1000 before passing to IQN
$env:STOCK_INVESTMENT_DSS_IQN_STATE_NORM_SCALE = "1000"

Invoke-DiagnosticExperiment `
    -Name "experiment_f_state_norm" `
    -Description "state_norm_scale=1000. Tests whether raw FinRL state scale (cash=1M) is the root cause of Q-value divergence and HOLD-collapse in seeds 2 and 5."
