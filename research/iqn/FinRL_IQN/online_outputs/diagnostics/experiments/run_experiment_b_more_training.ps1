# Experiment B: more training + slower epsilon decay
# Isolates: hypotheses 3 and 4 - replay dominance and premature exploration collapse.

. $PSScriptRoot\_common.ps1
Initialize-BaselineEnvironment

$env:STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_TOTAL_STEPS = "50000"
$env:STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_EVAL_INTERVAL = "10000"
$env:STOCK_INVESTMENT_DSS_IQN_EPSILON_DECAY_STEPS = "40000"

Invoke-DiagnosticExperiment -Name "experiment_b_more_training" `
    -Description "2x training steps (50k) and slower epsilon decay (40k). Same period, same seeds."
