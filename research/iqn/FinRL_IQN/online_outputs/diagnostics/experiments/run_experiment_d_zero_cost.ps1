# Experiment D: zero transaction costs
# Isolates: hypothesis 1 - asymmetric per-step BUY reward due to finrl cost.

. $PSScriptRoot\_common.ps1
Initialize-BaselineEnvironment

$env:STOCK_INVESTMENT_DSS_FINRL_ENV_BUY_COST_PCT = "0.0"
$env:STOCK_INVESTMENT_DSS_FINRL_ENV_SELL_COST_PCT = "0.0"

Invoke-DiagnosticExperiment -Name "experiment_d_zero_cost" `
    -Description "buy_cost_pct=0 and sell_cost_pct=0. Removes per-step asymmetry where BUY is guaranteed negative reward vs HOLD=0 cost."
