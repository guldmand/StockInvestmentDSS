# Experiment A: no CVaR in training (risk_lambda=0)
# Isolates: hypothesis 2 - CVaR penalty pushing Z(s, BUY) downward.

. $PSScriptRoot\_common.ps1
Initialize-BaselineEnvironment

$env:STOCK_INVESTMENT_DSS_IQN_BACKTEST_RISK_LAMBDA = "0.0"

Invoke-DiagnosticExperiment -Name "experiment_a_no_cvar" `
    -Description "risk_lambda=0.0 (no CVaR penalty in training/eval). Same period, same training length, same seeds as baseline."
