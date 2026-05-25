# Experiment C: longer training window
# Isolates: hypothesis 6 - regime/data distribution effects on BUY-Q estimation.

. $PSScriptRoot\_common.ps1
Initialize-BaselineEnvironment

$env:STOCK_INVESTMENT_DSS_DAILY_DATA_START = "2015-01-01"
$env:STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME = "2022-01-01"
$env:STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE = "2024-02-01"

Invoke-DiagnosticExperiment -Name "experiment_c_longer_window" `
    -Description "Wider train window 2015-2022 (instead of 2018-2023). More regime variation in replay. Same seeds and training length."
