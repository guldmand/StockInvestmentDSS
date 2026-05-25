# =============================================================================
# copilot-diagnostics/experiments/_common.ps1
# =============================================================================
# Shared baseline env vars for all IQN HOLD-collapse ablation experiments.
# Mirrors run_mode_b_repro_demo5_iqn.ps1 (demo_5 long-PIT) so we compare apples
# to apples against the 2026_05_20_054007_... baseline summary.
#
# Each experiment script:
#   1. dot-sources this file:        . $PSScriptRoot\_common.ps1
#   2. overrides the variables it wants to test
#   3. calls Invoke-DiagnosticExperiment -Name "<name>"
#
# This file does NOT modify anything in src/, configs/ or other project folders.
# =============================================================================

$ErrorActionPreference = "Stop"

# Project root resolved relative to this file (copilot-diagnostics/experiments/)
$Script:ProjectRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$Script:DiagnosticsRoot = (Resolve-Path "$PSScriptRoot\..").Path
$Script:ResultsRoot = Join-Path $Script:DiagnosticsRoot "results"

function Initialize-BaselineEnvironment {
    $env:PYTHONPATH = "src"

    # --- Dataset / universe (demo_5 long-PIT, same as run_mode_b) -------------
    $env:STOCK_INVESTMENT_DSS_DAILY_DATASET_ID  = "demo_5_long_2018_2024_v28_repro"
    $env:STOCK_INVESTMENT_DSS_PIT_SPLIT_ID      = "demo_5_long_2018_2024_v28_repro_pit"
    $env:STOCK_INVESTMENT_DSS_UNIVERSE_ID       = "demo_5"
    $env:STOCK_INVESTMENT_DSS_DAILY_DATA_UNIVERSE = "demo_5"
    $env:STOCK_INVESTMENT_DSS_FINRL_TICKERS     = "AAPL,MSFT,NVDA,AMZN,GOOGL"

    # --- PIT window (default; experiments may override) -----------------------
    $env:STOCK_INVESTMENT_DSS_DAILY_DATA_START  = "2018-01-01"
    $env:STOCK_INVESTMENT_DSS_DAILY_DATA_END    = "2024-02-01"
    $env:STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME = "2023-01-01"
    $env:STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE = "2024-02-01"
    $env:STOCK_INVESTMENT_DSS_PIT_MIN_TICKERS_PER_DATE = "5"

    # --- Reproducible data, no live download ----------------------------------
    $env:STOCK_INVESTMENT_DSS_DAILY_DATA_USE_CACHE     = "true"
    $env:STOCK_INVESTMENT_DSS_FORCE_FINRL_DOWNLOAD     = "false"
    $env:STOCK_INVESTMENT_DSS_DAILY_DATA_ALLOW_DOWNLOAD = "false"
    $env:STOCK_INVESTMENT_DSS_DAILY_DATA_IMPORT_FILE   = "data/market/daily/imports/market_data_full_500.csv"
    $env:STOCK_INVESTMENT_DSS_ALLOW_IMPORT_FALLBACK    = "true"
    $env:STOCK_INVESTMENT_DSS_REQUIRE_ALL_TICKERS      = "true"

    # --- FinRL env defaults (experiments may override cost) -------------------
    $env:STOCK_INVESTMENT_DSS_FINRL_ENV_INITIAL_AMOUNT  = "1000000.0"
    $env:STOCK_INVESTMENT_DSS_FINRL_ENV_HMAX            = "10000"
    $env:STOCK_INVESTMENT_DSS_FINRL_ENV_BUY_COST_PCT    = "0.001"
    $env:STOCK_INVESTMENT_DSS_FINRL_ENV_SELL_COST_PCT   = "0.001"
    $env:STOCK_INVESTMENT_DSS_FINRL_ENV_REWARD_SCALING  = "0.0001"

    # --- IQN preset + training schedule ---------------------------------------
    $env:STOCK_INVESTMENT_DSS_IQN_CONFIG_PRESET                  = "stockdss_long_v1"
    $env:STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_TOTAL_STEPS     = "25000"
    $env:STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_EVAL_INTERVAL   = "5000"
    $env:STOCK_INVESTMENT_DSS_IQN_LEARNING_STARTS                = "2000"
    $env:STOCK_INVESTMENT_DSS_IQN_EPSILON_DECAY_STEPS            = "25000"
    $env:STOCK_INVESTMENT_DSS_IQN_EPSILON_FINAL                  = "0.05"

    # --- Decision/eval scoring (q50 ablation, like the diagnosed baseline) ---
    $env:STOCK_INVESTMENT_DSS_IQN_BACKTEST_SCORE_MODE  = "q50"
    $env:STOCK_INVESTMENT_DSS_IQN_BACKTEST_RISK_LAMBDA = "0.75"
    $env:STOCK_INVESTMENT_DSS_IQN_DISABLE_CHANGE_STRATEGY = "true"

    # --- Seeds (all 5) ---------------------------------------------------------
    $env:STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST                = "1,2,3,4,5"
    $env:STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_STOP_ON_FAILURE     = "false"
    $env:STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_RUN_SUMMARY_AFTER   = "true"
    $env:STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_DEDUPLICATE_SEEDS   = "true"

    # --- W&B disabled for diagnostics (avoid polluting project) ---------------
    $env:STOCK_INVESTMENT_DSS_WANDB_ENABLED = "false"

    # --- Reward/action diagnostic seed list -----------------------------------
    $env:STOCK_INVESTMENT_DSS_IQN_REWARD_DIAGNOSTIC_SEEDS = "1,2,3,4,5"
}

function Get-RelevantEnvSnapshot {
    $keys = @(
        'PYTHONPATH',
        'STOCK_INVESTMENT_DSS_DAILY_DATASET_ID',
        'STOCK_INVESTMENT_DSS_PIT_SPLIT_ID',
        'STOCK_INVESTMENT_DSS_UNIVERSE_ID',
        'STOCK_INVESTMENT_DSS_FINRL_TICKERS',
        'STOCK_INVESTMENT_DSS_DAILY_DATA_START',
        'STOCK_INVESTMENT_DSS_DAILY_DATA_END',
        'STOCK_INVESTMENT_DSS_PIT_POINT_IN_TIME',
        'STOCK_INVESTMENT_DSS_PIT_TRADE_END_DATE',
        'STOCK_INVESTMENT_DSS_FINRL_ENV_BUY_COST_PCT',
        'STOCK_INVESTMENT_DSS_FINRL_ENV_SELL_COST_PCT',
        'STOCK_INVESTMENT_DSS_FINRL_ENV_REWARD_SCALING',
        'STOCK_INVESTMENT_DSS_IQN_CONFIG_PRESET',
        'STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_TOTAL_STEPS',
        'STOCK_INVESTMENT_DSS_IQN_LEARNING_CURVE_EVAL_INTERVAL',
        'STOCK_INVESTMENT_DSS_IQN_LEARNING_STARTS',
        'STOCK_INVESTMENT_DSS_IQN_EPSILON_DECAY_STEPS',
        'STOCK_INVESTMENT_DSS_IQN_EPSILON_FINAL',
        'STOCK_INVESTMENT_DSS_IQN_BACKTEST_SCORE_MODE',
        'STOCK_INVESTMENT_DSS_IQN_BACKTEST_RISK_LAMBDA',
        'STOCK_INVESTMENT_DSS_IQN_DISABLE_CHANGE_STRATEGY',
        'STOCK_INVESTMENT_DSS_IQN_MULTI_SEED_LIST',
        'STOCK_INVESTMENT_DSS_IQN_GRAD_CLIP_NORM',
        'STOCK_INVESTMENT_DSS_IQN_STATE_NORM_SCALE',
        'STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM'
    )
    $snapshot = @{}
    foreach ($k in $keys) {
        $val = [Environment]::GetEnvironmentVariable($k)
        if ($null -ne $val) { $snapshot[$k] = $val }
    }
    return $snapshot
}

function Invoke-DiagnosticExperiment {
    param(
        [Parameter(Mandatory)] [string] $Name,
        [string] $Description = ""
    )

    Push-Location $Script:ProjectRoot
    try {
        $expDir = Join-Path $Script:ResultsRoot $Name
        New-Item -ItemType Directory -Force -Path $expDir | Out-Null

        # Save config snapshot before run -------------------------------------
        $snapshot = Get-RelevantEnvSnapshot
        $payload = @{
            experiment_name = $Name
            description     = $Description
            started_at      = (Get-Date).ToString("o")
            env             = $snapshot
        }
        $payload | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $expDir "effective_config.json")

        Write-Host ""
        Write-Host "=========================================================="
        Write-Host " Experiment: $Name"
        if ($Description) { Write-Host "   $Description" }
        Write-Host " Results dir: $expDir"
        Write-Host "=========================================================="

        # Track latest run before launching so we can identify the new ones ---
        $beforeRuns = @{}
        $runsRoot = Join-Path $Script:ProjectRoot "outputs\runs"
        if (Test-Path $runsRoot) {
            Get-ChildItem $runsRoot -Directory | ForEach-Object { $beforeRuns[$_.Name] = $true }
        }

        # 1. Run multiseed launcher (5 seeds + summary) -----------------------
        $logFile = Join-Path $expDir "multiseed_launcher.log"
        Write-Host "[$Name] Launching multiseed (logs -> $logFile)"
        & python -m stock_investment_dss.runner.run_iqn_learning_curve_multiseed_launcher *>&1 |
            Tee-Object -FilePath $logFile

        # 2. Find new multiseed_summary run -----------------------------------
        $afterRuns = Get-ChildItem $runsRoot -Directory | Where-Object {
            -not $beforeRuns.ContainsKey($_.Name) -and $_.Name -like "*iqn_learning_curve_multiseed_summary"
        } | Sort-Object Name -Descending

        if (-not $afterRuns -or $afterRuns.Count -eq 0) {
            Write-Warning "[$Name] No new multiseed_summary run found. Skipping diagnostic step."
            return
        }

        $summaryRun = $afterRuns[0]
        Write-Host "[$Name] Using summary run: $($summaryRun.Name)"

        # 3. Run reward/action diagnostic against that summary ----------------
        $env:STOCK_INVESTMENT_DSS_IQN_REWARD_DIAGNOSTIC_SOURCE_SUMMARY_RUN_ID = $summaryRun.Name
        $diagLog = Join-Path $expDir "reward_action_diagnostic.log"
        Write-Host "[$Name] Running reward/action diagnostic"
        & python -m stock_investment_dss.runner.run_iqn_reward_action_diagnostic *>&1 |
            Tee-Object -FilePath $diagLog

        Remove-Item Env:STOCK_INVESTMENT_DSS_IQN_REWARD_DIAGNOSTIC_SOURCE_SUMMARY_RUN_ID -ErrorAction SilentlyContinue

        # Find newest reward_action_diagnostic run
        $diagRuns = Get-ChildItem $runsRoot -Directory |
            Where-Object { $_.Name -like "*iqn_reward_action_diagnostic" } |
            Sort-Object Name -Descending

        # 4. Copy aggregated artefacts into experiment results folder ---------
        $artefactCopies = @{
            'summary\iqn_learning_curve_multiseed_summary.json'          = 'multiseed_summary.json'
            'summary\iqn_learning_curve_multiseed_final_records.csv'     = 'multiseed_final_records.csv'
            'summary\iqn_learning_curve_multiseed_aggregate_by_step.csv' = 'multiseed_aggregate_by_step.csv'
            'data\iqn_learning_curve_multiseed_eval_records.csv'         = 'multiseed_eval_records.csv'
            'data\iqn_learning_curve_multiseed_run_index.csv'            = 'multiseed_run_index.csv'
        }
        foreach ($src in $artefactCopies.Keys) {
            $srcPath = Join-Path $summaryRun.FullName $src
            if (Test-Path $srcPath) {
                $dstPath = Join-Path $expDir $artefactCopies[$src]
                Copy-Item -Force $srcPath $dstPath
            }
        }

        if ($diagRuns -and $diagRuns.Count -gt 0) {
            $diagRun = $diagRuns[0]
            $diagArtefacts = @(
                'summary\iqn_reward_action_diagnostic_by_seed.csv',
                'summary\iqn_reward_action_diagnostic_training_by_action.csv',
                'summary\iqn_reward_action_diagnostic_status_comparison.csv',
                'summary\iqn_reward_action_diagnostic_summary.json',
                'summary\iqn_reward_action_diagnostic_summary.md',
                'plots\iqn_reward_action_training_action_counts.png',
                'plots\iqn_reward_action_hold_buy_gap.png'
            )
            foreach ($a in $diagArtefacts) {
                $srcPath = Join-Path $diagRun.FullName $a
                if (Test-Path $srcPath) {
                    $dstPath = Join-Path $expDir (Split-Path $a -Leaf)
                    Copy-Item -Force $srcPath $dstPath
                }
            }
        }

        # Append completion metadata
        $payload.completed_at = (Get-Date).ToString("o")
        $payload.summary_run_id = $summaryRun.Name
        if ($diagRuns -and $diagRuns.Count -gt 0) {
            $payload.reward_action_diagnostic_run_id = $diagRuns[0].Name
        }
        $payload | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $expDir "effective_config.json")

        Write-Host "[$Name] Done. Artefacts: $expDir"
    }
    finally {
        Pop-Location
    }
}
