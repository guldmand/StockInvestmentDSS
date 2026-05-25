# =============================================================================
# copilot-diagnostics/run_all.ps1
# =============================================================================
# Runs all IQN HOLD-collapse ablation experiments sequentially and then builds
# the cross-experiment report.
#
# Original ablations (D, A, B, C): ruled out cost asymmetry, CVaR, training
# length, and data window as root causes. Confirmed Q-value divergence.
#
# Experiment E: tested grad_clip_norm=1.0 — did NOT fix seeds 2 and 5.
#
# Experiment F: tests state_norm_scale=1000 (ROOT CAUSE HYPOTHESIS).
#   Raw FinRL cash=1,000,000 drives initial Q-value std ≈ 150,000.
#   With only 25 target updates (25k steps), the policy is dominated by
#   random init. Dividing by 1000 should reduce initial Q std to ~75.
#
# To run only Experiment F:
#   & .\copilot-diagnostics\experiments\run_experiment_f_state_norm.ps1
#
# Prerequisite:
#   conda activate stockdss
# =============================================================================

$ErrorActionPreference = "Continue"

$root = $PSScriptRoot
$results = Join-Path $root "results"
New-Item -ItemType Directory -Force -Path $results | Out-Null

$mainLog = Join-Path $results "run_all.log"
Start-Transcript -Path $mainLog -Append | Out-Null

Write-Host "=========================================================="
Write-Host " IQN HOLD-collapse diagnostic - run_all"
Write-Host " Started: $((Get-Date).ToString('o'))"
Write-Host "=========================================================="

$experiments = @(
    "run_experiment_d_zero_cost.ps1",
    "run_experiment_a_no_cvar.ps1",
    "run_experiment_b_more_training.ps1",
    "run_experiment_c_longer_window.ps1",
    "run_experiment_e_lower_grad_clip.ps1",
    "run_experiment_f_state_norm.ps1",
    "run_experiment_g_layer_norm.ps1"
)

foreach ($e in $experiments) {
    $path = Join-Path (Join-Path $root "experiments") $e
    Write-Host ""
    Write-Host "----------------------------------------------------------"
    Write-Host " Running: $e"
    Write-Host "----------------------------------------------------------"
    try {
        & $path
    } catch {
        Write-Warning "Experiment $e failed: $_"
    }
}

Write-Host ""
Write-Host "=========================================================="
Write-Host " Building cross-experiment comparison report"
Write-Host "=========================================================="
try {
    & python (Join-Path $root "compare.py")
} catch {
    Write-Warning "compare.py failed: $_"
}

Write-Host ""
Write-Host "Done. Report: $(Join-Path $results 'comparison_report.md')"
Stop-Transcript | Out-Null

