# LayerNorm Robustness Tests — README

## Purpose

These scripts verify that the D-IQN-DSS LayerNorm + q50 result is **not an artefact
of the specific 2018–2024 tech universe used in the primary thesis experiment**.

The primary result (`demo_5_layernorm_q50`, seeds 1–5) achieved:
- Mean return +77.25%, mean Sharpe 2.856, 5/5 active seeds
- Universe: AAPL, MSFT, NVDA, AMZN, GOOGL
- Train: 2018–2023, Eval: 2023–2024

These robustness tests check three generalisation axes:
1. **Time window shift** — same tickers, earlier period (2017–2023, includes bear market)
2. **Training data reduction** — same tickers, same eval, shorter history (2019–2024)
3. **Asset class generalisation** — different tickers (non-tech), same period (2018–2024)

All tests use seed=7 (not in the primary 1–5 seed set) to avoid any overlap.

---

## Scripts

### Test 1: `run_test_1_demo5_2017_2023_layernorm_q50.ps1`

| Parameter | Value |
|-----------|-------|
| Tickers | AAPL, MSFT, NVDA, AMZN, GOOGL |
| Train | 2017-01-01 → 2022-01-01 |
| Eval | 2022-01-01 → 2023-01-01 |
| Dataset ID | `robustness_demo5_2017_2023_layernorm_q50` |
| Seed | 7 |

**What this tests:** Whether LayerNorm + q50 works on the same universe in an *earlier*
and *harder* market regime. The 2022 evaluation year includes a major bear market
(S&P 500 −19%, NASDAQ −33%). A robust policy should at minimum avoid catastrophic
losses (no −20%+ drawdown), even if absolute return is negative.

**Pass criterion:** `total_trades > 0` (agent is not stuck in HOLD-collapse)  
**Fail criterion:** `total_trades == 0` or `return == 0.0` (HOLD-collapse re-appeared)  
**Bonus insight:** Negative return is acceptable given the 2022 regime — look at max_drawdown relative to benchmark.

---

### Test 2: `run_test_2_demo5_2019_2024_layernorm_q50.ps1`

| Parameter | Value |
|-----------|-------|
| Tickers | AAPL, MSFT, NVDA, AMZN, GOOGL |
| Train | 2019-01-01 → 2023-01-01 |
| Eval | 2023-01-01 → 2024-02-01 |
| Dataset ID | `robustness_demo5_2019_2024_layernorm_q50` |
| Seed | 7 |

**What this tests:** Whether the primary result is sensitive to having 2017–2018 data
in the training set. This run uses the same eval window (2023) but starts training
1 year later. If HOLD-collapse appears here but not in the primary result, it suggests
training history length matters for Q-convergence.

**Pass criterion:** `total_trades > 0` and `return > 0` (same eval regime as primary)  
**Fail criterion:** `total_trades == 0` (HOLD-collapse) or severe degradation (Sharpe < 1.0)

---

### Test 3: `run_test_3_nontech_2018_2024_layernorm_q50.ps1`

| Parameter | Value |
|-----------|-------|
| Tickers | JPM, XOM, UNH, KO, WMT |
| Train | 2018-01-01 → 2023-01-01 |
| Eval | 2023-01-01 → 2024-02-01 |
| Dataset ID | `robustness_nontech_2018_2024_layernorm_q50` |
| Seed | 7 |

**What this tests:** Whether LayerNorm's scale-normalisation generalises to a
different asset class. Non-tech stocks (financials, energy, healthcare, consumer
staples) have lower price levels (JPM ~$150, KO ~$60) and lower volatility than
NVDA or AMZN. The HOLD-collapse mechanism was driven by raw price scale in the
state vector — LayerNorm should handle different scale profiles equally.

**Pass criterion:** `total_trades > 0` (agent engages the portfolio)  
**Expected:** Lower absolute return than demo_5 primary (JPM+XOM+UNH+KO+WMT
underperformed AAPL+MSFT+NVDA+AMZN+GOOGL in 2023), but comparable Sharpe stability.  
**Fail criterion:** `total_trades == 0` (HOLD-collapse returned for different scale assets)

---

## How to Run

Run each script independently from the repository root:

```powershell
# Test 1: earlier window (bear market)
powershell -ExecutionPolicy Bypass -File .\copilot-diagnostics\experiments\layernorm_robustness_tests\run_test_1_demo5_2017_2023_layernorm_q50.ps1

# Test 2: shifted training window
powershell -ExecutionPolicy Bypass -File .\copilot-diagnostics\experiments\layernorm_robustness_tests\run_test_2_demo5_2019_2024_layernorm_q50.ps1

# Test 3: non-tech universe
powershell -ExecutionPolicy Bypass -File .\copilot-diagnostics\experiments\layernorm_robustness_tests\run_test_3_nontech_2018_2024_layernorm_q50.ps1
```

Each script runs seed=7 only (single seed). Runtime: ~5–8 minutes per script.

---

## How to Find Results After Running

Output folders appear under `outputs/runs/` sorted by timestamp. The most recent
run directories for each test will be named after the dataset ID.

**Quick summary — find latest multiseed summary CSV for each test:**

```powershell
# Find all robustness test output dirs
Get-ChildItem outputs\runs\ | Where-Object Name -like "*robustness*" | Sort-Object Name

# Show final records for latest Test 1 run
Get-ChildItem outputs\runs\ | Where-Object Name -like "*robustness_demo5_2017_2023*" |
  Sort-Object Name -Descending | Select-Object -First 1 |
  ForEach-Object { Get-Content "$($_.FullName)\summary\iqn_learning_curve_multiseed_final_records.csv" }

# Show final records for latest Test 3 run (non-tech)
Get-ChildItem outputs\runs\ | Where-Object Name -like "*robustness_nontech*" |
  Sort-Object Name -Descending | Select-Object -First 1 |
  ForEach-Object { Get-Content "$($_.FullName)\summary\iqn_learning_curve_multiseed_final_records.csv" }
```

**Check for HOLD-collapse (total_trades == 0):**

```powershell
Get-ChildItem outputs\runs\ | Where-Object Name -like "*robustness*" |
  Sort-Object Name -Descending | Select-Object -First 3 |
  ForEach-Object {
    $csv = "$($_.FullName)\summary\iqn_learning_curve_multiseed_final_records.csv"
    if (Test-Path $csv) {
      Write-Host "=== $($_.Name) ==="
      Import-Csv $csv | Select-Object seed, total_return_pct, total_trades, sharpe_ratio | Format-Table
    }
  }
```

---

## Important Notes

- These scripts do **not** modify any source files under `src/`
- All scripts run in Mode B (frozen data, no live download)
- The import file `data/market/daily/imports/market_data_full_500.csv` must contain
  the non-tech tickers (JPM, XOM, UNH, KO, WMT) for Test 3 to succeed.
  If not, the runner will fail with a missing tickers error — this is expected and
  means Test 3 requires a separate data import step first.
- Results are logged to W&B project `StockInvestmentDSS` (entity `guldmand-SDU`)
