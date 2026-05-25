# EDL v3.2 — Ablation Switching Plan

**Status:** Design only — no source code changed  
**Purpose:** Define all configuration flags, ablation matrix, and evaluation protocol  

---

## 1. Environment Variable Reference

All flags use the `STOCK_INVESTMENT_DSS_` prefix for namespace isolation.

### Core system flags

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `STOCK_INVESTMENT_DSS_USE_HIERARCHICAL_POLICY` | `true` / `false` | `true` | Enable/disable hierarchical decision policy (Stage 1-5) |
| `STOCK_INVESTMENT_DSS_USE_EDL` | `true` / `false` | `false` | Enable/disable EDL uncertainty layer |

### EDL configuration flags

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `STOCK_INVESTMENT_DSS_EDL_VARIANT` | `none` / `A` / `B` / `C` / `AB` / `AC` / `BC` / `ABC` | `none` | Which EDL model(s) to use |
| `STOCK_INVESTMENT_DSS_EDL_GATE_ENABLED` | `true` / `false` | `true` | Enable EDL gate (size reduction / force HOLD / human review) |
| `STOCK_INVESTMENT_DSS_EDL_INCLUDE_CHANGE_STRATEGY` | `true` / `false` | `false` | K=5 (CHANGE_STRATEGY class) vs K=4 |
| `STOCK_INVESTMENT_DSS_EDL_UNCERTAINTY_LAMBDA` | float ≥ 0.0 | `0.5` | λ_u: uncertainty penalty weight in score function |
| `STOCK_INVESTMENT_DSS_EDL_DISAGREEMENT_LAMBDA` | float ≥ 0.0 | `0.3` | λ_d: A/B/C model disagreement penalty weight |
| `STOCK_INVESTMENT_DSS_EDL_HORIZON_DAYS` | int > 0 | `20` | Look-ahead horizon for EDL-A oracle labels |
| `STOCK_INVESTMENT_DSS_EDL_LABEL_MODE` | `hindsight` / `rules` / `iqn_teacher` | `iqn_teacher` | Label source for dataset construction |
| `STOCK_INVESTMENT_DSS_EDL_MODEL_PATH` | file path | `""` | Path to trained `.pt` model checkpoint |

### Existing hierarchical policy flags (unchanged)

| Variable | Default |
|----------|---------|
| `STOCK_INVESTMENT_DSS_HIERARCHICAL_ACTION_TYPE` | `BUY` |
| `STOCK_INVESTMENT_DSS_HIERARCHICAL_CASH_WEIGHT` | `0.80` |
| `STOCK_INVESTMENT_DSS_HIERARCHICAL_TICKERS` | `AAPL,MSFT,NVDA,AMZN,GOOGL` |
| `STOCK_INVESTMENT_DSS_HIERARCHICAL_STRATEGY` | `balanced_v1` |
| `STOCK_INVESTMENT_DSS_HIERARCHICAL_DECISION_DATE` | latest in market data |
| `STOCK_INVESTMENT_DSS_HIERARCHICAL_STEPS` | `5` |

---

## 2. Ablation Matrix

The full evaluation protocol requires running 9 ablation conditions per strategy/risk profile combination.

### Primary ablation matrix

| Condition | `USE_HIERARCHICAL_POLICY` | `USE_EDL` | `EDL_VARIANT` | `EDL_GATE_ENABLED` | Notes |
|-----------|--------------------------|-----------|---------------|-------------------|-------|
| **Baseline** | `false` | `false` | — | — | IQN only, no hierarchical |
| **H-only** | `true` | `false` | — | — | Hierarchical + no EDL |
| **H+EDL-A** | `true` | `true` | `A` | `true` | Oracle labels |
| **H+EDL-B** | `true` | `true` | `B` | `true` | Rule labels |
| **H+EDL-C** | `true` | `true` | `C` | `true` | IQN teacher labels |
| **H+EDL-AB** | `true` | `true` | `AB` | `true` | A+B ensemble |
| **H+EDL-AC** | `true` | `true` | `AC` | `true` | A+C ensemble |
| **H+EDL-BC** | `true` | `true` | `BC` | `true` | B+C ensemble |
| **H+EDL-ABC** | `true` | `true` | `ABC` | `true` | Full ensemble |

### Secondary ablation: gate disabled

Run H+EDL-C also with `EDL_GATE_ENABLED=false` to isolate the contribution of the gate vs. the uncertainty-penalised score function.

### Secondary ablation: lambda sweep

For the best-performing EDL variant, sweep:
```
λ_u ∈ {0.0, 0.1, 0.25, 0.5, 1.0, 2.0}
λ_d ∈ {0.0, 0.1, 0.3, 0.5}
```

### Secondary ablation: horizon (EDL-A only)

```
HORIZON_DAYS ∈ {5, 10, 20, 60}
```

### Secondary ablation: class space

```
EDL_INCLUDE_CHANGE_STRATEGY = false (K=4)   [default]
EDL_INCLUDE_CHANGE_STRATEGY = true  (K=5)   [extended]
```

---

## 3. Evaluation Metrics

For each ablation condition, compute on **held-out test set (temporal split)**:

### Portfolio performance metrics

| Metric | Description |
|--------|-------------|
| `total_return` | Cumulative portfolio return over evaluation period |
| `annualized_return` | Annualised return |
| `sharpe_ratio` | Return / volatility (annualised) |
| `sortino_ratio` | Return / downside deviation |
| `max_drawdown` | Maximum peak-to-trough drawdown |
| `calmar_ratio` | annualized_return / abs(max_drawdown) |
| `cvar_5pct` | 5th percentile of daily return distribution |
| `win_rate` | Fraction of profitable decision steps |

### EDL-specific metrics

| Metric | Description |
|--------|-------------|
| `mean_vacuity` | Mean epistemic uncertainty over all decisions |
| `mean_confidence` | Mean selected_action_probability |
| `human_review_fraction` | Fraction of decisions flagged for review |
| `edl_agreement_rate` | Fraction where EDL agrees with selected action |
| `force_hold_fraction` | Fraction where gate issued FORCE_HOLD |
| `size_reduction_mean` | Mean size reduction ratio when REDUCE_SIZE fires |
| `ece` | Expected Calibration Error (v4.0: after calibration) |
| `brier_score` | Probabilistic accuracy of action predictions |

### Decision quality metrics

| Metric | Description |
|--------|-------------|
| `hold_rate` | Fraction of HOLD decisions |
| `buy_rate` | Fraction of BUY decisions |
| `sell_rate` | Fraction of SELL decisions |
| `rebalance_rate` | Fraction of REBALANCE decisions |
| `action_diversity` | Entropy of action distribution |

---

## 4. Ablation Run Protocol

### Step 1: Build labeled datasets (Phase 3)

```powershell
# EDL-B (rules-based labels, no future data needed)
$env:STOCK_INVESTMENT_DSS_EDL_LABEL_MODE = "rules"
python -m stock_investment_dss.runner.run_edl_action_dataset_builder

# EDL-C (IQN teacher labels, from existing hierarchical runs)
$env:STOCK_INVESTMENT_DSS_EDL_LABEL_MODE = "iqn_teacher"
python -m stock_investment_dss.runner.run_edl_action_dataset_builder

# EDL-A (hindsight oracle labels — requires market simulation)
$env:STOCK_INVESTMENT_DSS_EDL_LABEL_MODE = "hindsight"
$env:STOCK_INVESTMENT_DSS_EDL_HORIZON_DAYS = "20"
python -m stock_investment_dss.runner.run_edl_action_dataset_builder
```

### Step 2: Training smoke tests (Phase 3, small scale)

```powershell
# Train EDL-B (fastest — no future data)
$env:STOCK_INVESTMENT_DSS_EDL_VARIANT = "B"
python -m stock_investment_dss.runner.run_edl_action_training_smoke_test

# Train EDL-C
$env:STOCK_INVESTMENT_DSS_EDL_VARIANT = "C"
python -m stock_investment_dss.runner.run_edl_action_training_smoke_test
```

### Step 3: Inference smoke tests

```powershell
# Baseline: H-only
$env:STOCK_INVESTMENT_DSS_USE_EDL = "false"
python -m stock_investment_dss.runner.run_edl_action_inference_smoke_test

# H+EDL-C
$env:STOCK_INVESTMENT_DSS_USE_EDL = "true"
$env:STOCK_INVESTMENT_DSS_EDL_VARIANT = "C"
python -m stock_investment_dss.runner.run_edl_action_inference_smoke_test
```

### Step 4: Full ablation (Phase 3+, requires W&B)

```powershell
foreach ($variant in @("none","A","B","C","AB","AC","BC","ABC")) {
    $env:STOCK_INVESTMENT_DSS_EDL_VARIANT = $variant
    python -m stock_investment_dss.runner.run_edl_action_inference_smoke_test
}
```

---

## 5. W&B Integration Plan

For each ablation run, log to Weights & Biases under the project `d_iqn_dss_edl_ablation`:

```python
wandb.init(
    project="d_iqn_dss_edl_ablation",
    config={
        "use_hierarchical_policy": ...,
        "use_edl": ...,
        "edl_variant": ...,
        "edl_gate_enabled": ...,
        "edl_uncertainty_lambda": ...,
        "edl_disagreement_lambda": ...,
        "edl_horizon_days": ...,
        "edl_include_change_strategy": ...,
    },
)
```

Log per-step metrics:
```python
wandb.log({
    "edl/vacuity": u,
    "edl/confidence": p_selected,
    "edl/agreement": edl_agrees,
    "edl/gate": recommendation_gate,
    "portfolio/return": step_return,
    "portfolio/drawdown": current_drawdown,
})
```

Log per-run summary metrics:
```python
wandb.summary.update({
    "total_return": ...,
    "sharpe_ratio": ...,
    "max_drawdown": ...,
    "human_review_fraction": ...,
    "mean_vacuity": ...,
    "edl_agreement_rate": ...,
})
```

---

## 6. Thesis Ablation Table Template

The ablation produces a comparison table for the thesis:

| Condition | Return | Sharpe | Max DD | CVaR | Review% | Vacuity | EDL Agree% |
|-----------|--------|--------|--------|------|---------|---------|------------|
| Baseline (no H, no EDL) | | | | | — | — | — |
| H-only | | | | | — | — | — |
| H+EDL-A | | | | | | | |
| H+EDL-B | | | | | | | |
| H+EDL-C | | | | | | | |
| H+EDL-AB | | | | | | | |
| H+EDL-AC | | | | | | | |
| H+EDL-BC | | | | | | | |
| H+EDL-ABC | | | | | | | |

**Primary hypothesis:** H+EDL-C (or H+EDL-ABC) outperforms H-only on Sharpe ratio and CVaR, at the cost of increased human review rate, which is acceptable for a decision support system.
