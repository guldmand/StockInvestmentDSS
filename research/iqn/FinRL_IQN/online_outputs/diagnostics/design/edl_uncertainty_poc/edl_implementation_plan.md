# EDL Uncertainty PoC — Implementation Plan

**Module:** `stockdss_patch_v3_1_edl_uncertainty_poc`
**Version:** v3.1 (planned — not yet implemented)
**Date:** 2026-05-21
**Status:** Design phase — Markdown plan only. No source code yet.

---

## 1. Motivation

### The Two-Uncertainty Problem in D-IQN-DSS

The D-IQN-DSS system currently models **aleatoric uncertainty** (stochastic return variability) through IQN's distributional RL framework. IQN outputs a full return distribution, from which CVaR-based risk-sensitive objectives are derived. This is the *market uncertainty* component.

However, a complete uncertainty-aware decision support system also needs to model **epistemic uncertainty** — uncertainty about the model's own confidence in its estimates. This is the *model uncertainty* component.

**Why the distinction matters for an investor:**
- A recommendation with high aleatoric uncertainty (volatile stock) but low epistemic uncertainty (model is confident in its estimate) is qualitatively different from a recommendation where the model itself is uncertain.
- Investors operating in unfamiliar market regimes, low-data situations, or near fundamental data gaps need to know: "Is the model confident in this recommendation, or is it operating outside its training distribution?"

### The EDL Framework (Sensoy et al., NeurIPS 2018)

Evidential Deep Learning (EDL) addresses epistemic uncertainty in classification by replacing standard softmax outputs with **evidence-based Dirichlet distributions**. Instead of outputting class probabilities directly, the model outputs evidence `e_i ≥ 0` for each class, from which Dirichlet concentration parameters are derived:

```
α_i = e_i + 1
S = Σ α_i   (Dirichlet strength — total evidence)
p̂_i = α_i / S   (expected class probability)
u = K / S        (epistemic uncertainty / vacuity, K = number of classes)
```

High vacuity `u` indicates that the model has little evidence to support any class — i.e., high epistemic uncertainty. This maps directly to "should require human review" in a DSS context.

### How IQN + EDL Combine in D-IQN-DSS

```
IQN distributional RL
  └─ Models: P(return | state, action)
  └─ Captures: aleatoric / market uncertainty
  └─ Output: q10, q50, q90, CVaR

EDL uncertainty classifier
  └─ Models: P(recommendation_class | features)
  └─ Captures: epistemic / model-confidence uncertainty
  └─ Output: confidence_score, vacuity, recommendation_confidence_label
```

The combination provides investors with two distinct uncertainty signals:
1. "How volatile is the expected return?" (IQN)
2. "How confident is the model in this recommendation?" (EDL)

---

## 2. Scope of v3.1 PoC

### What v3.1 IS

- A **standalone EDL-inspired uncertainty layer** that reads from existing hierarchical policy audit output
- Uses **deterministic placeholder labels** (no training, no neural network) to simulate EDL-style evidence accumulation
- Produces human-readable `recommendation_confidence_label` (LOW / MEDIUM / HIGH) and `should_require_human_review`
- Writes full audit trail to `audit/edl_uncertainty_by_decision.csv`
- Architecturally designed to be replaced by a trained EDL classifier in v4.0

### What v3.1 IS NOT

- A trained neural network (no backpropagation, no weight learning)
- A calibrated probabilistic model
- A replacement for full EDL with proper Dirichlet training losses
- A production-ready uncertainty quantification system
- Research-grade EDL (that requires proper labeling, calibration, and evaluation)

All outputs are marked `source=edl_poc_placeholder` in audit records.

---

## 3. New Files Proposed (not yet implemented)

| File | Purpose |
|------|---------|
| `src/stock_investment_dss/uncertainty/__init__.py` | Package init |
| `src/stock_investment_dss/uncertainty/edl_classifier.py` | EDL-inspired evidence accumulation and Dirichlet approximation |
| `src/stock_investment_dss/uncertainty/recommendation_confidence.py` | Confidence label logic, human-review flag, warning generation |
| `src/stock_investment_dss/runner/run_edl_uncertainty_smoke_test.py` | Standalone runner reading from hierarchical audit output |
| `docs/EDL_Uncertainty_PoC_v3_1.md` | Full design and thesis documentation |

**Constraint: no existing src files will be modified.**
The EDL layer reads from already-written audit CSVs and writes its own output — it does not hook into `hierarchical_decision_policy.py` until v4.0.

---

## 4. Label Strategy for v3.1

Four label strategies were considered:

| Strategy | Description | Usable in v3.1? |
|----------|-------------|-----------------|
| A. Hindsight oracle | Label = 1 if trade was profitable ex-post | ❌ Requires backtesting results |
| B. Rule/baseline-policy | Label = 1 if action matches simple rule (e.g., buy above MA50) | ✅ Possible but weak signal |
| C. IQN/risk-policy teacher | Label = softmax argmax from trained IQN | ❌ Requires IQN connected and trained |
| D. Placeholder deterministic | Label derived from composite feature rules (high confidence if scores align) | ✅ Used in v3.1 |

**v3.1 uses Strategy D.** Confidence is computed as a weighted evidence accumulation from the input features (see `edl_feature_plan.md`). This is **NOT** real EDL training — it is a rule-based approximation that uses EDL mathematics to structure the output.

All labels are marked `label_strategy=placeholder_rule_based` in outputs.

---

## 5. Integration Roadmap

### v3.1 (this PoC)
- Standalone layer reading from hierarchical audit CSV
- Deterministic placeholder evidence
- Full audit schema (see `edl_audit_schema.md`)
- Smoke test confirmed working

### v4.0
- Train a proper EDL classification head
- Attach to `hierarchical_decision_policy.py` as a post-decision confidence layer
- Labels from hindsight oracle or IQN teacher
- Dirichlet loss function (UCE + KL regularizer from NeurIPS 2018)
- EDL uncertainty propagated back to Stage 3 size reduction in hierarchical policy

### v4.1
- Log EDL uncertainty metrics to W&B alongside IQN metrics
- Thesis figures: plot `vacuity` vs `return` over backtest window; show calibration curve
- EDL-gated human review trigger in live DSS UI

---

## 6. Constraints (confirmed)

- ❌ No modification to IQN kernel
- ❌ No modification to hierarchical policy source (`hierarchical_decision_policy.py` etc.)
- ❌ No training runs
- ❌ No package installs
- ❌ No source code in `copilot-diagnostics/`
- ✅ New files only in `src/stock_investment_dss/uncertainty/`
- ✅ Documentation in `docs/`
- ✅ Design plans in `copilot-diagnostics/design/edl_uncertainty_poc/`
