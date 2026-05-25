# EDL Uncertainty PoC — Thesis Alignment

**Module:** `stockdss_patch_v3_1_edl_uncertainty_poc`
**Status:** Design phase — plan only

---

## 1. Connection to the NeurIPS 2018 EDL Paper

Sensoy et al. (NeurIPS 2018) — *"Evidential Deep Learning to Quantify Classification Uncertainty"* — introduces the key framework that this PoC is inspired by.

### Core Paper Concepts and Their Use in D-IQN-DSS

| Paper concept | Formal definition | Use in D-IQN-DSS |
|---------------|-------------------|-----------------|
| Subjective Logic | Opinion = (belief, disbelief, uncertainty) | Maps to (HIGH, LOW, MEDIUM) confidence classes |
| Dirichlet distribution | Dir(p \| α) parameterises P over class simplex | Parameterises recommendation confidence distribution |
| Evidence e_i | Non-negative network output; e_i = ReLU(output_i) | In v3.1: deterministic rule-based; in v4.0: neural network output |
| Alpha parameters | α_i = e_i + 1 | Used directly in confidence label computation |
| Dirichlet strength S | S = Σ α_i | Denominator for expected probabilities and vacuity |
| Vacuity u | u = K / S | Primary epistemic uncertainty signal |
| UCE loss | L = Σ [−log p̂_i] Dirichlet-integrated expected cross-entropy | For v4.0: training loss for the EDL classification head |
| KL divergence regulariser | Penalises non-uniform Dirichlet for wrong-class predictions | For v4.0: prevents overconfident wrong predictions |

### What v3.1 Borrows from the Paper

v3.1 borrows the **mathematics** (Dirichlet parameterisation, vacuity formula) but not the **learning procedure** (the neural network and UCE loss). This is explicitly a "EDL-inspired" PoC, not a full EDL implementation.

The vacuity `u = K / S` is the key quantity: even without training, computing `u` from rule-based `α` values provides a structured, interpretable uncertainty estimate that is theoretically grounded in the paper's framework.

---

## 2. The IQN + EDL Duality

The thesis methodology section names both IQN and EDL as components of the system. This PoC makes their roles explicit:

```
┌────────────────────────────────────────────────────────────────┐
│                   Uncertainty in D-IQN-DSS                    │
│                                                                │
│  Aleatoric uncertainty (IQN)                                  │
│  ─────────────────────────                                     │
│  Source: stochastic returns / market volatility               │
│  Model: full return distribution P(G | state, action)         │
│  Representation: q10, q50, q90, CVaR                          │
│  → "The market is volatile / has fat tails"                   │
│                                                                │
│  Epistemic uncertainty (EDL)                                  │
│  ────────────────────────────                                  │
│  Source: model confidence / out-of-distribution inputs        │
│  Model: Dirichlet over recommendation confidence classes       │
│  Representation: vacuity u, confidence_score, label           │
│  → "The model is uncertain about THIS recommendation"         │
│                                                                │
│  Combined signal to investor:                                  │
│  "The expected return is volatile [IQN],                      │
│   and the model has limited confidence in its own estimate    │
│   [EDL]. Human review is recommended."                        │
└────────────────────────────────────────────────────────────────┘
```

This is a direct implementation of the thesis problem formulation's requirement to model **both** stochastic return variability and model uncertainty.

---

## 3. Fit With Thesis Problem Formulation

> *"The thesis investigates uncertainty-aware modeling by incorporating an epistemic uncertainty component inspired by Evidential Deep Learning. This allows the system to account not only for stochastic return variability but also for uncertainty in the model's own estimates, which is particularly relevant in financial environments characterized by limited data and regime shifts."*

The EDL uncertainty PoC directly operationalises this statement:

| Thesis requirement | EDL PoC component |
|--------------------|-------------------|
| Epistemic uncertainty estimate | `vacuity = K / S` |
| Model-confidence signal | `confidence_score = prob_high` |
| Relevant for limited data / regime shifts | `should_require_human_review` flag triggered in bear-market, OOD scenarios |
| Transparent to investor | `recommendation_confidence_label` + `uncertainty_warning` in plain language |
| Decision support (not autonomous) | Human review flag — system recommends, investor decides |
| Auditable | Full `edl_uncertainty_by_decision.csv` with all intermediate Dirichlet quantities |

---

## 4. How EDL Couples to Hierarchical Decision Policy (v4.0)

In v3.1, the EDL layer is **downstream** of the hierarchical policy (reads from its output). In v4.0, it will be integrated **inline**:

### v4.0 Integration Points

**A. Stage 3 — Size reduction via EDL uncertainty**

```python
# In size_selector.py (v4.0):
edl_confidence = edl_classifier.get_confidence(features)
if edl_confidence.uncertainty_score > 0.55:
    size_penalty *= 0.5   # Halve position size when model is uncertain
```

**B. Stage 4 — Risk validator: EDL-gated HOLD**

```python
# In hierarchical_decision_policy.py (v4.0):
if edl_result.should_require_human_review and strategy.id == "defensive_v1":
    # Force HOLD and log EDL reason
    return HierarchicalDecision(action=HOLD, reason="EDL_EPISTEMIC_UNCERTAINTY")
```

**C. Stage 5 — Audit ledger: EDL fields added**

The audit schema already includes placeholder columns for EDL in v3.0. In v4.0, these are populated by the live EDL layer:

```
recommendation_confidence_label
confidence_score
uncertainty_score
vacuity
should_require_human_review
edl_uncertainty_warning
```

---

## 5. W&B Integration Plan (v4.0)

When EDL is trained and connected to the live IQN run, the following metrics will be logged to Weights & Biases alongside IQN metrics:

| W&B metric | Description |
|------------|-------------|
| `edl/mean_vacuity` | Mean epistemic uncertainty across backtest window |
| `edl/mean_confidence` | Mean recommendation confidence |
| `edl/high_review_fraction` | Fraction of decisions requiring human review |
| `edl/calibration_ece` | Expected Calibration Error (v4.0 when labels available) |
| `edl/vacuity_vs_return` | Scatter plot: vacuity vs realised step return |
| `edl/confidence_vs_return` | Scatter plot: confidence vs realised step return |

These can be added to the existing run's W&B experiment under a new `edl/` metric namespace — no changes to IQN logging code.

---

## 6. Thesis Figure Suggestions

The following figures can be generated from EDL audit output for the thesis:

### Figure 1: Vacuity Over Time
Plot `uncertainty_score` (vacuity) across backtest dates, with action type annotations. Expected to show:
- Low vacuity in stable bull periods
- High vacuity around regime shifts (if bear-market scenario is included)
- High vacuity when IQN distribution is wide

### Figure 2: Confidence Distribution by Action Type
Bar chart: distribution of HIGH/MEDIUM/LOW confidence labels split by HOLD/BUY/SELL.
Expected: BUY decisions have higher average confidence than HOLD in bull periods; bear-market BUY decisions have lower confidence.

### Figure 3: IQN Spread vs EDL Vacuity (v4.0)
Scatter plot: `q90 - q10` (IQN aleatoric spread) vs `vacuity` (EDL epistemic uncertainty).
Hypothesis: low correlation → the two uncertainty types are capturing complementary signals, validating the two-uncertainty framework.

### Figure 4: Calibration Curve (v4.0)
Plot: predicted confidence vs actual accuracy on held-out labels.
Properly calibrated EDL should lie close to the diagonal. Deviation reveals overconfidence or underconfidence.

---

## 7. Limitations and Honest Scope

The thesis must present these limitations honestly:

| Limitation | Honest statement |
|------------|-----------------|
| v3.1 labels are deterministic placeholders | "EDL classification in v3.1 uses rule-based evidence accumulation, not a trained neural network. Results are illustrative of the framework, not calibrated uncertainty estimates." |
| No calibration evaluation | "Calibration of the EDL classifier is not evaluated in this PoC. Proper calibration requires a held-out labeled dataset with realised trade outcomes." |
| IQN not connected in smoke test | "The IQN distributional features (q10/q50/q90/CVaR) are not available in the v3.1 smoke test. Their inclusion in v4.0 is expected to improve confidence estimates." |
| Frozen fundamentals affect EDL input | "Fundamental score features used by the EDL classifier are derived from placeholder fundamentals in v3.0–v3.1. The quality of EDL estimates will improve when real FMP data is integrated." |
| No hindsight oracle comparison | "The gap between v3.1 (rule-based) and true EDL (trained on realised outcomes) cannot be quantified without a full backtesting evaluation loop — this is identified as future work." |

---

## 8. Version Roadmap

| Version | EDL Status |
|---------|-----------|
| v3.0 | No EDL — hierarchical policy PoC only |
| v3.1 | EDL-inspired rule-based uncertainty layer, standalone, no training |
| v4.0 | Trained EDL classification head; Dirichlet UCE loss; integrated into hierarchical policy Stage 3/4 |
| v4.1 | EDL + IQN joint evaluation; calibration curves; W&B logging; thesis figures |
