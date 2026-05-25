# src/stock_investment_dss/uncertainty/edl_classifier.py
"""
EDL-Inspired Evidence Classifier (v3.1 PoC)

This module implements the Dirichlet evidence accumulation layer of the
D-IQN-DSS uncertainty framework, inspired by Sensoy et al. (NeurIPS 2018)
"Evidential Deep Learning to Quantify Classification Uncertainty".

Architecture
------------
In a full EDL model (v4.0), a neural network outputs non-negative evidence
values e_i for each class, from which Dirichlet concentration parameters
α_i = e_i + 1 are derived. In v3.1, evidence values are computed from
deterministic rules applied to the input feature vector — no training.

Dirichlet quantities
--------------------
    α_i = e_i + 1                          (concentration parameter per class)
    S   = Σ α_i                            (Dirichlet strength / total evidence)
    p̂_i = α_i / S                          (expected class probability)
    u   = K / S   where K=3                (vacuity / epistemic uncertainty)

Output classes: K=3 — LOW / MEDIUM / HIGH confidence.

All outputs are marked source='edl_poc_placeholder'.
Real EDL requires: labeled outcomes, Dirichlet UCE loss, KL regulariser,
calibration evaluation. See docs/EDL_Uncertainty_PoC_v3_1.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Number of confidence classes (LOW / MEDIUM / HIGH)
# ---------------------------------------------------------------------------
_K = 3


@dataclass
class EDLFeatureVector:
    """
    Normalised [0, 1] input features for the EDL classifier.
    Missing/unavailable features should be set to 0.5 (neutral).
    """

    # Group 1 — Action / decision
    action_score_margin: float = 0.5  # margin between top and 2nd action score

    # Group 2 — Ticker scores (all [0, 1])
    final_ticker_score: float = 0.5
    score_variance: float = 0.0  # variance across 5 component scores (pre-normalised)
    value_score: float = 0.5
    quality_score: float = 0.5
    profitability_score: float = 0.5
    momentum_score: float = (
        0.5  # NB: raw momentum may be in [-1,1]; caller clips to [0,1]
    )
    risk_fit_score: float = 0.5

    # Group 3 — IQN distribution (optional; 0.0 = absent / neutral)
    q50: float = 0.0
    q_spread: float = 0.0  # q90 - q10; 0 = absent
    cvar: float = 0.0

    # Group 4 — Size / allocation
    risk_adj_fraction: float = 0.5  # risk_adjusted_allocation_fraction
    size_reduction_ratio: float = (
        1.0  # final_fraction / initial_bucket_fraction; 1 = no penalty
    )

    # Group 5 — Portfolio / market
    cash_weight: float = 0.8
    max_concentration: float = 0.0
    drawdown_norm: float = 1.0  # 1 + drawdown_from_recent_high, clipped to [0,1]
    price_vs_ma50_norm: float = 0.5  # (price_vs_ma50 + 1) / 2
    price_vs_ma200_norm: float = 0.5  # (price_vs_ma200 + 1) / 2

    # Metadata
    iqn_features_available: bool = False
    action_type: str = "BUY"


@dataclass
class DirichletResult:
    """
    Raw Dirichlet evidence and derived quantities for K=3 classes.
    Classes: index 0=LOW, 1=MEDIUM, 2=HIGH
    """

    evidence_low: float = 0.0
    evidence_medium: float = 0.5
    evidence_high: float = 0.0

    alpha_low: float = 1.0
    alpha_medium: float = 1.5
    alpha_high: float = 1.0

    dirichlet_strength: float = 3.5  # S = sum(alpha)

    prob_low: float = 0.0
    prob_medium: float = 0.0
    prob_high: float = 0.0

    vacuity: float = 1.0  # u = K / S
    confidence_score: float = 0.0  # = prob_high
    uncertainty_score: float = 1.0  # = vacuity

    evidence_total: float = 0.0  # S - K (net evidence above uniform prior)
    evidence_for: float = 0.0  # evidence_high (positive evidence)
    evidence_against: float = 0.0  # evidence_low (negative evidence)

    features_used: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Weights for each feature group's contribution to evidence_high
# (sum to 1.0 across groups; tunable in v4.0 via ablation)
# ---------------------------------------------------------------------------
_W_ACTION = 0.15  # Group 1: action score clarity
_W_TICKER = 0.40  # Group 2: ticker score quality and alignment
_W_IQN = 0.20  # Group 3: IQN distribution signal
_W_SIZE = 0.10  # Group 4: size allocation confidence
_W_MKTPORTF = 0.15  # Group 5: market / portfolio state


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _norm_momentum(raw_momentum: float) -> float:
    """Map momentum_score from [-1,1] to [0,1]."""
    return _clip((raw_momentum + 1.0) / 2.0)


def _norm_price_vs_trend(raw: float) -> float:
    """Map price_vs_ma50/200 (typically [-0.5, 0.5]) to [0, 1]."""
    return _clip((raw + 0.5) / 1.0)


class EDLClassifier:
    """
    Rule-based Dirichlet evidence classifier (v3.1 PoC placeholder).

    Computes EDL-inspired confidence quantities from a normalised feature
    vector. Uses weighted rule-based evidence accumulation rather than a
    trained neural network.

    Parameters
    ----------
    iqn_weight_boost : float
        Extra weight multiplier for IQN group when IQN features are available.
    score_variance_penalty_scale : float
        How strongly score variance (score contradiction) reduces evidence_high.
    bear_market_penalty : float
        Extra evidence_against added when bear-market indicators are active.
    """

    def __init__(
        self,
        iqn_weight_boost: float = 1.5,
        score_variance_penalty_scale: float = 2.0,
        bear_market_penalty: float = 0.3,
    ) -> None:
        self.iqn_weight_boost = iqn_weight_boost
        self.score_variance_penalty_scale = score_variance_penalty_scale
        self.bear_market_penalty = bear_market_penalty

    def classify(self, fv: EDLFeatureVector) -> DirichletResult:
        """
        Compute Dirichlet evidence and all derived quantities for a single decision.

        Parameters
        ----------
        fv : EDLFeatureVector
            Normalised feature vector for one decision step.

        Returns
        -------
        DirichletResult
        """
        features_used: list[str] = []

        # ----------------------------------------------------------------
        # Group 1 — Action score margin
        # High margin → clearer IQN preference → higher FOR evidence
        # ----------------------------------------------------------------
        g1_for = _W_ACTION * _clip(fv.action_score_margin)
        g1_against = _W_ACTION * (1.0 - _clip(fv.action_score_margin)) * 0.5
        features_used.append("action_score_margin")

        # ----------------------------------------------------------------
        # Group 2 — Ticker score quality and alignment
        # Mean component score → positive evidence
        # Variance across components → negative evidence (contradictions)
        # ----------------------------------------------------------------
        component_mean = _clip(
            (
                fv.value_score
                + fv.quality_score
                + fv.profitability_score
                + _norm_momentum(fv.momentum_score)
                + fv.risk_fit_score
            )
            / 5.0
        )
        variance_penalty = _clip(fv.score_variance * self.score_variance_penalty_scale)

        if fv.action_type.upper() == "HOLD":
            # For HOLD, use final_ticker_score = 0.5 neutral; low contribution
            g2_for = _W_TICKER * 0.5
            g2_against = _W_TICKER * 0.1
        else:
            g2_for = (
                _W_TICKER
                * _clip(fv.final_ticker_score)
                * (1.0 - variance_penalty * 0.5)
            )
            g2_against = _W_TICKER * variance_penalty
        features_used.extend(
            [
                "final_ticker_score",
                "score_variance",
                "value_score",
                "quality_score",
                "profitability_score",
                "momentum_score",
                "risk_fit_score",
            ]
        )

        # ----------------------------------------------------------------
        # Group 3 — IQN distribution (optional)
        # q50 > 0 → positive return expectation → FOR
        # wide q_spread → aleatoric uncertainty → reduces evidence
        # negative CVaR → tail risk → AGAINST
        # ----------------------------------------------------------------
        if fv.iqn_features_available:
            w_iqn = _W_IQN * self.iqn_weight_boost
            q50_signal = _clip((fv.q50 + 0.3) / 0.6)  # normalise [-0.3, 0.3] → [0, 1]
            spread_penalty = _clip(fv.q_spread / 0.6)  # wide spread → more AGAINST
            cvar_penalty = _clip((-fv.cvar) / 0.3) if fv.cvar < 0 else 0.0
            g3_for = w_iqn * q50_signal * (1.0 - spread_penalty * 0.5)
            g3_against = w_iqn * (spread_penalty * 0.3 + cvar_penalty * 0.4)
            features_used.extend(["q50", "q_spread", "cvar"])
        else:
            # IQN absent: small uncertainty penalty for non-HOLD actions
            if fv.action_type.upper() != "HOLD":
                g3_for = 0.0
                g3_against = _W_IQN * 0.3  # slight AGAINST: no IQN confirmation
            else:
                g3_for = _W_IQN * 0.3
                g3_against = 0.0
            features_used.append("iqn_features_available=False")

        # ----------------------------------------------------------------
        # Group 4 — Size reduction ratio
        # Ratio ≈ 1.0 → size not penalised → FOR confidence
        # Ratio << 1 → many risk flags fired → AGAINST confidence
        # ----------------------------------------------------------------
        size_ratio = _clip(fv.size_reduction_ratio)
        g4_for = _W_SIZE * size_ratio
        g4_against = _W_SIZE * (1.0 - size_ratio) * 0.8
        features_used.extend(["risk_adj_fraction", "size_reduction_ratio"])

        # ----------------------------------------------------------------
        # Group 5 — Portfolio / market state
        # Good cash buffer, low concentration, low drawdown, above MA trend → FOR
        # Poor cash, high concentration, high drawdown, below MA200 → AGAINST
        # ----------------------------------------------------------------
        cash_ok = _clip(fv.cash_weight / 0.8)  # 80% cash = full score
        concentration_ok = _clip(
            1.0 - fv.max_concentration / 0.5
        )  # 50% concentration = 0
        drawdown_ok = _clip(fv.drawdown_norm)  # already [0,1]; 1=no drawdown
        ma50_ok = _clip(fv.price_vs_ma50_norm)
        ma200_ok = _clip(fv.price_vs_ma200_norm)

        bear_market = fv.price_vs_ma200_norm < 0.45 and fv.action_type.upper() == "BUY"
        bear_penalty = self.bear_market_penalty if bear_market else 0.0

        g5_for = _W_MKTPORTF * (
            0.25 * cash_ok
            + 0.20 * concentration_ok
            + 0.25 * drawdown_ok
            + 0.15 * ma50_ok
            + 0.15 * ma200_ok
        )
        g5_against = (
            _W_MKTPORTF * bear_penalty + _W_MKTPORTF * (1.0 - drawdown_ok) * 0.3
        )
        features_used.extend(
            [
                "cash_weight",
                "max_concentration",
                "drawdown_norm",
                "price_vs_ma50_norm",
                "price_vs_ma200_norm",
            ]
        )

        # ----------------------------------------------------------------
        # Aggregate evidence
        # ----------------------------------------------------------------
        evidence_high = max(0.0, g1_for + g2_for + g3_for + g4_for + g5_for)
        evidence_low = max(
            0.0, g1_against + g2_against + g3_against + g4_against + g5_against
        )
        evidence_medium = max(0.0, 0.5 - abs(evidence_high - evidence_low) * 0.3)

        # Dirichlet concentration parameters: α_i = e_i + 1
        alpha_high = evidence_high + 1.0
        alpha_medium = evidence_medium + 1.0
        alpha_low = evidence_low + 1.0

        S = alpha_high + alpha_medium + alpha_low

        prob_high = alpha_high / S
        prob_medium = alpha_medium / S
        prob_low = alpha_low / S

        vacuity = _K / S  # u = K / S; higher = more uncertain

        return DirichletResult(
            evidence_low=round(evidence_low, 6),
            evidence_medium=round(evidence_medium, 6),
            evidence_high=round(evidence_high, 6),
            alpha_low=round(alpha_low, 6),
            alpha_medium=round(alpha_medium, 6),
            alpha_high=round(alpha_high, 6),
            dirichlet_strength=round(S, 6),
            prob_low=round(prob_low, 6),
            prob_medium=round(prob_medium, 6),
            prob_high=round(prob_high, 6),
            vacuity=round(vacuity, 6),
            confidence_score=round(prob_high, 6),
            uncertainty_score=round(vacuity, 6),
            evidence_total=round(S - _K, 6),
            evidence_for=round(evidence_high, 6),
            evidence_against=round(evidence_low, 6),
            features_used=features_used,
        )


# ---------------------------------------------------------------------------
# Feature vector builder — constructs EDLFeatureVector from audit DataFrames
# ---------------------------------------------------------------------------


def build_feature_vector(
    decision_row: "pd.Series",
    ticker_rows: "Optional[pd.DataFrame]",
    size_rows: "Optional[pd.DataFrame]",
    action_type: str,
) -> EDLFeatureVector:
    """
    Build an EDLFeatureVector from hierarchical policy audit data for a single decision.

    Parameters
    ----------
    decision_row : pd.Series
        Row from hierarchical_decision_by_step.csv.
    ticker_rows : pd.DataFrame or None
        Rows from ticker_score_table.csv matching this decision_id and selected ticker.
        May be None if HOLD or no ticker selected.
    size_rows : pd.DataFrame or None
        Rows from size_score_table.csv matching this decision_id.
        May be None if HOLD.
    action_type : str
        Selected action type for this decision.
    """

    def _safe_float(value, default: float = 0.5) -> float:
        try:
            v = float(value)
            return v if math.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    # Group 1 — action score margin (not present in current audit schema
    #           — use neutral 0.5 as IQN is not yet connected)
    # ------------------------------------------------------------------
    action_score_margin = _safe_float(
        decision_row.get("stage1_action_score_margin", 0.5), 0.5
    )

    # ------------------------------------------------------------------
    # Group 2 — ticker scores
    # ------------------------------------------------------------------
    selected_ticker = str(decision_row.get("selected_ticker", "") or "")

    if ticker_rows is not None and not ticker_rows.empty and selected_ticker:
        sel = ticker_rows[ticker_rows["ticker"].astype(str) == selected_ticker]
        if sel.empty:
            sel = ticker_rows.head(1)  # fallback to best-ranked row
        row = sel.iloc[0]
        final_ticker_score = _safe_float(row.get("final_ticker_score", 0.5), 0.5)
        value_score = _safe_float(row.get("value_score", 0.5), 0.5)
        quality_score = _safe_float(row.get("quality_score", 0.5), 0.5)
        profitability_score = _safe_float(row.get("profitability_score", 0.5), 0.5)
        momentum_score_raw = _safe_float(row.get("momentum_score", 0.0), 0.0)
        risk_fit_score = _safe_float(row.get("risk_fit_score", 0.5), 0.5)

        # Compute variance across 5 component scores (momentum normalised to [0,1])
        mom_norm = _clip((momentum_score_raw + 1.0) / 2.0)
        scores = [
            value_score,
            quality_score,
            profitability_score,
            mom_norm,
            risk_fit_score,
        ]
        mean_s = sum(scores) / len(scores)
        variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
    else:
        # HOLD or no ticker: neutral values
        final_ticker_score = 0.5
        value_score = quality_score = profitability_score = 0.5
        momentum_score_raw = 0.0
        risk_fit_score = 0.5
        variance = 0.0

    # ------------------------------------------------------------------
    # Group 4 — size / allocation
    # ------------------------------------------------------------------
    risk_adj_fraction = _safe_float(
        decision_row.get("risk_adjusted_allocation_fraction", 0.5), 0.5
    )

    # Determine initial bucket fraction from size label
    size_label = str(decision_row.get("selected_size", "") or "")
    _bucket_map = {
        "BUY_25": 0.25,
        "BUY_50": 0.5,
        "BUY_75": 0.75,
        "BUY_100": 1.0,
        "SELL_25": 0.25,
        "SELL_50": 0.5,
        "SELL_75": 0.75,
        "SELL_100": 1.0,
    }
    bucket_frac = _bucket_map.get(size_label, 0.5)
    if size_rows is not None and not size_rows.empty and size_label:
        sel_size = size_rows[size_rows["selected"].astype(str).str.lower() == "true"]
        if not sel_size.empty:
            bucket_frac = _safe_float(
                sel_size.iloc[0].get("fraction", bucket_frac), bucket_frac
            )

    size_reduction_ratio = (
        _clip(risk_adj_fraction / bucket_frac) if bucket_frac > 0 else 1.0
    )

    # ------------------------------------------------------------------
    # Group 5 — portfolio / market state
    # ------------------------------------------------------------------
    cash_weight = _safe_float(decision_row.get("portfolio_cash_weight", 0.8), 0.8)
    max_concentration = _safe_float(
        decision_row.get("portfolio_max_concentration", 0.0), 0.0
    )

    # Drawdown, MA features — may be present in future enriched audit rows
    drawdown_raw = _safe_float(decision_row.get("drawdown_from_recent_high", 0.0), 0.0)
    drawdown_norm = _clip(1.0 + drawdown_raw)  # e.g. -0.15 → 0.85; clipped to [0,1]
    price_vs_ma50_raw = _safe_float(decision_row.get("price_vs_ma50", 0.0), 0.0)
    price_vs_ma200_raw = _safe_float(decision_row.get("price_vs_ma200", 0.0), 0.0)
    price_vs_ma50_norm = _clip((price_vs_ma50_raw + 0.5) / 1.0)
    price_vs_ma200_norm = _clip((price_vs_ma200_raw + 0.5) / 1.0)

    # Bear-market from risk flags if available
    bear_market_signal = (
        str(decision_row.get("risk_bear_market_signal", "False")).lower() == "true"
    )
    if bear_market_signal:
        price_vs_ma200_norm = min(price_vs_ma200_norm, 0.40)

    return EDLFeatureVector(
        action_score_margin=action_score_margin,
        final_ticker_score=_clip(final_ticker_score),
        score_variance=_clip(variance, 0.0, 0.5),
        value_score=_clip(value_score),
        quality_score=_clip(quality_score),
        profitability_score=_clip(profitability_score),
        momentum_score=momentum_score_raw,
        risk_fit_score=_clip(risk_fit_score),
        q50=0.0,
        q_spread=0.0,
        cvar=0.0,
        risk_adj_fraction=_clip(risk_adj_fraction),
        size_reduction_ratio=_clip(size_reduction_ratio),
        cash_weight=_clip(cash_weight),
        max_concentration=_clip(max_concentration),
        drawdown_norm=drawdown_norm,
        price_vs_ma50_norm=price_vs_ma50_norm,
        price_vs_ma200_norm=price_vs_ma200_norm,
        iqn_features_available=False,
        action_type=action_type,
    )
