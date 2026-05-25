# src/stock_investment_dss/uncertainty/recommendation_confidence.py
"""
Recommendation Confidence Layer (v3.1 PoC)

Translates raw Dirichlet evidence quantities from EDLClassifier into:
- recommendation_confidence_label : LOW / MEDIUM / HIGH
- uncertainty_warning              : human-readable warning string
- should_require_human_review      : bool flag for investor-facing UI

Design
------
Thresholds for v3.1 are placeholder values. Proper calibration requires
a held-out labeled dataset with realised trade outcomes (v4.0).

All outputs are marked edl_model_version='edl_poc_v3_1_placeholder_rule_based'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from stock_investment_dss.uncertainty.edl_classifier import (
    DirichletResult,
    EDLFeatureVector,
)

# ---------------------------------------------------------------------------
# Thresholds (v3.1 placeholder — recalibrate in v4.0)
# ---------------------------------------------------------------------------
_HIGH_CONFIDENCE_MIN = 0.55  # prob_high >= this → HIGH label
_LOW_CONFIDENCE_MAX = 0.38  # prob_high <  this → LOW label
_HIGH_VACUITY_THRESH = 0.55  # vacuity >= this   → human review
_SCORE_VARIANCE_WARN = 0.12  # variance → contradiction warning
_DRAWDOWN_WARN = 0.75  # drawdown_norm < this → significant drawdown


@dataclass
class ConfidenceResult:
    """
    Full confidence assessment for a single decision.
    """

    recommendation_confidence_label: str  # LOW / MEDIUM / HIGH
    confidence_score: float
    uncertainty_score: float
    evidence_total: float
    evidence_for_recommendation: float
    evidence_against_recommendation: float
    uncertainty_warning: str
    should_require_human_review: bool
    human_review_reasons: List[str]
    edl_model_version: str = "edl_poc_v3_1_placeholder_rule_based"
    label_strategy: str = "placeholder_rule_based"
    source: str = "edl_poc_placeholder"
    iqn_features_available: bool = False

    def to_dict(self) -> dict:
        return {
            "recommendation_confidence_label": self.recommendation_confidence_label,
            "confidence_score": self.confidence_score,
            "uncertainty_score": self.uncertainty_score,
            "evidence_total": self.evidence_total,
            "evidence_for_recommendation": self.evidence_for_recommendation,
            "evidence_against_recommendation": self.evidence_against_recommendation,
            "uncertainty_warning": self.uncertainty_warning,
            "should_require_human_review": self.should_require_human_review,
            "edl_model_version": self.edl_model_version,
            "label_strategy": self.label_strategy,
            "source": self.source,
            "iqn_features_available": self.iqn_features_available,
        }


class RecommendationConfidenceEvaluator:
    """
    Evaluates Dirichlet outputs and feature context to produce a
    human-interpretable ConfidenceResult.

    Parameters
    ----------
    high_threshold : float
        Minimum prob_high for a HIGH label.
    low_threshold : float
        Maximum prob_high for a LOW label.
    vacuity_review_threshold : float
        Vacuity above which human review is required.
    score_variance_warn : float
        Score variance above which a contradiction warning is issued.
    drawdown_warn_norm : float
        Normalised drawdown below which a drawdown warning is issued.
    """

    def __init__(
        self,
        high_threshold: float = _HIGH_CONFIDENCE_MIN,
        low_threshold: float = _LOW_CONFIDENCE_MAX,
        vacuity_review_threshold: float = _HIGH_VACUITY_THRESH,
        score_variance_warn: float = _SCORE_VARIANCE_WARN,
        drawdown_warn_norm: float = _DRAWDOWN_WARN,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.vacuity_review_threshold = vacuity_review_threshold
        self.score_variance_warn = score_variance_warn
        self.drawdown_warn_norm = drawdown_warn_norm

    def evaluate(
        self,
        dirichlet: DirichletResult,
        feature_vector: EDLFeatureVector,
    ) -> ConfidenceResult:
        """
        Produce a ConfidenceResult from Dirichlet quantities and the
        original feature vector.
        """
        label = self._compute_label(dirichlet)
        warnings, review_reasons = self._compute_warnings(
            dirichlet, feature_vector, label
        )
        should_review = len(review_reasons) > 0
        warning_text = " | ".join(warnings) if warnings else ""

        return ConfidenceResult(
            recommendation_confidence_label=label,
            confidence_score=dirichlet.confidence_score,
            uncertainty_score=dirichlet.uncertainty_score,
            evidence_total=dirichlet.evidence_total,
            evidence_for_recommendation=dirichlet.evidence_for,
            evidence_against_recommendation=dirichlet.evidence_against,
            uncertainty_warning=warning_text,
            should_require_human_review=should_review,
            human_review_reasons=review_reasons,
            iqn_features_available=feature_vector.iqn_features_available,
        )

    # ------------------------------------------------------------------
    # Label logic
    # ------------------------------------------------------------------

    def _compute_label(self, d: DirichletResult) -> str:
        if d.prob_high >= self.high_threshold:
            return "HIGH"
        if d.prob_high < self.low_threshold:
            return "LOW"
        return "MEDIUM"

    # ------------------------------------------------------------------
    # Warning and review-flag logic
    # ------------------------------------------------------------------

    def _compute_warnings(
        self,
        d: DirichletResult,
        fv: EDLFeatureVector,
        label: str,
    ) -> tuple[List[str], List[str]]:
        """
        Returns (warnings list, human_review_reasons list).
        Warnings are human-readable explanations.
        Review reasons are short machine-readable tags.
        """
        warnings: List[str] = []
        reasons: List[str] = []

        # 1. High vacuity
        if d.vacuity >= self.vacuity_review_threshold:
            warnings.append(
                "High epistemic uncertainty: model evidence is insufficient "
                "to strongly support this recommendation."
            )
            reasons.append("high_vacuity")

        # 2. LOW label
        if label == "LOW":
            warnings.append(
                "Recommendation confidence is LOW: consider deferring "
                "this decision or seeking additional information."
            )
            reasons.append("low_confidence_label")

        # 3. Score contradiction (high variance across component scores)
        if (
            fv.score_variance > self.score_variance_warn
            and fv.action_type.upper() != "HOLD"
        ):
            warnings.append(
                "Score contradiction detected: fundamental and technical signals "
                "are conflicting. Investigate individual component scores."
            )
            reasons.append("score_contradiction")

        # 4. Bear-market + BUY
        if fv.price_vs_ma200_norm < 0.45 and fv.action_type.upper() == "BUY":
            pct_below = round((0.5 - fv.price_vs_ma200_norm) * 100, 1)
            warnings.append(
                f"Bear-market guard: selected ticker is ~{pct_below}% below MA200 "
                f"(price_vs_ma200_norm={fv.price_vs_ma200_norm:.3f}). "
                "BUY recommendation has elevated uncertainty."
            )
            reasons.append("bear_market_buy")

        # 5. Significant drawdown
        if (
            fv.drawdown_norm < self.drawdown_warn_norm
            and fv.action_type.upper() != "HOLD"
        ):
            drawdown_pct = round((1.0 - fv.drawdown_norm) * 100, 1)
            warnings.append(
                f"Significant drawdown detected: ticker is ~{drawdown_pct}% "
                "below recent high. Verify risk tolerance before acting."
            )
            reasons.append("significant_drawdown")

        # 6. IQN not connected (for non-HOLD actions)
        if not fv.iqn_features_available and fv.action_type.upper() != "HOLD":
            warnings.append(
                "IQN distribution features unavailable: confidence estimate is "
                "based on rule-based proxy only. Connect IQN to improve estimates."
            )
            reasons.append("iqn_not_connected")

        return warnings, reasons
