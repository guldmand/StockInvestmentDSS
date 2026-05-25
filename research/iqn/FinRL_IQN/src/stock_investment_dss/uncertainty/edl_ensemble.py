# src/stock_investment_dss/uncertainty/edl_ensemble.py
"""
EDL Action Ensemble (v3.2)

Combines predictions from multiple EDL classifiers (A, B, C variants)
via weighted probability averaging and computes:
- p_ensemble  : weighted average action probabilities
- u_ensemble  : weighted average vacuity
- u_conservative : max vacuity (conservative investment uncertainty)
- model_disagreement_score : fraction of sub-models disagreeing with selected action

Supported variants:
    none, A, B, C, AB, AC, BC, ABC

References
----------
See copilot-diagnostics/design/edl_uncertainty_poc/edl_v3_2_corrected_architecture.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from stock_investment_dss.uncertainty.edl_action_classes import (
    EDLActionConfig,
    get_action_classes,
    parse_variant_members,
)
from stock_investment_dss.uncertainty.edl_action_classifier import (
    EDLActionClassifier,
    EDLActionResult,
)

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ensemble result
# ---------------------------------------------------------------------------


@dataclass
class EDLEnsembleResult:
    """
    Combined result from an EDL ensemble (A/B/C or subset).

    Individual sub-model results:
        individual : dict[variant_id, EDLActionResult]

    Aggregate:
        p_ensemble            : weighted average probabilities per action
        u_ensemble            : weighted average vacuity
        u_conservative        : max(u_A, u_B, u_C) — conservative for investment
        model_disagreement_score : fraction of sub-models predicting a different action than selected
        ensemble_predicted_action : argmax of p_ensemble
        ensemble_weights      : weight per variant member

    Selected action quantities (from ensemble perspective):
        selected_action                  : input selected action
        ensemble_selected_probability    : p_ensemble for selected action
        ensemble_agrees_with_selected    : ensemble_predicted == selected
    """

    individual: Dict[str, EDLActionResult] = field(default_factory=dict)

    p_ensemble: Dict[str, float] = field(default_factory=dict)
    u_ensemble: float = 1.0
    u_conservative: float = 1.0
    model_disagreement_score: float = 0.0
    ensemble_predicted_action: str = "HOLD"
    ensemble_weights: Dict[str, float] = field(default_factory=dict)

    selected_action: str = "HOLD"
    ensemble_selected_probability: float = 0.0
    ensemble_agrees_with_selected: bool = False

    action_classes: List[str] = field(default_factory=list)
    num_members: int = 0

    def to_audit_dict(self) -> dict:
        """Flatten ensemble quantities for CSV audit output."""
        d: dict = {
            "ensemble_uncertainty": round(self.u_ensemble, 6),
            "ensemble_uncertainty_conservative": round(self.u_conservative, 6),
            "model_disagreement_score": round(self.model_disagreement_score, 6),
            "ensemble_predicted_action": self.ensemble_predicted_action,
            "ensemble_agrees_with_selected": self.ensemble_agrees_with_selected,
            "ensemble_selected_probability": round(
                self.ensemble_selected_probability, 6
            ),
            "ensemble_weights": str(self.ensemble_weights),
        }
        # Per-action ensemble probabilities
        for action in self.action_classes:
            d[f"p_ensemble_{action.lower()}"] = round(
                self.p_ensemble.get(action, 0.0), 6
            )
        return d


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


class EDLEnsemble:
    """
    Combines multiple EDLActionClassifier models (variants A, B, C).

    Parameters
    ----------
    classifiers : dict[str, EDLActionClassifier]
        Mapping from variant ID ('A', 'B', 'C') to classifier.
    weights : dict[str, float] or None
        Optional weights for each variant. If None, equal weights are used.
    config : EDLActionConfig
    """

    def __init__(
        self,
        classifiers: Dict[str, EDLActionClassifier],
        config: EDLActionConfig,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.classifiers = classifiers
        self.config = config
        self.members = sorted(classifiers.keys())
        self.action_classes = config.action_classes

        # Normalise weights
        if weights is None:
            w = 1.0 / max(len(self.members), 1)
            self.weights = {m: w for m in self.members}
        else:
            total = sum(weights.get(m, 1.0) for m in self.members)
            self.weights = {
                m: weights.get(m, 1.0) / max(total, 1e-8) for m in self.members
            }

    @classmethod
    def from_config(
        cls,
        config: EDLActionConfig,
        model_paths: Optional[Dict[str, str]] = None,
    ) -> "EDLEnsemble":
        """
        Build an EDLEnsemble from config, loading each variant model.

        Parameters
        ----------
        config : EDLActionConfig
        model_paths : dict[str, str] or None
            Optional dict mapping variant ID to .pt checkpoint path.
            If absent for a variant, that variant uses placeholder mode.
        """
        members = parse_variant_members(config.edl_variant)
        classifiers = {}
        for m in members:
            path = (model_paths or {}).get(m, config.model_path)
            classifiers[m] = EDLActionClassifier(
                config=config,
                model_path=path,
                source_variant=m,
            )
            status = "loaded" if not classifiers[m].is_placeholder() else "placeholder"
            logger.info("EDLEnsemble variant %s: %s", m, status)

        if not classifiers:
            # Single placeholder classifier for 'none' variant
            classifiers["none"] = EDLActionClassifier(
                config=config, source_variant="none"
            )

        return cls(classifiers=classifiers, config=config)

    def classify(
        self,
        features: np.ndarray,
        selected_action: str = "HOLD",
    ) -> EDLEnsembleResult:
        """
        Run all sub-model classifiers and combine results.

        Parameters
        ----------
        features : np.ndarray of shape (input_dim,)
        selected_action : str

        Returns
        -------
        EDLEnsembleResult
        """
        action_classes = self.action_classes
        individual: Dict[str, EDLActionResult] = {}

        for member, clf in self.classifiers.items():
            try:
                result = clf.predict(features, selected_action=selected_action)
                individual[member] = result
            except Exception as e:
                logger.warning("EDL variant %s classify failed: %s", member, e)
                individual[member] = clf._placeholder_result(
                    selected_action, action_classes, self.config.num_classes
                )

        return self._combine(individual, selected_action)

    def classify_batch(
        self,
        features: np.ndarray,
        selected_actions: List[str],
    ) -> List[EDLEnsembleResult]:
        """
        Run ensemble inference for a batch of feature vectors.
        """
        N = len(selected_actions)
        per_member: Dict[str, List[EDLActionResult]] = {}

        for member, clf in self.classifiers.items():
            try:
                per_member[member] = clf.predict_batch(features, selected_actions)
            except Exception as e:
                logger.warning("EDL variant %s batch classify failed: %s", member, e)
                per_member[member] = [
                    clf._placeholder_result(
                        sa, self.action_classes, self.config.num_classes
                    )
                    for sa in selected_actions
                ]

        results = []
        for i in range(N):
            individual_i = {m: per_member[m][i] for m in per_member}
            results.append(self._combine(individual_i, selected_actions[i]))
        return results

    # ------------------------------------------------------------------
    # Private combination logic
    # ------------------------------------------------------------------

    def _combine(
        self,
        individual: Dict[str, EDLActionResult],
        selected_action: str,
    ) -> EDLEnsembleResult:
        action_classes = self.action_classes
        sa = selected_action.upper().strip()

        # Weighted average probabilities
        p_ens: Dict[str, float] = {a: 0.0 for a in action_classes}
        u_sum = 0.0
        u_max = 0.0
        w_sum = 0.0

        for member, result in individual.items():
            w = self.weights.get(member, 0.0)
            w_sum += w
            for action in action_classes:
                p_ens[action] += w * result.probability_by_action.get(action, 0.0)
            u_sum += w * result.uncertainty_vacuity
            u_max = max(u_max, result.uncertainty_vacuity)

        # Normalise (guard against zero weight sum)
        if w_sum > 0:
            p_ens = {a: v / w_sum for a, v in p_ens.items()}
            u_ens = u_sum / w_sum
        else:
            p_ens = {a: 1.0 / max(len(action_classes), 1) for a in action_classes}
            u_ens = 1.0

        # Ensemble predicted action
        pred = max(p_ens, key=lambda a: p_ens[a])

        # Disagreement: fraction of sub-models that DON'T predict selected action
        # OR that predict a different action from the ensemble prediction
        n = len(individual)
        disagree_count = sum(
            1 for r in individual.values() if r.predicted_action.upper() != sa
        )
        disagreement = disagree_count / max(n, 1)

        ens_sel_prob = p_ens.get(sa, 0.0)

        return EDLEnsembleResult(
            individual=individual,
            p_ensemble=p_ens,
            u_ensemble=round(u_ens, 6),
            u_conservative=round(u_max, 6),
            model_disagreement_score=round(disagreement, 6),
            ensemble_predicted_action=pred,
            ensemble_weights=dict(self.weights),
            selected_action=sa,
            ensemble_selected_probability=round(ens_sel_prob, 6),
            ensemble_agrees_with_selected=(pred == sa),
            action_classes=action_classes,
            num_members=n,
        )
