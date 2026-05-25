# src/stock_investment_dss/uncertainty/edl_gate.py
"""
EDL Gate — Derived Decision Gate for D-IQN-DSS (v3.2)

The EDL gate translates EDL action-uncertainty outputs into actionable
investment decision modifications.

IMPORTANT: The gate is derived AFTER EDL classification.
The primary EDL Dirichlet class space is DSS actions (HOLD/BUY/SELL/REBALANCE).
The gate outputs are SECONDARY outputs: RECOMMEND_AS_IS / REDUCE_SIZE /
FORCE_HOLD / HUMAN_REVIEW / STRATEGY_REVIEW.

Gate logic
----------
1. If EDL agrees with selected action AND vacuity is low:
   → RECOMMEND_AS_IS (high confidence, no modification)

2. If EDL agrees but vacuity is medium:
   → REDUCE_SIZE (reduce allocation fraction by uncertainty penalty)

3. If EDL agrees but vacuity is high:
   → HUMAN_REVIEW (model is unsure even when agreeing)

4. If EDL disagrees with selected action AND vacuity < disagreement_force_hold_threshold:
   → HUMAN_REVIEW (EDL disagrees; investor should review)

5. If EDL disagrees AND vacuity >= disagreement_force_hold_threshold:
   → FORCE_HOLD (strong disagreement under high uncertainty → conservative)

6. If EDL strongly predicts REBALANCE (p_rebalance > rebalance_signal_threshold):
   → STRATEGY_REVIEW (or HUMAN_REVIEW if below threshold)

7. If EDL strongly predicts CHANGE_STRATEGY (p_cs > change_strategy_signal_threshold):
   → STRATEGY_REVIEW

8. High model disagreement (ensemble only):
   → Adds HUMAN_REVIEW flag

Size reduction formula (when gate = REDUCE_SIZE):
    final_fraction = max(0.0, original_fraction * (1 - lambda_u * vacuity))

References
----------
See copilot-diagnostics/design/edl_uncertainty_poc/edl_v3_2_corrected_architecture.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from stock_investment_dss.uncertainty.edl_action_classes import EDLActionConfig

# ---------------------------------------------------------------------------
# Gate output labels
# ---------------------------------------------------------------------------

GATE_RECOMMEND_AS_IS = "RECOMMEND_AS_IS"
GATE_REDUCE_SIZE = "REDUCE_SIZE"
GATE_FORCE_HOLD = "FORCE_HOLD"
GATE_HUMAN_REVIEW = "HUMAN_REVIEW"
GATE_STRATEGY_REVIEW = "STRATEGY_REVIEW"

ALL_GATE_OUTPUTS = [
    GATE_RECOMMEND_AS_IS,
    GATE_REDUCE_SIZE,
    GATE_FORCE_HOLD,
    GATE_HUMAN_REVIEW,
    GATE_STRATEGY_REVIEW,
]


# ---------------------------------------------------------------------------
# Gate configuration
# ---------------------------------------------------------------------------


@dataclass
class EDLGateConfig:
    """
    Thresholds and weights for the EDL gate.

    All thresholds are on vacuity u ∈ [0, 1].
    """

    # Vacuity thresholds
    recommend_as_is_max_vacuity: float = (
        0.35  # u < this → RECOMMEND_AS_IS (when agreeing)
    )
    reduce_size_max_vacuity: float = 0.55  # 0.35 ≤ u < 0.55 → REDUCE_SIZE
    # u ≥ reduce_size_max_vacuity → HUMAN_REVIEW (when agreeing)

    # Disagreement gate
    force_hold_min_vacuity: float = 0.45  # Disagree + u ≥ this → FORCE_HOLD
    # Disagree + u < this → HUMAN_REVIEW

    # Strategy signal
    rebalance_signal_threshold: float = 0.60  # p_rebalance > this → STRATEGY_REVIEW
    change_strategy_signal_threshold: float = 0.50  # p_cs > this → STRATEGY_REVIEW

    # Disagreement score
    disagreement_review_threshold: float = (
        0.50  # disagreement > this → add HUMAN_REVIEW
    )

    # Size reduction
    uncertainty_lambda: float = 0.5  # final = original * (1 - lambda * u)

    @classmethod
    def from_config(cls, edl_config: EDLActionConfig) -> "EDLGateConfig":
        return cls(uncertainty_lambda=edl_config.uncertainty_lambda)


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------


@dataclass
class EDLGateResult:
    """
    Output of the EDL gate for a single decision step.
    """

    recommendation_gate: str = GATE_RECOMMEND_AS_IS
    should_reduce_size: bool = False
    should_force_hold: bool = False
    should_require_human_review: bool = False
    should_strategy_review: bool = False

    final_action_after_edl_gate: str = "HOLD"
    final_size_after_edl_gate: str = ""
    final_fraction_after_edl_gate: float = 0.0
    original_selected_fraction: float = 0.0

    uncertainty_penalty: float = 0.0
    disagreement_penalty: float = 0.0

    reason_codes: List[str] = field(default_factory=list)

    def to_audit_dict(self) -> dict:
        return {
            "recommendation_gate": self.recommendation_gate,
            "should_reduce_size": self.should_reduce_size,
            "should_force_hold": self.should_force_hold,
            "should_require_human_review": self.should_require_human_review,
            "final_action_after_edl_gate": self.final_action_after_edl_gate,
            "final_size_after_edl_gate": self.final_size_after_edl_gate,
            "final_fraction_after_edl_gate": round(
                self.final_fraction_after_edl_gate, 6
            ),
            "original_selected_fraction": round(self.original_selected_fraction, 6),
            "uncertainty_penalty": round(self.uncertainty_penalty, 6),
            "disagreement_penalty": round(self.disagreement_penalty, 6),
            "reason_codes": "|".join(self.reason_codes),
        }


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------


class EDLGate:
    """
    Derives investment recommendation modification from EDL action-uncertainty outputs.

    Parameters
    ----------
    config : EDLGateConfig
    """

    def __init__(self, config: Optional[EDLGateConfig] = None) -> None:
        self.config = config or EDLGateConfig()

    @classmethod
    def from_edl_config(cls, edl_config: EDLActionConfig) -> "EDLGate":
        return cls(config=EDLGateConfig.from_config(edl_config))

    def apply(
        self,
        selected_action: str,
        selected_size: str,
        original_fraction: float,
        vacuity: float,
        edl_agrees: bool,
        edl_predicted_action: str,
        p_rebalance: float = 0.0,
        p_change_strategy: float = 0.0,
        disagreement_score: float = 0.0,
        uncertainty_penalty: float = 0.0,
        disagreement_penalty: float = 0.0,
    ) -> EDLGateResult:
        """
        Apply gate logic for a single decision.

        Parameters
        ----------
        selected_action : str
            Action selected by hierarchical policy.
        selected_size : str
            Size label (e.g. BUY_25, SELL_100).
        original_fraction : float
            Allocation fraction before gate.
        vacuity : float
            EDL epistemic uncertainty u = K/S.
        edl_agrees : bool
            True if EDL predicted action == selected action.
        edl_predicted_action : str
        p_rebalance : float
            EDL probability for REBALANCE class.
        p_change_strategy : float
            EDL probability for CHANGE_STRATEGY class (0 if K=4).
        disagreement_score : float
            Fraction of ensemble members disagreeing (0 if single model).
        uncertainty_penalty : float
            Pre-computed λ_u * u.
        disagreement_penalty : float
            Pre-computed λ_d * disagreement.

        Returns
        -------
        EDLGateResult
        """
        cfg = self.config
        sa = selected_action.upper()
        reasons: List[str] = []

        # ------------------------------------------------------------------
        # Step 1: Determine gate from vacuity and agreement
        # ------------------------------------------------------------------
        if edl_agrees:
            if vacuity < cfg.recommend_as_is_max_vacuity:
                gate = GATE_RECOMMEND_AS_IS
            elif vacuity < cfg.reduce_size_max_vacuity:
                gate = GATE_REDUCE_SIZE
                reasons.append("medium_vacuity")
            else:
                gate = GATE_HUMAN_REVIEW
                reasons.append("high_vacuity_with_agreement")
        else:
            # EDL disagrees
            reasons.append("edl_disagrees")
            if vacuity >= cfg.force_hold_min_vacuity:
                gate = GATE_FORCE_HOLD
                reasons.append("high_vacuity_with_disagreement")
            else:
                gate = GATE_HUMAN_REVIEW

        # ------------------------------------------------------------------
        # Step 2: Strategy signals (can upgrade gate)
        # ------------------------------------------------------------------
        if p_change_strategy > cfg.change_strategy_signal_threshold:
            gate = GATE_STRATEGY_REVIEW
            reasons.append("strong_change_strategy_signal")
        elif p_rebalance > cfg.rebalance_signal_threshold and sa not in (
            "REBALANCE",
            "HOLD",
        ):
            gate = GATE_STRATEGY_REVIEW
            reasons.append("strong_rebalance_signal")

        # ------------------------------------------------------------------
        # Step 3: Ensemble disagreement (can add HUMAN_REVIEW)
        # ------------------------------------------------------------------
        if disagreement_score > cfg.disagreement_review_threshold:
            if gate == GATE_RECOMMEND_AS_IS:
                gate = GATE_HUMAN_REVIEW
            reasons.append("high_ensemble_disagreement")

        # ------------------------------------------------------------------
        # Step 4: Compute final action and fraction
        # ------------------------------------------------------------------
        should_force_hold = gate == GATE_FORCE_HOLD
        should_reduce = gate == GATE_REDUCE_SIZE
        should_review = gate in (GATE_HUMAN_REVIEW, GATE_STRATEGY_REVIEW)

        final_action = "HOLD" if should_force_hold else sa
        final_fraction = original_fraction

        if should_force_hold:
            final_fraction = 0.0
            final_size = ""
        elif should_reduce:
            final_fraction = max(
                0.0, original_fraction * (1.0 - cfg.uncertainty_lambda * vacuity)
            )
            final_size = selected_size  # size label unchanged; fraction is reduced
        else:
            final_size = selected_size

        return EDLGateResult(
            recommendation_gate=gate,
            should_reduce_size=should_reduce,
            should_force_hold=should_force_hold,
            should_require_human_review=should_review or should_force_hold,
            should_strategy_review=(gate == GATE_STRATEGY_REVIEW),
            final_action_after_edl_gate=final_action,
            final_size_after_edl_gate=final_size,
            final_fraction_after_edl_gate=round(final_fraction, 6),
            original_selected_fraction=round(original_fraction, 6),
            uncertainty_penalty=round(uncertainty_penalty, 6),
            disagreement_penalty=round(disagreement_penalty, 6),
            reason_codes=reasons,
        )

    def null_gate(
        self,
        selected_action: str,
        selected_size: str,
        original_fraction: float,
    ) -> EDLGateResult:
        """
        Return a no-op gate result (when EDL is disabled or variant = none).
        All fields are pass-through; no modification to action or fraction.
        """
        return EDLGateResult(
            recommendation_gate=GATE_RECOMMEND_AS_IS,
            should_reduce_size=False,
            should_force_hold=False,
            should_require_human_review=False,
            final_action_after_edl_gate=selected_action.upper(),
            final_size_after_edl_gate=selected_size,
            final_fraction_after_edl_gate=round(original_fraction, 6),
            original_selected_fraction=round(original_fraction, 6),
            uncertainty_penalty=0.0,
            disagreement_penalty=0.0,
            reason_codes=["edl_disabled"],
        )
