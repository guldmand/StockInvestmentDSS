# src/stock_investment_dss/uncertainty/edl_action_classifier.py
"""
EDL Action Classifier — High-level wrapper for D-IQN-DSS (v3.2)

Wraps EDLActionNetwork with:
- train_step: single batch update
- predict: inference for one or more feature vectors
- save / load: checkpoint management
- EDLActionResult: structured inference result per decision

This module provides the interface used by:
- run_edl_action_training_smoke_test.py
- run_edl_action_inference_smoke_test.py
- EDLEnsemble (via per-variant classifiers)

References
----------
Sensoy et al. (NeurIPS 2018)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from stock_investment_dss.uncertainty.edl_action_classes import (
    EDLActionConfig,
    get_action_classes,
    idx_to_action,
)

try:
    import torch
    import torch.optim as optim

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore
    optim = None  # type: ignore


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for EDLActionClassifier. "
            "Install with: pip install torch"
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EDLActionResult:
    """
    Structured inference output for a single decision step.

    Primary EDL quantities (over action class space):
    -------------------------------------------------
    evidence_by_action : dict  — e_k per action
    alpha_by_action    : dict  — alpha_k = e_k + 1 per action
    probability_by_action : dict — p_k = alpha_k / S per action
    belief_by_action   : dict  — b_k = e_k / S per action
    dirichlet_strength : float — S = sum(alpha)
    uncertainty_vacuity: float — u = K / S

    Selected-action focus quantities:
    ---------------------------------
    predicted_action              : str  — argmax p_k
    selected_action               : str  — the action selected by HierarchicalDecisionPolicy
    selected_action_probability   : float
    selected_action_evidence      : float
    selected_action_belief        : float
    selected_action_uncertainty   : float  (= vacuity; same Dirichlet for all)
    edl_agrees_with_selected_action : bool

    Metadata:
    ---------
    num_classes     : int
    action_classes  : list[str]
    edl_model_version : str
    source_variant  : str  — A / B / C / none
    """

    evidence_by_action: Dict[str, float] = field(default_factory=dict)
    alpha_by_action: Dict[str, float] = field(default_factory=dict)
    probability_by_action: Dict[str, float] = field(default_factory=dict)
    belief_by_action: Dict[str, float] = field(default_factory=dict)

    dirichlet_strength: float = 0.0
    uncertainty_vacuity: float = 1.0

    predicted_action: str = "HOLD"
    selected_action: str = "HOLD"
    selected_action_probability: float = 0.0
    selected_action_evidence: float = 0.0
    selected_action_belief: float = 0.0
    selected_action_uncertainty: float = 1.0

    edl_agrees_with_selected_action: bool = False

    num_classes: int = 4
    action_classes: List[str] = field(default_factory=list)
    edl_model_version: str = "edl_v3_2"
    source_variant: str = "none"

    def to_audit_dict(self, decision_row: Optional[dict] = None) -> dict:
        """
        Flatten to a flat dictionary suitable for CSV audit output.

        Parameters
        ----------
        decision_row : dict or None
            Context from hierarchical policy audit (date, ticker, size, etc.)
        """
        base = dict(decision_row) if decision_row else {}

        # Per-action probabilities
        for action in self.action_classes:
            a = action.lower()
            base[f"p_{a}"] = round(self.probability_by_action.get(action, 0.0), 6)
            base[f"evidence_{a}"] = round(self.evidence_by_action.get(action, 0.0), 6)
            base[f"alpha_{a}"] = round(self.alpha_by_action.get(action, 1.0), 6)

        # Summary
        base.update(
            {
                "edl_predicted_action": self.predicted_action,
                "edl_agrees_with_selected_action": self.edl_agrees_with_selected_action,
                "selected_action_probability": round(
                    self.selected_action_probability, 6
                ),
                "selected_action_evidence": round(self.selected_action_evidence, 6),
                "selected_action_belief": round(self.selected_action_belief, 6),
                "selected_action_uncertainty": round(
                    self.selected_action_uncertainty, 6
                ),
                "dirichlet_strength": round(self.dirichlet_strength, 6),
                "uncertainty_vacuity": round(self.uncertainty_vacuity, 6),
                "edl_model_version": self.edl_model_version,
                "source_variant": self.source_variant,
            }
        )
        return base

    def __repr__(self) -> str:
        return (
            f"EDLActionResult("
            f"selected={self.selected_action} "
            f"predicted={self.predicted_action} "
            f"p_selected={self.selected_action_probability:.3f} "
            f"vacuity={self.uncertainty_vacuity:.3f} "
            f"agrees={self.edl_agrees_with_selected_action})"
        )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class EDLActionClassifier:
    """
    Wraps EDLActionNetwork for training and inference.

    In v3.2 PoC, if no trained model is available, a 'placeholder' classifier
    is used that produces uniform Dirichlet outputs (maximum vacuity).
    This is better than nothing: it correctly signals "no evidence available".

    Parameters
    ----------
    config : EDLActionConfig
    model_path : str or None
        Path to .pt checkpoint. If None or empty, placeholder mode is used.
    source_variant : str
        'A', 'B', 'C', or 'none' — which label strategy trained this model.
    """

    def __init__(
        self,
        config: EDLActionConfig,
        model_path: Optional[str] = None,
        source_variant: str = "none",
    ) -> None:
        self.config = config
        self.source_variant = source_variant
        self._model = None
        self._placeholder = True

        effective_path = model_path or config.model_path
        if effective_path and Path(effective_path).exists():
            try:
                self._load_network(effective_path)
                self._placeholder = False
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to load EDL model from '%s': %s — using placeholder.",
                    effective_path,
                    e,
                )

    def _load_network(self, path: str) -> None:
        _require_torch()
        from stock_investment_dss.uncertainty.edl_action_network import EDLActionNetwork

        self._model = EDLActionNetwork.load(path)
        self._model.eval()

    def is_placeholder(self) -> bool:
        """True if no trained model is loaded (uniform Dirichlet fallback)."""
        return self._placeholder

    def train_step(
        self,
        x: "torch.Tensor",
        y_onehot: "torch.Tensor",
        optimizer: "torch.optim.Optimizer",
        epoch: int,
        total_epochs: int,
        kl_lambda: float = 1.0,
    ) -> dict:
        """
        Single training batch step.

        Parameters
        ----------
        x : Tensor (batch, input_dim)
        y_onehot : Tensor (batch, K)
        optimizer : torch optimizer
        epoch, total_epochs : for KL annealing
        kl_lambda : KL regularisation weight

        Returns
        -------
        dict with 'loss', 'mse_loss', 'kl_loss' (float scalars)
        """
        _require_torch()
        from stock_investment_dss.uncertainty.edl_losses import edl_action_loss

        self._model.train()
        optimizer.zero_grad()

        out = self._model(x)
        loss, mse, kl = edl_action_loss(
            out["alpha"], y_onehot, epoch, total_epochs, kl_lambda
        )
        batch_loss = loss.mean()
        batch_loss.backward()
        optimizer.step()

        return {
            "loss": batch_loss.item(),
            "mse_loss": mse.mean().item(),
            "kl_loss": kl.mean().item(),
        }

    def predict(
        self,
        features: np.ndarray,
        selected_action: str = "HOLD",
    ) -> EDLActionResult:
        """
        Run inference for a single feature vector.

        Parameters
        ----------
        features : np.ndarray of shape (input_dim,) or (1, input_dim)
        selected_action : str
            The action selected by HierarchicalDecisionPolicy for this step.

        Returns
        -------
        EDLActionResult
        """
        action_classes = self.config.action_classes
        K = self.config.num_classes

        if self._placeholder or not _TORCH_AVAILABLE:
            # Uniform Dirichlet — maximum vacuity
            return self._placeholder_result(selected_action, action_classes, K)

        _require_torch()
        x = torch.tensor(features, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        out = self._model.predict(x)

        evidence = out["evidence"][0].tolist()
        alpha = out["alpha"][0].tolist()
        prob = out["prob"][0].tolist()
        belief = out["belief"][0].tolist()
        S = out["S"][0].item()
        u = out["vacuity"][0].item()

        return self._build_result(
            evidence, alpha, prob, belief, S, u, action_classes, K, selected_action
        )

    def predict_batch(
        self,
        features: np.ndarray,
        selected_actions: List[str],
    ) -> List[EDLActionResult]:
        """
        Run inference for a batch of feature vectors.

        Parameters
        ----------
        features : np.ndarray of shape (N, input_dim)
        selected_actions : list of str, length N

        Returns
        -------
        list of EDLActionResult, length N
        """
        action_classes = self.config.action_classes
        K = self.config.num_classes

        if self._placeholder or not _TORCH_AVAILABLE:
            return [
                self._placeholder_result(sa, action_classes, K)
                for sa in selected_actions
            ]

        _require_torch()
        x = torch.tensor(features, dtype=torch.float32)
        out = self._model.predict(x)

        results = []
        for i, sa in enumerate(selected_actions):
            evidence = out["evidence"][i].tolist()
            alpha = out["alpha"][i].tolist()
            prob = out["prob"][i].tolist()
            belief = out["belief"][i].tolist()
            S = out["S"][i].item()
            u = out["vacuity"][i].item()
            results.append(
                self._build_result(
                    evidence, alpha, prob, belief, S, u, action_classes, K, sa
                )
            )
        return results

    def save(self, path: str) -> None:
        """Save the underlying network to a .pt checkpoint."""
        if self._model is None:
            raise RuntimeError("No model to save — classifier is in placeholder mode.")
        self._model.save(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _placeholder_result(
        self,
        selected_action: str,
        action_classes: List[str],
        K: int,
    ) -> EDLActionResult:
        """Return a uniform-Dirichlet result (e_k=0 for all k → maximum vacuity)."""
        evidence = {a: 0.0 for a in action_classes}
        alpha = {a: 1.0 for a in action_classes}
        S = float(K)
        prob = {a: 1.0 / K for a in action_classes}
        belief = {a: 0.0 for a in action_classes}
        u = 1.0  # maximum vacuity

        return EDLActionResult(
            evidence_by_action=evidence,
            alpha_by_action=alpha,
            probability_by_action=prob,
            belief_by_action=belief,
            dirichlet_strength=S,
            uncertainty_vacuity=u,
            predicted_action=action_classes[0],  # HOLD (index 0) as uninformed prior
            selected_action=selected_action.upper(),
            selected_action_probability=1.0 / K,
            selected_action_evidence=0.0,
            selected_action_belief=0.0,
            selected_action_uncertainty=u,
            edl_agrees_with_selected_action=False,
            num_classes=K,
            action_classes=action_classes,
            edl_model_version="edl_v3_2_placeholder_uniform",
            source_variant=self.source_variant,
        )

    def _build_result(
        self,
        evidence: list,
        alpha: list,
        prob: list,
        belief: list,
        S: float,
        u: float,
        action_classes: List[str],
        K: int,
        selected_action: str,
    ) -> EDLActionResult:
        ev_dict = {a: evidence[i] for i, a in enumerate(action_classes)}
        al_dict = {a: alpha[i] for i, a in enumerate(action_classes)}
        p_dict = {a: prob[i] for i, a in enumerate(action_classes)}
        b_dict = {a: belief[i] for i, a in enumerate(action_classes)}

        predicted = action_classes[int(np.argmax(prob))]
        sa_upper = selected_action.upper().strip()
        sa_prob = p_dict.get(sa_upper, 1.0 / K)
        sa_ev = ev_dict.get(sa_upper, 0.0)
        sa_b = b_dict.get(sa_upper, 0.0)

        return EDLActionResult(
            evidence_by_action=ev_dict,
            alpha_by_action=al_dict,
            probability_by_action=p_dict,
            belief_by_action=b_dict,
            dirichlet_strength=S,
            uncertainty_vacuity=u,
            predicted_action=predicted,
            selected_action=sa_upper,
            selected_action_probability=sa_prob,
            selected_action_evidence=sa_ev,
            selected_action_belief=sa_b,
            selected_action_uncertainty=u,
            edl_agrees_with_selected_action=(predicted == sa_upper),
            num_classes=K,
            action_classes=action_classes,
            edl_model_version="edl_v3_2",
            source_variant=self.source_variant,
        )
