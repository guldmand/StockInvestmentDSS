# src/stock_investment_dss/uncertainty/edl_action_network.py
"""
EDL Action Network — PyTorch MLP with Evidence Head (v3.2)

Implements the neural network backbone for evidential classification over
DSS action classes (HOLD / BUY / SELL / REBALANCE [/ CHANGE_STRATEGY]).

Architecture
------------
Input: feature vector x ∈ ℝ^input_dim
Hidden: configurable MLP layers with SiLU activations + optional Dropout
Output: evidence e ∈ ℝ^K via Softplus (guarantees e_k >= 0)

Dirichlet quantities (computed in forward pass):
    alpha_k = e_k + 1
    S       = Σ_k alpha_k
    p_k     = alpha_k / S
    u       = K / S    (vacuity / epistemic uncertainty)
    b_k     = e_k / S  (belief mass)

No softmax is applied to the output. Probabilities are derived from the
Dirichlet parameterisation, not from a softmax distribution.

References
----------
Sensoy et al. (NeurIPS 2018)
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for EDLActionNetwork. "
            "Install with: pip install torch"
        )


# ---------------------------------------------------------------------------
# Default network hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_HIDDEN_DIMS: List[int] = [128, 64]
DEFAULT_DROPOUT: float = 0.1


class EDLActionNetwork(nn.Module if _TORCH_AVAILABLE else object):
    """
    MLP classifier with Softplus evidence output head for evidential learning.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    num_classes : int
        K — number of DSS action classes (4 or 5).
    hidden_dims : list of int
        Sizes of hidden layers. Default: [128, 64].
    dropout : float
        Dropout probability applied after each hidden layer.
    evidence_activation : str
        'softplus' (recommended) or 'relu'.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = DEFAULT_DROPOUT,
        evidence_activation: str = "softplus",
    ) -> None:
        _require_torch()
        super().__init__()

        if hidden_dims is None:
            hidden_dims = DEFAULT_HIDDEN_DIMS

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.evidence_activation_name = evidence_activation

        # Build MLP layers
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, num_classes)

        # Evidence activation
        if evidence_activation == "softplus":
            self.evidence_fn = nn.Softplus()
        elif evidence_activation == "relu":
            self.evidence_fn = nn.ReLU()
        else:
            raise ValueError(
                f"Unknown evidence_activation '{evidence_activation}'. "
                "Choose 'softplus' or 'relu'."
            )

    def forward(self, x: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)

        Returns
        -------
        dict with keys:
            evidence  : (batch, K) — non-negative evidence values
            alpha     : (batch, K) — Dirichlet concentration (evidence + 1)
            S         : (batch, 1) — Dirichlet strength
            prob      : (batch, K) — expected class probabilities (alpha / S)
            vacuity   : (batch, 1) — epistemic uncertainty u = K / S
            belief    : (batch, K) — belief mass (evidence / S)
        """
        logits = self.head(self.backbone(x))
        evidence = self.evidence_fn(logits)
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True)
        prob = alpha / S
        vacuity = self.num_classes / S
        belief = evidence / S

        return {
            "evidence": evidence,
            "alpha": alpha,
            "S": S,
            "prob": prob,
            "vacuity": vacuity,
            "belief": belief,
        }

    def predict(self, x: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        """Run forward pass in eval mode (no gradient)."""
        _require_torch()
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def save(self, path: str) -> None:
        """Save model weights and config to a .pt checkpoint."""
        _require_torch()
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "evidence_activation": self.evidence_activation_name,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls, path: str, hidden_dims: Optional[List[int]] = None
    ) -> "EDLActionNetwork":
        """Load model from a .pt checkpoint."""
        _require_torch()
        checkpoint = torch.load(path, map_location="cpu")
        net = cls(
            input_dim=checkpoint["input_dim"],
            num_classes=checkpoint["num_classes"],
            hidden_dims=hidden_dims or DEFAULT_HIDDEN_DIMS,
            evidence_activation=checkpoint.get("evidence_activation", "softplus"),
        )
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        return net
