# src/stock_investment_dss/uncertainty/edl_losses.py
"""
EDL Classification Loss Functions (v3.2)

Implements the Evidential Deep Learning loss for multi-class classification
as described in Sensoy et al. (NeurIPS 2018) Equation 4.

Loss components
---------------
1. Expected MSE / Bayes-risk term:
   L_MSE = sum_k [ (y_k - p_k)^2 + p_k*(1-p_k)/S ]

2. KL divergence toward uniform Dirichlet:
   Computed on evidence with true class zeroed out:
   alpha_tilde_k = y_k + (1 - y_k) * alpha_k
   KL = log Gamma(K) - sum_k log Gamma(alpha_tilde_k)
        + sum_k (alpha_tilde_k - 1) * [psi(alpha_tilde_k) - psi(S_tilde)]

3. Annealing:
   annealing_coeff = min(1.0, epoch / (total_epochs * anneal_frac))

4. Total loss:
   loss = L_MSE + annealing_coeff * kl_lambda * KL

No softmax is applied to the network output.
Input to loss is `alpha` (not raw logits), derived from evidence via Softplus.

References
----------
Sensoy, M., Kaplan, L., Kandemir, M. (2018).
"Evidential Deep Learning to Quantify Classification Uncertainty."
NeurIPS 2018.
"""

from __future__ import annotations

import math
from typing import Tuple

try:
    import torch
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore
    F = None  # type: ignore


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for EDL loss computation. "
            "Install with: pip install torch"
        )


# ---------------------------------------------------------------------------
# KL divergence helpers
# ---------------------------------------------------------------------------


def kl_divergence_uniform(alpha: "torch.Tensor") -> "torch.Tensor":
    """
    KL(Dir(alpha) || Dir(1,...,1)) — KL divergence from Dirichlet(alpha)
    to the uniform Dirichlet (all concentration parameters = 1).

    Parameters
    ----------
    alpha : Tensor of shape (batch, K)
        Dirichlet concentration parameters (all >= 1).

    Returns
    -------
    Tensor of shape (batch,)
        Per-sample KL divergence.
    """
    _require_torch()
    K = alpha.shape[-1]
    S = alpha.sum(dim=-1, keepdim=True)

    # log B(alpha) = sum log Gamma(alpha_k) - log Gamma(S)
    # log B(1,...,1) = sum log Gamma(1) - log Gamma(K) = 0 - log Gamma(K)
    # KL = log B(1,...,1) - log B(alpha)
    #    + sum_k (alpha_k - 1) * [psi(alpha_k) - psi(S)]

    log_b_alpha = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(S.squeeze(-1))
    log_b_uniform = torch.lgamma(
        torch.ones(K, dtype=alpha.dtype, device=alpha.device)
    ).sum() - torch.lgamma(
        torch.tensor(float(K), dtype=alpha.dtype, device=alpha.device)
    )

    digamma_diff = torch.digamma(alpha) - torch.digamma(S)
    kl = log_b_uniform - log_b_alpha + ((alpha - 1.0) * digamma_diff).sum(dim=-1)
    return kl.clamp(min=0.0)


def dirichlet_mse_loss(
    alpha: "torch.Tensor",
    y_onehot: "torch.Tensor",
) -> "torch.Tensor":
    """
    Expected MSE / Bayes-risk loss for Dirichlet classification.

    L = sum_k [ (y_k - p_k)^2 + p_k*(1-p_k)/S ]
      = sum_k [ (y_k - alpha_k/S)^2 + alpha_k*(S-alpha_k) / (S^2*(S+1)) ]

    Parameters
    ----------
    alpha : Tensor of shape (batch, K)
        Dirichlet concentration parameters (alpha_k = evidence_k + 1).
    y_onehot : Tensor of shape (batch, K)
        One-hot encoded true class labels.

    Returns
    -------
    Tensor of shape (batch,)
        Per-sample MSE loss.
    """
    _require_torch()
    S = alpha.sum(dim=-1, keepdim=True)
    p = alpha / S

    # Squared error term
    sq_err = (y_onehot - p) ** 2

    # Variance term: alpha_k*(S-alpha_k) / (S^2*(S+1))
    variance = alpha * (S - alpha) / (S * S * (S + 1.0))

    return (sq_err + variance).sum(dim=-1)


def annealing_coeff(epoch: int, total_epochs: int, anneal_frac: float = 0.5) -> float:
    """
    Linearly ramp the KL annealing coefficient from 0 to 1.

    annealing = min(1.0, epoch / (total_epochs * anneal_frac))

    Parameters
    ----------
    epoch : int
        Current training epoch (0-indexed).
    total_epochs : int
        Total number of training epochs.
    anneal_frac : float
        Fraction of training over which KL weight ramps to 1.0.

    Returns
    -------
    float in [0, 1]
    """
    if total_epochs <= 0 or anneal_frac <= 0:
        return 1.0
    return min(1.0, epoch / max(1, int(total_epochs * anneal_frac)))


# ---------------------------------------------------------------------------
# Main EDL loss module
# ---------------------------------------------------------------------------


def edl_action_loss(
    alpha: "torch.Tensor",
    y_onehot: "torch.Tensor",
    epoch: int,
    total_epochs: int,
    kl_lambda: float = 1.0,
    anneal_frac: float = 0.5,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Full EDL classification loss (Sensoy et al. 2018, Eq. 4).

    loss = L_MSE + annealing * kl_lambda * KL(Dir(alpha_tilde) || Dir(1,...,1))

    where alpha_tilde = y + (1-y)*alpha (evidence for incorrect classes only).

    Parameters
    ----------
    alpha : Tensor of shape (batch, K)
    y_onehot : Tensor of shape (batch, K), float
    epoch : int
    total_epochs : int
    kl_lambda : float
        Weight on the KL regularisation term.
    anneal_frac : float
        Fraction of total_epochs over which KL ramps to kl_lambda.

    Returns
    -------
    (total_loss, mse_loss, kl_loss) — all Tensors of shape (batch,)
    Caller typically calls total_loss.mean() for the batch loss.
    """
    _require_torch()

    # MSE / Bayes-risk term
    mse = dirichlet_mse_loss(alpha, y_onehot)

    # KL on alpha_tilde (zero out evidence for the true class)
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    kl = kl_divergence_uniform(alpha_tilde)

    coeff = annealing_coeff(epoch, total_epochs, anneal_frac)
    loss = mse + coeff * kl_lambda * kl

    return loss, mse, kl


# ---------------------------------------------------------------------------
# Pure Python sanity check (no torch required)
# ---------------------------------------------------------------------------


def sanity_check_vacuity(K: int, evidence_values: list) -> float:
    """
    Compute vacuity u = K / S from a list of evidence values.
    Pure Python — can be used without PyTorch for quick checks.

    Parameters
    ----------
    K : int
        Number of classes.
    evidence_values : list of float
        Non-negative evidence per class (length K).

    Returns
    -------
    float: vacuity in (0, 1]
    """
    if len(evidence_values) != K:
        raise ValueError(f"Expected {K} evidence values, got {len(evidence_values)}")
    alpha = [e + 1.0 for e in evidence_values]
    S = sum(alpha)
    return K / S
