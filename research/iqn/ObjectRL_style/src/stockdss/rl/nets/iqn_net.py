"""
Implicit Quantile Network used by the D-IQN-DSS agent.
"""

from __future__ import annotations

import torch
from torch import nn


class IQNNetwork(nn.Module):
    """
    IQN network.

    Input:
        states: shape [batch_size, state_dim]
        taus:   shape [batch_size, num_quantiles]

    Output:
        quantile_values: shape [batch_size, num_quantiles, action_dim]

    Meaning:
        quantile_values[b, q, a] = Z_tau_q(state_b, action_a)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        cosine_embedding_dim: int,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.cosine_embedding_dim = cosine_embedding_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        self.tau_embedding = nn.Sequential(
            nn.Linear(cosine_embedding_dim, hidden_dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        batch_size = states.shape[0]
        num_quantiles = taus.shape[1]

        state_features = self.state_encoder(states)

        i_pi = (
            torch.arange(
                1,
                self.cosine_embedding_dim + 1,
                device=states.device,
                dtype=torch.float32,
            )
            * torch.pi
        ).view(1, 1, -1)

        cosines = torch.cos(taus.unsqueeze(-1) * i_pi)

        tau_features = self.tau_embedding(
            cosines.view(batch_size * num_quantiles, self.cosine_embedding_dim)
        )

        state_features = state_features.unsqueeze(1).expand(
            batch_size, num_quantiles, self.hidden_dim
        )
        state_features = state_features.reshape(
            batch_size * num_quantiles, self.hidden_dim
        )

        combined = state_features * tau_features

        quantile_values = self.head(combined)
        quantile_values = quantile_values.view(
            batch_size, num_quantiles, self.action_dim
        )

        return quantile_values
