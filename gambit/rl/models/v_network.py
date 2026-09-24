"""State value network V(s) for IQL expectile regression."""

from __future__ import annotations

import torch
import torch.nn as nn


class VNetwork(nn.Module):
    """State value network V(s).

    Input:
        state [B, state_dim]

    Output:
        value [B]

    Trained with expectile regression against Q(s, a) targets.
    """

    def __init__(
        self,
        state_dim: int = 512,
        hidden_dim: int = 1024,
        num_hidden_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim

        layers: list[nn.Module] = []

        # First layer
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output: scalar value
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute V(s).

        Args:
            state: [B, state_dim]

        Returns:
            value: [B]
        """
        return self.net(state).squeeze(-1)
