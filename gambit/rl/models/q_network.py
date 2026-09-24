"""State-action value network Q(s, a) for IQL."""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """State-action value network Q(s, a).

    Input:
        state  [B, state_dim]
        action [B, action_dim]

    Output:
        q_value [B]

    IQL uses two independent QNetwork instances to reduce overestimation.
    """

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 14,
        hidden_dim: int = 1024,
        num_hidden_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers: list[nn.Module] = []
        in_dim = state_dim + action_dim

        # First layer
        layers.append(nn.Linear(in_dim, hidden_dim))
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

        # Output: scalar Q value
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q(s, a).

        Args:
            state:  [B, state_dim]
            action: [B, action_dim]

        Returns:
            q_value: [B]
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)
