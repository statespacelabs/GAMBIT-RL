"""Policy networks for behavior cloning and IQL actor training."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared MLP builder
# ---------------------------------------------------------------------------

def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 1024,
    num_hidden: int = 3,
    dropout: float = 0.1,
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    """Build a residual-free MLP with LayerNorm and ReLU."""
    layers: list[nn.Module] = []

    # Input layer
    layers.append(nn.Linear(in_dim, hidden_dim))
    layers.append(nn.LayerNorm(hidden_dim))
    layers.append(nn.ReLU())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    # Hidden layers
    for _ in range(num_hidden - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    # Output layer
    layers.append(nn.Linear(hidden_dim, out_dim))
    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# BC Policy
# ---------------------------------------------------------------------------

class BCPolicy(nn.Module):
    """Simple MLP policy for behavior cloning.

    Input:  state [B, state_dim]
    Output: action prediction [B, action_dim]

    The first implementation predicts the compact continuous/binary action
    vector directly. Binary dimensions can be passed through sigmoid during
    loss/inference.
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
        self.net = _build_mlp(
            in_dim=state_dim,
            out_dim=action_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden_layers,
            dropout=dropout,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict action from state.

        Args:
            state: [B, state_dim]

        Returns:
            action_pred: [B, action_dim]
        """
        return self.net(state)


# ---------------------------------------------------------------------------
# Gaussian Policy (for IQL actor)
# ---------------------------------------------------------------------------

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    """Continuous-action Gaussian policy for IQL actor training.

    Input:  state [B, state_dim]
    Output: mean [B, action_dim], log_std [B, action_dim]

    Used for advantage-weighted behavior cloning inside IQL.
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

        self.backbone = _build_mlp(
            in_dim=state_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden_layers,
            dropout=dropout,
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log_std from state.

        Returns:
            (mean, log_std), each [B, action_dim]
        """
        h = self.backbone(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action using reparameterization and return action plus log probability.

        During evaluation, use deterministic mean action instead.

        Returns:
            (action, log_prob), each [B, action_dim] and [B] respectively.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization trick
        noise = torch.randn_like(mean)
        action = mean + noise * std

        # Log probability under Gaussian
        log_prob = -0.5 * (
            ((action - mean) / (std + 1e-8)) ** 2
            + 2 * log_std
            + math.log(2 * math.pi)
        )
        # Sum over action dims for total log prob
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob

    def act(
        self, state: torch.Tensor, deterministic: bool = True
    ) -> torch.Tensor:
        """Produce an action for inference.

        If deterministic=True, return policy mean. Otherwise sample from the Gaussian.
        Binary dimensions should be thresholded outside this class using ActionSchema.
        """
        mean, log_std = self.forward(state)
        if deterministic:
            return mean
        std = log_std.exp()
        return mean + torch.randn_like(mean) * std

    def log_prob(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of a given action under the policy.

        Returns:
            log_prob: [B]
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        log_prob = -0.5 * (
            ((action - mean) / (std + 1e-8)) ** 2
            + 2 * log_std
            + math.log(2 * math.pi)
        )
        return log_prob.sum(dim=-1)
