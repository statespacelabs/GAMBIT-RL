"""Advanced policy networks using Residual MLPs and Mixture Density Networks (MDN)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class ResidualBlock(nn.Module):
    """A residual block with LayerNorm and ReLU."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.relu(self.ln(self.fc(x))))


def build_residual_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 2048,
    num_hidden: int = 4,
    dropout: float = 0.1,
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    """Build a Residual MLP backbone."""
    layers: list[nn.Module] = []

    # Input layer
    layers.append(nn.Linear(in_dim, hidden_dim))
    layers.append(nn.LayerNorm(hidden_dim))
    layers.append(nn.ReLU())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    # Residual Hidden layers
    for _ in range(num_hidden - 1):
        layers.append(ResidualBlock(hidden_dim, dropout))

    # Output layer
    layers.append(nn.Linear(hidden_dim, out_dim))
    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers)


class AdvancedBCPolicy(nn.Module):
    """Advanced Residual MLP policy for behavior cloning.
    
    Predicts the action vector directly using MSE/BCE.
    """
    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 14,
        hidden_dim: int = 2048,
        num_hidden_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = build_residual_mlp(
            in_dim=state_dim,
            out_dim=action_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden_layers,
            dropout=dropout,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class MDNPolicy(nn.Module):
    """Mixture Density Network (Gaussian Mixture) policy for IQL actor.
    
    Captures multi-modal action distributions for continuous actions (13 dims),
    and a Bernoulli distribution for the binary shoot action (1 dim).
    """
    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 14,
        num_components: int = 5,
        hidden_dim: int = 2048,
        num_hidden_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous_dim = action_dim - 1
        self.num_components = num_components

        self.backbone = build_residual_mlp(
            in_dim=state_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden_layers,
            dropout=dropout,
        )
        
        # Heads for the GMM (13 continuous dimensions)
        self.pi_head = nn.Linear(hidden_dim, num_components)
        self.mu_head = nn.Linear(hidden_dim, num_components * self.continuous_dim)
        self.log_std_head = nn.Linear(hidden_dim, num_components * self.continuous_dim)
        
        # Head for binary shoot action (1 dimension, outputting logit)
        self.shoot_logit_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute MDN parameters and shoot logit.
        
        Returns:
            pi_logits: [B, K]
            mu: [B, K, C]
            sigma: [B, K, C]
            shoot_logit: [B, 1]
        """
        h = self.backbone(state)
        
        pi_logits = self.pi_head(h)
        
        # View as [B, K, C]
        B = h.shape[0]
        mu = self.mu_head(h).view(B, self.num_components, self.continuous_dim)
        log_std = self.log_std_head(h).view(B, self.num_components, self.continuous_dim)
        
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_std)
        
        shoot_logit = self.shoot_logit_head(h)
        
        return pi_logits, mu, sigma, shoot_logit

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from the mixture and Bernoulli distribution.
        
        Returns:
            action: [B, 14] combined action array
            mdn_log_prob: log prob of the continuous actions
            shoot_logit: raw logit of the shoot action
        """
        pi_logits, mu, sigma, shoot_logit = self.forward(state)
        
        # Sample component k ~ Categorical(pi)
        pi_dist = torch.distributions.Categorical(logits=pi_logits)
        k = pi_dist.sample() # [B]
        
        # Gather mu and sigma for the chosen component
        # k: [B] -> [B, 1, C]
        k_expanded = k.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.continuous_dim)
        mu_k = torch.gather(mu, 1, k_expanded).squeeze(1) # [B, C]
        sigma_k = torch.gather(sigma, 1, k_expanded).squeeze(1) # [B, C]
        
        noise = torch.randn_like(mu_k)
        continuous_action = mu_k + noise * sigma_k
        
        # Compute continuous log prob
        mdn_log_prob = self._compute_log_prob(continuous_action, pi_logits, mu, sigma)
        
        # Sample shoot action from Bernoulli
        shoot_prob = torch.sigmoid(shoot_logit)
        shoot_action = torch.bernoulli(shoot_prob)
        
        # Combine actions: insert shoot_action at index 8
        B = continuous_action.shape[0]
        action = torch.zeros(B, self.action_dim, device=state.device)
        action[:, :8] = continuous_action[:, :8]
        action[:, 8:9] = shoot_action
        action[:, 9:] = continuous_action[:, 8:]
        
        return action, mdn_log_prob, shoot_logit

    def act(self, state: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Produce an action for inference."""
        pi_logits, mu, sigma, shoot_logit = self.forward(state)
        if deterministic:
            # Pick the component with the highest weight
            k = torch.argmax(pi_logits, dim=-1) # [B]
            k_expanded = k.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.continuous_dim)
            continuous_action = torch.gather(mu, 1, k_expanded).squeeze(1)
            
            shoot_action = (torch.sigmoid(shoot_logit) > 0.5).float()
            
            B = continuous_action.shape[0]
            action = torch.zeros(B, self.action_dim, device=state.device)
            action[:, :8] = continuous_action[:, :8]
            action[:, 8:9] = shoot_action
            action[:, 9:] = continuous_action[:, 8:]
            return action
        else:
            action, _, _ = self.sample(state)
            return action

    def log_prob(self, state: torch.Tensor, continuous_action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of a given continuous action under the MDN."""
        pi_logits, mu, sigma, _ = self.forward(state)
        return self._compute_log_prob(continuous_action, pi_logits, mu, sigma)
        
    def _compute_log_prob(
        self, action: torch.Tensor, pi_logits: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Helper to compute log \sum_k \pi_k N(a | \mu_k, \sigma_k)"""
        # action: [B, C] -> [B, 1, C]
        action_exp = action.unsqueeze(1)
        
        # log \pi_k: [B, K]
        pi_log_probs = F.log_softmax(pi_logits, dim=-1)
        
        # log N(a | \mu_k, \sigma_k) -> sum over C -> [B, K]
        # Normal log prob: -0.5 * (((x - mu)/sigma)^2 + 2*log(sigma) + log(2*pi))
        normal_log_probs = -0.5 * (((action_exp - mu) / sigma) ** 2 + 2 * torch.log(sigma) + math.log(2 * math.pi))
        component_log_probs = normal_log_probs.sum(dim=-1) # [B, K]
        
        # Total log prob: logsumexp_k( log(\pi_k) + log(N) )
        log_prob = torch.logsumexp(pi_log_probs + component_log_probs, dim=-1) # [B]
        return log_prob
