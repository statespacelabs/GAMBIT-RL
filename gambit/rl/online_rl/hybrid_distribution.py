"""Hybrid action distribution for PPO.

Combines 4 continuous dimensions (Normal with clamped sampling) and
4 binary dimensions (Independent Bernoulli).

Outputs an 8-dim action vector.
Provides joint log_prob and entropy.
"""

from __future__ import annotations

import torch
from torch.distributions import Normal, Bernoulli, Independent


class HybridDistribution:
    """Joint distribution over 4 continuous and 4 binary action dimensions.

    For continuous actions: uses Independent(Normal, 1).
    For binary actions: uses Independent(Bernoulli, 1).

    Sampling:
        Continuous outputs are clamped to [-1, 1] for Unity compatibility.
        NOTE: Unclamped actions are required for accurate log_prob calculation!
        We return both clamped (for env) and unclamped (for buffer) during sample.
    """

    def __init__(
        self,
        cont_mean: torch.Tensor,
        cont_logstd: torch.Tensor,
        binary_logits: torch.Tensor,
    ):
        """
        Args:
            cont_mean: [*, 4] mean of continuous actions.
            cont_logstd: [*, 4] log std of continuous actions.
            binary_logits: [*, 4] logits for binary actions.
        """
        self.cont_dist = Independent(Normal(cont_mean, cont_logstd.exp()), 1)
        self.binary_dist = Independent(Bernoulli(logits=binary_logits), 1)

    @property
    def cont_mean(self) -> torch.Tensor:
        return self.cont_dist.base_dist.loc

    @property
    def cont_logstd(self) -> torch.Tensor:
        return self.cont_dist.base_dist.scale.log()

    @property
    def binary_logits(self) -> torch.Tensor:
        return self.binary_dist.base_dist.logits

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample from the joint distribution.

        Returns:
            action_clamped: [*, 8] clamped continuous dims to [-1, 1], binary are 0/1.
                            (Use this for the environment)
            action_raw: [*, 8] unclamped continuous dims, binary are 0/1.
                        (Use this to store in the buffer and compute log_prob later)
        """
        cont_raw = self.cont_dist.sample()
        cont_clamped = torch.clamp(cont_raw, -1.0, 1.0)
        bin_sample = self.binary_dist.sample()

        action_clamped = torch.cat([cont_clamped, bin_sample], dim=-1)
        action_raw = torch.cat([cont_raw, bin_sample], dim=-1)

        return action_clamped, action_raw

    def log_prob(self, action_raw: torch.Tensor) -> torch.Tensor:
        """Compute the joint log probability of an action.

        Args:
            action_raw: [*, 8] unclamped action sampled from the distribution.

        Returns:
            log_p: [*] joint log probability.
        """
        cont_action = action_raw[..., :4]
        bin_action = action_raw[..., 4:]

        cont_logp = self.cont_dist.log_prob(cont_action)
        bin_logp = self.binary_dist.log_prob(bin_action)

        return cont_logp + bin_logp

    def entropy(self) -> torch.Tensor:
        """Compute the joint entropy.

        Returns:
            entropy: [*] joint entropy.
        """
        return self.cont_dist.entropy() + self.binary_dist.entropy()

    @property
    def mode(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the deterministic mode of the distribution."""
        cont_mean = self.cont_dist.base_dist.loc
        cont_clamped = torch.clamp(cont_mean, -1.0, 1.0)

        # Mode of Bernoulli: 1 if prob > 0.5 (logit > 0), else 0
        bin_mode = (self.binary_dist.base_dist.logits > 0).float()

        action_clamped = torch.cat([cont_clamped, bin_mode], dim=-1)
        action_raw = torch.cat([cont_mean, bin_mode], dim=-1)

        return action_clamped, action_raw
