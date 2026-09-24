"""Recurrent PPO Actor-Critic.

Uses the pretrained TelemetryEncoder, a GRU, and heads for hybrid action
distribution and value estimation. Includes weight-splitting initialization
from the feedforward CoreActionProjector.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from .telemetry_encoder import TelemetryEncoder
from .hybrid_distribution import HybridDistribution
from .core_action_projector import CoreActionProjector


class RecurrentActorCritic(nn.Module):
    """GRU-based actor-critic for online PPO.

    Architecture:
        obs_45 → TelemetryEncoder → z_tel [B,T,512]
               → GRU(512, hidden=512) → h_t [B,T,512]
               → actor_cont_mean [B,T,4]
               → actor_cont_logstd (parameter) [4]
               → actor_binary_logits [B,T,4]
               → value_head [B,T,1]
    """

    def __init__(self, tel_encoder: TelemetryEncoder):
        super().__init__()
        self.tel_encoder = tel_encoder

        # Temporal model
        self.gru = nn.GRU(
            input_size=512, hidden_size=512, num_layers=1, batch_first=True
        )

        # Actor heads
        self.actor_cont_mean = nn.Linear(512, 4)
        # Conservative exploration initial logstd ~ 0.37
        self.actor_cont_logstd = nn.Parameter(torch.full((4,), -1.0))
        self.actor_binary_logits = nn.Linear(512, 4)

        # Value head
        self.value_head = nn.Linear(512, 1)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize GRU hidden state.

        Returns:
            hidden: [1, B, 512] zero tensor.
        """
        return torch.zeros(1, batch_size, 512, device=device)

    def set_ppo_mode(self, training: bool = True) -> "RecurrentActorCritic":
        """Set PPO mode while keeping pretrained dropout deterministic."""
        self.train(training)
        for module in (
            self.tel_encoder.where_branch,
            self.tel_encoder.view_branch,
            self.tel_encoder.rhythm_branch,
            self.tel_encoder.telemetry_fusion,
        ):
            module.eval()
        return self

    def forward(
        self,
        obs_seq: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[HybridDistribution, torch.Tensor, torch.Tensor]:
        """Forward pass to get distributions and values.

        Args:
            obs_seq: [B, T, 45] telemetry observation.
            hidden: [1, B, 512] GRU hidden state. If None, initialized to zero.

        Returns:
            dist: HybridDistribution over 8-dim actions.
            value: [B, T] state value estimate.
            new_hidden: [1, B, 512] updated GRU hidden state.
        """
        B, T, _ = obs_seq.shape
        if hidden is None:
            hidden = self.init_hidden(B, obs_seq.device)

        z = self.tel_encoder(obs_seq)  # [B, T, 512]
        h, new_hidden = self.gru(z, hidden)  # h: [B, T, 512]

        cont_mean = self.actor_cont_mean(h)  # [B, T, 4]
        # Expand logstd to match batch/time dims
        cont_logstd = self.actor_cont_logstd.expand_as(cont_mean)
        bin_logits = self.actor_binary_logits(h)  # [B, T, 4]

        dist = HybridDistribution(cont_mean, cont_logstd, bin_logits)
        value = self.value_head(h).squeeze(-1)  # [B, T]

        return dist, value, new_hidden

    def get_value(
        self, obs_seq: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Get only the value estimate (useful for GAE bootstrapping)."""
        _, value, _ = self.forward(obs_seq, hidden)
        return value

    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        actions_raw: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities and entropies of given actions.

        Args:
            obs_seq: [B, T, 45]
            actions_raw: [B, T, 8] unclamped actions.
            hidden: [1, B, 512]

        Returns:
            log_prob: [B, T]
            entropy: [B, T]
            value: [B, T]
        """
        dist, value, _ = self.forward(obs_seq, hidden)
        log_prob = dist.log_prob(actions_raw)
        entropy = dist.entropy()
        return log_prob, entropy, value

    def predict_actions(
        self,
        obs_seq: torch.Tensor,
        hidden: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Deterministic prediction for distillation / evaluation.

        Returns the raw continuous mean for dims [0:4]. For the binary dims [4:8]:
          - return_logits=True  → raw logits (use with BCEWithLogitsLoss in training).
          - return_logits=False → sigmoid probabilities (for inspection/eval).

        The recurrent BC loss uses return_logits=True to avoid a double sigmoid.
        """
        dist, _, _ = self.forward(obs_seq, hidden)
        cont_mean = dist.cont_dist.base_dist.loc
        bin_logits = dist.binary_dist.base_dist.logits
        if return_logits:
            return torch.cat([cont_mean, bin_logits], dim=-1)
        bin_probs = torch.sigmoid(bin_logits)
        return torch.cat([cont_mean, bin_probs], dim=-1)

    @classmethod
    def from_distilled(cls, projector: CoreActionProjector) -> "RecurrentActorCritic":
        """Initialize from a trained CoreActionProjector.

        Copies TelemetryEncoder verbatim.
        Splits action_head [512→8] into actor_cont_mean [512→4] and actor_binary_logits [512→4].
        Initializes GRU and value_head randomly.
        """
        # Copy the encoder so later policy updates cannot mutate the projector.
        instance = cls(tel_encoder=deepcopy(projector.tel_encoder))

        # Split weights
        with torch.no_grad():
            action_w = projector.action_head.weight  # [8, 512]
            action_b = projector.action_head.bias  # [8]

            # Continuous mean (first 4 dims)
            instance.actor_cont_mean.weight.copy_(action_w[:4, :])
            instance.actor_cont_mean.bias.copy_(action_b[:4])

            # Binary logits (last 4 dims)
            instance.actor_binary_logits.weight.copy_(action_w[4:, :])
            instance.actor_binary_logits.bias.copy_(action_b[4:])

            # logstd remains -1.0

        return instance
