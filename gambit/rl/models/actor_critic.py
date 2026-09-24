"""IQL network container and factory."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch.nn as nn

from .policy import GaussianPolicy
from .q_network import QNetwork
from .v_network import VNetwork


@dataclass
class IQLNetworks:
    """Container for all IQL networks.

    Fields:
        actor:     GaussianPolicy.
        q1:        First QNetwork.
        q2:        Second QNetwork.
        v:         VNetwork.
        target_q1: Target copy of q1.
        target_q2: Target copy of q2.
    """

    actor: GaussianPolicy
    q1: QNetwork
    q2: QNetwork
    v: VNetwork
    target_q1: QNetwork
    target_q2: QNetwork

    def to(self, device, **kwargs) -> "IQLNetworks":
        """Move all networks to device."""
        self.actor = self.actor.to(device, **kwargs)
        self.q1 = self.q1.to(device, **kwargs)
        self.q2 = self.q2.to(device, **kwargs)
        self.v = self.v.to(device, **kwargs)
        self.target_q1 = self.target_q1.to(device, **kwargs)
        self.target_q2 = self.target_q2.to(device, **kwargs)
        return self

    def train(self) -> "IQLNetworks":
        """Set actor, q1, q2, v to train mode. Targets stay in eval."""
        self.actor.train()
        self.q1.train()
        self.q2.train()
        self.v.train()
        self.target_q1.eval()
        self.target_q2.eval()
        return self

    def eval(self) -> "IQLNetworks":
        """Set all networks to eval mode."""
        self.actor.eval()
        self.q1.eval()
        self.q2.eval()
        self.v.eval()
        self.target_q1.eval()
        self.target_q2.eval()
        return self


def build_iql_networks(
    state_dim: int = 512,
    action_dim: int = 14,
    hidden_dim: int = 1024,
    num_hidden_layers: int = 3,
    dropout: float = 0.1,
) -> IQLNetworks:
    """Build actor, twin Q networks, and V network for IQL.

    Uses MLPs first. Do not add recurrent/transformer policy logic until the
    latent-state version is working.
    """
    actor = GaussianPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
    )

    q1 = QNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
    )

    q2 = QNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
    )

    v = VNetwork(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
    )

    # Target networks are deep copies with frozen gradients
    target_q1 = copy.deepcopy(q1)
    target_q2 = copy.deepcopy(q2)
    for p in target_q1.parameters():
        p.requires_grad_(False)
    for p in target_q2.parameters():
        p.requires_grad_(False)

    return IQLNetworks(
        actor=actor,
        q1=q1,
        q2=q2,
        v=v,
        target_q1=target_q1,
        target_q2=target_q2,
    )
