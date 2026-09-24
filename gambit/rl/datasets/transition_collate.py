"""Collate function for offline RL transition samples."""

from __future__ import annotations

import torch

from .transition_dataset import TransitionBatch


def transition_collate_fn(samples: list[dict]) -> TransitionBatch:
    """Collate transition samples into a TransitionBatch.

    Stacks state, action, reward, next_state, done tensors.
    Preserves metadata as a list of dictionaries.
    """
    return TransitionBatch(
        state=torch.stack([s["state"] for s in samples], dim=0),
        action=torch.stack([s["action"] for s in samples], dim=0),
        reward=torch.stack([s["reward"] for s in samples], dim=0),
        next_state=torch.stack([s["next_state"] for s in samples], dim=0),
        done=torch.stack([s["done"] for s in samples], dim=0),
        metadata=[s["metadata"] for s in samples],
    )
