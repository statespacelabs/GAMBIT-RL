"""Opponent Pool for PPO Self-Play.

Manages the sampling of opponents (frozen checkpoints or scripted bots)
for the self-play training loop.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Optional

from .checkpoint_registry import CheckpointRegistry

logger = logging.getLogger(__name__)


class OpponentPool:
    """Manages frozen checkpoint opponents for self-play."""

    def __init__(self, registry: CheckpointRegistry):
        self.registry = registry

    def sample_opponent(
        self, current_rating: float = 1500.0
    ) -> tuple[Optional[str], Optional[dict]]:
        """Sample an opponent for the next rollout.

        Sampling strategy:
            10%: Unity scripted bot (returns None, None)
            20%: Random checkpoint from pool
            70%: Checkpoint near current learner rating

        Returns:
            (checkpoint_id, policy_state_dict)
            If checkpoint_id is None, the runner should use the scripted bot.
        """
        checkpoints = self.registry.list_all()

        if not checkpoints:
            logger.info("OpponentPool empty. Using scripted bot.")
            return None, None

        p = np.random.random()

        if p < 0.10:
            # Scripted bot
            return None, None

        elif p < 0.30 or len(checkpoints) < 3:
            # Random checkpoint
            idx = np.random.randint(len(checkpoints))
            chosen = checkpoints[idx]

        else:
            # Checkpoint near current rating
            # We don't have true Elo yet (needs Puppeteer), so we sample from the
            # most recent 20% of checkpoints or use basic distance heuristic.

            # Simple heuristic: sort by absolute difference in rating
            # Since ratings might all be 1500 initially, we add some noise
            # or just prefer recent ones if ratings are identical.

            ratings = np.array([c.rating for c in checkpoints])
            diffs = np.abs(ratings - current_rating)

            # Convert to probabilities (closer = higher prob)
            # Use softmax over negative differences
            temperature = 100.0  # Elo points scale
            logits = -diffs / temperature
            probs = np.exp(logits) / np.sum(np.exp(logits))

            idx = np.random.choice(len(checkpoints), p=probs)
            chosen = checkpoints[idx]

        state_dict = self.registry.load_policy_state(chosen.id)
        return chosen.id, state_dict

    def update_rating(self, checkpoint_id: str, new_rating: float) -> None:
        """Update the Elo rating of a checkpoint in the registry."""
        if checkpoint_id in self.registry.checkpoints:
            self.registry.checkpoints[checkpoint_id].rating = new_rating
            self.registry.save()
