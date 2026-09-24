"""Learning-rate scheduling helpers for PPO retention runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gambit.dataset_preparation.configs import PPOTrainConfig


def get_scheduled_lr(iter_idx: int, config: "PPOTrainConfig") -> float:
    """Return LR for ``iter_idx`` (1-based training iteration)."""
    if config.lr_schedule != "piecewise":
        return config.lr

    if iter_idx >= config.lr_milestone_3_iter:
        return config.lr_milestone_3_value
    if iter_idx >= config.lr_milestone_2_iter:
        return config.lr_milestone_2_value
    if iter_idx >= config.lr_milestone_1_iter:
        return config.lr_milestone_1_value
    return config.lr_initial
