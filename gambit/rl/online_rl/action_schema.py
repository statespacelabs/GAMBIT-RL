"""Canonical Phase 3 action schema and Phase 2 command conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CORE_ACTION_NAMES = (
    "move_x",
    "move_y",
    "look_dx",
    "look_dy",
    "shoot",
    "reload",
    "jump",
    "crouch",
)

# Indices in the Phase 2 14-dimensional summary schema.
CORE_ACTION_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]


@dataclass(frozen=True)
class ActionCommandConverter:
    """Convert Phase 2 physical summaries into normalized Unity commands."""

    max_move_speed: float = 6.0
    max_yaw_delta: float = 180.0
    max_pitch_delta: float = 180.0

    def __post_init__(self) -> None:
        for name, value in (
            ("max_move_speed", self.max_move_speed),
            ("max_yaw_delta", self.max_yaw_delta),
            ("max_pitch_delta", self.max_pitch_delta),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}")

    def phase2_action14_to_command8(self, action_14: np.ndarray) -> np.ndarray:
        """Return bounded move/look commands followed by exact binary commands."""
        action_14 = np.asarray(action_14, dtype=np.float32)
        if action_14.shape != (14,):
            raise ValueError(f"Expected action_14 shape (14,), got {action_14.shape}")
        if not np.isfinite(action_14).all():
            raise ValueError("Phase 2 action contains NaN or Inf")

        continuous = np.asarray(
            [
                action_14[0] / self.max_move_speed,
                action_14[1] / self.max_move_speed,
                action_14[2] / self.max_yaw_delta,
                action_14[3] / self.max_pitch_delta,
            ],
            dtype=np.float32,
        )
        continuous = np.clip(continuous, -1.0, 1.0)
        binary = (action_14[8:12] > 0.5).astype(np.float32)
        command = np.concatenate([continuous, binary]).astype(np.float32)
        self.validate_command8(command)
        return command

    @staticmethod
    def validate_command8(command_8: np.ndarray) -> None:
        command_8 = np.asarray(command_8)
        if command_8.shape != (8,):
            raise ValueError(f"Expected command_8 shape (8,), got {command_8.shape}")
        if not np.isfinite(command_8).all():
            raise ValueError("Command action contains NaN or Inf")
        if np.any(command_8[:4] < -1.0) or np.any(command_8[:4] > 1.0):
            raise ValueError(f"Continuous commands outside [-1, 1]: {command_8[:4]}")
        if not np.isin(command_8[4:], (0.0, 1.0)).all():
            raise ValueError(f"Binary commands are not 0/1: {command_8[4:]}")
