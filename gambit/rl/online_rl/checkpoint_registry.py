"""Checkpoint Registry for PPO Self-Play.

Manages persistent JSON registry of all frozen PPO checkpoints.
Stores metadata, paths, and evaluations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Metadata for a registered checkpoint."""

    id: str
    step: int
    path: str
    rating: float
    created_at: str
    eval_win_rate: float
    eval_damage_ratio: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointInfo":
        return cls(**data)


class CheckpointRegistry:
    """Persistent registry of PPO checkpoints.

    Copies policies to the registry directory and stores metadata in a JSON manifest.
    """

    def __init__(self, registry_dir: str | Path):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.registry_dir / "manifest.json"

        self.checkpoints: dict[str, CheckpointInfo] = {}
        self.load()

    def register(self, step: int, metrics: dict[str, float], state_dict: dict) -> str:
        """Register a new checkpoint policy.

        Args:
            step: Training iteration or step.
            metrics: Evaluation metrics (win_rate, damage_ratio).
            state_dict: The policy state dictionary to save.

        Returns:
            The generated checkpoint ID.
        """
        import datetime

        now = datetime.datetime.utcnow().isoformat()

        ckpt_id = f"ckpt_{step:08d}"
        file_path = self.registry_dir / f"{ckpt_id}.pt"

        # Save weights
        torch.save(state_dict, file_path)

        info = CheckpointInfo(
            id=ckpt_id,
            step=step,
            path=str(file_path.absolute()),
            rating=1500.0,  # Base Elo rating, to be updated by Puppeteer
            created_at=now,
            eval_win_rate=metrics.get("win_rate", 0.0),
            eval_damage_ratio=metrics.get("damage_ratio", 0.0),
        )

        self.checkpoints[ckpt_id] = info
        self.save()

        logger.info("Registered new checkpoint: %s", ckpt_id)
        return ckpt_id

    def list_all(self) -> list[CheckpointInfo]:
        """List all registered checkpoints sorted by step."""
        return sorted(list(self.checkpoints.values()), key=lambda c: c.step)

    def load_policy_state(self, checkpoint_id: str) -> dict:
        """Load the state dict of a checkpoint."""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found in registry.")

        info = self.checkpoints[checkpoint_id]
        path = Path(info.path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file {path} not found.")

        return torch.load(path, map_location="cpu", weights_only=False)

    def load(self) -> None:
        """Load registry manifest from disk."""
        if not self.manifest_path.exists():
            return

        with open(self.manifest_path, "r") as f:
            data = json.load(f)

        self.checkpoints = {k: CheckpointInfo.from_dict(v) for k, v in data.items()}

    def save(self) -> None:
        """Save registry manifest to disk."""
        data = {k: asdict(v) for k, v in self.checkpoints.items()}
        with open(self.manifest_path, "w") as f:
            json.dump(data, f, indent=2)
