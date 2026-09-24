"""Map-style dataset and batch container for Phase 2 offline RL transitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import pandas as pd


@dataclass
class TransitionBatch:
    """Batch object for offline RL training.

    Fields:
        state:      Tensor [B, state_dim].
        action:     Tensor [B, action_dim].
        reward:     Tensor [B].
        next_state: Tensor [B, state_dim].
        done:       Tensor [B], float 0/1.
        metadata:   Optional per-sample metadata for debugging/evaluation.
    """

    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor
    metadata: Optional[list[dict]] = None

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "TransitionBatch":
        """Move tensors to device while preserving metadata."""
        return TransitionBatch(
            state=self.state.to(device, non_blocking=non_blocking),
            action=self.action.to(device, non_blocking=non_blocking),
            reward=self.reward.to(device, non_blocking=non_blocking),
            next_state=self.next_state.to(device, non_blocking=non_blocking),
            done=self.done.to(device, non_blocking=non_blocking),
            metadata=self.metadata,
        )

    def __len__(self) -> int:
        return self.state.shape[0]


class TransitionDataset(Dataset):
    """Map-style dataset for Phase 2 offline RL transitions.

    Loads rows from phase2_transition_manifest.csv and returns state, action,
    reward, next_state, done, and metadata.

    The dataset loads z_raw from latent .npz files. It does not load
    videos or full telemetry during BC/IQL training.
    """

    def __init__(
        self,
        transition_manifest_path: str | Path,
        split: str,
        state_key: str = "z_raw",
        normalize_state: bool = False,
        state_scaler_path: str | Path | None = None,
        normalize_action: bool = False,
        action_scaler_path: str | Path | None = None,
        normalize_reward: bool = False,
        reward_stats_path: str | Path | None = None,
    ):
        """Load transition manifest and optional normalization statistics.

        Only rows matching split are included.
        """
        self.state_key = state_key

        df = pd.read_csv(transition_manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No transitions found for split='{split}' in {transition_manifest_path}"
            )

        # State normalization
        self.normalize_state = normalize_state
        self.state_mean: np.ndarray | None = None
        self.state_std: np.ndarray | None = None
        if normalize_state and state_scaler_path:
            with np.load(state_scaler_path) as sc:
                self.state_mean = sc["mean"].astype(np.float32)
                self.state_std = np.maximum(sc["std"].astype(np.float32), 1e-6)

        # Action normalization
        self.normalize_action = normalize_action
        self.action_mean: np.ndarray | None = None
        self.action_std: np.ndarray | None = None
        self.action_normalize_mask: np.ndarray | None = None
        if normalize_action and action_scaler_path:
            with np.load(action_scaler_path) as sc:
                self.action_mean = sc["mean"].astype(np.float32)
                self.action_std = np.maximum(sc["std"].astype(np.float32), 1e-6)
                if "normalize_mask" in sc:
                    self.action_normalize_mask = sc["normalize_mask"].astype(np.float32)

        # Reward normalization
        self.normalize_reward = normalize_reward
        self.reward_mean: float = 0.0
        self.reward_std: float = 1.0
        if normalize_reward and reward_stats_path:
            with open(reward_stats_path, "r") as f:
                stats = json.load(f)
            self.reward_mean = float(stats["mean"])
            self.reward_std = max(float(stats["std"]), 1e-6)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """Return one transition sample.

        Loads state z_raw, next_state z_raw, action_vector, reward, done flag.

        Returns:
            Dict ready for transition_collate_fn.
        """
        row = self.df.iloc[idx]

        # Load state latent
        with np.load(row["state_latent_path"]) as data:
            state = data[self.state_key].astype(np.float32)

        # Load next_state latent
        with np.load(row["next_state_latent_path"]) as data:
            next_state = data[self.state_key].astype(np.float32)

        # Load action
        with np.load(row["action_path"]) as data:
            action = data["action_vector"].astype(np.float32)

        reward = float(row["reward"])
        done = float(row["done"])

        # Apply normalization
        if self.normalize_state and self.state_mean is not None:
            state = (state - self.state_mean) / self.state_std
            next_state = (next_state - self.state_mean) / self.state_std

        if self.normalize_action and self.action_mean is not None:
            if self.action_normalize_mask is not None:
                # Only normalize continuous dims
                mask = self.action_normalize_mask > 0
                action[mask] = (
                    (action[mask] - self.action_mean[mask]) / self.action_std[mask]
                )
            else:
                action = (action - self.action_mean) / self.action_std

        if self.normalize_reward:
            reward = (reward - self.reward_mean) / self.reward_std

        return {
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "next_state": torch.from_numpy(next_state),
            "done": torch.tensor(done, dtype=torch.float32),
            "metadata": {
                "transition_id": str(row["transition_id"]),
                "session_id": str(row["session_id"]),
                "t": float(row["t"]),
            },
        }
