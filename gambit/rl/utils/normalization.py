"""Normalization utilities for offline RL state/action/reward standardization."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class RunningStandardizer:
    """Stores mean/std statistics and applies standard normalization.

    Used for states, continuous action dimensions, and rewards.
    """

    def __init__(
        self,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        eps: float = 1e-6,
    ):
        self.mean = mean
        self.std = std
        self.eps = eps

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize x using stored mean and std."""
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean) / (self.std + self.eps)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Inverse of normalize."""
        if self.mean is None or self.std is None:
            return x
        return x * (self.std + self.eps) + self.mean

    def save(self, path: str | Path) -> None:
        """Save statistics to .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean,
            std=self.std,
        )

    @classmethod
    def load(cls, path: str | Path, eps: float = 1e-6) -> "RunningStandardizer":
        """Load statistics from .npz."""
        with np.load(path) as data:
            return cls(
                mean=data["mean"].astype(np.float32),
                std=data["std"].astype(np.float32),
                eps=eps,
            )


def fit_npz_vector_stats(
    paths: list[str | Path],
    key: str,
    output_path: str | Path,
) -> dict:
    """Compute mean/std/min/max over vectors stored inside .npz files.

    Used for fitting state or action normalization statistics.

    Returns:
        Dictionary with mean, std, min, max arrays.
    """
    vectors: list[np.ndarray] = []

    for p in paths:
        with np.load(p) as data:
            if key in data:
                vectors.append(data[key].astype(np.float64))

    if not vectors:
        raise ValueError(f"No vectors found with key '{key}' in provided paths.")

    all_data = np.stack(vectors, axis=0)  # [N, dim]

    mean = np.mean(all_data, axis=0).astype(np.float32)
    std = np.std(all_data, axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    min_val = np.min(all_data, axis=0).astype(np.float32)
    max_val = np.max(all_data, axis=0).astype(np.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        mean=mean,
        std=std,
        min=min_val,
        max=max_val,
    )

    return {
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "count": len(vectors),
    }


def normalize_with_stats(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize x using precomputed mean and std."""
    return (x - mean) / (std + eps)
