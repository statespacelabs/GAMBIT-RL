"""RL checkpoint save/load utilities."""

from __future__ import annotations

from pathlib import Path

import torch


def save_rl_checkpoint(
    path: str | Path,
    model_state: dict,
    optimizer_state: dict | None,
    config: dict,
    step: int,
    metrics: dict,
) -> None:
    """Save BC or IQL checkpoint.

    Checkpoint includes model state, optimizer state, config, training step,
    and metrics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model_state,
        "config": config,
        "step": step,
        "metrics": metrics,
    }
    if optimizer_state is not None:
        checkpoint["optimizer"] = optimizer_state

    torch.save(checkpoint, path)


def load_rl_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> dict:
    """Load an RL checkpoint and map tensors to the requested device.

    Returns the full checkpoint dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(path, map_location=device, weights_only=False)
