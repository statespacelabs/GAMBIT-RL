"""Behavior cloning evaluation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from gambit.rl.models.policy import BCPolicy
from gambit.rl.utils.metrics import compute_action_metrics
from .train_bc import bc_loss


@torch.no_grad()
def evaluate_bc(
    policy: BCPolicy,
    dataloader: DataLoader,
    continuous_indices: list[int],
    binary_indices: list[int],
    device: torch.device,
) -> dict:
    """Evaluate BC policy on validation or test split.

    Reports action prediction losses and binary action metrics.

    Returns:
        Dictionary of evaluation metrics.
    """
    policy.eval()

    total_loss = 0.0
    total_metrics: dict[str, float] = {}
    n_batches = 0

    all_pred: list[torch.Tensor] = []
    all_true: list[torch.Tensor] = []

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)

        pred = policy(batch.state)
        loss, metrics = bc_loss(
            pred, batch.action,
            continuous_indices, binary_indices,
        )

        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v

        all_pred.append(pred.cpu())
        all_true.append(batch.action.cpu())
        n_batches += 1

    if n_batches == 0:
        return {"loss_total": float("inf")}

    # Average metrics
    avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}
    avg_metrics["loss_total"] = total_loss / n_batches

    # Compute detailed action metrics on concatenated predictions
    all_pred_t = torch.cat(all_pred, dim=0)
    all_true_t = torch.cat(all_true, dim=0)

    action_metrics = compute_action_metrics(
        all_pred_t, all_true_t,
        continuous_indices, binary_indices,
    )
    avg_metrics.update(action_metrics)

    policy.train()
    return avg_metrics
