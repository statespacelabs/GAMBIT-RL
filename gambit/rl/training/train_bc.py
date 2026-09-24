"""Behavior cloning training loop and loss computation."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gambit.dataset_preparation.configs import BCTrainConfig
from gambit.rl.datasets.transition_dataset import TransitionDataset
from gambit.rl.datasets.transition_collate import transition_collate_fn
from gambit.rl.models.policy import BCPolicy
from gambit.rl.utils.checkpointing import save_rl_checkpoint, load_rl_checkpoint
from gambit.rl.utils.metrics import compute_action_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def bc_loss(
    pred_action: torch.Tensor,
    true_action: torch.Tensor,
    continuous_indices: list[int],
    binary_indices: list[int],
    binary_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Compute mixed continuous/binary behavior cloning loss.

    Continuous action dimensions use MSE loss.
    Binary action dimensions use BCEWithLogitsLoss.

    Returns:
        total_loss: Scalar loss tensor.
        loss_metrics: Dictionary with per-type losses.
    """
    metrics: dict = {}

    total_loss = torch.tensor(0.0, device=pred_action.device)

    # Continuous loss (MSE)
    if continuous_indices:
        ci = torch.tensor(continuous_indices, device=pred_action.device, dtype=torch.long)
        pred_cont = pred_action[:, ci]
        true_cont = true_action[:, ci]
        cont_loss = F.mse_loss(pred_cont, true_cont)
        total_loss = total_loss + cont_loss
        metrics["loss_continuous"] = cont_loss.item()
        metrics["mae_continuous"] = (pred_cont - true_cont).abs().mean().item()

    # Binary loss (BCEWithLogitsLoss)
    if binary_indices:
        bi = torch.tensor(binary_indices, device=pred_action.device, dtype=torch.long)
        pred_bin = pred_action[:, bi]
        true_bin = true_action[:, bi]
        bin_loss = F.binary_cross_entropy_with_logits(pred_bin, true_bin)
        total_loss = total_loss + binary_weight * bin_loss
        metrics["loss_binary"] = bin_loss.item()

        # Binary accuracy
        with torch.no_grad():
            pred_labels = (pred_bin > 0).float()
            metrics["binary_accuracy"] = (pred_labels == true_bin).float().mean().item()

    metrics["loss_total"] = total_loss.item()
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_bc(config: BCTrainConfig) -> None:
    """Train behavior cloning policy on Phase 2 transitions.

    Training objective: state_t → action_t

    Saves best validation checkpoint, final checkpoint, and JSONL logs.
    """
    # Seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device(config.device)

    # Directories
    ckpt_dir = Path(config.checkpoint_dir)
    log_dir = Path(config.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "train_bc.jsonl"

    # Datasets
    train_ds = TransitionDataset(
        transition_manifest_path=config.transition_manifest_path,
        split="train",
        normalize_state=config.normalize_state,
        state_scaler_path=config.state_scaler_path or None,
        normalize_action=config.normalize_action,
        action_scaler_path=config.action_scaler_path or None,
        normalize_reward=config.normalize_reward,
        reward_stats_path=config.reward_stats_path or None,
    )

    val_ds = TransitionDataset(
        transition_manifest_path=config.transition_manifest_path,
        split="val",
        normalize_state=config.normalize_state,
        state_scaler_path=config.state_scaler_path or None,
        normalize_action=config.normalize_action,
        action_scaler_path=config.action_scaler_path or None,
        normalize_reward=config.normalize_reward,
        reward_stats_path=config.reward_stats_path or None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=transition_collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=transition_collate_fn,
        pin_memory=device.type == "cuda",
    )

    # Model
    policy = BCPolicy(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    continuous_indices = list(config.continuous_indices)
    binary_indices = list(config.binary_indices)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    logger.info("Starting BC training: %d train, %d val transitions", len(train_ds), len(val_ds))
    logger.info("Policy parameters: %d", sum(p.numel() for p in policy.parameters()))

    for epoch in range(1, config.epochs + 1):
        policy.train()
        epoch_losses: list[float] = []
        t_epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)

            pred = policy(batch.state)
            loss, metrics = bc_loss(
                pred, batch.action,
                continuous_indices, binary_indices,
                binary_weight=config.binary_loss_weight,
            )

            optimizer.zero_grad()
            loss.backward()

            if config.grad_clip_norm:
                nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip_norm)

            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1

            # Log periodically
            if global_step % 100 == 0:
                _log_metrics(log_path, {
                    "step": global_step,
                    "epoch": epoch,
                    "phase": "train",
                    **metrics,
                })

            # Limit steps per epoch if configured
            if config.steps_per_epoch and batch_idx + 1 >= config.steps_per_epoch:
                break

        epoch_time = time.time() - t_epoch_start
        avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        logger.info(
            "Epoch %d: train_loss=%.5f, time=%.1fs, steps=%d",
            epoch, avg_train_loss, epoch_time, global_step,
        )

        # Validation
        if epoch % config.val_every_epochs == 0:
            from .eval_bc import evaluate_bc

            val_metrics = evaluate_bc(
                policy, val_loader,
                continuous_indices, binary_indices,
                device,
            )
            val_loss = val_metrics["loss_total"]

            logger.info("Epoch %d: val_loss=%.5f", epoch, val_loss)

            _log_metrics(log_path, {
                "step": global_step,
                "epoch": epoch,
                "phase": "val",
                **val_metrics,
            })

            # Best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_rl_checkpoint(
                    path=ckpt_dir / "best.pt",
                    model_state=policy.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    config=config.to_dict(),
                    step=global_step,
                    metrics=val_metrics,
                )
                logger.info("New best val_loss=%.5f saved", val_loss)
            else:
                patience_counter += 1
                if config.patience and patience_counter >= config.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

    # Final checkpoint
    save_rl_checkpoint(
        path=ckpt_dir / "final.pt",
        model_state=policy.state_dict(),
        optimizer_state=optimizer.state_dict(),
        config=config.to_dict(),
        step=global_step,
        metrics={"train_loss": avg_train_loss},
    )
    logger.info("BC training complete. Global step=%d", global_step)


def _log_metrics(log_path: Path, metrics: dict) -> None:
    """Append metrics as a JSON line."""
    with open(log_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Train Behavior Cloning Policy")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    
    config = BCTrainConfig.from_yaml(args.config)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    train_bc(config)
