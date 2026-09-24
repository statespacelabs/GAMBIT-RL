"""Implicit Q-Learning (IQL) trainer for Phase 2 offline RL.

Assumes the encoder is frozen and states are cached z_raw vectors.
Does not load video during training.
"""

from __future__ import annotations

import copy
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gambit.dataset_preparation.configs import IQLTrainConfig
from gambit.rl.datasets.transition_dataset import TransitionDataset, TransitionBatch
from gambit.rl.datasets.transition_collate import transition_collate_fn
from gambit.rl.models.actor_critic import IQLNetworks, build_iql_networks
from gambit.rl.models.policy import BCPolicy
from gambit.rl.utils.checkpointing import save_rl_checkpoint, load_rl_checkpoint
from gambit.rl.utils.metrics import summarize_iql_values

logger = logging.getLogger(__name__)


class IQLTrainer:
    """Trainer for Implicit Q-Learning on Phase 2 transition data.

    This trainer assumes the encoder is frozen and states are cached z_raw
    vectors. It does not load video.

    IQL trains:
        V(s) using expectile regression
        Q(s,a) using Bellman backup
        actor using advantage-weighted behavior cloning
    """

    def __init__(self, config: IQLTrainConfig):
        """Initialize datasets, networks, optimizers, scalers, and logging.

        If config.init_actor_from_bc is set, load BC weights into the actor before
        IQL training begins.
        """
        self.config = config
        self.device = torch.device(config.device)

        # Seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Directories
        self.ckpt_dir = Path(config.checkpoint_dir)
        self.log_dir = Path(config.log_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "train_iql.jsonl"

        # Datasets
        self.train_ds = TransitionDataset(
            transition_manifest_path=config.transition_manifest_path,
            split="train",
            normalize_state=config.normalize_state,
            state_scaler_path=config.state_scaler_path or None,
            normalize_action=config.normalize_action,
            action_scaler_path=config.action_scaler_path or None,
            normalize_reward=config.normalize_reward,
            reward_stats_path=config.reward_stats_path or None,
        )

        self.val_ds = TransitionDataset(
            transition_manifest_path=config.transition_manifest_path,
            split="val",
            normalize_state=config.normalize_state,
            state_scaler_path=config.state_scaler_path or None,
            normalize_action=config.normalize_action,
            action_scaler_path=config.action_scaler_path or None,
            normalize_reward=config.normalize_reward,
            reward_stats_path=config.reward_stats_path or None,
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=transition_collate_fn,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=transition_collate_fn,
            pin_memory=self.device.type == "cuda",
        )

        # Networks
        self.networks = build_iql_networks(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            dropout=config.dropout,
        ).to(self.device)

        # Optionally initialize actor from BC checkpoint
        if config.init_actor_from_bc:
            self._init_actor_from_bc(config.init_actor_from_bc)

        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(
            self.networks.actor.parameters(),
            lr=config.actor_lr,
            weight_decay=config.weight_decay,
        )
        self.q_optimizer = torch.optim.AdamW(
            list(self.networks.q1.parameters()) + list(self.networks.q2.parameters()),
            lr=config.critic_lr,
            weight_decay=config.weight_decay,
        )
        self.v_optimizer = torch.optim.AdamW(
            self.networks.v.parameters(),
            lr=config.value_lr,
            weight_decay=config.weight_decay,
        )

        # IQL hyperparams
        self.discount = config.discount
        self.expectile = config.expectile
        self.temperature = config.temperature
        self.max_adv_weight = config.max_adv_weight
        self.target_update_rate = config.target_update_rate

        logger.info(
            "IQL Trainer: %d train, %d val transitions",
            len(self.train_ds), len(self.val_ds),
        )
        total_params = sum(
            sum(p.numel() for p in net.parameters())
            for net in [
                self.networks.actor, self.networks.q1,
                self.networks.q2, self.networks.v,
            ]
        )
        logger.info("Total trainable parameters: %d", total_params)

    def _init_actor_from_bc(self, bc_checkpoint_path: str) -> None:
        """Load BC policy weights into the IQL actor."""
        ckpt = load_rl_checkpoint(bc_checkpoint_path, self.device)
        bc_state = ckpt.get("model", ckpt)

        # BC policy has a single 'net' sequential; IQL actor has 'backbone' + heads.
        # Try direct loading first (if BC was a GaussianPolicy), then fall back to
        # mapping BCPolicy.net weights into the actor backbone.
        try:
            self.networks.actor.load_state_dict(bc_state, strict=False)
            logger.info("Loaded BC weights into IQL actor (direct)")
        except RuntimeError:
            # Map BCPolicy.net → GaussianPolicy.backbone (best effort)
            mapped = {}
            for k, v in bc_state.items():
                if k.startswith("net."):
                    new_key = k.replace("net.", "backbone.", 1)
                    mapped[new_key] = v
            self.networks.actor.load_state_dict(mapped, strict=False)
            logger.info("Loaded BC weights into IQL actor (mapped)")

    def train(self) -> None:
        """Run the IQL training loop.

        Saves periodic checkpoints and best validation checkpoint.
        """
        self.networks.train()
        train_iter = iter(self.train_loader)
        best_val_loss = float("inf")
        t_start = time.time()

        for step in range(1, self.config.total_steps + 1):
            # Get batch (cycle through dataloader)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            batch = batch.to(self.device, non_blocking=True)
            metrics = self.train_step(batch)

            # Logging
            if step % self.config.log_every_steps == 0:
                elapsed = time.time() - t_start
                metrics["step"] = step
                metrics["elapsed_s"] = elapsed
                metrics["steps_per_sec"] = step / elapsed
                self._log_metrics(metrics)

                logger.info(
                    "Step %d: v_loss=%.4f q_loss=%.4f actor_loss=%.4f",
                    step,
                    metrics.get("value_loss", 0),
                    metrics.get("q_loss", 0),
                    metrics.get("actor_loss", 0),
                )

            # Validation
            if step % self.config.val_every_steps == 0:
                from .eval_iql import evaluate_iql

                val_metrics = evaluate_iql(
                    self.networks, self.val_loader, self.device,
                )
                val_loss = val_metrics.get("total_loss", float("inf"))

                val_metrics["step"] = step
                val_metrics["phase"] = "val"
                self._log_metrics(val_metrics)

                logger.info("Step %d val: total_loss=%.4f", step, val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(step, val_metrics, "best.pt")

                self.networks.train()

            # Periodic save
            if step % self.config.save_every_steps == 0:
                self._save_checkpoint(step, metrics, f"step_{step:07d}.pt")

        # Final save
        self._save_checkpoint(step, metrics, "final.pt")
        logger.info("IQL training complete. Total steps=%d", step)

    def train_step(self, batch: TransitionBatch) -> dict:
        """Perform one IQL update step.

        Computes value loss, Q loss, actor loss, and advantage weights.

        Returns:
            Metrics dictionary for logging.
        """
        state = batch.state
        action = batch.action
        reward = batch.reward
        next_state = batch.next_state
        done = batch.done

        # ── Value loss (expectile regression) ──
        with torch.no_grad():
            q1_val = self.networks.target_q1(state, action)
            q2_val = self.networks.target_q2(state, action)
            q_min = torch.min(q1_val, q2_val)

        v = self.networks.v(state)
        value_loss = self.compute_value_loss(q_min, v)

        self.v_optimizer.zero_grad()
        value_loss.backward()
        if self.config.grad_clip_norm:
            nn.utils.clip_grad_norm_(
                self.networks.v.parameters(), self.config.grad_clip_norm,
            )
        self.v_optimizer.step()

        # ── Q loss (Bellman backup) ──
        with torch.no_grad():
            next_v = self.networks.v(next_state)

        q1 = self.networks.q1(state, action)
        q2 = self.networks.q2(state, action)
        q_loss = self.compute_q_loss(q1, q2, reward, done, next_v)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        if self.config.grad_clip_norm:
            nn.utils.clip_grad_norm_(
                list(self.networks.q1.parameters()) + list(self.networks.q2.parameters()),
                self.config.grad_clip_norm,
            )
        self.q_optimizer.step()

        # ── Actor loss (advantage-weighted BC) ──
        with torch.no_grad():
            q1_act = self.networks.target_q1(state, action)
            q2_act = self.networks.target_q2(state, action)
            q_min_act = torch.min(q1_act, q2_act)
            v_act = self.networks.v(state)
            advantage = q_min_act - v_act

        actor_loss = self.compute_actor_loss(state, action, advantage)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.config.grad_clip_norm:
            nn.utils.clip_grad_norm_(
                self.networks.actor.parameters(), self.config.grad_clip_norm,
            )
        self.actor_optimizer.step()

        # ── Target network update (EMA) ──
        self._update_targets()

        # ── Metrics ──
        metrics = {
            "value_loss": value_loss.item(),
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "v_mean": v.mean().item(),
            "advantage_mean": advantage.mean().item(),
            "advantage_max": advantage.max().item(),
            "reward_mean": reward.mean().item(),
        }

        return metrics

    def compute_value_loss(
        self,
        q_min: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IQL expectile value loss.

        q_min is min(Q1, Q2) for dataset actions.
        v is V(state).

        Uses asymmetric L2 loss with tau = expectile.
        """
        diff = q_min - v
        weight = torch.where(
            diff > 0,
            self.expectile,
            1.0 - self.expectile,
        )
        return (weight * diff.pow(2)).mean()

    def compute_q_loss(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Bellman Q loss.

        Target: reward + discount * (1 - done) * next_v
        """
        target = reward + self.discount * (1.0 - done) * next_v
        target = target.detach()

        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        return q1_loss + q2_loss

    def compute_actor_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        advantage: torch.Tensor,
    ) -> torch.Tensor:
        """Compute advantage-weighted behavior cloning loss.

        Dataset actions with higher advantage receive larger weights.
        Weights are clipped by config.max_adv_weight.
        """
        # Compute advantage weights
        weights = torch.exp(advantage * self.temperature)
        weights = torch.clamp(weights, max=self.max_adv_weight)
        weights = weights.detach()

        # Actor log probability
        log_prob = self.networks.actor.log_prob(state, action)

        # Weighted negative log-likelihood
        loss = -(weights * log_prob).mean()
        return loss

    def _update_targets(self) -> None:
        """EMA update of target Q networks."""
        tau = self.target_update_rate
        for target_p, p in zip(
            self.networks.target_q1.parameters(),
            self.networks.q1.parameters(),
        ):
            target_p.data.mul_(1 - tau).add_(p.data, alpha=tau)

        for target_p, p in zip(
            self.networks.target_q2.parameters(),
            self.networks.q2.parameters(),
        ):
            target_p.data.mul_(1 - tau).add_(p.data, alpha=tau)

    def _save_checkpoint(self, step: int, metrics: dict, filename: str) -> None:
        """Save IQL checkpoint."""
        save_rl_checkpoint(
            path=self.ckpt_dir / filename,
            model_state={
                "actor": self.networks.actor.state_dict(),
                "q1": self.networks.q1.state_dict(),
                "q2": self.networks.q2.state_dict(),
                "v": self.networks.v.state_dict(),
                "target_q1": self.networks.target_q1.state_dict(),
                "target_q2": self.networks.target_q2.state_dict(),
            },
            optimizer_state={
                "actor": self.actor_optimizer.state_dict(),
                "q": self.q_optimizer.state_dict(),
                "v": self.v_optimizer.state_dict(),
            },
            config=self.config.to_dict(),
            step=step,
            metrics=metrics,
        )

    def _log_metrics(self, metrics: dict) -> None:
        """Append metrics as a JSON line."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Train IQL Policy")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    
    config = IQLTrainConfig.from_yaml(args.config)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    trainer = IQLTrainer(config)
    trainer.train()
