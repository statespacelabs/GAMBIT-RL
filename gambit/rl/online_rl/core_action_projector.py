"""CoreActionProjector and distillation trainers.

Stage 1 (CoreActionDistiller): Feedforward distillation
  obs_45 → TelemetryEncoder → 512 → Linear → 8-dim action

Stage 2 (RecurrentDistiller): Recurrent BC warmup
  obs_seq [B,T,45] → TelemetryEncoder → GRU → actor heads → 8-dim action sequence

Both stages use the offline Phase 2 dataset with projected core actions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .telemetry_encoder import TelemetryEncoder
from .distillation_dataset import DistillationDataset
from .action_schema import ActionCommandConverter, CORE_ACTION_INDICES
from .distributed_utils import (
    barrier,
    is_main_process,
    reduce_metric_dict,
    unwrap_model,
)

logger = logging.getLogger(__name__)

# Re-export for convenience
__all__ = [
    "CORE_ACTION_INDICES",
    "CoreActionProjector",
    "CoreActionDistiller",
    "RecurrentDistiller",
]


class _RecurrentDistillDDPAdapter(nn.Module):
    """Routes distillation forward through DDP for gradient sync."""

    def __init__(self, actor_critic: nn.Module):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        batch_size = obs_seq.shape[0]
        hidden = self.actor_critic.init_hidden(batch_size, obs_seq.device)
        return self.actor_critic.predict_actions(
            obs_seq, hidden, return_logits=True
        )


class CoreActionProjector(nn.Module):
    """Feedforward action projection: obs_45 → TelemetryEncoder → 8-dim action.

    Used in Stage 1 distillation. The action_head is a single Linear(512→8)
    where outputs [0:4] are continuous action values and [4:8] are binary logits.

    After Stage 1, weights are split into RecurrentActorCritic heads via
    RecurrentActorCritic.from_distilled().
    """

    N_CONTINUOUS = 4  # move_x, move_y, look_dx, look_dy
    N_BINARY = 4  # shoot, reload, jump, crouch

    def __init__(self, tel_encoder: TelemetryEncoder):
        super().__init__()
        self.tel_encoder = tel_encoder
        self.action_head = nn.Linear(512, self.N_CONTINUOUS + self.N_BINARY)

    def forward(self, obs_45: torch.Tensor) -> torch.Tensor:
        """Predict 8-dim core action from telemetry.

        Args:
            obs_45: [B, T, 45] telemetry observation.

        Returns:
            action_8: [B, T, 8]. Dims [0:4] continuous, [4:8] binary logits.
        """
        z = self.tel_encoder(obs_45)  # [B, T, 512]
        return self.action_head(z)  # [B, T, 8]

    def predict(self, obs_45: torch.Tensor) -> torch.Tensor:
        """Predict with sigmoid applied to binary dims (for evaluation)."""
        raw = self.forward(obs_45)
        out = raw.clone()
        out[..., self.N_CONTINUOUS :] = torch.sigmoid(raw[..., self.N_CONTINUOUS :])
        return out


def _compute_distillation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_continuous: int = 4,
    pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute continuous MSE and class-weighted binary BCE with per-head metrics."""
    cont_pred = pred[..., :n_continuous]
    cont_target = target[..., :n_continuous]
    mse_loss = F.mse_loss(cont_pred, cont_target)

    bin_pred = pred[..., n_continuous:]
    bin_target = target[..., n_continuous:]
    if not torch.all((bin_target == 0.0) | (bin_target == 1.0)):
        raise ValueError("Binary distillation targets must be exactly 0 or 1")
    weight = pos_weight.to(pred.device) if pos_weight is not None else None
    bce_loss = F.binary_cross_entropy_with_logits(
        bin_pred, bin_target, pos_weight=weight
    )
    total = mse_loss + bce_loss

    with torch.no_grad():
        reduce_dims = tuple(range(bin_pred.ndim - 1))
        binary_pred = bin_pred >= 0
        binary_true = bin_target >= 0.5
        accuracy = (binary_pred == binary_true).float().mean(dim=reduce_dims)
        tp = (binary_pred & binary_true).float().sum(dim=reduce_dims)
        fp = (binary_pred & ~binary_true).float().sum(dim=reduce_dims)
        fn = (~binary_pred & binary_true).float().sum(dim=reduce_dims)
        tn = (~binary_pred & ~binary_true).float().sum(dim=reduce_dims)
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        predicted_positive_rate = binary_pred.float().mean(dim=reduce_dims)
        target_positive_rate = binary_true.float().mean(dim=reduce_dims)

    return total, {
        "total_loss": total.item(),
        "mse_loss": mse_loss.item(),
        "bce_loss": bce_loss.item(),
        "binary_accuracy_per_head": accuracy.cpu().tolist(),
        "binary_f1_per_head": f1.cpu().tolist(),
        "predicted_positive_rate_per_head": predicted_positive_rate.cpu().tolist(),
        "target_positive_rate_per_head": target_positive_rate.cpu().tolist(),
        "binary_tp_per_head": tp.cpu().tolist(),
        "binary_fp_per_head": fp.cpu().tolist(),
        "binary_fn_per_head": fn.cpu().tolist(),
        "binary_tn_per_head": tn.cpu().tolist(),
    }


def _new_metric_accum() -> dict:
    return {
        "total_loss_sum": 0.0,
        "mse_loss_sum": 0.0,
        "bce_loss_sum": 0.0,
        "weight_sum": 0.0,
        "tp": [0.0] * 4,
        "fp": [0.0] * 4,
        "fn": [0.0] * 4,
        "tn": [0.0] * 4,
    }


def _sample_weight(pred: torch.Tensor) -> int:
    """Number of positions (B or B*T) for loss weighting."""
    return int(pred.shape[:-1].numel())


def _accumulate_batch_metrics(
    accum: dict, loss: torch.Tensor, components: dict, weight: int
) -> None:
    accum["total_loss_sum"] += loss.item() * weight
    accum["mse_loss_sum"] += components["mse_loss"] * weight
    accum["bce_loss_sum"] += components["bce_loss"] * weight
    accum["weight_sum"] += weight
    for i in range(4):
        accum["tp"][i] += components["binary_tp_per_head"][i]
        accum["fp"][i] += components["binary_fp_per_head"][i]
        accum["fn"][i] += components["binary_fn_per_head"][i]
        accum["tn"][i] += components["binary_tn_per_head"][i]


def _finalize_distill_metrics(accum: dict, distributed: bool) -> dict:
    raw = {
        "total_loss_sum": accum["total_loss_sum"],
        "mse_loss_sum": accum["mse_loss_sum"],
        "bce_loss_sum": accum["bce_loss_sum"],
        "weight_sum": accum["weight_sum"],
        "tp": accum["tp"],
        "fp": accum["fp"],
        "fn": accum["fn"],
        "tn": accum["tn"],
    }
    reduced = reduce_metric_dict(raw) if distributed else raw
    w = max(reduced["weight_sum"], 1.0)
    tp = reduced["tp"]
    fp = reduced["fp"]
    fn = reduced["fn"]
    tn = reduced["tn"]
    binary_accuracy = []
    binary_f1 = []
    pred_pos_rate = []
    target_pos_rate = []
    for i in range(4):
        n = tp[i] + fp[i] + fn[i] + tn[i]
        n = max(n, 1.0)
        binary_accuracy.append((tp[i] + tn[i]) / n)
        prec = tp[i] / (tp[i] + fp[i] + 1e-8)
        rec = tp[i] / (tp[i] + fn[i] + 1e-8)
        binary_f1.append((2 * prec * rec) / (prec + rec + 1e-8))
        pred_pos_rate.append((tp[i] + fp[i]) / n)
        target_pos_rate.append((tp[i] + fn[i]) / n)
    return {
        "total_loss": reduced["total_loss_sum"] / w,
        "mse_loss": reduced["mse_loss_sum"] / w,
        "bce_loss": reduced["bce_loss_sum"] / w,
        "binary_accuracy_per_head": binary_accuracy,
        "binary_f1_per_head": binary_f1,
        "predicted_positive_rate_per_head": pred_pos_rate,
        "target_positive_rate_per_head": target_pos_rate,
    }


def _wrap_ddp(
    model: nn.Module,
    distributed: bool,
    local_rank: int,
    find_unused_parameters: bool,
) -> nn.Module:
    if not distributed:
        return model
    if next(model.parameters()).is_cuda:
        return DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
        )
    return DistributedDataParallel(
        model,
        find_unused_parameters=find_unused_parameters,
    )


def _build_train_loader(
    dataset: DistillationDataset,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    rank: int,
    world_size: int,
    pin_memory: bool = False,
) -> tuple[DataLoader, DistributedSampler | None]:
    train_sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        if distributed
        else None
    )
    loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": train_sampler is None,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,
    }
    if train_sampler is not None:
        loader_kwargs["sampler"] = train_sampler
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **loader_kwargs), train_sampler


def _build_val_loader(
    dataset: DistillationDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
) -> DataLoader:
    loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **loader_kwargs)


def _broadcast_passed(passed: bool, device: torch.device, distributed: bool) -> bool:
    if not distributed:
        return passed
    import torch.distributed as dist

    if is_main_process():
        flag = torch.tensor([1 if passed else 0], dtype=torch.int32, device=device)
    else:
        flag = torch.tensor([0], dtype=torch.int32, device=device)
    dist.broadcast(flag, src=0)
    return flag.item() == 1


def _log_binary_dataset_stats(
    dataset: DistillationDataset,
    pos_weight: torch.Tensor | None,
    label: str,
    max_samples: int = 20000,
) -> None:
    count = min(len(dataset.df), max_samples)
    if count == 0:
        raise RuntimeError(f"{label} dataset is empty")
    indices = torch.linspace(0, len(dataset.df) - 1, steps=count).long().tolist()
    targets = torch.stack(
        [
            torch.from_numpy(dataset._load_core_action(dataset.df.iloc[index]))
            for index in indices
        ]
    )
    binary = targets[..., 4:8]
    logger.info(
        "%s binary target positive rates=%s unique=%s pos_weight=%s",
        label,
        binary.float().mean(dim=0).tolist(),
        torch.unique(binary).tolist(),
        pos_weight.tolist() if pos_weight is not None else None,
    )


class CoreActionDistiller:
    """Stage 1: Feedforward distillation of CoreActionProjector.

    Trains TelemetryEncoder (projection layer) + action_head on single-frame
    (obs_45, core_action_8) pairs. Telemetry branches are frozen initially
    and optionally unfrozen after warmup_epochs.
    """

    def __init__(
        self,
        projector: CoreActionProjector,
        transition_manifest_path: str,
        window_manifest_path: str,
        action_converter: ActionCommandConverter | None = None,
        telemetry_fps: float = 30.0,
        step_seconds: float = 1.0,
        require_normalized_telemetry: bool = True,
        action_scaler_path: str | None = None,
        binary_pos_weight: list[float] | None = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        epochs: int = 30,
        warmup_epochs: int = 10,
        grad_clip_norm: float = 1.0,
        num_workers: int = 4,
        device: str = "cuda",
        checkpoint_dir: str = "experiments/phase3/distill/checkpoints",
        log_dir: str = "experiments/phase3/distill/logs",
        distributed: bool = False,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        max_batches: int | None = None,
    ):
        self.projector = projector
        self.device = torch.device(device)
        self.distributed = distributed
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.max_batches = max_batches
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.grad_clip_norm = grad_clip_norm
        self.num_workers = num_workers
        if binary_pos_weight is not None and len(binary_pos_weight) != 4:
            raise ValueError("binary_pos_weight must contain four values")
        self.binary_pos_weight = (
            torch.tensor(binary_pos_weight, dtype=torch.float32, device=self.device)
            if binary_pos_weight is not None
            else None
        )

        self.ckpt_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        if is_main_process():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        barrier()
        self.log_path = self.log_dir / "distill_stage1.jsonl"

        # Datasets
        self.train_ds = DistillationDataset(
            transition_manifest_path=transition_manifest_path,
            window_manifest_path=window_manifest_path,
            action_converter=action_converter,
            telemetry_fps=telemetry_fps,
            step_seconds=step_seconds,
            require_normalized_telemetry=require_normalized_telemetry,
            mode="single",
            split="train",
            action_scaler_path=action_scaler_path,
        )
        self.val_ds = DistillationDataset(
            transition_manifest_path=transition_manifest_path,
            window_manifest_path=window_manifest_path,
            action_converter=action_converter,
            telemetry_fps=telemetry_fps,
            step_seconds=step_seconds,
            require_normalized_telemetry=require_normalized_telemetry,
            mode="single",
            split="val",
            action_scaler_path=action_scaler_path,
        )

        if is_main_process():
            _log_binary_dataset_stats(
                self.train_ds, self.binary_pos_weight, self.__class__.__name__ + " train"
            )
            _log_binary_dataset_stats(
                self.val_ds, self.binary_pos_weight, self.__class__.__name__ + " val"
            )
        barrier()

    def train(self) -> CoreActionProjector:
        """Run Stage 1 feedforward distillation."""
        self.projector.to(self.device)
        model = _wrap_ddp(
            self.projector,
            self.distributed,
            self.local_rank,
            find_unused_parameters=True,
        )

        if is_main_process():
            logger.info("Running alignment verification...")
            alignment = self.train_ds.verify_alignment()
            passed = alignment["passed"]
            if not passed:
                logger.error(
                    "Distillation alignment check failed: %s",
                    alignment["errors"][:3],
                )
        else:
            passed = True
        if not _broadcast_passed(passed, self.device, self.distributed):
            raise RuntimeError("Distillation alignment check failed on rank 0")
        barrier()

        unwrap_model(model).tel_encoder.freeze_branches()

        optimizer = torch.optim.AdamW(
            [p for p in unwrap_model(model).parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        pin_memory = self.device.type == "cuda"
        train_loader, train_sampler = _build_train_loader(
            self.train_ds,
            self.batch_size,
            self.num_workers,
            self.distributed,
            self.rank,
            self.world_size,
            pin_memory=pin_memory,
        )
        val_loader = _build_val_loader(
            self.val_ds, self.batch_size, self.num_workers, pin_memory=pin_memory
        )

        if is_main_process():
            global_batch = self.batch_size * self.world_size
            logger.info(
                "Stage 1 distillation: %d train, %d val, %d epochs (unfreeze at %d) | "
                "per_device_batch_size=%d world_size=%d effective_global_batch_size=%d",
                len(self.train_ds),
                len(self.val_ds),
                self.epochs,
                self.warmup_epochs,
                self.batch_size,
                self.world_size,
                global_batch,
            )

        best_val_loss = float("inf")
        val_metrics: dict = {}

        for epoch in range(1, self.epochs + 1):
            if self.distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if epoch == self.warmup_epochs + 1:
                if isinstance(model, DistributedDataParallel):
                    underlying = model.module
                    del model
                else:
                    underlying = unwrap_model(model)
                underlying.tel_encoder.unfreeze_branches()
                model = _wrap_ddp(
                    underlying,
                    self.distributed,
                    self.local_rank,
                    find_unused_parameters=False,
                )
                optimizer = torch.optim.AdamW(
                    [
                        {
                            "params": unwrap_model(model).tel_encoder.projection.parameters(),
                            "lr": self.lr,
                        },
                        {
                            "params": unwrap_model(model).action_head.parameters(),
                            "lr": self.lr,
                        },
                        {
                            "params": unwrap_model(model).tel_encoder.where_branch.parameters(),
                            "lr": self.lr * 0.1,
                        },
                        {
                            "params": unwrap_model(model).tel_encoder.view_branch.parameters(),
                            "lr": self.lr * 0.1,
                        },
                        {
                            "params": unwrap_model(model).tel_encoder.rhythm_branch.parameters(),
                            "lr": self.lr * 0.1,
                        },
                        {
                            "params": unwrap_model(
                                model
                            ).tel_encoder.telemetry_fusion.parameters(),
                            "lr": self.lr * 0.1,
                        },
                    ],
                    weight_decay=self.weight_decay,
                )
                if is_main_process():
                    logger.info("Epoch %d: branches unfrozen with 0.1x LR", epoch)

            model.train()
            train_metrics = self._run_epoch(model, train_loader, optimizer)

            barrier()
            if is_main_process():
                eval_model = unwrap_model(model)
                eval_model.eval()
                with torch.no_grad():
                    val_metrics = self._run_epoch(eval_model, val_loader, optimizer=None)
            else:
                val_metrics = {}
            barrier()

            if is_main_process():
                logger.info(
                    "Epoch %d/%d: train_loss=%.4f val_loss=%.4f (mse=%.4f bce=%.4f)",
                    epoch,
                    self.epochs,
                    train_metrics["total_loss"],
                    val_metrics["total_loss"],
                    val_metrics["mse_loss"],
                    val_metrics["bce_loss"],
                )
                self._log_metrics(
                    {
                        "epoch": epoch,
                        "train": train_metrics,
                        "val": val_metrics,
                    }
                )

                if val_metrics["total_loss"] < best_val_loss:
                    best_val_loss = val_metrics["total_loss"]
                    self._save_checkpoint(model, epoch, val_metrics, "command_best.pt")
                    logger.info("  → new best (val_loss=%.4f)", best_val_loss)
            barrier()

        if is_main_process():
            self._save_checkpoint(model, self.epochs, val_metrics, "command_final.pt")
            logger.info(
                "Stage 1 distillation complete. Best val_loss=%.4f", best_val_loss
            )
        barrier()
        return unwrap_model(model)

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
    ) -> dict:
        """Run one epoch. If optimizer is None, evaluation only."""
        accum = _new_metric_accum()
        n_batches = 0

        for obs_45, core_action in loader:
            if self.max_batches is not None and n_batches >= self.max_batches:
                break
            obs_45 = obs_45.unsqueeze(1).to(self.device)
            core_action = core_action.to(self.device)

            pred = model(obs_45).squeeze(1)
            loss, components = _compute_distillation_loss(
                pred, core_action, pos_weight=self.binary_pos_weight
            )
            weight = _sample_weight(pred)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    unwrap_model(model).parameters(), self.grad_clip_norm
                )
                optimizer.step()

            _accumulate_batch_metrics(accum, loss, components, weight)
            n_batches += 1

        aggregate = self.distributed and optimizer is not None
        return _finalize_distill_metrics(accum, aggregate)

    def _save_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        metrics: dict,
        filename: str,
    ) -> None:
        underlying = unwrap_model(model)
        path = self.ckpt_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "projector_state": underlying.state_dict(),
                "tel_encoder_state": underlying.tel_encoder.state_dict(),
                "metrics": metrics,
            },
            path,
        )

    def _log_metrics(self, metrics: dict) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")


class RecurrentDistiller:
    """Stage 2: Recurrent BC warmup for RecurrentActorCritic.

    Trains the full pipeline on action sequences:
        tel_seq [B,T,45] → TelemetryEncoder → GRU → actor heads → action_8 [B,T,8]

    This ensures the GRU learns temporal dependencies before PPO starts.
    Without this stage, PPO would start with a random GRU and jerk around.
    """

    def __init__(
        self,
        actor_critic,  # RecurrentActorCritic (imported at runtime to avoid circular)
        transition_manifest_path: str,
        window_manifest_path: str,
        action_converter: ActionCommandConverter | None = None,
        telemetry_fps: float = 30.0,
        step_seconds: float = 1.0,
        require_normalized_telemetry: bool = True,
        action_scaler_path: str | None = None,
        binary_pos_weight: list[float] | None = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        seq_len: int = 32,
        epochs: int = 20,
        grad_clip_norm: float = 1.0,
        num_workers: int = 4,
        device: str = "cuda",
        checkpoint_dir: str = "experiments/phase3/distill/checkpoints",
        log_dir: str = "experiments/phase3/distill/logs",
        distributed: bool = False,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        max_batches: int | None = None,
    ):
        self.actor_critic = actor_critic
        self.device = torch.device(device)
        self.distributed = distributed
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.max_batches = max_batches
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.epochs = epochs
        self.grad_clip_norm = grad_clip_norm
        self.num_workers = num_workers
        if binary_pos_weight is not None and len(binary_pos_weight) != 4:
            raise ValueError("binary_pos_weight must contain four values")
        self.binary_pos_weight = (
            torch.tensor(binary_pos_weight, dtype=torch.float32, device=self.device)
            if binary_pos_weight is not None
            else None
        )

        self.ckpt_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        if is_main_process():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        barrier()
        self.log_path = self.log_dir / "distill_stage2.jsonl"

        self.train_ds = DistillationDataset(
            transition_manifest_path=transition_manifest_path,
            window_manifest_path=window_manifest_path,
            action_converter=action_converter,
            telemetry_fps=telemetry_fps,
            step_seconds=step_seconds,
            require_normalized_telemetry=require_normalized_telemetry,
            mode="sequence",
            seq_len=seq_len,
            split="train",
            action_scaler_path=action_scaler_path,
        )
        self.val_ds = DistillationDataset(
            transition_manifest_path=transition_manifest_path,
            window_manifest_path=window_manifest_path,
            action_converter=action_converter,
            telemetry_fps=telemetry_fps,
            step_seconds=step_seconds,
            require_normalized_telemetry=require_normalized_telemetry,
            mode="sequence",
            seq_len=seq_len,
            split="val",
            action_scaler_path=action_scaler_path,
        )

        if is_main_process():
            _log_binary_dataset_stats(
                self.train_ds, self.binary_pos_weight, self.__class__.__name__ + " train"
            )
            _log_binary_dataset_stats(
                self.val_ds, self.binary_pos_weight, self.__class__.__name__ + " val"
            )
        barrier()

    def train(self):
        """Run Stage 2 recurrent distillation."""
        self.actor_critic.to(self.device)
        self.actor_critic.set_ppo_mode(training=True)
        self.actor_critic.tel_encoder.freeze_branches()

        distill_adapter = _RecurrentDistillDDPAdapter(self.actor_critic)
        model = _wrap_ddp(
            distill_adapter,
            self.distributed,
            self.local_rank,
            find_unused_parameters=True,
        )

        optimizer = torch.optim.AdamW(
            [p for p in unwrap_model(model).actor_critic.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        pin_memory = self.device.type == "cuda"
        train_loader, train_sampler = _build_train_loader(
            self.train_ds,
            self.batch_size,
            self.num_workers,
            self.distributed,
            self.rank,
            self.world_size,
            pin_memory=pin_memory,
        )
        val_loader = _build_val_loader(
            self.val_ds, self.batch_size, self.num_workers, pin_memory=pin_memory
        )

        if is_main_process():
            global_batch = self.batch_size * self.world_size
            logger.info(
                "Stage 2 recurrent distillation: %d train seqs, %d val seqs, %d epochs | "
                "per_device_batch_size=%d world_size=%d effective_global_batch_size=%d",
                len(self.train_ds),
                len(self.val_ds),
                self.epochs,
                self.batch_size,
                self.world_size,
                global_batch,
            )

        best_val_loss = float("inf")
        val_metrics: dict = {}

        for epoch in range(1, self.epochs + 1):
            if self.distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            unwrap_model(model).actor_critic.set_ppo_mode(training=True)
            model.train()
            train_metrics = self._run_epoch(model, train_loader, optimizer)

            barrier()
            if is_main_process():
                eval_model = unwrap_model(model)
                eval_model.eval()
                with torch.no_grad():
                    val_metrics = self._run_epoch(eval_model, val_loader, optimizer=None)
            else:
                val_metrics = {}
            barrier()

            if is_main_process():
                logger.info(
                    "Epoch %d/%d: train_loss=%.4f val_loss=%.4f (mse=%.4f bce=%.4f)",
                    epoch,
                    self.epochs,
                    train_metrics["total_loss"],
                    val_metrics["total_loss"],
                    val_metrics["mse_loss"],
                    val_metrics["bce_loss"],
                )
                self._log_metrics(
                    {
                        "epoch": epoch,
                        "train": train_metrics,
                        "val": val_metrics,
                    }
                )

                if val_metrics["total_loss"] < best_val_loss:
                    best_val_loss = val_metrics["total_loss"]
                    self._save_checkpoint(
                        model, epoch, val_metrics, "command_recurrent_best.pt"
                    )
            barrier()

        if is_main_process():
            self._save_checkpoint(
                model, self.epochs, val_metrics, "command_recurrent_final.pt"
            )
            logger.info(
                "Stage 2 recurrent distillation complete. Best val_loss=%.4f",
                best_val_loss,
            )
        barrier()
        return unwrap_model(model).actor_critic

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
    ) -> dict:
        accum = _new_metric_accum()
        n_batches = 0
        underlying = unwrap_model(model)

        for obs_seq, action_seq in loader:
            if self.max_batches is not None and n_batches >= self.max_batches:
                break
            obs_seq = obs_seq.to(self.device)
            action_seq = action_seq.to(self.device)

            pred_actions = model(obs_seq)

            loss, components = _compute_distillation_loss(
                pred_actions, action_seq, pos_weight=self.binary_pos_weight
            )
            weight = _sample_weight(pred_actions)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    underlying.actor_critic.parameters(), self.grad_clip_norm
                )
                optimizer.step()

            _accumulate_batch_metrics(accum, loss, components, weight)
            n_batches += 1

        aggregate = self.distributed and optimizer is not None
        return _finalize_distill_metrics(accum, aggregate)

    def _save_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        metrics: dict,
        filename: str,
    ) -> None:
        underlying = unwrap_model(model).actor_critic
        path = self.ckpt_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "actor_critic_state": underlying.state_dict(),
                "metrics": metrics,
            },
            path,
        )

    def _log_metrics(self, metrics: dict) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
