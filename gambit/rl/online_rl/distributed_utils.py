"""Distributed training helpers for Phase 3A distillation (DDP).

Single-process runs when WORLD_SIZE is unset or 1. Multi-GPU via torchrun.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist


def init_distributed() -> tuple[bool, int, int, int, torch.device]:
    """Initialize distributed training if launched with torchrun.

    Returns:
        (distributed, rank, local_rank, world_size, device)
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
            device = torch.device(f"cuda:{local_rank}")
        else:
            backend = "gloo"
            device = torch.device("cpu")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    return distributed, rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def reduce_sum_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce sum in place; no-op when not distributed."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def reduce_mean_scalar(value_sum: float, count: float) -> float:
    """Global mean from local sum and count across ranks."""
    t = torch.tensor([value_sum, count], dtype=torch.float64, device="cpu")
    reduce_sum_tensor(t)
    if t[1].item() <= 0:
        return 0.0
    return (t[0] / t[1]).item()


def reduce_metric_dict(stats: dict[str, Any]) -> dict[str, Any]:
    """All-reduce scalar sums in stats dict (keys ending with _sum or plain count)."""
    out = dict(stats)
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            t = torch.tensor([float(value)], dtype=torch.float64)
            reduce_sum_tensor(t)
            out[key] = t.item()
        elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
            t = torch.tensor(value, dtype=torch.float64)
            reduce_sum_tensor(t)
            out[key] = t.tolist()
    return out
