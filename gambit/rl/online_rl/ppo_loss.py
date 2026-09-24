"""PPO objectives with a stable distilled-policy reference anchor."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .hybrid_distribution import HybridDistribution


def compute_reference_anchor(
    current: HybridDistribution,
    reference: HybridDistribution,
) -> torch.Tensor:
    """BC-style anchor between current and frozen reference distributions."""
    reference_mean = reference.cont_mean.detach()
    reference_binary_probs = torch.sigmoid(reference.binary_logits.detach())
    mean_loss = F.mse_loss(current.cont_mean, reference_mean)
    binary_loss = F.binary_cross_entropy_with_logits(
        current.binary_logits,
        reference_binary_probs,
    )
    return mean_loss + binary_loss


def compute_aim_smoothness(current_cont_mean: torch.Tensor) -> torch.Tensor:
    """Penalize temporal changes in current-policy look means."""
    if current_cont_mean.shape[1] <= 1:
        return current_cont_mean.sum() * 0.0
    look_mean = current_cont_mean[..., 2:4]
    return ((look_mean[:, 1:] - look_mean[:, :-1]) ** 2).mean()


def compute_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    reference_anchor: torch.Tensor,
    current_cont_mean: torch.Tensor,
    clip_ratio: float = 0.1,
    value_coef: float = 0.5,
    entropy_coef: float = 0.003,
    kl_coef: float = 0.05,
    smooth_coef: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute clipped PPO, value, entropy, reference and smoothness losses."""
    old_log_probs = old_log_probs.detach()
    advantages = advantages.detach()
    returns = returns.detach()

    adv_std = advantages.std(unbiased=False)
    if adv_std > 1e-8:
        advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

    ratio = torch.exp(new_log_probs - old_log_probs)
    surrogate = ratio * advantages
    clipped_surrogate = (
        torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    )
    policy_loss = -torch.min(surrogate, clipped_surrogate).mean()
    value_loss = F.mse_loss(new_values, returns)
    entropy_loss = -entropy.mean()
    smoothness_loss = compute_aim_smoothness(current_cont_mean)

    loss = (
        policy_loss
        + value_coef * value_loss
        + entropy_coef * entropy_loss
        + kl_coef * reference_anchor
        + smooth_coef * smoothness_loss
    )

    metrics = {
        "loss/total": loss.item(),
        "loss/policy": policy_loss.item(),
        "loss/value": value_loss.item(),
        "loss/entropy": entropy_loss.item(),
        "loss/reference_anchor": reference_anchor.item(),
        "loss/smoothness": smoothness_loss.item(),
        "metrics/clip_frac": (torch.abs(ratio - 1.0) > clip_ratio)
        .float()
        .mean()
        .item(),
        "metrics/ratio_mean": ratio.mean().item(),
        "metrics/ratio_std": ratio.std(unbiased=False).item(),
        "metrics/adv_mean": advantages.mean().item(),
        "metrics/adv_std": advantages.std(unbiased=False).item(),
    }
    return loss, metrics
