"""Metrics utilities for BC and IQL evaluation."""

from __future__ import annotations

import torch


def compute_action_metrics(
    pred_action: torch.Tensor,
    true_action: torch.Tensor,
    continuous_indices: list[int],
    binary_indices: list[int],
) -> dict:
    """Compute behavior cloning action metrics.

    Reports:
        continuous MSE, continuous MAE
        binary precision, recall, F1
        per-dimension breakdowns for key actions

    Returns:
        Dictionary of metric values.
    """
    metrics: dict = {}

    # Continuous metrics
    if continuous_indices:
        ci = torch.tensor(continuous_indices, dtype=torch.long)
        pred_cont = pred_action[:, ci]
        true_cont = true_action[:, ci]

        metrics["continuous_mse"] = ((pred_cont - true_cont) ** 2).mean().item()
        metrics["continuous_mae"] = (pred_cont - true_cont).abs().mean().item()

        # Per-dimension MAE for the first few continuous dims
        for i, idx in enumerate(continuous_indices[:4]):
            metrics[f"mae_dim{idx}"] = (
                pred_action[:, idx] - true_action[:, idx]
            ).abs().mean().item()

    # Binary metrics
    if binary_indices:
        bi = torch.tensor(binary_indices, dtype=torch.long)
        pred_bin = (pred_action[:, bi] > 0).float()
        true_bin = true_action[:, bi]

        # Overall binary metrics
        tp = (pred_bin * true_bin).sum().item()
        fp = (pred_bin * (1 - true_bin)).sum().item()
        fn = ((1 - pred_bin) * true_bin).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics["binary_precision"] = precision
        metrics["binary_recall"] = recall
        metrics["binary_f1"] = f1

        # Per-action binary metrics
        for i, idx in enumerate(binary_indices):
            pred_i = (pred_action[:, idx] > 0).float()
            true_i = true_action[:, idx]

            tp_i = (pred_i * true_i).sum().item()
            fp_i = (pred_i * (1 - true_i)).sum().item()
            fn_i = ((1 - pred_i) * true_i).sum().item()

            p_i = tp_i / (tp_i + fp_i + 1e-8)
            r_i = tp_i / (tp_i + fn_i + 1e-8)
            f1_i = 2 * p_i * r_i / (p_i + r_i + 1e-8)

            metrics[f"precision_dim{idx}"] = p_i
            metrics[f"recall_dim{idx}"] = r_i
            metrics[f"f1_dim{idx}"] = f1_i

    return metrics


def summarize_iql_values(
    q_values: torch.Tensor,
    v_values: torch.Tensor,
    advantages: torch.Tensor,
) -> dict:
    """Summarize Q, V, and advantage distributions for logging.

    Used to detect Q explosion, collapsed values, or advantage weights becoming
    too large.

    Returns:
        Dictionary of distribution statistics.
    """
    return {
        "q_mean": q_values.mean().item(),
        "q_std": q_values.std().item(),
        "q_min": q_values.min().item(),
        "q_max": q_values.max().item(),
        "v_mean": v_values.mean().item(),
        "v_std": v_values.std().item(),
        "v_min": v_values.min().item(),
        "v_max": v_values.max().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item(),
        "advantage_min": advantages.min().item(),
        "advantage_max": advantages.max().item(),
        "advantage_abs_mean": advantages.abs().mean().item(),
    }
