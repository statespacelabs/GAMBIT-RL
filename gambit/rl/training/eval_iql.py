"""IQL evaluation on validation/test transitions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gambit.rl.models.actor_critic import IQLNetworks
from gambit.rl.utils.metrics import summarize_iql_values


@torch.no_grad()
def evaluate_iql(
    networks: IQLNetworks,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate IQL networks on validation/test transitions.

    Reports:
        value loss
        Q loss
        actor imitation loss
        Q/V scale
        advantage statistics
        action distribution statistics

    Returns:
        Dictionary of evaluation metrics.
    """
    networks.eval()

    total_v_loss = 0.0
    total_q_loss = 0.0
    total_actor_loss = 0.0
    n_batches = 0

    all_q_vals: list[torch.Tensor] = []
    all_v_vals: list[torch.Tensor] = []
    all_advantages: list[torch.Tensor] = []

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)

        state = batch.state
        action = batch.action
        reward = batch.reward
        next_state = batch.next_state
        done = batch.done

        # Q values
        q1 = networks.q1(state, action)
        q2 = networks.q2(state, action)
        q_min = torch.min(q1, q2)

        # V values
        v = networks.v(state)
        next_v = networks.v(next_state)

        # Value loss (expectile)
        diff = q_min - v
        weight = torch.where(diff > 0, 0.7, 0.3)
        v_loss = (weight * diff.pow(2)).mean()
        total_v_loss += v_loss.item()

        # Q loss
        target = reward + 0.99 * (1.0 - done) * next_v
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        total_q_loss += (q1_loss + q2_loss).item()

        # Actor loss
        log_prob = networks.actor.log_prob(state, action)
        advantage = q_min - v
        weights = torch.exp(advantage * 3.0).clamp(max=100.0)
        actor_loss = -(weights * log_prob).mean()
        total_actor_loss += actor_loss.item()

        # Collect for distribution stats
        all_q_vals.append(q_min.cpu())
        all_v_vals.append(v.cpu())
        all_advantages.append(advantage.cpu())

        n_batches += 1

    if n_batches == 0:
        return {"total_loss": float("inf")}

    metrics = {
        "value_loss": total_v_loss / n_batches,
        "q_loss": total_q_loss / n_batches,
        "actor_loss": total_actor_loss / n_batches,
        "total_loss": (total_v_loss + total_q_loss + total_actor_loss) / n_batches,
    }

    # Distribution stats
    q_all = torch.cat(all_q_vals)
    v_all = torch.cat(all_v_vals)
    adv_all = torch.cat(all_advantages)

    value_stats = summarize_iql_values(q_all, v_all, adv_all)
    metrics.update(value_stats)

    networks.train()
    return metrics
