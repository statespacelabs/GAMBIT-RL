"""Evaluation and checkpoint-promotion helpers."""

from __future__ import annotations

import logging

import numpy as np
import torch

from .selfplay_runner import (
    _agent_observation,
    make_action_tuple,
    policy_action_to_buffers,
    validate_unity_action_spec,
)

logger = logging.getLogger(__name__)


def evaluate_policy(
    policy,
    env,
    n_episodes: int = 20,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate deterministically against the Unity scripted bot."""
    policy.eval()
    device_obj = torch.device(device)
    behavior_name = list(env.behavior_specs.keys())[0]
    action_spec = env.behavior_specs[behavior_name].action_spec
    validate_unity_action_spec(action_spec)

    wins = 0
    total_rewards: list[float] = []
    current_reward = 0.0
    episodes_completed = 0
    hidden_states: dict[int, torch.Tensor] = {}

    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    while episodes_completed < n_episodes:
        for agent_id in terminal_steps:
            current_reward += float(terminal_steps[agent_id].reward)
            total_rewards.append(current_reward)
            wins += int(current_reward > 0.0)
            episodes_completed += 1
            current_reward = 0.0
            hidden_states.pop(agent_id, None)

        if episodes_completed >= n_episodes:
            break
        if len(decision_steps) == 0:
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            continue

        agent_id = int(decision_steps.agent_id[0])
        obs = _agent_observation(decision_steps, agent_id)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device_obj)
        hidden = hidden_states.get(agent_id, policy.init_hidden(1, device_obj))
        with torch.no_grad():
            dist, _, next_hidden = policy(obs_tensor, hidden)
            action, _ = dist.mode
        continuous, discrete = policy_action_to_buffers(
            action.squeeze(0).squeeze(0).cpu().numpy(),
            action_spec,
        )
        env.set_action_for_agent(
            behavior_name,
            agent_id,
            make_action_tuple(continuous, discrete),
        )
        hidden_states[agent_id] = next_hidden
        current_reward += float(decision_steps[agent_id].reward)
        env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

    metrics = {
        "win_rate": wins / max(1, n_episodes),
        "avg_reward": float(np.mean(total_rewards) if total_rewards else 0.0),
        "damage_ratio": 1.0,
    }
    logger.info(
        "Evaluation completed: win_rate=%.2f avg_reward=%.2f",
        metrics["win_rate"],
        metrics["avg_reward"],
    )
    return metrics


def should_promote(
    eval_metrics: dict[str, float],
    threshold_win_rate: float = 0.55,
) -> bool:
    return (
        eval_metrics.get("win_rate", 0.0) >= threshold_win_rate
        or eval_metrics.get("damage_ratio", 0.0) >= 1.2
    )
