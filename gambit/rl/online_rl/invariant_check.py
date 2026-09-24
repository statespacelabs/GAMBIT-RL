"""Standalone Phase 3A pre-update PPO invariant and Unity rollout gate."""

from __future__ import annotations

from copy import deepcopy
import json
import logging
from pathlib import Path

import torch

from gambit.dataset_preparation.configs import DistillConfig, PPOTrainConfig
from .actor_critic import RecurrentActorCritic
from .core_action_projector import CoreActionProjector
from .reward_builder import RewardBuilder
from .rollout_buffer import RecurrentRolloutBuffer
from .selfplay_runner import SelfPlayRunner
from .telemetry_encoder import TelemetryEncoder

logger = logging.getLogger(__name__)


def _setup_unity_env(env_path: str, time_scale: float = 20.0):
    try:
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.engine_configuration_channel import (
            EngineConfigurationChannel,
        )
    except ImportError as exc:
        raise RuntimeError("mlagents_envs is required for invariant_only") from exc
    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(time_scale=time_scale)
    environment = UnityEnvironment(
        file_name=env_path or None,
        seed=42,
        side_channels=[engine_channel],
        no_graphics=True,
    )
    environment.reset()
    return environment


def load_recurrent_policy(
    distill_config: DistillConfig,
    device: torch.device,
) -> tuple[RecurrentActorCritic, Path]:
    checkpoint = Path(distill_config.checkpoint_dir) / "command_recurrent_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Recurrent distilled checkpoint not found: {checkpoint}"
        )
    encoder = TelemetryEncoder.from_encoder_checkpoint(
        distill_config.phase1_ckpt, device=device
    )
    projector = CoreActionProjector(encoder)
    actor_critic = RecurrentActorCritic.from_distilled(projector)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    actor_critic.load_state_dict(state["actor_critic_state"], strict=True)
    actor_critic.to(device)
    actor_critic.set_ppo_mode(training=False)
    return actor_critic, checkpoint


def _validate_buffer(buffer: RecurrentRolloutBuffer, expected_steps: int) -> None:
    if buffer.pos != expected_steps:
        raise RuntimeError(
            f"Invariant rollout collected {buffer.pos}/{expected_steps} steps"
        )
    for name in (
        "obs",
        "actions",
        "rewards",
        "values",
        "log_probs",
        "advantages",
        "returns",
    ):
        tensor = getattr(buffer, name)[: buffer.pos]
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Invariant buffer field {name} is non-finite")
    binary = buffer.actions[: buffer.pos, 4:8]
    if not torch.all((binary == 0.0) | (binary == 1.0)):
        raise RuntimeError("Invariant buffer contains non-binary actions")


def run_invariant_only(
    config: PPOTrainConfig,
    distill_config: DistillConfig,
) -> dict:
    """Collect and replay a rollout without constructing an optimizer."""
    if config.enable_ppo:
        raise ValueError("invariant_only requires enable_ppo:false")
    device = torch.device(config.device)
    actor_critic, checkpoint = load_recurrent_policy(distill_config, device)
    reference_policy = deepcopy(actor_critic)
    reference_policy.set_ppo_mode(training=False)
    for parameter in reference_policy.parameters():
        parameter.requires_grad_(False)

    reward_builder = RewardBuilder(
        dmg_dealt_idx=None,
        dmg_taken_idx=None,
        env_reward_coef=config.env_reward_coef,
        damage_dealt_coef=0.0,
        damage_taken_coef=0.0,
        kill_coef=config.kill_coef,
        death_coef=config.death_coef,
        miss_coef=0.0,
        jerk_coef=config.jerk_coef,
        spam_coef=config.spam_coef,
    )
    from .obs_normalizer import ObservationNormalizer

    environment = _setup_unity_env(config.env_path)
    try:
        runner = SelfPlayRunner(
            env=environment,
            reward_builder=reward_builder,
            device=config.device,
            normalizer=ObservationNormalizer(config.telemetry_stats_path),
        )
        buffer = RecurrentRolloutBuffer(
            size=config.rollout_smoke_steps,
            device=config.device,
        )
        rollout_metrics = runner.collect_rollouts(
            policy=actor_critic,
            buffer=buffer,
            n_steps=config.rollout_smoke_steps,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        _validate_buffer(buffer, config.rollout_smoke_steps)

        actor_critic.set_ppo_mode(training=True)
        ratios: list[torch.Tensor] = []
        max_deltas: list[float] = []
        clip_fractions: list[float] = []
        batches = 0
        with torch.no_grad():
            for batch in buffer.recurrent_minibatches(
                seq_len=config.seq_len,
                batch_size=config.batch_size,
            ):
                distribution, _, _ = actor_critic(batch["obs"], batch["hiddens"])
                new_log_prob = distribution.log_prob(batch["actions"])
                delta = new_log_prob - batch["log_probs"]
                ratio = torch.exp(delta)
                ratios.append(ratio.reshape(-1).cpu())
                max_deltas.append(float(delta.abs().max().item()))
                clip_fractions.append(
                    float((ratio - 1.0).abs().gt(config.ppo_clip).float().mean().item())
                )
                batches += 1
        if not batches:
            raise RuntimeError(
                "No complete recurrent minibatches were available for invariant replay"
            )
        ratio_values = torch.cat(ratios)
        actions = buffer.actions[: buffer.pos]
        continuous_raw = actions[:, :4]
        continuous_sent = continuous_raw.clamp(-1.0, 1.0)
        binary = actions[:, 4:8]
        behavior_spec = environment.behavior_specs[runner.behavior_name]
        observation_shapes = [
            tuple(spec.shape) for spec in behavior_spec.observation_specs
        ]
        report = {
            "stage": "invariant_only",
            "checkpoint_loaded": str(checkpoint.resolve()),
            "no_optimizer_step": True,
            "frozen_reference_created": True,
            "clip_fraction": max(clip_fractions),
            "max_abs_delta_logp": max(max_deltas),
            "ratio_mean": float(ratio_values.mean().item()),
            "ratio_std": float(ratio_values.std(unbiased=False).item()),
            "ratio_min": float(ratio_values.min().item()),
            "ratio_max": float(ratio_values.max().item()),
            "logp_finite": bool(
                torch.isfinite(buffer.log_probs[: buffer.pos]).all().item()
            ),
            "continuous_raw": {
                "min": continuous_raw.min(dim=0).values.cpu().tolist(),
                "max": continuous_raw.max(dim=0).values.cpu().tolist(),
                "mean": continuous_raw.mean(dim=0).cpu().tolist(),
                "std": continuous_raw.std(dim=0, unbiased=False).cpu().tolist(),
            },
            "continuous_sent": {
                "min": continuous_sent.min(dim=0).values.cpu().tolist(),
                "max": continuous_sent.max(dim=0).values.cpu().tolist(),
                "mean": continuous_sent.mean(dim=0).cpu().tolist(),
                "std": continuous_sent.std(dim=0, unbiased=False).cpu().tolist(),
            },
            "binary_action_rates": binary.mean(dim=0).cpu().tolist(),
            "behavior_spec": {
                "continuous_size": int(behavior_spec.action_spec.continuous_size),
                "discrete_branches": [
                    int(value) for value in behavior_spec.action_spec.discrete_branches
                ],
                "observation_shapes": observation_shapes,
            },
            "reward_damage_decision": {
                "option": "a",
                "source": "env_reward",
                "require_damage_signals": False,
                "observation_damage_indices_enabled": False,
                "obs_dim": 45,
                "unity_schema_changed": False,
            },
            **rollout_metrics,
        }
        if report["clip_fraction"] > config.max_preupdate_clip_frac:
            raise RuntimeError(
                f"Pre-update invariant failed: clip_fraction={report['clip_fraction']}"
            )
        target = Path(distill_config.log_dir) / "invariant_report.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("Invariant-only gate passed:\n%s", json.dumps(report, indent=2))
        return report
    finally:
        environment.close()
