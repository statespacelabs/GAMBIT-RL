"""Phase 3 repair-gated distillation, Unity smoke rollout, and PPO training."""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from gambit.dataset_preparation.configs import DistillConfig, PPOTrainConfig
from .action_schema import ActionCommandConverter
from .actor_critic import RecurrentActorCritic
from .checkpoint_registry import CheckpointRegistry
from .core_action_projector import (
    CoreActionDistiller,
    CoreActionProjector,
    RecurrentDistiller,
)
from .evaluate_policy import evaluate_policy, should_promote
from .opponent_pool import OpponentPool
from .ppo_loss import compute_ppo_loss, compute_reference_anchor
from .reward_builder import RewardBuilder
from .rollout_buffer import RecurrentRolloutBuffer
from .selfplay_runner import SelfPlayRunner
from .telemetry_encoder import TelemetryEncoder
from .vectorized_buffer import VectorizedRolloutBuffer
from .vectorized_runner import VectorizedUnityRunner
from .multi_area_runner import MultiAreaUnityRunner
from .lr_schedule import get_scheduled_lr
from .visual_runner import VisualMultiAreaUnityRunner
from .visual_actor_critic import VisualRecurrentActorCritic
from .obs_normalizer import ObservationNormalizer
from .distributed_utils import (
    barrier,
    cleanup_distributed,
    init_distributed,
    is_main_process,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _apply_scripted_shoot_env(config: PPOTrainConfig) -> None:
    """Export scripted-bot shoot pressure knobs to Unity via environment variables."""
    os.environ["SCRIPTED_SHOOT_DAMAGE_SCALE"] = str(config.scripted_shoot_damage_scale)
    os.environ["SCRIPTED_SHOOT_COOLDOWN_MULT"] = str(config.scripted_shoot_cooldown_mult)
    os.environ["SCRIPTED_SHOOT_AIM_THRESHOLD_DEG"] = str(
        config.scripted_shoot_aim_threshold_deg
    )
    if config.scripted_shoot_warmup_iters > 0:
        os.environ["SCRIPTED_SHOOT_WARMUP_ITERS"] = str(config.scripted_shoot_warmup_iters)
        os.environ["SCRIPTED_SHOOT_WARMUP_SEC_PER_ITER"] = str(
            config.scripted_shoot_warmup_sec_per_iter
        )
    else:
        os.environ.pop("SCRIPTED_SHOOT_WARMUP_ITERS", None)
        os.environ.pop("SCRIPTED_SHOOT_WARMUP_SEC_PER_ITER", None)


def _apply_phase3ac_env(config: PPOTrainConfig) -> None:
    """Export Phase 3AC weapon/reset debug flags to Unity."""
    if config.force_match_reset_on_episode_begin:
        os.environ["FORCE_MATCH_RESET_ON_EPISODE_BEGIN"] = "1"
    else:
        os.environ.pop("FORCE_MATCH_RESET_ON_EPISODE_BEGIN", None)
    if config.force_weapon_reset_on_episode_begin:
        os.environ["FORCE_WEAPON_RESET_ON_EPISODE_BEGIN"] = "1"
    else:
        os.environ.pop("FORCE_WEAPON_RESET_ON_EPISODE_BEGIN", None)
    if config.force_weapon_reset_on_respawn:
        os.environ["FORCE_WEAPON_RESET_ON_RESPAWN"] = "1"
    else:
        os.environ.pop("FORCE_WEAPON_RESET_ON_RESPAWN", None)
    if config.debug_infinite_ammo:
        os.environ["DEBUG_INFINITE_AMMO"] = "1"
    else:
        os.environ.pop("DEBUG_INFINITE_AMMO", None)
    if config.debug_disable_reload:
        os.environ["DEBUG_DISABLE_RELOAD"] = "1"
    else:
        os.environ.pop("DEBUG_DISABLE_RELOAD", None)
    if config.debug_force_can_fire_if_cooldown_ready:
        os.environ["DEBUG_FORCE_CAN_FIRE_IF_COOLDOWN_READY"] = "1"
    else:
        os.environ.pop("DEBUG_FORCE_CAN_FIRE_IF_COOLDOWN_READY", None)


def setup_unity_env(
    env_path: str,
    time_scale: float = 20.0,
    worker_id: int = 0,
    base_port: int = 5005,
    seed: int = 42,
):
    """Create and reset the ML-Agents environment."""
    try:
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.engine_configuration_channel import (
            EngineConfigurationChannel,
        )
    except ImportError as exc:
        raise RuntimeError(
            "mlagents_envs is required for Unity rollout and PPO training"
        ) from exc

    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(time_scale=time_scale)
    env = UnityEnvironment(
        file_name=env_path if env_path else None,
        worker_id=worker_id,
        base_port=base_port,
        seed=seed,
        side_channels=[engine_channel],
        no_graphics=True,
    )
    env.reset()
    for behavior_name, spec in env.behavior_specs.items():
        logger.info(
            "Unity behavior %s: observation_specs=%s action_spec=%s",
            behavior_name,
            spec.observation_specs,
            spec.action_spec,
        )
    return env


def _validate_local_inputs(
    config: PPOTrainConfig,
    distill_config: DistillConfig,
) -> None:
    required = {
        "Phase 1 checkpoint": distill_config.phase1_ckpt,
        "transition manifest": distill_config.transition_manifest_path,
        "window manifest": distill_config.window_manifest_path,
    }
    missing = [
        f"{name}: {path}" for name, path in required.items() if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError("Missing Phase 3 inputs:\n" + "\n".join(missing))


def _validate_rollout_buffer(
    buffer: RecurrentRolloutBuffer,
    expected_steps: int,
) -> None:
    if buffer.pos != expected_steps:
        raise RuntimeError(
            f"Smoke rollout collected {buffer.pos}/{expected_steps} steps"
        )
    fields = {
        "obs": buffer.obs[: buffer.pos],
        "actions": buffer.actions[: buffer.pos],
        "rewards": buffer.rewards[: buffer.pos],
        "values": buffer.values[: buffer.pos],
        "log_probs": buffer.log_probs[: buffer.pos],
        "advantages": buffer.advantages[: buffer.pos],
        "returns": buffer.returns[: buffer.pos],
    }
    for name, tensor in fields.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Smoke rollout buffer field {name} is nonfinite")
    binary = buffer.actions[: buffer.pos, 4:8]
    if not torch.all((binary == 0.0) | (binary == 1.0)):
        raise RuntimeError("Smoke rollout contains non-binary command actions")


def _build_action_converter(config: DistillConfig) -> ActionCommandConverter:
    return ActionCommandConverter(
        max_move_speed=config.max_move_speed,
        max_yaw_delta=config.max_yaw_delta,
        max_pitch_delta=config.max_pitch_delta,
    )


def train_ppo(config: PPOTrainConfig, distill_config: DistillConfig) -> None:
    """Run repaired distillation, a mandatory smoke rollout, then optional PPO."""
    _validate_local_inputs(config, distill_config)
    device = torch.device(config.device)
    action_converter = _build_action_converter(distill_config)

    logger.info(
        "Loading pretrained telemetry branches from %s",
        distill_config.phase1_ckpt,
    )
    telemetry_encoder = TelemetryEncoder.from_encoder_checkpoint(
        distill_config.phase1_ckpt,
        device=device,
    )

    checkpoint_dir = Path(distill_config.checkpoint_dir)
    stage1_checkpoint = checkpoint_dir / "command_best.pt"
    stage2_checkpoint = checkpoint_dir / "command_recurrent_best.pt"

    projector = CoreActionProjector(telemetry_encoder)
    stage1 = CoreActionDistiller(
        projector=projector,
        transition_manifest_path=distill_config.transition_manifest_path,
        window_manifest_path=distill_config.window_manifest_path,
        action_converter=action_converter,
        telemetry_fps=distill_config.telemetry_fps,
        step_seconds=distill_config.action_step_seconds,
        require_normalized_telemetry=distill_config.require_normalized_telemetry,
        action_scaler_path=None,
        binary_pos_weight=distill_config.binary_pos_weight,
        lr=distill_config.lr,
        weight_decay=distill_config.weight_decay,
        batch_size=distill_config.batch_size,
        epochs=distill_config.epochs,
        warmup_epochs=distill_config.warmup_epochs,
        grad_clip_norm=distill_config.grad_clip_norm,
        num_workers=distill_config.num_workers,
        checkpoint_dir=distill_config.checkpoint_dir,
        log_dir=distill_config.log_dir,
        device=config.device,
    )
    if stage1_checkpoint.exists() and not distill_config.force_retrain:
        state = torch.load(stage1_checkpoint, map_location=device, weights_only=False)
        projector.load_state_dict(state["projector_state"])
        projector.to(device)
        logger.info("Loaded command-distilled projector from %s", stage1_checkpoint)
    else:
        projector = stage1.train()

    actor_critic = RecurrentActorCritic.from_distilled(projector)
    stage2 = RecurrentDistiller(
        actor_critic=actor_critic,
        transition_manifest_path=distill_config.transition_manifest_path,
        window_manifest_path=distill_config.window_manifest_path,
        action_converter=action_converter,
        telemetry_fps=distill_config.telemetry_fps,
        step_seconds=distill_config.action_step_seconds,
        require_normalized_telemetry=distill_config.require_normalized_telemetry,
        action_scaler_path=None,
        binary_pos_weight=distill_config.binary_pos_weight,
        lr=distill_config.lr,
        weight_decay=distill_config.weight_decay,
        batch_size=distill_config.batch_size,
        seq_len=config.seq_len,
        epochs=distill_config.recurrent_epochs,
        grad_clip_norm=distill_config.grad_clip_norm,
        num_workers=distill_config.num_workers,
        checkpoint_dir=distill_config.checkpoint_dir,
        log_dir=distill_config.log_dir,
        device=config.device,
    )
    if stage2_checkpoint.exists() and not distill_config.force_retrain:
        state = torch.load(stage2_checkpoint, map_location=device, weights_only=False)
        actor_critic.load_state_dict(state["actor_critic_state"])
        logger.info("Loaded recurrent command warmup from %s", stage2_checkpoint)
    else:
        actor_critic = stage2.train()
    actor_critic.to(device)

    reference_policy = deepcopy(actor_critic)
    reference_policy.set_ppo_mode(training=False)
    for parameter in reference_policy.parameters():
        parameter.requires_grad_(False)

    if config.freeze_pretrained_branches:
        actor_critic.tel_encoder.freeze_branches()
    actor_critic.set_ppo_mode(training=True)

    reward_builder = RewardBuilder(
        dmg_dealt_idx=config.damage_dealt_obs_index,
        dmg_taken_idx=config.damage_taken_obs_index,
        env_reward_coef=config.env_reward_coef,
        damage_dealt_coef=config.damage_dealt_coef,
        damage_taken_coef=config.damage_taken_coef,
        kill_coef=config.kill_coef,
        death_coef=config.death_coef,
        miss_coef=config.miss_coef,
        jerk_coef=config.jerk_coef,
        spam_coef=config.spam_coef,
        aim_coef=config.aim_coef,
        aim_err_max=config.aim_err_max,
        aim_reward_mode=config.aim_reward_mode,
        bad_aim_threshold_deg=config.bad_aim_threshold_deg,
        bad_aim_coef=config.bad_aim_coef,
        bad_aim_scale_deg=config.bad_aim_scale_deg,
        aim_sigma_deg=config.aim_sigma_deg,
        use_true_aim_for_reward=config.use_true_aim_for_reward,
        true_aim_err_max=config.true_aim_err_max,
        true_aim_obs_index=config.true_aim_obs_index,
        shoot_bootstrap_metric=config.shoot_bootstrap_metric,
        true_shoot_bootstrap_threshold_deg=config.true_shoot_bootstrap_threshold_deg,
        shoot_bootstrap_coef=config.shoot_bootstrap_coef,
        shoot_bootstrap_aim_threshold_deg=config.shoot_bootstrap_aim_threshold_deg,
        shoot_bootstrap_require_fired=config.shoot_bootstrap_require_fired,
        shoot_bootstrap_require_los=config.shoot_bootstrap_require_los,
        shot_fired_obs_index=config.shot_fired_obs_index,
        damage_chain_coef=config.damage_chain_coef,
        damage_chain_window_sec=config.damage_chain_window_sec,
        damage_chain_window_steps=config.damage_chain_window_steps,
        learner_hit_env_reward_threshold=config.learner_hit_env_reward_threshold,
        hp_delta_reward_coef=config.hp_delta_reward_coef,
        hp_delta_reward_cap=config.hp_delta_reward_cap,
        opponent_hp_frac_obs_index=config.opponent_hp_frac_obs_index,
        hp_frac_to_damage_scale=config.hp_frac_to_damage_scale,
    )
    env = setup_unity_env(config.env_path)
    try:
        runner = SelfPlayRunner(
            env=env,
            reward_builder=reward_builder,
            device=config.device,
            normalizer=ObservationNormalizer(
                config.telemetry_stats_path, clip_value=config.obs_clip_value,
            ),
        )

        smoke_buffer = RecurrentRolloutBuffer(
            size=config.rollout_smoke_steps,
            device=config.device,
        )
        smoke_metrics = runner.collect_rollouts(
            policy=actor_critic,
            buffer=smoke_buffer,
            n_steps=config.rollout_smoke_steps,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        _validate_rollout_buffer(smoke_buffer, config.rollout_smoke_steps)
        logger.info(
            "%d-step rollout smoke test passed: %s",
            config.rollout_smoke_steps,
            smoke_metrics,
        )

        if not config.enable_ppo:
            logger.info(
                "PPO is disabled by config. Repair milestone completed after "
                "the valid Unity smoke rollout."
            )
            return

        if config.require_damage_signals and not reward_builder.has_damage_signals:
            raise ValueError(
                "PPO reward requires damage_dealt_obs_index and "
                "damage_taken_obs_index. Set them from the real Unity "
                "observation contract before enabling PPO."
            )

        registry = CheckpointRegistry(config.registry_dir)
        opponent_pool = OpponentPool(registry)
        registry.register(
            0,
            {"win_rate": 0.5, "damage_ratio": 1.0},
            actor_critic.state_dict(),
        )
        buffer = RecurrentRolloutBuffer(
            size=config.rollout_steps,
            device=config.device,
        )
        optimizer = torch.optim.AdamW(
            [p for p in actor_critic.parameters() if p.requires_grad],
            lr=config.lr,
            weight_decay=1e-4,
        )
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "ppo_train.jsonl"

        for iteration in range(1, config.num_iterations + 1):
            # Frozen checkpoint opponents are not yet controlled from Python.
            # Keep the first online milestone against the scripted Unity bot.
            _ = opponent_pool
            rollout_metrics = runner.collect_rollouts(
                policy=actor_critic,
                buffer=buffer,
                n_steps=config.rollout_steps,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )

            actor_critic.set_ppo_mode(training=True)
            total_loss = 0.0
            updates = 0
            preupdate_checked = False
            latest_loss_metrics: dict[str, float] = {}

            for epoch in range(config.ppo_epochs):
                for batch in buffer.recurrent_minibatches(
                    seq_len=config.seq_len,
                    batch_size=config.batch_size,
                ):
                    obs = batch["obs"]
                    actions = batch["actions"]
                    hidden = batch["hiddens"]
                    current_dist, new_values, _ = actor_critic(obs, hidden)
                    new_log_probs = current_dist.log_prob(actions)
                    entropy = current_dist.entropy()

                    if not preupdate_checked:
                        ratio = torch.exp(new_log_probs.detach() - batch["log_probs"])
                        clip_fraction = (
                            (torch.abs(ratio - 1.0) > config.ppo_clip)
                            .float()
                            .mean()
                            .item()
                        )
                        ratio_std = ratio.std(unbiased=False).item()
                        logger.info(
                            "Pre-update ratio check: mean=%.8f std=%.8f "
                            "clip_fraction=%.8f",
                            ratio.mean().item(),
                            ratio_std,
                            clip_fraction,
                        )
                        if clip_fraction > config.max_preupdate_clip_frac:
                            raise RuntimeError(
                                "PPO old/new log-prob invariant failed before "
                                f"optimization: clip_fraction={clip_fraction}"
                            )
                        preupdate_checked = True

                    anchor_hidden = torch.zeros_like(hidden)
                    anchor_current, _, _ = actor_critic(obs, anchor_hidden)
                    with torch.no_grad():
                        anchor_reference, _, _ = reference_policy(obs, anchor_hidden)
                    reference_anchor = compute_reference_anchor(
                        anchor_current,
                        anchor_reference,
                    )

                    loss, latest_loss_metrics = compute_ppo_loss(
                        new_log_probs=new_log_probs,
                        old_log_probs=batch["log_probs"],
                        advantages=batch["advantages"],
                        new_values=new_values,
                        returns=batch["returns"],
                        entropy=entropy,
                        reference_anchor=reference_anchor,
                        current_cont_mean=current_dist.cont_mean,
                        clip_ratio=config.ppo_clip,
                        value_coef=config.value_coef,
                        entropy_coef=config.entropy_coef,
                        kl_coef=config.kl_anchor_coef,
                        smooth_coef=config.smooth_coef,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        actor_critic.parameters(), config.max_grad_norm
                    )
                    optimizer.step()
                    total_loss += loss.item()
                    last_grad_norm = float(grad_norm)
                    updates += 1

            average_loss = total_loss / max(updates, 1)

            # ---- Post-update diagnostics (no grad): per-iteration policy movement ----
            kls, cfs, rmins, rmaxs, rmeans = [], [], [], [], []
            with torch.no_grad():
                for b in buffer.recurrent_minibatches(
                    seq_len=config.seq_len, batch_size=config.batch_size
                ):
                    d, _, _ = actor_critic(b["obs"], b["hiddens"])
                    nlp = d.log_prob(b["actions"])
                    r = torch.exp(nlp - b["log_probs"])
                    kls.append((b["log_probs"] - nlp).mean().item())
                    cfs.append((torch.abs(r - 1.0) > config.ppo_clip).float().mean().item())
                    rmins.append(r.min().item()); rmaxs.append(r.max().item()); rmeans.append(r.mean().item())
            _avg = lambda xs: float(sum(xs) / len(xs)) if xs else 0.0
            T = buffer.pos
            obs_b, act_b = buffer.obs[:T], buffer.actions[:T]
            adv_b, ret_b = buffer.advantages[:T], buffer.returns[:T]
            cont, binr = act_b[:, :4], act_b[:, 4:8]
            diag = {
                "approx_kl": _avg(kls),
                "postupdate_clip_frac": _avg(cfs),
                "ratio_min": float(min(rmins)) if rmins else 1.0,
                "ratio_mean": _avg(rmeans),
                "ratio_max": float(max(rmaxs)) if rmaxs else 1.0,
                "grad_norm": float(last_grad_norm) if "last_grad_norm" in locals() else 0.0,
                "obs_all_zero": bool(torch.all(obs_b == 0).item()),
                "obs_mean": float(obs_b.mean()), "obs_std": float(obs_b.std()),
                "obs_min": float(obs_b.min()), "obs_max": float(obs_b.max()),
                "adv_mean": float(adv_b.mean()), "adv_std": float(adv_b.std()),
                "ret_mean": float(ret_b.mean()), "ret_std": float(ret_b.std()),
                "cont_mean": float(cont.mean()), "cont_std": float(cont.std()),
                "rate_shoot": float((binr[:, 0] > 0.5).float().mean()),
                "rate_reload": float((binr[:, 1] > 0.5).float().mean()),
                "rate_jump": float((binr[:, 2] > 0.5).float().mean()),
                "rate_crouch": float((binr[:, 3] > 0.5).float().mean()),
                "finite": bool(torch.isfinite(adv_b).all() and torch.isfinite(ret_b).all()),
            }
            logger.info(
                "ITER %d | approx_kl=%.4f postclip=%.3f ratio[min=%.3f mean=%.3f max=%.3f] "
                "grad_norm=%.3f obs0=%s obs[mu=%.3f sd=%.3f rng=%.2f..%.2f] "
                "adv[mu=%.3f sd=%.3f] ret[mu=%.3f sd=%.3f] cont[mu=%.3f sd=%.3f] "
                "rates[s=%.2f r=%.2f j=%.2f c=%.2f] loss[p=%.3f v=%.3f ent=%.3f anc=%.3f]",
                iteration, diag["approx_kl"], diag["postupdate_clip_frac"],
                diag["ratio_min"], diag["ratio_mean"], diag["ratio_max"], diag["grad_norm"],
                diag["obs_all_zero"], diag["obs_mean"], diag["obs_std"], diag["obs_min"], diag["obs_max"],
                diag["adv_mean"], diag["adv_std"], diag["ret_mean"], diag["ret_std"],
                diag["cont_mean"], diag["cont_std"],
                diag["rate_shoot"], diag["rate_reload"], diag["rate_jump"], diag["rate_crouch"],
                latest_loss_metrics.get("loss/policy", 0.0), latest_loss_metrics.get("loss/value", 0.0),
                latest_loss_metrics.get("loss/entropy", 0.0),
                latest_loss_metrics.get("loss/reference_anchor", 0.0),
            )
            eval_metrics: dict[str, float] = {}
            if iteration % config.eval_interval == 0:
                eval_metrics = evaluate_policy(
                    policy=actor_critic,
                    env=env,
                    n_episodes=10,
                    device=config.device,
                )
                if should_promote(
                    eval_metrics,
                    threshold_win_rate=config.promote_win_rate,
                ):
                    registry.register(
                        step=iteration * config.rollout_steps,
                        metrics=eval_metrics,
                        state_dict=actor_critic.state_dict(),
                    )

            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "iteration": iteration,
                            "loss": average_loss,
                            **latest_loss_metrics,
                            **diag,
                            **rollout_metrics,
                            **eval_metrics,
                        }
                    )
                    + "\n"
                )
    finally:
        env.close()


def _behavioral_checkpoint_payload(
    iteration: int,
    actor_critic: RecurrentActorCritic,
    *,
    bscore: float,
    hit_count: float,
    kill_count: float,
    aim_err: float,
) -> dict:
    return {
        "iteration": iteration,
        "actor_critic_state": actor_critic.state_dict(),
        "bscore": bscore,
        "hit_count": hit_count,
        "kill_count": kill_count,
        "aim_err_mean": aim_err,
    }


def _save_behavioral_checkpoint(
    path: Path,
    iteration: int,
    actor_critic: RecurrentActorCritic,
    *,
    bscore: float,
    hit_count: float,
    kill_count: float,
    aim_err: float,
    tag: str,
) -> None:
    torch.save(
        _behavioral_checkpoint_payload(
            iteration,
            actor_critic,
            bscore=bscore,
            hit_count=hit_count,
            kill_count=kill_count,
            aim_err=aim_err,
        ),
        path,
    )
    logger.info(
        "%s iter=%d bscore=%.2f hits=%.0f kills=%.0f aim=%.1f -> %s",
        tag,
        iteration,
        bscore,
        hit_count,
        kill_count,
        aim_err,
        path,
    )


def _validate_vectorized_buffer(
    buffer: VectorizedRolloutBuffer,
    expected_steps: int,
    num_envs: int,
) -> None:
    """Validate vectorized buffer after rollout collection."""
    if buffer.pos != expected_steps:
        raise RuntimeError(
            f"Vectorized rollout collected {buffer.pos}/{expected_steps} steps"
        )
    T = buffer.pos
    fields = {
        "obs": buffer.obs[:T],
        "actions": buffer.actions[:T],
        "rewards": buffer.rewards[:T],
        "values": buffer.values[:T],
        "log_probs": buffer.log_probs[:T],
        "advantages": buffer.advantages[:T],
        "returns": buffer.returns[:T],
    }
    for name, tensor in fields.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Vectorized buffer field {name} is nonfinite")
    binary = buffer.actions[:T, :, 4:8]
    if not torch.all((binary == 0.0) | (binary == 1.0)):
        raise RuntimeError("Vectorized rollout contains non-binary command actions")


def _per_agent_metrics(rollout_metrics: dict[str, float], num_agents: int = 2) -> dict[str, float]:
    """Map multi-area per_area metrics to agent_a/agent_b for GvG smoke."""
    labels = ["a", "b"]
    out: dict[str, float] = {}
    for i in range(min(num_agents, len(labels))):
        lbl = labels[i]
        prefix = f"per_area/area{i}_"
        out[f"agent_{lbl}_shots"] = float(rollout_metrics.get(f"{prefix}shots", 0.0))
        out[f"agent_{lbl}_hits"] = float(rollout_metrics.get(f"{prefix}hits", 0.0))
        out[f"agent_{lbl}_kills"] = float(rollout_metrics.get(f"{prefix}kills", 0.0))
        out[f"agent_{lbl}_aim_mean"] = float(rollout_metrics.get(f"{prefix}aim_err", 0.0))
        out[f"agent_{lbl}_env_reward_sum"] = float(
            rollout_metrics.get(f"{prefix}env_reward_sum", 0.0)
        )
        out[f"agent_{lbl}_resets"] = float(rollout_metrics.get(f"{prefix}resets", 0.0))
    out["num_decisions_a"] = out.get("agent_a_shots", 0.0)
    out["num_decisions_b"] = out.get("agent_b_shots", 0.0)
    return out


def _check_gvg_smoke_stop(
    iteration: int,
    rollout_metrics: dict[str, float],
    config: PPOTrainConfig,
    fn: object,
) -> bool:
    """Return True if GvG smoke should stop early."""
    if not config.gvg_smoke:
        return False

    agent_metrics = _per_agent_metrics(rollout_metrics, num_agents=2)
    shots_a = agent_metrics.get("agent_a_shots", 0.0)
    shots_b = agent_metrics.get("agent_b_shots", 0.0)
    hits_a = agent_metrics.get("agent_a_hits", 0.0)
    hits_b = agent_metrics.get("agent_b_hits", 0.0)
    aim_a = agent_metrics.get("agent_a_aim_mean", 0.0)
    aim_b = agent_metrics.get("agent_b_aim_mean", 0.0)

    if not hasattr(fn, "_gvg_zero_shot_a"):
        fn._gvg_zero_shot_a = 0  # type: ignore[attr-defined]
        fn._gvg_zero_shot_b = 0  # type: ignore[attr-defined]
        fn._gvg_both_zero_hits = 0  # type: ignore[attr-defined]

    fn._gvg_zero_shot_a = fn._gvg_zero_shot_a + 1 if shots_a <= 0 else 0  # type: ignore[attr-defined]
    fn._gvg_zero_shot_b = fn._gvg_zero_shot_b + 1 if shots_b <= 0 else 0  # type: ignore[attr-defined]
    if hits_a <= 0 and hits_b <= 0:
        fn._gvg_both_zero_hits += 1  # type: ignore[attr-defined]
    else:
        fn._gvg_both_zero_hits = 0  # type: ignore[attr-defined]

    if fn._gvg_zero_shot_a >= config.gvg_zero_shot_streak_stop or fn._gvg_zero_shot_b >= config.gvg_zero_shot_streak_stop:  # type: ignore[attr-defined]
        logger.info(
            "GVG_STOP iter=%d reason=zero_shots streak_a=%d streak_b=%d",
            iteration,
            fn._gvg_zero_shot_a,  # type: ignore[attr-defined]
            fn._gvg_zero_shot_b,  # type: ignore[attr-defined]
        )
        return True
    if fn._gvg_both_zero_hits >= config.gvg_both_zero_hits_streak_stop:  # type: ignore[attr-defined]
        logger.info(
            "GVG_STOP iter=%d reason=both_zero_hits streak=%d",
            iteration,
            config.gvg_both_zero_hits_streak_stop,
        )
        return True
    total_shots = float(rollout_metrics.get("aim/total_shots", 0.0))
    if config.gvg_min_total_shots_per_iter > 0 and total_shots < config.gvg_min_total_shots_per_iter:
        if not hasattr(fn, "_gvg_low_shot_iters"):
            fn._gvg_low_shot_iters = 0  # type: ignore[attr-defined]
        fn._gvg_low_shot_iters += 1  # type: ignore[attr-defined]
        if fn._gvg_low_shot_iters >= config.gvg_zero_shot_streak_stop:  # type: ignore[attr-defined]
            logger.info(
                "GVG_STOP iter=%d reason=low_shots total=%.0f min=%.0f",
                iteration, total_shots, config.gvg_min_total_shots_per_iter,
            )
            return True
    else:
        if hasattr(fn, "_gvg_low_shot_iters"):
            fn._gvg_low_shot_iters = 0  # type: ignore[attr-defined]
    if iteration > config.gvg_aim_drift_start_iter and (
        aim_a > config.gvg_aim_drift_stop_deg or aim_b > config.gvg_aim_drift_stop_deg
    ):
        logger.info(
            "GVG_STOP iter=%d reason=aim_drift aim_a=%.1f aim_b=%.1f",
            iteration, aim_a, aim_b,
        )
        return True
    return False


def _check_preserve_stop(
    iteration: int,
    hit_count: float,
    aim_err: float,
    config: PPOTrainConfig,
    fn: object,
) -> bool:
    """Return True if preservation PPO should stop."""
    if not config.preserve_stop_enabled:
        return False

    if iteration == 1:
        fn._preserve_iter1_hits = float(hit_count)  # type: ignore[attr-defined]
        return False

    if iteration < config.preserve_stop_start_iter:
        return False

    if not hasattr(fn, "_preserve_zero_hit_iters"):
        fn._preserve_zero_hit_iters = 0  # type: ignore[attr-defined]

    iter1_hits = getattr(fn, "_preserve_iter1_hits", 0.0)  # type: ignore[attr-defined]
    recent = list(fn._recent_history)  # type: ignore[attr-defined]
    window = recent[-min(5, len(recent)):]
    mean_recent_aim = sum(r["aim_err"] for r in window) / max(len(window), 1)
    mean_recent_hits = sum(r["hits"] for r in window) / max(len(window), 1)

    if hit_count <= 0:
        fn._preserve_zero_hit_iters += 1  # type: ignore[attr-defined]
    else:
        fn._preserve_zero_hit_iters = 0  # type: ignore[attr-defined]

    if mean_recent_aim > config.preserve_stop_aim_threshold_deg:
        logger.info(
            "PRESERVE_STOP iter=%d reason=AIM_DRIFT mean_aim=%.1f threshold=%.1f",
            iteration, mean_recent_aim, config.preserve_stop_aim_threshold_deg,
        )
        return True
    if iter1_hits > 0 and mean_recent_hits < config.preserve_stop_min_hits_fraction_of_iter1 * iter1_hits:
        logger.info(
            "PRESERVE_STOP iter=%d reason=HIT_RETENTION mean_hits=%.2f iter1=%.0f frac=%.2f",
            iteration, mean_recent_hits, iter1_hits,
            config.preserve_stop_min_hits_fraction_of_iter1,
        )
        return True
    if fn._preserve_zero_hit_iters >= config.preserve_stop_zero_hits_iters:  # type: ignore[attr-defined]
        logger.info(
            "PRESERVE_STOP iter=%d reason=ZERO_HITS streak=%d",
            iteration, fn._preserve_zero_hit_iters,  # type: ignore[attr-defined]
        )
        return True
    return False


def train_ppo_vectorized(config: PPOTrainConfig, distill_config: DistillConfig) -> None:
    """Run distillation warmup, then vectorized multi-env PPO."""
    _validate_local_inputs(config, distill_config)
    device = torch.device(config.device)
    action_converter = _build_action_converter(distill_config)

    logger.info(
        "Loading pretrained telemetry branches from %s",
        distill_config.phase1_ckpt,
    )
    telemetry_encoder = TelemetryEncoder.from_encoder_checkpoint(
        distill_config.phase1_ckpt, device=device,
    )

    checkpoint_dir = Path(distill_config.checkpoint_dir)
    stage1_checkpoint = checkpoint_dir / "command_best.pt"
    stage2_checkpoint = checkpoint_dir / "command_recurrent_best.pt"

    projector = CoreActionProjector(telemetry_encoder)
    if stage1_checkpoint.exists() and not distill_config.force_retrain:
        state = torch.load(stage1_checkpoint, map_location=device, weights_only=False)
        projector.load_state_dict(state["projector_state"])
        projector.to(device)
        logger.info("Loaded command-distilled projector from %s", stage1_checkpoint)
    else:
        stage1 = CoreActionDistiller(
            projector=projector,
            transition_manifest_path=distill_config.transition_manifest_path,
            window_manifest_path=distill_config.window_manifest_path,
            action_converter=action_converter,
            telemetry_fps=distill_config.telemetry_fps,
            step_seconds=distill_config.action_step_seconds,
            require_normalized_telemetry=distill_config.require_normalized_telemetry,
            action_scaler_path=None,
            binary_pos_weight=distill_config.binary_pos_weight,
            lr=distill_config.lr,
            weight_decay=distill_config.weight_decay,
            batch_size=distill_config.batch_size,
            epochs=distill_config.epochs,
            warmup_epochs=distill_config.warmup_epochs,
            grad_clip_norm=distill_config.grad_clip_norm,
            num_workers=distill_config.num_workers,
            checkpoint_dir=distill_config.checkpoint_dir,
            log_dir=distill_config.log_dir,
            device=config.device,
        )
        projector = stage1.train()

    actor_critic = RecurrentActorCritic.from_distilled(projector)
    if stage2_checkpoint.exists() and not distill_config.force_retrain:
        state = torch.load(stage2_checkpoint, map_location=device, weights_only=False)
        actor_critic.load_state_dict(state["actor_critic_state"])
        logger.info("Loaded recurrent command warmup from %s", stage2_checkpoint)
    else:
        stage2 = RecurrentDistiller(
            actor_critic=actor_critic,
            transition_manifest_path=distill_config.transition_manifest_path,
            window_manifest_path=distill_config.window_manifest_path,
            action_converter=action_converter,
            telemetry_fps=distill_config.telemetry_fps,
            step_seconds=distill_config.action_step_seconds,
            require_normalized_telemetry=distill_config.require_normalized_telemetry,
            action_scaler_path=None,
            binary_pos_weight=distill_config.binary_pos_weight,
            lr=distill_config.lr,
            weight_decay=distill_config.weight_decay,
            batch_size=distill_config.batch_size,
            seq_len=config.seq_len,
            epochs=distill_config.recurrent_epochs,
            grad_clip_norm=distill_config.grad_clip_norm,
            num_workers=distill_config.num_workers,
            checkpoint_dir=distill_config.checkpoint_dir,
            log_dir=distill_config.log_dir,
            device=config.device,
        )
        actor_critic = stage2.train()
    actor_critic.to(device)

    # Resume from a behavioral checkpoint if specified (overrides distilled init)
    if config.resume_from:
        resume_path = Path(config.resume_from)
        if not resume_path.is_absolute():
            resume_path = Path(config.log_dir) / resume_path
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            actor_critic.load_state_dict(ckpt["actor_critic_state"])
            logger.info(
                "Resumed from checkpoint %s (iter=%s bscore=%s)",
                resume_path,
                ckpt.get("iteration", "?"),
                ckpt.get("bscore", "?"),
            )
        else:
            raise FileNotFoundError(
                f"resume_from checkpoint not found: {resume_path}"
            )

    # Upgrade to visual actor-critic if enabled
    if config.enable_visual:
        logger.info("VISUAL MODE: Upgrading to VisualRecurrentActorCritic")
        tel_state = actor_critic.state_dict()
        visual_ac = VisualRecurrentActorCritic(
            tel_encoder=actor_critic.tel_encoder,
            visual_proj_dim=config.visual_proj_dim,
            visual_pretrained=True,
        )
        missing, unexpected = visual_ac.load_state_dict(tel_state, strict=False)
        visual_keys = [k for k in missing if "video_encoder" in k or "visual_fusion" in k]
        other_missing = [k for k in missing if k not in visual_keys]
        if other_missing:
            logger.warning("Unexpected missing keys on visual upgrade: %s", other_missing)
        logger.info(
            "Visual upgrade: %d visual keys initialized fresh, %d unexpected",
            len(visual_keys), len(unexpected),
        )
        if config.visual_encoder_ckpt and Path(config.visual_encoder_ckpt).exists():
            enc_state = torch.load(
                config.visual_encoder_ckpt, map_location=device, weights_only=False,
            )
            enc_sd = enc_state.get(
                "ema", enc_state.get("model", enc_state.get("model_state_dict", enc_state))
            )
            cleaned = {}
            for key, value in enc_sd.items():
                while key.startswith(("module.", "_orig_mod.")):
                    key = key.removeprefix("module.").removeprefix("_orig_mod.")
                cleaned[key] = value
            ve_state = {
                k.removeprefix("video_encoder."): v
                for k, v in cleaned.items()
                if k.startswith("video_encoder.")
            }
            if ve_state:
                visual_ac.video_encoder.load_state_dict(ve_state, strict=True)
                logger.info("Loaded VideoEncoder from Phase 1 (%d params)", len(ve_state))
        visual_ac.freeze_visual()
        visual_ac.to(device)
        actor_critic = visual_ac

    reference_policy = deepcopy(actor_critic)
    reference_policy.set_ppo_mode(training=False)
    for parameter in reference_policy.parameters():
        parameter.requires_grad_(False)

    if config.freeze_pretrained_branches:
        actor_critic.tel_encoder.freeze_branches()
    actor_critic.set_ppo_mode(training=True)

    reward_builder = RewardBuilder(
        dmg_dealt_idx=config.damage_dealt_obs_index,
        dmg_taken_idx=config.damage_taken_obs_index,
        env_reward_coef=config.env_reward_coef,
        damage_dealt_coef=config.damage_dealt_coef,
        damage_taken_coef=config.damage_taken_coef,
        kill_coef=config.kill_coef,
        death_coef=config.death_coef,
        miss_coef=config.miss_coef,
        jerk_coef=config.jerk_coef,
        spam_coef=config.spam_coef,
        aim_coef=config.aim_coef,
        aim_err_max=config.aim_err_max,
        aim_reward_mode=config.aim_reward_mode,
        bad_aim_threshold_deg=config.bad_aim_threshold_deg,
        bad_aim_coef=config.bad_aim_coef,
        bad_aim_scale_deg=config.bad_aim_scale_deg,
        aim_sigma_deg=config.aim_sigma_deg,
        use_true_aim_for_reward=config.use_true_aim_for_reward,
        true_aim_err_max=config.true_aim_err_max,
        true_aim_obs_index=config.true_aim_obs_index,
        shoot_bootstrap_metric=config.shoot_bootstrap_metric,
        true_shoot_bootstrap_threshold_deg=config.true_shoot_bootstrap_threshold_deg,
        shoot_bootstrap_coef=config.shoot_bootstrap_coef,
        shoot_bootstrap_aim_threshold_deg=config.shoot_bootstrap_aim_threshold_deg,
        shoot_bootstrap_require_fired=config.shoot_bootstrap_require_fired,
        shoot_bootstrap_require_los=config.shoot_bootstrap_require_los,
        shot_fired_obs_index=config.shot_fired_obs_index,
        damage_chain_coef=config.damage_chain_coef,
        damage_chain_window_sec=config.damage_chain_window_sec,
        damage_chain_window_steps=config.damage_chain_window_steps,
        learner_hit_env_reward_threshold=config.learner_hit_env_reward_threshold,
        hp_delta_reward_coef=config.hp_delta_reward_coef,
        hp_delta_reward_cap=config.hp_delta_reward_cap,
        opponent_hp_frac_obs_index=config.opponent_hp_frac_obs_index,
        hp_frac_to_damage_scale=config.hp_frac_to_damage_scale,
    )

    normalizer = ObservationNormalizer(
        config.telemetry_stats_path, clip_value=config.obs_clip_value,
    )
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    _apply_scripted_shoot_env(config)
    _apply_phase3ac_env(config)
    if config.enable_visual and config.num_unity_areas > 0:
        os.environ["ENABLE_VISUAL_OBS"] = "1"
        vec_runner = VisualMultiAreaUnityRunner(
            env_path=config.env_path,
            num_areas=config.num_unity_areas,
            reward_builder=reward_builder,
            device=config.device,
            base_port=config.env_base_port,
            seed=config.env_seed,
            time_scale=config.time_scale,
            normalizer=normalizer,
            visual_size=config.visual_frame_size,
        )
        effective_envs = config.num_unity_areas
    elif config.num_unity_areas > 0:
        vec_runner = MultiAreaUnityRunner(
            env_path=config.env_path,
            num_areas=config.num_unity_areas,
            reward_builder=reward_builder,
            device=config.device,
            base_port=config.env_base_port,
            seed=config.env_seed,
            time_scale=config.time_scale,
            normalizer=normalizer,
            shot_debug_enabled=config.shot_debug_enabled or config.shot_debug_true_unity,
            shot_debug_sample_limit_per_area=config.shot_debug_sample_limit_per_area,
            shot_debug_log_path=str(log_dir / "shot_debug.jsonl"),
            shot_debug_true_unity=config.shot_debug_true_unity,
            shot_debug_unity_log_path=str(log_dir / "shot_debug_unity.jsonl"),
            opponent_hp_frac_obs_index=config.opponent_hp_frac_obs_index,
            target_hurtbox_inflate=config.target_hurtbox_inflate,
            learner_shoot_ray_mode=config.learner_shoot_ray_mode,
            ray_correction_alpha=config.ray_correction_alpha,
            ray_correction_require_los=config.ray_correction_require_los,
            ray_correction_max_angle_deg=config.ray_correction_max_angle_deg,
            log_weapon_state_debug=config.log_weapon_state_debug,
            weapon_state_debug_log_path=str(log_dir / "weapon_state_debug.jsonl"),
        )
        effective_envs = config.num_unity_areas
    else:
        vec_runner = VectorizedUnityRunner(
            env_path=config.env_path,
            num_envs=config.num_envs,
            reward_builder=reward_builder,
            device=config.device,
            base_port=config.env_base_port,
            base_seed=config.env_seed,
            time_scale=config.time_scale,
            normalizer=normalizer,
            parallel_step=config.use_threadpool_step,
        )
        effective_envs = config.num_envs
    try:
        smoke_buffer = VectorizedRolloutBuffer(
            rollout_steps=config.rollout_smoke_steps,
            num_envs=effective_envs,
            device=config.device,
        )
        smoke_metrics = vec_runner.collect_rollouts(
            policy=actor_critic,
            buffer=smoke_buffer,
            n_steps=config.rollout_smoke_steps,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        _validate_vectorized_buffer(
            smoke_buffer, config.rollout_smoke_steps, effective_envs
        )
        logger.info(
            "%d-step x %d-env vectorized rollout smoke passed: %s",
            config.rollout_smoke_steps,
            effective_envs,
            {
                k: (f"{v:.4f}" if isinstance(v, (int, float)) else str(v))
                for k, v in smoke_metrics.items()
                if not k.startswith("rollout/reward/")
            },
        )

        if not config.enable_ppo:
            logger.info("PPO disabled. Vectorized smoke rollout complete.")
            return

        if config.require_damage_signals and not reward_builder.has_damage_signals:
            raise ValueError(
                "PPO reward requires damage_dealt_obs_index and "
                "damage_taken_obs_index."
            )

        # Reset per-run state on the function object
        train_ppo_vectorized._best_bscore = float("-inf")  # type: ignore[attr-defined]
        train_ppo_vectorized._best_late_bscore = float("-inf")  # type: ignore[attr-defined]
        train_ppo_vectorized._best_retention_score = float("-inf")  # type: ignore[attr-defined]
        history_len = max(config.drift_window_iters, config.retention_history_len)
        train_ppo_vectorized._recent_history = deque(maxlen=history_len)  # type: ignore[attr-defined]
        train_ppo_vectorized._best_retention_iter = 0  # type: ignore[attr-defined]
        train_ppo_vectorized._best_retention_aim = 999.0  # type: ignore[attr-defined]
        train_ppo_vectorized._best_retention_hits = 0.0  # type: ignore[attr-defined]
        train_ppo_vectorized._collapse_counter = 0  # type: ignore[attr-defined]

        if config.eval_only:
            logger.info(
                "[TRAINING DISABLED] eval_only=true; optimizer steps will be skipped"
            )

        registry = CheckpointRegistry(config.registry_dir)
        opponent_pool = OpponentPool(registry)
        registry.register(
            0, {"win_rate": 0.5, "damage_ratio": 1.0}, actor_critic.state_dict(),
        )

        buffer = VectorizedRolloutBuffer(
            rollout_steps=config.rollout_steps,
            num_envs=effective_envs,
            device=config.device,
        )
        optimizer = torch.optim.AdamW(
            [p for p in actor_critic.parameters() if p.requires_grad],
            lr=config.lr_initial if config.lr_schedule == "piecewise" else config.lr,
            weight_decay=1e-4,
        )
        log_path = log_dir / "ppo_train.jsonl"

        for iteration in range(1, config.num_iterations + 1):
            _ = opponent_pool

            lr_now = get_scheduled_lr(iteration, config)
            for group in optimizer.param_groups:
                group["lr"] = lr_now

            # Visual fusion unfreeze at specified iteration
            if (
                config.enable_visual
                and config.visual_unfreeze_fusion_at_iter > 0
                and iteration == config.visual_unfreeze_fusion_at_iter
                and hasattr(actor_critic, "unfreeze_fusion")
            ):
                actor_critic.unfreeze_fusion()
                optimizer.add_param_group({
                    "params": [p for p in actor_critic.visual_fusion.parameters() if p.requires_grad],
                    "lr": config.lr * 0.1,
                })
                logger.info(
                    "VISUAL_UNFREEZE iter=%d — fusion gate now trainable",
                    iteration,
                )

            logger.info("ITER_START iter=%d/%d pid=%d",
                        iteration, config.num_iterations, os.getpid())

            if (
                config.hard_restart_unity_each_iteration
                and iteration > 1
                and isinstance(vec_runner, MultiAreaUnityRunner)
            ):
                vec_runner.hard_restart()

            action_seed = None
            if config.fixed_action_seed_per_iteration:
                action_seed = config.env_seed + iteration

            logger.info("ROLLOUT_START iter=%d", iteration)
            rollout_metrics = vec_runner.collect_rollouts(
                policy=actor_critic,
                buffer=buffer,
                n_steps=config.rollout_steps,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                force_done_every_n=config.force_done_every_n,
                deterministic_policy_actions=config.deterministic_policy_actions,
                action_seed=action_seed,
            )
            logger.info("ROLLOUT_DONE iter=%d", iteration)

            logger.info("UPDATE_START iter=%d", iteration)
            t_update_start = time.monotonic()
            total_loss = 0.0
            updates = 0
            preupdate_checked = False
            latest_loss_metrics: dict[str, float] = {}
            epochs_completed = 0
            early_stopped = False
            last_grad_norm = 0.0

            if config.eval_only:
                logger.info(
                    "[TRAINING DISABLED] eval_only=true; optimizer steps will be skipped"
                )
                actor_critic.eval()
            else:
                actor_critic.set_ppo_mode(training=True)

            if not config.eval_only:
                for epoch in range(config.ppo_epochs):
                    epoch_kls: list[float] = []
                    for batch in buffer.recurrent_minibatches(
                        seq_len=config.seq_len,
                        batch_size=config.batch_size,
                    ):
                        obs = batch["obs"]
                        actions = batch["actions"]
                        hidden = batch["hiddens"]
                        current_dist, new_values, _ = actor_critic(obs, hidden)
                        new_log_probs = current_dist.log_prob(actions)
                        entropy = current_dist.entropy()

                        if not preupdate_checked:
                            ratio = torch.exp(
                                new_log_probs.detach() - batch["log_probs"]
                            )
                            clip_fraction = (
                                (torch.abs(ratio - 1.0) > config.ppo_clip)
                                .float()
                                .mean()
                                .item()
                            )
                            logger.info(
                                "Pre-update ratio: mean=%.8f std=%.8f clip_frac=%.8f",
                                ratio.mean().item(),
                                ratio.std(unbiased=False).item(),
                                clip_fraction,
                            )
                            if clip_fraction > config.max_preupdate_clip_frac:
                                raise RuntimeError(
                                    f"PPO log-prob invariant failed: "
                                    f"clip_fraction={clip_fraction}"
                                )
                            preupdate_checked = True

                        anchor_hidden = torch.zeros_like(hidden)
                        anchor_current, _, _ = actor_critic(obs, anchor_hidden)
                        with torch.no_grad():
                            anchor_reference, _, _ = reference_policy(
                                obs, anchor_hidden
                            )
                        reference_anchor = compute_reference_anchor(
                            anchor_current, anchor_reference,
                        )

                        loss, latest_loss_metrics = compute_ppo_loss(
                            new_log_probs=new_log_probs,
                            old_log_probs=batch["log_probs"],
                            advantages=batch["advantages"],
                            new_values=new_values,
                            returns=batch["returns"],
                            entropy=entropy,
                            reference_anchor=reference_anchor,
                            current_cont_mean=current_dist.cont_mean,
                            clip_ratio=config.ppo_clip,
                            value_coef=config.value_coef,
                            entropy_coef=config.entropy_coef,
                            kl_coef=config.kl_anchor_coef,
                            smooth_coef=config.smooth_coef,
                        )
                        optimizer.zero_grad()
                        loss.backward()
                        grad_norm = nn.utils.clip_grad_norm_(
                            actor_critic.parameters(), config.max_grad_norm
                        )
                        optimizer.step()
                        with torch.no_grad():
                            actor_critic.actor_cont_logstd.clamp_(
                                min=config.min_log_std, max=config.max_log_std
                            )
                        total_loss += loss.item()
                        last_grad_norm = float(grad_norm)
                        updates += 1
                        with torch.no_grad():
                            mb_kl = (batch["log_probs"] - new_log_probs).mean().item()
                        epoch_kls.append(mb_kl)

                    epochs_completed += 1
                    mean_epoch_kl = float(sum(epoch_kls) / len(epoch_kls)) if epoch_kls else 0.0
                    if config.target_kl is not None and mean_epoch_kl > config.target_kl:
                        logger.info(
                            "PPO_EARLY_STOP iter=%d epoch=%d mean_kl=%.4f target_kl=%.4f",
                            iteration, epoch, mean_epoch_kl, config.target_kl,
                        )
                        early_stopped = True
                        break

            update_time = time.monotonic() - t_update_start
            average_loss = total_loss / max(updates, 1)

            # ---- Post-update diagnostics (no grad): per-iteration + per-env ----
            kls, cfs, rmins, rmaxs, rmeans = [], [], [], [], []
            with torch.no_grad():
                for b in buffer.recurrent_minibatches(
                    seq_len=config.seq_len, batch_size=config.batch_size
                ):
                    d, _, _ = actor_critic(b["obs"], b["hiddens"])
                    nlp = d.log_prob(b["actions"])
                    r = torch.exp(nlp - b["log_probs"])
                    kls.append((b["log_probs"] - nlp).mean().item())
                    cfs.append((torch.abs(r - 1.0) > config.ppo_clip).float().mean().item())
                    rmins.append(r.min().item()); rmaxs.append(r.max().item()); rmeans.append(r.mean().item())
            _avg = lambda xs: float(sum(xs) / len(xs)) if xs else 0.0
            Tn = buffer.pos
            E = effective_envs
            obs_b, act_b = buffer.obs[:Tn], buffer.actions[:Tn]
            adv_b, ret_b, val_b = buffer.advantages[:Tn], buffer.returns[:Tn], buffer.values[:Tn]
            rew_b, done_b = buffer.rewards[:Tn], buffer.dones[:Tn]
            rd = getattr(vec_runner, "last_diag", {})
            raw_log, envr_log, dcounts = rd.get("raw_obs_log"), rd.get("env_reward_log"), rd.get("done_counts")
            per_env = []
            for e in range(E):
                oe, ae = obs_b[:, e], act_b[:, e]
                oe_abs = oe.abs()
                pe = {
                    "env": e,
                    "obs_all_zero": bool(torch.all(oe == 0).item()),
                    "norm_obs_mean": float(oe.mean()), "norm_obs_std": float(oe.std()),
                    "norm_obs_min": float(oe.min()), "norm_obs_max": float(oe.max()),
                    "norm_finite": bool(torch.isfinite(oe).all()),
                    "outlier_gt10": float((oe_abs > 10).float().mean()),
                    "outlier_gt25": float((oe_abs > 25).float().mean()),
                    "outlier_gt50": float((oe_abs > 50).float().mean()),
                    "outlier_gt100": float((oe_abs > 100).float().mean()),
                    "shaped_reward_mean": float(rew_b[:, e].mean()),
                    "shaped_reward_min": float(rew_b[:, e].min()),
                    "shaped_reward_max": float(rew_b[:, e].max()),
                    "done_count": int(done_b[:, e].sum().item()),
                    "rate_shoot": float((ae[:, 4] > 0.5).float().mean()),
                    "rate_reload": float((ae[:, 5] > 0.5).float().mean()),
                    "rate_jump": float((ae[:, 6] > 0.5).float().mean()),
                    "rate_crouch": float((ae[:, 7] > 0.5).float().mean()),
                }
                if raw_log is not None:
                    r_ = raw_log[:, e]
                    pe.update({"raw_obs_mean": float(r_.mean()), "raw_obs_std": float(r_.std()),
                               "raw_obs_min": float(r_.min()), "raw_obs_max": float(r_.max())})
                if envr_log is not None:
                    ee = envr_log[:, e]
                    pe.update({"env_reward_mean": float(ee.mean()), "env_reward_min": float(ee.min()),
                               "env_reward_max": float(ee.max())})
                if dcounts is not None:
                    pe["hidden_resets"] = int(dcounts[e])
                per_env.append(pe)
            with torch.no_grad():
                logstd_vals = actor_critic.actor_cont_logstd.cpu()
                act_std_per_dim = logstd_vals.exp().tolist()
                cont_act = act_b[:, :, :4]  # [T, E, 4] continuous actions
                act_mean_per_dim = cont_act.mean(dim=(0, 1)).cpu().tolist()
                act_std_sampled = cont_act.std(dim=(0, 1)).cpu().tolist()
            diag = {
                "approx_kl": _avg(kls),
                "postupdate_clip_frac": _avg(cfs),
                "ratio_min": float(min(rmins)) if rmins else 1.0,
                "ratio_mean": _avg(rmeans),
                "ratio_max": float(max(rmaxs)) if rmaxs else 1.0,
                "grad_norm": float(last_grad_norm) if "last_grad_norm" in locals() else 0.0,
                "buffer_shape": list(buffer.obs.shape),
                "adv_mean": float(adv_b.mean()), "adv_std": float(adv_b.std()),
                "adv_min": float(adv_b.min()), "adv_max": float(adv_b.max()),
                "ret_mean": float(ret_b.mean()), "ret_std": float(ret_b.std()),
                "val_mean": float(val_b.mean()), "val_std": float(val_b.std()),
                "finite": bool(torch.isfinite(adv_b).all() and torch.isfinite(ret_b).all()
                               and torch.isfinite(obs_b).all()),
                "per_env": per_env,
                "action/logstd": logstd_vals.tolist(),
                "action/std_policy": act_std_per_dim,
                "action/mean_sampled": act_mean_per_dim,
                "action/std_sampled": act_std_sampled,
                "ppo/epochs_completed": epochs_completed,
                "ppo/early_stopped": early_stopped,
            }
            logger.info(
                "ITER %d | approx_kl=%.4f postclip=%.3f ratio[min=%.3f mean=%.3f max=%.3f] "
                "grad_norm=%.3f buffer=%s adv[mu=%.3f sd=%.3f] ret[mu=%.3f sd=%.3f] "
                "loss[p=%.3f v=%.3f ent=%.3f anc=%.3f sm=%.3f tot=%.3f] finite=%s "
                "logstd=[%.2f,%.2f,%.2f,%.2f]",
                iteration, diag["approx_kl"], diag["postupdate_clip_frac"],
                diag["ratio_min"], diag["ratio_mean"], diag["ratio_max"], diag["grad_norm"],
                diag["buffer_shape"], diag["adv_mean"], diag["adv_std"], diag["ret_mean"], diag["ret_std"],
                latest_loss_metrics.get("loss/policy", 0.0), latest_loss_metrics.get("loss/value", 0.0),
                latest_loss_metrics.get("loss/entropy", 0.0),
                latest_loss_metrics.get("loss/reference_anchor", 0.0),
                latest_loss_metrics.get("loss/smoothness", 0.0),
                latest_loss_metrics.get("loss/total", 0.0), diag["finite"],
                *diag["action/logstd"],
            )
            for pe in per_env:
                logger.info(
                    "  env%d obs0=%s norm[mu=%.3f sd=%.3f rng=%.2f..%.2f fin=%s] "
                    "out[>10=%.3f >25=%.3f >50=%.3f >100=%.3f] "
                    "raw[mu=%.2f sd=%.2f rng=%.1f..%.1f] envR[mu=%.4f min=%.4f max=%.4f] "
                    "shapedR_mu=%.4f dones=%d resets=%d rates[s=%.2f r=%.2f j=%.2f c=%.2f]",
                    pe["env"], pe["obs_all_zero"], pe["norm_obs_mean"], pe["norm_obs_std"],
                    pe["norm_obs_min"], pe["norm_obs_max"], pe["norm_finite"],
                    pe["outlier_gt10"], pe["outlier_gt25"], pe["outlier_gt50"], pe["outlier_gt100"],
                    pe.get("raw_obs_mean", 0.0), pe.get("raw_obs_std", 0.0),
                    pe.get("raw_obs_min", 0.0), pe.get("raw_obs_max", 0.0),
                    pe.get("env_reward_mean", 0.0), pe.get("env_reward_min", 0.0), pe.get("env_reward_max", 0.0),
                    pe["shaped_reward_mean"], pe["done_count"], pe.get("hidden_resets", 0),
                    pe["rate_shoot"], pe["rate_reload"], pe["rate_jump"], pe["rate_crouch"],
                )
            eval_metrics: dict[str, float] = {}

            if iteration % config.eval_interval == 0 and effective_envs == 1:
                eval_metrics = evaluate_policy(
                    policy=actor_critic,
                    env=vec_runner.envs[0],
                    n_episodes=10,
                    device=config.device,
                )
                if should_promote(
                    eval_metrics, threshold_win_rate=config.promote_win_rate,
                ):
                    registry.register(
                        step=iteration * config.rollout_steps * effective_envs,
                        metrics=eval_metrics,
                        state_dict=actor_critic.state_dict(),
                    )
            elif iteration % config.eval_interval == 0:
                logger.info(
                    "Skipping evaluate_policy: multi-area env (%d areas) "
                    "is incompatible with single-agent evaluator",
                    effective_envs,
                )

            throughput = {
                "timing/update_time_sec": update_time,
                "timing/rollout_time_sec": rollout_metrics.get(
                    "rollout/rollout_time_sec", 0.0
                ),
                "timing/total_env_steps": rollout_metrics.get(
                    "rollout/total_env_steps", 0.0
                ),
                "timing/steps_per_sec": rollout_metrics.get(
                    "rollout/steps_per_sec", 0.0
                ),
            }

            obs_norm_stats = normalizer.get_stats_and_reset()

            logger.info(
                "iter %d/%d | loss=%.4f | updates=%d | "
                "rollout=%.1fs | update=%.1fs | "
                "env_steps=%d | steps/s=%.0f | episodes=%d | "
                "obs_clip_rate=%.4f",
                iteration,
                config.num_iterations,
                average_loss,
                updates,
                throughput["timing/rollout_time_sec"],
                update_time,
                int(throughput["timing/total_env_steps"]),
                throughput["timing/steps_per_sec"],
                int(rollout_metrics.get("rollout/episodes_completed", 0)),
                obs_norm_stats.get("obs_norm/clip_rate", 0.0),
            )

            logger.info("UPDATE_DONE iter=%d", iteration)

            logger.info("JSONL_WRITE iter=%d", iteration)
            config_metrics = {
                "config/shoot_bootstrap_aim_threshold_deg": (
                    config.shoot_bootstrap_aim_threshold_deg
                ),
                "config/shoot_bootstrap_coef": config.shoot_bootstrap_coef,
                "config/scripted_shoot_damage_scale": (
                    config.scripted_shoot_damage_scale
                ),
                "config/scripted_shoot_cooldown_mult": (
                    config.scripted_shoot_cooldown_mult
                ),
                "config/target_hurtbox_inflate": config.target_hurtbox_inflate,
                "config/learner_shoot_ray_mode": config.learner_shoot_ray_mode,
                "config/ray_correction_alpha": config.ray_correction_alpha,
                "config/ray_correction_require_los": float(
                    1.0 if config.ray_correction_require_los else 0.0
                ),
                "config/ray_correction_max_angle_deg": config.ray_correction_max_angle_deg,
                "config/shoot_bootstrap_metric": config.shoot_bootstrap_metric,
                "config/true_shoot_bootstrap_threshold_deg": (
                    config.true_shoot_bootstrap_threshold_deg
                ),
                "config/use_true_aim_for_reward": config.use_true_aim_for_reward,
                "config/true_aim_err_max": config.true_aim_err_max,
                "config/aim_reward_mode": config.aim_reward_mode,
                "config/eval_only": config.eval_only,
            }
            if config.log_per_agent_metrics:
                config_metrics.update(_per_agent_metrics(rollout_metrics, num_agents=2))
            unity_shot_log = log_dir / "shot_debug_unity.jsonl"
            if unity_shot_log.exists():
                from .shot_geometry_summary import compact_shot_geometry_metrics

                rollout_metrics.update(
                    compact_shot_geometry_metrics(unity_shot_log)
                )
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "iteration": iteration,
                            "loss": average_loss,
                            "ppo/lr": lr_now,
                            **config_metrics,
                            **latest_loss_metrics,
                            **diag,
                            **rollout_metrics,
                            **eval_metrics,
                            **throughput,
                            **obs_norm_stats,
                        }
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            # --- Behavioral checkpoint saving (skip in eval-only mode) ---
            hit_count = rollout_metrics.get("aim/total_hits", 0)
            kill_count = rollout_metrics.get("aim/total_kills", 0)
            pos_reward = rollout_metrics.get("aim/total_pos_rewards", 0)
            aim_err = rollout_metrics.get("aim/err_mean", 999.0)
            bscore = (
                1.0 * hit_count
                + 5.0 * kill_count
                + 0.1 * pos_reward
                - 0.02 * aim_err
            )
            if not config.eval_only:
                if not hasattr(train_ppo_vectorized, "_best_bscore"):
                    train_ppo_vectorized._best_bscore = float("-inf")  # type: ignore[attr-defined]
                if bscore > train_ppo_vectorized._best_bscore:  # type: ignore[attr-defined]
                    train_ppo_vectorized._best_bscore = bscore  # type: ignore[attr-defined]
                    _save_behavioral_checkpoint(
                        log_dir / "best_behavioral.pt",
                        iteration,
                        actor_critic,
                        bscore=bscore,
                        hit_count=hit_count,
                        kill_count=kill_count,
                        aim_err=aim_err,
                        tag="BEST_BEHAVIORAL",
                    )

            learner_damage = rollout_metrics.get("damage/learner_damage_dealt", 0.0)
            train_ppo_vectorized._recent_history.append(  # type: ignore[attr-defined]
                {
                    "hits": hit_count,
                    "aim_err": aim_err,
                    "kills": kill_count,
                    "learner_damage": learner_damage,
                }
            )
            if not config.eval_only:
                if iteration >= 80:
                    if bscore > train_ppo_vectorized._best_late_bscore:  # type: ignore[attr-defined]
                        train_ppo_vectorized._best_late_bscore = bscore  # type: ignore[attr-defined]
                        _save_behavioral_checkpoint(
                            log_dir / "best_late_behavioral.pt",
                            iteration,
                            actor_critic,
                            bscore=bscore,
                            hit_count=hit_count,
                            kill_count=kill_count,
                            aim_err=aim_err,
                            tag="BEST_LATE_BEHAVIORAL",
                        )

                if iteration >= config.retention_min_iter:
                    recent = list(train_ppo_vectorized._recent_history)  # type: ignore[attr-defined]
                    window = recent[-config.retention_history_len:]
                    if len(window) >= min(10, config.retention_history_len):
                        avg_hits = sum(r["hits"] for r in window) / len(window)
                        avg_aim = sum(r["aim_err"] for r in window) / len(window)
                        avg_kills = sum(r.get("kills", 0) for r in window) / len(window)
                        avg_dmg = sum(r.get("learner_damage", 0) for r in window) / len(window)
                        nonzero_hit_iters = sum(1 for r in window if r["hits"] > 0)
                        retention_score = (
                            avg_hits
                            - 0.03 * avg_aim
                            + 2.0 * avg_kills
                            + 0.02 * avg_dmg
                            + 0.5 * nonzero_hit_iters
                        )
                        if (
                            retention_score
                            > train_ppo_vectorized._best_retention_score  # type: ignore[attr-defined]
                        ):
                            train_ppo_vectorized._best_retention_score = (  # type: ignore[attr-defined]
                                retention_score
                            )
                            train_ppo_vectorized._best_retention_iter = iteration  # type: ignore[attr-defined]
                            train_ppo_vectorized._best_retention_aim = avg_aim  # type: ignore[attr-defined]
                            train_ppo_vectorized._best_retention_hits = avg_hits  # type: ignore[attr-defined]
                            _save_behavioral_checkpoint(
                                log_dir / "best_retention.pt",
                                iteration,
                                actor_critic,
                                bscore=bscore,
                                hit_count=hit_count,
                                kill_count=kill_count,
                                aim_err=aim_err,
                                tag="BEST_RETENTION",
                            )
                            logger.info(
                                "RETENTION_SCORE iter=%d score=%.3f "
                                "avg_hits=%.2f avg_aim=%.1f avg_kills=%.2f avg_dmg=%.2f "
                                "nonzero_iters=%d",
                                iteration,
                                retention_score,
                                avg_hits,
                                avg_aim,
                                avg_kills,
                                avg_dmg,
                                nonzero_hit_iters,
                            )
                torch.save(
                    {
                        "iteration": iteration,
                        "actor_critic_state": actor_critic.state_dict(),
                    },
                    log_dir / "latest.pt",
                )

            logger.info("ITER_DONE iter=%d pid=%d", iteration, os.getpid())

            # Early collapse detection: stop if policy is stuck in facing-away attractor
            if config.early_stop_on_collapse:
                if not hasattr(train_ppo_vectorized, "_collapse_counter"):
                    train_ppo_vectorized._collapse_counter = 0  # type: ignore[attr-defined]
                if (
                    aim_err > config.collapse_aim_threshold
                    and hit_count == 0
                ):
                    train_ppo_vectorized._collapse_counter += 1  # type: ignore[attr-defined]
                else:
                    train_ppo_vectorized._collapse_counter = 0  # type: ignore[attr-defined]
                if train_ppo_vectorized._collapse_counter >= config.collapse_zero_hits_iters:  # type: ignore[attr-defined]
                    logger.info(
                        "COLLAPSE_STOP iter=%d aim=%.1f hits=%d "
                        "consecutive_collapse_iters=%d",
                        iteration, aim_err, hit_count,
                        train_ppo_vectorized._collapse_counter,  # type: ignore[attr-defined]
                    )
                    break

            drift_stop = False
            if config.early_stop_on_drift and iteration >= config.drift_start_iter:
                recent = list(train_ppo_vectorized._recent_history)  # type: ignore[attr-defined]
                window = recent[-config.drift_window_iters:]
                if len(window) >= config.drift_window_iters:
                    mean_aim_window = sum(r["aim_err"] for r in window) / len(window)
                    mean_hits_window = sum(r["hits"] for r in window) / len(window)
                    if (
                        mean_aim_window > config.drift_aim_threshold
                        and mean_hits_window < config.drift_min_hits
                    ):
                        _save_behavioral_checkpoint(
                            log_dir / "drift_stop_latest.pt",
                            iteration,
                            actor_critic,
                            bscore=bscore,
                            hit_count=hit_count,
                            kill_count=kill_count,
                            aim_err=aim_err,
                            tag="DRIFT_STOP",
                        )
                        logger.info(
                            "DRIFT_STOP iter=%d mean_aim=%.1f mean_hits=%.2f",
                            iteration, mean_aim_window, mean_hits_window,
                        )
                        drift_stop = True

                    best_ret_iter = train_ppo_vectorized._best_retention_iter  # type: ignore[attr-defined]
                    if (
                        not drift_stop
                        and best_ret_iter > 0
                        and iteration - best_ret_iter > config.drift_patience_after_best
                    ):
                        best_ret_aim = train_ppo_vectorized._best_retention_aim  # type: ignore[attr-defined]
                        best_ret_hits = train_ppo_vectorized._best_retention_hits  # type: ignore[attr-defined]
                        if (
                            mean_aim_window > best_ret_aim + 30.0
                            or mean_hits_window < 0.7 * best_ret_hits
                        ):
                            _save_behavioral_checkpoint(
                                log_dir / "patience_stop_latest.pt",
                                iteration,
                                actor_critic,
                                bscore=bscore,
                                hit_count=hit_count,
                                kill_count=kill_count,
                                aim_err=aim_err,
                                tag="PATIENCE_STOP",
                            )
                            logger.info(
                                "PATIENCE_STOP iter=%d best_ret_iter=%d "
                                "mean_aim=%.1f mean_hits=%.2f",
                                iteration, best_ret_iter,
                                mean_aim_window, mean_hits_window,
                            )
                            drift_stop = True
            if drift_stop:
                break

            if _check_preserve_stop(
                iteration, hit_count, aim_err, config, train_ppo_vectorized
            ):
                break

            if _check_gvg_smoke_stop(
                iteration, rollout_metrics, config, train_ppo_vectorized
            ):
                break
    finally:
        vec_runner.close()
        if config.shot_debug_true_unity or config.write_ray_vs_hurtbox_summary:
            try:
                from .shot_geometry_summary import (
                    write_compact_metrics,
                    write_ray_vs_hurtbox_summary,
                    write_shot_geometry_summary,
                )

                summary_path = write_shot_geometry_summary(log_dir)
                if summary_path:
                    logger.info("SHOT_GEOMETRY_SUMMARY -> %s", summary_path)
                if config.write_ray_vs_hurtbox_summary or config.shot_debug_true_unity:
                    ray_path = write_ray_vs_hurtbox_summary(log_dir)
                    if ray_path:
                        logger.info("RAY_VS_HURTBOX_SUMMARY -> %s", ray_path)
                compact_path = write_compact_metrics(log_dir)
                if compact_path and config.write_compact_metrics:
                    logger.info("COMPACT_METRICS -> %s", compact_path)
            except Exception as exc:
                logger.warning("SHOT_GEOMETRY_SUMMARY failed: %s", exc)


def _feedforward_only(config: PPOTrainConfig, distill_config: DistillConfig) -> None:
    """Run only Stage 1 feedforward distillation, then exit."""
    distributed, rank, local_rank, world_size, device = init_distributed()
    try:
        if is_main_process():
            _validate_local_inputs(config, distill_config)
        barrier()

        device_str = str(device) if distributed else config.device
        action_converter = _build_action_converter(distill_config)

        if is_main_process():
            logger.info(
                "Loading pretrained telemetry branches from %s",
                distill_config.phase1_ckpt,
            )
        tel_encoder = TelemetryEncoder.from_encoder_checkpoint(
            distill_config.phase1_ckpt,
            device=torch.device(device_str),
        )

        projector = CoreActionProjector(tel_encoder)
        stage1 = CoreActionDistiller(
            projector=projector,
            transition_manifest_path=distill_config.transition_manifest_path,
            window_manifest_path=distill_config.window_manifest_path,
            action_converter=action_converter,
            telemetry_fps=distill_config.telemetry_fps,
            step_seconds=distill_config.action_step_seconds,
            require_normalized_telemetry=distill_config.require_normalized_telemetry,
            action_scaler_path=None,
            binary_pos_weight=distill_config.binary_pos_weight,
            lr=distill_config.lr,
            weight_decay=distill_config.weight_decay,
            batch_size=distill_config.resolved_batch_size(),
            epochs=distill_config.epochs,
            warmup_epochs=distill_config.warmup_epochs,
            grad_clip_norm=distill_config.grad_clip_norm,
            num_workers=distill_config.resolved_num_workers(),
            checkpoint_dir=distill_config.checkpoint_dir,
            log_dir=distill_config.log_dir,
            device=device_str,
            distributed=distributed,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            max_batches=distill_config.max_batches,
        )
        stage1.train()
        if is_main_process():
            logger.info("Feedforward distillation complete.")
    finally:
        cleanup_distributed()


def _recurrent_only(config: PPOTrainConfig, distill_config: DistillConfig) -> None:
    """Run Stage 2 recurrent warmup (loads Stage 1 checkpoint)."""
    distributed, rank, local_rank, world_size, device = init_distributed()
    try:
        if is_main_process():
            _validate_local_inputs(config, distill_config)
        barrier()

        device_str = str(device) if distributed else config.device
        torch_device = torch.device(device_str)
        action_converter = _build_action_converter(distill_config)

        tel_encoder = TelemetryEncoder.from_encoder_checkpoint(
            distill_config.phase1_ckpt,
            device=torch_device,
        )
        projector = CoreActionProjector(tel_encoder)

        stage1_ckpt = Path(distill_config.checkpoint_dir) / "command_best.pt"
        barrier()
        if is_main_process() and not stage1_ckpt.exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found at {stage1_ckpt}. "
                "Run feedforward_only stage first."
            )
        barrier()

        state = torch.load(stage1_ckpt, map_location=torch_device, weights_only=False)
        projector.load_state_dict(state["projector_state"])
        projector.to(torch_device)
        if is_main_process():
            logger.info("Loaded Stage 1 projector from %s", stage1_ckpt)
        barrier()

        actor_critic = RecurrentActorCritic.from_distilled(projector)
        stage2 = RecurrentDistiller(
            actor_critic=actor_critic,
            transition_manifest_path=distill_config.transition_manifest_path,
            window_manifest_path=distill_config.window_manifest_path,
            action_converter=action_converter,
            telemetry_fps=distill_config.telemetry_fps,
            step_seconds=distill_config.action_step_seconds,
            require_normalized_telemetry=distill_config.require_normalized_telemetry,
            action_scaler_path=None,
            binary_pos_weight=distill_config.binary_pos_weight,
            lr=distill_config.lr,
            weight_decay=distill_config.weight_decay,
            batch_size=distill_config.resolved_recurrent_batch_size(),
            seq_len=config.seq_len,
            epochs=distill_config.recurrent_epochs,
            grad_clip_norm=distill_config.grad_clip_norm,
            num_workers=distill_config.resolved_num_workers(),
            checkpoint_dir=distill_config.checkpoint_dir,
            log_dir=distill_config.log_dir,
            device=device_str,
            distributed=distributed,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            max_batches=distill_config.max_batches,
        )
        stage2.train()
        if is_main_process():
            logger.info("Recurrent distillation complete.")
    finally:
        cleanup_distributed()


def _unity_smoke(config: PPOTrainConfig, distill_config: DistillConfig) -> None:
    """Load distilled policy, run 100-step Unity smoke test only."""
    _validate_local_inputs(config, distill_config)
    device = torch.device(config.device)

    tel_encoder = TelemetryEncoder.from_encoder_checkpoint(
        distill_config.phase1_ckpt,
        device=device,
    )
    projector = CoreActionProjector(tel_encoder)

    stage2_ckpt = Path(distill_config.checkpoint_dir) / "command_recurrent_best.pt"
    if not stage2_ckpt.exists():
        raise FileNotFoundError(
            f"Stage 2 checkpoint not found at {stage2_ckpt}. "
            "Run recurrent_only stage first."
        )

    actor_critic = RecurrentActorCritic.from_distilled(projector)
    state = torch.load(stage2_ckpt, map_location=device, weights_only=False)
    actor_critic.load_state_dict(state["actor_critic_state"])
    actor_critic.to(device)
    logger.info("Loaded recurrent policy from %s", stage2_ckpt)

    reward_builder = RewardBuilder(
        dmg_dealt_idx=config.damage_dealt_obs_index,
        dmg_taken_idx=config.damage_taken_obs_index,
        env_reward_coef=config.env_reward_coef,
        damage_dealt_coef=config.damage_dealt_coef,
        damage_taken_coef=config.damage_taken_coef,
        kill_coef=config.kill_coef,
        death_coef=config.death_coef,
        miss_coef=config.miss_coef,
        jerk_coef=config.jerk_coef,
        spam_coef=config.spam_coef,
        aim_coef=config.aim_coef,
        aim_err_max=config.aim_err_max,
        aim_reward_mode=config.aim_reward_mode,
        bad_aim_threshold_deg=config.bad_aim_threshold_deg,
        bad_aim_coef=config.bad_aim_coef,
        bad_aim_scale_deg=config.bad_aim_scale_deg,
        aim_sigma_deg=config.aim_sigma_deg,
        use_true_aim_for_reward=config.use_true_aim_for_reward,
        true_aim_err_max=config.true_aim_err_max,
        true_aim_obs_index=config.true_aim_obs_index,
        shoot_bootstrap_metric=config.shoot_bootstrap_metric,
        true_shoot_bootstrap_threshold_deg=config.true_shoot_bootstrap_threshold_deg,
        shoot_bootstrap_coef=config.shoot_bootstrap_coef,
        shoot_bootstrap_aim_threshold_deg=config.shoot_bootstrap_aim_threshold_deg,
        shoot_bootstrap_require_fired=config.shoot_bootstrap_require_fired,
        shoot_bootstrap_require_los=config.shoot_bootstrap_require_los,
        shot_fired_obs_index=config.shot_fired_obs_index,
        damage_chain_coef=config.damage_chain_coef,
        damage_chain_window_sec=config.damage_chain_window_sec,
        damage_chain_window_steps=config.damage_chain_window_steps,
        learner_hit_env_reward_threshold=config.learner_hit_env_reward_threshold,
        hp_delta_reward_coef=config.hp_delta_reward_coef,
        hp_delta_reward_cap=config.hp_delta_reward_cap,
        opponent_hp_frac_obs_index=config.opponent_hp_frac_obs_index,
        hp_frac_to_damage_scale=config.hp_frac_to_damage_scale,
    )

    env = setup_unity_env(config.env_path)
    try:
        runner = SelfPlayRunner(
            env=env, reward_builder=reward_builder, device=config.device,
            normalizer=ObservationNormalizer(
                config.telemetry_stats_path, clip_value=config.obs_clip_value,
            ),
        )
        buffer = RecurrentRolloutBuffer(
            size=config.rollout_smoke_steps, device=config.device
        )
        metrics = runner.collect_rollouts(
            policy=actor_critic,
            buffer=buffer,
            n_steps=config.rollout_smoke_steps,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        _validate_rollout_buffer(buffer, config.rollout_smoke_steps)
        logger.info(
            "%d-step Unity smoke test PASSED: %s",
            config.rollout_smoke_steps,
            metrics,
        )
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 gated pipeline")
    parser.add_argument("--ppo_config", required=True)
    parser.add_argument("--distill_config", required=True)
    parser.add_argument(
        "--stage",
        choices=[
            "feedforward_distill_only",
            "recurrent_distill_only",
            "invariant_only",
            "unity_smoke",
            "ppo",
            "full",
        ],
        default="full",
        help="Which stage to run.",
    )
    parser.add_argument(
        "--enable_ppo", action="store_true", help="Override enable_ppo in config"
    )
    parser.add_argument("--env_path", default=None, help="Override Unity build path")
    parser.add_argument(
        "--num_iterations", type=int, default=None, help="Override num_iterations"
    )
    parser.add_argument(
        "--num_envs", type=int, default=None, help="Override num_envs"
    )
    parser.add_argument(
        "--num_unity_areas", type=int, default=None,
        help="Override num_unity_areas (multi-area single-process mode)",
    )
    parser.add_argument(
        "--obs_clip_value", type=float, default=None,
        help="Override obs_clip_value (post-normalization clipping)",
    )
    parser.add_argument(
        "--force_done_every_n", type=int, default=None,
        help="Force episode boundary every N steps (pseudo-episodes)",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate",
    )
    arguments = parser.parse_args()

    ppo_cfg = PPOTrainConfig.from_yaml(arguments.ppo_config)
    distill_cfg = DistillConfig.from_yaml(arguments.distill_config)

    if arguments.enable_ppo:
        ppo_cfg.enable_ppo = True
    if arguments.env_path is not None:
        ppo_cfg.env_path = arguments.env_path
    if arguments.num_iterations is not None:
        ppo_cfg.num_iterations = arguments.num_iterations
    if arguments.num_envs is not None:
        ppo_cfg.num_envs = arguments.num_envs
    if arguments.num_unity_areas is not None:
        ppo_cfg.num_unity_areas = arguments.num_unity_areas
    if arguments.obs_clip_value is not None:
        ppo_cfg.obs_clip_value = arguments.obs_clip_value
    if arguments.force_done_every_n is not None:
        ppo_cfg.force_done_every_n = arguments.force_done_every_n
    if arguments.lr is not None:
        ppo_cfg.lr = arguments.lr

    if arguments.stage == "feedforward_distill_only":
        _feedforward_only(ppo_cfg, distill_cfg)
    elif arguments.stage == "recurrent_distill_only":
        _recurrent_only(ppo_cfg, distill_cfg)
    elif arguments.stage == "invariant_only":
        from .invariant_check import run_invariant_only

        run_invariant_only(ppo_cfg, distill_cfg)
    elif arguments.stage == "unity_smoke":
        _unity_smoke(ppo_cfg, distill_cfg)
    elif arguments.stage == "ppo":
        if not arguments.enable_ppo:
            raise SystemExit("PPO stage requires explicit --enable_ppo")
        ppo_cfg.enable_ppo = True
        if ppo_cfg.num_envs > 1 or ppo_cfg.num_unity_areas > 0:
            train_ppo_vectorized(ppo_cfg, distill_cfg)
        else:
            train_ppo(ppo_cfg, distill_cfg)
    else:
        if (ppo_cfg.num_envs > 1 or ppo_cfg.num_unity_areas > 0) and ppo_cfg.enable_ppo:
            train_ppo_vectorized(ppo_cfg, distill_cfg)
        else:
            train_ppo(ppo_cfg, distill_cfg)
