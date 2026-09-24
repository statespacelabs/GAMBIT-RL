#!/usr/bin/env python3
"""Phase 3v2 tiny PPO canary."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.combat_ppo_update import collect_rollout, compute_loss_and_backward
from scripts.combat_runtime import (
    OBS_DIM,
    RuntimeConfig,
    action_to_unity,
    append_jsonl,
    collect_all_weapon_events,
    configure_unity_env,
    get_first_decision,
    launch_unity_env,
    load_policy_and_normalizer,
    manifest,
    sha256_file,
    summarize_weapon_events,
    utc_now,
    validate_specs,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="")
    parser.add_argument("--env-path", default=RuntimeConfig.env_path)
    parser.add_argument("--normalizer-path", default=RuntimeConfig.normalizer_path)
    parser.add_argument("--ppo-config", default=RuntimeConfig.ppo_config)
    parser.add_argument("--distill-config", default=RuntimeConfig.distill_config)
    parser.add_argument(
        "--output-dir",
        default="experiments/phase3v2/phase3v2_a_tiny_ppo_canary",
    )
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-decisions", type=int, default=128)
    parser.add_argument("--base-port", type=int, default=51305)
    parser.add_argument("--seed", type=int, default=7421)
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--target-kl", type=float, default=RuntimeConfig.target_kl)
    parser.add_argument("--max-grad-norm", type=float, default=RuntimeConfig.max_grad_norm)
    parser.add_argument("--kl-anchor-coef", type=float, default=0.0)
    parser.add_argument("--kl-anchor-checkpoint", default="")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--game-mode", default="GambitVsScripted")
    parser.add_argument("--player-b-bot-mode", default="StrafeAndFace")
    parser.add_argument("--num-areas", type=int, default=1)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--aim-coef", type=float, default=0.0)
    parser.add_argument("--aim-err-max", type=float, default=150.0)
    parser.add_argument("--shoot-bootstrap-coef", type=float, default=0.0)
    parser.add_argument("--shoot-bootstrap-aim-threshold-deg", type=float, default=30.0)
    parser.add_argument("--jerk-coef", type=float, default=0.0)
    parser.add_argument("--spam-coef", type=float, default=0.0)
    parser.add_argument("--min-log-std", type=float, default=-2.5)
    parser.add_argument("--max-log-std", type=float, default=-0.1)
    parser.add_argument("--scripted-shoot-damage-scale", type=float, default=1.0)
    parser.add_argument("--scripted-shoot-cooldown-mult", type=float, default=1.0)
    parser.add_argument("--scripted-shoot-aim-threshold-deg", type=float, default=15.0)
    parser.add_argument("--scripted-shoot-warmup-iters", type=int, default=0)
    parser.add_argument("--mean-action-saturation-penalty", action="store_true", default=RuntimeConfig.mean_action_saturation_penalty)
    parser.add_argument("--mean-action-saturation-threshold", type=float, default=RuntimeConfig.mean_action_saturation_threshold)
    parser.add_argument("--mean-action-saturation-coef", type=float, default=RuntimeConfig.mean_action_saturation_coef)
    parser.add_argument("--look-action-l2-reward-coef", type=float, default=RuntimeConfig.look_action_l2_reward_coef)
    parser.add_argument("--mean-action-l2-loss-coef", type=float, default=RuntimeConfig.mean_action_l2_loss_coef)
    parser.add_argument("--saturation-abs095-stop-rate", type=float, default=RuntimeConfig.saturation_abs095_stop_rate)
    parser.add_argument("--saturation-start-iter", type=int, default=RuntimeConfig.saturation_start_iter)
    parser.add_argument("--value-only", action="store_true", default=RuntimeConfig.value_only)
    parser.add_argument("--bc-aux-coef", type=float, default=RuntimeConfig.bc_aux_coef)
    parser.add_argument("--bc-aux-dataset-path", default=RuntimeConfig.bc_aux_dataset_path)
    parser.add_argument("--bc-aux-batch-size", type=int, default=RuntimeConfig.bc_aux_batch_size)
    parser.add_argument("--actor-update-enabled", action="store_true", default=RuntimeConfig.actor_update_enabled)
    parser.add_argument("--no-actor-update", dest="actor_update_enabled", action="store_false")
    parser.add_argument("--value-update-enabled", action="store_true", default=RuntimeConfig.value_update_enabled)
    parser.add_argument("--no-value-update", dest="value_update_enabled", action="store_false")
    parser.add_argument("--freeze-look-action-head", action="store_true", default=RuntimeConfig.freeze_look_action_head)
    parser.add_argument("--freeze-movement-action-head", action="store_true", default=RuntimeConfig.freeze_movement_action_head)
    parser.add_argument("--train-shoot-head", action="store_true", default=RuntimeConfig.train_shoot_head)
    parser.add_argument("--train-value-head", action="store_true", default=RuntimeConfig.train_value_head)
    args = parser.parse_args()
    if args.config:
        data = yaml.safe_load(Path(args.config).read_text()) or {}
        mapping = {
            "resume_from": "resume_from",
            "normalizer_path": "normalizer_path",
            "game_mode": "game_mode",
            "player_b_bot_mode": "player_b_bot_mode",
            "num_iterations": "updates",
            "time_scale": "time_scale",
            "lr_initial": "lr",
            "lr": "lr",
            "target_kl": "target_kl",
            "max_grad_norm": "max_grad_norm",
            "kl_anchor_coef": "kl_anchor_coef",
            "kl_anchor_checkpoint": "kl_anchor_checkpoint",
            "seed": "seed",
            "entropy_coef": "entropy_coef",
            "aim_coef": "aim_coef",
            "aim_err_max": "aim_err_max",
            "shoot_bootstrap_coef": "shoot_bootstrap_coef",
            "shoot_bootstrap_aim_threshold_deg": "shoot_bootstrap_aim_threshold_deg",
            "jerk_coef": "jerk_coef",
            "spam_coef": "spam_coef",
            "min_log_std": "min_log_std",
            "max_log_std": "max_log_std",
            "scripted_shoot_damage_scale": "scripted_shoot_damage_scale",
            "scripted_shoot_cooldown_mult": "scripted_shoot_cooldown_mult",
            "scripted_shoot_aim_threshold_deg": "scripted_shoot_aim_threshold_deg",
            "scripted_shoot_warmup_iters": "scripted_shoot_warmup_iters",
            "mean_action_saturation_penalty": "mean_action_saturation_penalty",
            "mean_action_saturation_threshold": "mean_action_saturation_threshold",
            "mean_action_saturation_coef": "mean_action_saturation_coef",
            "look_action_l2_reward_coef": "look_action_l2_reward_coef",
            "mean_action_l2_loss_coef": "mean_action_l2_loss_coef",
            "saturation_abs095_stop_rate": "saturation_abs095_stop_rate",
            "saturation_start_iter": "saturation_start_iter",
            "value_only": "value_only",
            "bc_aux_coef": "bc_aux_coef",
            "bc_aux_dataset_path": "bc_aux_dataset_path",
            "bc_aux_batch_size": "bc_aux_batch_size",
            "actor_update_enabled": "actor_update_enabled",
            "value_update_enabled": "value_update_enabled",
            "freeze_look_action_head": "freeze_look_action_head",
            "freeze_movement_action_head": "freeze_movement_action_head",
            "train_shoot_head": "train_shoot_head",
            "train_value_head": "train_value_head",
        }
        for key, attr in mapping.items():
            if key in data and getattr(args, attr, None) == parser.get_default(attr):
                setattr(args, attr, data[key])
    int_fields = {"updates", "base_port", "seed", "num_areas", "scripted_shoot_warmup_iters", "saturation_start_iter", "bc_aux_batch_size"}
    bool_fields = {"mean_action_saturation_penalty", "value_only", "actor_update_enabled", "value_update_enabled", "freeze_look_action_head", "freeze_movement_action_head", "train_shoot_head", "train_value_head"}
    float_fields = {
        "time_scale", "lr", "target_kl", "max_grad_norm", "kl_anchor_coef", "entropy_coef", "aim_coef", "aim_err_max",
        "shoot_bootstrap_coef", "shoot_bootstrap_aim_threshold_deg", "jerk_coef",
        "spam_coef", "min_log_std", "max_log_std", "scripted_shoot_damage_scale",
        "scripted_shoot_cooldown_mult", "scripted_shoot_aim_threshold_deg",
        "mean_action_saturation_threshold", "mean_action_saturation_coef", "look_action_l2_reward_coef", "mean_action_l2_loss_coef", "saturation_abs095_stop_rate", "bc_aux_coef",
    }
    for field in int_fields:
        setattr(args, field, int(getattr(args, field)))
    for field in bool_fields:
        value = getattr(args, field)
        if isinstance(value, str):
            value = value.lower() in {"1", "true", "yes", "on"}
        setattr(args, field, bool(value))
    for field in float_fields:
        setattr(args, field, float(getattr(args, field)))
    return args


def parameter_delta(before: dict[str, torch.Tensor], policy) -> float:
    total = 0.0
    with torch.no_grad():
        for name, param in policy.state_dict().items():
            if not torch.is_floating_point(param):
                continue
            total += float((param.detach().cpu() - before[name]).abs().sum().item())
    return total


def evaluate_reloaded(policy, normalizer, env, behavior_name, action_spec, cfg, device, decisions: int) -> dict:
    policy.set_ppo_mode(training=False)
    hidden = policy.init_hidden(1, device)
    agent_id, obs, _ = get_first_decision(env, behavior_name)
    valid_actions = 0
    terminals = 0
    fires = 0
    rewards = []
    for _ in range(decisions):
        obs_norm = normalizer.normalize_tensor(
            torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, OBS_DIM)
        )
        with torch.no_grad():
            dist, value, next_hidden = policy(obs_norm, hidden)
            action_clamped, _ = dist.mode
        if not torch.isfinite(value).all() or not torch.isfinite(action_clamped).all():
            raise RuntimeError("reloaded policy produced NaN/Inf")
        action_np = action_clamped.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        action_to_unity(env, behavior_name, action_spec, agent_id, action_np)
        t0 = time.monotonic()
        env.step()
        if time.monotonic() - t0 > cfg.deadlock_step_seconds:
            raise RuntimeError("Unity deadlock during reload eval")
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if len(terminal_steps):
            terminals += len(terminal_steps)
            hidden.zero_()
        else:
            hidden = next_hidden.detach()
        if len(decision_steps):
            idx = 0
            obs = np.asarray(decision_steps.obs[0][idx], dtype=np.float32)
            agent_id = int(decision_steps.agent_id[idx])
            rewards.append(float(decision_steps.reward[idx]))
        else:
            agent_id, obs, reward = get_first_decision(env, behavior_name)
            rewards.append(reward)
        if obs[44] > 0.5:
            fires += 1
        valid_actions += 1
    return {
        "eval_decisions": valid_actions,
        "eval_terminals": terminals,
        "eval_reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "eval_fire_count_A": fires,
        "reloaded_actions_valid": valid_actions == decisions,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = RuntimeConfig(
        env_path=args.env_path,
        normalizer_path=args.normalizer_path,
        ppo_config=args.ppo_config,
        distill_config=args.distill_config,
        output_dir=args.output_dir,
        base_port=args.base_port,
        seed=args.seed,
        time_scale=args.time_scale,
        device=args.device,
        rollout_steps=args.rollout_steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        target_kl=args.target_kl,
        max_grad_norm=args.max_grad_norm,
        kl_anchor_coef=args.kl_anchor_coef,
        kl_anchor_checkpoint=args.kl_anchor_checkpoint,
        resume_from=args.resume_from,
        game_mode=args.game_mode,
        player_b_bot_mode=args.player_b_bot_mode,
        num_areas=args.num_areas,
        entropy_coef=args.entropy_coef,
        aim_coef=args.aim_coef,
        aim_err_max=args.aim_err_max,
        shoot_bootstrap_coef=args.shoot_bootstrap_coef,
        shoot_bootstrap_aim_threshold_deg=args.shoot_bootstrap_aim_threshold_deg,
        jerk_coef=args.jerk_coef,
        spam_coef=args.spam_coef,
        min_log_std=args.min_log_std,
        max_log_std=args.max_log_std,
        scripted_shoot_damage_scale=args.scripted_shoot_damage_scale,
        scripted_shoot_cooldown_mult=args.scripted_shoot_cooldown_mult,
        scripted_shoot_aim_threshold_deg=args.scripted_shoot_aim_threshold_deg,
        scripted_shoot_warmup_iters=args.scripted_shoot_warmup_iters,
        mean_action_saturation_penalty=args.mean_action_saturation_penalty,
        mean_action_saturation_threshold=args.mean_action_saturation_threshold,
        mean_action_saturation_coef=args.mean_action_saturation_coef,
        look_action_l2_reward_coef=args.look_action_l2_reward_coef,
        mean_action_l2_loss_coef=args.mean_action_l2_loss_coef,
        saturation_abs095_stop_rate=args.saturation_abs095_stop_rate,
        saturation_start_iter=args.saturation_start_iter,
        value_only=args.value_only,
        bc_aux_coef=args.bc_aux_coef,
        bc_aux_dataset_path=args.bc_aux_dataset_path,
        bc_aux_batch_size=args.bc_aux_batch_size,
        actor_update_enabled=args.actor_update_enabled,
        value_update_enabled=args.value_update_enabled,
        freeze_look_action_head=args.freeze_look_action_head,
        freeze_movement_action_head=args.freeze_movement_action_head,
        train_shoot_head=args.train_shoot_head,
        train_value_head=args.train_value_head,
    )
    weapon_log = configure_unity_env(out_dir, cfg)
    policy, normalizer, _, _, policy_checkpoint, device = load_policy_and_normalizer(cfg)
    if cfg.value_only or not cfg.actor_update_enabled:
        for name, param in policy.named_parameters():
            param.requires_grad_(bool(cfg.value_update_enabled and name.startswith("value_head.")))
    else:
        if not cfg.value_update_enabled:
            for name, param in policy.named_parameters():
                if name.startswith("value_head."):
                    param.requires_grad_(False)
        if cfg.freeze_movement_action_head or cfg.freeze_look_action_head or cfg.train_shoot_head:
            freeze_cont_rows = []
            if cfg.freeze_movement_action_head:
                freeze_cont_rows.extend([0, 1])
            if cfg.freeze_look_action_head:
                freeze_cont_rows.extend([2, 3])
            if freeze_cont_rows and hasattr(policy, "actor_cont_mean"):
                rows = torch.as_tensor(sorted(set(freeze_cont_rows)), device=policy.actor_cont_mean.weight.device)
                policy.actor_cont_mean.weight.register_hook(lambda grad, rows=rows: grad.index_fill(0, rows, 0.0))
                policy.actor_cont_mean.bias.register_hook(lambda grad, rows=rows: grad.index_fill(0, rows, 0.0))
                if hasattr(policy, "actor_cont_logstd"):
                    policy.actor_cont_logstd.register_hook(lambda grad, rows=rows: grad.index_fill(0, rows, 0.0))
            if cfg.train_shoot_head and hasattr(policy, "actor_binary_logits"):
                rows = torch.as_tensor([1, 2, 3], device=policy.actor_binary_logits.weight.device)
                policy.actor_binary_logits.weight.register_hook(lambda grad, rows=rows: grad.index_fill(0, rows, 0.0))
                policy.actor_binary_logits.bias.register_hook(lambda grad, rows=rows: grad.index_fill(0, rows, 0.0))
    if cfg.bc_aux_coef > 0.0:
        if not cfg.bc_aux_dataset_path:
            raise RuntimeError("bc_aux_coef > 0 requires bc_aux_dataset_path")
        aux = np.load(cfg.bc_aux_dataset_path)
        if "obs" not in aux or "action" not in aux:
            raise RuntimeError(f"BC aux dataset must contain obs/action keys, found {list(aux.keys())}")
        cfg.bc_aux_obs_array = np.asarray(aux["obs"], dtype=np.float32)
        cfg.bc_aux_action_array = np.asarray(aux["action"], dtype=np.float32)
        if cfg.bc_aux_obs_array.ndim != 2 or cfg.bc_aux_obs_array.shape[1] != 45:
            raise RuntimeError(f"BC aux obs shape invalid: {cfg.bc_aux_obs_array.shape}")
        if cfg.bc_aux_action_array.ndim != 2 or cfg.bc_aux_action_array.shape[1] != 8:
            raise RuntimeError(f"BC aux action shape invalid: {cfg.bc_aux_action_array.shape}")
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("no trainable parameters for PPO")
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.lr)
    anchor_policy = None
    if cfg.kl_anchor_coef > 0.0:
        anchor_policy = copy.deepcopy(policy)
        if cfg.kl_anchor_checkpoint:
            anchor_cfg = copy.deepcopy(cfg)
            anchor_cfg.resume_from = cfg.kl_anchor_checkpoint
            anchor_policy, _, _, _, _, _ = load_policy_and_normalizer(anchor_cfg)
        anchor_policy.to(device)
        anchor_policy.eval()
        for param in anchor_policy.parameters():
            param.requires_grad_(False)
    write_json(out_dir / "run_manifest.json", manifest(cfg, sys.argv, policy_checkpoint))
    serializable_cfg = {k: v for k, v in cfg.__dict__.items() if not k.endswith("_array")}
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump({"runtime_config": serializable_cfg}, sort_keys=True), encoding="utf-8")
    status = "PASS"
    failures: list[str] = []
    best_bscore = float("-inf")
    best_retention = float("-inf")
    metrics: dict = {
        "start_utc": utc_now(),
        "normalizer_path": str(normalizer.path),
        "normalizer_sha256": sha256_file(normalizer.path),
        "updates_requested": args.updates,
        "policy_init_source": getattr(policy, "policy_init_source", "unknown"),
        "self_play": False,
        "checkpoints": [],
    }
    env = None
    initial_state = {
        k: v.detach().cpu().clone()
        for k, v in policy.state_dict().items()
        if torch.is_floating_point(v)
    }
    try:
        env = launch_unity_env(cfg)
        behavior_name, action_spec, spec_report = validate_specs(env)
        metrics["runtime_spec"] = spec_report
        zero_fire_streak = 0
        zero_shoot_streak = 0
        zero_hit_high_aim_streak = 0
        for update in range(args.updates):
            buffer, rollout = collect_rollout(policy, normalizer, env, behavior_name, action_spec, cfg, device)
            loss = compute_loss_and_backward(policy, buffer, cfg, optimizer, anchor_policy=anchor_policy, normalizer=normalizer)
            rewards = np.asarray(rollout["raw_rewards"], dtype=np.float32)
            weapon_counts = summarize_weapon_events(collect_all_weapon_events(weapon_log))
            update_metrics = {
                "update_index": update + 1,
                "learning_rate": cfg.lr,
                "reward_mean": float(rewards.mean()) if rewards.size else 0.0,
                "episode_length_mean": float(np.mean(rollout["episode_lengths"])) if rollout["episode_lengths"] else 0.0,
                "death_count": int(rollout["terminal_count"]),
                "terminal_count": int(rollout["terminal_count"]),
                "fire_count_A": int(rollout["fire_count"]),
                "fire_count_B": None,
                "shoot_pressed_count": int(rollout.get("shoot_pressed_count", 0)),
                "fired_to_pressed_ratio": float(rollout.get("fired_to_pressed_ratio", 0.0)),
                "hit_count": int(rollout.get("hit_count", 0)),
                "aim_mean": float(rollout.get("aim_mean", 999.0)),
                "action_stats": rollout.get("action_stats", {}),
                "reward_component_means": rollout.get("reward_component_means", {}),
                "reward_component_nonzero": rollout.get("reward_component_nonzero", {}),
                "env_reward_mean": float(rollout.get("env_reward_mean", 0.0)),
                "env_reward_nonzero_count": int(rollout.get("env_reward_nonzero_count", 0)),
                "actor_cont_logstd": policy.actor_cont_logstd.detach().cpu().float().tolist() if hasattr(policy, "actor_cont_logstd") else [],
                "blocked_fire_count_by_reason": weapon_counts["blocked_fire_count_by_reason"],
                "hard_blocked_fire_count_by_reason": weapon_counts["hard_blocked_fire_count_by_reason"],
                "owner_mismatch_count": weapon_counts["owner_mismatch_count"],
            }
            action_stats = update_metrics.get("action_stats") or {}
            update_metrics.update({
                "mu_look_sat95": float(action_stats.get("mu_look_abs_gt_0_95_rate", 0.0)),
                "mu_look_sat80": float(action_stats.get("mu_look_abs_gt_0_80_rate", 0.0)),
                "mu_continuous_sat95": float(action_stats.get("mu_continuous_abs_gt_0_95_rate", 0.0)),
                "mu_continuous_sat80": float(action_stats.get("mu_continuous_abs_gt_0_80_rate", 0.0)),
                "sampled_look_sat95": float(action_stats.get("sampled_look_abs_gt_0_95_rate", action_stats.get("look_abs_gt_0_95_rate", 0.0))),
                "sampled_look_sat80": float(action_stats.get("sampled_look_abs_gt_0_80_rate", action_stats.get("look_abs_gt_0_80_rate", 0.0))),
                "sampled_continuous_sat95": float(action_stats.get("sampled_continuous_abs_gt_0_95_rate", action_stats.get("continuous_abs_gt_0_95_rate", 0.0))),
                "sampled_continuous_sat80": float(action_stats.get("sampled_continuous_abs_gt_0_80_rate", action_stats.get("continuous_abs_gt_0_80_rate", 0.0))),
                "env_applied_look_sat95": float(action_stats.get("env_applied_look_abs_gt_0_95_rate", 0.0)),
                "env_applied_look_sat80": float(action_stats.get("env_applied_look_abs_gt_0_80_rate", 0.0)),
                "env_applied_continuous_sat95": float(action_stats.get("env_applied_continuous_abs_gt_0_95_rate", 0.0)),
                "env_applied_continuous_sat80": float(action_stats.get("env_applied_continuous_abs_gt_0_80_rate", 0.0)),
            })
            update_metrics.update(loss)
            if abs(update_metrics["approx_kl"]) > cfg.target_kl:
                raise RuntimeError(f"target KL exceeded: {update_metrics['approx_kl']} > {cfg.target_kl}")
            if update_metrics["hard_blocked_fire_count_by_reason"] or update_metrics["owner_mismatch_count"] != 0:
                raise RuntimeError("hard weapon-state failure observed during update")
            zero_fire_streak = zero_fire_streak + 1 if update_metrics["fire_count_A"] == 0 else 0
            zero_shoot_streak = zero_shoot_streak + 1 if update_metrics["shoot_pressed_count"] == 0 else 0
            if update_metrics["aim_mean"] > 170.0 and update_metrics["hit_count"] == 0:
                zero_hit_high_aim_streak += 1
            else:
                zero_hit_high_aim_streak = 0
            cont_sat95 = float(update_metrics.get("mu_look_sat95", update_metrics.get("sampled_continuous_sat95", (update_metrics.get("action_stats") or {}).get("continuous_abs_gt_0_95_rate", 0.0))))
            if zero_fire_streak >= 10:
                raise RuntimeError("10-iteration zero-fire streak")
            if zero_shoot_streak >= 5:
                raise RuntimeError("shoot_pressed collapsed to zero for 5 consecutive iterations")
            if zero_hit_high_aim_streak >= 10:
                raise RuntimeError("persistent aim >170 with zero hits")
            if update + 1 > cfg.saturation_start_iter and cont_sat95 > cfg.saturation_abs095_stop_rate:
                raise RuntimeError(f"policy mean look abs>0.95 rate after iter {cfg.saturation_start_iter} exceeded {cfg.saturation_abs095_stop_rate}: {cont_sat95}")
            if abs(update_metrics["approx_kl"]) > 1.0:
                raise RuntimeError(f"approx KL exploded: {update_metrics['approx_kl']}")
            if update_metrics["entropy"] < 1e-4:
                raise RuntimeError("entropy collapsed near zero")
            if hasattr(policy, "actor_cont_logstd"):
                with torch.no_grad():
                    policy.actor_cont_logstd.clamp_(cfg.min_log_std, cfg.max_log_std)
            append_jsonl(out_dir / "ppo_update_metrics.jsonl", update_metrics)
            append_jsonl(out_dir / "ppo_train.jsonl", update_metrics)
            ckpt_path = ckpt_dir / f"canary_update_{update + 1:03d}.pt"
            torch.save({
                "actor_critic_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "update": update + 1,
                "metrics": update_metrics,
                "normalizer_path": str(normalizer.path),
                "normalizer_sha256": sha256_file(normalizer.path),
            }, ckpt_path)
            metrics["checkpoints"].append({
                "path": str(ckpt_path),
                "sha256": sha256_file(ckpt_path),
                "update": update + 1,
            })
            latest_path = out_dir / "latest.pt"
            torch.save(torch.load(ckpt_path, map_location="cpu", weights_only=False), latest_path)
            bscore = float(update_metrics.get("hit_count", 0)) - 0.03 * float(update_metrics.get("aim_mean", 999.0)) + 0.1 * float(update_metrics.get("fire_count_A", 0))
            if bscore > best_bscore:
                best_bscore = bscore
                torch.save(torch.load(ckpt_path, map_location="cpu", weights_only=False), out_dir / "best_behavioral.pt")
            retention_score = bscore + 0.5 * float(update_metrics.get("fire_count_A", 0) > 0)
            if update + 1 >= max(1, args.updates - 20) and retention_score > best_retention:
                best_retention = retention_score
                torch.save(torch.load(ckpt_path, map_location="cpu", weights_only=False), out_dir / "best_retention.pt")
        delta = parameter_delta(initial_state, policy)
        metrics["parameter_abs_delta_sum"] = delta
        if delta <= 0.0:
            raise RuntimeError("parameters did not change during canary")
        final_ckpt = Path(metrics["checkpoints"][-1]["path"])
        reloaded = copy.deepcopy(policy)
        state = torch.load(final_ckpt, map_location=device, weights_only=False)
        reloaded.load_state_dict(state["actor_critic_state"], strict=True)
        reloaded.to(device)
        metrics["reload_eval"] = evaluate_reloaded(
            reloaded,
            normalizer,
            env,
            behavior_name,
            action_spec,
            cfg,
            device,
            decisions=args.eval_decisions,
        )
        if not metrics["reload_eval"]["reloaded_actions_valid"]:
            raise RuntimeError("reload evaluation produced invalid actions")
    except Exception as exc:
        status = "FAIL"
        failures.append(str(exc))
    finally:
        if env is not None:
            env.close()
    weapon_summary = summarize_weapon_events(collect_all_weapon_events(weapon_log))
    metrics.update(weapon_summary)
    metrics["end_utc"] = utc_now()
    metrics["status"] = status
    metrics["failures"] = failures
    if weapon_summary["owner_mismatch_count"] != 0 or weapon_summary["hard_blocked_fire_count_by_reason"]:
        metrics["status"] = "FAIL"
        metrics["failures"].append("hard weapon-state failure observed")
    write_json(out_dir / "tiny_ppo_canary_summary.json", metrics)
    write_json(out_dir / "run_summary.json", metrics)
    (out_dir / "tiny_ppo_canary_report.md").write_text(
        "\n".join([
            "# Phase 3v2 Tiny PPO Canary",
            "",
            f"Status: {metrics['status']}",
            f"Updates: {args.updates}",
            f"Parameter abs delta sum: {metrics.get('parameter_abs_delta_sum')}",
            f"Checkpoints: {len(metrics.get('checkpoints', []))}",
            f"Reload eval: {metrics.get('reload_eval')}",
            f"Failures: {metrics['failures']}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(metrics["status"])
    if metrics["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
