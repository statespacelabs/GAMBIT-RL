#!/usr/bin/env python3
"""Phase 3v2 PPO rollout/buffer/loss smoke."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.combat_runtime import (
    ACTION_DIM,
    OBS_DIM,
    RuntimeConfig,
    action_to_unity,
    append_jsonl,
    assert_finite_tensor,
    build_reward_builder,
    collect_all_weapon_events,
    configure_unity_env,
    explained_variance,
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
from gambit.rl.online_rl.rollout_buffer import RecurrentRolloutBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-path", default=RuntimeConfig.env_path)
    parser.add_argument("--normalizer-path", default=RuntimeConfig.normalizer_path)
    parser.add_argument("--ppo-config", default=RuntimeConfig.ppo_config)
    parser.add_argument("--distill-config", default=RuntimeConfig.distill_config)
    parser.add_argument(
        "--output-dir",
        default="experiments/phase3v2/phase3v2_a_ppo_loss_smoke",
    )
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=51205)
    parser.add_argument("--seed", type=int, default=7411)
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--one-optimizer-step", action="store_true")
    return parser.parse_args()


def collect_rollout(policy, normalizer, env, behavior_name, action_spec, cfg, device):
    reward_builder = build_reward_builder(cfg)
    buffer = RecurrentRolloutBuffer(size=cfg.rollout_steps, device=str(device))
    hidden = policy.init_hidden(1, device)
    prev_action = None
    agent_id, obs, _ = get_first_decision(env, behavior_name)
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    cur_reward = 0.0
    cur_len = 0
    raw_rewards = []
    aim_values = []
    sampled_action_values = []
    mu_action_values = []
    env_action_values = []
    component_sums: dict[str, float] = {}
    component_nonzero: dict[str, int] = {}
    env_rewards = []
    terminal_count = 0
    fire_count = 0
    shoot_pressed_count = 0
    hit_count = 0

    for _ in range(cfg.rollout_steps):
        aim_values.append(float(max(0.0, obs[22])))
        obs_norm = normalizer.normalize_tensor(
            torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, OBS_DIM)
        )
        hidden_before = hidden.detach().clone()
        with torch.no_grad():
            dist, value, next_hidden = policy(obs_norm, hidden)
            action_clamped, action_raw = dist.sample()
            mode_clamped, _ = dist.mode
            log_prob = dist.log_prob(action_raw)
        action_np = action_clamped.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        mu_np = mode_clamped.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        sampled_action_values.append(action_np.copy())
        mu_action_values.append(mu_np.copy())
        env_action_values.append(action_np.copy())
        if action_np[4] > 0.5:
            shoot_pressed_count += 1
        action_to_unity(env, behavior_name, action_spec, agent_id, action_np)
        t0 = time.monotonic()
        env.step()
        if time.monotonic() - t0 > cfg.deadlock_step_seconds:
            raise RuntimeError("Unity step exceeded deadlock threshold")
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        done = False
        env_reward = 0.0
        if agent_id in terminal_steps:
            idx = list(terminal_steps.agent_id).index(agent_id)
            next_obs = np.asarray(terminal_steps.obs[0][idx], dtype=np.float32)
            env_reward = float(terminal_steps.reward[idx])
            done = True
            terminal_count += 1
        elif agent_id in decision_steps:
            idx = list(decision_steps.agent_id).index(agent_id)
            next_obs = np.asarray(decision_steps.obs[0][idx], dtype=np.float32)
            env_reward = float(decision_steps.reward[idx])
        else:
            agent_id, next_obs, env_reward = get_first_decision(env, behavior_name)

        shaped, components = reward_builder.compute(
            obs_current=obs,
            obs_next=next_obs,
            action=action_np,
            previous_action=prev_action,
            env_reward=env_reward,
            is_terminal=done,
        )
        look_l2_coef = float(getattr(cfg, "look_action_l2_reward_coef", 0.0))
        if look_l2_coef > 0.0:
            look_l2_penalty = look_l2_coef * float(action_np[2] ** 2 + action_np[3] ** 2)
            shaped -= look_l2_penalty
            components["look_action_l2_penalty"] = -look_l2_penalty
        env_rewards.append(float(env_reward))
        for key, value_component in components.items():
            value_float = float(value_component)
            component_sums[key] = component_sums.get(key, 0.0) + value_float
            if abs(value_float) > 1.0e-12:
                component_nonzero[key] = component_nonzero.get(key, 0) + 1
        if next_obs[44] > 0.5:
            fire_count += 1
        if env_reward > 0.05:
            hit_count += 1
        buffer.add(
            obs=obs_norm.squeeze(0).squeeze(0),
            action=action_raw.squeeze(0).squeeze(0),
            reward=shaped,
            done=done,
            value=value.squeeze(),
            log_prob=log_prob.squeeze(),
            hidden=hidden_before,
        )
        raw_rewards.append(float(shaped))
        cur_reward += env_reward
        cur_len += 1
        prev_action = action_np.copy()
        obs = next_obs
        if done:
            episode_rewards.append(cur_reward)
            episode_lengths.append(cur_len)
            cur_reward = 0.0
            cur_len = 0
            prev_action = None
            hidden.zero_()
            agent_id, obs, _ = get_first_decision(env, behavior_name)
        else:
            hidden = next_hidden.detach()

    last_obs_norm = normalizer.normalize_tensor(
        torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, OBS_DIM)
    )
    with torch.no_grad():
        last_value = policy.get_value(last_obs_norm, hidden).squeeze()
    buffer.compute_returns_and_advantages(last_value=last_value, gamma=cfg.gamma, lam=cfg.gae_lambda)
    sampled_actions = np.asarray(sampled_action_values, dtype=np.float32) if sampled_action_values else np.zeros((0, ACTION_DIM), dtype=np.float32)
    mu_actions = np.asarray(mu_action_values, dtype=np.float32) if mu_action_values else np.zeros((0, ACTION_DIM), dtype=np.float32)
    env_actions = np.asarray(env_action_values, dtype=np.float32) if env_action_values else np.zeros((0, ACTION_DIM), dtype=np.float32)

    def summarize_actions(actions: np.ndarray, prefix: str = "") -> dict:
        stem = f"{prefix}_" if prefix else ""
        if not len(actions):
            return {
                f"{stem}mean": [0.0] * ACTION_DIM,
                f"{stem}std": [0.0] * ACTION_DIM,
                f"{stem}min": [0.0] * ACTION_DIM,
                f"{stem}max": [0.0] * ACTION_DIM,
                f"{stem}abs_gt_0_95_rate": [0.0] * 4,
                f"{stem}abs_gt_0_80_rate": [0.0] * 4,
                f"{stem}look_abs_gt_0_95_rate": 0.0,
                f"{stem}look_abs_gt_0_80_rate": 0.0,
                f"{stem}continuous_abs_gt_0_95_rate": 0.0,
                f"{stem}continuous_abs_gt_0_80_rate": 0.0,
            }
        return {
            f"{stem}mean": actions.mean(axis=0).astype(float).tolist(),
            f"{stem}std": actions.std(axis=0).astype(float).tolist(),
            f"{stem}min": actions.min(axis=0).astype(float).tolist(),
            f"{stem}max": actions.max(axis=0).astype(float).tolist(),
            f"{stem}abs_gt_0_95_rate": np.mean(np.abs(actions[:, :4]) > 0.95, axis=0).astype(float).tolist(),
            f"{stem}abs_gt_0_80_rate": np.mean(np.abs(actions[:, :4]) > 0.80, axis=0).astype(float).tolist(),
            f"{stem}look_abs_gt_0_95_rate": float(np.mean(np.abs(actions[:, 2:4]) > 0.95)),
            f"{stem}look_abs_gt_0_80_rate": float(np.mean(np.abs(actions[:, 2:4]) > 0.80)),
            f"{stem}continuous_abs_gt_0_95_rate": float(np.mean(np.abs(actions[:, :4]) > 0.95)),
            f"{stem}continuous_abs_gt_0_80_rate": float(np.mean(np.abs(actions[:, :4]) > 0.80)),
        }

    action_stats = summarize_actions(sampled_actions)
    action_stats.update(summarize_actions(sampled_actions, "sampled"))
    action_stats.update(summarize_actions(mu_actions, "mu"))
    action_stats.update(summarize_actions(env_actions, "env_applied"))
    action_stats["sampled"] = summarize_actions(sampled_actions, "sampled")
    action_stats["mu"] = summarize_actions(mu_actions, "mu")
    action_stats["env_applied"] = summarize_actions(env_actions, "env_applied")
    n = max(1, cfg.rollout_steps)
    reward_component_means = {key: value / n for key, value in component_sums.items()}
    env_reward_arr = np.asarray(env_rewards, dtype=np.float32)
    return buffer, {
        "raw_rewards": raw_rewards,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "terminal_count": terminal_count,
        "fire_count": fire_count,
        "shoot_pressed_count": shoot_pressed_count,
        "fired_to_pressed_ratio": float(fire_count / shoot_pressed_count) if shoot_pressed_count else 0.0,
        "hit_count": hit_count,
        "aim_mean": float(np.mean(aim_values)) if aim_values else 999.0,
        "action_stats": action_stats,
        "reward_component_means": reward_component_means,
        "reward_component_nonzero": component_nonzero,
        "env_reward_mean": float(env_reward_arr.mean()) if env_reward_arr.size else 0.0,
        "env_reward_nonzero_count": int(np.count_nonzero(np.abs(env_reward_arr) > 1.0e-12)) if env_reward_arr.size else 0,
    }


def compute_loss_and_backward(policy, buffer, cfg, optimizer=None, anchor_policy=None, normalizer=None):
    policy.set_ppo_mode(training=True)
    batches = list(buffer.recurrent_minibatches(seq_len=cfg.seq_len, batch_size=cfg.batch_size))
    if not batches:
        raise RuntimeError("no recurrent minibatches available")
    batch = batches[0]
    for key, tensor in batch.items():
        assert_finite_tensor(key, tensor)
    log_prob, entropy, values = policy.evaluate_actions(batch["obs"], batch["actions"], batch["hiddens"])
    assert_finite_tensor("new_log_prob", log_prob)
    assert_finite_tensor("entropy", entropy)
    assert_finite_tensor("values", values)
    ratio = torch.exp(log_prob - batch["log_probs"])
    if not torch.isfinite(ratio).all() or float(ratio.max().item()) > 100.0:
        raise RuntimeError("PPO ratio non-finite or extreme")
    clipped = torch.clamp(ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip)
    policy_loss = -torch.min(ratio * batch["advantages"], clipped * batch["advantages"]).mean()
    value_loss = torch.nn.functional.mse_loss(values, batch["returns"])
    entropy_mean = entropy.mean()
    kl_anchor_loss = torch.zeros((), dtype=policy_loss.dtype, device=policy_loss.device)
    if anchor_policy is not None and getattr(cfg, "kl_anchor_coef", 0.0) > 0.0:
        with torch.no_grad():
            anchor_log_prob, _, _ = anchor_policy.evaluate_actions(batch["obs"], batch["actions"], batch["hiddens"])
        assert_finite_tensor("anchor_log_prob", anchor_log_prob)
        kl_anchor_loss = torch.nn.functional.mse_loss(log_prob, anchor_log_prob)
    mean_dist, _, _ = policy(batch["obs"], batch["hiddens"])
    mean_look = torch.clamp(mean_dist.cont_mean[..., 2:4], -1.0, 1.0)
    saturation_loss = torch.zeros((), dtype=policy_loss.dtype, device=policy_loss.device)
    if getattr(cfg, "mean_action_saturation_penalty", False):
        saturation_loss = torch.relu(mean_look.abs() - float(getattr(cfg, "mean_action_saturation_threshold", 0.70))).pow(2).mean()
    mean_action_l2_loss = torch.zeros((), dtype=policy_loss.dtype, device=policy_loss.device)
    if float(getattr(cfg, "mean_action_l2_loss_coef", 0.0)) > 0.0:
        mean_action_l2_loss = mean_look.pow(2).mean()
    bc_aux_loss = torch.zeros((), dtype=policy_loss.dtype, device=policy_loss.device)
    if float(getattr(cfg, "bc_aux_coef", 0.0)) > 0.0:
        if normalizer is None:
            raise RuntimeError("BC auxiliary loss requires normalizer")
        obs_array = getattr(cfg, "bc_aux_obs_array", None)
        action_array = getattr(cfg, "bc_aux_action_array", None)
        if obs_array is None or action_array is None:
            raise RuntimeError("BC auxiliary arrays were not loaded")
        n_aux = int(obs_array.shape[0])
        batch_size = min(int(getattr(cfg, "bc_aux_batch_size", 256)), n_aux)
        idx = torch.randint(0, n_aux, (batch_size,), device=batch["obs"].device)
        obs_np = obs_array[idx.detach().cpu().numpy()]
        action_np = action_array[idx.detach().cpu().numpy()]
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=batch["obs"].device).view(batch_size, 1, OBS_DIM)
        target_action = torch.as_tensor(action_np, dtype=torch.float32, device=batch["obs"].device).view(batch_size, 1, ACTION_DIM)
        obs_norm = normalizer.normalize_tensor(obs_tensor)
        hidden = policy.init_hidden(batch_size, batch["obs"].device)
        aux_dist, _, _ = policy(obs_norm, hidden)
        cont_pred = torch.clamp(aux_dist.cont_mean, -1.0, 1.0)
        cont_target = target_action[..., :4].clamp(-1.0, 1.0)
        bin_target = target_action[..., 4:].clamp(0.0, 1.0)
        cont_loss = torch.nn.functional.mse_loss(cont_pred, cont_target)
        bin_loss = torch.nn.functional.binary_cross_entropy_with_logits(aux_dist.binary_logits, bin_target)
        bc_aux_loss = cont_loss + bin_loss
    total_loss = (
        policy_loss
        + cfg.value_coef * value_loss
        - cfg.entropy_coef * entropy_mean
        + float(getattr(cfg, "kl_anchor_coef", 0.0)) * kl_anchor_loss
        + float(getattr(cfg, "mean_action_saturation_coef", 0.0)) * saturation_loss
        + float(getattr(cfg, "mean_action_l2_loss_coef", 0.0)) * mean_action_l2_loss
        + float(getattr(cfg, "bc_aux_coef", 0.0)) * bc_aux_loss
    )
    approx_kl = (batch["log_probs"] - log_prob).mean()
    clip_fraction = ((ratio - 1.0).abs() > cfg.ppo_clip).float().mean()
    for name, tensor in {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_mean,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "total_loss": total_loss,
        "kl_anchor_loss": kl_anchor_loss,
        "mean_action_saturation_loss": saturation_loss,
        "mean_action_l2_loss": mean_action_l2_loss,
        "bc_aux_loss": bc_aux_loss,
    }.items():
        assert_finite_tensor(name, tensor)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    bad_grad = []
    for name, param in policy.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            bad_grad.append(name)
    if bad_grad:
        raise RuntimeError(f"non-finite gradients: {bad_grad[:8]}")
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
    if not torch.isfinite(grad_norm):
        raise RuntimeError("grad norm is NaN/Inf")
    if optimizer is not None:
        optimizer.step()
    return {
        "obs_shape": list(buffer.obs[: buffer.pos].shape),
        "action_shape": list(buffer.actions[: buffer.pos].shape),
        "return_mean": float(buffer.returns[: buffer.pos].mean().item()),
        "advantage_mean": float(buffer.advantages[: buffer.pos].mean().item()),
        "advantage_std": float(buffer.advantages[: buffer.pos].std().item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy_mean.item()),
        "approx_kl": float(approx_kl.item()),
        "clip_fraction": float(clip_fraction.item()),
        "total_loss": float(total_loss.item()),
        "kl_anchor_loss": float(kl_anchor_loss.item()),
        "mean_action_saturation_loss": float(saturation_loss.item()),
        "mean_action_l2_loss": float(mean_action_l2_loss.item()),
        "bc_aux_loss": float(bc_aux_loss.item()),
        "grad_norm": float(grad_norm.item()),
        "explained_variance": explained_variance(values, batch["returns"]),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
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
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    weapon_log = configure_unity_env(out_dir)
    policy, normalizer, _, _, policy_checkpoint, device = load_policy_and_normalizer(cfg)
    write_json(out_dir / "run_manifest.json", manifest(cfg, sys.argv, policy_checkpoint))
    status = "PASS"
    failures: list[str] = []
    env = None
    summary = {
        "start_utc": utc_now(),
        "normalizer_path": str(normalizer.path),
        "normalizer_sha256": sha256_file(normalizer.path),
        "optimizer_step_performed": bool(args.one_optimizer_step),
        "policy_init_source": getattr(policy, "policy_init_source", "unknown"),
    }
    try:
        env = launch_unity_env(cfg)
        behavior_name, action_spec, spec_report = validate_specs(env)
        summary["runtime_spec"] = spec_report
        buffer, rollout = collect_rollout(policy, normalizer, env, behavior_name, action_spec, cfg, device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr) if args.one_optimizer_step else None
        loss = compute_loss_and_backward(policy, buffer, cfg, optimizer)
        rewards = np.asarray(rollout["raw_rewards"], dtype=np.float32)
        summary.update(loss)
        summary.update({
            "reward_mean": float(rewards.mean()) if rewards.size else 0.0,
            "reward_min": float(rewards.min()) if rewards.size else 0.0,
            "reward_max": float(rewards.max()) if rewards.size else 0.0,
            "num_decisions": int(buffer.pos),
            "num_episodes": len(rollout["episode_rewards"]),
            "num_terminals": int(rollout["terminal_count"]),
            "fire_count_A": int(rollout["fire_count"]),
            "fire_count_B": None,
            "hit_count": int(rollout.get("hit_count", 0)),
            "aim_mean": float(rollout.get("aim_mean", 999.0)),
        })
        append_jsonl(out_dir / "loss_metrics.jsonl", summary)
    except Exception as exc:
        status = "FAIL"
        failures.append(str(exc))
    finally:
        if env is not None:
            env.close()
    weapon_summary = summarize_weapon_events(collect_all_weapon_events(weapon_log))
    summary.update(weapon_summary)
    summary["end_utc"] = utc_now()
    summary["status"] = status
    summary["failures"] = failures
    if weapon_summary["owner_mismatch_count"] != 0 or weapon_summary["hard_blocked_fire_count_by_reason"]:
        summary["status"] = "FAIL"
        summary["failures"].append("hard weapon-state failure observed")
    write_json(out_dir / "ppo_loss_smoke_summary.json", summary)
    (out_dir / "ppo_loss_smoke_report.md").write_text(
        "\n".join([
            "# Phase 3v2 PPO Loss Smoke",
            "",
            f"Status: {summary['status']}",
            f"Obs shape: {summary.get('obs_shape')}",
            f"Action shape: {summary.get('action_shape')}",
            f"Policy loss: {summary.get('policy_loss')}",
            f"Value loss: {summary.get('value_loss')}",
            f"Entropy: {summary.get('entropy')}",
            f"Approx KL: {summary.get('approx_kl')}",
            f"Grad norm: {summary.get('grad_norm')}",
            f"Failures: {summary['failures']}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(summary["status"])
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
