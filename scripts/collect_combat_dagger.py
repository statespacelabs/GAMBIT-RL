#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.combat_runtime import (
    OBS_DIM,
    RuntimeConfig,
    collect_all_weapon_events,
    configure_unity_env,
    launch_unity_env,
    load_policy_and_normalizer,
    sha256_file,
    summarize_weapon_events,
    validate_specs,
    write_json,
)
from gambit.rl.online_rl.selfplay_runner import make_action_tuple, policy_action_to_buffers
from scripts.combat_observations import build_slot_map_from_agent_ids

AREA_SPACING = 500.0
HARD_BLOCK_REASONS = {
    "NO_OWNER",
    "WRONG_OWNER",
    "WEAPON_DISABLED",
    "WEAPON_INACTIVE",
    "UNKNOWN",
    "ROUND_OVER",
    "DEAD_OR_RESPAWNING",
}
DEFAULT_CKPT = "experiments/phase3v2/phase3v2_b_selected_artifacts/gvg_seed_v000.pt"
DEFAULT_NORMALIZER = "experiments/phase3v2/normalizers/phase3v2_a_obs45_no_pressure.pt"


def collect_step_records(dec, ter):
    records = []
    if len(dec):
        obs_batch = np.asarray(dec.obs[0], dtype=np.float32)
        for row, aid_raw in enumerate(dec.agent_id):
            records.append((int(aid_raw), obs_batch[row]))
    if len(ter):
        obs_batch = np.asarray(ter.obs[0], dtype=np.float32)
        for row, aid_raw in enumerate(ter.agent_id):
            records.append((int(aid_raw), obs_batch[row]))
    return records


def side_area(obs: np.ndarray, num_areas: int) -> tuple[str, int]:
    area = int(round(float(obs[0]) / AREA_SPACING))
    area = max(0, min(num_areas - 1, area))
    return ("A" if float(obs[2]) <= -78.5 else "B"), area


def checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        return payload.get("actor_critic_state", payload.get("model_state_dict", payload))
    return payload


def restore_policy_tensors(policy: torch.nn.Module, ckpt_path: str | Path) -> list[str]:
    state = checkpoint_state(ckpt_path)
    current = policy.state_dict()
    restored: list[str] = []
    with torch.no_grad():
        for name, tensor in state.items():
            if name not in current or not torch.is_tensor(tensor):
                continue
            if tuple(current[name].shape) != tuple(tensor.shape):
                continue
            current[name].copy_(tensor.to(device=current[name].device, dtype=current[name].dtype))
            restored.append(name)
    return restored


class JsonlTailer:
    def __init__(self, path: Path):
        self.path = path
        self.offset = 0

    def read_new(self) -> list[dict]:
        if not self.path.exists():
            return []
        data = self.path.read_text(encoding="utf-8", errors="replace")
        chunk = data[self.offset :]
        self.offset = len(data)
        out = []
        for line in chunk.splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"event": "json_decode_error", "raw": line})
        return out


def action_mode_for_step(mode: str, rng: np.random.Generator, step: int) -> str:
    if mode == "mixed":
        if step % 10 in (0, 1, 2):
            return "deterministic"
        return "stochastic" if rng.random() < 0.65 else "deterministic"
    return mode


def look_sat(action: np.ndarray) -> float:
    return float(np.max(np.abs(action[2:4])))


def target_visible(obs: np.ndarray) -> bool:
    aim = float(max(0.0, obs[22]))
    return bool(np.isfinite(aim) and aim < 150.0)


def keep_for_emphasis(
    emphasis: str,
    side: str,
    obs: np.ndarray,
    post_reset: bool,
    damage_taken: float,
    rng: np.random.Generator,
) -> bool:
    aim = float(max(0.0, obs[22]))
    if emphasis == "A":
        return side == "A"
    if emphasis == "B":
        return side == "B"
    if emphasis == "sideB_heavy":
        return side == "B" or rng.random() < 0.12
    if emphasis == "highaim":
        return 60.0 <= aim <= 150.0 or rng.random() < 0.25
    if emphasis == "midaim":
        return 35.0 <= aim < 90.0
    if emphasis == "underfire_strict":
        return damage_taken > 0.0
    if emphasis == "postreset":
        return post_reset or rng.random() < 0.25
    if emphasis == "underfire":
        return damage_taken > 0.0 or rng.random() < 0.35
    return True


def append_record(store: dict[str, list], rec: dict) -> None:
    for key, value in rec.items():
        store[key].append(value)


def summarize_store(
    store: dict[str, list],
    weapon_summary: dict,
    args: argparse.Namespace,
    status: str,
    failures: list[str],
) -> dict:
    n = len(store["obs"])
    sides = Counter(str(x) for x in store["agent_side"])
    hard = weapon_summary.get("hard_blocked_fire_count_by_reason") or {}
    owner = int(weapon_summary.get("owner_mismatch_count", 0) or 0)
    summary = {
        "status": status,
        "failures": failures,
        "job": args.job_name,
        "emphasis": args.emphasis,
        "action_mode": args.action_mode,
        "samples": n,
        "agent_side_counts": dict(sides),
        "A_samples": int(sides.get("A", 0)),
        "B_samples": int(sides.get("B", 0)),
        "post_reset_count": int(sum(float(x) > 0.5 for x in store["post_reset_flag"])),
        "terminal_count": int(sum(float(x) > 0.5 for x in store["terminal_flag"])),
        "death_count": int(sum(float(x) > 0.5 for x in store["death_flag"])),
        "under_fire_count": int(sum(float(x) > 0.0 for x in store["damage_taken"])),
        "shoot_pressed_count": int(sum(float(x) > 0.5 for x in store["shoot_pressed"])),
        "shot_fired_count": int(sum(float(x) > 0.5 for x in store["shot_fired"])),
        "hit_count": int(sum(float(x) > 0.5 for x in store["hit"])),
        "target_visible_count": int(sum(float(x) > 0.5 for x in store["target_visible"])),
        "target_los_count": int(sum(float(x) > 0.5 for x in store["target_los"])),
        "combat_ready_count": int(sum(float(x) > 0.5 for x in store["combat_ready"])),
        "aim_mean": float(mean(store["aim_error"])) if n else 999.0,
        "distance_mean": float(mean(store["distance_to_target"])) if n else 0.0,
        "mu_look_sat95_rate": float(np.mean(np.asarray(store["mu_look_sat"], dtype=np.float32) > 0.95)) if n else 0.0,
        "sampled_look_sat95_rate": float(np.mean(np.asarray(store["sampled_look_sat"], dtype=np.float32) > 0.95)) if n else 0.0,
        "env_look_sat95_rate": float(np.mean(np.asarray(store["env_look_sat"], dtype=np.float32) > 0.95)) if n else 0.0,
        "obs44_mismatch_count": 0,
        **weapon_summary,
    }
    if hard or owner:
        summary["status"] = "FAIL"
    if n < args.target_samples and status == "PASS":
        summary["status"] = "FAIL"
        summary["failures"] = list(summary["failures"]) + [f"samples {n} < target {args.target_samples}"]
    return summary


def write_shard(out: Path, store: dict[str, list]) -> None:
    np.savez_compressed(
        out / "policy_state_shard.npz",
        obs=np.asarray(store["obs"], dtype=np.float32),
        agent_side=np.asarray(store["agent_side"], dtype=str),
        area_id=np.asarray(store["area_id"], dtype=np.int32),
        episode_id=np.asarray(store["episode_id"], dtype=np.int32),
        agent_id=np.asarray(store["agent_id"], dtype=np.int32),
        policy_mu=np.asarray(store["policy_mu"], dtype=np.float32),
        policy_logstd=np.asarray(store["policy_logstd"], dtype=np.float32),
        sampled_action=np.asarray(store["sampled_action"], dtype=np.float32),
        env_applied_action=np.asarray(store["env_applied_action"], dtype=np.float32),
        mu_look_sat=np.asarray(store["mu_look_sat"], dtype=np.float32),
        sampled_look_sat=np.asarray(store["sampled_look_sat"], dtype=np.float32),
        env_look_sat=np.asarray(store["env_look_sat"], dtype=np.float32),
        shoot_pressed=np.asarray(store["shoot_pressed"], dtype=np.float32),
        shot_fired=np.asarray(store["shot_fired"], dtype=np.float32),
        hit=np.asarray(store["hit"], dtype=np.float32),
        damage_dealt=np.asarray(store["damage_dealt"], dtype=np.float32),
        damage_taken=np.asarray(store["damage_taken"], dtype=np.float32),
        aim_error=np.asarray(store["aim_error"], dtype=np.float32),
        yaw_error_deg=np.asarray(store["yaw_error_deg"], dtype=np.float32),
        pitch_error_deg=np.asarray(store["pitch_error_deg"], dtype=np.float32),
        true_aim_error=np.asarray(store["true_aim_error"], dtype=np.float32),
        distance_to_target=np.asarray(store["distance_to_target"], dtype=np.float32),
        target_visible=np.asarray(store["target_visible"], dtype=np.float32),
        target_los=np.asarray(store["target_los"], dtype=np.float32),
        combat_ready=np.asarray(store["combat_ready"], dtype=np.float32),
        post_reset_flag=np.asarray(store["post_reset_flag"], dtype=np.float32),
        terminal_flag=np.asarray(store["terminal_flag"], dtype=np.float32),
        death_flag=np.asarray(store["death_flag"], dtype=np.float32),
        blocked_fire_reason=np.asarray(store["blocked_fire_reason"], dtype=str),
        action_mode_used=np.asarray(store["action_mode_used"], dtype=str),
        collector_checkpoint=np.asarray(store["collector_checkpoint"], dtype=str),
    )


def run(args: argparse.Namespace) -> dict:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = RuntimeConfig(
        env_path=args.env_path,
        normalizer_path=args.normalizer,
        output_dir=str(out),
        base_port=args.base_port,
        seed=args.seed,
        time_scale=args.time_scale,
        device=args.device,
        resume_from=args.checkpoint,
        game_mode="GambitVsGambit",
        player_b_bot_mode="None",
        num_areas=args.num_areas,
        min_log_std=-20.0,
        max_log_std=5.0,
    )
    weapon_log = configure_unity_env(out, cfg)
    tailer = JsonlTailer(weapon_log)
    rng = np.random.default_rng(args.seed)
    cfg_a = copy.deepcopy(cfg)
    cfg_b = copy.deepcopy(cfg)
    policy_a, normalizer, _, _, _, device = load_policy_and_normalizer(cfg_a)
    policy_b, _, _, _, _, _ = load_policy_and_normalizer(cfg_b)
    restored_a = restore_policy_tensors(policy_a, args.checkpoint)
    restored_b = restore_policy_tensors(policy_b, args.checkpoint)
    policy_a.eval()
    policy_b.eval()
    if hasattr(policy_a, "set_ppo_mode"):
        policy_a.set_ppo_mode(training=False)
    if hasattr(policy_b, "set_ppo_mode"):
        policy_b.set_ppo_mode(training=False)
    for param in policy_a.parameters():
        param.requires_grad_(False)
    for param in policy_b.parameters():
        param.requires_grad_(False)
    write_json(
        out / "policy_state_manifest.json",
        {
            "phase": "phase3v2_b_gvg_dagger_v2_policy_state_collection",
            "job": args.job_name,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
            "normalizer": args.normalizer,
            "normalizer_sha256": sha256_file(Path(args.normalizer)),
            "env_path": args.env_path,
            "env_sha256": sha256_file(Path(args.env_path)),
            "target_samples": args.target_samples,
            "emphasis": args.emphasis,
            "action_mode": args.action_mode,
            "num_areas": args.num_areas,
            "seed": args.seed,
            "base_port": args.base_port,
            "time_scale": args.time_scale,
            "side_assignment": "stable_initial_agent_id_slot_map",
            "derived_fields": {
                "agent_side": "A if obs[2] <= -78.5 else B; area from round(obs[0]/500)",
                "target_visible": "derived as finite obs[22] < 150 deg",
                "target_los": "same conservative derived indicator as target_visible; Unity LOS is not separately exposed in obs45",
                "combat_ready": "derived as not terminal and no hard blocked-fire reason on the transition",
                "distance_to_target": "obs[9]",
                "true_aim_error": "obs[38], logged only; not used as reward",
            },
            "restored_tensor_count_a": len(restored_a),
            "restored_tensor_count_b": len(restored_b),
        },
    )
    keys = [
        "obs",
        "agent_side",
        "area_id",
        "episode_id",
        "agent_id",
        "policy_mu",
        "policy_logstd",
        "sampled_action",
        "env_applied_action",
        "mu_look_sat",
        "sampled_look_sat",
        "env_look_sat",
        "shoot_pressed",
        "shot_fired",
        "hit",
        "damage_dealt",
        "damage_taken",
        "aim_error",
        "yaw_error_deg",
        "pitch_error_deg",
        "true_aim_error",
        "distance_to_target",
        "target_visible",
        "target_los",
        "combat_ready",
        "post_reset_flag",
        "terminal_flag",
        "death_flag",
        "blocked_fire_reason",
        "action_mode_used",
        "collector_checkpoint",
    ]
    store = {key: [] for key in keys}
    hidden: dict[int, torch.Tensor] = {}
    episode: dict[int, int] = {}
    decisions_since_reset: dict[int, int] = {}
    next_episode_id = 0
    status = "PASS"
    failures: list[str] = []
    env = None
    step_idx = 0
    max_steps = args.max_env_steps if args.max_env_steps > 0 else args.target_samples * 8
    try:
        env = launch_unity_env(cfg)
        behavior, action_spec, _spec = validate_specs(env)
        slot_seed_obs = {}
        for _ in range(100):
            dec0, ter0 = env.get_steps(behavior)
            for aid, obs in collect_step_records(dec0, ter0):
                slot_seed_obs[aid] = obs
            if len(slot_seed_obs) >= args.num_areas * 2:
                break
            env.step()
            step_idx += 1
        slot_map = build_slot_map_from_agent_ids(slot_seed_obs, args.num_areas) if slot_seed_obs else {}
        while len(store["obs"]) < args.target_samples and step_idx < max_steps:
            dec, _ter = env.get_steps(behavior)
            if len(dec) == 0:
                t0 = time.monotonic()
                env.step()
                if time.monotonic() - t0 > cfg.deadlock_step_seconds:
                    raise RuntimeError("Unity deadlock while waiting for GvG decisions")
                step_idx += 1
                continue
            obs_batch = np.asarray(dec.obs[0], dtype=np.float32)
            pending: dict[int, dict] = {}
            mode_used = action_mode_for_step(args.action_mode, rng, step_idx)
            for row, aid_raw in enumerate(dec.agent_id):
                aid = int(aid_raw)
                obs = obs_batch[row].astype(np.float32)
                side, area = slot_map.get(aid, side_area(obs, args.num_areas))
                policy = policy_a if side == "A" else policy_b
                if aid not in episode:
                    episode[aid] = next_episode_id
                    next_episode_id += 1
                    decisions_since_reset[aid] = 0
                post_reset = decisions_since_reset.get(aid, 0) < args.post_reset_window
                hidden_in = hidden.get(aid, policy.init_hidden(1, device))
                obs_norm = normalizer.normalize_tensor(
                    torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, OBS_DIM)
                )
                with torch.no_grad():
                    dist, _value, next_hidden = policy(obs_norm, hidden_in)
                    mu_c, _mu_raw = dist.mode
                    sample_c, _sample_raw = dist.sample()
                mu = mu_c.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
                sample = sample_c.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
                action = mu.copy() if mode_used == "deterministic" else sample.copy()
                logstd = (
                    policy.actor_cont_logstd.detach().cpu().numpy().astype(np.float32)
                    if hasattr(policy, "actor_cont_logstd")
                    else np.zeros(4, dtype=np.float32)
                )
                continuous, discrete = policy_action_to_buffers(action, action_spec)
                env.set_action_for_agent(behavior, aid, make_action_tuple(continuous, discrete))
                pending[aid] = {
                    "obs": obs,
                    "side": side,
                    "area": area,
                    "episode": episode[aid],
                    "post_reset": post_reset,
                    "hidden": next_hidden.detach(),
                    "mu": mu,
                    "sample": sample,
                    "action": action,
                    "logstd": logstd,
                    "mode_used": mode_used,
                }
            t0 = time.monotonic()
            env.step()
            dur = time.monotonic() - t0
            if dur > cfg.deadlock_step_seconds:
                raise RuntimeError(f"Unity deadlock/slow step during GvG collection: {dur:.1f}s")
            new_events = tailer.read_new()
            blocked_reasons = [
                str(event.get("blocked_fire_reason", "UNKNOWN"))
                for event in new_events
                if event.get("event") == "blocked_fire"
            ]
            hard_seen = [reason for reason in blocked_reasons if reason in HARD_BLOCK_REASONS]
            reason_for_step = ";".join(blocked_reasons)
            dec2, ter2 = env.get_steps(behavior)
            post: dict[int, tuple[np.ndarray, float, bool]] = {}
            if len(dec2):
                obs2_batch = np.asarray(dec2.obs[0], dtype=np.float32)
                for row, aid_raw in enumerate(dec2.agent_id):
                    post[int(aid_raw)] = (obs2_batch[row].astype(np.float32), float(dec2.reward[row]), False)
            if len(ter2):
                obs2_batch = np.asarray(ter2.obs[0], dtype=np.float32)
                for row, aid_raw in enumerate(ter2.agent_id):
                    post[int(aid_raw)] = (obs2_batch[row].astype(np.float32), float(ter2.reward[row]), True)
            for aid, rec0 in pending.items():
                obs = rec0["obs"]
                obs2, reward, done = post.get(aid, (obs, 0.0, False))
                side = rec0["side"]
                action = rec0["action"]
                shot_fired = float(obs2[44] > 0.5)
                hit = float(reward > 0.05)
                damage_dealt = float(max(0.0, reward))
                damage_taken = float(max(0.0, -reward))
                post_reset = bool(rec0["post_reset"])
                if keep_for_emphasis(args.emphasis, side, obs, post_reset, damage_taken, rng):
                    combat_ready = bool((not done) and not hard_seen)
                    append_record(
                        store,
                        {
                            "obs": obs.copy(),
                            "agent_side": side,
                            "area_id": int(rec0["area"]),
                            "episode_id": int(rec0["episode"]),
                            "agent_id": int(aid),
                            "policy_mu": rec0["mu"].copy(),
                            "policy_logstd": rec0["logstd"].copy(),
                            "sampled_action": rec0["sample"].copy(),
                            "env_applied_action": action.copy(),
                            "mu_look_sat": look_sat(rec0["mu"]),
                            "sampled_look_sat": look_sat(rec0["sample"]),
                            "env_look_sat": look_sat(action),
                            "shoot_pressed": float(action[4] > 0.5),
                            "shot_fired": shot_fired,
                            "hit": hit,
                            "damage_dealt": damage_dealt,
                            "damage_taken": damage_taken,
                            "aim_error": float(max(0.0, obs[22])),
                            "yaw_error_deg": float(obs[20]),
                            "pitch_error_deg": float(obs[21]),
                            "true_aim_error": float(max(0.0, obs[38]))
                            if len(obs) > 38 and np.isfinite(obs[38])
                            else 999.0,
                            "distance_to_target": float(obs[9])
                            if len(obs) > 9 and np.isfinite(obs[9])
                            else -1.0,
                            "target_visible": float(target_visible(obs)),
                            "target_los": float(target_visible(obs)),
                            "combat_ready": float(combat_ready),
                            "post_reset_flag": float(post_reset),
                            "terminal_flag": float(done),
                            "death_flag": float(done and reward < 0.0),
                            "blocked_fire_reason": reason_for_step,
                            "action_mode_used": rec0["mode_used"],
                            "collector_checkpoint": args.checkpoint,
                        },
                    )
                if done:
                    episode[aid] = next_episode_id
                    next_episode_id += 1
                    decisions_since_reset[aid] = 0
                    hidden[aid] = (policy_a if side == "A" else policy_b).init_hidden(1, device)
                else:
                    decisions_since_reset[aid] = decisions_since_reset.get(aid, 0) + 1
                    hidden[aid] = rec0["hidden"]
                if len(store["obs"]) >= args.target_samples:
                    break
            step_idx += 1
    except Exception as exc:
        status = "FAIL"
        failures.append(str(exc))
    finally:
        if env is not None:
            env.close()
    weapon_summary = summarize_weapon_events(collect_all_weapon_events(weapon_log))
    write_shard(out, store)
    summary = summarize_store(store, weapon_summary, args, status, failures)
    summary["env_steps"] = step_idx
    summary["max_env_steps"] = max_steps
    write_json(out / "policy_state_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if summary["status"] != "PASS":
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--job-name", default="gvgdagger2_collect")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--normalizer", default=DEFAULT_NORMALIZER)
    parser.add_argument("--target-samples", type=int, required=True)
    parser.add_argument(
        "--emphasis",
        choices=["A", "B", "sideB_heavy", "postreset", "underfire", "underfire_strict", "highaim", "midaim", "balanced"],
        required=True,
    )
    parser.add_argument("--action-mode", choices=["deterministic", "stochastic", "mixed"], default="mixed")
    parser.add_argument("--seed", type=int, default=9200)
    parser.add_argument("--base-port", type=int, default=64500)
    parser.add_argument("--env-path", default="artifacts/unity/combat/Builds/BotArena.x86_64")
    parser.add_argument("--time-scale", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-areas", type=int, default=2)
    parser.add_argument("--post-reset-window", type=int, default=8)
    parser.add_argument("--max-env-steps", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
