#!/usr/bin/env python3
"""Shared Phase 3v2 PPO-readiness utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gambit.dataset_preparation.configs import DistillConfig, PPOTrainConfig
from gambit.rl.online_rl.invariant_check import load_recurrent_policy
from gambit.rl.online_rl.actor_critic import RecurrentActorCritic
from gambit.rl.online_rl.core_action_projector import CoreActionProjector
from gambit.rl.online_rl.reward_builder import RewardBuilder
from gambit.rl.online_rl.selfplay_runner import make_action_tuple, policy_action_to_buffers
from gambit.rl.online_rl.telemetry_encoder import TelemetryEncoder

OBS_DIM = 45
ACTION_DIM = 8
HARD_BLOCK_REASONS = {
    "NO_OWNER",
    "WRONG_OWNER",
    "WEAPON_DISABLED",
    "WEAPON_INACTIVE",
    "UNKNOWN",
    "ROUND_OVER",
    "DEAD_OR_RESPAWNING",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_text(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


class CertifiedObsNormalizer:
    """Normalizer for the certified `phase3v2_a_obs45_no_pressure.pt` artifact."""

    def __init__(self, path: str | Path, device: torch.device):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"normalizer not found: {self.path}")
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        self.payload = payload
        self.obs_schema_version = str(payload.get("obs_schema_version", ""))
        self.obs_dim = int(payload.get("obs_dim", -1))
        if self.obs_dim != OBS_DIM:
            raise ValueError(f"normalizer obs_dim {self.obs_dim} != {OBS_DIM}")
        mean = payload.get("mean")
        std = payload.get("effective_standard_deviation", payload.get("standard_deviation"))
        if mean is None or std is None:
            raise ValueError("normalizer missing mean/std tensors")
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.as_tensor(std, dtype=torch.float32, device=device)
        if self.mean.shape != (OBS_DIM,) or self.std.shape != (OBS_DIM,):
            raise ValueError(f"normalizer tensor shapes mean={self.mean.shape} std={self.std.shape}")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("normalizer contains NaN/Inf")
        if torch.any(self.std <= 0):
            bad = torch.nonzero(self.std <= 0).flatten().tolist()
            raise ValueError(f"normalizer std <= 0 at dims {bad}")
        clip_policy = payload.get("clipping_policy", {}) or {}
        self.clip_value = float(clip_policy.get("clip_value", 0.0)) if clip_policy.get("enabled", False) else 0.0

    def normalize_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != OBS_DIM:
            raise ValueError(f"obs last dim {obs.shape[-1]} != {OBS_DIM}")
        out = (obs.to(self.mean.device, dtype=torch.float32) - self.mean) / self.std
        if self.clip_value > 0:
            out = torch.clamp(out, -self.clip_value, self.clip_value)
        if not torch.isfinite(out).all():
            raise ValueError("normalized observation tensor contains NaN/Inf")
        return out


@dataclass
class RuntimeConfig:
    env_path: str = "artifacts/unity/combat/Builds/BotArena.x86_64"
    normalizer_path: str = "experiments/phase3v2/normalizers/phase3v2_a_obs45_no_pressure.pt"
    ppo_config: str = "configs/ppo_production.yaml"
    distill_config: str = "configs/distill_production.yaml"
    output_dir: str = ""
    base_port: int = 51005
    seed: int = 7311
    time_scale: float = 20.0
    device: str = "cuda"
    rollout_steps: int = 128
    seq_len: int = 16
    batch_size: int = 4
    ppo_clip: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.003
    value_coef: float = 0.5
    lr: float = 1.0e-5
    target_kl: float = 0.005
    max_grad_norm: float = 0.5
    kl_anchor_coef: float = 0.0
    kl_anchor_checkpoint: str = ""
    deadlock_step_seconds: float = 30.0
    resume_from: str = ""
    game_mode: str = "GambitVsScripted"
    player_b_bot_mode: str = "StrafeAndFace"
    num_areas: int = 1
    aim_coef: float = 0.0
    aim_err_max: float = 150.0
    aim_reward_mode: str = "proximity_next"
    shoot_bootstrap_coef: float = 0.0
    shoot_bootstrap_aim_threshold_deg: float = 30.0
    shoot_bootstrap_require_fired: bool = True
    jerk_coef: float = 0.0
    spam_coef: float = 0.0
    min_log_std: float = -2.5
    max_log_std: float = -0.1
    mean_action_saturation_penalty: bool = False
    mean_action_saturation_threshold: float = 0.70
    mean_action_saturation_coef: float = 0.0
    look_action_l2_reward_coef: float = 0.0
    mean_action_l2_loss_coef: float = 0.0
    saturation_abs095_stop_rate: float = 0.10
    saturation_start_iter: int = 20
    value_only: bool = False
    bc_aux_coef: float = 0.0
    bc_aux_dataset_path: str = ""
    bc_aux_batch_size: int = 256
    actor_update_enabled: bool = True
    value_update_enabled: bool = True
    freeze_look_action_head: bool = False
    freeze_movement_action_head: bool = False
    train_shoot_head: bool = False
    train_value_head: bool = True
    scripted_shoot_damage_scale: float = 1.0
    scripted_shoot_cooldown_mult: float = 1.0
    scripted_shoot_aim_threshold_deg: float = 15.0
    scripted_shoot_warmup_iters: int = 0
    scripted_shoot_warmup_sec_per_iter: float = 45.0


def configure_unity_env(out_dir: Path, cfg: RuntimeConfig | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    weapon_log = out_dir / "weapon_state_debug.jsonl"
    if weapon_log.exists():
        weapon_log.unlink()
    cfg = cfg or RuntimeConfig()
    os.environ["GAME_MODE"] = cfg.game_mode
    os.environ["PLAYER_B_BOT_MODE"] = cfg.player_b_bot_mode
    os.environ["NUM_AREAS"] = str(cfg.num_areas)
    os.environ["TARGET_HURTBOX_INFLATE"] = "0"
    os.environ["LEARNER_SHOOT_RAY_MODE"] = "current"
    os.environ["RAY_CORRECTION_ALPHA"] = "0"
    os.environ["RAY_CORRECTION_REQUIRE_LOS"] = "1"
    os.environ["RAY_CORRECTION_MAX_ANGLE_DEG"] = "30"
    os.environ["HIT_PROBE_ORACLE"] = "0"
    debug_level = os.environ.get("DEBUG_LOG_LEVEL", "summary").strip().lower()
    if debug_level not in {"summary", "full"}:
        raise ValueError(f"invalid DEBUG_LOG_LEVEL: {debug_level}")
    os.environ["DEBUG_LOG_LEVEL"] = debug_level
    os.environ["MAX_DEBUG_LOG_BYTES"] = os.environ.get("MAX_DEBUG_LOG_BYTES", "5000000")
    os.environ["MAX_JSONL_ROWS"] = os.environ.get("MAX_JSONL_ROWS", "50000")
    os.environ["LOG_ROTATE_BYTES"] = os.environ.get("LOG_ROTATE_BYTES", "5000000")
    os.environ["WEAPON_STATE_DEBUG"] = "1" if debug_level == "full" else "0"
    os.environ["WEAPON_STATE_DEBUG_PATH"] = str(weapon_log)
    os.environ["SCRIPTED_SHOOT_DAMAGE_SCALE"] = str(cfg.scripted_shoot_damage_scale)
    os.environ["SCRIPTED_SHOOT_COOLDOWN_MULT"] = str(cfg.scripted_shoot_cooldown_mult)
    os.environ["SCRIPTED_SHOOT_AIM_THRESHOLD_DEG"] = str(cfg.scripted_shoot_aim_threshold_deg)
    if cfg.scripted_shoot_warmup_iters > 0:
        os.environ["SCRIPTED_SHOOT_WARMUP_ITERS"] = str(cfg.scripted_shoot_warmup_iters)
        os.environ["SCRIPTED_SHOOT_WARMUP_SEC_PER_ITER"] = str(cfg.scripted_shoot_warmup_sec_per_iter)
    else:
        os.environ.pop("SCRIPTED_SHOOT_WARMUP_ITERS", None)
        os.environ.pop("SCRIPTED_SHOOT_WARMUP_SEC_PER_ITER", None)
    return weapon_log


def launch_unity_env(cfg: RuntimeConfig):
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=cfg.time_scale)
    env = UnityEnvironment(
        file_name=cfg.env_path,
        worker_id=1,
        base_port=cfg.base_port,
        seed=cfg.seed,
        side_channels=[channel],
        no_graphics=True,
    )
    env.reset()
    return env


def _load_actor_critic_checkpoint(policy, checkpoint: Path, device: torch.device) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("actor_critic_state", payload.get("model_state_dict", payload))
    policy.load_state_dict(state, strict=True)


def load_policy_and_normalizer(cfg: RuntimeConfig):
    ppo_cfg = PPOTrainConfig.from_yaml(cfg.ppo_config)
    distill_cfg = DistillConfig.from_yaml(cfg.distill_config)
    ppo_cfg.device = cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
    ppo_cfg.env_path = cfg.env_path
    ppo_cfg.rollout_steps = cfg.rollout_steps
    ppo_cfg.rollout_smoke_steps = cfg.rollout_steps
    ppo_cfg.seq_len = cfg.seq_len
    ppo_cfg.batch_size = cfg.batch_size
    ppo_cfg.ppo_clip = cfg.ppo_clip
    ppo_cfg.gamma = cfg.gamma
    ppo_cfg.gae_lambda = cfg.gae_lambda
    ppo_cfg.entropy_coef = cfg.entropy_coef
    ppo_cfg.value_coef = cfg.value_coef
    ppo_cfg.lr = cfg.lr
    ppo_cfg.max_grad_norm = cfg.max_grad_norm
    ppo_cfg.enable_ppo = False
    device = torch.device(ppo_cfg.device)
    policy_init_source = "recurrent_distilled_checkpoint"
    try:
        policy, policy_checkpoint = load_recurrent_policy(distill_cfg, device)
    except FileNotFoundError:
        phase1 = Path(distill_cfg.phase1_ckpt)
        if not phase1.exists():
            fallback = PROJECT_ROOT / "experiments/checkpoints/large/best_checkpoint.pt"
            if fallback.exists():
                phase1 = fallback
        if not phase1.exists():
            raise
        encoder = TelemetryEncoder.from_encoder_checkpoint(str(phase1), device=device)
        policy = RecurrentActorCritic.from_distilled(CoreActionProjector(encoder))
        policy_checkpoint = phase1
        policy_init_source = "phase1_encoder_random_policy_heads"
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        _load_actor_critic_checkpoint(policy, resume_path, device)
        policy_checkpoint = resume_path
        policy_init_source = f"resume_from:{resume_path}"
    policy.to(device)
    policy.policy_init_source = policy_init_source
    if hasattr(policy, "actor_cont_logstd"):
        with torch.no_grad():
            policy.actor_cont_logstd.clamp_(cfg.min_log_std, cfg.max_log_std)
    policy.set_ppo_mode(training=False)
    normalizer = CertifiedObsNormalizer(cfg.normalizer_path, device)
    return policy, normalizer, ppo_cfg, distill_cfg, policy_checkpoint, device


def validate_specs(env) -> tuple[str, Any, dict[str, Any]]:
    behavior_name = next(iter(env.behavior_specs))
    spec = env.behavior_specs[behavior_name]
    obs_shapes = [list(item.shape) for item in spec.observation_specs]
    continuous = int(spec.action_spec.continuous_size)
    branches = [int(x) for x in spec.action_spec.discrete_branches]
    action_dim = continuous + len(branches)
    if obs_shapes != [[OBS_DIM]]:
        raise RuntimeError(f"runtime observation spec mismatch: {obs_shapes}")
    if continuous != 4 or branches[:4] != [2, 2, 2, 2] or action_dim != ACTION_DIM:
        raise RuntimeError(f"runtime action spec mismatch: continuous={continuous} branches={branches}")
    return behavior_name, spec.action_spec, {
        "obs_shapes": obs_shapes,
        "continuous_size": continuous,
        "discrete_branches": branches,
        "action_dim": action_dim,
    }


def get_first_decision(env, behavior_name: str, max_steps: int = 200) -> tuple[int, np.ndarray, float]:
    for _ in range(max_steps):
        decision_steps, _ = env.get_steps(behavior_name)
        if len(decision_steps):
            obs_batch = np.asarray(decision_steps.obs[0], dtype=np.float32)
            agent_id = int(decision_steps.agent_id[0])
            obs = obs_batch[0]
            if obs.shape != (OBS_DIM,):
                raise RuntimeError(f"decision obs shape {obs.shape} != {(OBS_DIM,)}")
            if not np.isfinite(obs).all():
                raise RuntimeError("decision obs contains NaN/Inf")
            return agent_id, obs, float(decision_steps.reward[0])
        t0 = time.monotonic()
        env.step()
        if time.monotonic() - t0 > 30.0:
            raise RuntimeError("Unity step exceeded deadlock threshold while waiting for decision")
    raise RuntimeError("no decision received from Unity")


def action_to_unity(env, behavior_name: str, action_spec: Any, agent_id: int, action: np.ndarray) -> None:
    if action.shape != (ACTION_DIM,):
        raise RuntimeError(f"action shape {action.shape} != {(ACTION_DIM,)}")
    if not np.isfinite(action).all():
        raise RuntimeError("action contains NaN/Inf")
    if np.any(action[:4] < -1.0) or np.any(action[:4] > 1.0):
        raise RuntimeError(f"continuous action out of range: {action[:4].tolist()}")
    if not np.all((action[4:] == 0.0) | (action[4:] == 1.0)):
        raise RuntimeError(f"binary action not 0/1: {action[4:].tolist()}")
    continuous, discrete = policy_action_to_buffers(action, action_spec)
    env.set_action_for_agent(behavior_name, agent_id, make_action_tuple(continuous, discrete))


def collect_all_weapon_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"event": "json_decode_error", "raw": line})
    return events


def summarize_weapon_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    blocked = Counter()
    hard = Counter()
    owner_mismatches = 0
    for event in events:
        if event.get("event") != "blocked_fire":
            continue
        reason = str(event.get("blocked_fire_reason", "UNKNOWN"))
        blocked[reason] += 1
        if reason in HARD_BLOCK_REASONS:
            hard[reason] += 1
        if reason == "WRONG_OWNER":
            owner_mismatches += 1
    return {
        "blocked_fire_count_by_reason": dict(blocked),
        "hard_blocked_fire_count_by_reason": dict(hard),
        "owner_mismatch_count": owner_mismatches,
    }


def build_reward_builder(cfg: RuntimeConfig | None = None) -> RewardBuilder:
    cfg = cfg or RuntimeConfig()
    return RewardBuilder(
        dmg_dealt_idx=None,
        dmg_taken_idx=None,
        env_reward_coef=1.0,
        damage_dealt_coef=0.0,
        damage_taken_coef=0.0,
        kill_coef=0.0,
        death_coef=0.0,
        miss_coef=0.0,
        jerk_coef=cfg.jerk_coef,
        spam_coef=cfg.spam_coef,
        aim_coef=cfg.aim_coef,
        aim_err_max=cfg.aim_err_max,
        aim_reward_mode=cfg.aim_reward_mode,
        shoot_bootstrap_coef=cfg.shoot_bootstrap_coef,
        shoot_bootstrap_aim_threshold_deg=cfg.shoot_bootstrap_aim_threshold_deg,
        shoot_bootstrap_require_fired=cfg.shoot_bootstrap_require_fired,
        shot_fired_obs_index=44,
        damage_chain_coef=0.0,
        hp_delta_reward_coef=0.0,
    )


def manifest(cfg: RuntimeConfig, command: list[str], policy_checkpoint: Path | None = None) -> dict[str, Any]:
    repo_commit = run_text(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT) or os.environ.get("REPO_COMMIT", "unavailable")
    return {
        "created_utc": utc_now(),
        "command": command,
        "runtime_config": asdict(cfg),
        "repository": str(PROJECT_ROOT),
        "repository_git_commit": repo_commit,
        "unity_executable_sha256": sha256_file(Path(cfg.env_path)),
        "normalizer_sha256": sha256_file(Path(cfg.normalizer_path)),
        "policy_checkpoint": str(policy_checkpoint) if policy_checkpoint else None,
        "policy_checkpoint_sha256": sha256_file(policy_checkpoint) if policy_checkpoint else None,
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": run_text(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"]),
        },
    }


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN/Inf")


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    y = returns.detach().flatten()
    pred = values.detach().flatten()
    var_y = torch.var(y)
    if float(var_y.item()) <= 1e-8:
        return 0.0
    return float((1.0 - torch.var(y - pred) / (var_y + 1e-8)).item())
