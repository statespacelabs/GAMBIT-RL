"""Phase 2 configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import yaml


@dataclass
class Phase2CacheConfig:
    """Configuration for encoder latent caching.

    Fields define encoder checkpoint path, window manifest path, output latent
    directory, image size, sequence length, batch size, workers, device, and AMP.
    """

    encoder_checkpoint_path: str = ""
    window_manifest_path: str = ""
    output_latent_dir: str = "data/phase2/latents"
    image_size: tuple[int, int] = (160, 240)
    seq_len: int = 150
    fps: float = 30.0
    action_vocab_size: int = 14
    batch_size: int = 128
    num_workers: int = 8
    device: str = "cuda"
    amp: bool = True
    telemetry_stats_path: str = ""


@dataclass
class Phase2DatasetConfig:
    """Configuration for building action, reward, and transition datasets.

    Fields define window size, stride, action schema, reward config, manifest
    paths, scaler paths, and output directories.
    """

    source_manifest_path: str = ""
    window_manifest_path: str = "data/phase2/manifests/phase2_window_manifest.csv"
    transition_manifest_path: str = (
        "data/phase2/manifests/phase2_transition_manifest.csv"
    )
    latent_index_path: str = ""

    window_seconds: float = 5.0
    stride_seconds: float = 1.0
    step_seconds: float = 1.0
    fps: float = 30.0
    seq_len: int = 150

    action_dir: str = "data/phase2/actions"
    action_scaler_path: str = "data/phase2/scalers/action_scaler.npz"
    state_scaler_path: str = "data/phase2/scalers/state_scaler.npz"
    reward_stats_path: str = "data/phase2/rewards/reward_stats.json"


@dataclass
class BCTrainConfig:
    """Configuration for behavior cloning training.

    Includes transition manifest path, state/action dimensions, batch size,
    optimizer settings, continuous/binary action indices, checkpoint directory,
    logging settings, and training duration.
    """

    transition_manifest_path: str = ""

    state_dim: int = 512
    action_dim: int = 14
    hidden_dim: int = 1024
    num_hidden_layers: int = 3
    dropout: float = 0.1

    continuous_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    binary_indices: tuple[int, ...] = (8, 9, 10, 11)
    count_indices: tuple[int, ...] = (7, 12, 13)

    normalize_state: bool = False
    state_scaler_path: str = ""
    normalize_action: bool = False
    action_scaler_path: str = ""
    normalize_reward: bool = False
    reward_stats_path: str = ""

    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: Optional[float] = 1.0
    binary_loss_weight: float = 1.0

    epochs: int = 100
    steps_per_epoch: int = 1000
    val_every_epochs: int = 1
    patience: int = 10

    checkpoint_dir: str = "experiments/phase2/bc/checkpoints"
    log_dir: str = "experiments/phase2/bc/logs"

    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "BCTrainConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # Convert list fields to tuples
        for key in ("continuous_indices", "binary_indices", "count_indices"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        if "image_size" in data and isinstance(data["image_size"], list):
            data["image_size"] = tuple(data["image_size"])
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IQLTrainConfig:
    """Configuration for IQL training.

    Includes transition manifest path, state/action dimensions, discount,
    expectile, temperature, max advantage weight, optimizer settings, reward
    normalization, BC initialization path, checkpoint directory, and update count.
    """

    transition_manifest_path: str = ""

    state_dim: int = 512
    action_dim: int = 14
    hidden_dim: int = 1024
    num_hidden_layers: int = 3
    dropout: float = 0.1

    continuous_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    binary_indices: tuple[int, ...] = (8, 9, 10, 11)

    # IQL hyperparameters
    discount: float = 0.99
    expectile: float = 0.7
    temperature: float = 3.0
    max_adv_weight: float = 100.0
    target_update_rate: float = 0.005

    normalize_state: bool = False
    state_scaler_path: str = ""
    normalize_action: bool = False
    action_scaler_path: str = ""
    normalize_reward: bool = False
    reward_stats_path: str = ""

    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: Optional[float] = 1.0

    total_steps: int = 500_000
    val_every_steps: int = 5000
    save_every_steps: int = 10000
    log_every_steps: int = 100

    init_actor_from_bc: str = ""  # Path to BC checkpoint, empty = skip

    checkpoint_dir: str = "experiments/phase2/iql/checkpoints"
    log_dir: str = "experiments/phase2/iql/logs"

    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "IQLTrainConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        for key in ("continuous_indices", "binary_indices"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DistillConfig:
    """Configuration for Phase 3A distillation."""

    transition_manifest_path: str = ""
    window_manifest_path: str = ""
    action_scaler_path: str = ""
    phase1_ckpt: str = ""
    telemetry_fps: float = 30.0
    action_step_seconds: float = 1.0
    max_move_speed: float = 6.0
    max_yaw_delta: float = 180.0
    max_pitch_delta: float = 180.0
    require_normalized_telemetry: bool = True
    binary_pos_weight: Optional[list[float]] = None

    obs_dim: int = 45
    action_dim: int = 8
    hidden_dim: int = 256
    num_layers: int = 3

    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    warmup_epochs: int = 10
    grad_clip_norm: float = 1.0

    checkpoint_dir: str = "experiments/phase3/distill/checkpoints"
    log_dir: str = "experiments/phase3/distill/logs"

    # Recurrent BC warmup
    recurrent_epochs: int = 10
    # If True, always retrain both stages even if checkpoints already exist.
    force_retrain: bool = False

    num_workers: int = 4
    device: str = "cuda"

    # DDP: per-GPU batch and DataLoader workers (fall back to batch_size / num_workers)
    per_device_batch_size: Optional[int] = None
    recurrent_per_device_batch_size: Optional[int] = None
    num_workers_per_rank: Optional[int] = None
    max_batches: Optional[int] = None

    def resolved_batch_size(self) -> int:
        return (
            self.per_device_batch_size
            if self.per_device_batch_size is not None
            else self.batch_size
        )

    def resolved_recurrent_batch_size(self) -> int:
        if self.recurrent_per_device_batch_size is not None:
            return self.recurrent_per_device_batch_size
        return self.resolved_batch_size()

    def resolved_num_workers(self) -> int:
        return (
            self.num_workers_per_rank
            if self.num_workers_per_rank is not None
            else self.num_workers
        )

    @classmethod
    def from_yaml(cls, path: str) -> "DistillConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass
class PPOTrainConfig:
    """Configuration for Phase 3A PPO Self-Play."""

    env_path: str = ""
    enable_ppo: bool = False
    rollout_smoke_steps: int = 100
    freeze_pretrained_branches: bool = True
    max_preupdate_clip_frac: float = 0.001

    num_envs: int = 1
    time_scale: float = 20.0
    env_base_port: int = 5005
    env_seed: int = 42
    # Telemetry stats NPZ for ONLINE observation normalization (required for PPO).
    telemetry_stats_path: str = ""
    # Test mode (Part C): force an episode boundary every N steps to exercise the
    # per-env recurrent hidden-state reset path. None/0 = disabled (normal runs).
    force_done_every_n: Optional[int] = None
    # EXPERIMENTAL: ThreadPool env stepping. Default False (serial). Measured to
    # REGRESS throughput (mlagents step is GIL-bound) — do not enable for real runs.
    use_threadpool_step: bool = False
    # Multi-area mode: run N arenas inside ONE Unity process instead of N
    # separate processes. Requires a Unity build compiled with multi-area
    # GameModeBootstrapper support. When > 1, MultiAreaUnityRunner is used
    # instead of VectorizedUnityRunner. Default 0 = disabled (use num_envs
    # separate processes as before).
    num_unity_areas: int = 0
    # Post-normalization observation clipping. Multi-area arenas have spatial
    # offsets that produce huge raw position values; clipping prevents them
    # from destabilizing the policy. 0 = disabled.
    obs_clip_value: float = 10.0

    num_iterations: int = 1000
    rollout_steps: int = 4096
    seq_len: int = 32
    ppo_epochs: int = 4
    batch_size: int = 16  # seqs per minibatch
    ppo_clip: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    max_grad_norm: float = 1.0

    entropy_coef: float = 0.003
    value_coef: float = 0.5
    kl_anchor_coef: float = 0.05
    smooth_coef: float = 0.02

    damage_dealt_obs_index: Optional[int] = None
    damage_taken_obs_index: Optional[int] = None
    require_damage_signals: bool = False
    env_reward_coef: float = 1.0
    damage_dealt_coef: float = 1.0
    damage_taken_coef: float = 1.0
    kill_coef: float = 0.5
    death_coef: float = 0.5
    miss_coef: float = 0.05
    jerk_coef: float = 0.01
    spam_coef: float = 0.01
    aim_coef: float = 0.0
    aim_err_max: float = 45.0
    aim_reward_mode: str = "proximity_next"
    bad_aim_threshold_deg: float = 90.0
    bad_aim_coef: float = 0.0
    bad_aim_scale_deg: float = 90.0
    aim_sigma_deg: float = 60.0
    min_log_std: float = -3.0
    max_log_std: float = 2.0
    target_kl: Optional[float] = None
    resume_from: Optional[str] = None
    early_stop_on_collapse: bool = False
    collapse_aim_threshold: float = 140.0
    collapse_zero_hits_iters: int = 5

    # Shooting-head bootstrap reward (Phase 3S / 3T).
    shoot_bootstrap_coef: float = 0.0
    shoot_bootstrap_aim_threshold_deg: float = 30.0
    shoot_bootstrap_require_fired: bool = False
    shoot_bootstrap_require_los: bool = False
    shot_fired_obs_index: Optional[int] = 44

    # Sustained-damage chain bonus (Phase 3T).
    damage_chain_coef: float = 0.0
    damage_chain_window_sec: float = 3.0
    damage_chain_window_steps: int = 45
    learner_hit_env_reward_threshold: float = 0.05

    # HP-delta shaping (Phase 3U).
    hp_delta_reward_coef: float = 0.0
    hp_delta_reward_cap: float = 0.05
    opponent_hp_frac_obs_index: Optional[int] = 39
    hp_frac_to_damage_scale: float = 100.0

    # LR schedule + drift retention (Phase 3U).
    lr_schedule: str = "none"
    lr_initial: float = 3.0e-5
    lr_milestone_1_iter: int = 50
    lr_milestone_1_value: float = 1.5e-5
    lr_milestone_2_iter: int = 100
    lr_milestone_2_value: float = 7.5e-6
    lr_milestone_3_iter: int = 160
    lr_milestone_3_value: float = 3.0e-6

    early_stop_on_drift: bool = False
    drift_window_iters: int = 20
    drift_aim_threshold: float = 90.0
    drift_min_hits: float = 3.0
    drift_start_iter: int = 60
    drift_patience_after_best: int = 40
    retention_min_iter: int = 50
    retention_history_len: int = 20

    # Shot geometry debug (Phase 3U / 3V).
    shot_debug_enabled: bool = False
    shot_debug_sample_limit_per_area: int = 20
    shot_debug_true_unity: bool = False

    # Phase 3Y learner shoot geometry curriculum/debug.
    target_hurtbox_inflate: float = 0.0
    learner_shoot_ray_mode: str = "current"

    # Phase 3AA partial ray correction toward target chest.
    ray_correction_alpha: float = 0.0
    ray_correction_require_los: bool = True
    ray_correction_max_angle_deg: float = 30.0

    # Phase 3AB preservation / eval-only validation.
    eval_only: bool = False
    preserve_stop_enabled: bool = False
    preserve_stop_start_iter: int = 5
    preserve_stop_aim_threshold_deg: float = 75.0
    preserve_stop_min_hits_fraction_of_iter1: float = 0.50
    preserve_stop_zero_hits_iters: int = 3
    write_ray_vs_hurtbox_summary: bool = False
    log_per_agent_metrics: bool = False
    gvg_smoke: bool = False
    agent_a_checkpoint: str = ""
    agent_b_checkpoint: str = ""

    # Phase 3AC weapon / reset state isolation.
    log_weapon_state_debug: bool = False
    write_compact_metrics: bool = True
    hard_restart_unity_each_iteration: bool = False
    force_match_reset_on_episode_begin: bool = False
    force_weapon_reset_on_episode_begin: bool = False
    force_weapon_reset_on_respawn: bool = False
    debug_infinite_ammo: bool = False
    debug_disable_reload: bool = False
    debug_force_can_fire_if_cooldown_ready: bool = False
    deterministic_policy_actions: bool = False
    fixed_action_seed_per_iteration: bool = False
    gvg_zero_shot_streak_stop: int = 10
    gvg_both_zero_hits_streak_stop: int = 20
    gvg_min_total_shots_per_iter: float = 0.0
    gvg_min_total_hits_window: float = 0.0
    gvg_stop_if_either_agent_no_action_iters: int = 0
    gvg_aim_drift_stop_deg: float = 120.0
    gvg_aim_drift_start_iter: int = 10
    save_best_gvg_behavioral: bool = False
    save_best_gvg_retention: bool = False
    save_latest: bool = True
    resume_agent_a_from: str = ""
    resume_agent_b_from: str = ""
    repeat_processes: int = 0

    # Phase 3Z true ray-to-hurtbox aim metric.
    use_true_aim_for_reward: bool = False
    true_aim_err_max: float = 30.0
    true_aim_obs_index: int = 38
    shoot_bootstrap_metric: str = "old_aim"
    true_shoot_bootstrap_threshold_deg: float = 5.0

    # Visual encoder (Phase 3S-appendix).
    enable_visual: bool = False
    visual_encoder_ckpt: str = ""
    visual_frame_size: tuple[int, int] = (120, 160)
    visual_proj_dim: int = 512
    visual_freeze_backbone: bool = True
    visual_unfreeze_fusion_at_iter: int = 0

    # Scripted opponent shooting pressure (Phase 3R shooter curriculum bridge).
    scripted_shoot_damage_scale: float = 1.0
    scripted_shoot_cooldown_mult: float = 1.0
    scripted_shoot_aim_threshold_deg: float = 15.0
    scripted_shoot_warmup_iters: int = 0
    scripted_shoot_warmup_sec_per_iter: float = 45.0

    eval_interval: int = 50
    promote_win_rate: float = 0.55

    registry_dir: str = "experiments/phase3/registry"
    log_dir: str = "experiments/phase3/ppo/logs"
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str) -> "PPOTrainConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
