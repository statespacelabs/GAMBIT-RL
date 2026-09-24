#!/usr/bin/env python3
"""Goal 9 fair-telemetry PPO model, reward, checkpoint, and selection primitives."""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.navigation_policy import (
    ACTOR_DIM,
    LOS_INDEX,
    NavigatorConfig,
    Phase5Navigator,
    load_checkpoint as load_navigator,
)

READY_PREREQUISITE = "PHASE5_DAGGER_NAVIGATOR_READY"
READY = "PHASE5_MAP_GENERAL_HUNTER_READY"
CHECKPOINT_SCHEMA = "phase5_map_general_hunter_ppo_checkpoint_v001"
CRITIC_DIM = 428
PRIVILEGED_OFFSET = 363
EXACT_DISTANCE_INDEX = 375


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class HunterBranch:
    branch_id: str
    seed: int
    reward_scale: float = 1.0
    beacon_dropout: float | None = None
    gru_hidden: int = 128
    tactical_lr: float = 1e-4
    bc_start: float = 0.20
    bc_floor: float = 0.05
    smoothness_coef: float = 0.0
    reaction_delay_steps: int = 0


BRANCHES: tuple[HunterBranch, ...] = (
    HunterBranch("gpu0_base_seed_a", 59001),
    HunterBranch("gpu1_base_seed_b", 59002),
    HunterBranch("gpu2_base_seed_c", 59003),
    HunterBranch("gpu3_reward_scale_075", 59004, reward_scale=0.75),
    HunterBranch("gpu4_reward_scale_125", 59005, reward_scale=1.25),
    HunterBranch("gpu5_beacon_dropout_035", 59006, beacon_dropout=0.35),
    HunterBranch("gpu6_gru256_continuation", 59007, gru_hidden=256),
    HunterBranch(
        "gpu7_low_lr_strong_bc", 59008, tactical_lr=3e-5,
        bc_start=0.35, bc_floor=0.10,
    ),
)


@dataclass(frozen=True)
class ReacquireVariant:
    branch: HunterBranch
    memory_steps: int
    cue_dropout: float
    stale_timeout_seconds: float
    reacquisition_reward: float
    route_change_penalty: float


REACQUIRE_VARIANTS: tuple[ReacquireVariant, ...] = (
    ReacquireVariant(HunterBranch("gpu0_reacq_base", 61001), 128, 0.25, 4.0, 0.06, 0.010),
    ReacquireVariant(HunterBranch("gpu1_memory64", 61002), 64, 0.25, 4.0, 0.06, 0.010),
    ReacquireVariant(HunterBranch("gpu2_memory256", 61003), 256, 0.25, 4.0, 0.06, 0.010),
    ReacquireVariant(HunterBranch("gpu3_dropout050", 61004), 128, 0.50, 4.0, 0.06, 0.010),
    ReacquireVariant(HunterBranch("gpu4_stale2500ms", 61005), 128, 0.25, 2.5, 0.06, 0.010),
    ReacquireVariant(HunterBranch("gpu5_gru256", 61006, gru_hidden=256), 192, 0.25, 4.0, 0.06, 0.010),
    ReacquireVariant(HunterBranch("gpu6_reacq_reward010", 61007), 128, 0.25, 4.0, 0.10, 0.010),
    ReacquireVariant(
        HunterBranch("gpu7_route_penalty030", 61008, tactical_lr=3e-5, bc_start=0.30, bc_floor=0.10),
        128, 0.35, 3.5, 0.08, 0.030,
    ),
)


@dataclass(frozen=True)
class PeekerVariant:
    branch: HunterBranch
    dwell_min_steps: int
    dwell_max_steps: int
    risk: float
    repeek_penalty: float


PEEKER_VARIANTS: tuple[PeekerVariant, ...] = (
    PeekerVariant(HunterBranch("gpu0_peek_base_a", 62001), 12, 28, 0.50, 0.020),
    PeekerVariant(HunterBranch("gpu1_peek_base_b", 62002), 12, 28, 0.50, 0.020),
    PeekerVariant(HunterBranch("gpu2_short_dwell", 62003), 7, 18, 0.65, 0.020),
    PeekerVariant(HunterBranch("gpu3_long_dwell", 62004), 22, 45, 0.40, 0.020),
    PeekerVariant(HunterBranch("gpu4_aggressive_risk", 62005), 10, 25, 0.80, 0.020),
    PeekerVariant(HunterBranch("gpu5_conservative_risk", 62006), 16, 34, 0.25, 0.020),
    PeekerVariant(HunterBranch("gpu6_strong_repeek_penalty", 62007), 12, 28, 0.50, 0.045),
    PeekerVariant(
        HunterBranch("gpu7_gru256_repeek", 62008, gru_hidden=256, tactical_lr=3e-5,
                     bc_start=0.30, bc_floor=0.10),
        14, 32, 0.45, 0.035,
    ),
)


@dataclass(frozen=True)
class LeagueLane:
    gpu: int
    lane_id: str
    role: str
    branch: HunterBranch


LEAGUE_LANES: tuple[LeagueLane, ...] = (
    LeagueLane(0, "main_A", "main", HunterBranch("league_main_A", 63001)),
    LeagueLane(1, "main_B", "main", HunterBranch("league_main_B", 63002, tactical_lr=8e-5)),
    LeagueLane(2, "main_C", "main_lower_lr", HunterBranch(
        "league_main_C_lower_lr", 63003, tactical_lr=3e-5, bc_start=0.25,
    )),
    LeagueLane(3, "anti_camp_exploiter", "anti_camp", HunterBranch(
        "league_anti_camp", 63004, reward_scale=1.20,
    )),
    LeagueLane(4, "anti_repeek_exploiter", "anti_repeek", HunterBranch(
        "league_anti_repeek", 63005, bc_start=0.28, bc_floor=0.08,
    )),
    LeagueLane(5, "kiter_escape_exploiter", "kiter_escape", HunterBranch(
        "league_kiter_escape", 63006, reward_scale=1.10,
    )),
    LeagueLane(6, "aggressive_rusher_exploiter", "aggressive_rusher", HunterBranch(
        "league_aggressive_rusher", 63007, reward_scale=1.25,
    )),
    LeagueLane(7, "humanlike_generalist", "humanlike", HunterBranch(
        "league_humanlike_generalist", 63008, tactical_lr=5e-5,
        bc_start=0.35, bc_floor=0.15, smoothness_coef=0.08,
        reaction_delay_steps=2,
    )),
)


def branch_by_id(branch_id: str) -> HunterBranch:
    for branch in (
        *BRANCHES,
        *(variant.branch for variant in REACQUIRE_VARIANTS),
        *(variant.branch for variant in PEEKER_VARIANTS),
        *(lane.branch for lane in LEAGUE_LANES),
    ):
        if branch.branch_id == branch_id:
            return branch
    raise KeyError(branch_id)


def reacquire_variant_by_id(branch_id: str) -> ReacquireVariant:
    for variant in REACQUIRE_VARIANTS:
        if variant.branch.branch_id == branch_id:
            return variant
    raise KeyError(branch_id)


def peeker_variant_by_id(branch_id: str) -> PeekerVariant:
    for variant in PEEKER_VARIANTS:
        if variant.branch.branch_id == branch_id:
            return variant
    raise KeyError(branch_id)


def league_lane_by_branch_id(branch_id: str) -> LeagueLane:
    for lane in LEAGUE_LANES:
        if lane.branch.branch_id == branch_id:
            return lane
    raise KeyError(branch_id)


class PrivilegedValue(nn.Module):
    """Critic-only network; its input is never passed to the actor."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CRITIC_DIM, 192), nn.LayerNorm(192), nn.SiLU(),
            nn.Linear(192, 128), nn.SiLU(), nn.Linear(128, 1),
        )

    def forward(self, critic_observation: torch.Tensor) -> torch.Tensor:
        if critic_observation.shape[-1] != CRITIC_DIM:
            raise ValueError("critic input must be phase5_privileged_critic_obs_v001 width 428")
        return self.net(critic_observation).squeeze(-1)


class HunterActorCritic(nn.Module):
    def __init__(self, navigator: Phase5Navigator, branch: HunterBranch):
        super().__init__()
        self.navigator = navigator
        self.critic = PrivilegedValue()
        self.branch = branch

    def actor(
        self, actor_observation: torch.Tensor, hidden: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        prediction, next_hidden = self.navigator(actor_observation, hidden)
        if self.branch.reaction_delay_steps > 0:
            previous_action = actor_observation[..., 16:20]
            reaction_alpha = 1.0 / float(self.branch.reaction_delay_steps + 1)
            prediction = dict(prediction)
            prediction["mean"] = previous_action + reaction_alpha * (
                prediction["mean"] - previous_action
            )
        return prediction, next_hidden

    def value(self, critic_observation: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_observation)


def _copy_overlap(target: torch.Tensor, source: torch.Tensor) -> None:
    slices = tuple(slice(0, min(a, b)) for a, b in zip(target.shape, source.shape))
    target[slices].copy_(source[slices])


def continue_from_goal7(
    navigator_path: str | Path,
    branch: HunterBranch,
    device: torch.device | str,
) -> HunterActorCritic:
    parent, payload, _ = load_navigator(navigator_path, device)
    parent_config = parent.config
    config = replace(
        parent_config,
        branch_id=branch.branch_id,
        seed=branch.seed,
        gru_hidden=branch.gru_hidden,
        hunt_dropout=(
            branch.beacon_dropout
            if branch.beacon_dropout is not None
            else parent_config.hunt_dropout
        ),
    )
    navigator = Phase5Navigator(config).to(device)
    with torch.no_grad():
        source_state = parent.state_dict()
        target_state = navigator.state_dict()
        for name, target in target_state.items():
            source = source_state.get(name)
            if source is not None:
                _copy_overlap(target, source)
        navigator.load_state_dict(target_state, strict=True)
    model = HunterActorCritic(navigator, branch).to(device)
    model.navigator.train()
    model.critic.train()
    return model


def assert_actor_critic_separation(model: HunterActorCritic) -> None:
    actor_width = model.navigator.context_encoder[0].in_features + (
        model.navigator.ray_encoder[0].in_features
    )
    if actor_width != ACTOR_DIM:
        raise RuntimeError("actor width changed or privileged fields entered actor encoder")
    if model.critic.net[0].in_features != CRITIC_DIM:
        raise RuntimeError("critic width mismatch")


@dataclass
class RewardLedger:
    shaping_abs: float = 0.0
    first_contact: bool = False
    los_previous: bool = False
    stuck_run: int = 0
    stuck_latched: bool = False
    collision_run: int = 0
    prior_distance_m: float | None = None
    prior_action: np.ndarray | None = None

    def transition(
        self,
        actor: np.ndarray,
        critic: np.ndarray,
        action: np.ndarray,
        reward_scale: float,
        route_change_penalty: float = 0.005,
    ) -> tuple[float, dict[str, float]]:
        if actor.shape != (ACTOR_DIM,) or critic.shape != (CRITIC_DIM,):
            raise ValueError("reward requires actor231 and critic428")
        if not np.isfinite(actor).all() or not np.isfinite(critic).all():
            raise RuntimeError("non-finite reward input")
        distance_m = float(critic[EXACT_DISTANCE_INDEX] * 100.0)
        components = {
            "progress": 0.0, "contact": 0.0, "reacquisition": 0.0,
            "stuck": 0.0, "collision_loop": 0.0, "oscillation": 0.0,
        }
        if self.prior_distance_m is not None:
            components["progress"] = float(np.clip(
                (self.prior_distance_m - distance_m) * 0.002 * reward_scale,
                -0.01, 0.01,
            ))
        los = bool(actor[LOS_INDEX] > 0.5)
        if los and not self.los_previous:
            if self.first_contact:
                components["reacquisition"] = 0.03
            else:
                components["contact"] = 0.05
                self.first_contact = True
        self.stuck_run = self.stuck_run + 1 if actor[26] > 0.60 else 0
        if self.stuck_run >= 100 and not self.stuck_latched:
            components["stuck"] = -0.05
            self.stuck_latched = True
        if self.stuck_run == 0:
            self.stuck_latched = False
        collision = actor[28] > 0.55 and actor[26] > 0.35
        self.collision_run = self.collision_run + 1 if collision else 0
        if self.collision_run and self.collision_run % 50 == 0:
            components["collision_loop"] = -0.01
        if self.prior_action is not None:
            flip = np.sign(action[:2]) * np.sign(self.prior_action[:2]) < 0
            if int(flip.sum()) >= 2 and float(np.linalg.norm(action[:2])) > 0.6:
                components["oscillation"] = -float(np.clip(route_change_penalty, 0.0, 0.05))
        proposed = sum(components.values())
        remaining = max(0.0, 0.20 - self.shaping_abs)
        applied = float(np.clip(proposed, -remaining, remaining))
        self.shaping_abs += abs(applied)
        self.prior_distance_m = distance_m
        self.prior_action = np.asarray(action, dtype=np.float32).copy()
        self.los_previous = los
        return applied, components

    def terminal(self, outcome: str, reacquisition_reward: float = 0.03) -> float:
        if outcome == "kill":
            return 1.0
        if outcome == "death":
            return -1.0
        if outcome == "timeout":
            return -0.75
        if outcome == "reacquired":
            return float(np.clip(reacquisition_reward, 0.0, 0.20))
        raise ValueError(outcome)


def normal_log_prob(
    action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor,
) -> torch.Tensor:
    variance = torch.exp(2.0 * log_std)
    return (-0.5 * (
        (action - mean).square() / variance + 2.0 * log_std
        + math.log(2.0 * math.pi)
    )).sum(-1)


def conservative_ppo_loss(
    model: HunterActorCritic,
    batch: dict[str, torch.Tensor],
    *,
    bc_weight: float,
    clip: float = 0.10,
) -> dict[str, torch.Tensor]:
    actor = batch["actor_obs"].float().unsqueeze(1)
    hidden = batch["nav_hidden"].float().unsqueeze(0)
    prediction, _ = model.actor(actor, hidden)
    mean = prediction["mean"][:, 0]
    log_std = prediction["log_std"][:, 0]
    action = batch["policy_action"].float()
    old_log_prob = normal_log_prob(
        action, batch["old_mean"].float(), batch["old_log_std"].float(),
    ).detach()
    new_log_prob = normal_log_prob(action, mean, log_std)
    values = model.value(batch["critic_obs"].float())
    reward = batch["reward"].float()
    advantage = (reward - values.detach())
    valid = (
        (batch["visible"].float() < 0.5)
        & (batch["intervened"].float() < 0.5)
    )
    if valid.any():
        normalized = advantage[valid]
        normalized = (normalized - normalized.mean()) / (normalized.std(unbiased=False) + 1e-5)
        ratio = torch.exp(new_log_prob[valid] - old_log_prob[valid])
        surrogate = torch.min(
            ratio * normalized,
            ratio.clamp(1.0 - clip, 1.0 + clip) * normalized,
        )
        policy_loss = -surrogate.mean()
        approx_kl = (old_log_prob[valid] - new_log_prob[valid]).mean().abs()
    else:
        policy_loss = mean.new_zeros(())
        approx_kl = mean.new_zeros(())
    value_loss = 0.5 * F.mse_loss(values, reward)
    bc_loss = F.smooth_l1_loss(mean, batch["bc_target"].float())
    anchor = F.mse_loss(mean, batch["old_mean"].float())
    tactical = F.cross_entropy(
        prediction["mode_logits"][:, 0], batch["old_mode"].long(),
    )
    previous_action = actor[:, 0, 16:20]
    smoothness = F.smooth_l1_loss(mean, previous_action)
    total = (
        policy_loss + 0.5 * value_loss + bc_weight * bc_loss
        + 0.05 * anchor + 0.02 * tactical
        + model.branch.smoothness_coef * smoothness
    )
    return {
        "total": total,
        "policy": policy_loss,
        "value": value_loss,
        "bc": bc_loss,
        "anchor": anchor,
        "tactical": tactical,
        "smoothness": smoothness,
        "approx_kl": approx_kl,
        "valid_fraction": valid.float().mean(),
    }


def trainable_parameters(model: HunterActorCritic) -> Iterable[nn.Parameter]:
    return model.parameters()


def save_checkpoint(
    path: str | Path,
    model: HunterActorCritic,
    optimizer: torch.optim.Optimizer,
    environment_steps: int,
    metadata: dict[str, Any],
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "actor_schema": "phase5_actor_obs_v001",
        "critic_schema": "phase5_privileged_critic_obs_v001",
        "actor_dim": ACTOR_DIM,
        "critic_dim": CRITIC_DIM,
        "branch": asdict(model.branch),
        "navigator_config": asdict(model.navigator.config),
        "navigator_state_dict": model.navigator.state_dict(),
        "critic_state_dict": model.critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "environment_steps": int(environment_steps),
        "metadata": metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: torch.device | str,
    *,
    load_optimizer: bool = False,
) -> tuple[HunterActorCritic, dict[str, Any], torch.optim.Optimizer | None]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise RuntimeError("Goal 9 checkpoint schema mismatch")
    branch = HunterBranch(**payload["branch"])
    navigator = Phase5Navigator(NavigatorConfig(**payload["navigator_config"])).to(device)
    navigator.load_state_dict(payload["navigator_state_dict"], strict=True)
    model = HunterActorCritic(navigator, branch).to(device)
    model.critic.load_state_dict(payload["critic_state_dict"], strict=True)
    optimizer = None
    if load_optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=branch.tactical_lr, weight_decay=1e-4)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    assert_actor_critic_separation(model)
    return model, payload, optimizer


def rank_for_halving(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = list(records)
    for record in records:
        if any("heldout" in str(key).lower() for key in record):
            raise RuntimeError("held-out metrics cannot drive successive halving")
    return sorted(records, key=lambda row: (
        -float(row["validation_contact_rate"]),
        float(row["validation_timeout_rate"]),
        float(row["validation_stuck_rate"]),
        float(row["mean_approx_kl"]),
        str(row["candidate_id"]),
    ))
