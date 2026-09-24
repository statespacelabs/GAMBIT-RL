#!/usr/bin/env python3
"""Model, data, and certification primitives for the Phase 5 navigator."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ACTOR_DIM = 231
LOCAL45_DIM = 45
ACTION_DIM = 8
MODE_COUNT = 7
LOS_INDEX = 193
RAY_SLICE = slice(30, 182)
HUNT_SLICE = slice(211, 227)
READY_PREREQUISITE = "PHASE5_GENERIC_DAGGER_DATA_READY"
READY = "PHASE5_DAGGER_NAVIGATOR_READY"
PARENT_ID = "p44g3_gvg_generalist_roster_mix_generalist_safe_from_s050_wna_u30"
PARENT_SHA256 = "4ce907c625f228546e6f65df3a55021a8afc7b86f366927eb00e06e61f517c4d"
LOCAL45_NORMALIZER_SHA256 = "28df2464221f65798ff4e4c98eb0d8d340f18b8765b4ca306fcbc45da4e89b5e"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class NavigatorConfig:
    branch_id: str
    seed: int
    gru_hidden: int = 128
    ray_width: int = 64
    auxiliary_weight: float = 0.20
    hunt_dropout: float = 0.15
    hunt_noise: float = 0.0
    hunt_quantization: float = 0.0
    log_std_init: float = -2.3


BRANCHES: tuple[NavigatorConfig, ...] = (
    NavigatorConfig("gpu0_base_gru128_seed_a", 57001),
    NavigatorConfig("gpu1_base_gru128_seed_b", 57002),
    NavigatorConfig("gpu2_base_gru128_seed_c", 57003),
    NavigatorConfig("gpu3_gru256", 57004, gru_hidden=256),
    NavigatorConfig("gpu4_stronger_auxiliary", 57005, auxiliary_weight=0.60),
    NavigatorConfig("gpu5_wider_ray_embedding", 57006, ray_width=128),
    NavigatorConfig("gpu6_higher_hunt_dropout", 57007, hunt_dropout=0.40),
    NavigatorConfig(
        "gpu7_lower_precision_noisy_hunt",
        57008,
        hunt_noise=0.08,
        hunt_quantization=0.125,
    ),
)


def branch_by_id(branch_id: str) -> NavigatorConfig:
    for config in BRANCHES:
        if config.branch_id == branch_id:
            return config
    raise KeyError(branch_id)


class Phase5Navigator(nn.Module):
    """Actor231 -> MLP256 -> GRU -> navigation/tactical/blend heads."""

    def __init__(self, config: NavigatorConfig):
        super().__init__()
        self.config = config
        self.training_augmentation_enabled = True
        ray_dim = RAY_SLICE.stop - RAY_SLICE.start
        context_dim = ACTOR_DIM - ray_dim
        self.ray_encoder = nn.Sequential(
            nn.Linear(ray_dim, config.ray_width),
            nn.SiLU(),
            nn.Linear(config.ray_width, 64),
            nn.SiLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 192),
            nn.SiLU(),
        )
        self.mlp256 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.gru = nn.GRU(256, config.gru_hidden, batch_first=True)
        self.move_head = nn.Linear(config.gru_hidden, 2)
        self.look_head = nn.Linear(config.gru_hidden, 2)
        self.mode_head = nn.Linear(config.gru_hidden, MODE_COUNT)
        self.blend_head = nn.Linear(config.gru_hidden, 1)
        self.corner_head = nn.Linear(config.gru_hidden, 3)
        self.log_std = nn.Parameter(torch.full((4,), config.log_std_init))

    def initial_hidden(self, batch: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(1, batch, self.config.gru_hidden, device=device)

    def _augment(self, observation: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.training_augmentation_enabled:
            return observation
        output = observation.clone()
        hunt = output[..., HUNT_SLICE]
        if self.config.hunt_dropout > 0:
            keep = torch.rand_like(hunt[..., :1]) >= self.config.hunt_dropout
            hunt = hunt * keep
        if self.config.hunt_noise > 0:
            hunt = hunt + torch.randn_like(hunt) * self.config.hunt_noise
        if self.config.hunt_quantization > 0:
            step = self.config.hunt_quantization
            hunt = torch.round(hunt / step) * step
        output[..., HUNT_SLICE] = hunt.clamp(-1.0, 1.0)
        return output

    def forward(
        self, observation: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if observation.ndim == 2:
            observation = observation.unsqueeze(1)
        if observation.shape[-1] != ACTOR_DIM:
            raise ValueError(f"expected actor231, got {tuple(observation.shape)}")
        observation = self._augment(observation)
        rays = observation[..., RAY_SLICE]
        context = torch.cat(
            (observation[..., : RAY_SLICE.start], observation[..., RAY_SLICE.stop :]),
            dim=-1,
        )
        encoded = self.mlp256(
            torch.cat((self.ray_encoder(rays), self.context_encoder(context)), dim=-1)
        )
        if hidden is None:
            hidden = self.initial_hidden(encoded.shape[0], encoded.device)
        recurrent, next_hidden = self.gru(encoded, hidden)
        move = torch.tanh(self.move_head(recurrent))
        look = torch.tanh(self.look_head(recurrent))
        output = {
            "mean": torch.cat((move, look), dim=-1),
            "log_std": self.log_std.clamp(-5.0, 0.0).expand_as(
                torch.cat((move, look), dim=-1)
            ),
            "mode_logits": self.mode_head(recurrent),
            "blend": torch.sigmoid(self.blend_head(recurrent)).squeeze(-1),
            "next_corner": torch.tanh(self.corner_head(recurrent)),
        }
        return output, next_hidden


def loss_terms(
    prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], config: NavigatorConfig
) -> dict[str, torch.Tensor]:
    target = torch.cat((batch["teacher_move"], batch["navigation_look_bias"]), dim=-1)
    navigation = F.smooth_l1_loss(prediction["mean"], target)
    mode = F.cross_entropy(
        prediction["mode_logits"].reshape(-1, MODE_COUNT),
        batch["teacher_tactical_mode"].reshape(-1).long(),
    )
    blend = F.binary_cross_entropy(prediction["blend"], batch["combat_navigation_blend_gate"])
    valid = batch["teacher_next_corner_valid"].bool()
    if valid.any():
        corner = F.smooth_l1_loss(
            prediction["next_corner"][valid],
            batch["teacher_next_corner_direction_local"][valid],
        )
    else:
        corner = navigation.new_zeros(())
    std_regularizer = prediction["log_std"].exp().mean() * 0.002
    total = navigation + 0.25 * mode + 0.20 * blend + config.auxiliary_weight * corner + std_regularizer
    return {
        "total": total,
        "navigation": navigation,
        "mode": mode,
        "blend": blend,
        "corner": corner,
    }


ARRAYS = (
    "actor_obs",
    "teacher_move",
    "navigation_look_bias",
    "teacher_tactical_mode",
    "combat_navigation_blend_gate",
    "teacher_next_corner_direction_local",
    "teacher_next_corner_valid",
    "area_id",
    "episode_id",
    "fixed_step",
)


def manifest_npz_paths(
    manifest_path: str | Path, *, validation: bool
) -> list[Path]:
    manifest_path = Path(manifest_path).resolve()
    document = json.loads(manifest_path.read_text())
    if document.get("status") != READY_PREREQUISITE:
        raise RuntimeError("Goal 6 prerequisite is not ready")
    root = Path(__file__).resolve().parent.parent
    paths: list[Path] = []
    for shard in document.get("shards", []):
        is_validation = shard.get("split") == "validation" or int(shard.get("gpu", -1)) == 7
        if is_validation != validation:
            continue
        for round_record in shard.get("rounds", []):
            path = root / round_record["npz"]["path"]
            if sha256_file(path) != round_record["npz"]["sha256"]:
                raise RuntimeError(f"dataset hash mismatch: {path}")
            paths.append(path)
    if not paths:
        raise RuntimeError("no matching dataset shards")
    return paths


def supplemental_npz_paths(
    supplemental_dir: str | Path, *, validation: bool
) -> list[Path]:
    root = Path(supplemental_dir).resolve()
    matrix = json.loads((root / "COLLECTION_RUN.json").read_text())
    if matrix.get("status") != "PASS" or matrix.get("gpu_count") != 8:
        raise RuntimeError("supplemental eight-GPU collection did not pass")
    paths: list[Path] = []
    for shard_dir in sorted(root.glob("gpu*")):
        result = json.loads((shard_dir / "shard_result.json").read_text())
        if result.get("status") != "PASS":
            raise RuntimeError(f"supplemental shard failed: {shard_dir.name}")
        shard = result["shard"]
        is_validation = shard.get("split") == "validation" or int(shard.get("gpu", -1)) == 7
        if is_validation != validation:
            continue
        for round_result in result.get("rounds", []):
            if round_result.get("status") != "PASS":
                raise RuntimeError(f"supplemental round failed: {shard_dir.name}")
            path = shard_dir / round_result["round_id"] / "teacher_rollout.npz"
            if sha256_file(path) != round_result["npz"]["sha256"]:
                raise RuntimeError(f"supplemental dataset hash mismatch: {path}")
            with np.load(path) as source:
                hidden = ~source["exact_los"].astype(bool)
                if hidden.any() and not (source["actor_obs"][hidden, 224] > 0.5).any():
                    raise RuntimeError(f"supplemental shard has no hidden hunt cues: {path}")
            paths.append(path)
    if not paths:
        raise RuntimeError("no matching supplemental shards")
    return paths


class SequenceCorpus:
    """Certified NPZ corpus sampled without crossing area/episode/step boundaries."""

    def __init__(self, paths: Iterable[Path], sequence_length: int = 32):
        self.sequence_length = sequence_length
        self.segments: list[dict[str, np.ndarray]] = []
        self.starts: list[tuple[int, int]] = []
        self.raw_sample_count = 0
        for path in paths:
            with np.load(path) as source:
                raw = {name: np.asarray(source[name]) for name in ARRAYS}
            count = len(raw["actor_obs"])
            self.raw_sample_count += count
            if raw["actor_obs"].shape != (count, ACTOR_DIM):
                raise RuntimeError(f"actor schema mismatch: {path}")
            if not all(np.isfinite(raw[name]).all() for name in ARRAYS):
                raise RuntimeError(f"non-finite training data: {path}")
            # Rows from parallel areas are interleaved. Regroup first, then split
            # again at every fixed-step gap so recurrent windows never leak state.
            keys = np.stack((raw["area_id"], raw["episode_id"]), axis=1)
            for area_id, episode_id in np.unique(keys, axis=0):
                indices = np.flatnonzero(
                    (raw["area_id"] == area_id)
                    & (raw["episode_id"] == episode_id)
                )
                boundaries = np.flatnonzero(np.diff(raw["fixed_step"][indices]) != 1) + 1
                for run in np.split(indices, boundaries):
                    if len(run) < sequence_length:
                        continue
                    segment = {name: raw[name][run] for name in ARRAYS}
                    segment_id = len(self.segments)
                    self.segments.append(segment)
                    for start in range(0, len(run) - sequence_length + 1):
                        self.starts.append((segment_id, start))
        if not self.starts:
            raise RuntimeError("corpus contains no contiguous sequences")

    @property
    def sample_count(self) -> int:
        return self.raw_sample_count

    def sample(self, batch_size: int, rng: random.Random, device: torch.device) -> dict[str, torch.Tensor]:
        selected = [self.starts[rng.randrange(len(self.starts))] for _ in range(batch_size)]
        result: dict[str, list[np.ndarray]] = {name: [] for name in ARRAYS[:-3]}
        for segment_id, start in selected:
            segment = self.segments[segment_id]
            end = start + self.sequence_length
            for name in result:
                result[name].append(segment[name][start:end])
        return {
            name: torch.as_tensor(np.stack(values), device=device)
            for name, values in result.items()
        }


@torch.no_grad()
def evaluate_corpus(
    model: Phase5Navigator,
    corpus: SequenceCorpus,
    device: torch.device,
    batches: int = 32,
    batch_size: int = 64,
    seed: int = 57999,
) -> dict[str, float]:
    prior = model.training
    model.eval()
    rng = random.Random(seed)
    totals: dict[str, float] = {key: 0.0 for key in ("total", "navigation", "mode", "blend", "corner")}
    mode_correct = 0
    mode_count = 0
    mu_saturated = 0
    mu_count = 0
    for _ in range(batches):
        batch = corpus.sample(batch_size, rng, device)
        prediction, _ = model(batch["actor_obs"].float())
        terms = loss_terms(prediction, batch, model.config)
        for key in totals:
            totals[key] += float(terms[key])
        mode_correct += int(
            (prediction["mode_logits"].argmax(-1) == batch["teacher_tactical_mode"]).sum()
        )
        mode_count += batch["teacher_tactical_mode"].numel()
        mu_saturated += int((prediction["mean"].abs() >= 0.95).sum())
        mu_count += prediction["mean"].numel()
    if prior:
        model.train()
    metrics = {f"validation_{key}_loss": value / batches for key, value in totals.items()}
    metrics["validation_mode_accuracy"] = mode_correct / mode_count
    metrics["mu_saturation_rate"] = mu_saturated / mu_count
    return metrics


def stochastic_action(
    prediction: dict[str, torch.Tensor], stochastic: bool, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, dict[str, float]]:
    mean = prediction["mean"]
    sampled = mean
    if stochastic:
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        sampled = mean + noise * prediction["log_std"].exp()
    applied = sampled.clamp(-1.0, 1.0)
    return applied, {
        "mu_saturation_rate": float((mean.abs() >= 0.95).float().mean()),
        "sampled_saturation_rate": float((sampled.abs() >= 1.0).float().mean()),
        "env_applied_saturation_rate": float((applied.abs() >= 0.999).float().mean()),
    }


def compose_actions(
    actor_observation: np.ndarray,
    navigation_action: np.ndarray,
    combat_action: np.ndarray,
) -> np.ndarray:
    """Leakage-safe handoff: local45 combat is used only under actor-visible LOS."""
    actor = np.asarray(actor_observation, dtype=np.float32)
    navigation = np.asarray(navigation_action, dtype=np.float32)
    combat = np.asarray(combat_action, dtype=np.float32)
    if actor.ndim != 2 or actor.shape[1] != ACTOR_DIM:
        raise ValueError("actor observation must be [batch,231]")
    output = np.zeros((len(actor), ACTION_DIM), dtype=np.float32)
    output[:, :4] = np.clip(navigation[:, :4], -1.0, 1.0)
    visible = actor[:, LOS_INDEX] > 0.5
    output[visible] = combat[visible]
    return output


class FairReacquisitionSearch:
    """Actor231-only last-seen/cue search policy; no privileged suffix is accepted."""

    def __init__(self, stale_timeout_seconds: float = 4.0) -> None:
        self.stale_timeout_seconds = float(np.clip(stale_timeout_seconds, 0.5, 20.0))
        self.search_side: dict[int, int] = {}
        self.hidden_ticks: dict[int, int] = {}

    def reset(self, agent_id: int) -> None:
        self.search_side.pop(int(agent_id), None)
        self.hidden_ticks.pop(int(agent_id), None)

    def apply(
        self,
        actor_observation: np.ndarray,
        navigation_action: np.ndarray,
        agent_ids: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        actor = np.asarray(actor_observation, dtype=np.float32)
        output = np.asarray(navigation_action, dtype=np.float32).copy()
        ids = [int(value) for value in agent_ids]
        if actor.ndim != 2 or actor.shape[1] != ACTOR_DIM or len(ids) != len(actor):
            raise ValueError("reacquisition search requires [batch,231] fair observations")
        active = np.zeros(len(actor), dtype=np.bool_)
        for row, agent_id in enumerate(ids):
            if actor[row, LOS_INDEX] > 0.5:
                self.reset(agent_id)
                continue
            active[row] = True
            ticks = self.hidden_ticks.get(agent_id, 0) + 1
            self.hidden_ticks[agent_id] = ticks
            angle_degrees: float | None = None
            last_seen_age = float(actor[row, 206]) * 10.0
            last_seen_distance = float(np.hypot(actor[row, 203], actor[row, 205]))
            if actor[row, 202] > 0.5 and last_seen_age < self.stale_timeout_seconds and last_seen_distance > 0.025:
                angle_degrees = float(np.degrees(np.arctan2(actor[row, 203], actor[row, 205])))
            elif actor[row, 207] > 0.5 and actor[row, 210] < 0.9:
                angle_degrees = float(np.degrees(np.arctan2(actor[row, 208], actor[row, 209])))
            elif actor[row, 224] > 0.5:
                bearing_bin = int(np.argmax(actor[row, 211:219]))
                angle_degrees = -180.0 + (bearing_bin + 0.5) * 45.0
            if angle_degrees is None:
                side = self.search_side.setdefault(agent_id, 1 if agent_id % 2 == 0 else -1)
                if ticks % 100 == 0:
                    side = -side
                    self.search_side[agent_id] = side
                angle_degrees = float(side * 75.0)
            ray_clearance = np.asarray(
                [actor[row, 30 + sector * 8] for sector in range(8)], dtype=np.float32
            )
            front_blocked = ray_clearance[0] < 0.08 or actor[row, 28] > 0.45
            if front_blocked:
                right = float(max(ray_clearance[1], ray_clearance[2]))
                left = float(max(ray_clearance[6], ray_clearance[7]))
                side = 1 if right >= left else -1
                self.search_side[agent_id] = side
                output[row, 0] = 0.75 * side
                output[row, 1] = 0.15
                output[row, 2] = 0.85 * side
            else:
                output[row, 0] = float(np.clip(np.sin(np.radians(angle_degrees)) * 0.35, -0.5, 0.5))
                output[row, 1] = 0.90 if abs(angle_degrees) < 80.0 else 0.25
                output[row, 2] = float(np.clip(angle_degrees / 70.0, -1.0, 1.0))
            output[row, 3] = 0.0
        return output, active


class MapIndependentSafetyLayer:
    """Persistent actor231-only blocked-motion/wall-follow adapter."""

    def __init__(self, flip_interval: int = 500) -> None:
        if flip_interval < 0:
            raise ValueError("flip_interval must be non-negative")
        self.flip_interval = int(flip_interval)
        self.wall_side: dict[int, int] = {}
        self.clear_ticks: dict[int, int] = {}
        self.hidden_ticks: dict[int, int] = {}

    def reset(self, agent_id: int) -> None:
        self.wall_side.pop(int(agent_id), None)
        self.clear_ticks.pop(int(agent_id), None)
        self.hidden_ticks.pop(int(agent_id), None)

    def apply(
        self,
        actor_observation: np.ndarray,
        navigation_action: np.ndarray,
        agent_ids: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        actor = np.asarray(actor_observation, dtype=np.float32)
        output = np.asarray(navigation_action, dtype=np.float32).copy()
        ids = [int(value) for value in agent_ids]
        if actor.ndim != 2 or actor.shape[1] != ACTOR_DIM or len(ids) != len(actor):
            raise ValueError("actor observation/agent IDs must be [batch,231]/[batch]")
        intervened = np.zeros(len(actor), dtype=np.bool_)
        for row, agent_id in enumerate(ids):
            if actor[row, LOS_INDEX] > 0.5:
                self.reset(agent_id)
                continue
            self.hidden_ticks[agent_id] = self.hidden_ticks.get(agent_id, 0) + 1
            if actor[row, 224] <= 0.5:
                continue
            bearing_values = actor[row, 211:219]
            if bearing_values.max(initial=0.0) <= 0.5:
                continue
            bearing_bin = int(np.argmax(bearing_values))
            target_sector = (bearing_bin + 4) % 8
            ray_clearance = np.asarray(
                [actor[row, 30 + sector * 8] for sector in range(8)],
                dtype=np.float32,
            )
            target_blocked = ray_clearance[target_sector] < 0.075
            collision_pressure = actor[row, 28] > 0.25 or actor[row, 27] > 0.35
            active = agent_id in self.wall_side
            if active and ray_clearance[target_sector] >= 0.14 and not collision_pressure:
                self.clear_ticks[agent_id] = self.clear_ticks.get(agent_id, 0) + 1
                if self.clear_ticks[agent_id] >= 8:
                    self.wall_side.pop(agent_id, None)
                    self.clear_ticks.pop(agent_id, None)
                    active = False
            else:
                self.clear_ticks[agent_id] = 0
            if not active and (target_blocked or collision_pressure):
                clockwise = ray_clearance[(target_sector + 1) % 8]
                counter = ray_clearance[(target_sector - 1) % 8]
                self.wall_side[agent_id] = 1 if clockwise >= counter else -1
                active = True
            if active and self.flip_interval > 0 and self.hidden_ticks[agent_id] % self.flip_interval == 0:
                self.wall_side[agent_id] = -self.wall_side[agent_id]
                self.clear_ticks[agent_id] = 0
            if not active:
                continue
            side = self.wall_side[agent_id]
            chosen = target_sector
            for offset in range(1, 8):
                candidate = (target_sector + side * offset) % 8
                if ray_clearance[candidate] >= 0.06:
                    chosen = candidate
                    break
            if ray_clearance[chosen] < 0.025:
                chosen = int(np.argmax(ray_clearance))
            sector_angles = np.arange(8, dtype=np.float32) * 45.0
            sector_angles[sector_angles > 180.0] -= 360.0
            chosen_angle = float(sector_angles[chosen])
            output[row, 0] = float(
                np.clip(np.sin(np.radians(chosen_angle)) * 0.75, -0.75, 0.75)
            )
            output[row, 1] = 0.85 if abs(chosen_angle) < 100.0 else 0.15
            output[row, 2] = float(np.clip(chosen_angle / 75.0, -1.0, 1.0))
            intervened[row] = True
        return output, intervened


def apply_map_independent_safety_layer(
    actor_observation: np.ndarray,
    navigation_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stateless convenience wrapper used by unit tests."""
    actor = np.asarray(actor_observation, dtype=np.float32)
    layer = MapIndependentSafetyLayer()
    return layer.apply(actor, navigation_action, range(len(actor)))


def persistent_mean_saturation(values: Iterable[float], threshold: float = 0.35) -> bool:
    rates = [float(value) for value in values]
    return len(rates) >= 3 and all(value > threshold for value in rates[-3:])


def selection_key(record: dict[str, Any]) -> tuple[float, float, float, str]:
    """Lower is better. Only validation fields may participate."""
    return (
        float(record["validation_navigation_loss"]),
        float(record["validation_total_loss"]),
        float(record["mu_saturation_rate"]),
        str(record["candidate_id"]),
    )


def rank_candidates(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = list(records)
    forbidden = ("heldout", "test_contact", "heldout_contact")
    for record in records:
        if any(any(token in key.lower() for token in forbidden) for key in record):
            raise RuntimeError("heldout/test metrics are forbidden during selection")
    return sorted(records, key=selection_key)


def save_checkpoint(
    path: str | Path,
    model: Phase5Navigator,
    optimizer: torch.optim.Optimizer,
    update: int,
    metadata: dict[str, Any],
) -> None:
    payload = {
        "schema_version": "phase5_navigator_checkpoint_v001",
        "actor_schema": "phase5_actor_obs_v001",
        "actor_dim": ACTOR_DIM,
        "config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "update": int(update),
        "metadata": metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path, device: torch.device | str, load_optimizer: bool = False
) -> tuple[Phase5Navigator, dict[str, Any], torch.optim.Optimizer | None]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != "phase5_navigator_checkpoint_v001":
        raise RuntimeError("navigator checkpoint schema mismatch")
    if payload.get("actor_dim") != ACTOR_DIM:
        raise RuntimeError("navigator checkpoint actor dimension mismatch")
    config = NavigatorConfig(**payload["config"])
    model = Phase5Navigator(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer = None
    if load_optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return model, payload, optimizer
