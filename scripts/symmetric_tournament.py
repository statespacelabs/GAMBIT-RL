#!/usr/bin/env python3
"""Symmetric dual-policy terminal tournament runner for Phase 6 Goal 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.autonomous_combat_baseline import make_action_tuple  # noqa: E402
from scripts.collect_navigation_dagger import action_from_distribution  # noqa: E402
from scripts.navigation_ppo import CRITIC_DIM  # noqa: E402
from scripts.navigation_policy import (  # noqa: E402
    ACTOR_DIM,
    LOCAL45_DIM,
    MapIndependentSafetyLayer,
    compose_actions,
    load_checkpoint,
)
from scripts.local_geometry_peeker import LocalGeometryPeeker, detect_local_cover  # noqa: E402

FIXED_STEP_SECONDS = 0.02
MATCH_SCHEMA_VERSION = "phase6_tournament_match_v001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-path", required=True)
    parser.add_argument("--policy-a-id", required=True)
    parser.add_argument("--policy-a", required=True)
    parser.add_argument("--policy-a-sha256", required=True)
    parser.add_argument("--policy-b-id", required=True)
    parser.add_argument("--policy-b", required=True)
    parser.add_argument("--policy-b-sha256", required=True)
    parser.add_argument("--combat-a", required=True)
    parser.add_argument("--combat-a-sha256", required=True)
    parser.add_argument("--combat-b", required=True)
    parser.add_argument("--combat-b-sha256", required=True)
    parser.add_argument("--normalizer-a", required=True)
    parser.add_argument("--normalizer-a-sha256", required=True)
    parser.add_argument("--normalizer-b", required=True)
    parser.add_argument("--normalizer-b-sha256", required=True)
    parser.add_argument(
        "--mode-a", choices=("deterministic", "stochastic"), required=True
    )
    parser.add_argument(
        "--mode-b", choices=("deterministic", "stochastic"), required=True
    )
    parser.add_argument(
        "--side-assignment", choices=("base", "swapped"), required=True
    )
    parser.add_argument("--layout-manifest", required=True)
    parser.add_argument("--layout-seed", type=int, required=True)
    parser.add_argument("--num-areas", type=int, default=8)
    parser.add_argument("--target-matches", type=int, required=True)
    parser.add_argument("--match-timeout-seconds", type=float, default=12.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--time-scale", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-fixed-steps", type=int, default=50000)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def split_observations(
    values: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_width: dict[int, np.ndarray] = {}
    for value in values:
        if value.ndim != 2:
            continue
        width = int(value.shape[1])
        if width in by_width:
            raise RuntimeError(f"duplicate observation width {width}")
        by_width[width] = np.asarray(value, dtype=np.float32)
    expected = {1, LOCAL45_DIM, ACTOR_DIM, CRITIC_DIM}
    if set(by_width) != expected:
        raise RuntimeError(
            f"tournament sensor widths must be {sorted(expected)}, "
            f"got {sorted(by_width)}"
        )
    role = by_width[1]
    local45 = by_width[LOCAL45_DIM]
    actor = by_width[ACTOR_DIM]
    critic = by_width[CRITIC_DIM]
    if len({len(role), len(local45), len(actor), len(critic)}) != 1:
        raise RuntimeError("tournament sensors are row-misaligned")
    if not all(np.isfinite(item).all() for item in (role, local45, actor, critic)):
        raise RuntimeError("tournament observation contains NaN/Inf")
    if not np.allclose(actor, critic[:, :ACTOR_DIM], atol=1e-5):
        raise RuntimeError("critic actor prefix differs from fair actor sensor")
    return role[:, 0], local45, actor, critic


def make_peeker(payload: dict[str, Any], seed_offset: int) -> LocalGeometryPeeker | None:
    config = payload.get("metadata", {}).get("peeker_config")
    if not config:
        return None
    return LocalGeometryPeeker(
        seed=int(config["branch"]["seed"]) + seed_offset,
        dwell_min=int(config["dwell_min_steps"]),
        dwell_max=int(config["dwell_max_steps"]),
        risk=float(config["risk"]),
        repeek_penalty=float(config["repeek_penalty"]),
    )


@dataclass
class MatchTracker:
    decision_count: int = 0
    first_contact_step: int | None = None
    had_contact: bool = False
    contact_was_lost: bool = False
    reacquisition_events: int = 0
    stuck_steps: int = 0
    collision_steps: int = 0
    cover_entries: int = 0
    behind_cover: bool = False
    peek_events: int = 0
    mean_saturated: int = 0
    mean_values: int = 0
    sampled_saturated: int = 0
    sampled_values: int = 0
    applied_saturated: int = 0
    applied_values: int = 0
    action_digest: Any = field(default_factory=hashlib.sha256)

    def observe(self, actor: np.ndarray) -> None:
        visible = bool(actor[193] > 0.5)
        if visible and self.first_contact_step is None:
            self.first_contact_step = self.decision_count
        if visible and self.had_contact and self.contact_was_lost:
            self.reacquisition_events += 1
            self.contact_was_lost = False
        elif not visible and self.had_contact:
            self.contact_was_lost = True
        self.had_contact = self.had_contact or visible
        self.stuck_steps += int(actor[26] > 0.60)
        self.collision_steps += int(actor[28] > 0.60)
        behind = detect_local_cover(actor).behind_cover
        if behind and not self.behind_cover:
            self.cover_entries += 1
        self.behind_cover = behind
        self.decision_count += 1

    def action(
        self,
        mean: np.ndarray,
        raw_sample: np.ndarray,
        applied: np.ndarray,
        peek_started: bool,
    ) -> None:
        self.mean_saturated += int((np.abs(mean) >= 0.95).sum())
        self.mean_values += int(mean.size)
        self.sampled_saturated += int((np.abs(raw_sample) >= 1.0).sum())
        self.sampled_values += int(raw_sample.size)
        self.applied_saturated += int((np.abs(applied[:4]) >= 0.999).sum())
        self.applied_values += 4
        self.peek_events += int(peek_started)
        self.action_digest.update(np.asarray(applied, dtype="<f4").tobytes())

    def snapshot(self) -> dict[str, Any]:
        return {
            "decision_count": self.decision_count,
            "contact_time_seconds": (
                None
                if self.first_contact_step is None
                else self.first_contact_step * FIXED_STEP_SECONDS
            ),
            "stuck_time_seconds": self.stuck_steps * FIXED_STEP_SECONDS,
            "collision_time_seconds": self.collision_steps * FIXED_STEP_SECONDS,
            "cover_events": self.cover_entries,
            "peek_events": self.peek_events,
            "reacquisition_events": self.reacquisition_events,
            "policy_mean_action_saturation": (
                self.mean_saturated / max(1, self.mean_values)
            ),
            "sampled_action_saturation": (
                self.sampled_saturated / max(1, self.sampled_values)
            ),
            "environment_applied_saturation": (
                self.applied_saturated / max(1, self.applied_values)
            ),
            "applied_action_sha256": self.action_digest.hexdigest(),
        }


class PolicyRuntime:
    def __init__(
        self,
        *,
        policy_id: str,
        checkpoint: Path,
        checkpoint_sha256: str,
        combat_checkpoint: Path,
        combat_sha256: str,
        normalizer_path: Path,
        normalizer_sha256: str,
        mode: str,
        seed: int,
        seed_offset: int,
        device: torch.device,
        env_path: str,
        output_dir: Path,
        base_port: int,
        time_scale: float,
    ) -> None:
        from scripts.combat_runtime import (
            RuntimeConfig,
            load_policy_and_normalizer,
        )

        for label, path, expected in (
            ("navigator", checkpoint, checkpoint_sha256),
            ("combat", combat_checkpoint, combat_sha256),
            ("normalizer", normalizer_path, normalizer_sha256),
        ):
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(
                    f"{policy_id} {label} hash mismatch: {actual} != {expected}"
                )
        self.policy_id = policy_id
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = checkpoint_sha256
        self.combat_checkpoint = combat_checkpoint
        self.combat_sha256 = combat_sha256
        self.normalizer_path = normalizer_path
        self.normalizer_sha256 = normalizer_sha256
        self.mode = mode
        self.device = device
        self.model, self.payload, _ = load_checkpoint(checkpoint, device)
        self.model.eval()
        self.peeker = make_peeker(self.payload, seed_offset)
        self.safety = MapIndependentSafetyLayer()
        self.nav_hidden: dict[int, torch.Tensor] = {}
        self.combat_hidden: dict[int, torch.Tensor] = {}
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed + seed_offset)
        runtime = RuntimeConfig(
            env_path=env_path,
            normalizer_path=str(normalizer_path),
            resume_from=str(combat_checkpoint),
            output_dir=str(output_dir / f"runtime_{policy_id}"),
            base_port=base_port,
            seed=seed + seed_offset,
            time_scale=time_scale,
            device=str(device),
            game_mode="GambitVsGambit",
            player_b_bot_mode="Idle",
            num_areas=1,
        )
        (
            self.combat,
            self.normalizer,
            _,
            _,
            _,
            self.combat_device,
        ) = load_policy_and_normalizer(runtime)
        self.combat.set_ppo_mode(training=False)

    def act(
        self,
        *,
        actor: np.ndarray,
        local45: np.ndarray,
        ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool]]:
        combat_actions = action_from_distribution(
            self.combat,
            self.normalizer,
            self.combat_device,
            local45,
            ids,
            self.combat_hidden,
        )
        hidden_before = torch.cat(
            [
                self.nav_hidden.get(
                    agent_id, self.model.initial_hidden(1, self.device)
                )
                for agent_id in ids
            ],
            dim=1,
        )
        with torch.no_grad():
            prediction, next_hidden = self.model(
                torch.as_tensor(actor, device=self.device).unsqueeze(1),
                hidden_before,
            )
        mean = prediction["mean"][:, 0]
        raw = mean
        if self.mode == "stochastic":
            raw = mean + torch.randn(
                mean.shape,
                device=mean.device,
                dtype=mean.dtype,
                generator=self.generator,
            ) * prediction["log_std"][:, 0].exp()
        sampled = raw.clamp(-1.0, 1.0)
        navigation = np.zeros((len(ids), 8), dtype=np.float32)
        navigation[:, :4] = sampled.cpu().numpy()
        navigation, _ = self.safety.apply(actor, navigation, ids)
        previous_phases = [
            self.peeker.states.get(agent_id).phase
            if self.peeker is not None and agent_id in self.peeker.states
            else "idle"
            for agent_id in ids
        ]
        if self.peeker is not None:
            applied, _, _, _ = self.peeker.apply(
                actor, navigation, combat_actions, ids, None
            )
        else:
            applied = compose_actions(actor, navigation, combat_actions)
        peek_started = [
            self.peeker is not None
            and agent_id in self.peeker.states
            and previous_phases[index] != "peek"
            and self.peeker.states[agent_id].phase == "peek"
            for index, agent_id in enumerate(ids)
        ]
        for index, agent_id in enumerate(ids):
            self.nav_hidden[agent_id] = next_hidden[
                :, index : index + 1
            ].detach()
        return (
            applied,
            mean.detach().cpu().numpy(),
            raw.detach().cpu().numpy(),
            peek_started,
        )

    def reset(self, agent_id: int) -> int:
        self.nav_hidden.pop(agent_id, None)
        self.combat_hidden.pop(agent_id, None)
        self.safety.reset(agent_id)
        if self.peeker is not None:
            self.peeker.reset(agent_id)
        return int(
            agent_id in self.nav_hidden
            or agent_id in self.combat_hidden
            or agent_id in self.safety.wall_side
            or agent_id in self.safety.clear_ticks
            or agent_id in self.safety.hidden_ticks
            or (
                self.peeker is not None
                and agent_id in self.peeker.states
            )
        )


def configure_environment(
    config: argparse.Namespace, output: Path
) -> dict[str, Path]:
    paths = {
        "matches": output / "matches.jsonl",
        "result": output / "run_summary.json",
        "audit": output / "tournament_unity_audit.jsonl",
        "layout": output / "layout_audit.json",
        "runtime": output / "runtime_summary.json",
        "unity": output / "unity.log",
        "weapon": output / "weapon_state.jsonl",
    }
    output.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        if path.exists():
            path.unlink()
    manifest = Path(config.layout_manifest).resolve()
    environment = {
        "GAME_MODE": "GambitVsGambit",
        "PLAYER_B_BOT_MODE": "Idle",
        "NUM_AREAS": str(config.num_areas),
        "ENABLE_VISUAL_OBS": "0",
        "PHASE5_TELEMETRY_SCHEMA": "phase3v2_c_local45",
        "PHASE5_NAVIGATOR_EVAL": "1",
        "PHASE5_MAP_GENERAL_PPO": "1",
        "PHASE5_NAVMESH_UPPER_BOUND": "0",
        "PHASE5_ENABLE_NAVMESH_ORACLE": "0",
        "PHASE5_PROCEDURAL_LAYOUT": "1",
        "PHASE5_LAYOUT_MANIFEST": str(manifest),
        "PHASE5_LAYOUT_MANIFEST_SHA256": sha256_file(manifest),
        "PHASE5_LAYOUT_SPLIT": "validation",
        "PHASE5_LAYOUT_SEEDS": ",".join(
            [str(config.layout_seed)] * config.num_areas
        ),
        "PHASE5_LAYOUT_AUDIT_PATH": str(paths["layout"].resolve()),
        "PHASE5_AUTONOMOUS_SESSION": "0",
        "PHASE5_NAVMESH_ORACLE_LABEL": "0",
        "PHASE4_5_CONTINUOUS_CHALLENGERS": "0",
        "PHASE4_5_DEFERRED_CONTINUATION": "0",
        "FORCE_MATCH_RESET_ON_EPISODE_BEGIN": "0",
        "TARGET_HURTBOX_INFLATE": "0",
        "LEARNER_SHOOT_RAY_MODE": "camera_forward",
        "RAY_CORRECTION_ALPHA": "0",
        "HIT_PROBE_ORACLE": "0",
        "PHASE5_RUNTIME_METRICS": "1",
        "PHASE5_RUNTIME_TRACE_PATH": "",
        "PHASE5_RUNTIME_SUMMARY_PATH": str(paths["runtime"].resolve()),
        "PHASE5_GENERIC_TEACHER": "0",
        "PHASE6_SWAP_SPAWNS": "0",
        "PHASE6_TOURNAMENT_HARNESS": "1",
        "PHASE6_TOURNAMENT_AUDIT_PATH": str(paths["audit"].resolve()),
        "PHASE6_TOURNAMENT_TIMEOUT_SECONDS": str(
            config.match_timeout_seconds
        ),
        "DEBUG_LOG_LEVEL": "summary",
        "WEAPON_STATE_DEBUG": "1",
        "WEAPON_STATE_DEBUG_PATH": str(paths["weapon"].resolve()),
    }
    os.environ.update(environment)
    for key in (
        "PHASE4_MAP_CONTROL_ENABLED",
        "PHASE4_MAP_ID",
        "PHASE4_MAP_MIX_PROFILE",
        "PHASE4_MAP_SEED",
    ):
        os.environ.pop(key, None)
    return paths


def main() -> int:
    config = parse_args()
    if config.num_areas <= 0 or config.target_matches <= 0:
        raise RuntimeError("num areas and target matches must be positive")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    layout_manifest = json.loads(Path(config.layout_manifest).read_text())
    if layout_manifest.get("split") != "validation":
        raise RuntimeError("Goal 3 is restricted to validation layouts")
    layouts = {
        int(row["seed"]): row for row in layout_manifest["layouts"]
    }
    if config.layout_seed not in layouts:
        raise RuntimeError("layout seed is not in the validation manifest")
    layout = layouts[config.layout_seed]

    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.engine_configuration_channel import (
        EngineConfigurationChannel,
    )

    output = Path(config.output_dir).resolve()
    paths = configure_environment(config, output)
    device = torch.device(config.device)
    policy_a = PolicyRuntime(
        policy_id=config.policy_a_id,
        checkpoint=Path(config.policy_a).resolve(),
        checkpoint_sha256=config.policy_a_sha256,
        combat_checkpoint=Path(config.combat_a).resolve(),
        combat_sha256=config.combat_a_sha256,
        normalizer_path=Path(config.normalizer_a).resolve(),
        normalizer_sha256=config.normalizer_a_sha256,
        mode=config.mode_a,
        seed=config.seed,
        seed_offset=1000,
        device=device,
        env_path=config.env_path,
        output_dir=output,
        base_port=config.base_port,
        time_scale=config.time_scale,
    )
    policy_b = PolicyRuntime(
        policy_id=config.policy_b_id,
        checkpoint=Path(config.policy_b).resolve(),
        checkpoint_sha256=config.policy_b_sha256,
        combat_checkpoint=Path(config.combat_b).resolve(),
        combat_sha256=config.combat_b_sha256,
        normalizer_path=Path(config.normalizer_b).resolve(),
        normalizer_sha256=config.normalizer_b_sha256,
        mode=config.mode_b,
        seed=config.seed,
        seed_offset=2000,
        device=device,
        env_path=config.env_path,
        output_dir=output,
        base_port=config.base_port,
        time_scale=config.time_scale,
    )
    if (
        policy_a is policy_b
        or policy_a.model is policy_b.model
        or policy_a.normalizer is policy_b.normalizer
        or policy_a.nav_hidden is policy_b.nav_hidden
        or policy_a.combat_hidden is policy_b.combat_hidden
        or policy_a.safety is policy_b.safety
    ):
        raise RuntimeError("policy runtimes are not independent")

    physical_runtime = (
        {"A": policy_a, "B": policy_b}
        if config.side_assignment == "base"
        else {"A": policy_b, "B": policy_a}
    )
    logical_slot = (
        {"policy_a": "A", "policy_b": "B"}
        if config.side_assignment == "base"
        else {"policy_a": "B", "policy_b": "A"}
    )

    engine = EngineConfigurationChannel()
    engine.set_configuration_parameters(
        time_scale=config.time_scale,
        target_frame_rate=-1,
        capture_frame_rate=0,
    )
    env = None
    failure = ""
    fixed_steps = 0
    observation_rows = 0
    action_rows = 0
    nonfinite_observation_values = 0
    nonfinite_action_values = 0
    actor_prefix_max_error = 0.0
    action_abs_max = 0.0
    hidden_state_leakage_events = 0
    reset_events = 0
    applied_digest = hashlib.sha256()
    agent_area: dict[int, int] = {}
    agent_slot: dict[int, str] = {}
    area_agents: dict[int, dict[str, int]] = {
        index: {} for index in range(config.num_areas)
    }
    trackers: dict[int, MatchTracker] = {}
    area_match_index = {index: 0 for index in range(config.num_areas)}
    boundaries: list[dict[str, Any]] = []
    boundary_seen_step: dict[int, int] = {}
    started = time.perf_counter()

    def establish_mapping(
        role: np.ndarray, ids: list[int]
    ) -> None:
        if agent_area:
            return
        slot_a = [ids[index] for index in np.flatnonzero(role > 0.5)]
        slot_b = [ids[index] for index in np.flatnonzero(role < 0.5)]
        if len(slot_a) != config.num_areas or len(slot_b) != config.num_areas:
            raise RuntimeError(
                f"cannot map {len(slot_a)}/{len(slot_b)} agents to "
                f"{config.num_areas} areas"
            )
        for area_id, agent_id in enumerate(slot_a):
            agent_area[agent_id] = area_id
            agent_slot[agent_id] = "A"
            area_agents[area_id]["A"] = agent_id
        for area_id, agent_id in enumerate(slot_b):
            agent_area[agent_id] = area_id
            agent_slot[agent_id] = "B"
            area_agents[area_id]["B"] = agent_id

    try:
        env = UnityEnvironment(
            file_name=config.env_path,
            worker_id=0,
            base_port=config.base_port,
            seed=config.seed,
            side_channels=[engine],
            no_graphics=True,
            additional_args=[
                "--phase5-headless",
                "-logFile",
                str(paths["unity"]),
            ],
            timeout_wait=config.timeout,
        )
        env.reset()
        behaviors = list(env.behavior_specs)
        if len(behaviors) != 1:
            raise RuntimeError(f"expected one behavior, got {behaviors}")
        behavior = behaviors[0]
        action_spec = env.behavior_specs[behavior].action_spec

        for fixed_steps in range(1, config.max_fixed_steps + 1):
            decisions, terminals = env.get_steps(behavior)
            if len(decisions):
                role, local45, actor, critic = split_observations(decisions.obs)
                ids = [int(value) for value in decisions.agent_id]
                establish_mapping(role, ids)
                observation_rows += len(ids)
                nonfinite_observation_values += int(
                    (~np.isfinite(local45)).sum()
                    + (~np.isfinite(actor)).sum()
                    + (~np.isfinite(critic)).sum()
                )
                actor_prefix_max_error = max(
                    actor_prefix_max_error,
                    float(
                        np.max(
                            np.abs(actor - critic[:, :ACTOR_DIM]),
                            initial=0.0,
                        )
                    ),
                )
                applied = np.zeros((len(ids), 8), dtype=np.float32)
                for physical_slot, rows in (
                    ("A", np.flatnonzero(role > 0.5)),
                    ("B", np.flatnonzero(role < 0.5)),
                ):
                    if not len(rows):
                        continue
                    runtime = physical_runtime[physical_slot]
                    side_ids = [ids[index] for index in rows]
                    for row, agent_id in zip(rows, side_ids):
                        trackers.setdefault(agent_id, MatchTracker()).observe(
                            actor[row]
                        )
                    side_actions, means, raw_samples, peek_started = runtime.act(
                        actor=actor[rows],
                        local45=local45[rows],
                        ids=side_ids,
                    )
                    applied[rows] = side_actions
                    for local_index, agent_id in enumerate(side_ids):
                        trackers[agent_id].action(
                            means[local_index],
                            raw_samples[local_index],
                            side_actions[local_index],
                            peek_started[local_index],
                        )
                nonfinite_action_values += int((~np.isfinite(applied)).sum())
                action_abs_max = max(
                    action_abs_max,
                    float(np.max(np.abs(applied), initial=0.0)),
                )
                if nonfinite_action_values:
                    raise RuntimeError("tournament action contains NaN/Inf")
                if action_abs_max > 1.000001:
                    raise RuntimeError(
                        f"tournament action out of range: {action_abs_max}"
                    )
                action_rows += len(applied)
                applied_digest.update(
                    np.asarray(applied, dtype="<f4").tobytes()
                )
                env.set_actions(
                    behavior, make_action_tuple(applied, action_spec)
                )

            if len(terminals):
                terminal_role, _, terminal_actor, terminal_critic = (
                    split_observations(terminals.obs)
                )
                terminal_ids = [int(value) for value in terminals.agent_id]
                if not agent_area:
                    establish_mapping(terminal_role, terminal_ids)
                actor_prefix_max_error = max(
                    actor_prefix_max_error,
                    float(
                        np.max(
                            np.abs(
                                terminal_actor
                                - terminal_critic[:, :ACTOR_DIM]
                            ),
                            initial=0.0,
                        )
                    ),
                )
                terminal_rewards = {
                    int(agent_id): float(terminals.reward[index])
                    for index, agent_id in enumerate(terminals.agent_id)
                }
                terminal_areas = {
                    agent_area[agent_id]
                    for agent_id in terminal_ids
                    if agent_id in agent_area
                }
                for area_id in sorted(terminal_areas):
                    if boundary_seen_step.get(area_id) == fixed_steps:
                        continue
                    if len(boundaries) >= config.target_matches:
                        continue
                    boundary_seen_step[area_id] = fixed_steps
                    area_match_index[area_id] += 1
                    ids_by_slot = area_agents[area_id]
                    tracker_a = trackers.get(
                        ids_by_slot["A"], MatchTracker()
                    ).snapshot()
                    tracker_b = trackers.get(
                        ids_by_slot["B"], MatchTracker()
                    ).snapshot()
                    boundaries.append(
                        {
                            "area_id": area_id,
                            "match_index": area_match_index[area_id],
                            "fixed_step": fixed_steps,
                            "physical_slot_a": tracker_a,
                            "physical_slot_b": tracker_b,
                            "terminal_reward_slot_a": terminal_rewards.get(
                                ids_by_slot["A"]
                            ),
                            "terminal_reward_slot_b": terminal_rewards.get(
                                ids_by_slot["B"]
                            ),
                        }
                    )
                    for physical_slot in ("A", "B"):
                        agent_id = ids_by_slot[physical_slot]
                        hidden_state_leakage_events += physical_runtime[
                            physical_slot
                        ].reset(agent_id)
                        trackers[agent_id] = MatchTracker()
                        reset_events += 1
            if len(boundaries) >= config.target_matches:
                break
            env.step()
        else:
            raise RuntimeError(
                f"tournament exceeded max_fixed_steps={config.max_fixed_steps}"
            )
    except Exception as exc:
        failure = str(exc)
    finally:
        if env is not None:
            env.close()

    for _ in range(100):
        if (
            paths["audit"].is_file()
            and paths["layout"].is_file()
            and paths["runtime"].is_file()
        ):
            break
        time.sleep(0.05)
    audit_rows = read_jsonl(paths["audit"])
    ready_rows = [row for row in audit_rows if row.get("event") == "area_ready"]
    terminal_rows = [
        row for row in audit_rows if row.get("event") == "terminal"
    ]
    boundary_keys = {
        (int(row["area_id"]), int(row["match_index"]))
        for row in boundaries
    }
    relevant_terminal_rows = [
        row
        for row in terminal_rows
        if (int(row["area_id"]), int(row["match_index"])) in boundary_keys
    ]
    ignored_post_target_terminal_events = (
        len(terminal_rows) - len(relevant_terminal_rows)
    )
    terminal_by_key = {
        (int(row["area_id"]), int(row["match_index"])): row
        for row in relevant_terminal_rows
    }
    audit_terminal_count_mismatch = int(
        len(relevant_terminal_rows) != len(boundaries)
        or len(terminal_by_key) != len(boundaries)
    )
    joined_rows = []
    missing_terminal_outcomes = 0
    side_config_mismatches = 0
    spawn_a = layout["spawn_a"]
    spawn_b = layout["spawn_b"]
    for sequence, boundary in enumerate(boundaries, start=1):
        key = (boundary["area_id"], boundary["match_index"])
        terminal = terminal_by_key.get(key)
        if terminal is None:
            missing_terminal_outcomes += 1
            continue
        terminal_type = terminal["terminal_type"]
        killer_slot = terminal["killer_slot"]
        policy_a_slot = logical_slot["policy_a"]
        if terminal_type == "timeout":
            payoff = 0.0
        elif terminal_type == "kill" and killer_slot in {"A", "B"}:
            payoff = 1.0 if killer_slot == policy_a_slot else -1.0
        else:
            missing_terminal_outcomes += 1
            continue
        stats_a = (
            boundary["physical_slot_a"]
            if policy_a_slot == "A"
            else boundary["physical_slot_b"]
        )
        stats_b = (
            boundary["physical_slot_b"]
            if policy_a_slot == "A"
            else boundary["physical_slot_a"]
        )
        hits_a = int(
            terminal[
                "hits_by_slot_a"
                if policy_a_slot == "A"
                else "hits_by_slot_b"
            ]
        )
        hits_b = int(
            terminal[
                "hits_by_slot_b"
                if policy_a_slot == "A"
                else "hits_by_slot_a"
            ]
        )
        physical_spawn_a = spawn_a if policy_a_slot == "A" else spawn_b
        physical_spawn_b = spawn_b if policy_a_slot == "A" else spawn_a
        joined_rows.append(
            {
                "schema_version": MATCH_SCHEMA_VERSION,
                "run_id": output.name,
                "match_sequence": sequence,
                "area_id": boundary["area_id"],
                "area_match_index": boundary["match_index"],
                "policy_a_id": config.policy_a_id,
                "policy_a_sha256": config.policy_a_sha256,
                "policy_b_id": config.policy_b_id,
                "policy_b_sha256": config.policy_b_sha256,
                "policy_a_mode": config.mode_a,
                "policy_b_mode": config.mode_b,
                "side_assignment": config.side_assignment,
                "policy_a_physical_slot": policy_a_slot,
                "policy_b_physical_slot": logical_slot["policy_b"],
                "map_source": "phase5_procedural_validation",
                "layout_family": layout["family"],
                "layout_sha256": layout["layout_sha256"],
                "layout_seed": config.layout_seed,
                "spawn_pair": {
                    "policy_a": physical_spawn_a,
                    "policy_b": physical_spawn_b,
                },
                "terminal_type": terminal_type,
                "winner_policy_id": (
                    None
                    if payoff == 0
                    else config.policy_a_id
                    if payoff > 0
                    else config.policy_b_id
                ),
                "timeout": terminal_type == "timeout",
                "terminal_payoff_policy_a": payoff,
                "terminal_payoff_policy_b": -payoff,
                "terminal_reward_slot_a_diagnostic": boundary[
                    "terminal_reward_slot_a"
                ],
                "terminal_reward_slot_b_diagnostic": boundary[
                    "terminal_reward_slot_b"
                ],
                "payoff_source": "unity_terminal_event_only",
                "dense_reward_used_for_payoff": False,
                "duration_seconds": float(terminal["duration_seconds"]),
                "policy_a_contact_time_seconds": stats_a[
                    "contact_time_seconds"
                ],
                "policy_b_contact_time_seconds": stats_b[
                    "contact_time_seconds"
                ],
                "contact_time_seconds": min(
                    value
                    for value in (
                        stats_a["contact_time_seconds"],
                        stats_b["contact_time_seconds"],
                    )
                    if value is not None
                )
                if (
                    stats_a["contact_time_seconds"] is not None
                    or stats_b["contact_time_seconds"] is not None
                )
                else None,
                "policy_a_damage": 20 * hits_a,
                "policy_b_damage": 20 * hits_b,
                "policy_a_kills": int(payoff > 0),
                "policy_b_kills": int(payoff < 0),
                "policy_a_deaths": int(payoff < 0),
                "policy_b_deaths": int(payoff > 0),
                "policy_a_stuck_time_seconds": stats_a[
                    "stuck_time_seconds"
                ],
                "policy_b_stuck_time_seconds": stats_b[
                    "stuck_time_seconds"
                ],
                "policy_a_collision_time_seconds": stats_a[
                    "collision_time_seconds"
                ],
                "policy_b_collision_time_seconds": stats_b[
                    "collision_time_seconds"
                ],
                "policy_a_cover_events": stats_a["cover_events"],
                "policy_b_cover_events": stats_b["cover_events"],
                "policy_a_peek_events": stats_a["peek_events"],
                "policy_b_peek_events": stats_b["peek_events"],
                "policy_a_reacquisition_events": stats_a[
                    "reacquisition_events"
                ],
                "policy_b_reacquisition_events": stats_b[
                    "reacquisition_events"
                ],
                "policy_a_mean_saturation": stats_a[
                    "policy_mean_action_saturation"
                ],
                "policy_b_mean_saturation": stats_b[
                    "policy_mean_action_saturation"
                ],
                "policy_a_sampled_saturation": stats_a[
                    "sampled_action_saturation"
                ],
                "policy_b_sampled_saturation": stats_b[
                    "sampled_action_saturation"
                ],
                "policy_a_applied_saturation": stats_a[
                    "environment_applied_saturation"
                ],
                "policy_b_applied_saturation": stats_b[
                    "environment_applied_saturation"
                ],
                "policy_a_action_sha256": stats_a[
                    "applied_action_sha256"
                ],
                "policy_b_action_sha256": stats_b[
                    "applied_action_sha256"
                ],
                "reset_generation": int(terminal["reset_generation"]),
            }
        )

    with paths["matches"].open("w", encoding="utf-8") as handle:
        for row in joined_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    layout_audit = (
        json.loads(paths["layout"].read_text())
        if paths["layout"].is_file()
        else {}
    )
    runtime_audit = (
        json.loads(paths["runtime"].read_text())
        if paths["runtime"].is_file()
        else {}
    )
    weapon_rows = read_jsonl(paths["weapon"])
    weapon_ownership_mismatches = sum(
        row.get("blocked_fire_reason") in {"NO_OWNER", "WRONG_OWNER"}
        for row in weapon_rows
    )
    unity_text = (
        paths["unity"].read_text(errors="replace")
        if paths["unity"].is_file()
        else ""
    )
    weapon_fire_mismatches = unity_text.count(
        "First post-reset shot failed"
    )
    human_controller_count = sum(
        int(row.get("human_controller_count", -1)) for row in ready_rows
    )
    side_config_mismatches += int(
        logical_slot["policy_a"]
        != ("A" if config.side_assignment == "base" else "B")
    )
    side_config_mismatches += int(
        len(ready_rows) != config.num_areas
        or layout_audit.get("area_count") != config.num_areas
        or any(
            int(row.get("seed", -1)) != config.layout_seed
            for row in layout_audit.get("areas", [])
        )
    )
    terminal_payoffs = {
        float(row["terminal_payoff_policy_a"]) for row in joined_rows
    }
    elapsed = time.perf_counter() - started
    status = (
        "PASS"
        if not failure
        and len(joined_rows) == config.target_matches
        and audit_terminal_count_mismatch == 0
        and missing_terminal_outcomes == 0
        and hidden_state_leakage_events == 0
        and side_config_mismatches == 0
        and nonfinite_observation_values == 0
        and nonfinite_action_values == 0
        and actor_prefix_max_error <= 1e-5
        and action_abs_max <= 1.000001
        and terminal_payoffs <= {-1.0, 0.0, 1.0}
        and weapon_ownership_mismatches == 0
        and weapon_fire_mismatches == 0
        and human_controller_count == 0
        and layout_audit.get("status") == "PASS"
        and runtime_audit.get("status") == "PASS"
        and runtime_audit.get("headless_audit", {}).get("status") == "PASS"
        else "FAIL"
    )
    result = {
        "schema_version": "phase6_tournament_run_summary_v001",
        "status": status,
        "failure_reason": failure,
        "policy_a": {
            "id": config.policy_a_id,
            "checkpoint": str(Path(config.policy_a).resolve()),
            "sha256": config.policy_a_sha256,
            "mode": config.mode_a,
            "combat_sha256": config.combat_a_sha256,
            "normalizer_sha256": config.normalizer_a_sha256,
        },
        "policy_b": {
            "id": config.policy_b_id,
            "checkpoint": str(Path(config.policy_b).resolve()),
            "sha256": config.policy_b_sha256,
            "mode": config.mode_b,
            "combat_sha256": config.combat_b_sha256,
            "normalizer_sha256": config.normalizer_b_sha256,
        },
        "independent_runtime_objects": True,
        "independent_recurrent_states": True,
        "independent_normalizer_objects": id(policy_a.normalizer)
        != id(policy_b.normalizer),
        "side_assignment": config.side_assignment,
        "logical_slot_assignment": logical_slot,
        "layout_split": "validation",
        "layout_seed": config.layout_seed,
        "layout_family": layout["family"],
        "layout_sha256": layout["layout_sha256"],
        "num_areas": config.num_areas,
        "target_matches": config.target_matches,
        "matches_completed": len(joined_rows),
        "terminal_counts": {
            "win": sum(
                row["terminal_payoff_policy_a"] > 0 for row in joined_rows
            ),
            "draw": sum(
                row["terminal_payoff_policy_a"] == 0 for row in joined_rows
            ),
            "loss": sum(
                row["terminal_payoff_policy_a"] < 0 for row in joined_rows
            ),
        },
        "missing_terminal_outcomes": missing_terminal_outcomes,
        "unity_audit_terminal_count": len(relevant_terminal_rows),
        "unity_audit_total_terminal_events": len(terminal_rows),
        "ignored_post_target_terminal_events": (
            ignored_post_target_terminal_events
        ),
        "unity_audit_terminal_count_mismatch": audit_terminal_count_mismatch,
        "hidden_state_leakage_events": hidden_state_leakage_events,
        "reset_events": reset_events,
        "side_config_mismatches": side_config_mismatches,
        "observation_rows": observation_rows,
        "action_rows": action_rows,
        "observation_nonfinite_values": nonfinite_observation_values,
        "action_nonfinite_values": nonfinite_action_values,
        "critic_prefix_max_abs_error": actor_prefix_max_error,
        "action_abs_max": action_abs_max,
        "weapon_ownership_mismatch_count": weapon_ownership_mismatches,
        "weapon_fire_mismatch_count": weapon_fire_mismatches,
        "human_controller_count": human_controller_count,
        "actor_schema": "phase5_actor_obs_v001",
        "actor_dim": ACTOR_DIM,
        "actor_received_opponent_id": False,
        "actor_received_map_identity": False,
        "actor_received_privileged_suffix": False,
        "audit_only_sensor_added": False,
        "manual_input_events": 0,
        "rendered": False,
        "headless": True,
        "final_heldout_used": False,
        "payoff_source": "unity_terminal_event_only",
        "dense_reward_used_for_payoff": False,
        "fixed_environment_steps": fixed_steps,
        "elapsed_seconds": elapsed,
        "matches_per_second": len(joined_rows) / max(elapsed, 1e-9),
        "area_steps_per_second": (
            fixed_steps * config.num_areas / max(elapsed, 1e-9)
        ),
        "applied_action_sha256": applied_digest.hexdigest(),
        "agent_area_mapping": {
            str(agent_id): {
                "area_id": agent_area[agent_id],
                "physical_slot": agent_slot[agent_id],
            }
            for agent_id in sorted(agent_area)
        },
        "matches_path": str(paths["matches"]),
        "matches_sha256": sha256_file(paths["matches"]),
        "source_runtime_audits": {
            "unity_tournament": str(paths["audit"]),
            "layout": str(paths["layout"]),
            "runtime": str(paths["runtime"]),
            "weapon": str(paths["weapon"]),
            "unity_log": str(paths["unity"]),
        },
        "device": str(device),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
    }
    paths["result"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
