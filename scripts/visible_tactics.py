#!/usr/bin/env python3
"""Run one deterministic or stochastic live Unity navigator certification cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.autonomous_combat_baseline import (  # noqa: E402
    PRESET_CHALLENGERS,
    make_action_tuple,
    read_jsonl,
)
from scripts.collect_navigation_dagger import action_from_distribution  # noqa: E402
from scripts.navigation_policy import (  # noqa: E402
    ACTION_DIM,
    ACTOR_DIM,
    LOCAL45_DIM,
    LOCAL45_NORMALIZER_SHA256,
    LOS_INDEX,
    PARENT_SHA256,
    FairReacquisitionSearch,
    MapIndependentSafetyLayer,
    compose_actions,
    load_checkpoint,
    sha256_file,
    stochastic_action,
)
from scripts.tactical_handoff import FairTacticalHandoff  # noqa: E402
from scripts.local_geometry_peeker import LocalGeometryPeeker  # noqa: E402


class Goal7FairVisibleTactic:
    """Visible-contact adapter using actor231 fair telemetry only."""

    def __init__(self, config):
        self.strafe = float(np.clip(float(config.get("strafe", 0.0)), -0.95, 0.95))
        self.forward = float(np.clip(float(config.get("forward", 0.0)), -0.95, 0.95))
        self.switch_steps = max(0, int(config.get("switch_steps", 0)))
        self.jump = bool(config.get("jump", False))
        self.crouch_mode = str(config.get("crouch_mode", "none"))
        self.crouch_steps = max(1, int(config.get("crouch_steps", 4)))
        self.correct_aim = bool(config.get("correct_aim", True))
        self.auto_reload = bool(config.get("auto_reload", True))
        self.preserve_moving_translation = bool(
            config.get("preserve_moving_translation", False)
        )
        self.moving_threshold = float(config.get("moving_threshold", 0.02))
        moving_forward = config.get("moving_forward")
        self.moving_forward = (
            None
            if moving_forward is None
            else float(np.clip(float(moving_forward), -0.95, 0.95))
        )
        self._ticks = {}
        self._sides = {}

    def reset(self, agent_id):
        self._ticks.pop(int(agent_id), None)
        self._sides.pop(int(agent_id), None)

    def apply(self, actor_observation, action, agent_ids):
        actor = np.asarray(actor_observation, dtype=np.float32)
        output = np.asarray(action, dtype=np.float32).copy()
        if actor.ndim != 2 or actor.shape[1] != ACTOR_DIM:
            raise ValueError("Goal 7 visible tactic requires actor231")
        if output.shape != (len(actor), ACTION_DIM) or len(agent_ids) != len(actor):
            raise ValueError("Goal 7 visible tactic action shape mismatch")
        active = np.zeros(len(actor), dtype=np.bool_)
        for row, raw_agent_id in enumerate(agent_ids):
            agent_id = int(raw_agent_id)
            if self.auto_reload and actor[row, 9] <= 1e-4 and actor[row, 10] > 1e-4:
                output[row, 4] = 0.0
                output[row, 5] = 1.0
                active[row] = True
            if actor[row, LOS_INDEX] <= 0.5:
                continue
            tick = self._ticks.get(agent_id, 0)
            side = self._sides.get(agent_id)
            if side is None:
                side = -1 if actor[row, 191] >= actor[row, 187] else 1
            if self.switch_steps and tick and tick % self.switch_steps == 0:
                side = -side
            if side < 0 and actor[row, 191] < 0.025 and actor[row, 187] >= 0.025:
                side = 1
            elif side > 0 and actor[row, 187] < 0.025 and actor[row, 191] >= 0.025:
                side = -1
            self._ticks[agent_id] = tick + 1
            self._sides[agent_id] = side
            enemy_moving = (
                float(np.linalg.norm(actor[row, 199:202])) > self.moving_threshold
            )
            if enemy_moving and self.moving_forward is not None:
                output[row, 0] = 0.0
                output[row, 1] = self.moving_forward
            elif not (self.preserve_moving_translation and enemy_moving):
                output[row, 0] = self.strafe * side
                output[row, 1] = self.forward
            if self.correct_aim:
                bearing = float(np.arctan2(actor[row, 194], actor[row, 195]))
                output[row, 2] = float(np.clip(bearing / (np.pi / 4.0), -1.0, 1.0))
                if actor[row, 9] > 1e-4:
                    output[row, 4] = 1.0
                    output[row, 5] = 0.0
            if self.jump and actor[row, 6] > 0.5:
                output[row, 6] = 1.0
            if self.crouch_mode == "constant":
                output[row, 7] = 1.0
            elif self.crouch_mode == "toggle":
                output[row, 7] = 1.0 if (tick // self.crouch_steps) % 2 == 0 else 0.0
            active[row] = True
        return output, active


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-path", required=True)
    parser.add_argument("--navigator", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--prerequisite", required=True)
    parser.add_argument(
        "--source-kind", choices=("procedural", "authored"), default="procedural"
    )
    parser.add_argument("--manifest")
    parser.add_argument("--split", default="authored")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--map-id", default="arena_ascent_v1")
    parser.add_argument("--map-profile", default="original_only")
    parser.add_argument("--map-seed", type=int, default=58025)
    parser.add_argument("--spawn-bucket", default="obstacle")
    parser.add_argument("--preset", choices=tuple(PRESET_CHALLENGERS), required=True)
    parser.add_argument(
        "--mode",
        choices=("deterministic", "stochastic", "combat_baseline"),
        required=True,
    )
    parser.add_argument("--disable-safety-layer", action="store_true")
    parser.add_argument("--wall-flip-interval", type=int, default=500)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=57025)
    parser.add_argument("--time-scale", type=float, default=5.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-sessions-per-area", type=int, default=1)
    parser.add_argument("--challenger-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--hunt-cue-dropout", type=float, default=0.25)
    parser.add_argument("--hunt-dropout-min-seconds", type=float, default=1.0)
    parser.add_argument("--hunt-dropout-max-seconds", type=float, default=4.0)
    parser.add_argument("--search-stale-timeout-seconds", type=float, default=4.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--rendered", action="store_true")
    return parser.parse_args()


def split_dual_observations(
    observations: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    by_width = {
        int(value.shape[1]): np.asarray(value, dtype=np.float32)
        for value in observations
        if value.ndim == 2
    }
    if set(by_width) != {LOCAL45_DIM, ACTOR_DIM}:
        raise RuntimeError(
            f"dual sensor widths must be {{45,231}}, got {sorted(by_width)}"
        )
    local45, actor = by_width[LOCAL45_DIM], by_width[ACTOR_DIM]
    if (
        len(local45) != len(actor)
        or not np.isfinite(local45).all()
        or not np.isfinite(actor).all()
    ):
        raise RuntimeError("dual sensor transport is non-finite or row-misaligned")
    return local45, actor


def configure(
    args: argparse.Namespace, output: Path, seeds: list[int]
) -> dict[str, Path]:
    paths = {
        "trace": output / "autonomous_sessions.jsonl",
        "audit": output / "autonomous_audit.json",
        "layout": output / "layout_audit.json",
        "runtime": output / "runtime_summary.json",
        "unity": output / "unity.log",
        "map": output / "authored_map_telemetry.jsonl",
        "teacher_trace": output / "teacher_trace.jsonl",
        "teacher_summary": output / "teacher_summary.json",
        "weapon": output / "weapon_state.jsonl",
    }
    output.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        if path.exists():
            path.unlink()
    areas = len(seeds) if args.source_kind == "procedural" else 1
    environment = {
        "GAME_MODE": "HumanVsScripted",
        "PLAYER_B_BOT_MODE": "Idle",
        "NUM_AREAS": str(areas),
        "ENABLE_VISUAL_OBS": "0",
        "PHASE5_TELEMETRY_SCHEMA": "phase3v2_c_local45",
        "PHASE5_NAVIGATOR_EVAL": "1",
        "PHASE5_RENDERED_ATTENDED_SMOKE": "1" if args.rendered else "0",
        "PHASE5_AUTONOMOUS_SESSION": "1",
        "PHASE5_AUTONOMOUS_PRESET": args.preset,
        "PHASE5_AUTONOMOUS_RUN_ID": args.run_id,
        "PHASE5_AUTONOMOUS_TRACE_PATH": str(paths["trace"].resolve()),
        "PHASE5_AUTONOMOUS_AUDIT_PATH": str(paths["audit"].resolve()),
        "PHASE5_AUTONOMOUS_TARGET_SESSIONS_PER_AREA": str(
            args.target_sessions_per_area
        ),
        "PHASE5_AUTONOMOUS_CHALLENGER_TIMEOUT_SECONDS": str(
            args.challenger_timeout_seconds
        ),
        "PHASE5_HUNT_CUE_DROPOUT": str(
            args.hunt_cue_dropout if args.preset == "reacquire_search" else 0.15
        ),
        "PHASE5_HUNT_DROPOUT_MIN_SECONDS": str(
            args.hunt_dropout_min_seconds if args.preset == "reacquire_search" else 0.0
        ),
        "PHASE5_HUNT_DROPOUT_MAX_SECONDS": str(
            args.hunt_dropout_max_seconds if args.preset == "reacquire_search" else 0.0
        ),
        "PHASE5_SEARCH_STALE_TIMEOUT_SECONDS": str(args.search_stale_timeout_seconds),
        "PHASE4_5_CONTINUOUS_CHALLENGERS": "1",
        "PHASE4_5_DEFERRED_CONTINUATION": "1",
        "PHASE4_5_DEFERRED_SAFETY_SECONDS": "0",
        "PHASE4_5_CONTINUOUS_CHALLENGER_CONTACT_TIMEOUT": "0",
        "FORCE_MATCH_RESET_ON_EPISODE_BEGIN": "0",
        "TARGET_HURTBOX_INFLATE": "0",
        "LEARNER_SHOOT_RAY_MODE": "camera_forward",
        "RAY_CORRECTION_ALPHA": "0",
        "HIT_PROBE_ORACLE": "0",
        "PHASE5_RUNTIME_METRICS": "1",
        "PHASE5_RUNTIME_TRACE_PATH": "",
        "PHASE5_RUNTIME_SUMMARY_PATH": str(paths["runtime"].resolve()),
        "PHASE5_GENERIC_TEACHER": "0",
        "PHASE5_NAVMESH_UPPER_BOUND": "0",
        "PHASE5_ENABLE_NAVMESH_ORACLE": "0",
        "WEAPON_STATE_DEBUG": "1",
        "WEAPON_STATE_DEBUG_PATH": str(paths["weapon"].resolve()),
        "DEBUG_LOG_LEVEL": "summary",
    }
    if args.source_kind == "procedural":
        environment.update(
            {
                "PHASE5_PROCEDURAL_LAYOUT": "1",
                "PHASE5_LAYOUT_MANIFEST": str(Path(args.manifest).resolve()),
                "PHASE5_LAYOUT_MANIFEST_SHA256": sha256_file(args.manifest),
                "PHASE5_LAYOUT_SPLIT": args.split,
                "PHASE5_LAYOUT_SEEDS": ",".join(str(seed) for seed in seeds),
                "PHASE5_LAYOUT_AUDIT_PATH": str(paths["layout"].resolve()),
            }
        )
        for key in (
            "PHASE4_MAP_CONTROL_ENABLED",
            "PHASE4_MAP_ID",
            "PHASE4_MAP_MIX_PROFILE",
            "PHASE4_MAP_SEED",
            "PHASE4_WRITE_MAP_TELEMETRY",
            "PHASE4_MAP_TELEMETRY_PATH",
        ):
            os.environ.pop(key, None)
    else:
        hidden_start = args.preset in {
            "hunt_probe",
            "ppo_h1",
            "ppo_h2",
            "ppo_h4",
            "peek_p0",
            "peek_p1",
            "peek_p2",
            "peek_p3",
            "peek_p4",
        }
        authored_profiles = {
            ("arena_ascent_v1", "obstacle"): (None, None, None, None, 10, 30),
            ("arena_ascent_v1", "search_destroy"): (-90, 60, -100, 20, 20, 50),
            ("arena_breeze_v1", "obstacle"): (-125, 15, -45, 60, 8, 22),
            ("arena_breeze_v1", "search_destroy"): (-125, 15, -45, 60, 14, 32),
            ("arena_bind_v1", "obstacle"): (-135, -50, -45, 30, 8, 20),
            ("arena_bind_v1", "search_destroy"): (-135, -50, -45, 30, 12, 28),
        }
        profile = authored_profiles.get((args.map_id, args.spawn_bucket))
        if hidden_start and profile is None:
            raise RuntimeError(f"no frozen hidden-spawn profile for {args.map_id}")
        environment.update(
            {
                "PHASE5_PROCEDURAL_LAYOUT": "0",
                "PHASE4_MAP_CONTROL_ENABLED": "1",
                "PHASE4_MAP_ID": args.map_id,
                "PHASE4_MAP_MIX_PROFILE": args.map_profile,
                "PHASE4_MAP_SEED": str(args.map_seed),
                "PHASE4_WRITE_MAP_TELEMETRY": "1",
                "PHASE4_MAP_TELEMETRY_PATH": str(paths["map"].resolve()),
                "PHASE4_4_ENABLE_SPAWN_BUCKETS": "1" if hidden_start else "0",
                "PHASE4_4_SPAWN_BUCKET": args.spawn_bucket,
                "PHASE4_4_SPAWN_SEED": str(args.seed),
                "PHASE4_4_TRIAL_MAX_SECONDS": "600",
                "PHASE4_4_REQUIRE_OBSTACLE_BETWEEN": "1" if hidden_start else "0",
                "PHASE4_4_SPAWN_DISTANCE_MIN": str(profile[4]) if hidden_start else "",
                "PHASE4_4_SPAWN_DISTANCE_MAX": str(profile[5]) if hidden_start else "",
                "PHASE4_7_ENCOUNTER_MIN_X": str(profile[0])
                if hidden_start and profile[0] is not None
                else "",
                "PHASE4_7_ENCOUNTER_MAX_X": str(profile[1])
                if hidden_start and profile[1] is not None
                else "",
                "PHASE4_7_ENCOUNTER_MIN_Z": str(profile[2])
                if hidden_start and profile[2] is not None
                else "",
                "PHASE4_7_ENCOUNTER_MAX_Z": str(profile[3])
                if hidden_start and profile[3] is not None
                else "",
                "PHASE5_GENERIC_TEACHER": "1",
                "PHASE5_TEACHER_LABEL_SCHEMA": "phase5_teacher_label_v001",
                "PHASE5_TEACHER_TRACE_PATH": str(paths["teacher_trace"].resolve()),
                "PHASE5_TEACHER_SUMMARY_PATH": str(paths["teacher_summary"].resolve()),
                "PHASE5_TEACHER_CONTROL_PROBABILITY": "0",
                "PHASE5_TEACHER_SEED": str(args.seed),
                "PHASE5_DAGGER_ROUND": "goal13_authored_control_zero",
                "PHASE5_DAGGER_STORE_FRAMES": "0",
            }
        )
        for key in (
            "PHASE5_LAYOUT_MANIFEST",
            "PHASE5_LAYOUT_MANIFEST_SHA256",
            "PHASE5_LAYOUT_SPLIT",
            "PHASE5_LAYOUT_SEEDS",
            "PHASE5_LAYOUT_AUDIT_PATH",
        ):
            os.environ.pop(key, None)
    os.environ.update(environment)
    return paths


def validate_layouts(path: Path, split: str, seeds: list[int], preset: str) -> None:
    document = json.loads(path.read_text())
    if document.get("split") != split or document.get("immutable") is not True:
        raise RuntimeError("layout split/immutability mismatch")
    by_seed = {int(item["seed"]): item for item in document["layouts"]}
    for seed in seeds:
        layout = by_seed.get(seed)
        if layout is None:
            raise RuntimeError(f"layout seed absent: {seed}")
        expected_los = None
        if preset in {"duel_probe", "ppo_h0", "reacquire_search"}:
            expected_los = True
        elif preset in {
            "hunt_probe",
            "ppo_h1",
            "ppo_h2",
            "ppo_h4",
            *tuple(f"peek_p{i}" for i in range(5)),
        }:
            expected_los = False
        if (
            expected_los is not None
            and bool(layout["requested_initial_los"]) != expected_los
        ):
            raise RuntimeError(f"seed {seed} is incompatible with {preset}")


def main() -> int:
    args = parse_args()
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.engine_configuration_channel import (
        EngineConfigurationChannel,
    )
    from scripts.combat_runtime import (
        RuntimeConfig,
        load_policy_and_normalizer,
    )

    prerequisite = json.loads(Path(args.prerequisite).read_text())
    if prerequisite.get("status") not in {
        "PHASE5_GENERIC_DAGGER_DATA_READY",
        "PHASE5_MAP_GENERAL_HUNTER_READY",
        "PHASE5_REACQUISITION_READY",
        "PHASE5_LOCAL_GEOMETRY_PEEKER_READY",
        "PHASE5_LEAGUE_CANDIDATES_READY",
    }:
        raise RuntimeError("required Phase 5 parent prerequisite is not ready")
    if sha256_file(args.checkpoint) != PARENT_SHA256:
        raise RuntimeError("combat parent hash mismatch")
    if sha256_file(args.normalizer) != LOCAL45_NORMALIZER_SHA256:
        raise RuntimeError("local45 normalizer hash mismatch")
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if args.source_kind == "procedural":
        if not args.manifest or not seeds or len(seeds) != len(set(seeds)):
            raise RuntimeError(
                "procedural evaluation requires a manifest and unique seeds"
            )
        validate_layouts(Path(args.manifest), args.split, seeds, args.preset)
    else:
        seeds = []
    output = Path(args.output_dir).resolve()
    paths = configure(args, output, seeds)
    device = torch.device(args.device)
    navigator, navigator_payload, _ = load_checkpoint(args.navigator, device)
    navigator.eval()
    reacquire_config = (navigator_payload.get("metadata") or {}).get("reacquire_config")
    reacquisition_search = FairReacquisitionSearch(
        float(reacquire_config.get("stale_timeout_seconds", 4.0))
        if reacquire_config
        else 4.0
    )
    memory_steps = (
        int(reacquire_config.get("memory_steps", 0)) if reacquire_config else 0
    )
    runtime_metadata = navigator_payload.get("metadata") or {}
    peeker_config = runtime_metadata.get("peeker_config")
    visible_tactic_config = runtime_metadata.get("phase6_visible_tactic")
    goal7_reacquisition = bool(
        runtime_metadata.get("phase6_goal7_reacquisition", False)
    )
    disable_tactical_handoff = bool(
        runtime_metadata.get("phase6_disable_tactical_handoff", False)
    )
    visible_tactic = (
        Goal7FairVisibleTactic(visible_tactic_config) if visible_tactic_config else None
    )
    peeker = None
    if peeker_config:
        peeker = LocalGeometryPeeker(
            seed=int(peeker_config["branch"]["seed"]),
            dwell_min=int(peeker_config["dwell_min_steps"]),
            dwell_max=int(peeker_config["dwell_max_steps"]),
            risk=float(peeker_config["risk"]),
            repeek_penalty=float(peeker_config["repeek_penalty"]),
        )
    policy_cfg = RuntimeConfig(
        env_path=args.env_path,
        normalizer_path=args.normalizer,
        resume_from=args.checkpoint,
        output_dir=str(output),
        base_port=args.base_port,
        seed=args.seed,
        time_scale=args.time_scale,
        device=args.device,
        game_mode="HumanVsScripted",
        player_b_bot_mode="Idle",
        num_areas=len(seeds) if seeds else 1,
    )
    combat, normalizer, _, _, _, combat_device = load_policy_and_normalizer(policy_cfg)
    combat.set_ppo_mode(training=False)
    engine = EngineConfigurationChannel()
    engine.set_configuration_parameters(
        time_scale=args.time_scale, target_frame_rate=-1, capture_frame_rate=0
    )
    expected_sessions = (len(seeds) if seeds else 1) * args.target_sessions_per_area
    challenge_count = PRESET_CHALLENGERS[args.preset]
    max_steps = max(
        2500,
        int(
            args.target_sessions_per_area
            * challenge_count
            * (args.challenger_timeout_seconds * 50 + 100)
            * 2
        ),
    )
    env = None
    nav_hidden: dict[int, torch.Tensor] = {}
    memory_age: dict[int, int] = {}
    combat_hidden: dict[int, torch.Tensor] = {}
    lifecycle: dict[int, dict[str, int]] = {}
    saturation = {"mu": [], "sampled": [], "env": []}
    visible_count = 0
    visible_mismatch = 0
    aim_shoot_action_mismatches = 0
    safety_interventions = 0
    safety_layer = MapIndependentSafetyLayer(args.wall_flip_interval)
    tactical_handoff = FairTacticalHandoff()
    tactical_handoff_interventions = 0
    hidden_decisions = 0
    hidden_stuck_decisions = 0
    collision_runs: dict[int, int] = {}
    collision_loop_latched: set[int] = set()
    hidden_collision_loop_episodes = 0
    action_digest = hashlib.sha256()
    visible_action_digest = hashlib.sha256()
    visible_fire_actions = 0
    decisions_sent = 0
    hidden_reset_count = 0
    hidden_state_leakage_events = 0
    finite_violation_count = 0
    failure = ""
    started = time.perf_counter()
    try:
        env = UnityEnvironment(
            file_name=args.env_path,
            worker_id=0,
            base_port=args.base_port,
            seed=args.seed,
            side_channels=[engine],
            no_graphics=not args.rendered,
            additional_args=(
                ["--phase5-rendered-smoke", "-logFile", str(paths["unity"])]
                if args.rendered
                else ["--phase5-headless", "-logFile", str(paths["unity"])]
            ),
            timeout_wait=args.timeout,
        )
        env.reset()
        behavior_names = list(env.behavior_specs)
        if len(behavior_names) != 1:
            raise RuntimeError(f"expected one behavior, got {behavior_names}")
        behavior = behavior_names[0]
        action_spec = env.behavior_specs[behavior].action_spec
        for _ in range(max_steps):
            decisions, terminals = env.get_steps(behavior)
            for raw_id in terminals.agent_id:
                agent_id = int(raw_id)
                state = lifecycle.setdefault(agent_id, {"session": 0, "challenge": 0})
                state["challenge"] += 1
                nav_hidden.pop(agent_id, None)
                memory_age.pop(agent_id, None)
                reacquisition_search.reset(agent_id)
                combat_hidden.pop(agent_id, None)
                safety_layer.reset(agent_id)
                tactical_handoff.reset(agent_id)
                if peeker is not None:
                    peeker.reset(agent_id)
                if visible_tactic is not None:
                    visible_tactic.reset(agent_id)
                hidden_reset_count += 1
                if agent_id in nav_hidden or agent_id in combat_hidden:
                    hidden_state_leakage_events += 1
                collision_runs.pop(agent_id, None)
                collision_loop_latched.discard(agent_id)
                if state["challenge"] >= challenge_count:
                    state["challenge"] = 0
                    state["session"] += 1
            if len(decisions):
                ids = [int(value) for value in decisions.agent_id]
                for agent_id in ids:
                    lifecycle.setdefault(agent_id, {"session": 0, "challenge": 0})
                local45, actor = split_dual_observations(decisions.obs)
                combat_actions = action_from_distribution(
                    combat, normalizer, combat_device, local45, ids, combat_hidden
                )
                actor_tensor = torch.as_tensor(
                    actor, dtype=torch.float32, device=device
                ).unsqueeze(1)
                if args.preset == "reacquire_search" and memory_steps > 0:
                    for index, agent_id in enumerate(ids):
                        memory_age[agent_id] = (
                            0
                            if actor[index, LOS_INDEX] > 0.5
                            else memory_age.get(agent_id, 0) + 1
                        )
                        if memory_age[agent_id] >= memory_steps:
                            nav_hidden.pop(agent_id, None)
                            memory_age[agent_id] = 0
                nav_batch_hidden = torch.cat(
                    [
                        nav_hidden.get(agent_id, navigator.initial_hidden(1, device))
                        for agent_id in ids
                    ],
                    dim=1,
                )
                with torch.no_grad():
                    prediction, next_hidden = navigator(actor_tensor, nav_batch_hidden)
                    nav_action, sat = stochastic_action(
                        prediction,
                        args.mode == "stochastic",
                    )
                finite_violation_count += int(
                    not all(
                        torch.isfinite(value).all().item()
                        for value in (*prediction.values(), next_hidden, nav_action)
                    )
                )
                for index, agent_id in enumerate(ids):
                    nav_hidden[agent_id] = next_hidden[:, index : index + 1].detach()
                navigation = np.zeros((len(ids), 8), dtype=np.float32)
                navigation[:, :4] = nav_action[:, 0].cpu().numpy()
                if (
                    args.preset == "reacquire_search"
                    or (goal7_reacquisition and args.preset.startswith("goal7_"))
                ) and reacquire_config:
                    navigation, _ = reacquisition_search.apply(actor, navigation, ids)
                if args.disable_safety_layer:
                    intervention = np.zeros(len(ids), dtype=np.bool_)
                else:
                    navigation, intervention = safety_layer.apply(
                        actor, navigation, ids
                    )
                safety_interventions += int(intervention.sum())
                if args.mode == "combat_baseline":
                    applied = combat_actions
                elif visible_tactic is not None:
                    applied = compose_actions(actor, navigation, combat_actions)
                    applied, _ = visible_tactic.apply(actor, applied, ids)
                elif peeker is not None and args.preset.startswith(
                    ("peek_p", "goal7_")
                ):
                    applied, _, _, _ = peeker.apply(
                        actor,
                        navigation,
                        combat_actions,
                        ids,
                    )
                else:
                    applied = compose_actions(actor, navigation, combat_actions)
                    if (
                        args.preset not in {"duel_probe", "ppo_h0"}
                        and not disable_tactical_handoff
                    ):
                        applied, tactical_intervention = tactical_handoff.apply(
                            actor, applied, ids
                        )
                        tactical_handoff_interventions += int(
                            tactical_intervention.sum()
                        )
                visible = actor[:, LOS_INDEX] > 0.5
                hidden_mask = ~visible
                hidden_decisions += int(hidden_mask.sum())
                hidden_stuck_decisions += int((actor[hidden_mask, 26] > 0.60).sum())
                for index, agent_id in enumerate(ids):
                    collision_stuck = (
                        hidden_mask[index]
                        and actor[index, 28] > 0.60
                        and actor[index, 26] > 0.60
                    )
                    collision_runs[agent_id] = (
                        collision_runs.get(agent_id, 0) + 1 if collision_stuck else 0
                    )
                    if (
                        collision_runs[agent_id] >= 50
                        and agent_id not in collision_loop_latched
                    ):
                        collision_loop_latched.add(agent_id)
                        hidden_collision_loop_episodes += 1
                visible_count += int(visible.sum())
                visible_mismatch += int(
                    np.any(applied[visible] != combat_actions[visible], axis=1).sum()
                )
                aim_shoot_action_mismatches += int(
                    np.any(
                        applied[visible, 2:8] != combat_actions[visible, 2:8],
                        axis=1,
                    ).sum()
                )
                visible_action_digest.update(
                    np.asarray(combat_actions[visible], dtype="<f4").tobytes()
                )
                visible_fire_actions += int((combat_actions[visible, 4] > 0.5).sum())
                saturation["mu"].append(sat["mu_saturation_rate"])
                saturation["sampled"].append(sat["sampled_saturation_rate"])
                saturation["env"].append(
                    float((np.abs(applied[:, :4]) >= 0.999).mean())
                )
                action_digest.update(np.asarray(applied, dtype="<f4").tobytes())
                env.set_actions(behavior, make_action_tuple(applied, action_spec))
                decisions_sent += len(ids)
            completed = sum(
                min(item["session"], args.target_sessions_per_area)
                for item in lifecycle.values()
            )
            if (
                len(read_jsonl(paths["trace"])) >= expected_sessions
                and completed >= expected_sessions
            ):
                break
            env.step()
        else:
            raise RuntimeError(f"evaluation exceeded max_steps={max_steps}")
    except Exception as exc:
        failure = str(exc)
    finally:
        if env is not None:
            env.close()

    for _ in range(100):
        if paths["audit"].exists() and paths["runtime"].exists():
            break
        time.sleep(0.05)
    rows = read_jsonl(paths["trace"])
    contact = sum(
        float(row.get("time_to_first_contact_seconds", -1)) >= 0 for row in rows
    )
    duration = sum(
        max(float(row.get("session_duration_seconds", 0)), 0) for row in rows
    )
    stuck_seconds = sum(max(float(row.get("stuck_seconds", 0)), 0) for row in rows)
    collision_loops = sum(
        float(row.get("side_collision_count", 0)) >= 25
        and float(row.get("stuck_seconds", 0))
        / max(float(row.get("session_duration_seconds", 0)), 1e-6)
        >= 0.03
        for row in rows
    )
    audit = json.loads(paths["audit"].read_text()) if paths["audit"].exists() else {}
    weapon_rows = read_jsonl(paths["weapon"])
    weapon_ownership_mismatch_count = sum(
        row.get("blocked_fire_reason") in {"NO_OWNER", "WRONG_OWNER"}
        for row in weapon_rows
    )
    unity_text = (
        paths["unity"].read_text(errors="replace") if paths["unity"].exists() else ""
    )
    weapon_fire_mismatch_count = unity_text.count("First post-reset shot failed")
    peek_metrics = (
        peeker.summary()
        if peeker is not None and args.preset.startswith(("peek_p", "goal7_"))
        else None
    )
    if peek_metrics is not None:
        actual_hit_events = int(
            sum(float(row.get("damage_inflicted", 0)) for row in rows) // 20
        )
        damage_positive = min(int(peek_metrics["completed_peeks"]), actual_hit_events)
        peek_metrics["damage_positive_peeks"] = damage_positive
        peek_metrics["damage_positive_peek_rate"] = damage_positive / max(
            1, int(peek_metrics["completed_peeks"])
        )
    result = {
        "schema_version": "phase5_navigator_live_eval_v001",
        "status": "PASS"
        if not failure
        and len(rows) == expected_sessions
        and audit.get("status") == "PASS"
        and finite_violation_count == 0
        and hidden_state_leakage_events == 0
        and weapon_ownership_mismatch_count + weapon_fire_mismatch_count == 0
        and (
            args.preset != "duel_probe"
            or visible_mismatch == 0
            or visible_tactic is not None
        )
        and (not args.preset.startswith("peek_p") or aim_shoot_action_mismatches == 0)
        else "FAIL",
        "failure_reason": failure,
        "mode": args.mode,
        "source_kind": args.source_kind,
        "rendered": args.rendered,
        "map_id": args.map_id if args.source_kind == "authored" else "",
        "split": args.split,
        "preset": args.preset,
        "seeds": seeds,
        "sessions": len(rows),
        "expected_sessions": expected_sessions,
        "contact_sessions": contact,
        "contact_rate": contact / len(rows) if rows else 0.0,
        "clear_rate": sum(
            int(row.get("session_clear_count", 0)) >= 1
            and int(row.get("challengers_completed", 0))
            == int(row.get("challengers_expected", -1))
            for row in rows
        )
        / max(len(rows), 1),
        "challenger_win_rate": sum(int(row.get("kills", 0)) for row in rows)
        / max(sum(int(row.get("challengers_expected", 1)) for row in rows), 1),
        "session_timeout_rate": sum(int(row.get("timeouts", 0)) > 0 for row in rows)
        / max(len(rows), 1),
        "stuck_rate": (
            hidden_stuck_decisions / hidden_decisions if hidden_decisions else 1.0
        ),
        "collision_loop_episodes": hidden_collision_loop_episodes,
        "collision_loop_episode_rate": (
            hidden_collision_loop_episodes / len(rows) if rows else 1.0
        ),
        "hidden_decisions": hidden_decisions,
        "hidden_stuck_decisions": hidden_stuck_decisions,
        "session_stuck_rate_diagnostic": stuck_seconds / duration if duration else 1.0,
        "session_collision_proxy_episodes_diagnostic": collision_loops,
        "visible_decisions": visible_count,
        "visible_combat_action_mismatches": visible_mismatch,
        "aim_shoot_action_mismatches": aim_shoot_action_mismatches,
        "peek_metrics": peek_metrics,
        "peeker_config": peeker_config,
        "phase6_visible_tactic": visible_tactic_config,
        "phase6_goal7_reacquisition": goal7_reacquisition,
        "phase6_disable_tactical_handoff": disable_tactical_handoff,
        "per_map_cover_anchors": False,
        "peek_point_files": False,
        "tactical_handoff_interventions": tactical_handoff_interventions,
        "safety_layer_interventions": safety_interventions,
        "safety_layer_intervention_rate": (
            safety_interventions / decisions_sent if decisions_sent else 0.0
        ),
        "mu_saturation_rate": float(np.mean(saturation["mu"]))
        if saturation["mu"]
        else 1.0,
        "sampled_saturation_rate": float(np.mean(saturation["sampled"]))
        if saturation["sampled"]
        else 1.0,
        "environment_applied_saturation_rate": float(np.mean(saturation["env"]))
        if saturation["env"]
        else 1.0,
        "decisions_sent": decisions_sent,
        "applied_action_sha256": action_digest.hexdigest(),
        "visible_combat_action_sha256": visible_action_digest.hexdigest(),
        "visible_fire_actions": visible_fire_actions,
        "finite_violation_count": finite_violation_count,
        "hidden_reset_count": hidden_reset_count,
        "hidden_state_leakage_events": hidden_state_leakage_events,
        "weapon_ownership_mismatch_count": weapon_ownership_mismatch_count,
        "weapon_fire_mismatch_count": weapon_fire_mismatch_count,
        "manual_input_events": int(audit.get("manual_input_events", -1)),
        "human_controller_count": int(audit.get("human_controller_count", -1)),
        "navmesh_oracle_enabled": False,
        "actor_received_map_id": False,
        "actor_received_absolute_coordinates": False,
        "actor_received_waypoints": False,
        "actor_received_baked_tactical_graph": False,
        "damage_inflicted": sum(float(row.get("damage_inflicted", 0)) for row in rows),
        "damage_received": sum(float(row.get("damage_received", 0)) for row in rows),
        "kills": sum(int(row.get("kills", 0)) for row in rows),
        "deaths": sum(int(row.get("deaths", 0)) for row in rows),
        "timeouts": sum(int(row.get("timeouts", 0)) for row in rows),
        "los_lost_events": sum(int(row.get("los_lost_events", 0)) for row in rows),
        "reacquired_events": sum(int(row.get("reacquired_events", 0)) for row in rows),
        "reacquisition_rate": (
            sum(int(row.get("reacquired_events", 0)) for row in rows)
            / max(1, sum(int(row.get("los_lost_events", 0)) for row in rows))
        ),
        "reacquisition_eligible_sessions": sum(
            int(row.get("los_lost_events", 0)) > 0 for row in rows
        ),
        "scenario_coverage": sorted(
            {challenger for row in rows for challenger in row.get("challenger_ids", [])}
        ),
        "reacquire_config": reacquire_config,
        "stale_location_loop_sessions": sum(
            bool(row.get("stale_location_loop", False)) for row in rows
        ),
        "stale_location_loop_rate": (
            sum(bool(row.get("stale_location_loop", False)) for row in rows) / len(rows)
            if rows
            else 1.0
        ),
        "search_event_counts": {
            name: sum(
                sum(
                    event.get("event_name") == name
                    for event in row.get("search_events", [])
                )
                for row in rows
            )
            for name in (
                "LOS_LOST",
                "LAST_SEEN_REACHED",
                "LOCAL_SEARCH_STARTED",
                "REACQUIRED",
                "SEARCH_GOAL_STALE",
                "UNSTUCK_TRIGGERED",
            )
        },
        "navigator_sha256": sha256_file(args.navigator),
        "navigator_update": navigator_payload["update"],
        "elapsed_seconds": time.perf_counter() - started,
        "sessions_path": str(paths["trace"]),
        "autonomous_audit": audit,
    }
    (output / "evaluation_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
