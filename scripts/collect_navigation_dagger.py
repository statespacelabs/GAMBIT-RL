#!/usr/bin/env python3
"""Eight-GPU Phase 5 generic-teacher DAgger collection driver."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.autonomous_combat_baseline import (  # noqa: E402
    NORMALIZER_SHA256,
    PARENT_ID,
    PARENT_SHA256,
    PRESET_CHALLENGERS,
    make_action_tuple,
    read_jsonl,
)

READY_PREREQUISITE = "PHASE5_AUTONOMOUS_HARNESS_READY"
LABEL_SCHEMA = "phase5_teacher_label_v001"
ROW_SCHEMA = "phase5_dagger_rollout_v001"
ACTOR_SCHEMA = "phase5_actor_obs_v001"
ROUND_SPECS = (
    ("teacher_100", 1.00),
    ("teacher_050", 0.50),
    ("teacher_025", 0.25),
    ("teacher_010", 0.10),
    ("learner_labels", 0.00),
)
MODE_NAMES = (
    "DIRECT_PURSUIT",
    "FOLLOW_PATH_CORNER",
    "WALL_FOLLOW_LEFT",
    "WALL_FOLLOW_RIGHT",
    "ESCAPE_STUCK",
    "COMBAT_HANDOFF",
    "REACQUIRE_LAST_SEEN",
)
FORBIDDEN_ROW_KEYS = {
    "map_id",
    "map_name",
    "absolute_world_position",
    "path_nodes",
    "waypoints",
    "exact_hidden_target_position",
    "exact_hidden_target_relative_vector",
}


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def action_from_distribution(
    policy: Any,
    normalizer: Any,
    device: Any,
    batch: np.ndarray,
    ids: list[int],
    hidden: dict[int, Any],
) -> np.ndarray:
    import torch

    observations = torch.as_tensor(
        batch,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    normalized = normalizer.normalize_tensor(observations)
    hidden_batch = torch.cat(
        [
            hidden.get(agent_id, policy.init_hidden(1, device))
            for agent_id in ids
        ],
        dim=1,
    )
    with torch.no_grad():
        distribution, values, next_hidden = policy(normalized, hidden_batch)
        actions = distribution.mode[0].squeeze(1).detach().cpu().numpy()
    if (
        not torch.isfinite(values).all()
        or not torch.isfinite(next_hidden).all()
        or not np.isfinite(actions).all()
    ):
        raise RuntimeError("frozen combat expert inference produced NaN/Inf")
    for index, agent_id in enumerate(ids):
        hidden[agent_id] = next_hidden[:, index : index + 1].detach()
    return np.asarray(actions, dtype=np.float32)


def validate_teacher_rows(
    rows: list[dict[str, Any]],
    control_probability: float,
) -> dict[str, Any]:
    failures: list[str] = []
    modes = Counter()
    intervention_rows = 0
    los_rows = 0
    finite_violations = 0
    for index, row in enumerate(rows):
        missing = FORBIDDEN_ROW_KEYS & set(row)
        if missing:
            failures.append(f"row {index} forbidden keys: {sorted(missing)}")
        if row.get("schema_version") != ROW_SCHEMA:
            failures.append(f"row {index} rollout schema mismatch")
        if row.get("label_schema") != LABEL_SCHEMA:
            failures.append(f"row {index} label schema mismatch")
        if row.get("student_observation_schema") != ACTOR_SCHEMA:
            failures.append(f"row {index} actor schema mismatch")
        arrays = {
            "actor_observation": 231,
            "policy_action": 8,
            "teacher_action": 8,
            "executed_action": 8,
            "teacher_move": 2,
            "navigation_look_bias": 2,
            "teacher_next_corner_direction_local": 3,
        }
        for key, size in arrays.items():
            value = np.asarray(row.get(key, []), dtype=np.float64)
            if value.shape != (size,):
                failures.append(f"row {index} {key} shape {value.shape}")
            finite_violations += int((~np.isfinite(value)).sum())
        gate = float(row.get("combat_navigation_blend_gate", math.nan))
        if not math.isfinite(gate) or not 0.0 <= gate <= 1.0:
            failures.append(f"row {index} invalid blend gate")
        mode = str(row.get("teacher_mode", ""))
        if mode not in MODE_NAMES:
            failures.append(f"row {index} unknown teacher mode {mode!r}")
        modes[mode] += 1
        if row.get("teacher_controlled") is True:
            intervention_rows += 1
        if row.get("exact_los") is True:
            los_rows += 1
            policy = np.asarray(row["policy_action"], dtype=np.float32)
            executed = np.asarray(row["executed_action"], dtype=np.float32)
            if not np.array_equal(policy, executed):
                failures.append(
                    f"row {index} did not hand LOS control to frozen combat expert"
                )
        if (
            row.get("weapon_failure")
            or row.get("owner_failure")
            or row.get("observation_failure")
        ):
            failures.append(f"row {index} runtime failure flag")
    if finite_violations:
        failures.append(f"non-finite array values: {finite_violations}")
    if not rows:
        failures.append("no teacher rollout rows")
    if control_probability == 1.0 and rows:
        hidden = [row for row in rows if not row.get("exact_los")]
        if hidden and not all(row.get("teacher_controlled") is True for row in hidden):
            failures.append("teacher_100 failed to control every hidden row")
    if control_probability == 0.0 and intervention_rows != 0:
        failures.append("learner_labels round contains teacher interventions")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures[:100],
        "rows": len(rows),
        "intervention_rows": intervention_rows,
        "los_rows": los_rows,
        "mode_counts": dict(sorted(modes.items())),
        "finite_violations": finite_violations,
    }


def write_npz(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)

    def stack(key: str, width: int) -> np.ndarray:
        if not rows:
            return np.empty((0, width), dtype=np.float32)
        return np.asarray([row[key] for row in rows], dtype=np.float32)

    mode_to_index = {name: index for index, name in enumerate(MODE_NAMES)}
    arrays = {
        "actor_obs": stack("actor_observation", 231),
        "policy_action": stack("policy_action", 8),
        "teacher_action": stack("teacher_action", 8),
        "executed_action": stack("executed_action", 8),
        "teacher_move": stack("teacher_move", 2),
        "navigation_look_bias": stack("navigation_look_bias", 2),
        "teacher_next_corner_direction_local": stack(
            "teacher_next_corner_direction_local", 3
        ),
        "teacher_tactical_mode": np.asarray(
            [mode_to_index[row["teacher_mode"]] for row in rows],
            dtype=np.int8,
        ),
        "combat_navigation_blend_gate": np.asarray(
            [row["combat_navigation_blend_gate"] for row in rows],
            dtype=np.float32,
        ),
        "teacher_next_corner_valid": np.asarray(
            [row["teacher_next_corner_valid"] for row in rows],
            dtype=np.bool_,
        ),
        "teacher_controlled": np.asarray(
            [row["teacher_controlled"] for row in rows],
            dtype=np.bool_,
        ),
        "exact_los": np.asarray(
            [row["exact_los"] for row in rows],
            dtype=np.bool_,
        ),
        "contact": np.asarray(
            [row["contact"] for row in rows],
            dtype=np.bool_,
        ),
        "stuck": np.asarray(
            [row["stuck"] for row in rows],
            dtype=np.bool_,
        ),
        "area_id": np.asarray([row["area_id"] for row in rows], dtype=np.int16),
        "episode_id": np.asarray(
            [row["episode_id"] for row in rows],
            dtype=np.int32,
        ),
        "fixed_step": np.asarray(
            [row["fixed_step"] for row in rows],
            dtype=np.int32,
        ),
        "control_probability": np.asarray(
            [row["control_probability"] for row in rows],
            dtype=np.float32,
        ),
    }
    np.savez_compressed(path, **arrays)
    reopened = np.load(path)
    if set(reopened.files) != set(arrays):
        raise RuntimeError("NPZ array schema mismatch after write")
    if reopened["actor_obs"].shape != (len(rows), 231):
        raise RuntimeError("NPZ actor observation shape mismatch")
    return {
        "path": str(path),
        "sha256": file_hash(path),
        "samples": len(rows),
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for key, value in arrays.items()
        },
    }


def configure_environment(args: argparse.Namespace, output: Path) -> dict[str, Path]:
    paths = {
        "teacher_trace": output / "teacher_rollout.jsonl",
        "teacher_summary": output / "teacher_summary.json",
        "autonomous_trace": output / "autonomous_sessions.jsonl",
        "autonomous_audit": output / "autonomous_audit.json",
        "runtime_summary": output / "runtime_summary.json",
        "layout_audit": output / "layout_audit.json",
        "map_telemetry": output / "authored_map_telemetry.jsonl",
        "unity_log": output / "unity.log",
    }
    for path in paths.values():
        if path.exists():
            path.unlink()
    common = {
        "GAME_MODE": "HumanVsScripted",
        "PLAYER_B_BOT_MODE": "Idle",
        "NUM_AREAS": str(args.areas),
        "ENABLE_VISUAL_OBS": "0",
        "PHASE5_TELEMETRY_SCHEMA": "phase3v2_c_local45",
        "PHASE5_AUTONOMOUS_SESSION": "1",
        "PHASE5_AUTONOMOUS_PRESET": args.preset,
        "PHASE5_AUTONOMOUS_RUN_ID": args.run_id,
        "PHASE5_AUTONOMOUS_TRACE_PATH": str(paths["autonomous_trace"].resolve()),
        "PHASE5_AUTONOMOUS_AUDIT_PATH": str(paths["autonomous_audit"].resolve()),
        "PHASE5_AUTONOMOUS_TARGET_SESSIONS_PER_AREA": str(
            args.target_sessions_per_area
        ),
        "PHASE5_AUTONOMOUS_CHALLENGER_TIMEOUT_SECONDS": str(
            args.challenger_timeout_seconds
        ),
        "PHASE5_GENERIC_TEACHER": "1",
        "PHASE5_TEACHER_LABEL_SCHEMA": LABEL_SCHEMA,
        "PHASE5_TEACHER_TRACE_PATH": str(paths["teacher_trace"].resolve()),
        "PHASE5_TEACHER_SUMMARY_PATH": str(paths["teacher_summary"].resolve()),
        "PHASE5_TEACHER_CONTROL_PROBABILITY": str(args.control_probability),
        "PHASE5_TEACHER_SEED": str(args.seed),
        "PHASE5_DAGGER_ROUND": args.round_id,
        "PHASE5_DAGGER_STORE_FRAMES": "0",
        "PHASE5_ENABLE_NAVMESH_ORACLE": "1",
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
        "PHASE5_RUNTIME_SUMMARY_PATH": str(paths["runtime_summary"].resolve()),
        "DEBUG_LOG_LEVEL": "summary",
    }
    os.environ.update(common)
    if args.source_kind == "procedural":
        manifest = Path(args.manifest).resolve()
        os.environ.update(
            {
                "PHASE5_PROCEDURAL_LAYOUT": "1",
                "PHASE5_LAYOUT_MANIFEST": str(manifest),
                "PHASE5_LAYOUT_MANIFEST_SHA256": file_hash(manifest),
                "PHASE5_LAYOUT_SPLIT": args.split,
                "PHASE5_LAYOUT_SEEDS": args.seeds,
                "PHASE5_LAYOUT_AUDIT_PATH": str(
                    paths["layout_audit"].resolve()
                ),
                "PHASE4_MAP_CONTROL_ENABLED": "0",
                "PHASE4_4_ENABLE_SPAWN_BUCKETS": "0",
            }
        )
        for key in (
            "PHASE4_MAP_ID",
            "PHASE4_MAP_MIX_PROFILE",
            "PHASE4_MAP_SEED",
            "PHASE4_WRITE_MAP_TELEMETRY",
            "PHASE4_MAP_TELEMETRY_PATH",
            "PHASE4_4_REQUIRE_OBSTACLE_BETWEEN",
            "PHASE4_4_REQUIRE_INITIAL_LOS",
        ):
            os.environ.pop(key, None)
    else:
        os.environ.update(
            {
                "PHASE5_PROCEDURAL_LAYOUT": "0",
                "PHASE4_MAP_CONTROL_ENABLED": "1",
                "PHASE4_MAP_ID": args.map_id,
                "PHASE4_MAP_MIX_PROFILE": args.map_profile,
                "PHASE4_MAP_SEED": str(args.map_seed),
                "PHASE4_WRITE_MAP_TELEMETRY": "1",
                "PHASE4_MAP_TELEMETRY_PATH": str(
                    paths["map_telemetry"].resolve()
                ),
                "PHASE4_4_ENABLE_SPAWN_BUCKETS": "0",
                "PHASE4_4_SPAWN_BUCKET": args.spawn_bucket,
                "PHASE4_4_SPAWN_SEED": str(args.seed),
                "PHASE4_4_TRIAL_MAX_SECONDS": "600",
                "PHASE4_4_REQUIRE_OBSTACLE_BETWEEN": "0",
            }
        )
        for key in (
            "PHASE5_LAYOUT_MANIFEST",
            "PHASE5_LAYOUT_MANIFEST_SHA256",
            "PHASE5_LAYOUT_SPLIT",
            "PHASE5_LAYOUT_SEEDS",
            "PHASE5_LAYOUT_AUDIT_PATH",
            "PHASE4_4_REQUIRE_INITIAL_LOS",
        ):
            os.environ.pop(key, None)
    return paths


def run_job(args: argparse.Namespace) -> int:
    import torch
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.engine_configuration_channel import (
        EngineConfigurationChannel,
    )
    from scripts.combat_runtime import (
        RuntimeConfig,
        load_policy_and_normalizer,
        validate_specs,
    )

    prerequisite = json.loads(Path(args.prerequisite).read_text())
    if prerequisite.get("status") != READY_PREREQUISITE:
        raise RuntimeError("Goal 5 prerequisite is not ready")
    if file_hash(args.checkpoint) != PARENT_SHA256:
        raise RuntimeError("strongest combat parent hash mismatch")
    if file_hash(args.normalizer) != NORMALIZER_SHA256:
        raise RuntimeError("frozen local45 normalizer hash mismatch")
    if args.preset not in PRESET_CHALLENGERS:
        raise RuntimeError(f"unknown autonomous preset {args.preset}")
    if not 0.0 <= args.control_probability <= 1.0:
        raise RuntimeError("teacher control probability outside [0,1]")
    if args.source_kind == "procedural":
        seeds = [int(value) for value in args.seeds.split(",") if value]
        if len(seeds) != args.areas:
            raise RuntimeError("procedural seed count must match areas")
    elif args.areas != 1:
        raise RuntimeError("authored collection requires exactly one area")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = configure_environment(args, output)
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
        num_areas=args.areas,
    )
    policy, normalizer, _, _, _, device = load_policy_and_normalizer(policy_cfg)
    policy.set_ppo_mode(training=False)
    engine = EngineConfigurationChannel()
    engine.set_configuration_parameters(
        time_scale=args.time_scale,
        target_frame_rate=-1,
        capture_frame_rate=0,
    )

    expected_sessions = args.areas * args.target_sessions_per_area
    challenge_count = PRESET_CHALLENGERS[args.preset]
    lifecycle: dict[int, dict[str, int]] = {}
    hidden: dict[int, torch.Tensor] = {}
    hidden_resets = 0
    decisions_sent = 0
    failure = ""
    env = None
    started = time.perf_counter()
    max_steps = max(
        4000,
        int(
            args.target_sessions_per_area
            * challenge_count
            * (args.challenger_timeout_seconds * 50 + 150)
            * 2
        ),
    )
    try:
        env = UnityEnvironment(
            file_name=args.env_path,
            worker_id=0,
            base_port=args.base_port,
            seed=args.seed,
            side_channels=[engine],
            no_graphics=True,
            additional_args=[
                "--phase5-headless",
                "-job-worker-count",
                str(args.job_worker_count),
                "-logFile",
                str(paths["unity_log"].resolve()),
            ],
            timeout_wait=args.timeout,
        )
        env.reset()
        behavior, action_spec, specs = validate_specs(env)
        for _ in range(max_steps):
            decisions, terminals = env.get_steps(behavior)
            for agent_raw in terminals.agent_id:
                agent_id = int(agent_raw)
                state = lifecycle.setdefault(
                    agent_id,
                    {"session": 0, "challenge": 0},
                )
                state["challenge"] += 1
                hidden.pop(agent_id, None)
                hidden_resets += 1
                if state["challenge"] >= challenge_count:
                    state["challenge"] = 0
                    state["session"] += 1

            if len(decisions):
                ids = [int(value) for value in decisions.agent_id]
                for agent_id in ids:
                    lifecycle.setdefault(
                        agent_id,
                        {"session": 0, "challenge": 0},
                    )
                batch = np.asarray(decisions.obs[0], dtype=np.float32)
                if batch.shape != (len(ids), 45) or not np.isfinite(batch).all():
                    raise RuntimeError(
                        f"frozen combat transport invalid shape={batch.shape}"
                    )
                actions = action_from_distribution(
                    policy,
                    normalizer,
                    device,
                    batch,
                    ids,
                    hidden,
                )
                env.set_actions(
                    behavior,
                    make_action_tuple(actions, action_spec),
                )
                decisions_sent += len(ids)

            completed = sum(
                min(state["session"], args.target_sessions_per_area)
                for state in lifecycle.values()
            )
            session_rows = read_jsonl(paths["autonomous_trace"])
            if len(session_rows) >= expected_sessions and completed >= expected_sessions:
                break
            env.step()
        else:
            raise RuntimeError(
                f"job exceeded max_steps={max_steps}; "
                f"sessions={len(read_jsonl(paths['autonomous_trace']))}/"
                f"{expected_sessions}"
            )
    except Exception as exc:
        failure = str(exc)
    finally:
        if env is not None:
            env.close()

    for _ in range(100):
        if (
            paths["teacher_summary"].exists()
            and paths["autonomous_audit"].exists()
            and paths["runtime_summary"].exists()
        ):
            break
        time.sleep(0.05)

    teacher_rows = read_jsonl(paths["teacher_trace"])
    row_audit = validate_teacher_rows(
        teacher_rows,
        args.control_probability,
    )
    npz = write_npz(teacher_rows, output / "teacher_rollout.npz")
    teacher_summary = (
        json.loads(paths["teacher_summary"].read_text())
        if paths["teacher_summary"].exists()
        else {}
    )
    autonomous_rows = read_jsonl(paths["autonomous_trace"])
    autonomous_audit = (
        json.loads(paths["autonomous_audit"].read_text())
        if paths["autonomous_audit"].exists()
        else {}
    )
    runtime_summary = (
        json.loads(paths["runtime_summary"].read_text())
        if paths["runtime_summary"].exists()
        else {}
    )
    layout_audit = (
        json.loads(paths["layout_audit"].read_text())
        if paths["layout_audit"].exists()
        else {}
    )
    map_rows = read_jsonl(paths["map_telemetry"])
    contacts = sum(
        float(row.get("time_to_first_contact_seconds", -1)) >= 0
        for row in autonomous_rows
    )
    mode_counts = teacher_summary.get("mode_counts", [])
    status = "PASS" if (
        not failure
        and row_audit["status"] == "PASS"
        and len(autonomous_rows) == expected_sessions
        and autonomous_audit.get("status") == "PASS"
        and teacher_summary.get("status") == "PASS"
        and runtime_summary.get("status") == "PASS"
        and runtime_summary.get("headless_audit", {}).get("status") == "PASS"
        and hidden_resets >= expected_sessions * challenge_count
        and teacher_summary.get("frame_count") == 0
        and teacher_summary.get("contains_world_coordinates") is False
        and teacher_summary.get("contains_map_identifiers") is False
        and teacher_summary.get("contains_path_nodes_or_waypoints") is False
        and teacher_summary.get("weapon_failures") == 0
        and teacher_summary.get("owner_failures") == 0
        and teacher_summary.get("observation_failures") == 0
        and (
            args.source_kind != "procedural"
            or layout_audit.get("status") == "PASS"
        )
        and (
            args.source_kind != "authored"
            or any(row.get("status") == "PASS" for row in map_rows)
        )
    ) else "FAIL"
    result = {
        "schema_version": "phase5_dagger_collection_job_v001",
        "status": status,
        "failure_reason": failure,
        "run_id": args.run_id,
        "round_id": args.round_id,
        "control_probability": args.control_probability,
        "gpu": args.gpu,
        "source_kind": args.source_kind,
        "split": args.split,
        "family": args.family,
        "layout_seeds": [
            int(value) for value in args.seeds.split(",") if value
        ],
        "authored_map_reference": (
            {
                "map_id": args.map_id,
                "map_profile": args.map_profile,
                "map_seed": args.map_seed,
            }
            if args.source_kind == "authored"
            else None
        ),
        "preset": args.preset,
        "expected_sessions": expected_sessions,
        "complete_sessions": len(autonomous_rows),
        "contact_sessions": contacts,
        "contact_rate": contacts / len(autonomous_rows)
        if autonomous_rows
        else 0.0,
        "hidden_resets": hidden_resets,
        "decisions_sent": decisions_sent,
        "samples": len(teacher_rows),
        "row_audit": row_audit,
        "teacher_summary": teacher_summary,
        "npz": npz,
        "mode_counts": mode_counts,
        "build_executable_sha256": file_hash(args.env_path),
        "build_managed_assembly_sha256": file_hash(
            Path(args.env_path).resolve().parent
            / (Path(args.env_path).name.split(".x86_64")[0] + "_Data")
            / "Managed/Assembly-CSharp.dll"
        ),
        "collector_sha256": file_hash(Path(__file__).resolve()),
        "time_scale": args.time_scale,
        "challenger_timeout_seconds": args.challenger_timeout_seconds,
        "job_worker_count": args.job_worker_count,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "normalizer_sha256": NORMALIZER_SHA256,
        "wall_seconds": time.perf_counter() - started,
        "specs": specs if "specs" in locals() else {},
    }
    (output / "collection_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": status,
                "run_id": args.run_id,
                "samples": len(teacher_rows),
                "sessions": len(autonomous_rows),
                "contact_rate": result["contact_rate"],
                "failure": failure,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if status == "PASS" else 1


def hidden_family_seeds(
    document: dict[str, Any],
    family: str,
    count: int,
) -> list[int]:
    values = [
        int(item["seed"])
        for item in document["layouts"]
        if item["family"] == family
        and item["requested_initial_los"] is False
    ]
    if len(values) < count:
        raise RuntimeError(
            f"need {count} hidden seeds for {family}, found {len(values)}"
        )
    return values[:count]


def build_shards(args: argparse.Namespace) -> list[dict[str, Any]]:
    layouts = Path(args.layouts_dir).resolve()
    train = json.loads((layouts / "TRAIN_LAYOUTS.json").read_text())
    validation = json.loads((layouts / "VALIDATION_LAYOUTS.json").read_text())
    count = args.areas
    families = (
        "open_scattered_cover",
        "parallel_corridors",
        "rooms_doorways",
        "pillars_alternating_cover",
        "narrow_wide_transitions",
    )
    shards = []
    for gpu, family in enumerate(families):
        shards.append(
            {
                "gpu": gpu,
                "shard_id": f"gpu{gpu}_{family}",
                "source_kind": "procedural",
                "manifest": str(layouts / "TRAIN_LAYOUTS.json"),
                "split": "train",
                "family": family,
                "seeds": hidden_family_seeds(train, family, count),
                "areas": count,
                "preset": "hunt_probe",
            }
        )
    shards.append(
        {
            "gpu": 5,
            "shard_id": "gpu5_authored_ascent",
            "source_kind": "authored",
            "manifest": "",
            "split": "authored",
            "family": "authored_demo",
            "seeds": [],
            "areas": 1,
            "preset": "demo_5",
            "map_id": "arena_ascent_v1",
            "map_profile": "original_only",
            "map_seed": args.seed + 5000,
            "spawn_bucket": "search_destroy",
        }
    )
    moving_seeds = [
        int(item["seed"])
        for item in train["layouts"]
        if item["parameters"]["target_motion"]
        in {"strafe", "alternating_cover", "patrol"}
    ][:count]
    if len(moving_seeds) < count:
        raise RuntimeError("not enough moving-target training layouts")
    shards.append(
        {
            "gpu": 6,
            "shard_id": "gpu6_moving_kiting",
            "source_kind": "procedural",
            "manifest": str(layouts / "TRAIN_LAYOUTS.json"),
            "split": "train",
            "family": "moving_kiting_mix",
            "seeds": moving_seeds,
            "areas": count,
            "preset": "endurance_20",
        }
    )
    validation_hidden = [
        int(item["seed"])
        for item in validation["layouts"]
        if item["requested_initial_los"] is False
    ][:count]
    if len(validation_hidden) < count:
        raise RuntimeError("not enough hidden validation layouts")
    shards.append(
        {
            "gpu": 7,
            "shard_id": "gpu7_validation_heldback",
            "source_kind": "procedural",
            "manifest": str(layouts / "VALIDATION_LAYOUTS.json"),
            "split": "validation",
            "family": "heldback_validation_mix",
            "seeds": validation_hidden,
            "areas": count,
            "preset": "hunt_probe",
        }
    )
    return shards


def run_shard(args: argparse.Namespace) -> int:
    shard = json.loads(Path(args.shard_config).read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    returncodes = {}
    for index, (round_id, probability) in enumerate(ROUND_SPECS):
        round_dir = output / round_id
        round_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run-job",
            "--prerequisite",
            args.prerequisite,
            "--env-path",
            args.env_path,
            "--checkpoint",
            args.checkpoint,
            "--normalizer",
            args.normalizer,
            "--output-dir",
            str(round_dir),
            "--run-id",
            f"{shard['shard_id']}:{round_id}",
            "--round-id",
            round_id,
            "--control-probability",
            str(probability),
            "--gpu",
            str(shard["gpu"]),
            "--device",
            args.device,
            "--source-kind",
            shard["source_kind"],
            "--manifest",
            shard.get("manifest", ""),
            "--split",
            shard["split"],
            "--family",
            shard["family"],
            "--seeds",
            ",".join(str(value) for value in shard.get("seeds", [])),
            "--areas",
            str(shard["areas"]),
            "--preset",
            shard["preset"],
            "--base-port",
            str(args.base_port + index * 2),
            "--seed",
            str(args.seed + index * 101),
            "--time-scale",
            str(args.time_scale),
            "--target-sessions-per-area",
            str(args.target_sessions_per_area),
            "--challenger-timeout-seconds",
            str(args.challenger_timeout_seconds),
            "--job-worker-count",
            str(args.job_worker_count),
            "--map-id",
            shard.get("map_id", "arena_ascent_v1"),
            "--map-profile",
            shard.get("map_profile", "original_only"),
            "--map-seed",
            str(shard.get("map_seed", args.seed)),
            "--spawn-bucket",
            shard.get("spawn_bucket", "search_destroy"),
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (round_dir / "driver.log").write_text(completed.stdout)
        returncodes[round_id] = completed.returncode
        result_path = round_dir / "collection_result.json"
        if result_path.exists():
            results.append(json.loads(result_path.read_text()))
    status = "PASS" if (
        len(results) == len(ROUND_SPECS)
        and all(code == 0 for code in returncodes.values())
        and all(result.get("status") == "PASS" for result in results)
    ) else "FAIL"
    summary = {
        "schema_version": "phase5_dagger_shard_v001",
        "status": status,
        "shard": shard,
        "rounds": results,
        "returncodes": returncodes,
        "samples": sum(result.get("samples", 0) for result in results),
        "complete_sessions": sum(
            result.get("complete_sessions", 0) for result in results
        ),
    }
    (output / "shard_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if status == "PASS" else 1


def launch_matrix(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shards = build_shards(args)
    plan = {
        "schema_version": "phase5_dagger_collection_plan_v001",
        "rounds": [
            {"round_id": name, "teacher_control_probability": probability}
            for name, probability in ROUND_SPECS
        ],
        "runtime": {
            "time_scale": args.time_scale,
            "challenger_timeout_seconds": args.challenger_timeout_seconds,
            "job_worker_count": args.job_worker_count,
            "cpu_cores_per_shard": args.cpu_cores_per_shard,
            "areas_per_procedural_shard": args.areas,
        },
        "shards": shards,
    }
    (output / "COLLECTION_PLAN.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    processes = []
    for shard in shards:
        shard_dir = output / shard["shard_id"]
        shard_dir.mkdir(parents=True, exist_ok=True)
        cpu_start = shard["gpu"] * args.cpu_cores_per_shard
        cpu_end = cpu_start + args.cpu_cores_per_shard - 1
        shard["cpu_affinity"] = list(range(cpu_start, cpu_end + 1))
        config_path = shard_dir / "shard_config.json"
        config_path.write_text(json.dumps(shard, indent=2, sort_keys=True) + "\n")
        log = (shard_dir / "shard_driver.log").open("w")
        command = [
            "taskset",
            "-c",
            f"{cpu_start}-{cpu_end}",
            sys.executable,
            str(Path(__file__).resolve()),
            "run-shard",
            "--prerequisite",
            args.prerequisite,
            "--env-path",
            args.env_path,
            "--checkpoint",
            args.checkpoint,
            "--normalizer",
            args.normalizer,
            "--shard-config",
            str(config_path),
            "--output-dir",
            str(shard_dir),
            "--base-port",
            str(args.base_port + shard["gpu"] * 100),
            "--seed",
            str(args.seed + shard["gpu"] * 1000),
            "--time-scale",
            str(args.time_scale),
            "--target-sessions-per-area",
            str(args.target_sessions_per_area),
            "--challenger-timeout-seconds",
            str(args.challenger_timeout_seconds),
            "--job-worker-count",
            str(args.job_worker_count),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(shard["gpu"])
        env["HIP_VISIBLE_DEVICES"] = str(shard["gpu"])
        env["ROCR_VISIBLE_DEVICES"] = str(shard["gpu"])
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((shard, process, log))
    returncodes = {}
    for shard, process, log in processes:
        returncodes[shard["shard_id"]] = process.wait()
        log.close()
    status = "PASS" if all(code == 0 for code in returncodes.values()) else "FAIL"
    summary = {
        "schema_version": "phase5_dagger_collection_matrix_v001",
        "status": status,
        "gpu_count": len({shard["gpu"] for shard in shards}),
        "runtime": plan["runtime"],
        "returncodes": returncodes,
        "shards": shards,
    }
    (output / "COLLECTION_RUN.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if status == "PASS" else 1


def add_job_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prerequisite", required=True)
    parser.add_argument("--env-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--control-probability", type=float, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--source-kind",
        choices=("procedural", "authored"),
        required=True,
    )
    parser.add_argument("--manifest", default="")
    parser.add_argument("--split", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--areas", type=int, required=True)
    parser.add_argument("--preset", choices=tuple(PRESET_CHALLENGERS), required=True)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--target-sessions-per-area", type=int, default=1)
    parser.add_argument("--challenger-timeout-seconds", type=float, default=6.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--job-worker-count", type=int, default=15)
    parser.add_argument("--map-id", default="arena_ascent_v1")
    parser.add_argument("--map-profile", default="original_only")
    parser.add_argument("--map-seed", type=int, default=56025)
    parser.add_argument("--spawn-bucket", default="search_destroy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    job = subparsers.add_parser("run-job")
    add_job_args(job)
    job.set_defaults(func=run_job)

    shard = subparsers.add_parser("run-shard")
    shard.add_argument("--prerequisite", required=True)
    shard.add_argument("--env-path", required=True)
    shard.add_argument("--checkpoint", required=True)
    shard.add_argument("--normalizer", required=True)
    shard.add_argument("--shard-config", required=True)
    shard.add_argument("--output-dir", required=True)
    shard.add_argument("--base-port", type=int, required=True)
    shard.add_argument("--seed", type=int, required=True)
    shard.add_argument("--time-scale", type=float, default=20.0)
    shard.add_argument("--target-sessions-per-area", type=int, default=1)
    shard.add_argument("--challenger-timeout-seconds", type=float, default=6.0)
    shard.add_argument("--device", default="cuda")
    shard.add_argument("--job-worker-count", type=int, default=15)
    shard.set_defaults(func=run_shard)

    launch = subparsers.add_parser("launch-matrix")
    launch.add_argument("--prerequisite", required=True)
    launch.add_argument("--env-path", required=True)
    launch.add_argument("--layouts-dir", required=True)
    launch.add_argument("--checkpoint", required=True)
    launch.add_argument("--normalizer", required=True)
    launch.add_argument("--output-dir", required=True)
    launch.add_argument("--areas", type=int, default=4)
    launch.add_argument("--base-port", type=int, default=57000)
    launch.add_argument("--seed", type=int, default=56025)
    launch.add_argument("--time-scale", type=float, default=5.0)
    launch.add_argument("--target-sessions-per-area", type=int, default=1)
    launch.add_argument("--challenger-timeout-seconds", type=float, default=6.0)
    launch.add_argument("--job-worker-count", type=int, default=15)
    launch.add_argument("--cpu-cores-per-shard", type=int, default=16)
    launch.set_defaults(func=launch_matrix)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
