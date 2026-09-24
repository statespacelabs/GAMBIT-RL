#!/usr/bin/env python3
"""Persistent eight-GPU baseline driver for Phase5AutonomousSession."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PARENT_ID = "p44g3_gvg_generalist_roster_mix_generalist_safe_from_s050_wna_u30"
PARENT_SHA256 = "4ce907c625f228546e6f65df3a55021a8afc7b86f366927eb00e06e61f517c4d"
NORMALIZER_SHA256 = "28df2464221f65798ff4e4c98eb0d8d340f18b8765b4ca306fcbc45da4e89b5e"
POLICIES = ("strongest_parent", "direct_steering")
PRESET_CHALLENGERS = {
    "hunt_probe": 1,
    "duel_probe": 1,
    "ppo_h0": 1,
    "ppo_h1": 1,
    "ppo_h2": 1,
    "ppo_h3": 1,
    "ppo_h4": 1,
    "ppo_h5_scripted": 5,
    "reacquire_search": 1,
    "peek_p0": 1,
    "peek_p1": 1,
    "peek_p2": 1,
    "peek_p3": 1,
    "peek_p4": 1,
    "demo_5": 5,
    "endurance_20": 20,
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_action_tuple(actions: np.ndarray, action_spec: Any):
    from mlagents_envs.base_env import ActionTuple

    actions = np.asarray(actions, dtype=np.float32)
    continuous = np.clip(actions[:, :4], -1.0, 1.0)
    branches = len(action_spec.discrete_branches)
    discrete = np.zeros((len(actions), branches), dtype=np.int32)
    discrete[:, :4] = (actions[:, 4:8] > 0.5).astype(np.int32)
    return ActionTuple(continuous=continuous, discrete=discrete)


def direct_steering_action(obs: np.ndarray) -> np.ndarray:
    """Simple deterministic target-relative steering over frozen local45."""
    rel_x, rel_y, rel_z = (float(obs[0]), float(obs[1]), float(obs[2]))
    distance = max(0.0, float(obs[9]))
    planar = max(1e-6, math.hypot(rel_x, rel_z))
    yaw_deg = math.degrees(math.atan2(rel_x, rel_z))
    pitch_deg = math.degrees(math.atan2(rel_y, planar))
    action = np.zeros(8, dtype=np.float32)
    action[2] = np.clip(yaw_deg / 45.0, -1.0, 1.0)
    action[3] = np.clip(pitch_deg / 35.0, -1.0, 1.0)
    if abs(yaw_deg) < 65.0 and distance > 4.5:
        action[1] = 1.0
    elif abs(yaw_deg) >= 65.0:
        action[0] = 0.6 if yaw_deg > 0 else -0.6
    action[4] = 1.0 if abs(yaw_deg) < 8.0 and abs(pitch_deg) < 8.0 else 0.0
    return action


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def validate_layout_selection(
    manifest_path: Path,
    split: str,
    seeds: list[int],
    preset: str,
) -> dict[int, dict[str, Any]]:
    document = json.loads(manifest_path.read_text())
    if document.get("split") != split or document.get("immutable") is not True:
        raise RuntimeError("layout manifest split/immutability mismatch")
    by_seed = {int(item["seed"]): item for item in document["layouts"]}
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("layout seeds must be unique within a cell")
    selected = {}
    for seed in seeds:
        if seed not in by_seed:
            raise RuntimeError(f"seed {seed} is absent from {manifest_path}")
        layout = by_seed[seed]
        if layout.get("validation", {}).get("status") != "PASS":
            raise RuntimeError(f"seed {seed} did not pass frozen validation")
        if preset == "hunt_probe" and layout["requested_initial_los"] is not False:
            raise RuntimeError(f"hunt_probe seed {seed} is not hidden")
        if preset == "duel_probe" and layout["requested_initial_los"] is not True:
            raise RuntimeError(f"duel_probe seed {seed} is not visible")
        if preset == "reacquire_search" and layout["requested_initial_los"] is not True:
            raise RuntimeError(f"reacquire_search seed {seed} is not visible")
        if preset.startswith("peek_p") and layout["requested_initial_los"] is not False:
            raise RuntimeError(f"{preset} seed {seed} is not a hidden-cover layout")
        selected[seed] = layout
    return selected


def policy_for_session(session_ordinal: int) -> str:
    return POLICIES[session_ordinal % len(POLICIES)]


def run_cell(args: argparse.Namespace) -> int:
    import torch
    import yaml
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.engine_configuration_channel import (
        EngineConfigurationChannel,
    )
    from scripts.combat_runtime import (
        RuntimeConfig,
        load_policy_and_normalizer,
        validate_specs,
    )

    prerequisite = yaml.safe_load(Path(args.prerequisite).read_text())
    if prerequisite.get("status") != "PHASE5_PROCEDURAL_LAYOUT_SUITE_READY":
        raise RuntimeError("Goal 4 prerequisite is not ready")
    if args.preset not in PRESET_CHALLENGERS:
        raise RuntimeError(f"unknown preset {args.preset}")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    selected = validate_layout_selection(
        Path(args.manifest), args.split, seeds, args.preset
    )
    if sha256_file(args.checkpoint) != PARENT_SHA256:
        raise RuntimeError("strongest parent checkpoint hash mismatch")
    if sha256_file(args.normalizer) != NORMALIZER_SHA256:
        raise RuntimeError("frozen local45 normalizer hash mismatch")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "autonomous_sessions.jsonl"
    audit_path = output / "autonomous_audit.json"
    layout_audit_path = output / "layout_audit.json"
    runtime_summary_path = output / "runtime_summary.json"
    unity_log_path = output / "unity.log"
    result_path = output / "cell_result.json"
    annotated_path = output / "annotated_sessions.jsonl"
    for path in (
        trace_path,
        audit_path,
        layout_audit_path,
        runtime_summary_path,
        unity_log_path,
        result_path,
        annotated_path,
    ):
        if path.exists():
            path.unlink()

    run_id = f"{args.cell_id}:seed:{args.seed}"
    os.environ.update(
        {
            "GAME_MODE": "HumanVsScripted",
            "PLAYER_B_BOT_MODE": "Idle",
            "NUM_AREAS": str(len(seeds)),
            "ENABLE_VISUAL_OBS": "0",
            "PHASE5_TELEMETRY_SCHEMA": "phase3v2_c_local45",
            "PHASE5_PROCEDURAL_LAYOUT": "1",
            "PHASE5_LAYOUT_MANIFEST": str(Path(args.manifest).resolve()),
            "PHASE5_LAYOUT_MANIFEST_SHA256": sha256_file(args.manifest),
            "PHASE5_LAYOUT_SPLIT": args.split,
            "PHASE5_LAYOUT_SEEDS": ",".join(str(seed) for seed in seeds),
            "PHASE5_LAYOUT_AUDIT_PATH": str(layout_audit_path.resolve()),
            "PHASE5_AUTONOMOUS_SESSION": "1",
            "PHASE5_AUTONOMOUS_PRESET": args.preset,
            "PHASE5_AUTONOMOUS_RUN_ID": run_id,
            "PHASE5_AUTONOMOUS_TRACE_PATH": str(trace_path.resolve()),
            "PHASE5_AUTONOMOUS_AUDIT_PATH": str(audit_path.resolve()),
            "PHASE5_AUTONOMOUS_TARGET_SESSIONS_PER_AREA": str(
                args.target_sessions_per_area
            ),
            "PHASE5_AUTONOMOUS_CHALLENGER_TIMEOUT_SECONDS": str(
                args.challenger_timeout_seconds
            ),
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
            "PHASE5_RUNTIME_SUMMARY_PATH": str(runtime_summary_path.resolve()),
            "DEBUG_LOG_LEVEL": "summary",
        }
    )

    cfg = RuntimeConfig(
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
        num_areas=len(seeds),
    )
    policy, normalizer, _, _, _, device = load_policy_and_normalizer(cfg)
    policy.set_ppo_mode(training=False)
    engine = EngineConfigurationChannel()
    engine.set_configuration_parameters(
        time_scale=args.time_scale,
        target_frame_rate=-1,
        capture_frame_rate=0,
    )

    env = None
    hidden: dict[int, torch.Tensor] = {}
    lifecycle: dict[int, dict[str, int]] = {}
    hidden_reset_count = 0
    decision_rows = 0
    parent_decisions = 0
    direct_decisions = 0
    finite_violations = 0
    failure = ""
    process_pid = -1
    started = time.perf_counter()
    expected_sessions = len(seeds) * args.target_sessions_per_area
    challenge_count = PRESET_CHALLENGERS[args.preset]
    max_steps = int(
        max(
            2000,
            args.target_sessions_per_area
            * challenge_count
            * (args.challenger_timeout_seconds * 50.0 + 100.0)
            * 2.0,
        )
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
                "-logFile",
                str(unity_log_path.resolve()),
            ],
            timeout_wait=args.timeout,
        )
        env.reset()
        behavior, action_spec, specs = validate_specs(env)
        process = getattr(env, "_process", None)
        process_pid = int(getattr(process, "pid", -1))
        for step in range(max_steps):
            decisions, terminals = env.get_steps(behavior)
            for aid_raw in terminals.agent_id:
                aid = int(aid_raw)
                state = lifecycle.setdefault(
                    aid, {"session": 0, "challenge": 0, "area": -1}
                )
                state["challenge"] += 1
                hidden.pop(aid, None)
                hidden_reset_count += 1
                if state["challenge"] >= challenge_count:
                    state["challenge"] = 0
                    state["session"] += 1

            if len(decisions):
                ids = [int(value) for value in decisions.agent_id]
                for aid in ids:
                    lifecycle.setdefault(
                        aid, {"session": 0, "challenge": 0, "area": -1}
                    )
                discovered = sorted(lifecycle)
                if len(discovered) >= len(seeds):
                    for area, aid in enumerate(discovered[: len(seeds)]):
                        lifecycle[aid]["area"] = area

                batch = np.asarray(decisions.obs[0], dtype=np.float32)
                finite_violations += int((~np.isfinite(batch)).sum())
                actions = np.zeros((len(ids), 8), dtype=np.float32)
                parent_indices = [
                    index
                    for index, aid in enumerate(ids)
                    if policy_for_session(lifecycle[aid]["session"])
                    == "strongest_parent"
                ]
                if parent_indices:
                    parent_obs = torch.as_tensor(
                        batch[parent_indices],
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(1)
                    parent_norm = normalizer.normalize_tensor(parent_obs)
                    parent_hidden = torch.cat(
                        [
                            hidden.get(
                                ids[index],
                                policy.init_hidden(1, device),
                            )
                            for index in parent_indices
                        ],
                        dim=1,
                    )
                    with torch.no_grad():
                        distribution, values, next_hidden = policy(
                            parent_norm, parent_hidden
                        )
                        parent_actions = (
                            distribution.mode[0]
                            .squeeze(1)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    if (
                        not torch.isfinite(values).all()
                        or not torch.isfinite(next_hidden).all()
                        or not np.isfinite(parent_actions).all()
                    ):
                        raise RuntimeError("parent inference produced NaN/Inf")
                    for local_index, batch_index in enumerate(parent_indices):
                        aid = ids[batch_index]
                        actions[batch_index] = parent_actions[local_index]
                        hidden[aid] = next_hidden[
                            :, local_index : local_index + 1
                        ].detach()
                    parent_decisions += len(parent_indices)

                for index, aid in enumerate(ids):
                    if policy_for_session(lifecycle[aid]["session"]) == "direct_steering":
                        actions[index] = direct_steering_action(batch[index])
                        direct_decisions += 1
                finite_violations += int((~np.isfinite(actions)).sum())
                env.set_actions(
                    behavior,
                    make_action_tuple(actions, action_spec),
                )
                decision_rows += len(ids)

            rows = read_jsonl(trace_path)
            completed_agent_sessions = sum(
                min(state["session"], args.target_sessions_per_area)
                for state in lifecycle.values()
            )
            if (
                len(rows) >= expected_sessions
                and completed_agent_sessions >= expected_sessions
            ):
                break
            env.step()
        else:
            raise RuntimeError(
                f"cell exceeded max_steps={max_steps} with "
                f"{len(read_jsonl(trace_path))}/{expected_sessions} sessions"
            )
    except Exception as exc:
        failure = str(exc)
    finally:
        if env is not None:
            env.close()

    for _ in range(50):
        if audit_path.exists() and runtime_summary_path.exists():
            break
        time.sleep(0.1)
    rows = read_jsonl(trace_path)
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    layout_audit = (
        json.loads(layout_audit_path.read_text()) if layout_audit_path.exists() else {}
    )
    runtime_summary = (
        json.loads(runtime_summary_path.read_text())
        if runtime_summary_path.exists()
        else {}
    )
    annotated = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "cell_id": args.cell_id,
                "gpu": args.gpu,
                "candidate_policy": policy_for_session(
                    int(row["session_ordinal"])
                ),
                "parent_id": PARENT_ID,
            }
        )
        annotated.append(item)
    annotated_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in annotated)
    )
    process_tokens = {row.get("process_start_token") for row in rows}
    policies_observed = {row["candidate_policy"] for row in annotated}
    row_pass = all(
        row.get("status") == "PASS"
        and row.get("no_manual_input") is True
        and row.get("manual_input_events") == 0
        and row.get("reset_corruptions") == 0
        and row.get("missing_telemetry") == 0
        and row.get("process_restart_count") == 0
        and row.get("challengers_completed") == row.get("challengers_expected")
        for row in rows
    )
    status = "PASS" if (
        not failure
        and len(rows) == expected_sessions
        and row_pass
        and process_tokens and len(process_tokens) == 1
        and policies_observed == set(POLICIES)
        and audit.get("status") == "PASS"
        and audit.get("completed_sessions") == expected_sessions
        and audit.get("human_controller_count") == 0
        and audit.get("manual_input_events") == 0
        and audit.get("process_restart_count") == 0
        and layout_audit.get("status") == "PASS"
        and runtime_summary.get("headless_audit", {}).get("status") == "PASS"
        and finite_violations == 0
        and hidden_reset_count
        >= expected_sessions * challenge_count
    ) else "FAIL"
    result = {
        "schema_version": "phase5_autonomous_baseline_cell_v001",
        "status": status,
        "cell_id": args.cell_id,
        "gpu": args.gpu,
        "preset": args.preset,
        "split": args.split,
        "layout_seeds": seeds,
        "layout_families": [selected[seed]["family"] for seed in seeds],
        "target_sessions_per_area": args.target_sessions_per_area,
        "expected_sessions": expected_sessions,
        "complete_sessions": len(rows),
        "process_pid": process_pid,
        "process_start_tokens": sorted(process_tokens),
        "process_restarts": max(0, len(process_tokens) - 1),
        "decision_rows": decision_rows,
        "parent_decisions": parent_decisions,
        "direct_decisions": direct_decisions,
        "hidden_reset_count": hidden_reset_count,
        "finite_violations": finite_violations,
        "wall_seconds": time.perf_counter() - started,
        "policies_observed": sorted(policies_observed),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "normalizer_sha256": NORMALIZER_SHA256,
        "build_executable_sha256": sha256_file(args.env_path),
        "autonomous_audit": audit,
        "layout_audit_status": layout_audit.get("status"),
        "headless_audit_status": runtime_summary.get("headless_audit", {}).get(
            "status"
        ),
        "failure_reason": failure,
        "specs": specs if "specs" in locals() else {},
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "cell": args.cell_id,
                "sessions": len(rows),
                "hidden_resets": hidden_reset_count,
                "failure": failure,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if status == "PASS" else 1


def select_family(
    document: dict[str, Any],
    family: str,
    count: int,
    los: bool | None = None,
) -> list[int]:
    selected = [
        int(item["seed"])
        for item in document["layouts"]
        if item["family"] == family
        and (los is None or item["requested_initial_los"] is los)
    ]
    if len(selected) < count:
        raise RuntimeError(
            f"need {count} seeds for {family} los={los}, found {len(selected)}"
        )
    return selected[:count]


def build_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    layouts = Path(args.layouts_dir)
    train = json.loads((layouts / "TRAIN_LAYOUTS.json").read_text())
    validation = json.loads((layouts / "VALIDATION_LAYOUTS.json").read_text())
    heldout = json.loads((layouts / "HELDOUT_LAYOUTS.json").read_text())
    areas = args.areas
    validation_visible = [
        int(item["seed"])
        for item in validation["layouts"]
        if item["requested_initial_los"] is True
    ][:areas]
    heldout_seeds = [int(item["seed"]) for item in heldout["layouts"][:areas]]
    mixed_hidden = [
        int(item["seed"])
        for item in train["layouts"]
        if item["requested_initial_los"] is False
    ][:areas]
    corridor_visible_count = sum(
        item["family"] == "parallel_corridors"
        and item["requested_initial_los"] is True
        for item in train["layouts"]
    )
    corridor_visible = select_family(
        train,
        "parallel_corridors",
        min(areas, corridor_visible_count),
        True,
    )
    if len(corridor_visible) < areas:
        corridor_visible.extend(
            int(item["seed"])
            for item in train["layouts"]
            if item["requested_initial_los"] is True
            and int(item["seed"]) not in corridor_visible
        )
        corridor_visible = corridor_visible[:areas]
    cells = [
        ("gpu0_open_hunt", 0, "train", "TRAIN_LAYOUTS.json", "hunt_probe",
         select_family(train, "open_scattered_cover", areas, False)),
        ("gpu1_corridor_duel", 1, "train", "TRAIN_LAYOUTS.json", "duel_probe",
         corridor_visible),
        ("gpu2_rooms_demo", 2, "train", "TRAIN_LAYOUTS.json", "demo_5",
         select_family(train, "rooms_doorways", areas)),
        ("gpu3_pillars_endurance", 3, "train", "TRAIN_LAYOUTS.json", "endurance_20",
         select_family(train, "pillars_alternating_cover", areas)),
        ("gpu4_transition_demo", 4, "train", "TRAIN_LAYOUTS.json", "demo_5",
         select_family(train, "narrow_wide_transitions", areas)),
        ("gpu5_validation_duel", 5, "validation", "VALIDATION_LAYOUTS.json", "duel_probe",
         validation_visible),
        ("gpu6_heldout_endurance", 6, "heldout", "HELDOUT_LAYOUTS.json", "endurance_20",
         heldout_seeds),
        ("gpu7_mixed_hunt", 7, "train", "TRAIN_LAYOUTS.json", "hunt_probe",
         mixed_hidden),
    ]
    if any(len(cell[5]) != areas for cell in cells):
        raise RuntimeError("one or more matrix cells lacks the requested area count")
    return [
        {
            "cell_id": cell_id,
            "gpu": gpu,
            "split": split,
            "manifest": str((layouts / manifest).resolve()),
            "preset": preset,
            "seeds": seeds,
        }
        for cell_id, gpu, split, manifest, preset, seeds in cells
    ]


def launch_matrix(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cells = build_matrix(args)
    (output / "MATRIX_PLAN.json").write_text(
        json.dumps(
            {
                "schema_version": "phase5_autonomous_matrix_plan_v001",
                "cells": cells,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    processes: list[tuple[dict[str, Any], subprocess.Popen[str], Any]] = []
    for index, cell in enumerate(cells):
        cell_dir = output / cell["cell_id"]
        cell_dir.mkdir(parents=True, exist_ok=True)
        log_handle = (cell_dir / "driver.log").open("w")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run-cell",
            "--prerequisite",
            args.prerequisite,
            "--env-path",
            args.env_path,
            "--manifest",
            cell["manifest"],
            "--split",
            cell["split"],
            "--seeds",
            ",".join(str(value) for value in cell["seeds"]),
            "--preset",
            cell["preset"],
            "--checkpoint",
            args.checkpoint,
            "--normalizer",
            args.normalizer,
            "--output-dir",
            str(cell_dir),
            "--cell-id",
            cell["cell_id"],
            "--gpu",
            str(cell["gpu"]),
            "--device",
            "cuda",
            "--base-port",
            str(args.base_port + index * 20),
            "--seed",
            str(args.seed + index * 1000),
            "--time-scale",
            str(args.time_scale),
            "--target-sessions-per-area",
            str(args.target_sessions_per_area),
            "--challenger-timeout-seconds",
            str(args.challenger_timeout_seconds),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(cell["gpu"])
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((cell, process, log_handle))

    returncodes = {}
    for cell, process, handle in processes:
        returncodes[cell["cell_id"]] = process.wait()
        handle.close()
    summary = {
        "schema_version": "phase5_autonomous_matrix_run_v001",
        "status": "PASS" if all(code == 0 for code in returncodes.values()) else "FAIL",
        "returncodes": returncodes,
        "gpu_count": len({cell["gpu"] for cell in cells}),
        "cells": cells,
    }
    (output / "MATRIX_RUN.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["status"] == "PASS" else 1


def add_common_cell_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prerequisite", required=True)
    parser.add_argument("--env-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--preset", choices=tuple(PRESET_CHALLENGERS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-port", type=int, default=56000)
    parser.add_argument("--seed", type=int, default=55025)
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--target-sessions-per-area", type=int, default=2)
    parser.add_argument("--challenger-timeout-seconds", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-cell")
    add_common_cell_args(run)
    run.set_defaults(func=run_cell)

    launch = subparsers.add_parser("launch-matrix")
    launch.add_argument("--prerequisite", required=True)
    launch.add_argument("--env-path", required=True)
    launch.add_argument("--layouts-dir", required=True)
    launch.add_argument("--checkpoint", required=True)
    launch.add_argument("--normalizer", required=True)
    launch.add_argument("--output-dir", required=True)
    launch.add_argument("--areas", type=int, default=7)
    launch.add_argument("--base-port", type=int, default=56000)
    launch.add_argument("--seed", type=int, default=55025)
    launch.add_argument("--time-scale", type=float, default=20.0)
    launch.add_argument("--target-sessions-per-area", type=int, default=2)
    launch.add_argument("--challenger-timeout-seconds", type=float, default=1.0)
    launch.set_defaults(func=launch_matrix)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
