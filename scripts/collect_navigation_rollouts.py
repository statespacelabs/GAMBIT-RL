#!/usr/bin/env python3
"""Collect a fixed-count real-Unity rollout for one Goal 9 curriculum lane."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.autonomous_combat_baseline import make_action_tuple  # noqa: E402
from scripts.collect_navigation_dagger import action_from_distribution  # noqa: E402
from scripts.navigation_ppo import (  # noqa: E402
    CRITIC_DIM,
    READY_PREREQUISITE,
    RewardLedger,
    load_checkpoint,
    peeker_variant_by_id,
    reacquire_variant_by_id,
    sha256_file,
)
from scripts.local_geometry_peeker import (  # noqa: E402
    ENEMY_HEALTH_CRITIC_INDEX,
    LocalGeometryPeeker,
)
from scripts.navigation_policy import (  # noqa: E402
    ACTOR_DIM,
    LOCAL45_DIM,
    LOCAL45_NORMALIZER_SHA256,
    LOS_INDEX,
    PARENT_SHA256,
    FairReacquisitionSearch,
    MapIndependentSafetyLayer,
    compose_actions,
)
from scripts.navigation_policy import load_checkpoint as load_navigator_checkpoint  # noqa: E402
from scripts.tactical_handoff import FairTacticalHandoff  # noqa: E402


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-path", required=True)
    parser.add_argument("--hunter", required=True)
    parser.add_argument("--combat-checkpoint", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--prerequisite", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument(
        "--stage",
        choices=(*tuple(f"H{i}" for i in range(6)), "R0", "L0", *tuple(f"P{i}" for i in range(6))),
        required=True,
    )
    parser.add_argument("--preset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-decisions", type=int, default=2048)
    parser.add_argument("--time-scale", type=float, default=10.0)
    parser.add_argument("--wall-flip-interval", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--neural-opponent", action="store_true")
    parser.add_argument("--opponent-navigator")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--challenger-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--hunt-cue-dropout", type=float, default=0.25)
    parser.add_argument("--hunt-dropout-min-seconds", type=float, default=1.0)
    parser.add_argument("--hunt-dropout-max-seconds", type=float, default=4.0)
    parser.add_argument("--search-stale-timeout-seconds", type=float, default=4.0)
    return parser.parse_args()


def split_observations(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        raise RuntimeError(f"Goal 9 sensor widths must be {sorted(expected)}, got {sorted(by_width)}")
    role, local45, actor, critic = (
        by_width[1], by_width[LOCAL45_DIM], by_width[ACTOR_DIM], by_width[CRITIC_DIM],
    )
    if len({len(role), len(local45), len(actor), len(critic)}) != 1:
        raise RuntimeError("Goal 9 sensors are row-misaligned")
    if not all(np.isfinite(item).all() for item in (role, local45, actor, critic)):
        raise RuntimeError("Goal 9 sensor transport contains NaN/Inf")
    if not np.allclose(actor, critic[:, :ACTOR_DIM], atol=1e-5):
        raise RuntimeError("critic actor prefix differs from fair actor sensor")
    return role[:, 0], local45, actor, critic


def configure(config: argparse.Namespace, output: Path, seeds: list[int]) -> dict[str, Path]:
    paths = {
        "rollout": output / "rollout.npz",
        "result": output / "collection_result.json",
        "trace": output / "autonomous_sessions.jsonl",
        "audit": output / "autonomous_audit.json",
        "layout": output / "layout_audit.json",
        "runtime": output / "runtime_summary.json",
        "unity": output / "unity.log",
        "weapon": output / "weapon_state.jsonl",
    }
    output.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        if path.exists():
            path.unlink()
    manifest = Path(config.manifest).resolve()
    neural = config.neural_opponent
    environment = {
        "GAME_MODE": "GambitVsGambit" if neural else "HumanVsScripted",
        "PLAYER_B_BOT_MODE": "Idle",
        "NUM_AREAS": str(len(seeds)),
        "ENABLE_VISUAL_OBS": "0",
        "PHASE5_TELEMETRY_SCHEMA": "phase3v2_c_local45",
        "PHASE5_NAVIGATOR_EVAL": "1",
        "PHASE5_MAP_GENERAL_PPO": "1",
        "PHASE5_NAVMESH_UPPER_BOUND": "0",
        "PHASE5_ENABLE_NAVMESH_ORACLE": "0",
        "PHASE5_PROCEDURAL_LAYOUT": "1",
        "PHASE5_LAYOUT_MANIFEST": str(manifest),
        "PHASE5_LAYOUT_MANIFEST_SHA256": sha256_file(manifest),
        "PHASE5_LAYOUT_SPLIT": config.split,
        "PHASE5_LAYOUT_SEEDS": ",".join(str(seed) for seed in seeds),
        "PHASE5_LAYOUT_AUDIT_PATH": str(paths["layout"].resolve()),
        "PHASE5_AUTONOMOUS_SESSION": "0" if neural else "1",
        "PHASE5_AUTONOMOUS_PRESET": config.preset,
        "PHASE5_AUTONOMOUS_RUN_ID": config.run_id,
        "PHASE5_AUTONOMOUS_TRACE_PATH": str(paths["trace"].resolve()),
        "PHASE5_AUTONOMOUS_AUDIT_PATH": str(paths["audit"].resolve()),
        "PHASE5_AUTONOMOUS_TARGET_SESSIONS_PER_AREA": "1000",
        "PHASE5_AUTONOMOUS_CHALLENGER_TIMEOUT_SECONDS": str(config.challenger_timeout_seconds),
        "PHASE5_HUNT_CUE_DROPOUT": str(
            config.hunt_cue_dropout if config.preset == "reacquire_search" else 0.15
        ),
        "PHASE5_HUNT_DROPOUT_MIN_SECONDS": str(
            config.hunt_dropout_min_seconds if config.preset == "reacquire_search" else 0.0
        ),
        "PHASE5_HUNT_DROPOUT_MAX_SECONDS": str(
            config.hunt_dropout_max_seconds if config.preset == "reacquire_search" else 0.0
        ),
        "PHASE5_SEARCH_STALE_TIMEOUT_SECONDS": str(config.search_stale_timeout_seconds),
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
        "DEBUG_LOG_LEVEL": "summary",
        "WEAPON_STATE_DEBUG": "1",
        "WEAPON_STATE_DEBUG_PATH": str(paths["weapon"].resolve()),
    }
    os.environ.update(environment)
    for key in (
        "PHASE4_MAP_CONTROL_ENABLED", "PHASE4_MAP_ID", "PHASE4_MAP_MIX_PROFILE",
        "PHASE4_MAP_SEED", "PHASE5_NAVMESH_ORACLE_LABEL",
    ):
        os.environ.pop(key, None)
    return paths


def main() -> int:
    config = args()
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
    from scripts.combat_runtime import RuntimeConfig, load_policy_and_normalizer

    prerequisite = json.loads(Path(config.prerequisite).read_text())
    if prerequisite.get("status") not in {
        READY_PREREQUISITE, "PHASE5_MAP_GENERAL_HUNTER_READY",
        "PHASE5_REACQUISITION_READY",
        "PHASE5_LOCAL_GEOMETRY_PEEKER_READY",
    }:
        raise RuntimeError("required navigator/hunter prerequisite is not ready")
    if sha256_file(config.combat_checkpoint) != PARENT_SHA256:
        raise RuntimeError("frozen combat checkpoint hash mismatch")
    if sha256_file(config.normalizer) != LOCAL45_NORMALIZER_SHA256:
        raise RuntimeError("frozen local45 normalizer hash mismatch")
    seeds = [int(value) for value in config.seeds.split(",") if value]
    if not seeds or len(seeds) != len(set(seeds)):
        raise RuntimeError("curriculum needs unique layout seeds")
    manifest = json.loads(Path(config.manifest).read_text())
    by_seed = {int(row["seed"]): row for row in manifest["layouts"]}
    if any(seed not in by_seed for seed in seeds) or manifest.get("split") != config.split:
        raise RuntimeError("curriculum seed/split mismatch")
    if config.stage in {"H0", "R0"} and any(not by_seed[seed]["requested_initial_los"] for seed in seeds):
        raise RuntimeError(f"{config.stage} requires visible layouts")
    if config.stage in {"H1", "H2", "H4", "H5"} and any(by_seed[seed]["requested_initial_los"] for seed in seeds):
        raise RuntimeError(f"{config.stage} requires hidden layouts")
    if (config.stage.startswith("P") or config.stage == "L0") and any(by_seed[seed]["requested_initial_los"] for seed in seeds):
        raise RuntimeError(f"{config.stage} requires hidden-cover layouts")
    if config.opponent_navigator and not config.neural_opponent:
        raise RuntimeError("a frozen opponent navigator requires --neural-opponent")
    output = Path(config.output_dir).resolve()
    paths = configure(config, output, seeds)
    device = torch.device(config.device)
    model, payload, _ = load_checkpoint(config.hunter, device)
    model.eval()
    try:
        reacquire_variant = reacquire_variant_by_id(model.branch.branch_id)
    except KeyError:
        reacquire_variant = None
    inherited_reacquire = payload.get("metadata", {}).get("reacquire_config")
    if reacquire_variant is None and inherited_reacquire:
        from types import SimpleNamespace
        inherited_reacquire = dict(inherited_reacquire)
        inherited_reacquire["branch"] = SimpleNamespace(
            **inherited_reacquire["branch"]
        )
        reacquire_variant = SimpleNamespace(**inherited_reacquire)
    try:
        peeker_variant = peeker_variant_by_id(model.branch.branch_id)
    except KeyError:
        peeker_variant = None
    peeker_config = payload.get("metadata", {}).get("peeker_config")
    if peeker_variant is not None:
        peeker_config = {
            "branch": {"seed": peeker_variant.branch.seed},
            "dwell_min_steps": peeker_variant.dwell_min_steps,
            "dwell_max_steps": peeker_variant.dwell_max_steps,
            "risk": peeker_variant.risk,
            "repeek_penalty": peeker_variant.repeek_penalty,
        }
    peeker = (
        LocalGeometryPeeker(
            seed=int(peeker_config["branch"]["seed"]),
            dwell_min=int(peeker_config["dwell_min_steps"]),
            dwell_max=int(peeker_config["dwell_max_steps"]),
            risk=float(peeker_config["risk"]),
            repeek_penalty=float(peeker_config["repeek_penalty"]),
        ) if peeker_config is not None else None
    )
    opponent_navigator = None
    opponent_payload: dict = {}
    opponent_peeker = None
    if config.opponent_navigator:
        opponent_navigator, opponent_payload, _ = load_navigator_checkpoint(
            config.opponent_navigator, device,
        )
        opponent_navigator.eval()
        opponent_peeker_config = opponent_payload.get("metadata", {}).get("peeker_config")
        if opponent_peeker_config:
            opponent_peeker = LocalGeometryPeeker(
                seed=int(opponent_peeker_config["branch"]["seed"]) + 100000,
                dwell_min=int(opponent_peeker_config["dwell_min_steps"]),
                dwell_max=int(opponent_peeker_config["dwell_max_steps"]),
                risk=float(opponent_peeker_config["risk"]),
                repeek_penalty=float(opponent_peeker_config["repeek_penalty"]),
            )
    reacquisition_search = FairReacquisitionSearch(
        reacquire_variant.stale_timeout_seconds if reacquire_variant else 4.0
    )
    policy_config = RuntimeConfig(
        env_path=config.env_path,
        normalizer_path=config.normalizer,
        resume_from=config.combat_checkpoint,
        output_dir=str(output), base_port=config.base_port,
        seed=config.seed, time_scale=config.time_scale,
        device=config.device,
        game_mode="GambitVsGambit" if config.neural_opponent else "HumanVsScripted",
        player_b_bot_mode="Idle", num_areas=len(seeds),
    )
    combat, normalizer, _, _, _, combat_device = load_policy_and_normalizer(policy_config)
    combat.set_ppo_mode(training=False)
    engine = EngineConfigurationChannel()
    engine.set_configuration_parameters(
        time_scale=config.time_scale, target_frame_rate=-1, capture_frame_rate=0,
    )
    safety = MapIndependentSafetyLayer(config.wall_flip_interval)
    opponent_safety = MapIndependentSafetyLayer(config.wall_flip_interval)
    tactical_handoff = FairTacticalHandoff()
    tactical_handoff_interventions = 0
    nav_hidden: dict[int, torch.Tensor] = {}
    opponent_nav_hidden: dict[int, torch.Tensor] = {}
    memory_age: defaultdict[int, int] = defaultdict(int)
    combat_hidden: dict[int, torch.Tensor] = {}
    ledgers: dict[int, RewardLedger] = {}
    pending: dict[int, int] = {}
    episodes: defaultdict[int, int] = defaultdict(int)
    arrays: dict[str, list] = {name: [] for name in (
        "actor_obs", "critic_obs", "policy_action", "old_mean", "old_log_std",
        "old_mode", "nav_hidden", "bc_target", "reward", "visible", "intervened",
        "agent_id", "episode_id", "fixed_step", "done",
    )}
    fixed_step = 0
    visible_mismatches = 0
    aim_shoot_action_mismatches = 0
    critic_prefix_max_error = 0.0
    candidate_decisions = 0
    opponent_decisions = 0
    candidate_visible_decisions = 0
    candidate_visible_fire_actions = 0
    policy_mean_look_saturated = 0
    policy_mean_look_values = 0
    hidden_state_leakage_events = 0
    candidate_terminals = {"kill": 0, "death": 0, "timeout": 0, "reacquired": 0}
    reward_component_totals: defaultdict[str, float] = defaultdict(float)
    max_shaping_abs = 0.0
    env = None
    failure = ""
    started = time.perf_counter()
    try:
        env = UnityEnvironment(
            file_name=config.env_path, worker_id=0,
            base_port=config.base_port, seed=config.seed,
            side_channels=[engine], no_graphics=True,
            additional_args=["--phase5-headless", "-logFile", str(paths["unity"])],
            timeout_wait=config.timeout,
        )
        env.reset()
        behaviors = list(env.behavior_specs)
        if len(behaviors) != 1:
            raise RuntimeError(f"expected one behavior, got {behaviors}")
        behavior = behaviors[0]
        action_spec = env.behavior_specs[behavior].action_spec
        max_ticks = max(5000, config.target_decisions * 8)
        for _ in range(max_ticks):
            decisions, terminals = env.get_steps(behavior)
            if len(terminals):
                terminal_role, _, terminal_actor, _ = split_observations(terminals.obs)
                for row, raw_id in enumerate(terminals.agent_id):
                    agent_id = int(raw_id)
                    if terminal_role[row] < 0.5:
                        opponent_nav_hidden.pop(agent_id, None)
                        combat_hidden.pop(agent_id, None)
                        opponent_safety.reset(agent_id)
                        if opponent_peeker is not None:
                            opponent_peeker.reset(agent_id)
                        if agent_id in opponent_nav_hidden:
                            hidden_state_leakage_events += 1
                        continue
                    outcome = (
                        "kill" if float(terminals.reward[row]) > 0
                        else "death" if float(terminals.reward[row]) < 0
                        else "reacquired" if config.preset == "reacquire_search"
                        and terminal_actor[row, LOS_INDEX] > 0.5 else "timeout"
                    )
                    if agent_id in pending:
                        index = pending.pop(agent_id)
                        arrays["reward"][index] += ledgers[agent_id].terminal(
                            outcome,
                            reacquire_variant.reacquisition_reward if reacquire_variant else 0.03,
                        )
                        arrays["done"][index] = 1.0
                    candidate_terminals[outcome] += 1
                    episodes[agent_id] += 1
                    nav_hidden.pop(agent_id, None)
                    memory_age.pop(agent_id, None)
                    combat_hidden.pop(agent_id, None)
                    ledgers.pop(agent_id, None)
                    safety.reset(agent_id)
                    reacquisition_search.reset(agent_id)
                    tactical_handoff.reset(agent_id)
                    if peeker is not None:
                        peeker.reset(agent_id)
                    if agent_id in nav_hidden:
                        hidden_state_leakage_events += 1
            if len(decisions):
                role, local45, actor, critic = split_observations(decisions.obs)
                critic_prefix_max_error = max(
                    critic_prefix_max_error,
                    float(np.max(np.abs(actor - critic[:, :ACTOR_DIM]))),
                )
                ids = [int(value) for value in decisions.agent_id]
                combat_actions = action_from_distribution(
                    combat, normalizer, combat_device, local45, ids, combat_hidden,
                )
                applied = combat_actions.copy()
                candidate_rows = np.flatnonzero(role > 0.5)
                if len(candidate_rows):
                    candidate_ids = [ids[index] for index in candidate_rows]
                    if reacquire_variant is not None:
                        for local_row, source_row in enumerate(candidate_rows):
                            agent_id = candidate_ids[local_row]
                            memory_age[agent_id] = (
                                0 if actor[source_row, LOS_INDEX] > 0.5
                                else memory_age[agent_id] + 1
                            )
                            if memory_age[agent_id] >= reacquire_variant.memory_steps:
                                nav_hidden.pop(agent_id, None)
                                memory_age[agent_id] = 0
                    hidden_before = torch.cat([
                        nav_hidden.get(agent_id, model.navigator.initial_hidden(1, device))
                        for agent_id in candidate_ids
                    ], dim=1)
                    with torch.no_grad():
                        prediction, next_hidden = model.actor(
                            torch.as_tensor(actor[candidate_rows], device=device).unsqueeze(1),
                            hidden_before,
                        )
                    mean = prediction["mean"][:, 0]
                    log_std = prediction["log_std"][:, 0]
                    policy_mean_look_saturated += int((mean[:, 2:4].abs() > 0.95).sum().item())
                    policy_mean_look_values += int(mean[:, 2:4].numel())
                    generator = torch.Generator(device=device)
                    generator.manual_seed(config.seed + fixed_step)
                    sampled = mean + torch.randn(
                        mean.shape, generator=generator, device=device,
                    ) * log_std.exp()
                    policy_action = sampled.clamp(-1.0, 1.0).cpu().numpy()
                    navigation = np.zeros((len(candidate_rows), 8), dtype=np.float32)
                    navigation[:, :4] = policy_action
                    if config.preset == "reacquire_search":
                        navigation, _ = reacquisition_search.apply(
                            actor[candidate_rows], navigation, candidate_ids,
                        )
                    safe_navigation, interventions = safety.apply(
                        actor[candidate_rows], navigation, candidate_ids,
                    )
                    if peeker is not None:
                        composed, bc_target, peek_rewards, _ = peeker.apply(
                            actor[candidate_rows], safe_navigation,
                            combat_actions[candidate_rows], candidate_ids,
                            critic[candidate_rows, ENEMY_HEALTH_CRITIC_INDEX],
                        )
                    else:
                        composed = compose_actions(
                            actor[candidate_rows], safe_navigation,
                            combat_actions[candidate_rows],
                        )
                        composed, tactical_intervention = tactical_handoff.apply(
                            actor[candidate_rows], composed, candidate_ids,
                        )
                        tactical_handoff_interventions += int(tactical_intervention.sum())
                        bc_target = safe_navigation
                        peek_rewards = np.zeros(len(candidate_rows), dtype=np.float32)
                    applied[candidate_rows] = composed
                    visible = actor[candidate_rows, LOS_INDEX] > 0.5
                    candidate_visible_decisions += int(visible.sum())
                    candidate_visible_fire_actions += int((composed[visible, 4] > 0.5).sum())
                    visible_mismatches += int(np.any(
                        composed[visible] != combat_actions[candidate_rows][visible], axis=1,
                    ).sum())
                    aim_shoot_action_mismatches += int(np.any(
                        composed[visible, 2:8]
                        != combat_actions[candidate_rows][visible, 2:8], axis=1,
                    ).sum())
                    for local_row, source_row in enumerate(candidate_rows):
                        agent_id = candidate_ids[local_row]
                        nav_hidden[agent_id] = next_hidden[:, local_row:local_row + 1].detach()
                        if candidate_decisions >= config.target_decisions:
                            continue
                        ledger = ledgers.setdefault(agent_id, RewardLedger())
                        if peeker is not None:
                            shaping = float(peek_rewards[local_row])
                            components = {"cover_peek": shaping}
                            state = peeker.states[agent_id]
                            max_shaping_abs = max(max_shaping_abs, state.shaping_abs)
                        else:
                            shaping, components = ledger.transition(
                                actor[source_row], critic[source_row],
                                policy_action[local_row], model.branch.reward_scale,
                                reacquire_variant.route_change_penalty if reacquire_variant else 0.005,
                            )
                            max_shaping_abs = max(max_shaping_abs, ledger.shaping_abs)
                        for name, value in components.items():
                            reward_component_totals[name] += value
                        index = len(arrays["reward"])
                        pending[agent_id] = index
                        arrays["actor_obs"].append(actor[source_row].copy())
                        arrays["critic_obs"].append(critic[source_row].copy())
                        arrays["policy_action"].append(policy_action[local_row].copy())
                        arrays["old_mean"].append(mean[local_row].cpu().numpy().copy())
                        arrays["old_log_std"].append(log_std[local_row].cpu().numpy().copy())
                        arrays["old_mode"].append(int(prediction["mode_logits"][local_row, 0].argmax().item()))
                        arrays["nav_hidden"].append(hidden_before[0, local_row].cpu().numpy().copy())
                        arrays["bc_target"].append(bc_target[local_row, :4].copy())
                        arrays["reward"].append(float(shaping))
                        arrays["visible"].append(float(visible[local_row]))
                        arrays["intervened"].append(float(interventions[local_row]))
                        arrays["agent_id"].append(agent_id)
                        arrays["episode_id"].append(episodes[agent_id])
                        arrays["fixed_step"].append(fixed_step)
                        arrays["done"].append(0.0)
                        candidate_decisions += 1
                opponent_rows = np.flatnonzero(role < 0.5)
                if opponent_navigator is not None and len(opponent_rows):
                    opponent_ids = [ids[index] for index in opponent_rows]
                    opponent_hidden_before = torch.cat([
                        opponent_nav_hidden.get(
                            agent_id, opponent_navigator.initial_hidden(1, device)
                        ) for agent_id in opponent_ids
                    ], dim=1)
                    with torch.no_grad():
                        opponent_prediction, opponent_next_hidden = opponent_navigator(
                            torch.as_tensor(actor[opponent_rows], device=device).unsqueeze(1),
                            opponent_hidden_before,
                        )
                    opponent_navigation = np.zeros((len(opponent_rows), 8), dtype=np.float32)
                    opponent_navigation[:, :4] = np.clip(
                        opponent_prediction["mean"][:, 0].cpu().numpy(), -1.0, 1.0,
                    )
                    opponent_navigation, _ = opponent_safety.apply(
                        actor[opponent_rows], opponent_navigation, opponent_ids,
                    )
                    if opponent_peeker is not None:
                        opponent_composed, _, _, _ = opponent_peeker.apply(
                            actor[opponent_rows], opponent_navigation,
                            combat_actions[opponent_rows], opponent_ids,
                        )
                    else:
                        opponent_composed = compose_actions(
                            actor[opponent_rows], opponent_navigation,
                            combat_actions[opponent_rows],
                        )
                    applied[opponent_rows] = opponent_composed
                    for local_row, agent_id in enumerate(opponent_ids):
                        opponent_nav_hidden[agent_id] = opponent_next_hidden[
                            :, local_row:local_row + 1
                        ].detach()
                    opponent_decisions += len(opponent_rows)
                env.set_actions(behavior, make_action_tuple(applied, action_spec))
            env.step()
            fixed_step += 1
            if candidate_decisions >= config.target_decisions:
                break
        else:
            raise RuntimeError("fixed rollout exceeded tick budget")
    except Exception as exc:
        failure = str(exc)
    finally:
        if env is not None:
            env.close()

    weapon_rows = []
    if paths["weapon"].exists():
        for line in paths["weapon"].read_text(errors="replace").splitlines():
            if line.strip():
                try:
                    weapon_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    failure = failure or "malformed weapon audit row"
    ownership_mismatch_count = sum(
        row.get("blocked_fire_reason") in {"NO_OWNER", "WRONG_OWNER"}
        for row in weapon_rows
    )
    unity_text = paths["unity"].read_text(errors="replace") if paths["unity"].exists() else ""
    fire_mismatch_count = unity_text.count("First post-reset shot failed")
    mean_look_saturation_rate = (
        policy_mean_look_saturated / max(1, policy_mean_look_values)
    )
    zero_fire_collapse = (
        candidate_visible_decisions >= 20 and candidate_visible_fire_actions == 0
    )

    status = (
        "PASS" if not failure
        and candidate_decisions == config.target_decisions
        and (config.stage != "H0" or visible_mismatches == 0)
        and (not (config.stage.startswith("P") or config.stage == "L0")
             or aim_shoot_action_mismatches == 0)
        and critic_prefix_max_error <= 1e-5
        and max_shaping_abs <= 0.20 + 1e-6
        and (config.stage != "L0" or mean_look_saturation_rate <= 0.10)
        and (config.stage != "L0" or hidden_state_leakage_events == 0)
        and (config.stage != "L0" or ownership_mismatch_count + fire_mismatch_count == 0)
        and (config.stage != "L0" or not zero_fire_collapse)
        else "FAIL"
    )
    if arrays["actor_obs"]:
        packed = {
            name: np.asarray(values, dtype=(
                np.float32 if name not in {"old_mode", "agent_id", "episode_id", "fixed_step"}
                else np.int64
            ))
            for name, values in arrays.items()
        }
        np.savez_compressed(paths["rollout"], **packed)
    result = {
        "schema_version": "phase5_map_general_hunter_rollout_v001",
        "status": status,
        "failure_reason": failure,
        "stage": config.stage,
        "preset": config.preset,
        "split": config.split,
        "seeds": seeds,
        "fixed_environment_steps": fixed_step,
        "candidate_decisions": candidate_decisions,
        "target_decisions": config.target_decisions,
        "opponent_decisions": opponent_decisions,
        "neural_opponent": config.neural_opponent,
        "neural_opponent_checkpoint_sha256": (
            sha256_file(config.opponent_navigator) if config.opponent_navigator
            else PARENT_SHA256 if config.neural_opponent else ""
        ),
        "opponent_navigator": str(Path(config.opponent_navigator).resolve()) if config.opponent_navigator else "",
        # Script controllers, the external Phase 4 combat parent, and an explicit
        # Phase 5 navigator are all immutable for the lifetime of this process.
        "opponent_frozen_within_window": True,
        "simultaneous_opponent_updates": False,
        "scripted_opponent": not config.neural_opponent,
        "actor_input_schema": "phase5_actor_obs_v001",
        "actor_input_width": ACTOR_DIM,
        "critic_input_schema": "phase5_privileged_critic_obs_v001",
        "critic_input_width": CRITIC_DIM,
        "critic_prefix_max_abs_error": critic_prefix_max_error,
        "actor_received_privileged_suffix": False,
        "map_identity_actor_input": False,
        "absolute_coordinate_actor_input": False,
        "navmesh_actor_input": False,
        "visible_combat_action_mismatches": visible_mismatches,
        "aim_shoot_action_mismatches": aim_shoot_action_mismatches,
        "policy_mean_look_saturation_rate": mean_look_saturation_rate,
        "policy_mean_look_saturation_limit": 0.10,
        "candidate_visible_decisions": candidate_visible_decisions,
        "candidate_visible_fire_actions": candidate_visible_fire_actions,
        "zero_fire_collapse": zero_fire_collapse,
        "hidden_state_leakage_events": hidden_state_leakage_events,
        "weapon_ownership_mismatch_count": ownership_mismatch_count,
        "weapon_fire_mismatch_count": fire_mismatch_count,
        "tactical_handoff_interventions": tactical_handoff_interventions,
        "reward_terminal_counts": candidate_terminals,
        "reward_component_totals": dict(reward_component_totals),
        "max_navigation_shaping_abs_per_life": max_shaping_abs,
        "navigation_shaping_cap": 0.20,
        "peek_metrics": peeker.summary() if peeker is not None else None,
        "peeker_config": peeker_config,
        "per_map_cover_anchors": False,
        "peek_point_files": False,
        "rollout": str(paths["rollout"]),
        "rollout_sha256": sha256_file(paths["rollout"]) if paths["rollout"].exists() else "",
        "hunter_checkpoint_sha256": sha256_file(config.hunter),
        "combat_checkpoint_sha256": sha256_file(config.combat_checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "reacquire_config": (
            {
                "memory_steps": reacquire_variant.memory_steps,
                "cue_dropout": reacquire_variant.cue_dropout,
                "stale_timeout_seconds": reacquire_variant.stale_timeout_seconds,
                "reacquisition_reward": reacquire_variant.reacquisition_reward,
                "route_change_penalty": reacquire_variant.route_change_penalty,
                "gru_hidden": reacquire_variant.branch.gru_hidden,
            } if reacquire_variant else None
        ),
        "source_runtime_audits": {
            "layout": str(paths["layout"]),
            "runtime": str(paths["runtime"]),
            "autonomous": str(paths["audit"]) if not config.neural_opponent else "not_applicable_dual_agent",
        },
    }
    paths["result"].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
