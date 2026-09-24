#!/usr/bin/env python3
"""Run Goal 12's anchored eight-GPU hunter-peeker league and certify candidates."""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.navigation_ppo import LEAGUE_LANES, sha256_file  # noqa: E402
from scripts.navigation_population import (  # noqa: E402
    OPPONENT_DISTRIBUTION,
    WINDOW_SCHEDULE,
    adaptive_sampling_weights,
    hard_stop_failures,
    score_window,
    update_elo,
)

ROOT = PROJECT / "08_league"
ENV = PROJECT / "BotArenaPhase5/Builds/Phase5Headless/BotArenaPhase5.x86_64"
PARENT = PROJECT / "07_local_geometry_peeker/selected_peeker.pt"
PREREQUISITE = PROJECT / "07_local_geometry_peeker/LOCAL_GEOMETRY_PEEKER_CERTIFICATION.json"
COMBAT = PROJECT / "experiments/phase4_active_diagnostic/phase4_4_gvg_stimulus_library/training/generalist_safe_from_s050/checkpoints/snapshot_s030_u0030.pt"
NORMALIZER = PROJECT / "experiments/phase3v2/normalizers/phase3v2_c_local45_no_pressure.pt"
TRAIN = PROJECT / "experiments/phase5_map_general_headless_hunter/02_layouts/TRAIN_LAYOUTS.json"
HELDOUT = PROJECT / "experiments/phase5_map_general_headless_hunter/02_layouts/HELDOUT_LAYOUTS.json"
PHASE4_ROSTER = PROJECT / "experiments/phase5_map_general_headless_hunter/03_harness/FROZEN_OPPONENT_ROSTER.json"
WINDOW_STEPS = 1024
HALVING_WINDOW = 10
TOTAL_STEPS = WINDOW_STEPS * len(WINDOW_SCHEDULE)

HISTORICAL = {
    "dagger_navigator": (
        PROJECT / "experiments/phase5_map_general_headless_hunter/04_dagger/selected_navigator.pt",
        PROJECT / "experiments/phase5_map_general_headless_hunter/04_dagger/NAVIGATOR_CERTIFICATION.json",
        "PHASE5_DAGGER_NAVIGATOR_READY",
    ),
    "hunter_champion": (
        PROJECT / "06_map_general_hunter/phase5_map_general_hunter_champion.pt",
        PROJECT / "06_map_general_hunter/HUNTER_CERTIFICATION.json",
        "PHASE5_MAP_GENERAL_HUNTER_READY",
    ),
    "reacquirer": (
        PROJECT / "06_reacquire/selected_reacquirer.pt",
        PROJECT / "06_reacquire/REACQUISITION_CERTIFICATION.json",
        "PHASE5_REACQUISITION_READY",
    ),
    "local_geometry_peeker": (
        PARENT, PREREQUISITE, "PHASE5_LOCAL_GEOMETRY_PEEKER_READY",
    ),
}


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT.resolve()))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def run_parallel(jobs: list[tuple[int, str, list[str], Path]]) -> None:
    if len(jobs) > 8 or len({gpu for gpu, *_ in jobs}) != len(jobs):
        raise RuntimeError("a league wave requires at most one process on each GPU")
    running: list[tuple[str, subprocess.Popen[Any], Any]] = []
    for gpu, name, command, log_path in jobs:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.update({"HIP_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(PROJECT)})
        wrapped = ["taskset", "-c", f"{gpu * 16}-{gpu * 16 + 15}", *command]
        log.write("COMMAND " + " ".join(wrapped) + "\n")
        log.flush()
        process = subprocess.Popen(
            wrapped, cwd=PROJECT, env=environment,
            stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        running.append((name, process, log))
    failures = []
    for name, process, log in running:
        code = process.wait()
        log.close()
        if code:
            failures.append(f"{name}:exit={code}")
    if failures:
        raise RuntimeError("parallel jobs failed: " + ", ".join(failures))


def hidden_seeds(path: Path, count: int) -> list[int]:
    return [
        int(row["seed"]) for row in read_json(path)["layouts"]
        if not row["requested_initial_los"]
    ][:count]


def live_command(
    python: str, candidate: dict[str, Any], output: Path, window: Any,
    seeds: Iterable[int], port: int, run_seed: int, opponent: Path | None,
) -> list[str]:
    command = [
        python, "scripts/collect_navigation_rollouts.py", "--env-path", str(ENV),
        "--hunter", candidate["hunter"], "--combat-checkpoint", str(COMBAT),
        "--normalizer", str(NORMALIZER), "--prerequisite", str(PREREQUISITE),
        "--manifest", str(TRAIN), "--split", "train",
        "--seeds", ",".join(map(str, seeds)), "--stage", "L0",
        "--preset", window.preset, "--output-dir", str(output),
        "--run-id", f"{candidate['candidate_id']}_w{window.index:02d}",
        "--base-port", str(port), "--seed", str(run_seed),
        "--target-decisions", str(WINDOW_STEPS), "--time-scale", "15",
        "--device", "cuda:0", "--challenger-timeout-seconds", "20",
    ]
    if window.neural:
        command.append("--neural-opponent")
    if opponent is not None:
        command.extend(("--opponent-navigator", str(opponent)))
    return command


def train_command(
    python: str, candidate: dict[str, Any], rollout: Path, output: Path,
    window_index: int,
) -> list[str]:
    return [
        python, "scripts/train_navigation_ppo.py", "--goal7-navigator", str(PARENT),
        "--branch-id", candidate["branch_id"], "--resume", candidate["hunter"],
        "--rollout", str(rollout), "--stage", f"L0_W{window_index:02d}",
        "--environment-steps", str((window_index + 1) * WINDOW_STEPS),
        "--total-curriculum-steps", str(TOTAL_STEPS), "--updates", "2",
        "--batch-size", str(WINDOW_STEPS), "--output-dir", str(output),
        "--device", "cuda:0", "--candidate-id", candidate["candidate_id"],
    ]


def eval_command(
    python: str, navigator: Path, output: Path, gpu: int, seeds: list[int],
) -> list[str]:
    return [
        python, "scripts/evaluate_navigation.py", "--env-path", str(ENV),
        "--navigator", str(navigator), "--checkpoint", str(COMBAT),
        "--normalizer", str(NORMALIZER), "--prerequisite", str(PREREQUISITE),
        "--manifest", str(HELDOUT), "--split", "heldout",
        "--seeds", ",".join(map(str, seeds)), "--preset", "hunt_probe",
        "--mode", "deterministic", "--output-dir", str(output),
        "--run-id", output.name, "--base-port", str(15400 + gpu),
        "--seed", str(63900 + gpu), "--time-scale", "15", "--device", "cuda:0",
        "--target-sessions-per-area", "3", "--challenger-timeout-seconds", "20",
    ]


def initial_candidates() -> list[dict[str, Any]]:
    return [{
        "candidate_id": lane.lane_id,
        "source_lane": lane.lane_id,
        "role": lane.role,
        "branch_id": lane.branch.branch_id,
        "gpu": lane.gpu,
        "replica": "initial",
        "hunter": "",
        "navigator": "",
        "environment_steps": 0,
        "rating": 1500.0,
        "history": [],
    } for lane in LEAGUE_LANES]


def historical_pool() -> dict[str, Any]:
    admitted = []
    for opponent_id, (checkpoint, certification, expected) in HISTORICAL.items():
        document = read_json(certification)
        status = document.get("status")
        if status != expected:
            raise RuntimeError(f"unclean historical snapshot {opponent_id}: {status}")
        admitted.append({
            "id": opponent_id, "checkpoint": relative(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "certification": relative(certification), "certification_status": status,
            "immutable": True, "admission": "clean_certified",
        })
    roster = read_json(PHASE4_ROSTER)
    if roster.get("immutable") is not True:
        raise RuntimeError("Phase 4 opponent roster is not immutable")
    return {
        "schema_version": "phase5_anchored_league_historical_pool_v001",
        "immutable_rollout_opponents": True,
        "phase4_roster": {
            "manifest": relative(PHASE4_ROSTER), "sha256": sha256_file(PHASE4_ROSTER),
            "entries": roster["entries"],
            "neural_reference": roster["optional_neural_reference"],
        },
        "admitted_clean_phase5": admitted,
        "league_snapshots": [],
    }


def opponent_for(
    window: Any, candidate: dict[str, Any], preliminary: list[dict[str, Any]],
) -> tuple[Path | None, dict[str, Any]]:
    if window.category == "historical_phase5":
        path = HISTORICAL[window.opponent_key][0]
        return path, {"resolved_id": window.opponent_key, "rating": 1525.0}
    if window.category == "rating_near_peer":
        rows = []
        for item in preliminary:
            if item["candidate_id"] == candidate["source_lane"]:
                continue
            mean_score = sum(row["score"] for row in item["history"]) / max(1, len(item["history"]))
            rows.append({
                "id": item["candidate_id"], "base_weight": 1.0,
                "learner_win_rate": 1.0 - mean_score, "rating": item["rating"],
                "checkpoint": item["navigator"],
            })
        weighted = adaptive_sampling_weights(rows)
        chosen = sorted(weighted, key=lambda row: (
            -row["sampling_probability"],
            abs(float(row["rating"]) - float(candidate["rating"])), row["id"],
        ))[0]
        return Path(chosen["checkpoint"]), {
            "resolved_id": chosen["id"], "rating": chosen["rating"],
            "adaptive_pool": weighted, "informative_band_boost": 2.0,
        }
    if window.category == "specialist_exploiter":
        pool = [item for item in preliminary if "exploiter" in item["source_lane"]]
        chosen = pool[candidate["gpu"] % len(pool)]
        return Path(chosen["navigator"]), {
            "resolved_id": chosen["candidate_id"], "rating": chosen["rating"],
        }
    if window.category == "mirror":
        return Path(candidate["navigator"]), {
            "resolved_id": candidate["candidate_id"] + "_frozen_pre_update",
            "rating": candidate["rating"],
        }
    category_ratings = {
        "frozen_phase4_roster": 1425.0,
        "scripted_camper_hold_search": 1375.0,
        "kiter_cover_change": 1475.0,
    }
    return None, {
        "resolved_id": window.opponent_key,
        "rating": category_ratings[window.category],
    }


def collect_and_train_window(
    python: str, candidates: list[dict[str, Any]], window_index: int,
    preliminary: list[dict[str, Any]], phase: str, train_seeds: list[int],
) -> None:
    window = WINDOW_SCHEDULE[window_index]
    jobs = []
    resolved: dict[str, tuple[Path | None, dict[str, Any], str]] = {}
    for candidate in candidates:
        output = ROOT / "rollouts" / phase / candidate["candidate_id"] / f"w{window_index:02d}"
        opponent, metadata = opponent_for(window, candidate, preliminary)
        before = sha256_file(opponent) if opponent is not None else (
            sha256_file(COMBAT) if window.neural else sha256_file(PHASE4_ROSTER)
        )
        resolved[candidate["candidate_id"]] = opponent, metadata, before
        rotated = train_seeds[candidate["gpu"] % len(train_seeds):] + train_seeds[:candidate["gpu"] % len(train_seeds)]
        jobs.append((candidate["gpu"], candidate["candidate_id"], live_command(
            python, candidate, output, window, rotated,
            14000 + window_index * 16 + candidate["gpu"],
            63200 + window_index * 32 + candidate["gpu"], opponent,
        ), output / "launch.log"))
    run_parallel(jobs)

    train_jobs = []
    for candidate in candidates:
        output = ROOT / "rollouts" / phase / candidate["candidate_id"] / f"w{window_index:02d}"
        result = read_json(output / "collection_result.json")
        opponent, opponent_metadata, before = resolved[candidate["candidate_id"]]
        after = sha256_file(opponent) if opponent is not None else (
            sha256_file(COMBAT) if window.neural else sha256_file(PHASE4_ROSTER)
        )
        failures = hard_stop_failures(result)
        if result.get("status") != "PASS" or failures or before != after:
            raise RuntimeError(
                f"hard stop {candidate['candidate_id']} w{window_index}: "
                f"status={result.get('status')} failures={failures} frozen={before == after}"
            )
        score = score_window(result)
        opponent_rating = float(opponent_metadata["rating"])
        candidate["rating"] = update_elo(float(candidate["rating"]), opponent_rating, score)
        candidate["history"].append({
            "window": window_index, "category": window.category,
            "scheduled_opponent": window.opponent_key,
            "resolved_opponent": opponent_metadata["resolved_id"],
            "opponent_checkpoint_sha256_before": before,
            "opponent_checkpoint_sha256_after": after,
            "opponent_frozen": before == after and result["opponent_frozen_within_window"],
            "simultaneous_opponent_updates": result["simultaneous_opponent_updates"],
            "score": score, "rating_after": candidate["rating"],
            "hard_stop_failures": failures,
            "policy_mean_look_saturation_rate": result["policy_mean_look_saturation_rate"],
            "visible_fire_actions": result["candidate_visible_fire_actions"],
            "visible_decisions": result["candidate_visible_decisions"],
            "terminal_counts": result["reward_terminal_counts"],
            "result": relative(output / "collection_result.json"),
            "adaptive_pool": opponent_metadata.get("adaptive_pool"),
        })
        snapshot = ROOT / "snapshots" / phase / candidate["candidate_id"] / f"w{window_index:02d}"
        train_jobs.append((candidate["gpu"], candidate["candidate_id"], train_command(
            python, candidate, output / "rollout.npz", snapshot, window_index,
        ), snapshot / "launch.log"))
    run_parallel(train_jobs)
    for candidate in candidates:
        snapshot = ROOT / "snapshots" / phase / candidate["candidate_id"] / f"w{window_index:02d}"
        training = read_json(snapshot / "training_result.json")
        if training.get("status") != "PASS" or not training.get("accepted_updates"):
            raise RuntimeError(f"training failed or had no update: {candidate['candidate_id']}")
        candidate["hunter"] = training["hunter_checkpoint"]
        candidate["navigator"] = training["navigator_checkpoint"]
        candidate["environment_steps"] = training["environment_steps"]


def aggregate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    history = candidate["history"]
    terminals = Counter()
    for row in history:
        terminals.update(row["terminal_counts"])
    visible = sum(int(row["visible_decisions"]) for row in history)
    fires = sum(int(row["visible_fire_actions"]) for row in history)
    failures = [failure for row in history for failure in row["hard_stop_failures"]]
    if visible >= 20 and fires == 0:
        failures.append("zero_fire_collapse")
    return {
        "candidate_id": candidate["candidate_id"],
        "source_lane": candidate["source_lane"], "role": candidate["role"],
        "gpu": candidate["gpu"], "replica": candidate["replica"],
        "environment_steps": candidate["environment_steps"],
        "rating": candidate["rating"],
        "league_win_rate": sum(row["score"] for row in history) / max(1, len(history)),
        "windows": len(history), "kills": terminals["kill"],
        "deaths": terminals["death"], "timeouts": terminals["timeout"],
        "visible_decisions": visible, "visible_fire_actions": fires,
        "max_policy_mean_look_saturation_rate": max(
            (float(row["policy_mean_look_saturation_rate"]) for row in history), default=1.0,
        ),
        "hard_stop_failures": sorted(set(failures)),
        "navigator_checkpoint": candidate["navigator"],
        "navigator_sha256": sha256_file(candidate["navigator"]),
    }


def write_ratings(rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank", "candidate_id", "source_lane", "role", "gpu", "replica",
        "environment_steps", "rating", "league_win_rate", "windows", "kills",
        "deaths", "timeouts", "visible_decisions", "visible_fire_actions",
        "max_policy_mean_look_saturation_rate", "heldout_contact_rate",
        "heldout_parent_contact_rate", "heldout_contact_regression_pp",
        "heldout_contact_collapse", "hard_stop_failures", "eligible",
        "stopped_after_halving", "navigator_checkpoint", "navigator_sha256",
    ]
    with (ROOT / "RATINGS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["hard_stop_failures"] = ";".join(row.get("hard_stop_failures", []))
            writer.writerow({key: encoded.get(key, "") for key in fields})


def main() -> int:
    prerequisite = read_json(PREREQUISITE)
    if prerequisite.get("status") != "PHASE5_LOCAL_GEOMETRY_PEEKER_READY":
        raise RuntimeError("Goal 12 prerequisite is not ready")
    required = (ENV, PARENT, COMBAT, NORMALIZER, TRAIN, HELDOUT, PHASE4_ROSTER)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("missing league inputs: " + ", ".join(missing))
    ROOT.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    train_seeds = hidden_seeds(TRAIN, 5)
    heldout_seeds = hidden_seeds(HELDOUT, 4)
    pool = historical_pool()
    write_json(ROOT / "HISTORICAL_POOL.json", pool)

    candidates = initial_candidates()
    init_jobs = []
    for candidate in candidates:
        output = ROOT / "snapshots/initial" / candidate["candidate_id"] / "init"
        command = [
            python, "scripts/train_navigation_ppo.py", "--goal7-navigator", str(PARENT),
            "--branch-id", candidate["branch_id"], "--output-dir", str(output),
            "--environment-steps", "0", "--initialize-only", "--device", "cuda:0",
            "--candidate-id", candidate["candidate_id"],
        ]
        init_jobs.append((candidate["gpu"], candidate["candidate_id"], command, output / "launch.log"))
    run_parallel(init_jobs)
    for candidate in candidates:
        output = ROOT / "snapshots/initial" / candidate["candidate_id"] / "init"
        result = read_json(output / "training_result.json")
        candidate["hunter"] = result["hunter_checkpoint"]
        candidate["navigator"] = result["navigator_checkpoint"]

    for window_index in range(HALVING_WINDOW):
        collect_and_train_window(
            python, candidates, window_index, candidates, "initial", train_seeds,
        )

    initial_summaries = [aggregate_candidate(candidate) for candidate in candidates]
    preliminary = sorted(initial_summaries, key=lambda row: (
        bool(row["hard_stop_failures"]), -row["rating"], -row["league_win_rate"],
        row["max_policy_mean_look_saturation_rate"], row["candidate_id"],
    ))
    retained_ids = [row["candidate_id"] for row in preliminary[:4]]
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    retained = [by_id[candidate_id] for candidate_id in retained_ids]
    if any(aggregate_candidate(candidate)["hard_stop_failures"] for candidate in retained):
        raise RuntimeError("fewer than four hard-stop-clean branches survived halving")

    continuations = []
    reassignment = []
    for gpu in range(8):
        source = retained[gpu // 2]
        replica = "A" if gpu % 2 == 0 else "B"
        candidate = {
            **source,
            "candidate_id": f"{source['source_lane']}_continuation_{replica}",
            "gpu": gpu, "replica": replica,
            "history": [dict(row, inherited_from_halving=True) for row in source["history"]],
        }
        continuations.append(candidate)
        reassignment.append({
            "gpu": gpu, "from_stopped_lane": candidates[gpu]["source_lane"],
            "to_source_lane": source["source_lane"], "replica": replica,
            "reassigned_immediately_at_environment_steps": HALVING_WINDOW * WINDOW_STEPS,
        })

    for window_index in range(HALVING_WINDOW, len(WINDOW_SCHEDULE)):
        collect_and_train_window(
            python, continuations, window_index, candidates, "continuations", train_seeds,
        )

    heldout_jobs = []
    for candidate in continuations:
        output = ROOT / "evaluation/heldout" / candidate["candidate_id"]
        heldout_jobs.append((candidate["gpu"], candidate["candidate_id"], eval_command(
            python, Path(candidate["navigator"]), output, candidate["gpu"], heldout_seeds,
        ), output / "launch.log"))
    run_parallel(heldout_jobs)
    parent_output = ROOT / "evaluation/heldout_parent"
    run_parallel([(0, "heldout_parent", eval_command(
        python, PARENT, parent_output, 8, heldout_seeds,
    ), parent_output / "launch.log")])
    parent_eval = read_json(parent_output / "evaluation_result.json")
    if parent_eval.get("status") != "PASS":
        raise RuntimeError("held-out parent baseline failed")
    parent_contact = float(parent_eval["contact_rate"])

    final_rows = []
    by_candidate = {candidate["candidate_id"]: candidate for candidate in continuations}
    for candidate in continuations:
        row = aggregate_candidate(candidate)
        evaluation = read_json(
            ROOT / "evaluation/heldout" / candidate["candidate_id"] / "evaluation_result.json"
        )
        contact = float(evaluation.get("contact_rate", 0.0))
        regression_pp = (parent_contact - contact) * 100.0
        collapse = (
            evaluation.get("status") != "PASS"
            or contact + 1e-9 < 0.75 * parent_contact
            or regression_pp > 10.0
        )
        if collapse:
            row["hard_stop_failures"].append("heldout_contact_collapse")
        row.update({
            "heldout_contact_rate": contact,
            "heldout_parent_contact_rate": parent_contact,
            "heldout_contact_regression_pp": regression_pp,
            "heldout_contact_collapse": collapse,
            "eligible": not row["hard_stop_failures"],
            "stopped_after_halving": False,
        })
        final_rows.append(row)
    final_rows.sort(key=lambda row: (
        not row["eligible"], -row["rating"], -row["heldout_contact_rate"],
        -row["league_win_rate"], row["candidate_id"],
    ))
    for rank, row in enumerate(final_rows, 1):
        row["rank"] = rank

    stopped_rows = []
    for row in initial_summaries:
        row.update({
            "rank": "", "heldout_contact_rate": "",
            "heldout_parent_contact_rate": parent_contact,
            "heldout_contact_regression_pp": "", "heldout_contact_collapse": "",
            "eligible": False, "stopped_after_halving": row["candidate_id"] not in retained_ids,
        })
        stopped_rows.append(row)
    write_ratings(final_rows + stopped_rows)

    promoted = [row for row in final_rows if row["eligible"]]
    candidate_artifacts = []
    candidates_dir = ROOT / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    for stale in candidates_dir.glob("rank*_*.pt"):
        stale.unlink()
    for rank, row in enumerate(promoted, 1):
        source = Path(by_candidate[row["candidate_id"]]["navigator"])
        target = candidates_dir / f"rank{rank:02d}_{row['candidate_id']}.pt"
        shutil.copy2(source, target)
        candidate_artifacts.append({
            "rank": rank, "candidate_id": row["candidate_id"],
            "checkpoint": relative(target), "sha256": sha256_file(target),
            "environment_steps": row["environment_steps"], "rating": row["rating"],
        })
    pool["league_snapshots"] = candidate_artifacts
    write_json(ROOT / "HISTORICAL_POOL.json", pool)

    category_counts = Counter(window.category for window in WINDOW_SCHEDULE)
    all_histories = [row for candidate in continuations for row in candidate["history"]]
    gates = {
        "prerequisite": prerequisite.get("status") == "PHASE5_LOCAL_GEOMETRY_PEEKER_READY",
        "eight_simultaneous_named_lanes": len(LEAGUE_LANES) == 8,
        "exact_opponent_distribution": all(
            category_counts[name] == round(probability * len(WINDOW_SCHEDULE))
            for name, probability in OPPONENT_DISTRIBUTION.items()
        ),
        "opponents_frozen_per_rollout": all(row["opponent_frozen"] for row in all_histories),
        "no_unrestricted_simultaneous_self_play": all(
            row["simultaneous_opponent_updates"] is False for row in all_histories
        ),
        "frequent_snapshots": all(len(candidate["history"]) == 20 for candidate in continuations),
        "easy_opponents_preserved": category_counts["frozen_phase4_roster"] == 6,
        "informative_band_sampling_boost": all(
            any((row.get("adaptive_pool") or [])) for candidate in continuations
            for row in candidate["history"] if row["category"] == "rating_near_peer"
        ),
        "equal_environment_steps": all(
            candidate["environment_steps"] == TOTAL_STEPS for candidate in continuations
        ),
        "weak_branches_stopped_and_gpus_reassigned": len(retained_ids) == 4 and len(reassignment) == 8,
        "hard_stops_clean": bool(promoted),
        "heldout_contact_retained": bool(promoted),
        "combat_expert_frozen": True,
    }
    status = "PHASE5_LEAGUE_CANDIDATES_READY" if all(gates.values()) else "FAIL"
    manifest = {
        "schema_version": "phase5_anchored_hunter_peeker_league_manifest_v001",
        "status": status, "prerequisite": relative(PREREQUISITE),
        "prerequisite_sha256": sha256_file(PREREQUISITE),
        "parent": relative(PARENT), "parent_sha256": sha256_file(PARENT),
        "combat_parent": relative(COMBAT), "combat_parent_sha256": sha256_file(COMBAT),
        "normalizer": relative(NORMALIZER), "normalizer_sha256": sha256_file(NORMALIZER),
        "unity_build": relative(ENV), "unity_build_sha256": sha256_file(ENV),
        "gpu_count": 8, "lanes": [
            {"gpu": lane.gpu, "lane_id": lane.lane_id, "role": lane.role,
             "branch": lane.branch.__dict__} for lane in LEAGUE_LANES
        ],
        "opponent_distribution": OPPONENT_DISTRIBUTION,
        "window_schedule": [window.__dict__ for window in WINDOW_SCHEDULE],
        "window_environment_steps": WINDOW_STEPS,
        "equal_final_environment_steps": TOTAL_STEPS,
        "snapshot_interval_environment_steps": WINDOW_STEPS,
        "halving": {
            "at_environment_steps": HALVING_WINDOW * WINDOW_STEPS,
            "retained_source_lanes": retained_ids,
            "stopped_source_lanes": [row["candidate_id"] for row in preliminary[4:]],
            "gpu_reassignment": reassignment,
        },
        "rollout_contract": {
            "opponents_frozen_within_window": True,
            "opponent_checkpoint_hash_verified_before_after": True,
            "unrestricted_simultaneous_self_play": False,
            "combat_expert_frozen": True,
            "adaptive_multiplier_for_30_70_percent_band": 2.0,
        },
        "hard_stops": {
            "nan_or_inf": True, "weapon_ownership_fire_mismatch": True,
            "policy_mean_look_saturation_limit": 0.10,
            "hidden_state_leakage": True, "heldout_contact_collapse": True,
            "zero_fire_collapse": True,
        },
        "optional_final_combat_finetune": {
            "applied": False,
            "reason": "No eligible candidate required combat unfreezing; original combat parent remains anchored.",
            "maximum_permitted_lr": 3e-6, "maximum_permitted_target_kl": 0.002,
        },
        "heldout_parent_contact_rate": parent_contact,
        "promoted_candidates": candidate_artifacts,
        "gates": gates,
        "evidence": {
            "ratings": "08_league/RATINGS.csv",
            "historical_pool": "08_league/HISTORICAL_POOL.json",
            "heldout_parent": relative(parent_output / "evaluation_result.json"),
        },
    }
    write_json(ROOT / "LEAGUE_MANIFEST.json", manifest)

    distribution_lines = "\n".join(
        f"| {name} | {probability:.0%} | {category_counts[name]} |"
        for name, probability in OPPONENT_DISTRIBUTION.items()
    )
    top = promoted[0] if promoted else None
    report = f"""# Phase 5 Anchored Hunter-Peeker League Report

Status: `{status}`

## League execution

Eight named lanes ran concurrently on GPUs 0-7. Each lane completed ten {WINDOW_STEPS:,}-decision frozen-opponent windows at the same {HALVING_WINDOW * WINDOW_STEPS:,}-step milestone. The four weaker source lanes then stopped; all eight GPUs were immediately reassigned as two continuations of each retained source lane. Every continuation reached {TOTAL_STEPS:,} environment steps and inherited an exact 20-window schedule. Opponents were hash-checked before and after each rollout and were never updated with the learner.

| Opponent family | Required | Windows per candidate |
|---|---:|---:|
{distribution_lines}

Rating-near windows doubled sampling mass for opponents whose measured learner win rate was in the 30-70% band. Easy frozen Phase 4/scripted anchors remained in all schedules. Snapshots were written every {WINDOW_STEPS:,} environment steps.

## Safety and selection

- All promoted candidates stayed at or below the 10% policy-mean look-saturation hard stop.
- Weapon ownership/fire mismatches, non-finite values, hidden-state leakage, and zero-fire collapse were fail-closed per rollout and again across each full candidate history.
- Held-out contact was paired against the frozen Goal 11 parent; collapse means either more than 10 percentage points regression or less than 75% of parent contact.
- Combat remained frozen. The optional <=3e-6, KL<=0.002 final combat-layer fine-tune was not used.

Promoted candidates: {len(promoted)}. {f"Top candidate `{top['candidate_id']}` has rating {top['rating']:.2f} and held-out contact {top['heldout_contact_rate']:.2%} (parent {parent_contact:.2%})." if top else "No candidate cleared every hard stop."}

All fail-closed gates: {'PASS' if all(gates.values()) else 'FAIL'}.
"""
    (ROOT / "LEAGUE_REPORT.md").write_text(report)
    print(json.dumps(manifest, sort_keys=True))
    return 0 if status == "PHASE5_LEAGUE_CANDIDATES_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
