#!/usr/bin/env python3
"""Freeze and execute the Goal 3 CUDA examiner-bank round robin."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "experiments/paper_goal3_cuda_examiner_bank_v001"
RUNNER = ROOT / "scripts/examiner_match.py"
TOURNAMENT_SOURCE = ROOT / "scripts/symmetric_tournament.py"
BUILD = ROOT / (
    "experiments/phase6_population_league_psro/03_tournament/derived_unity/"
    "BotArenaPhase6Tournament/Builds/BotArena.x86_64"
)
LAYOUT_MANIFEST = ROOT / (
    "experiments/phase5_map_general_headless_hunter/02_layouts/"
    "VALIDATION_LAYOUTS.json"
)
LOCAL45_NORMALIZER = ROOT / (
    "experiments/phase3v2/normalizers/"
    "phase3v2_c_local45_no_pressure.pt"
)
ACTOR231_NORMALIZER = ROOT / (
    "experiments/phase5_map_general_headless_hunter/02_telemetry/"
    "normalizers/phase5_actor_obs_v001_normalizer_v001.json"
)
COMBAT_EXPERT = ROOT / (
    "experiments/phase4_active_diagnostic/phase4_4_gvg_stimulus_library/"
    "training/generalist_safe_from_s050/checkpoints/snapshot_s030_u0030.pt"
)

EXPECTED = {
    "build": "0bfe51325720c950b4ad75ce7c3b65595fd7e939908520ec33600919919327a8",
    "layout_manifest": "20593f1a6928c1fbff209205f9b8bde736bdd9fa0b98a54f6b8cf757875dae4f",
    "local45_normalizer": "28df2464221f65798ff4e4c98eb0d8d340f18b8765b4ca306fcbc45da4e89b5e",
    "actor231_normalizer": "d131048ccc5da298154156966d88117867aa434f360c909de573003c2e74b940",
    "combat_expert": "4ce907c625f228546e6f65df3a55021a8afc7b86f366927eb00e06e61f517c4d",
}
ROOMS_LAYOUT_SEEDS = [550002, 550007, 550012, 550017]
PAIRED_SEED_COUNT = 20

ANCHORS = {
    "bank_anchor_face_shooter": {
        "forward": 0.0,
        "strafe": 0.0,
        "max_look": 0.80,
        "shoot_threshold_deg": 14.0,
        "switch_period_decisions": 48,
    },
    "bank_anchor_strafe_shooter": {
        "forward": 0.0,
        "strafe": 0.65,
        "max_look": 0.80,
        "shoot_threshold_deg": 18.0,
        "switch_period_decisions": 48,
    },
    "bank_anchor_rusher": {
        "forward": 0.75,
        "strafe": 0.15,
        "max_look": 0.80,
        "shoot_threshold_deg": 22.0,
        "switch_period_decisions": 80,
    },
    "bank_anchor_kiter": {
        "forward": -0.55,
        "strafe": 0.30,
        "max_look": 0.80,
        "shoot_threshold_deg": 16.0,
        "switch_period_decisions": 72,
    },
    "bank_anchor_cover_sweeper": {
        "forward": 0.15,
        "strafe": 0.75,
        "max_look": 0.80,
        "shoot_threshold_deg": 18.0,
        "switch_period_decisions": 96,
    },
    "bank_anchor_search_pursuer": {
        "forward": 0.35,
        "strafe": 0.0,
        "max_look": 0.80,
        "shoot_threshold_deg": 24.0,
        "switch_period_decisions": 64,
    },
}

LEARNED = [
    {
        "canonical_id": "bank_original_combat_expert",
        "family": "phase4_visible_combat_only",
        "style": "humanlike stationary contact duelist; intentionally weak hunt",
        "diagnostic_purpose": "isolates contact-to-kill conversion from navigation",
        "path": (
            "experiments/phase4_active_diagnostic/phase4_4_gvg_stimulus_library/"
            "training/generalist_safe_from_s050/checkpoints/snapshot_s030_u0030.pt"
        ),
        "sha256": EXPECTED["combat_expert"],
        "combat_path": (
            "experiments/phase4_active_diagnostic/phase4_4_gvg_stimulus_library/"
            "training/generalist_safe_from_s050/checkpoints/snapshot_s030_u0030.pt"
        ),
        "combat_sha256": EXPECTED["combat_expert"],
    },
    {
        "canonical_id": "bank_phase4_anti_strafe",
        "family": "phase4_fair_search_combat",
        "style": "anti-strafe pressure specialist with fair hidden search",
        "diagnostic_purpose": "tests lateral movement and anti-strafe robustness",
        "path": (
            "experiments/phase4_active_diagnostic/phase4_4_gvg_stimulus_library/"
            "training/gvg_anti_strafe_pressure/checkpoints/snapshot_s030_u0030.pt"
        ),
        "sha256": "78343458d203d238b2077d72faeb135e7681c15851d3d384cbaef1a1e8481108",
    },
    {
        "canonical_id": "bank_phase4_generalist",
        "family": "phase4_fair_search_combat",
        "style": "balanced roster generalist with fair hidden search",
        "diagnostic_purpose": "broad combat and reacquisition benchmark",
        "path": (
            "experiments/phase4_active_diagnostic/phase4_4_gvg_stimulus_library/"
            "training/gvg_generalist_roster_mix/checkpoints/snapshot_s050_u0050.pt"
        ),
        "sha256": "b75beb4af9fe7ad244a6bacae65b2eb91861f44746a31bac0633985283cef20e",
    },
    {
        "canonical_id": "bank_phase4_obstacle_search",
        "family": "phase4_fair_search_combat",
        "style": "obstacle-search pursuer with strongest screened hunt",
        "diagnostic_purpose": "tests search, obstacle recovery, and conversion",
        "path": (
            "experiments/phase4_active_diagnostic/phase4_4_gvg_stimulus_library/"
            "phase4_4g2_10x_expansion/training/wave_01/"
            "w01_search_obstacle_pursuer_repaired_expansion/checkpoints/"
            "snapshot_s006_u0060.pt"
        ),
        "sha256": "f510d6a43714c53d0a1f0e4bafe512846e9f127428d1b038953263957c417eb9",
    },
    {
        "canonical_id": "bank_phase5_hunter",
        "family": "phase5_hybrid_navigator_combat",
        "style": "forward-aggressive map-general hunter",
        "diagnostic_purpose": "tests pursuit pressure and map-general hunt",
        "path": "06_map_general_hunter/phase5_map_general_hunter_champion.pt",
        "sha256": "4cdc378ac5689396847e3a201d729ec8c531f7efd8efd5700c1afb92cccc2b9e",
        "combat_path": (
            "experiments/phase4_active_diagnostic/phase4_4_gvg_stimulus_library/"
            "training/generalist_safe_from_s050/checkpoints/snapshot_s030_u0030.pt"
        ),
        "combat_sha256": EXPECTED["combat_expert"],
    },
    {
        "canonical_id": "bank_phase5_peeker",
        "family": "phase5_hybrid_navigator_combat",
        "style": "cautious cover-heavy peeker",
        "diagnostic_purpose": "tests cover interaction, patience, and reacquisition",
        "path": "09_champion/frozen/local_geometry_peeker.pt",
        "sha256": "b28eb5f85cacce5c53d34312a8688900bba552d30100793160cd09d194b00aa8",
        "combat_path": (
            "experiments/phase4_active_diagnostic/phase4_4_gvg_stimulus_library/"
            "training/generalist_safe_from_s050/checkpoints/snapshot_s030_u0030.pt"
        ),
        "combat_sha256": EXPECTED["combat_expert"],
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_once(path: Path, payload: Any) -> None:
    if path.is_file():
        if read_json(path) != payload:
            raise RuntimeError(f"frozen artifact differs: {path}")
        return
    atomic_json(path, payload)


def assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: {actual} != {expected}")


def anchor_spec() -> dict[str, Any]:
    return {
        "schema_version": "phase6_examiner_anchor_spec_v001",
        "status": "FROZEN",
        "action_schema": {
            "continuous": ["move_x", "move_z", "turn_yaw", "look_pitch"],
            "binary": ["fire", "jump", "reload", "reserved"],
            "range": [-1.0, 1.0],
        },
        "observation_policy": {
            "visible_target_teacher": (
                "actor LOS gate plus local45 indices 20 yaw, 21 pitch, 22 aim"
            ),
            "hidden_target_behavior": (
                "FairReacquisitionSearch and MapIndependentSafetyLayer over "
                "phase5_actor_obs_v001 only"
            ),
            "forbidden": [
                "critic suffix",
                "hidden-target coordinates",
                "absolute world coordinates",
                "map identity",
                "opponent identity",
                "NavMesh oracle",
            ],
        },
        "anchors": ANCHORS,
    }


def build_candidates(output: Path) -> list[dict[str, Any]]:
    spec_path = output / "00_contract/ANCHOR_CONTROLLER_SPEC.json"
    spec_sha = sha256_file(spec_path)
    runner_sha = sha256_file(RUNNER)
    candidates: list[dict[str, Any]] = []
    anchor_metadata = {
        "bank_anchor_face_shooter": (
            "stationary precision face shooter",
            "pure visible aim-and-fire conversion floor",
        ),
        "bank_anchor_strafe_shooter": (
            "periodic lateral strafe shooter",
            "lateral tracking and conversion anchor",
        ),
        "bank_anchor_rusher": (
            "forward-aggressive close-range rusher",
            "pressure and close-range conversion anchor",
        ),
        "bank_anchor_kiter": (
            "backpedaling evasive kiter",
            "pursuit and evasive-target diagnostic",
        ),
        "bank_anchor_cover_sweeper": (
            "wide periodic cover-changing sweeper",
            "angle change and cover reacquisition diagnostic",
        ),
        "bank_anchor_search_pursuer": (
            "moderate forward search pursuer",
            "search-to-contact transition anchor",
        ),
    }
    for canonical_id, parameters in ANCHORS.items():
        style, purpose = anchor_metadata[canonical_id]
        controller_identity = {
            "adapter_sha256": runner_sha,
            "anchor_spec_sha256": spec_sha,
            "canonical_id": canonical_id,
            "parameters": parameters,
        }
        candidates.append(
            {
                "canonical_id": canonical_id,
                "candidate_class": "parametric_anchor",
                "family": "fair_parametric_anchor",
                "style": style,
                "difficulty_parameters": parameters,
                "diagnostic_purpose": purpose,
                "checkpoint": str(spec_path.relative_to(ROOT)),
                "checkpoint_sha256": spec_sha,
                "model_or_controller_sha256": canonical_hash(controller_identity),
                "controller_identity": controller_identity,
                "combat_checkpoint": str(COMBAT_EXPERT.relative_to(ROOT)),
                "combat_sha256": EXPECTED["combat_expert"],
            }
        )
    for source in LEARNED:
        row = dict(source)
        row["candidate_class"] = "learned_or_hybrid"
        row["difficulty_parameters"] = {
            "action_mode": "deterministic",
            "sampling_temperature": 0.0,
            "training_updates": 0,
        }
        row["checkpoint"] = row.pop("path")
        row["checkpoint_sha256"] = row.pop("sha256")
        row["model_or_controller_sha256"] = row["checkpoint_sha256"]
        if "combat_path" not in row:
            row["combat_path"] = row["checkpoint"]
            row["combat_sha256"] = row["checkpoint_sha256"]
        row["combat_checkpoint"] = row.pop("combat_path")
        candidates.append(row)
    for row in candidates:
        row["normalizers"] = {
            "local45": {
                "path": str(LOCAL45_NORMALIZER.relative_to(ROOT)),
                "sha256": EXPECTED["local45_normalizer"],
                "update": False,
            },
            "actor231": {
                "path": str(ACTOR231_NORMALIZER.relative_to(ROOT)),
                "sha256": EXPECTED["actor231_normalizer"],
                "update": False,
                "runtime_relevance": (
                    "navigator/fair-search schema freeze; not a mutable input"
                ),
            },
        }
        row["observation_schema"] = {
            "actor": "phase5_actor_obs_v001",
            "actor_dim": 231,
            "visible_combat": "phase3v2_c_local45",
            "visible_combat_dim": 45,
            "critic_suffix_available_to_actor": False,
        }
        row["action_schema"] = "phase3v2_eight_action_v001"
        row["fairness"] = {
            "hidden_target_coordinates": False,
            "map_identity": False,
            "opponent_identity": False,
            "navmesh_oracle": False,
            "final_heldout_used": False,
        }
    return candidates


def verify_inputs(candidates: list[dict[str, Any]]) -> None:
    assert_hash(BUILD, EXPECTED["build"], "CUDA tournament build")
    assert_hash(
        LAYOUT_MANIFEST, EXPECTED["layout_manifest"], "validation manifest"
    )
    assert_hash(
        LOCAL45_NORMALIZER,
        EXPECTED["local45_normalizer"],
        "local45 normalizer",
    )
    assert_hash(
        ACTOR231_NORMALIZER,
        EXPECTED["actor231_normalizer"],
        "actor231 normalizer",
    )
    assert_hash(COMBAT_EXPERT, EXPECTED["combat_expert"], "combat expert")
    if not RUNNER.is_file() or not TOURNAMENT_SOURCE.is_file():
        raise RuntimeError("missing Goal 3 runner source")
    for row in candidates:
        assert_hash(
            ROOT / row["checkpoint"],
            row["checkpoint_sha256"],
            row["canonical_id"],
        )
        assert_hash(
            ROOT / row["combat_checkpoint"],
            row["combat_sha256"],
            row["canonical_id"] + " combat",
        )


def make_schedule(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = sorted(row["canonical_id"] for row in candidates)
    canonical_pairs = list(itertools.combinations(ids, 2))
    pair_index = {pair: index for index, pair in enumerate(canonical_pairs)}
    shuffled_pairs = list(canonical_pairs)
    random.Random(603031).shuffle(shuffled_pairs)
    schedule: list[dict[str, Any]] = []
    execution_index = 0
    for paired_seed_index in range(PAIRED_SEED_COUNT):
        layout_seed = ROOMS_LAYOUT_SEEDS[
            paired_seed_index % len(ROOMS_LAYOUT_SEEDS)
        ]
        runtime_seed = 960000 + paired_seed_index
        for pair in shuffled_pairs:
            for side_assignment in ("base", "swapped"):
                pidx = pair_index[pair]
                schedule.append(
                    {
                        "execution_index": execution_index,
                        "run_id": (
                            f"pair_{pidx:02d}_seed_{paired_seed_index:02d}_"
                            f"{side_assignment}"
                        ),
                        "pair_index": pidx,
                        "policy_a_id": pair[0],
                        "policy_b_id": pair[1],
                        "paired_seed_index": paired_seed_index,
                        "paired_seed_set_id": (
                            f"rooms_doorways_paired_{paired_seed_index:02d}"
                        ),
                        "runtime_seed": runtime_seed,
                        "layout_seed": layout_seed,
                        "layout_family": "rooms_doorways",
                        "side_assignment": side_assignment,
                    }
                )
                execution_index += 1
    return schedule


def prepare(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contract_dir = output / "00_contract"
    contract_dir.mkdir(parents=True, exist_ok=True)
    write_once(contract_dir / "ANCHOR_CONTROLLER_SPEC.json", anchor_spec())
    candidates = build_candidates(output)
    verify_inputs(candidates)
    schedule = make_schedule(candidates)
    candidate_payload = {
        "schema_version": "phase6_cuda_examiner_candidates_v001",
        "status": "FROZEN_PRE_EXECUTION",
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    schedule_payload = {
        "schema_version": "phase6_cuda_examiner_round_robin_schedule_v001",
        "status": "FROZEN_PRE_EXECUTION",
        "candidate_count": len(candidates),
        "unordered_pair_count": len(candidates) * (len(candidates) - 1) // 2,
        "paired_seeds_per_unordered_pair": PAIRED_SEED_COUNT,
        "side_assignments": ["base", "swapped"],
        "execution_count": len(schedule),
        "layout_family": "rooms_doorways",
        "layout_seeds": ROOMS_LAYOUT_SEEDS,
        "schedule": schedule,
    }
    write_once(contract_dir / "CANDIDATE_LIBRARY.json", candidate_payload)
    write_once(contract_dir / "ROUND_ROBIN_SCHEDULE.json", schedule_payload)
    frozen_hashes = {
        "match_adapter": {
            "path": str(RUNNER.relative_to(ROOT)),
            "sha256": sha256_file(RUNNER),
        },
        "tournament_source": {
            "path": str(TOURNAMENT_SOURCE.relative_to(ROOT)),
            "sha256": sha256_file(TOURNAMENT_SOURCE),
        },
        "tournament_build": {
            "path": str(BUILD.relative_to(ROOT)),
            "sha256": EXPECTED["build"],
        },
        "validation_manifest": {
            "path": str(LAYOUT_MANIFEST.relative_to(ROOT)),
            "sha256": EXPECTED["layout_manifest"],
            "internal_manifest_sha256": (
                "fc3c432a17b80290ef04b6df30acc07bbd5ed065ea487dc5995c7e7b4916b777"
            ),
        },
        "local45_normalizer": {
            "path": str(LOCAL45_NORMALIZER.relative_to(ROOT)),
            "sha256": EXPECTED["local45_normalizer"],
        },
        "actor231_normalizer": {
            "path": str(ACTOR231_NORMALIZER.relative_to(ROOT)),
            "sha256": EXPECTED["actor231_normalizer"],
        },
    }
    contract = {
        "schema_version": "phase6_cuda_examiner_calibration_contract_v001",
        "status": "FROZEN_PRE_EXECUTION",
        "prerequisite": "TANDEMFPS_CUDA_BOT_BANK_SCREENED",
        "training_performed": False,
        "calibration_device": "cuda:0",
        "primary_target": {"minimum": 6, "maximum": 8, "selected": 8},
        "reserve_target": 2,
        "composition_target": {
            "primary_parametric_anchors": 4,
            "primary_learned_or_hybrid": 4,
            "reserve_parametric_anchors": 1,
            "reserve_learned_or_hybrid": 1,
        },
        "selection_rule": (
            "terminal-only Bayesian/Bradley-Terry calibration followed by "
            "within-class payoff-profile diversity and reliability selection"
        ),
        "round_robin": {
            "unordered_pairs": len(candidates) * (len(candidates) - 1) // 2,
            "paired_seeds_per_pair": PAIRED_SEED_COUNT,
            "side_swapped": True,
            "total_matches": len(schedule),
            "map_source": "current Phase 5 procedural tournament build",
            "validation_layout_family": "rooms_doorways",
            "final_heldout_used": False,
            "terminal_and_damage_complete_required": True,
        },
        "fairness": {
            "actor_schema": "phase5_actor_obs_v001",
            "hidden_target_coordinates": False,
            "absolute_world_coordinates": False,
            "map_identity": False,
            "opponent_identity": False,
            "navmesh_oracle": False,
            "human_controller": False,
        },
        "frozen_hashes": frozen_hashes,
    }
    write_once(contract_dir / "CALIBRATION_CONTRACT.json", contract)
    print(
        json.dumps(
            {
                "status": "PREPARED",
                "output": str(output),
                "candidates": len(candidates),
                "matches": len(schedule),
                "schedule_sha256": sha256_file(
                    contract_dir / "ROUND_ROBIN_SCHEDULE.json"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return candidates, schedule


def command_for(
    item: dict[str, Any],
    candidate_by_id: dict[str, dict[str, Any]],
    output: Path,
    base_port: int,
    time_scale: float,
    device: str,
) -> tuple[list[str], Path, Path, Path]:
    a = candidate_by_id[item["policy_a_id"]]
    b = candidate_by_id[item["policy_b_id"]]
    run_dir = output / "01_round_robin/raw" / item["run_id"]
    logs = output / "01_round_robin/logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / (str(item["run_id"]) + ".stdout.log")
    stderr_path = logs / (str(item["run_id"]) + ".stderr.log")
    command = [
        sys.executable,
        str(RUNNER),
        "--env-path",
        str(BUILD),
        "--policy-a-id",
        a["canonical_id"],
        "--policy-a",
        str(ROOT / a["checkpoint"]),
        "--policy-a-sha256",
        a["checkpoint_sha256"],
        "--policy-b-id",
        b["canonical_id"],
        "--policy-b",
        str(ROOT / b["checkpoint"]),
        "--policy-b-sha256",
        b["checkpoint_sha256"],
        "--combat-a",
        str(ROOT / a["combat_checkpoint"]),
        "--combat-a-sha256",
        a["combat_sha256"],
        "--combat-b",
        str(ROOT / b["combat_checkpoint"]),
        "--combat-b-sha256",
        b["combat_sha256"],
        "--normalizer-a",
        str(LOCAL45_NORMALIZER),
        "--normalizer-a-sha256",
        EXPECTED["local45_normalizer"],
        "--normalizer-b",
        str(LOCAL45_NORMALIZER),
        "--normalizer-b-sha256",
        EXPECTED["local45_normalizer"],
        "--mode-a",
        "deterministic",
        "--mode-b",
        "deterministic",
        "--side-assignment",
        item["side_assignment"],
        "--layout-manifest",
        str(LAYOUT_MANIFEST),
        "--layout-seed",
        str(item["layout_seed"]),
        "--num-areas",
        "1",
        "--target-matches",
        "1",
        "--match-timeout-seconds",
        "30",
        "--output-dir",
        str(run_dir),
        "--base-port",
        str(base_port + int(item["execution_index"])),
        "--seed",
        str(item["runtime_seed"]),
        "--time-scale",
        str(time_scale),
        "--timeout",
        "180",
        "--max-fixed-steps",
        "450000",
        "--device",
        device,
    ]
    return command, run_dir, stdout_path, stderr_path


def valid_existing(run_dir: Path) -> bool:
    summary_path = run_dir / "run_summary.json"
    matches_path = run_dir / "matches.jsonl"
    if not summary_path.is_file() or not matches_path.is_file():
        return False
    summary = read_json(summary_path)
    return (
        summary.get("status") == "PASS"
        and summary.get("matches_completed") == 1
        and summary.get("target_matches") == 1
        and sum(1 for _ in matches_path.open(encoding="utf-8")) == 1
    )


def run_one(
    item: dict[str, Any],
    candidate_by_id: dict[str, dict[str, Any]],
    output: Path,
    base_port: int,
    time_scale: float,
    device: str,
) -> dict[str, Any]:
    command, run_dir, stdout_path, stderr_path = command_for(
        item, candidate_by_id, output, base_port, time_scale, device
    )
    if valid_existing(run_dir):
        summary = read_json(run_dir / "run_summary.json")
        return {
            "run_id": item["run_id"],
            "status": "PASS",
            "resumed": True,
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "matches_sha256": summary.get("matches_sha256"),
        }
    if run_dir.exists() or stdout_path.exists() or stderr_path.exists():
        raise RuntimeError(
            "incomplete or untracked prior output blocks safe resume: " + str(item["run_id"])
        )
    started = time.perf_counter()
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HIP_VISIBLE_DEVICES": "",
            "MVH_RENDERED": "0",
            "PHASE5_NAVMESH_UPPER_BOUND": "0",
            "PHASE5_ENABLE_NAVMESH_ORACLE": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "RESEARCH_HUD_STATE_PATH": str(
                output / "01_round_robin/hud" / (str(item["run_id"]) + ".json")
            ),
        }
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    summary_path = run_dir / "run_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    status = (
        "PASS"
        if completed.returncode == 0
        and summary.get("status") == "PASS"
        and valid_existing(run_dir)
        else "FAIL"
    )
    return {
        "run_id": item["run_id"],
        "status": status,
        "resumed": False,
        "returncode": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "failure_reason": summary.get("failure_reason") or completed.stderr[-1000:],
        "matches_sha256": summary.get("matches_sha256"),
    }


def execute(
    output: Path,
    candidates: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    *,
    base_port: int,
    max_parallel: int,
    time_scale: float,
    device: str,
) -> None:
    verify_inputs(candidates)
    contract_dir = output / "00_contract"
    frozen_adapter_sha = read_json(
        contract_dir / "CALIBRATION_CONTRACT.json"
    )["frozen_hashes"]["match_adapter"]["sha256"]
    assert_hash(RUNNER, frozen_adapter_sha, "frozen match adapter")
    manifest_path = output / "01_round_robin/EXECUTION_MANIFEST.json"
    prior = read_json(manifest_path) if manifest_path.is_file() else {}
    result_by_id = {
        row["run_id"]: row for row in prior.get("executions", [])
    }
    pending = []
    for item in schedule:
        run_dir = output / "01_round_robin/raw" / item["run_id"]
        if valid_existing(run_dir):
            result_by_id[item["run_id"]] = {
                "run_id": item["run_id"],
                "status": "PASS",
                "resumed": True,
                "matches_sha256": read_json(
                    run_dir / "run_summary.json"
                ).get("matches_sha256"),
            }
        else:
            if run_dir.exists():
                raise RuntimeError(
                    "incomplete prior run requires audit: " + str(item["run_id"])
                )
            pending.append(item)
    candidate_by_id = {
        row["canonical_id"]: row for row in candidates
    }
    started = time.time()
    print(
        json.dumps(
            {
                "status": "RUNNING",
                "completed": len(schedule) - len(pending),
                "pending": len(pending),
                "max_parallel": max_parallel,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    def save(status: str) -> None:
        ordered = [
            result_by_id[item["run_id"]]
            for item in schedule
            if item["run_id"] in result_by_id
        ]
        atomic_json(
            manifest_path,
            {
                "schema_version": "phase6_cuda_examiner_execution_manifest_v001",
                "status": status,
                "scheduled": len(schedule),
                "completed": len(ordered),
                "pass_count": sum(row["status"] == "PASS" for row in ordered),
                "fail_count": sum(row["status"] != "PASS" for row in ordered),
                "wall_seconds_current_invocation": time.time() - started,
                "executions": ordered,
            },
        )

    failures: list[dict[str, Any]] = []
    completed_now = 0
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(
                run_one,
                item,
                candidate_by_id,
                output,
                base_port,
                time_scale,
                device,
            ): item
            for item in pending
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "run_id": item["run_id"],
                    "status": "FAIL",
                    "resumed": False,
                    "failure_reason": repr(exc),
                }
            result_by_id[item["run_id"]] = result
            completed_now += 1
            if result["status"] != "PASS":
                failures.append(result)
            if completed_now % 20 == 0 or failures:
                save("FAIL" if failures else "RUNNING")
                print(
                    json.dumps(
                        {
                            "completed_this_invocation": completed_now,
                            "total_completed": len(result_by_id),
                            "scheduled": len(schedule),
                            "failures": len(failures),
                            "elapsed_seconds": round(time.time() - started, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if failures:
                for queued in futures:
                    queued.cancel()
                break
    if failures:
        save("FAIL")
        raise RuntimeError(f"round robin failure: {failures[0]}")
    if len(result_by_id) != len(schedule):
        save("INCOMPLETE")
        raise RuntimeError("round robin did not complete all scheduled matches")
    verify_inputs(candidates)
    assert_hash(RUNNER, frozen_adapter_sha, "post-run match adapter")
    save("COMPLETE")
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "matches": len(schedule),
                "elapsed_seconds": round(time.time() - started, 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare", "run")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-port", type=int, default=21000)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output.resolve()
    candidates, schedule = prepare(output)
    if args.command == "run":
        execute(
            output,
            candidates,
            schedule,
            base_port=args.base_port,
            max_parallel=args.max_parallel,
            time_scale=args.time_scale,
            device=args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
