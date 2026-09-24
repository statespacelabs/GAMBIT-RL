#!/usr/bin/env python3
"""Run the Goal 10 eight-GPU fair-memory/reacquisition sweep."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.navigation_ppo import REACQUIRE_VARIANTS, sha256_file  # noqa: E402

ROOT = PROJECT / "06_reacquire"
ENV = PROJECT / "BotArenaPhase5/Builds/Phase5Headless/BotArenaPhase5.x86_64"
PARENT = PROJECT / "06_map_general_hunter/phase5_map_general_hunter_champion.pt"
CERT = PROJECT / "06_map_general_hunter/HUNTER_CERTIFICATION.json"
COMBAT = PROJECT / "experiments/phase4_active_diagnostic/phase4_4_gvg_stimulus_library/training/generalist_safe_from_s050/checkpoints/snapshot_s030_u0030.pt"
NORMALIZER = PROJECT / "experiments/phase3v2/normalizers/phase3v2_c_local45_no_pressure.pt"
TRAIN = PROJECT / "experiments/phase5_map_general_headless_hunter/02_layouts/TRAIN_LAYOUTS.json"
VALIDATION = PROJECT / "experiments/phase5_map_general_headless_hunter/02_layouts/VALIDATION_LAYOUTS.json"


def run_parallel(jobs: list[tuple[int, str, list[str], Path]]) -> None:
    running: list[tuple[str, subprocess.Popen[Any], Any]] = []
    for gpu, name, command, log_path in jobs:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.update({
            "HIP_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": str(PROJECT),
        })
        wrapped = ["taskset", "-c", f"{gpu * 16}-{gpu * 16 + 15}", *command]
        log.write("COMMAND " + " ".join(wrapped) + "\n")
        log.flush()
        running.append((name, subprocess.Popen(
            wrapped, cwd=PROJECT, env=environment,
            stdout=log, stderr=subprocess.STDOUT, text=True,
        ), log))
    failures = []
    for name, process, log in running:
        code = process.wait()
        log.close()
        if code:
            failures.append(f"{name}:exit={code}")
    if failures:
        raise RuntimeError("parallel jobs failed: " + ", ".join(failures))


def visible_seeds(path: Path, count: int) -> list[int]:
    document = json.loads(path.read_text())
    return [int(row["seed"]) for row in document["layouts"] if row["requested_initial_los"]][:count]


def main() -> int:
    prerequisite = json.loads(CERT.read_text())
    if prerequisite.get("status") != "PHASE5_MAP_GENERAL_HUNTER_READY":
        raise RuntimeError("Goal 9 prerequisite is not ready")
    ROOT.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    train_seeds = visible_seeds(TRAIN, 7)
    validation_seeds = [550008]

    init_jobs = []
    for gpu, variant in enumerate(REACQUIRE_VARIANTS):
        output = ROOT / "sweep/init" / f"gpu{gpu}_{variant.branch.branch_id}"
        init_jobs.append((gpu, variant.branch.branch_id, [
            python, "scripts/train_navigation_ppo.py",
            "--goal7-navigator", str(PARENT), "--branch-id", variant.branch.branch_id,
            "--output-dir", str(output), "--environment-steps", "0",
            "--initialize-only", "--device", "cuda:0",
        ], output / "launch.log"))
    run_parallel(init_jobs)

    rollout_jobs = []
    for gpu, variant in enumerate(REACQUIRE_VARIANTS):
        source = ROOT / "sweep/init" / f"gpu{gpu}_{variant.branch.branch_id}/hunter_s000000.pt"
        output = ROOT / "sweep/rollout" / f"gpu{gpu}_{variant.branch.branch_id}"
        rollout_jobs.append((gpu, variant.branch.branch_id, [
            python, "scripts/collect_navigation_rollouts.py",
            "--env-path", str(ENV), "--hunter", str(source),
            "--combat-checkpoint", str(COMBAT), "--normalizer", str(NORMALIZER),
            "--prerequisite", str(CERT), "--manifest", str(TRAIN), "--split", "train",
            "--seeds", ",".join(map(str, train_seeds)), "--stage", "R0",
            "--preset", "reacquire_search", "--output-dir", str(output),
            "--run-id", f"goal10_rollout_gpu{gpu}", "--base-port", str(9000 + gpu),
            "--seed", str(61100 + gpu), "--target-decisions", "4096",
            "--time-scale", "15", "--device", "cuda:0",
            "--hunt-cue-dropout", str(variant.cue_dropout),
            "--hunt-dropout-min-seconds", "1", "--hunt-dropout-max-seconds", "4",
            "--search-stale-timeout-seconds", str(variant.stale_timeout_seconds),
            "--challenger-timeout-seconds", "20",
        ], output / "launch.log"))
    run_parallel(rollout_jobs)

    train_jobs = []
    for gpu, variant in enumerate(REACQUIRE_VARIANTS):
        init = ROOT / "sweep/init" / f"gpu{gpu}_{variant.branch.branch_id}/hunter_s000000.pt"
        rollout = ROOT / "sweep/rollout" / f"gpu{gpu}_{variant.branch.branch_id}/rollout.npz"
        output = ROOT / "sweep/train" / f"gpu{gpu}_{variant.branch.branch_id}"
        train_jobs.append((gpu, variant.branch.branch_id, [
            python, "scripts/train_navigation_ppo.py", "--goal7-navigator", str(PARENT),
            "--branch-id", variant.branch.branch_id, "--resume", str(init),
            "--rollout", str(rollout), "--stage", "R0", "--environment-steps", "4096",
            "--total-curriculum-steps", "4096", "--updates", "8", "--batch-size", "256",
            "--output-dir", str(output), "--device", "cuda:0",
        ], output / "launch.log"))
    run_parallel(train_jobs)

    eval_jobs = []
    for gpu, variant in enumerate(REACQUIRE_VARIANTS):
        navigator = ROOT / "sweep/train" / f"gpu{gpu}_{variant.branch.branch_id}/navigator_s004096.pt"
        output = ROOT / "sweep/validation" / f"gpu{gpu}_{variant.branch.branch_id}"
        eval_jobs.append((gpu, variant.branch.branch_id, [
            python, "scripts/evaluate_navigation.py", "--env-path", str(ENV),
            "--navigator", str(navigator), "--checkpoint", str(COMBAT),
            "--normalizer", str(NORMALIZER), "--prerequisite", str(CERT),
            "--manifest", str(VALIDATION), "--split", "validation",
            "--seeds", ",".join(map(str, validation_seeds)),
            "--preset", "reacquire_search", "--mode", "deterministic",
            "--output-dir", str(output), "--run-id", f"goal10_validation_gpu{gpu}",
            "--base-port", str(9100 + gpu), "--seed", str(61200 + gpu),
            "--time-scale", "15", "--device", "cuda:0",
            "--target-sessions-per-area", "7", "--challenger-timeout-seconds", "20",
            "--hunt-cue-dropout", str(variant.cue_dropout),
            "--hunt-dropout-min-seconds", "1", "--hunt-dropout-max-seconds", "4",
            "--search-stale-timeout-seconds", str(variant.stale_timeout_seconds),
        ], output / "launch.log"))
    run_parallel(eval_jobs)

    records = []
    for gpu, variant in enumerate(REACQUIRE_VARIANTS):
        stem = f"gpu{gpu}_{variant.branch.branch_id}"
        evaluation_path = ROOT / "sweep/validation" / stem / "evaluation_result.json"
        training_path = ROOT / "sweep/train" / stem / "training_result.json"
        evaluation = json.loads(evaluation_path.read_text())
        training = json.loads(training_path.read_text())
        records.append({
            "gpu": gpu, "branch_id": variant.branch.branch_id,
            "memory_steps": variant.memory_steps, "cue_dropout": variant.cue_dropout,
            "stale_timeout_seconds": variant.stale_timeout_seconds,
            "gru_hidden": variant.branch.gru_hidden,
            "reacquisition_reward": variant.reacquisition_reward,
            "route_change_penalty": variant.route_change_penalty,
            "environment_steps": 4096,
            "validation_reacquisition_rate": evaluation["reacquisition_rate"],
            "validation_timeout_rate": evaluation["timeouts"] / max(1, evaluation["sessions"]),
            "validation_stale_loop_rate": evaluation["stale_location_loop_rate"],
            "validation_stuck_rate": evaluation["stuck_rate"],
            "los_lost_events": evaluation["los_lost_events"],
            "reacquired_events": evaluation["reacquired_events"],
            "mean_approx_kl": training["mean_approx_kl"],
            "bc_auxiliary_weight": training["metadata"]["bc_auxiliary_weight"],
            "navigator": str((ROOT / "sweep/train" / stem / "navigator_s004096.pt").relative_to(PROJECT)),
            "evaluation": str(evaluation_path.relative_to(PROJECT)),
        })
    ranking = sorted(records, key=lambda row: (
        -row["validation_reacquisition_rate"], row["validation_timeout_rate"],
        row["validation_stale_loop_rate"], row["validation_stuck_rate"],
        row["mean_approx_kl"], row["branch_id"],
    ))
    winner = ranking[0]
    selected_source = PROJECT / winner["navigator"]
    selected = ROOT / "selected_reacquirer.pt"
    shutil.copy2(selected_source, selected)
    result = {
        "schema_version": "phase5_reacquisition_sweep_v001",
        "status": "PASS", "gpu_count": 8, "environment_step_milestone": 4096,
        "selection_used_heldout": False, "parent": str(PARENT.relative_to(PROJECT)),
        "parent_sha256": sha256_file(PARENT), "combat_expert_frozen": True,
        "train_seeds": train_seeds, "validation_seeds": validation_seeds,
        "records": records, "ranking": [row["branch_id"] for row in ranking],
        "selected_branch": winner["branch_id"], "selected_reacquirer": str(selected.relative_to(PROJECT)),
        "selected_sha256": sha256_file(selected),
    }
    (ROOT / "SWEEP_RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
