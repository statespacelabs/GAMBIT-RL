#!/usr/bin/env python3
"""Launch the Goal 9 eight-GPU curriculum with fixed-step successive halving."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.navigation_ppo import BRANCHES, rank_for_halving, sha256_file  # noqa: E402


@dataclass
class Candidate:
    gpu: int
    candidate_id: str
    branch_id: str
    hunter: Path
    navigator: Path
    environment_steps: int
    lineage: list[str]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="06_map_general_hunter")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--env-path", default="BotArenaPhase5/Builds/Phase5Headless/BotArenaPhase5.x86_64")
    parser.add_argument("--time-scale", type=float, default=10.0)
    return parser.parse_args()


def paths(root: Path) -> dict[str, Path]:
    return {
        "curriculum": root / "CURRICULUM.json",
        "goal7": PROJECT / "experiments/phase5_map_general_headless_hunter/04_dagger/selected_navigator.pt",
        "goal7_cert": PROJECT / "experiments/phase5_map_general_headless_hunter/04_dagger/NAVIGATOR_CERTIFICATION.json",
        "goal6": PROJECT / "experiments/phase5_map_general_headless_hunter/04_dagger/DAGGER_DATASET_MANIFEST.json",
        "combat": PROJECT / "experiments/phase4_active_diagnostic/phase4_4_gvg_stimulus_library/training/generalist_safe_from_s050/checkpoints/snapshot_s030_u0030.pt",
        "normalizer": PROJECT / "experiments/phase3v2/normalizers/phase3v2_c_local45_no_pressure.pt",
        "train": PROJECT / "experiments/phase5_map_general_headless_hunter/02_layouts/TRAIN_LAYOUTS.json",
        "validation": PROJECT / "experiments/phase5_map_general_headless_hunter/02_layouts/VALIDATION_LAYOUTS.json",
    }


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in command)


def run_parallel(jobs: list[tuple[int, str, list[str], Path]]) -> None:
    processes: list[tuple[str, subprocess.Popen, Any]] = []
    for gpu, name, command, log_path in jobs:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.update({
            "HIP_VISIBLE_DEVICES": str(gpu),
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": str(PROJECT),
        })
        cpu_start = gpu * 16
        wrapped = ["taskset", "-c", f"{cpu_start}-{cpu_start + 15}", *command]
        log.write("COMMAND " + command_text(wrapped) + "\n")
        log.flush()
        process = subprocess.Popen(
            wrapped, cwd=PROJECT, env=environment,
            stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        processes.append((name, process, log))
    failures: list[str] = []
    for name, process, log in processes:
        code = process.wait()
        log.close()
        if code != 0:
            failures.append(f"{name}:exit={code}")
    if failures:
        raise RuntimeError("parallel jobs failed: " + ", ".join(failures))


def initialize(config: argparse.Namespace, root: Path, fixed: dict[str, Path]) -> list[Candidate]:
    jobs = []
    candidates = []
    for gpu, branch in enumerate(BRANCHES):
        output = root / "sweep/init" / f"gpu{gpu}_{branch.branch_id}"
        command = [
            config.python, "scripts/train_navigation_ppo.py",
            "--goal7-navigator", str(fixed["goal7"]),
            "--branch-id", branch.branch_id,
            "--output-dir", str(output),
            "--environment-steps", "0", "--initialize-only",
            "--candidate-id", branch.branch_id,
            "--device", "cuda:0",
        ]
        jobs.append((gpu, branch.branch_id, command, output / "launch.log"))
        candidates.append(Candidate(
            gpu, branch.branch_id, branch.branch_id,
            output / "hunter_s000000.pt", output / "navigator_s000000.pt",
            0, [branch.branch_id],
        ))
    run_parallel(jobs)
    return candidates


def collect_stage(
    config: argparse.Namespace,
    root: Path,
    fixed: dict[str, Path],
    stage: dict[str, Any],
    candidates: list[Candidate],
) -> dict[int, list[Path]]:
    stage_name = stage["stage"]
    increment_by_gpu = {
        candidate.gpu: int(stage["milestone_steps"]) - candidate.environment_steps
        for candidate in candidates
    }
    if any(value <= 0 for value in increment_by_gpu.values()):
        raise RuntimeError(f"non-positive {stage_name} fixed-step increment")
    rollout_paths: dict[int, list[Path]] = {candidate.gpu: [] for candidate in candidates}
    waves = [("scripted", False, 1.0)]
    if stage_name == "H5":
        waves = [("scripted", False, 0.5), ("neural", True, 0.5)]
    for wave_index, (wave_name, neural, fraction) in enumerate(waves):
        jobs = []
        for candidate in candidates:
            count = int(increment_by_gpu[candidate.gpu] * fraction)
            output = root / f"sweep/{stage_name}/rollouts" / f"gpu{candidate.gpu}_{candidate.candidate_id}_{wave_name}"
            command = [
                config.python, "scripts/collect_navigation_rollouts.py",
                "--env-path", str((PROJECT / config.env_path).resolve()),
                "--hunter", str(candidate.hunter),
                "--combat-checkpoint", str(fixed["combat"]),
                "--normalizer", str(fixed["normalizer"]),
                "--prerequisite", str(fixed["goal7_cert"]),
                "--manifest", str(fixed["train"]), "--split", "train",
                "--seeds", ",".join(str(value) for value in stage["seeds"]),
                "--stage", stage_name, "--preset", stage["preset"],
                "--output-dir", str(output),
                "--run-id", f"goal9_{stage_name.lower()}_gpu{candidate.gpu}_{wave_name}",
                "--base-port", str(6800 + int(stage_name[1]) * 100 + wave_index * 20 + candidate.gpu),
                "--seed", str(59100 + int(stage_name[1]) * 100 + candidate.gpu),
                "--target-decisions", str(count),
                "--time-scale", str(config.time_scale), "--device", "cuda:0",
            ]
            if neural:
                command.append("--neural-opponent")
            jobs.append((candidate.gpu, f"{stage_name}:{candidate.candidate_id}:{wave_name}", command, output / "launch.log"))
            rollout_paths[candidate.gpu].append(output / "rollout.npz")
        run_parallel(jobs)
    return rollout_paths


def train_stage(
    config: argparse.Namespace,
    root: Path,
    fixed: dict[str, Path],
    stage: dict[str, Any],
    candidates: list[Candidate],
    rollouts: dict[int, list[Path]],
) -> list[Candidate]:
    jobs = []
    outputs: dict[int, Path] = {}
    for candidate in candidates:
        output = root / f"sweep/{stage['stage']}/train" / f"gpu{candidate.gpu}_{candidate.candidate_id}"
        outputs[candidate.gpu] = output
        command = [
            config.python, "scripts/train_navigation_ppo.py",
            "--goal7-navigator", str(fixed["goal7"]),
            "--branch-id", candidate.branch_id,
            "--output-dir", str(output), "--resume", str(candidate.hunter),
            "--stage", stage["stage"],
            "--environment-steps", str(stage["milestone_steps"]),
            "--candidate-id", candidate.candidate_id,
            "--updates", "8", "--batch-size", "256", "--device", "cuda:0",
        ]
        for rollout in rollouts[candidate.gpu]:
            command.extend(("--rollout", str(rollout)))
        jobs.append((candidate.gpu, f"train:{stage['stage']}:{candidate.candidate_id}", command, output / "launch.log"))
    run_parallel(jobs)
    updated = []
    for candidate in candidates:
        output = outputs[candidate.gpu]
        steps = int(stage["milestone_steps"])
        result = json.loads((output / "training_result.json").read_text())
        if result.get("status") != "PASS" or float(result.get("mean_approx_kl", 1)) >= 0.01:
            raise RuntimeError(f"training gate failed: {candidate.candidate_id}")
        updated.append(Candidate(
            candidate.gpu, candidate.candidate_id, result["branch_id"],
            output / f"hunter_s{steps:06d}.pt",
            output / f"navigator_s{steps:06d}.pt",
            steps, candidate.lineage,
        ))
    return updated


def evaluate_milestone(
    config: argparse.Namespace,
    root: Path,
    fixed: dict[str, Path],
    stage_name: str,
    candidates: list[Candidate],
    validation_seeds: list[int],
) -> list[dict[str, Any]]:
    jobs = []
    outputs: dict[int, Path] = {}
    for candidate in candidates:
        output = root / f"sweep/{stage_name}/validation" / f"gpu{candidate.gpu}_{candidate.candidate_id}"
        outputs[candidate.gpu] = output
        command = [
            config.python, "scripts/evaluate_navigation.py",
            "--env-path", str((PROJECT / config.env_path).resolve()),
            "--navigator", str(candidate.navigator),
            "--checkpoint", str(fixed["combat"]),
            "--normalizer", str(fixed["normalizer"]),
            "--prerequisite", str(fixed["goal6"]),
            "--manifest", str(fixed["validation"]), "--split", "validation",
            "--seeds", ",".join(str(value) for value in validation_seeds),
            "--preset", "hunt_probe", "--mode", "deterministic",
            "--output-dir", str(output),
            "--run-id", f"goal9_{stage_name.lower()}_validation_gpu{candidate.gpu}",
            "--base-port", str(7600 + int(stage_name[1]) * 20 + candidate.gpu),
            "--seed", str(59600 + candidate.gpu), "--time-scale", str(config.time_scale),
            "--device", "cuda:0", "--target-sessions-per-area", "1",
            "--challenger-timeout-seconds", "30",
        ]
        jobs.append((candidate.gpu, f"eval:{stage_name}:{candidate.candidate_id}", command, output / "launch.log"))
    run_parallel(jobs)
    records = []
    for candidate in candidates:
        result = json.loads((outputs[candidate.gpu] / "evaluation_result.json").read_text())
        sessions = max(1, int(result["sessions"]))
        training = json.loads((candidate.hunter.parent / "training_result.json").read_text())
        records.append({
            "candidate_id": candidate.candidate_id,
            "gpu": candidate.gpu,
            "branch_id": candidate.branch_id,
            "environment_steps": candidate.environment_steps,
            "validation_contact_rate": float(result["contact_rate"]),
            "validation_timeout_rate": float(result["timeouts"]) / sessions,
            "validation_stuck_rate": float(result["stuck_rate"]),
            "mean_approx_kl": float(training["mean_approx_kl"]),
            "navigator_sha256": sha256_file(candidate.navigator),
            "evaluation_result": str((outputs[candidate.gpu] / "evaluation_result.json").relative_to(PROJECT)),
        })
    return rank_for_halving(records)


def reassign(candidates: list[Candidate], ranking: list[dict[str, Any]], retain: int, stage: str) -> list[Candidate]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    retained_ids = [row["candidate_id"] for row in ranking[:retain]]
    retained = [by_id[candidate_id] for candidate_id in retained_ids]
    occupied = {candidate.gpu for candidate in retained}
    free = [gpu for gpu in range(8) if gpu not in occupied]
    top_two = retained[:2]
    output = list(retained)
    for index, gpu in enumerate(free):
        parent = top_two[index % len(top_two)]
        candidate_id = f"{parent.branch_id}_{stage.lower()}_reuse_gpu{gpu}"
        output.append(Candidate(
            gpu, candidate_id, parent.branch_id,
            parent.hunter, parent.navigator, parent.environment_steps,
            [*parent.lineage, candidate_id],
        ))
    return sorted(output, key=lambda candidate: candidate.gpu)


def main() -> int:
    config = arguments()
    root = (PROJECT / config.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    fixed = paths(root)
    curriculum = json.loads(fixed["curriculum"].read_text())
    prerequisite = json.loads(fixed["goal7_cert"].read_text())
    if prerequisite.get("status") != "PHASE5_DAGGER_NAVIGATOR_READY":
        raise RuntimeError("Goal 7 prerequisite is not ready")
    candidates = initialize(config, root, fixed)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for stage in curriculum["stages"]:
        rollouts = collect_stage(config, root, fixed, stage, candidates)
        candidates = train_stage(config, root, fixed, stage, candidates, rollouts)
        if stage["stage"] in {"H1", "H3", "H5"}:
            ranking = evaluate_milestone(
                config, root, fixed, stage["stage"], candidates,
                curriculum["selection_validation_seeds"],
            )
            history.append({"stage": stage["stage"], "ranking": ranking})
            if stage["stage"] == "H1":
                candidates = reassign(candidates, ranking, 4, "H1")
            elif stage["stage"] == "H3":
                candidates = reassign(candidates, ranking, 2, "H3")
    final_ranking = history[-1]["ranking"]
    selected_id = final_ranking[0]["candidate_id"]
    selected = next(candidate for candidate in candidates if candidate.candidate_id == selected_id)
    manifest = {
        "schema_version": "phase5_map_general_hunter_sweep_v001",
        "status": "PASS",
        "gpu_count": 8,
        "all_gpus_used_each_stage": True,
        "fixed_environment_step_milestones": [row["milestone_steps"] for row in curriculum["stages"]],
        "successive_halving": curriculum["successive_halving"],
        "history": history,
        "selected": {
            "candidate_id": selected.candidate_id,
            "branch_id": selected.branch_id,
            "hunter_checkpoint": str(selected.hunter.relative_to(PROJECT)),
            "hunter_sha256": sha256_file(selected.hunter),
            "navigator_checkpoint": str(selected.navigator.relative_to(PROJECT)),
            "navigator_sha256": sha256_file(selected.navigator),
            "environment_steps": selected.environment_steps,
            "lineage": selected.lineage,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (root / "SWEEP_RESULTS.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["selected"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
