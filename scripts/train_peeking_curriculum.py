#!/usr/bin/env python3
"""Run Goal 11 P0-P5 with eight-GPU successive halving (8 -> 4 -> 2 -> 1)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.navigation_ppo import PEEKER_VARIANTS, sha256_file  # noqa: E402

ROOT = PROJECT / "07_local_geometry_peeker"
ENV = PROJECT / "BotArenaPhase5/Builds/Phase5Headless/BotArenaPhase5.x86_64"
PARENT = PROJECT / "06_reacquire/selected_reacquirer.pt"
CERT = PROJECT / "06_reacquire/REACQUISITION_CERTIFICATION.json"
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
    layouts = json.loads(path.read_text())["layouts"]
    return [int(row["seed"]) for row in layouts if not row["requested_initial_los"]][:count]


def live_command(
    python: str, hunter: Path, output: Path, *, stage: str, preset: str,
    seeds: Iterable[int], split: str, manifest: Path, port: int, seed: int,
    target: int, neural: bool = False,
) -> list[str]:
    command = [
        python, "scripts/collect_navigation_rollouts.py", "--env-path", str(ENV),
        "--hunter", str(hunter), "--combat-checkpoint", str(COMBAT),
        "--normalizer", str(NORMALIZER), "--prerequisite", str(CERT),
        "--manifest", str(manifest), "--split", split,
        "--seeds", ",".join(map(str, seeds)), "--stage", stage,
        "--preset", preset, "--output-dir", str(output),
        "--run-id", output.name, "--base-port", str(port), "--seed", str(seed),
        "--target-decisions", str(target), "--time-scale", "15",
        "--device", "cuda:0", "--challenger-timeout-seconds", "20",
    ]
    if neural:
        command.append("--neural-opponent")
    return command


def train_command(
    python: str, branch_id: str, resume: Path, rollouts: Iterable[Path],
    output: Path, *, stage: str, steps: int,
) -> list[str]:
    command = [
        python, "scripts/train_navigation_ppo.py", "--goal7-navigator", str(PARENT),
        "--branch-id", branch_id, "--resume", str(resume), "--stage", stage,
        "--environment-steps", str(steps), "--total-curriculum-steps", "4096",
        "--updates", "8", "--batch-size", "256", "--output-dir", str(output),
        "--device", "cuda:0",
    ]
    for rollout in rollouts:
        command.extend(("--rollout", str(rollout)))
    return command


def eval_command(
    python: str, navigator: Path, output: Path, *, preset: str,
    seed_value: int, port: int, run_seed: int, sessions: int = 2,
) -> list[str]:
    return [
        python, "scripts/evaluate_navigation.py", "--env-path", str(ENV),
        "--navigator", str(navigator), "--checkpoint", str(COMBAT),
        "--normalizer", str(NORMALIZER), "--prerequisite", str(CERT),
        "--manifest", str(VALIDATION), "--split", "validation",
        "--seeds", str(seed_value), "--preset", preset, "--mode", "deterministic",
        "--output-dir", str(output), "--run-id", output.name,
        "--base-port", str(port), "--seed", str(run_seed), "--time-scale", "15",
        "--device", "cuda:0", "--target-sessions-per-area", str(sessions),
        "--challenger-timeout-seconds", "20",
    ]


def read_eval(path: Path) -> dict[str, Any]:
    result = json.loads((path / "evaluation_result.json").read_text())
    metrics = result.get("peek_metrics") or {}
    return {
        "status": result["status"],
        "cover_use_rate": float(metrics.get("cover_use_rate", 0)),
        "completed_peek_rate": float(metrics.get("completed_peek_rate", 0)),
        "damage_positive_peek_rate": float(metrics.get("damage_positive_peek_rate", 0)),
        "rapid_same_side_repeek_rate": float(metrics.get("rapid_same_side_repeek_rate", 1)),
        "aim_shoot_action_mismatches": int(result.get("aim_shoot_action_mismatches", 1)),
        "contact_rate": float(result.get("contact_rate", 0)),
        "damage_inflicted": float(result.get("damage_inflicted", 0)),
        "source": str((path / "evaluation_result.json").relative_to(PROJECT)),
    }


def aggregate(branch_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "cover_use_rate", "completed_peek_rate", "damage_positive_peek_rate",
        "rapid_same_side_repeek_rate", "contact_rate", "damage_inflicted",
    )
    record: dict[str, Any] = {"branch_id": branch_id, "evaluations": rows}
    for key in keys:
        record[key] = sum(float(row[key]) for row in rows) / max(1, len(rows))
    record["status"] = "PASS" if all(
        row["status"] == "PASS" and row["aim_shoot_action_mismatches"] == 0
        for row in rows
    ) else "FAIL"
    return record


def ranking(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda row: (
        row["status"] != "PASS", -row["cover_use_rate"],
        -row["completed_peek_rate"], -row["damage_positive_peek_rate"],
        row["rapid_same_side_repeek_rate"], -row["contact_rate"],
        -row["damage_inflicted"], row["branch_id"],
    ))


def final_ranking(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """At promotion, explicit pass gates outrank aggregate shaping metrics."""
    def gate_failures(row: dict[str, Any]) -> int:
        return sum((
            row["cover_use_rate"] < 0.45,
            row["completed_peek_rate"] < 0.30,
            row["damage_positive_peek_rate"] < 0.25,
            row["rapid_same_side_repeek_rate"] >= 0.20,
        ))

    return sorted(records, key=lambda row: (
        row["status"] != "PASS", gate_failures(row),
        -row["cover_use_rate"], -row["completed_peek_rate"],
        -row["damage_positive_peek_rate"], row["rapid_same_side_repeek_rate"],
        -row["contact_rate"], -row["damage_inflicted"], row["branch_id"],
    ))


def reselect_existing() -> int:
    path = ROOT / "SWEEP_RESULTS.json"
    result = json.loads(path.read_text())
    result["round2_ranking"] = final_ranking(result["round2_ranking"])
    winner = result["round2_ranking"][0]
    selected_source = ROOT / "sweep/r2_train" / winner["branch_id"] / "navigator_s004096.pt"
    selected = ROOT / "selected_peeker.pt"
    shutil.copy2(selected_source, selected)
    result["selected_branch"] = winner["branch_id"]
    result["selected_sha256"] = sha256_file(selected)
    result["final_selection_enforced_explicit_tactical_gates"] = True
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


def main() -> int:
    prerequisite = json.loads(CERT.read_text())
    if prerequisite.get("status") != "PHASE5_REACQUISITION_READY":
        raise RuntimeError("Goal 10 prerequisite is not ready")
    ROOT.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    train_seeds = hidden_seeds(TRAIN, 5)
    validation_seeds = hidden_seeds(VALIDATION, 5)

    init_jobs = []
    for gpu, variant in enumerate(PEEKER_VARIANTS):
        output = ROOT / "sweep/r0_init" / variant.branch.branch_id
        command = [
            python, "scripts/train_navigation_ppo.py", "--goal7-navigator", str(PARENT),
            "--branch-id", variant.branch.branch_id, "--output-dir", str(output),
            "--environment-steps", "0", "--initialize-only", "--device", "cuda:0",
        ]
        init_jobs.append((gpu, variant.branch.branch_id, command, output / "launch.log"))
    run_parallel(init_jobs)

    r0_collect = []
    for gpu, variant in enumerate(PEEKER_VARIANTS):
        branch = variant.branch.branch_id
        hunter = ROOT / "sweep/r0_init" / branch / "hunter_s000000.pt"
        output = ROOT / "sweep/r0_p0" / branch
        r0_collect.append((gpu, branch, live_command(
            python, hunter, output, stage="P0", preset="peek_p0",
            seeds=train_seeds, split="train", manifest=TRAIN,
            port=12000 + gpu, seed=62100 + gpu, target=1024,
        ), output / "launch.log"))
    run_parallel(r0_collect)

    r0_train = []
    for gpu, variant in enumerate(PEEKER_VARIANTS):
        branch = variant.branch.branch_id
        output = ROOT / "sweep/r0_train" / branch
        r0_train.append((gpu, branch, train_command(
            python, branch, ROOT / "sweep/r0_init" / branch / "hunter_s000000.pt",
            [ROOT / "sweep/r0_p0" / branch / "rollout.npz"], output,
            stage="P0", steps=1024,
        ), output / "launch.log"))
    run_parallel(r0_train)

    r0_eval = []
    for gpu, variant in enumerate(PEEKER_VARIANTS):
        branch = variant.branch.branch_id
        navigator = ROOT / "sweep/r0_train" / branch / "navigator_s001024.pt"
        output = ROOT / "sweep/r0_validation" / branch
        r0_eval.append((gpu, branch, eval_command(
            python, navigator, output, preset="peek_p0", seed_value=validation_seeds[2],
            port=12100 + gpu, run_seed=62200 + gpu,
        ), output / "launch.log"))
    run_parallel(r0_eval)
    round0 = ranking([
        aggregate(variant.branch.branch_id, [read_eval(
            ROOT / "sweep/r0_validation" / variant.branch.branch_id
        )]) for variant in PEEKER_VARIANTS
    ])
    top4 = [row["branch_id"] for row in round0[:4]]

    r1_collect = []
    for gpu, (branch, stage, preset) in enumerate(
        (branch, stage, f"peek_{stage.lower()}")
        for branch in top4 for stage in ("P1", "P2")
    ):
        hunter = ROOT / "sweep/r0_train" / branch / "hunter_s001024.pt"
        output = ROOT / "sweep/r1_collect" / branch / stage.lower()
        r1_collect.append((gpu, f"{branch}:{stage}", live_command(
            python, hunter, output, stage=stage, preset=preset,
            seeds=train_seeds, split="train", manifest=TRAIN,
            port=12200 + gpu, seed=62300 + gpu, target=1024,
        ), output / "launch.log"))
    run_parallel(r1_collect)

    r1_train = []
    for gpu, branch in enumerate(top4):
        output = ROOT / "sweep/r1_train" / branch
        rollouts = [
            ROOT / "sweep/r1_collect" / branch / stage / "rollout.npz"
            for stage in ("p1", "p2")
        ]
        r1_train.append((gpu, branch, train_command(
            python, branch, ROOT / "sweep/r0_train" / branch / "hunter_s001024.pt",
            rollouts, output, stage="P1_P2", steps=2048,
        ), output / "launch.log"))
    run_parallel(r1_train)

    r1_eval = []
    for gpu, (branch, stage, seed_index) in enumerate(
        (branch, stage, index) for branch in top4
        for index, stage in enumerate(("P1", "P2"))
    ):
        navigator = ROOT / "sweep/r1_train" / branch / "navigator_s002048.pt"
        output = ROOT / "sweep/r1_validation" / branch / stage.lower()
        r1_eval.append((gpu, f"{branch}:{stage}", eval_command(
            python, navigator, output, preset=f"peek_{stage.lower()}",
            seed_value=validation_seeds[seed_index + 1], port=12300 + gpu,
            run_seed=62400 + gpu,
        ), output / "launch.log"))
    run_parallel(r1_eval)
    round1 = ranking([
        aggregate(branch, [read_eval(ROOT / "sweep/r1_validation" / branch / stage)
                           for stage in ("p1", "p2")])
        for branch in top4
    ])
    top2 = [row["branch_id"] for row in round1[:2]]

    final_stages = (
        ("P3", "peek_p3", False), ("P4", "peek_p4", False),
        ("P5", "peek_p4", False), ("P5", "peek_p4", True),
    )
    r2_collect = []
    for gpu, (branch, stage_index, stage_spec) in enumerate(
        (branch, index, spec) for branch in top2 for index, spec in enumerate(final_stages)
    ):
        stage, preset, neural = stage_spec
        hunter = ROOT / "sweep/r1_train" / branch / "hunter_s002048.pt"
        label = f"{stage.lower()}_{'neural' if neural else 'scripted'}_{stage_index}"
        output = ROOT / "sweep/r2_collect" / branch / label
        r2_collect.append((gpu, f"{branch}:{label}", live_command(
            python, hunter, output, stage=stage, preset=preset,
            seeds=train_seeds, split="train", manifest=TRAIN,
            port=12400 + gpu, seed=62500 + gpu, target=1024, neural=neural,
        ), output / "launch.log"))
    run_parallel(r2_collect)

    r2_train = []
    for gpu, branch in enumerate(top2):
        rollouts = sorted((ROOT / "sweep/r2_collect" / branch).glob("*/rollout.npz"))
        output = ROOT / "sweep/r2_train" / branch
        r2_train.append((gpu, branch, train_command(
            python, branch, ROOT / "sweep/r1_train" / branch / "hunter_s002048.pt",
            rollouts, output, stage="P3_P4_P5", steps=4096,
        ), output / "launch.log"))
    run_parallel(r2_train)

    validation_presets = ("peek_p0", "peek_p2", "peek_p3", "peek_p4")
    r2_eval = []
    for gpu, (branch, preset_index, preset) in enumerate(
        (branch, index, preset) for branch in top2
        for index, preset in enumerate(validation_presets)
    ):
        navigator = ROOT / "sweep/r2_train" / branch / "navigator_s004096.pt"
        output = ROOT / "sweep/r2_validation" / branch / preset
        r2_eval.append((gpu, f"{branch}:{preset}", eval_command(
            python, navigator, output, preset=preset,
            seed_value=validation_seeds[preset_index], port=12500 + gpu,
            run_seed=62600 + gpu,
        ), output / "launch.log"))
    run_parallel(r2_eval)
    round2 = final_ranking([
        aggregate(branch, [read_eval(ROOT / "sweep/r2_validation" / branch / preset)
                           for preset in validation_presets])
        for branch in top2
    ])
    winner = round2[0]
    selected_source = ROOT / "sweep/r2_train" / winner["branch_id"] / "navigator_s004096.pt"
    selected = ROOT / "selected_peeker.pt"
    shutil.copy2(selected_source, selected)
    result = {
        "schema_version": "phase5_local_geometry_peeker_sweep_v001",
        "status": "PASS", "gpu_count": 8,
        "successive_halving": {"round0": 8, "round1": 4, "round2": 2, "selected": 1},
        "fixed_environment_step_milestones": [1024, 2048, 4096],
        "curriculum_stages": ["P0", "P1", "P2", "P3", "P4", "P5_scripted", "P5_neural"],
        "selection_used_heldout": False, "freed_gpus_reassigned_to_top_branches": True,
        "parent": str(PARENT.relative_to(PROJECT)), "parent_sha256": sha256_file(PARENT),
        "combat_expert_frozen": True, "per_map_cover_anchors": False,
        "peek_point_files": False, "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "round0_ranking": round0, "round0_retained": top4,
        "round1_ranking": round1, "round1_retained": top2,
        "round2_ranking": round2, "selected_branch": winner["branch_id"],
        "selected_peeker": str(selected.relative_to(PROJECT)),
        "selected_sha256": sha256_file(selected),
    }
    (ROOT / "SWEEP_RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(reselect_existing() if "--reselect-only" in sys.argv else main())
