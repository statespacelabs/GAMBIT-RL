#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.combat_specialist_families import (
    ACTION_DIM,
    ENV_PATH,
    FAMILY_BY_ID,
    OBS_DIM,
    OBS_SCHEMA_VERSION,
    OUT,
    PHASE3_SCRIPTED_BOT_MODES,
    PRODUCTION_SELECTOR_MANIFEST,
    SCRIPTED_MODES,
    TRAINING_CATEGORY_WEIGHTS,
    append_jsonl,
    choose_spawn_bucket,
    choose_weighted,
    ensure_out,
    expert_checkpoint_map,
    family_row,
    family_snapshot_dir,
    family_training_dir,
    load_selector_manifest,
    normalizer_path,
    read_json_safe,
    read_jsonl,
    resolve_expert_path,
    safe_float,
    selector_checkpoint_path,
    sha256_file,
    summarize_train_jsonl,
    validate_selector_manifest,
    write_csv,
    write_json,
    write_md,
)
from scripts.unity_port_allocator import acquire_port, release_port


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4.4g anchored GvG candidate-family training wrapper.")
    parser.add_argument("--family-id", required=True, choices=sorted(FAMILY_BY_ID))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--updates", type=int, default=int(os.environ.get("PHASE4_4G_UPDATES", "120")))
    parser.add_argument("--segment-updates", type=int, default=int(os.environ.get("PHASE4_4G_SEGMENT_UPDATES", "10")))
    parser.add_argument("--steps-per-iter", type=int, default=int(os.environ.get("PHASE4_4G_STEPS_PER_ITER", "256")))
    parser.add_argument("--num-areas", type=int, default=int(os.environ.get("PHASE4_4G_NUM_AREAS", "2")))
    parser.add_argument("--base-port", type=int, default=65100)
    parser.add_argument("--seed", type=int, default=44800)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--phase-name", default="phase4_4g_large_gvg_stimulus_library")
    parser.add_argument("--lr-initial", type=float, default=None)
    parser.add_argument("--target-kl-override", type=float, default=None)
    parser.add_argument("--kl-anchor-coef-override", type=float, default=None)
    parser.add_argument("--entropy-coef-override", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--min-log-std", type=float, default=None)
    parser.add_argument("--max-log-std", type=float, default=None)
    parser.add_argument("--mean-action-saturation-threshold", type=float, default=None)
    parser.add_argument("--spawn-mix-json", default="")
    parser.add_argument("--category-weights-json", default="")
    parser.add_argument("--time-scale", type=float, default=float(os.environ.get("PHASE4_4G_TIME_SCALE", "5.0")))
    parser.add_argument("--env-path", default=str(ENV_PATH))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=os.environ.get("PHASE4_4_PYTHON", sys.executable))
    parser.add_argument("--snapshot-stride", type=int, default=int(os.environ.get("PHASE4_4G_SNAPSHOT_STRIDE", "10")))
    parser.add_argument("--disable-safe-port-allocator", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _write_yaml_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _focus_checkpoint_pool(spec, manifest: dict[str, Any]) -> list[tuple[str, Path]]:
    experts = expert_checkpoint_map(manifest)
    out: list[tuple[str, Path]] = []
    for focus in spec.focus_roster:
        if focus in SCRIPTED_MODES:
            continue
        if focus == "temporal_selector_v001":
            for expert in manifest.get("expert_order") or experts:
                if expert in experts:
                    out.append((f"temporal_selector_pool:{expert}", Path(str(experts[expert]["path"]))))
            continue
        if focus in experts:
            out.append((focus, Path(str(experts[focus]["path"]))))
    if not out:
        out = [(name, Path(str(row["path"]))) for name, row in sorted(experts.items())]
    return out


def _latest_snapshot_pool(current_family: str) -> list[tuple[str, Path]]:
    pool_csv = os.environ.get("PHASE4_4G_SNAPSHOT_POOL_CSV", "")
    if pool_csv and Path(pool_csv).exists():
        rows: list[tuple[str, Path]] = []
        with Path(pool_csv).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ckpt = Path(str(row.get("snapshot_path") or row.get("candidate_parent_path") or ""))
                name = str(row.get("candidate_id") or row.get("snapshot_id") or ckpt.stem)
                if ckpt.exists():
                    rows.append((f"smoke_keep:{name}", ckpt))
        if rows:
            return rows
    rows: list[tuple[str, Path]] = []
    for ckpt in sorted((OUT / "training").glob("*/checkpoints/*.pt")):
        if ckpt.is_file():
            rows.append((f"snapshot:{ckpt.parent.parent.name}:{ckpt.stem}", ckpt))
    return rows


def _copy_selected_snapshots(
    segment_dir: Path,
    family_dir: Path,
    segment_index: int,
    global_start: int,
    segment_updates: int,
    stride: int,
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    target_dir = family_dir / "checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted((segment_dir / "checkpoints").glob("*.pt"))
    latest = segment_dir / "latest.pt"
    if latest.exists() and latest not in checkpoints:
        checkpoints.append(latest)
    by_dest: dict[Path, Path] = {}
    by_update: dict[Path, int] = {}
    for ckpt in checkpoints:
        if ckpt.name == "latest.pt":
            global_update = global_start + segment_updates
        else:
            local_update = 0
            digits = "".join(ch for ch in ckpt.stem if ch.isdigit())
            if digits:
                local_update = int(digits[-3:])
            global_update = global_start + max(local_update, 1)
        if stride > 1 and global_update % stride != 0 and ckpt.name != "latest.pt":
            continue
        dest = target_dir / f"snapshot_s{segment_index:03d}_u{global_update:04d}.pt"
        by_dest[dest] = ckpt
        by_update[dest] = global_update
    for dest, ckpt in by_dest.items():
        global_update = by_update[dest]
        shutil.copy2(ckpt, dest)
        copied.append(
            {
                "snapshot_id": f"{family_dir.name}_s{segment_index:03d}_u{global_update:04d}",
                "family_id": family_dir.name,
                "segment_index": segment_index,
                "global_update": global_update,
                "snapshot_path": str(dest),
                "sha256": sha256_file(dest),
            }
        )
    return copied


def _run_command(cmd: list[str], cwd: Path, env: dict[str, str], stdout: Path, stderr: Path, dry_run: bool) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        write_json(stdout.with_suffix(".dry_run.json"), {"cmd": cmd, "env": {k: env[k] for k in sorted(env) if k.startswith("PHASE4_4") or k.startswith("CUDA")}})
        return 0
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        return subprocess.run(cmd, cwd=cwd, env=env, stdout=out, stderr=err).returncode


def _base_env(args: argparse.Namespace, bucket: str, segment_dir: Path, seed: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PHASE4_4_ENABLE_SPAWN_BUCKETS"] = "1"
    env["PHASE4_4_SPAWN_BUCKET"] = bucket
    env["PHASE4_4_SPAWN_SEED"] = str(seed)
    env["PHASE4_4_TRIAL_MAX_SECONDS"] = "60"
    env["PHASE4_4_WRITE_SPAWN_TELEMETRY"] = "1"
    env["PHASE4_4_SPAWN_TELEMETRY_PATH"] = str(segment_dir / "phase44_spawn_telemetry.jsonl")
    if bucket == "close":
        env["PHASE4_4_REQUIRE_INITIAL_LOS"] = "1"
    if bucket in {"obstacle", "search_destroy"}:
        env["PHASE4_4_REQUIRE_OBSTACLE_BETWEEN"] = "1"
    return env


def _override(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def _json_weights(raw: str, fallback: dict[str, float]) -> dict[str, float]:
    if not raw:
        return dict(fallback)
    data = json.loads(raw)
    return {str(k): float(v) for k, v in data.items()}


def _segment_config(spec, args: argparse.Namespace, segment_dir: Path, current_ckpt: Path, opponent_name: str, opponent_path: Path, norm: Path, segment_updates: int, seed: int) -> dict[str, Any]:
    return {
        "phase": args.phase_name,
        "learner_agent": "A",
        "resume_agent_a_from": str(current_ckpt),
        "resume_agent_b_from": str(opponent_path),
        "normalizer_path": str(norm),
        "num_unity_areas": args.num_areas,
        "time_scale": args.time_scale,
        "action_mode": "stochastic",
        "num_iterations": segment_updates,
        "seed": seed,
        "lr": _override(args.lr_initial, spec.lr),
        "lr_initial": _override(args.lr_initial, spec.lr),
        "target_kl": _override(args.target_kl_override, spec.target_kl),
        "max_grad_norm": _override(args.max_grad_norm, 0.5),
        "kl_anchor_coef": _override(args.kl_anchor_coef_override, spec.kl_anchor_coef),
        "kl_anchor_checkpoint": str(current_ckpt),
        "entropy_coef": _override(args.entropy_coef_override, spec.entropy_coef),
        "min_log_std": _override(args.min_log_std, -2.5),
        "max_log_std": _override(args.max_log_std, -1.2),
        "mean_action_saturation_threshold": _override(args.mean_action_saturation_threshold, 0.65),
        "aim_coef": spec.aim_coef,
        "shoot_bootstrap_coef": spec.shoot_bootstrap_coef,
        "jerk_coef": spec.jerk_coef,
        "spam_coef": spec.spam_coef,
        "freeze_movement_action_head": spec.freeze_movement_action_head,
        "freeze_look_action_head": spec.freeze_look_action_head,
        "train_shoot_head": spec.train_shoot_head,
        "actor_update_enabled": True,
        "value_update_enabled": True,
        "normalizer_update": False,
        "opponent_name": opponent_name,
        "output_dir": str(segment_dir),
    }


def _run_gvg_segment(spec, args, segment_dir: Path, current_ckpt: Path, opponent_name: str, opponent_path: Path, norm: Path, segment_updates: int, port: int, seed: int, env: dict[str, str]) -> int:
    cfg = _segment_config(spec, args, segment_dir, current_ckpt, opponent_name, opponent_path, norm, segment_updates, seed)
    cfg_path = segment_dir / "segment_config.yaml"
    _write_yaml_json(cfg_path, cfg)
    cmd = [
        args.python,
        "scripts/train_combat_ppo.py",
        "--config",
        str(cfg_path),
        "--output-dir",
        str(segment_dir),
        "--env-path",
        args.env_path,
        "--base-port",
        str(port),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--iterations",
        str(segment_updates),
        "--steps-per-iter",
        str(args.steps_per_iter),
        "--action-mode",
        "stochastic",
    ]
    return _run_command(cmd, PROJECT_ROOT, env, segment_dir / "stdout.log", segment_dir / "stderr.log", args.dry_run)


def _run_scripted_segment(spec, args, segment_dir: Path, current_ckpt: Path, norm: Path, segment_updates: int, port: int, seed: int, env: dict[str, str], scripted_id: str) -> int:
    cmd = [
        args.python,
        "scripts/combat_ppo_canary.py",
        "--output-dir",
        str(segment_dir),
        "--env-path",
        args.env_path,
        "--normalizer-path",
        str(norm),
        "--resume-from",
        str(current_ckpt),
        "--kl-anchor-checkpoint",
        str(current_ckpt),
        "--updates",
        str(segment_updates),
        "--rollout-steps",
        str(args.steps_per_iter),
        "--seq-len",
        "16",
        "--batch-size",
        "4",
        "--eval-decisions",
        str(args.steps_per_iter),
        "--base-port",
        str(port),
        "--seed",
        str(seed),
        "--time-scale",
        str(args.time_scale),
        "--device",
        args.device,
        "--lr",
        str(_override(args.lr_initial, spec.lr)),
        "--target-kl",
        str(_override(args.target_kl_override, spec.target_kl)),
        "--kl-anchor-coef",
        str(_override(args.kl_anchor_coef_override, spec.kl_anchor_coef)),
        "--entropy-coef",
        str(_override(args.entropy_coef_override, spec.entropy_coef)),
        "--aim-coef",
        str(spec.aim_coef),
        "--shoot-bootstrap-coef",
        str(spec.shoot_bootstrap_coef),
        "--jerk-coef",
        str(spec.jerk_coef),
        "--spam-coef",
        str(spec.spam_coef),
        "--game-mode",
        "GambitVsScripted",
        "--player-b-bot-mode",
        PHASE3_SCRIPTED_BOT_MODES[scripted_id],
        "--num-areas",
        str(args.num_areas),
        "--mean-action-saturation-penalty",
        "--saturation-start-iter",
        "10",
        "--saturation-abs095-stop-rate",
        "0.50",
    ]
    if spec.freeze_movement_action_head:
        cmd.append("--freeze-movement-action-head")
    if spec.freeze_look_action_head:
        cmd.append("--freeze-look-action-head")
    if spec.train_shoot_head:
        cmd.append("--train-shoot-head")
    return _run_command(cmd, PROJECT_ROOT, env, segment_dir / "stdout.log", segment_dir / "stderr.log", args.dry_run)


def main() -> int:
    args = parse_args()
    ensure_out()
    spec = FAMILY_BY_ID[args.family_id]
    family_dir = Path(args.output_dir) if args.output_dir else family_training_dir(spec.family_id)
    family_dir.mkdir(parents=True, exist_ok=True)
    family_snapshot_dir(spec.family_id).mkdir(parents=True, exist_ok=True)

    manifest = load_selector_manifest()
    failures = validate_selector_manifest(manifest)
    norm = normalizer_path(manifest)
    parent = Path(args.resume_from) if args.resume_from else resolve_expert_path(spec.parent_expert, manifest)
    if not parent.exists():
        failures.append(f"missing parent checkpoint: {parent}")
    if not norm.exists():
        failures.append(f"missing normalizer: {norm}")
    if not selector_checkpoint_path(manifest).exists():
        failures.append(f"missing selector checkpoint: {selector_checkpoint_path(manifest)}")
    if failures:
        write_json(family_dir / "candidate_summary.json", {"status": "FAIL", "failures": failures, "family_id": spec.family_id})
        return 1

    rng = random.Random(args.seed)
    current_ckpt = parent
    spawn_mix = _json_weights(args.spawn_mix_json, spec.spawn_bucket_mix)
    category_weights = _json_weights(args.category_weights_json, TRAINING_CATEGORY_WEIGHTS)
    schedule_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    completed_updates = 0
    status = "PASS"
    hard_failures: list[str] = []
    frozen_pool = _focus_checkpoint_pool(spec, manifest)

    manifest_payload = {
        "phase": args.phase_name,
        "family": family_row(spec),
        "production_selector_manifest": str(PRODUCTION_SELECTOR_MANIFEST),
        "production_selector_manifest_sha256": sha256_file(PRODUCTION_SELECTOR_MANIFEST),
        "selector_checkpoint": str(selector_checkpoint_path(manifest)),
        "selector_checkpoint_sha256": sha256_file(selector_checkpoint_path(manifest)),
        "normalizer_path": str(norm),
        "normalizer_sha256": sha256_file(norm),
        "obs_schema_version": OBS_SCHEMA_VERSION,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "normalizer_update": False,
        "schema_update": False,
        "phase4_5_live_roster_update": False,
        "training_category_weights": dict(TRAINING_CATEGORY_WEIGHTS),
        "effective_training_category_weights": dict(category_weights),
        "effective_spawn_bucket_mix": dict(spawn_mix),
        "updates_requested": args.updates,
        "segment_updates": args.segment_updates,
        "snapshot_stride": args.snapshot_stride,
        "resume_from": str(parent),
        "use_safe_port_allocator": not args.disable_safe_port_allocator,
        "safe_ppo_overrides": {
            "lr_initial": args.lr_initial,
            "target_kl": args.target_kl_override,
            "kl_anchor_coef": args.kl_anchor_coef_override,
            "entropy_coef": args.entropy_coef_override,
            "max_grad_norm": args.max_grad_norm,
            "min_log_std": args.min_log_std,
            "max_log_std": args.max_log_std,
            "mean_action_saturation_threshold": args.mean_action_saturation_threshold,
        },
    }
    write_json(family_dir / "run_manifest.json", manifest_payload)
    _write_yaml_json(family_dir / "resolved_config.yaml", manifest_payload)

    total_segments = max(1, (args.updates + args.segment_updates - 1) // args.segment_updates)
    for segment_index in range(1, total_segments + 1):
        remaining = args.updates - completed_updates
        if remaining <= 0:
            break
        segment_updates = min(args.segment_updates, remaining)
        category = choose_weighted(category_weights, rng)
        bucket = choose_weighted(spawn_mix, rng)
        seed = args.seed + segment_index
        gpu_id = int(os.environ.get("PHASE4_4G_PHYSICAL_GPU") or spec.assigned_gpu)
        run_id = os.environ.get("PHASE4_4G_RUN_ID", "phase4_4g_training")
        safe_port_lock = False
        if args.disable_safe_port_allocator:
            port = args.base_port + segment_index * 10
        else:
            port = acquire_port(gpu_id, f"{spec.family_id}_segment_{segment_index:03d}", run_id)
            safe_port_lock = True
        segment_dir = family_dir / "segments" / f"segment_{segment_index:03d}_{category}_{bucket}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        env = _base_env(args, bucket, segment_dir, seed)

        fallback_reason = ""
        opponent_name = ""
        opponent_path: Path | None = None
        code = 0
        if category == "snapshot_pool":
            pool = _latest_snapshot_pool(spec.family_id)
            if pool:
                opponent_name, opponent_path = rng.choice(pool)
            else:
                fallback_reason = "snapshot_pool_empty_fallback_to_frozen_roster"
                opponent_name, opponent_path = rng.choice(frozen_pool)
        elif category == "self_mirror":
            opponent_name, opponent_path = "self_mirror_current_policy", current_ckpt
        elif category == "scripted_retention":
            scripted_choices = [x for x in spec.focus_roster if x in SCRIPTED_MODES] or list(SCRIPTED_MODES)
            scripted_id = str(rng.choice(scripted_choices))
            opponent_name = scripted_id
            code = _run_scripted_segment(spec, args, segment_dir, current_ckpt, norm, segment_updates, port, seed, env, scripted_id)
        else:
            opponent_name, opponent_path = rng.choice(frozen_pool)

        if category != "scripted_retention":
            assert opponent_path is not None
            code = _run_gvg_segment(spec, args, segment_dir, current_ckpt, opponent_name, opponent_path, norm, segment_updates, port, seed, env)

        summary_path = segment_dir / ("tiny_ppo_canary_summary.json" if category == "scripted_retention" else "gvg_one_sided_summary.json")
        summary = read_json_safe(summary_path, {})
        segment_status = "DRY_RUN" if args.dry_run else ("PASS" if code == 0 and summary.get("status") == "PASS" else "FAIL")
        if category == "scripted_retention":
            rows = read_jsonl(segment_dir / "ppo_train.jsonl")
            train_summary = summarize_train_jsonl(rows)
            latest = segment_dir / "latest.pt"
        else:
            rows = read_jsonl(segment_dir / "gvg_train.jsonl")
            train_summary = summarize_train_jsonl(rows, "A")
            latest = segment_dir / "latest.pt"

        segment_row = {
            "family_id": spec.family_id,
            "segment_index": segment_index,
            "category": category,
            "spawn_bucket": bucket,
            "opponent_name": opponent_name,
            "opponent_path": str(opponent_path or ""),
            "fallback_reason": fallback_reason,
            "segment_updates": segment_updates,
            "global_update_start": completed_updates,
            "global_update_end": completed_updates + segment_updates,
            "seed": seed,
            "base_port": port,
            "safe_port_allocator": not args.disable_safe_port_allocator,
            "status": segment_status,
            "summary_path": str(summary_path),
            "latest_checkpoint": str(latest) if latest.exists() else "",
            **{f"train_{k}": v for k, v in train_summary.items()},
        }
        schedule_rows.append(segment_row)
        append_jsonl(family_dir / "train.jsonl", segment_row)

        copied = _copy_selected_snapshots(segment_dir, family_dir, segment_index, completed_updates, segment_updates, args.snapshot_stride)
        for row in copied:
            row.update(
                {
                    "category": category,
                    "spawn_bucket": bucket,
                    "opponent_name": opponent_name,
                    "segment_status": segment_status,
                    "train_late_hits_mean": train_summary["late_hits_mean"],
                    "train_late_fired_steps_mean": train_summary["late_fired_steps_mean"],
                    "train_late_aim_mean": train_summary["late_aim_mean"],
                }
            )
        snapshot_rows.extend(copied)
        completed_updates += segment_updates

        if latest.exists():
            current_ckpt = latest
            shutil.copy2(latest, family_dir / "latest.pt")
        if safe_port_lock:
            release_port(port)
        if segment_status != "PASS" and not args.dry_run:
            status = "FAIL"
            hard_failures.append(f"segment {segment_index} failed: {summary.get('failures') or code}")
            break

    write_csv(family_dir / "segment_schedule.csv", schedule_rows)
    write_csv(family_dir / "snapshot_index.csv", snapshot_rows)
    best_snapshot = ""
    if snapshot_rows:
        best_row = sorted(
            snapshot_rows,
            key=lambda r: (
                safe_float(r.get("train_late_hits_mean"), 0.0),
                safe_float(r.get("train_late_fired_steps_mean"), 0.0),
                -safe_float(r.get("train_late_aim_mean"), 999.0),
            ),
            reverse=True,
        )[0]
        best_snapshot = str(best_row["snapshot_path"])
        if Path(best_snapshot).exists():
            shutil.copy2(best_snapshot, family_dir / "best_behavioral.pt")
            shutil.copy2(best_snapshot, family_dir / "best_retention.pt")
    if not snapshot_rows and not args.dry_run:
        status = "FAIL"
        hard_failures.append("no snapshots harvested")

    summary = {
        "phase": args.phase_name,
        "family_id": spec.family_id,
        "status": status if not args.dry_run else "DRY_RUN",
        "failures": hard_failures,
        "updates_completed": completed_updates,
        "segments_completed": len(schedule_rows),
        "snapshots_harvested": len(snapshot_rows),
        "best_behavioral": best_snapshot,
        "best_retention": best_snapshot,
        "latest": str(family_dir / "latest.pt") if (family_dir / "latest.pt").exists() else "",
        "normalizer_path": str(norm),
        "normalizer_sha256": sha256_file(norm),
        "schema_update": False,
        "normalizer_update": False,
        "phase4_5_live_roster_update": False,
    }
    write_json(family_dir / "candidate_summary.json", summary)
    write_md(
        family_dir / "candidate_report.md",
        [
            f"# Phase 4.4g `{spec.family_id}`",
            "",
            f"Status: `{summary['status']}`",
            f"Goal: {spec.goal}",
            f"GPU assignment: `{spec.assigned_gpu}`",
            f"Updates completed: `{completed_updates}`",
            f"Snapshots harvested: `{len(snapshot_rows)}`",
            f"Best behavioral: `{best_snapshot}`",
            "",
            "No normalizer, schema, temporal selector, or Phase 4.5 live roster mutation is performed by this wrapper.",
            "",
            "Failures:",
            *[f"- {failure}" for failure in (hard_failures or ["none"])],
        ],
    )
    print(summary["status"])
    return 0 if summary["status"] in {"PASS", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
