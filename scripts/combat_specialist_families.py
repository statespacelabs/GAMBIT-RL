#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(os.environ.get("RL_GRADER_ROOT", str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "experiments/phase4_active_diagnostic/phase4_4_gvg_stimulus_library"
P43 = ROOT / "experiments/phase4_active_diagnostic/phase4_3_offline_puppeteer_sim"
PRODUCTION_SELECTOR_MANIFEST = ROOT / "experiments/phase3v2/phase3v2_c_selected_artifacts/gvg_temporal_selector_local45_v001.json"
ENV_PATH = Path(os.environ.get("PHASE4_4_ENV_PATH", "artifacts/unity/combat/Builds/BotArena.x86_64"))

OBS_SCHEMA_VERSION = "phase3v2_c_local45"
OBS_DIM = 45
ACTION_DIM = 8
ALLOWED_GPUS = [1, 2, 3, 4, 5, 6, 7]

SPAWN_BUCKETS = ["close", "mid", "far", "obstacle", "search_destroy"]
DEFAULT_BUCKET_MIX = {
    "close": 0.20,
    "mid": 0.30,
    "far": 0.20,
    "obstacle": 0.20,
    "search_destroy": 0.10,
}
SEARCH_OBSTACLE_BUCKET_MIX = {
    "close": 0.10,
    "mid": 0.15,
    "far": 0.20,
    "obstacle": 0.35,
    "search_destroy": 0.20,
}

CURRENT_ROSTER = [
    "scripted_idle",
    "scripted_face_opponent",
    "scripted_strafe_and_face",
    "v000",
    "visible_residual",
    "underfire_residual",
    "dag21_default",
    "temporal_selector_v001",
]

PRODUCTION_POLICY_ALIASES = {
    "v000": "gvg_seed_local45_v000",
    "dag21_default": "dag21_default",
    "dag21_retention_heavy": "dag21_retention_heavy",
    "visible_residual": "br_cap010_visible_e003",
    "underfire_residual": "br_cap010_underfire_e003",
}

SCRIPTED_MODES = {
    "scripted_idle": "scripted_idle",
    "scripted_face_opponent": "scripted_face_opponent",
    "scripted_strafe_and_face": "scripted_strafe_and_face",
}

PHASE3_SCRIPTED_BOT_MODES = {
    "scripted_idle": "Idle",
    "scripted_face_opponent": "FaceOpponent",
    "scripted_strafe_and_face": "StrafeAndFaceShoot",
}

TRAINING_CATEGORY_WEIGHTS = {
    "frozen_roster": 0.60,
    "snapshot_pool": 0.25,
    "self_mirror": 0.10,
    "scripted_retention": 0.05,
}

POOL_SOURCE_DEFAULT = ROOT / "experiments/phase3v2/phase3v2_d_production_sparring_baseline/manifest_resolved_pool_sources.json"
POOL_SOLUTION_DEFAULT = ROOT / "experiments/phase3v2/phase3v2_c_v23_policy_pool/maximin/MAXIMIN_SOLUTION.json"


@dataclass(frozen=True)
class GvgFamilySpec:
    family_id: str
    assigned_gpu: int
    goal: str
    parent_expert: str
    focus_roster: tuple[str, ...]
    bucket_mix_name: str
    lr: float
    target_kl: float
    kl_anchor_coef: float
    entropy_coef: float
    aim_coef: float
    shoot_bootstrap_coef: float
    jerk_coef: float
    spam_coef: float
    freeze_movement_action_head: bool = False
    freeze_look_action_head: bool = False
    train_shoot_head: bool = False

    @property
    def spawn_bucket_mix(self) -> dict[str, float]:
        if self.bucket_mix_name == "search_obstacle":
            return dict(SEARCH_OBSTACLE_BUCKET_MIX)
        return dict(DEFAULT_BUCKET_MIX)


FAMILIES: list[GvgFamilySpec] = [
    GvgFamilySpec(
        "gvg_generalist_roster_mix",
        1,
        "robust mid/high generalist",
        "dag21_default",
        ("v000", "visible_residual", "underfire_residual", "dag21_default"),
        "default",
        6.0e-7,
        0.0012,
        0.035,
        0.004,
        0.9,
        0.035,
        0.0010,
        0.0010,
    ),
    GvgFamilySpec(
        "gvg_anti_temporal_selector",
        2,
        "pressure current production baseline without modifying it",
        "dag21_default",
        ("temporal_selector_v001", "dag21_default", "underfire_residual", "visible_residual"),
        "default",
        5.5e-7,
        0.0010,
        0.045,
        0.004,
        1.0,
        0.040,
        0.0010,
        0.0010,
    ),
    GvgFamilySpec(
        "gvg_anti_v000_visible",
        3,
        "counter older supervised/local45 aiming styles",
        "visible_residual",
        ("v000", "visible_residual"),
        "default",
        6.0e-7,
        0.0012,
        0.040,
        0.004,
        0.9,
        0.040,
        0.0010,
        0.0010,
    ),
    GvgFamilySpec(
        "gvg_anti_strafe_pressure",
        4,
        "punish lateral rhythm and predictable strafing",
        "dag21_default",
        ("scripted_strafe_and_face", "dag21_default", "v000"),
        "default",
        6.5e-7,
        0.0010,
        0.040,
        0.003,
        1.0,
        0.055,
        0.0007,
        0.0010,
        freeze_movement_action_head=True,
    ),
    GvgFamilySpec(
        "gvg_underfire_retaliator",
        5,
        "performs well after taking damage / under pressure",
        "underfire_residual",
        ("underfire_residual", "dag21_default", "temporal_selector_v001"),
        "default",
        6.0e-7,
        0.0010,
        0.045,
        0.004,
        0.9,
        0.045,
        0.0008,
        0.0010,
    ),
    GvgFamilySpec(
        "gvg_evasive_survivor",
        6,
        "hard-to-hit evasive/survival style",
        "visible_residual",
        ("visible_residual", "underfire_residual", "dag21_default"),
        "default",
        5.0e-7,
        0.0010,
        0.055,
        0.006,
        0.7,
        0.025,
        0.0005,
        0.0005,
    ),
    GvgFamilySpec(
        "gvg_search_obstacle_pursuer",
        7,
        "performs under far, obstacle, and search_destroy spawns",
        "dag21_retention_heavy",
        ("dag21_default", "visible_residual", "temporal_selector_v001"),
        "search_obstacle",
        4.5e-7,
        0.0010,
        0.060,
        0.006,
        0.8,
        0.030,
        0.0005,
        0.0007,
    ),
]

FAMILY_BY_ID = {spec.family_id: spec for spec in FAMILIES}


def ensure_out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_safe(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return read_json(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"json_decode_error": True, "raw": line})
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def load_selector_manifest() -> dict[str, Any]:
    return read_json(PRODUCTION_SELECTOR_MANIFEST)


def expert_checkpoint_map(manifest: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    data = manifest or load_selector_manifest()
    return {str(row["expert"]): dict(row) for row in data.get("expert_checkpoints", [])}


def validate_selector_manifest(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if manifest.get("obs_schema_version") != OBS_SCHEMA_VERSION:
        failures.append(f"obs_schema_version mismatch: {manifest.get('obs_schema_version')}")
    if int(manifest.get("obs_dim", -1)) != OBS_DIM:
        failures.append(f"obs_dim mismatch: {manifest.get('obs_dim')}")
    if int(manifest.get("action_dim", -1)) != ACTION_DIM:
        failures.append(f"action_dim mismatch: {manifest.get('action_dim')}")
    if "gvg_seed_local45_v001.pt" in json.dumps(manifest):
        failures.append("unexpected gvg_seed_local45_v001.pt reference")
    return failures


def normalizer_path(manifest: dict[str, Any] | None = None) -> Path:
    data = manifest or load_selector_manifest()
    return Path(str(data["normalizer_path"]))


def selector_checkpoint_path(manifest: dict[str, Any] | None = None) -> Path:
    data = manifest or load_selector_manifest()
    return Path(str(data["selector_checkpoint_path"]))


def resolve_expert_path(expert: str, manifest: dict[str, Any] | None = None) -> Path:
    experts = expert_checkpoint_map(manifest)
    if expert not in experts:
        raise KeyError(f"expert {expert!r} missing from selector manifest")
    return Path(str(experts[expert]["path"]))


def family_training_dir(family_id: str) -> Path:
    return OUT / "training" / family_id


def family_snapshot_dir(family_id: str) -> Path:
    return family_training_dir(family_id) / "checkpoints"


def family_row(spec: GvgFamilySpec) -> dict[str, Any]:
    row = asdict(spec)
    row["spawn_bucket_mix"] = spec.spawn_bucket_mix
    row["training_category_weights"] = dict(TRAINING_CATEGORY_WEIGHTS)
    return row


def all_family_rows() -> list[dict[str, Any]]:
    return [family_row(spec) for spec in FAMILIES]


def choose_weighted(items: dict[str, float], rng: random.Random) -> str:
    total = sum(max(0.0, float(v)) for v in items.values())
    if total <= 0:
        return next(iter(items))
    pick = rng.random() * total
    acc = 0.0
    for key, weight in items.items():
        acc += max(0.0, float(weight))
        if pick <= acc:
            return key
    return next(reversed(items))


def choose_spawn_bucket(spec: GvgFamilySpec, rng: random.Random) -> str:
    return choose_weighted(spec.spawn_bucket_mix, rng)


def candidate_snapshot_rows(path: Path = OUT / "PHASE4_4G_RAW_SNAPSHOT_INDEX.csv") -> list[dict[str, str]]:
    return read_csv_rows(path)


def pool_source_member_name(entity_id: str) -> str:
    return PRODUCTION_POLICY_ALIASES.get(entity_id, entity_id)


def roster_policy_source_members(manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    data = manifest or load_selector_manifest()
    members = []
    seen: set[str] = set()
    for row in data.get("expert_checkpoints", []):
        expert = str(row["expert"])
        member_name = pool_source_member_name(expert)
        if member_name in seen:
            continue
        seen.add(member_name)
        members.append(
            {
                "name": member_name,
                "alias": expert,
                "path": str(row["path"]),
                "sha256": row.get("sha256") or sha256_file(row["path"]),
                "status": "PASS",
                "source": "production_selector_manifest_expert",
            }
        )
    return members


def build_policy_sources(path: Path, candidate_rows: list[dict[str, Any]], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    members = roster_policy_source_members(manifest)
    seen = {m["name"] for m in members}
    for row in candidate_rows:
        name = str(row.get("candidate_id") or row.get("snapshot_id") or "").strip()
        ckpt = str(row.get("checkpoint_path") or row.get("snapshot_path") or "").strip()
        if not name or not ckpt or name in seen:
            continue
        members.append(
            {
                "name": name,
                "path": ckpt,
                "sha256": row.get("sha256") or sha256_file(ckpt),
                "status": "PASS",
                "source": "phase4_4g_quarantined_candidate",
            }
        )
        seen.add(name)
    payload = {
        "source_manifest": str(PRODUCTION_SELECTOR_MANIFEST),
        "source_manifest_sha256": sha256_file(PRODUCTION_SELECTOR_MANIFEST),
        "members": members,
    }
    write_json(path, payload)
    return payload


def extract_side(summary: dict[str, Any], side: str) -> dict[str, Any]:
    return dict(summary.get(side, {}) or {})


def side_score(side: dict[str, Any]) -> float:
    damage = safe_float(side.get("damage_dealt"), 0.0) - safe_float(side.get("damage_taken"), 0.0)
    hits = safe_float(side.get("hits"), 0.0)
    fired = safe_float(side.get("fired_steps"), 0.0)
    aim = safe_float(side.get("late_aim_mean"), 120.0)
    return damage + 0.25 * hits + 0.01 * fired - 0.02 * max(0.0, aim - 60.0)


def summarize_train_jsonl(rows: list[dict[str, Any]], learner_side: str = "A") -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "fired_steps": 0,
            "hits": 0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "late_fired_steps_mean": 0.0,
            "late_hits_mean": 0.0,
            "late_aim_mean": 999.0,
        }
    vals = []
    for row in rows:
        if learner_side in row and isinstance(row[learner_side], dict):
            vals.append(row[learner_side])
        else:
            vals.append(row)
    late = vals[-10:]
    return {
        "rows": len(rows),
        "fired_steps": int(sum(safe_float(v.get("fired_steps", v.get("fire_count_A", 0))) for v in vals)),
        "hits": int(sum(safe_float(v.get("hits", v.get("hit_count", 0))) for v in vals)),
        "damage_dealt": float(sum(safe_float(v.get("damage_dealt"), 0.0) for v in vals)),
        "damage_taken": float(sum(safe_float(v.get("damage_taken"), 0.0) for v in vals)),
        "late_fired_steps_mean": mean([safe_float(v.get("fired_steps", v.get("fire_count_A", 0))) for v in late]) if late else 0.0,
        "late_hits_mean": mean([safe_float(v.get("hits", v.get("hit_count", 0))) for v in late]) if late else 0.0,
        "late_aim_mean": mean([safe_float(v.get("aim_mean"), 999.0) for v in late]) if late else 999.0,
    }
