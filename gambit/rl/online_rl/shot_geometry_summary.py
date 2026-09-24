"""Analyze Phase 3V true Unity shot geometry logs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(vals)
    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "median": s[len(s) // 2],
        "min": s[0],
        "max": s[-1],
    }


def analyze_shot_geometry(
    unity_jsonl: Path,
    proxy_jsonl: Path | None = None,
) -> dict[str, Any]:
    rows = []
    if unity_jsonl.exists():
        with open(unity_jsonl, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    total = len(rows)
    aligned = [r for r in rows if r.get("aligned_at_fire")]
    ray_hits = [r for r in rows if r.get("raycast_hit_true")]
    same_step_dmg = [r for r in rows if float(r.get("damage_applied_same_step", 0)) > 0]
    same_step_reward = [r for r in rows if float(r.get("env_reward_same_step", 0)) > 0.01]

    aim_all = [float(r.get("aim_err_at_fire", 0)) for r in rows]
    aim_aligned = [float(r.get("aim_err_at_fire", 0)) for r in aligned]
    aim_hits = [float(r.get("aim_err_at_fire", 0)) for r in ray_hits]
    aim_misses = [
        float(r.get("aim_err_at_fire", 0)) for r in rows if not r.get("raycast_hit_true")
    ]
    dist_all = [float(r.get("target_distance_at_fire", 0)) for r in rows]

    miss_reasons = Counter(
        r.get("raycast_miss_reason", "unknown") or "unknown"
        for r in rows
        if not r.get("raycast_hit_true")
    )
    body_parts = Counter(r.get("raycast_hit_body_part", "") for r in ray_hits)
    colliders = Counter(r.get("raycast_hit_collider", "") for r in ray_hits)

    aligned_miss = [r for r in aligned if not r.get("raycast_hit_true")]

    summary: dict[str, Any] = {
        "total_sampled_shots": total,
        "aligned_sampled_shots": len(aligned),
        "aligned_fraction": len(aligned) / total if total else 0.0,
        "true_raycast_hits": len(ray_hits),
        "true_raycast_hit_rate": len(ray_hits) / total if total else 0.0,
        "true_raycast_misses": total - len(ray_hits),
        "damage_same_step": len(same_step_dmg),
        "damage_same_step_rate": len(same_step_dmg) / total if total else 0.0,
        "env_reward_same_step": len(same_step_reward),
        "env_reward_same_step_rate": len(same_step_reward) / total if total else 0.0,
        "aligned_but_miss": len(aligned_miss),
        "aligned_miss_fraction_of_aligned": (
            len(aligned_miss) / len(aligned) if aligned else 0.0
        ),
        "aim_err_all": _stats(aim_all),
        "aim_err_aligned": _stats(aim_aligned),
        "aim_err_hits": _stats(aim_hits),
        "aim_err_misses": _stats(aim_misses),
        "target_distance_at_fire": _stats(dist_all),
        "weapon_spread_angle": _stats(
            [float(r.get("weapon_spread_angle", 0)) for r in rows]
        ),
        "recoil_state": _stats([float(r.get("recoil_state", 0)) for r in rows]),
        "miss_reason_distribution": dict(miss_reasons),
        "hit_body_part_distribution": dict(body_parts),
        "hit_collider_distribution": dict(colliders),
        "fires_with_same_step_damage": len(same_step_dmg),
        "fires_with_same_step_env_reward": len(same_step_reward),
        "fires_with_damage_within_3_steps": None,
    }

    if rows:
        center_vs_chest = []
        for r in rows:
            cx = float(r.get("target_position_x", 0))
            cy = float(r.get("target_position_y", 0))
            cz = float(r.get("target_position_z", 0))
            tx = float(r.get("target_chest_x", cx))
            ty = float(r.get("target_chest_y", cy))
            tz = float(r.get("target_chest_z", cz))
            center_vs_chest.append(
                ((tx - cx) ** 2 + (ty - cy) ** 2 + (tz - cz) ** 2) ** 0.5
            )
        summary["center_to_chest_offset_m"] = _stats(center_vs_chest)

    if proxy_jsonl and proxy_jsonl.exists():
        proxy_rows = [
            json.loads(line)
            for line in open(proxy_jsonl, encoding="utf-8")
            if line.strip()
        ]
        proxy_hits = sum(1 for r in proxy_rows if r.get("raycast_hit_at_fire"))
        summary["proxy_jsonl_samples"] = len(proxy_rows)
        summary["proxy_hit_rate"] = proxy_hits / len(proxy_rows) if proxy_rows else 0.0

    return summary


def compact_shot_geometry_metrics(
    unity_jsonl: Path,
    prefix: str = "shot_geometry",
) -> dict[str, float]:
    """Return flat metrics dict for ppo_train.jsonl from Unity shot log."""
    if not unity_jsonl.exists():
        return {}
    summary = analyze_shot_geometry(unity_jsonl)
    total = summary.get("total_sampled_shots", 0)
    aligned = summary.get("aligned_sampled_shots", 0)
    hits = summary.get("true_raycast_hits", 0)
    out: dict[str, float] = {
        f"{prefix}/sample_count": float(total),
        f"{prefix}/true_raycast_hit_rate": float(summary.get("true_raycast_hit_rate", 0.0)),
        f"{prefix}/aligned_sample_fraction": float(summary.get("aligned_fraction", 0.0)),
        f"{prefix}/aligned_but_miss_rate": float(
            summary.get("aligned_miss_fraction_of_aligned", 0.0)
        ),
        f"{prefix}/damage_same_step_rate": float(
            summary.get("damage_same_step_rate", 0.0)
        ),
        f"{prefix}/aim_err_at_fire_mean": float(
            summary.get("aim_err_all", {}).get("mean", 0.0)
        ),
        f"{prefix}/aim_err_at_fire_median": float(
            summary.get("aim_err_all", {}).get("median", 0.0)
        ),
        f"{prefix}/aim_err_hits_mean": float(
            summary.get("aim_err_hits", {}).get("mean", 0.0)
        ),
        f"{prefix}/aim_err_misses_mean": float(
            summary.get("aim_err_misses", {}).get("mean", 0.0)
        ),
    }
    if aligned > 0:
        out[f"{prefix}/aligned_true_hit_rate"] = float(hits) / float(aligned)
    else:
        out[f"{prefix}/aligned_true_hit_rate"] = 0.0

    ray_summary = analyze_ray_vs_hurtbox(unity_jsonl)
    if ray_summary.get("sample_count", 0) > 0:
        out[f"{prefix}/ray_to_hurtbox_min_distance_mean"] = float(
            ray_summary.get("ray_to_hurtbox_min_distance_mean", 0.0)
        )
        out[f"{prefix}/weapon_ray_vs_crosshair_angle_mean"] = float(
            ray_summary.get("weapon_ray_vs_crosshair_angle_mean", 0.0)
        )
        if "ray_correction_applied_rate" in ray_summary:
            out[f"{prefix}/ray_correction_applied_rate"] = float(
                ray_summary["ray_correction_applied_rate"]
            )
            out[f"{prefix}/ray_correction_angle_applied_deg_mean"] = float(
                ray_summary.get("ray_correction_angle_applied_deg_mean", 0.0)
            )
            out[f"{prefix}/ray_correction_angle_applied_deg_median"] = float(
                ray_summary.get("ray_correction_angle_applied_deg_median", 0.0)
            )
    return out


def analyze_ray_vs_hurtbox(unity_jsonl: Path) -> dict[str, Any]:
    rows = []
    if unity_jsonl.exists():
        with open(unity_jsonl, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        return {"sample_count": 0}

    ray_dist = [float(r.get("ray_to_hurtbox_min_distance", 0)) for r in rows]
    true_err = [float(r.get("true_ray_hurtbox_error_deg", r.get("angle_weapon_ray_to_hurtbox_closest", 0))) for r in rows]
    old_err = [float(r.get("aim_err_at_fire", 0)) for r in rows]
    wvsc = [float(r.get("weapon_ray_vs_crosshair_angle", 0)) for r in rows]
    aligned_miss = [
        r for r in rows if r.get("aligned_at_fire") and not r.get("raycast_hit_true")
    ]
    aligned_miss_dist = [
        float(r.get("ray_to_hurtbox_min_distance", 0)) for r in aligned_miss
    ]
    inflate_vals = [float(r.get("target_hurtbox_inflate", 0)) for r in rows]
    ray_modes = [r.get("learner_shoot_ray_mode", "current") for r in rows]
    corr_applied = [bool(r.get("ray_correction_applied", False)) for r in rows]
    corr_angles = [
        float(r.get("ray_correction_angle_applied_deg", 0)) for r in rows
    ]
    corr_alphas = [float(r.get("ray_correction_alpha", 0)) for r in rows]

    result = {
        "sample_count": len(rows),
        "target_hurtbox_inflate": inflate_vals[0] if inflate_vals else 0.0,
        "learner_shoot_ray_mode": ray_modes[0] if ray_modes else "current",
        "ray_to_hurtbox_min_distance_mean": sum(ray_dist) / len(ray_dist),
        "ray_to_hurtbox_min_distance_median": sorted(ray_dist)[len(ray_dist) // 2],
        "ray_to_hurtbox_min_distance_for_aligned_misses_mean": (
            sum(aligned_miss_dist) / len(aligned_miss_dist) if aligned_miss_dist else 0.0
        ),
        "weapon_ray_vs_crosshair_angle_mean": sum(wvsc) / len(wvsc),
        "weapon_ray_vs_crosshair_angle_median": sorted(wvsc)[len(wvsc) // 2],
        "true_ray_hurtbox_error_mean": sum(true_err) / len(true_err),
        "true_ray_hurtbox_error_median": sorted(true_err)[len(true_err) // 2],
        "old_aim_err_at_fire_mean": sum(old_err) / len(old_err),
        "aligned_miss_count": len(aligned_miss),
    }
    if corr_applied:
        applied_n = sum(1 for x in corr_applied if x)
        result["ray_correction_applied_rate"] = applied_n / len(corr_applied)
        result["ray_correction_alpha"] = corr_alphas[0] if corr_alphas else 0.0
        if corr_angles:
            result["ray_correction_angle_applied_deg_mean"] = sum(corr_angles) / len(
                corr_angles
            )
            result["ray_correction_angle_applied_deg_median"] = sorted(corr_angles)[
                len(corr_angles) // 2
            ]
    return result


def write_ray_vs_hurtbox_summary(
    log_dir: Path,
    unity_jsonl_name: str = "shot_debug_unity.jsonl",
    output_name: str = "ray_vs_hurtbox_summary.md",
) -> Path | None:
    log_dir = Path(log_dir)
    unity_path = log_dir / unity_jsonl_name
    if not unity_path.exists():
        return None

    summary = analyze_ray_vs_hurtbox(unity_path)
    if summary.get("sample_count", 0) == 0:
        return None

    out = log_dir / output_name
    lines = [
        "# Ray vs Hurtbox Summary",
        "",
        f"Source: `{unity_path.name}`",
        f"- Samples: {summary['sample_count']}",
        f"- target_hurtbox_inflate: {summary.get('target_hurtbox_inflate', 0.0)}",
        f"- learner_shoot_ray_mode: {summary.get('learner_shoot_ray_mode', 'current')}",
        "",
        "## Ray to hurtbox distance (m)",
        f"- Mean: {summary['ray_to_hurtbox_min_distance_mean']:.3f}",
        f"- Median: {summary['ray_to_hurtbox_min_distance_median']:.3f}",
        f"- Aligned-miss mean: "
        f"{summary['ray_to_hurtbox_min_distance_for_aligned_misses_mean']:.3f}",
        "",
        "## Weapon ray vs crosshair (degrees)",
        f"- Mean: {summary['weapon_ray_vs_crosshair_angle_mean']:.2f}°",
        f"- Median: {summary['weapon_ray_vs_crosshair_angle_median']:.2f}°",
        "",
        "## Old vs true aim error (degrees)",
        f"- Old aim err at fire mean: {summary.get('old_aim_err_at_fire_mean', 0):.2f}°",
        f"- True ray hurtbox err mean: {summary.get('true_ray_hurtbox_error_mean', 0):.2f}°",
        f"- True ray hurtbox err median: {summary.get('true_ray_hurtbox_error_median', 0):.2f}°",
        "",
        f"- Aligned misses: {summary['aligned_miss_count']}",
    ]
    if "ray_correction_applied_rate" in summary:
        lines.extend([
            "",
            "## Ray correction (Phase 3AA)",
            f"- ray_correction_alpha: {summary.get('ray_correction_alpha', 0.0)}",
            f"- correction applied rate: {summary['ray_correction_applied_rate']:.3%}",
            f"- correction angle mean: "
            f"{summary.get('ray_correction_angle_applied_deg_mean', 0):.2f}°",
            f"- correction angle median: "
            f"{summary.get('ray_correction_angle_applied_deg_median', 0):.2f}°",
        ])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = log_dir / "ray_vs_hurtbox_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def write_shot_geometry_summary(
    log_dir: Path,
    unity_jsonl_name: str = "shot_debug_unity.jsonl",
    proxy_jsonl_name: str = "shot_debug.jsonl",
    output_name: str = "shot_geometry_summary.md",
) -> Path | None:
    log_dir = Path(log_dir)
    unity_path = log_dir / unity_jsonl_name
    proxy_path = log_dir / proxy_jsonl_name
    if not unity_path.exists():
        return None

    summary = analyze_shot_geometry(
        unity_path, proxy_path if proxy_path.exists() else None
    )
    out = log_dir / output_name

    lines = [
        "# Shot Geometry Summary",
        "",
        f"Source: `{unity_path.name}`",
        "",
        "## Counts",
        f"- Total sampled shots: {summary['total_sampled_shots']}",
        f"- Aligned sampled shots: {summary['aligned_sampled_shots']} "
        f"({summary['aligned_fraction']:.1%})",
        f"- True raycast hits: {summary['true_raycast_hits']} "
        f"(rate {summary['true_raycast_hit_rate']:.3%})",
        f"- True raycast misses: {summary['true_raycast_misses']}",
        f"- Aligned but miss: {summary['aligned_but_miss']} "
        f"({summary['aligned_miss_fraction_of_aligned']:.1%} of aligned)",
        "",
        "## Damage / reward timing",
        f"- Damage same step: {summary['damage_same_step']} "
        f"(rate {summary['damage_same_step_rate']:.3%})",
        f"- Env reward same step (>0.01): {summary['env_reward_same_step']} "
        f"(rate {summary['env_reward_same_step_rate']:.3%})",
        "",
        "## Aim error at fire (degrees)",
        "",
        "| Group | Count | Mean | Median | Min | Max |",
        "|-------|-------|------|--------|-----|-----|",
    ]
    for label, key in [
        ("All", "aim_err_all"),
        ("Aligned", "aim_err_aligned"),
        ("Hits", "aim_err_hits"),
        ("Misses", "aim_err_misses"),
    ]:
        s = summary[key]
        lines.append(
            f"| {label} | {int(s['count'])} | {s['mean']:.1f} | {s['median']:.1f} | "
            f"{s['min']:.1f} | {s['max']:.1f} |"
        )

    lines.extend([
        "",
        "## Target distance at fire",
        f"- Mean: {summary['target_distance_at_fire']['mean']:.1f}m, "
        f"median: {summary['target_distance_at_fire']['median']:.1f}m",
        "",
        "## Spread / recoil",
        f"- Weapon spread angle: mean {summary['weapon_spread_angle']['mean']:.2f}°",
        f"- Recoil state: mean {summary['recoil_state']['mean']:.2f}",
        "",
        "## Miss reasons",
    ])
    for reason, count in sorted(
        summary["miss_reason_distribution"].items(), key=lambda x: -x[1]
    ):
        lines.append(f"- {reason}: {count}")

    lines.extend(["", "## Hit body parts"])
    for part, count in sorted(
        summary["hit_body_part_distribution"].items(), key=lambda x: -x[1]
    ):
        if part:
            lines.append(f"- {part}: {count}")

    if "center_to_chest_offset_m" in summary:
        c = summary["center_to_chest_offset_m"]
        lines.extend([
            "",
            "## Target point mismatch (center vs chest)",
            f"- Offset mean: {c['mean']:.2f}m, median: {c['median']:.2f}m",
        ])

    if summary.get("proxy_jsonl_samples"):
        lines.extend([
            "",
            "## Python proxy comparison",
            f"- Proxy samples: {summary['proxy_jsonl_samples']}",
            f"- Proxy hit rate: {summary['proxy_hit_rate']:.3%}",
        ])

    return out


def write_compact_metrics(
    log_dir: Path,
    unity_jsonl_name: str = "shot_debug_unity.jsonl",
    output_name: str = "compact_metrics.json",
    rollout_metrics: dict | None = None,
) -> Path | None:
    """Write flat geometry + optional rollout metrics JSON for Phase 3AB job6."""
    log_dir = Path(log_dir)
    unity_path = log_dir / unity_jsonl_name
    if not unity_path.exists():
        return None

    shot_summary = analyze_shot_geometry(unity_path)
    ray_summary = analyze_ray_vs_hurtbox(unity_path)
    compact: dict[str, float | int | str] = {
        "true_raycast_hit_rate": float(shot_summary.get("true_raycast_hit_rate", 0.0)),
        "aligned_but_miss_rate": float(
            shot_summary.get("aligned_miss_fraction_of_aligned", 0.0)
        ),
        "same_step_damage_rate": float(shot_summary.get("damage_same_step_rate", 0.0)),
        "correction_applied_rate": float(
            ray_summary.get("ray_correction_applied_rate", 0.0)
        ),
        "correction_angle_mean": float(
            ray_summary.get("ray_correction_angle_applied_deg_mean", 0.0)
        ),
        "correction_angle_median": float(
            ray_summary.get("ray_correction_angle_applied_deg_median", 0.0)
        ),
        "true_ray_hurtbox_error_mean": float(
            ray_summary.get("true_ray_hurtbox_error_mean", 0.0)
        ),
        "ray_to_hurtbox_distance_mean": float(
            ray_summary.get("ray_to_hurtbox_min_distance_mean", 0.0)
        ),
        "old_aim_err_at_fire_mean": float(
            ray_summary.get("old_aim_err_at_fire_mean", 0.0)
        ),
        "sample_count": int(ray_summary.get("sample_count", 0)),
        "learner_shoot_ray_mode": str(ray_summary.get("learner_shoot_ray_mode", "current")),
    }
    if rollout_metrics:
        for key in ("damage/kills", "damage/target_hp_min", "aim/total_hits"):
            if key in rollout_metrics:
                compact[key.replace("/", "_")] = rollout_metrics[key]

    out = log_dir / output_name
    out.write_text(json.dumps(compact, indent=2), encoding="utf-8")
    return out
