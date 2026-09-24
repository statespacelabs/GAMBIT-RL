#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_policy_states(root: Path) -> dict[str, np.ndarray]:
    paths = sorted(root.glob("*/policy_state_shard.npz"))
    if not paths:
        raise FileNotFoundError(f"no policy_state_shard.npz files under {root}")
    keys = [
        "obs",
        "agent_side",
        "area_id",
        "episode_id",
        "policy_mu",
        "policy_logstd",
        "sampled_action",
        "env_applied_action",
        "shoot_pressed",
        "shot_fired",
        "hit",
        "damage_dealt",
        "damage_taken",
        "aim_error",
        "yaw_error_deg",
        "pitch_error_deg",
        "distance_to_target",
        "target_visible",
        "target_los",
        "combat_ready",
        "post_reset_flag",
        "terminal_flag",
        "death_flag",
        "blocked_fire_reason",
        "action_mode_used",
    ]
    parts: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    source_job = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        n = len(data["obs"])
        for key in keys:
            if key not in data.files:
                raise KeyError(f"{path} missing required field {key}")
            parts[key].append(data[key])
        source_job.append(np.asarray([path.parent.name] * n, dtype=str))
    out = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    out["source_job"] = np.concatenate(source_job, axis=0)
    return out


def choose_side_balanced_indices(
    rng: np.random.Generator,
    side: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    ambiguous: np.ndarray,
    target_positive_rate: float,
    ambiguous_fraction: float,
    priority_mask: np.ndarray | None = None,
    priority_mult: float = 1.0,
) -> np.ndarray:
    selected_by_side = []
    side_totals = []
    candidates = {}
    for label in ("A", "B"):
        side_mask = side == label
        pos = np.flatnonzero(side_mask & positive)
        neg = np.flatnonzero(side_mask & negative)
        amb = np.flatnonzero(side_mask & ambiguous)
        max_total_by_pos = int(np.floor(len(pos) / target_positive_rate)) if target_positive_rate > 0 else 0
        max_total_by_nonpos = int(np.floor((len(neg) + len(amb)) / max(1e-6, 1.0 - target_positive_rate)))
        total = max(0, min(max_total_by_pos, max_total_by_nonpos))
        candidates[label] = (pos, neg, amb)
        side_totals.append(total)
    side_total = min(side_totals)
    if side_total <= 0:
        raise RuntimeError(f"unable to form balanced relabel set at target positive rate {target_positive_rate}")
    pos_keep = max(1, int(round(side_total * target_positive_rate)))
    nonpos_keep = side_total - pos_keep
    amb_keep = min(int(round(nonpos_keep * ambiguous_fraction)), min(len(candidates["A"][2]), len(candidates["B"][2])))
    neg_keep = nonpos_keep - amb_keep

    def choose(pool: np.ndarray, size: int) -> np.ndarray:
        if size <= 0:
            return np.asarray([], dtype=np.int64)
        if priority_mask is None or priority_mult <= 1.0:
            return rng.choice(pool, size=size, replace=False)
        weights = np.ones(len(pool), dtype=np.float64)
        weights[np.asarray(priority_mask[pool], dtype=bool)] *= priority_mult
        weights = weights / max(float(np.sum(weights)), 1.0e-12)
        return rng.choice(pool, size=size, replace=False, p=weights)

    for label in ("A", "B"):
        pos, neg, amb = candidates[label]
        chosen = [
            choose(pos, pos_keep),
            choose(neg, neg_keep),
        ]
        if amb_keep > 0:
            chosen.append(choose(amb, amb_keep))
        selected_by_side.append(np.concatenate(chosen))
    selected = np.concatenate(selected_by_side)
    rng.shuffle(selected)
    return selected


def make_labels(
    data: dict[str, np.ndarray],
    idx: np.ndarray,
    yaw_scale: float,
    pitch_scale: float,
    max_look_label: float,
) -> dict[str, np.ndarray]:
    obs = data["obs"][idx].astype(np.float32)
    action = data["env_applied_action"][idx].astype(np.float32).copy()
    yaw = data["yaw_error_deg"][idx].astype(np.float32)
    pitch = data["pitch_error_deg"][idx].astype(np.float32)
    aim = data["aim_error"][idx].astype(np.float32)
    visible = data["target_visible"][idx].astype(np.float32) > 0.5
    los = data["target_los"][idx].astype(np.float32) > 0.5
    positive = visible & los & (aim <= 35.0)
    negative = (~visible) | (~los) | (aim >= 75.0)
    ambiguous = ~(positive | negative)
    action[:, 2] = np.clip(yaw / yaw_scale, -max_look_label, max_look_label)
    action[:, 3] = np.clip(-pitch / pitch_scale, -max_look_label, max_look_label)
    action[:, 4] = positive.astype(np.float32)
    shoot_mask = (positive | negative).astype(np.float32)
    shoot_intent = positive.astype(np.float32)
    return {
        "obs": obs,
        "action": action.astype(np.float32),
        "shoot_intent": shoot_intent.astype(np.float32),
        "shoot_mask": shoot_mask.astype(np.float32),
        "aim_error": aim.astype(np.float32),
        "yaw_error_deg": yaw.astype(np.float32),
        "pitch_error_deg": pitch.astype(np.float32),
        "target_visible": visible.astype(np.float32),
        "target_los": los.astype(np.float32),
        "positive": positive,
        "negative": negative,
        "ambiguous": ambiguous,
    }


def sample_weights(data: dict[str, np.ndarray], idx: np.ndarray, labels: dict[str, np.ndarray]) -> np.ndarray:
    side = data["agent_side"][idx].astype(str)
    weights = np.ones(len(idx), dtype=np.float32)
    weights[side == "B"] *= 1.25
    weights[data["damage_taken"][idx].astype(np.float32) > 0.0] *= 2.0
    weights[data["post_reset_flag"][idx].astype(np.float32) > 0.5] *= 2.0
    visible_no_hit = (
        (data["target_visible"][idx].astype(np.float32) > 0.5)
        & (data["hit"][idx].astype(np.float32) < 0.5)
        & (data["aim_error"][idx].astype(np.float32) <= 75.0)
    )
    weights[visible_no_hit] *= 1.5
    weights[labels["ambiguous"]] *= 0.25
    return weights / max(float(np.mean(weights)), 1e-6)


def sign_accuracy(values: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & (np.abs(values) > 1.0) & (np.abs(labels) > 1.0e-6)
    if not np.any(valid):
        return 1.0
    return float(np.mean(np.sign(values[valid]) == np.sign(labels[valid])))


def summarize(data: dict[str, np.ndarray], idx: np.ndarray, labels: dict[str, np.ndarray], weights: np.ndarray) -> dict[str, Any]:
    side = data["agent_side"][idx].astype(str)
    action = labels["action"]
    look = action[:, 2:4]
    aim = labels["aim_error"]
    off_target = aim >= 30.0
    yaw_acc = sign_accuracy(labels["yaw_error_deg"], action[:, 2], off_target)
    pitch_acc = sign_accuracy(-labels["pitch_error_deg"], action[:, 3], off_target)
    positives_respect_gate = bool(
        np.all((labels["target_visible"][labels["positive"]] > 0.5) & (labels["target_los"][labels["positive"]] > 0.5))
    )
    pos_rate = float(np.mean(labels["shoot_intent"] > 0.5)) if len(idx) else 0.0
    sat95 = float(np.mean(np.abs(look) > 0.95)) if len(idx) else 0.0
    sat80 = float(np.mean(np.abs(look) > 0.80)) if len(idx) else 0.0
    side_counts = Counter(side)
    side_a = int(side_counts.get("A", 0))
    side_b = int(side_counts.get("B", 0))
    side_balance_delta = abs(side_a - side_b) / max(side_a + side_b, 1)
    post_reset_count = int(np.sum(data["post_reset_flag"][idx].astype(np.float32) > 0.5))
    under_fire_count = int(np.sum(data["damage_taken"][idx].astype(np.float32) > 0.0))
    visible_no_hit_count = int(
        np.sum(
            (data["target_visible"][idx].astype(np.float32) > 0.5)
            & (data["hit"][idx].astype(np.float32) < 0.5)
            & (data["aim_error"][idx].astype(np.float32) <= 75.0)
        )
    )
    failures = []
    if len(idx) < 500000:
        failures.append(f"relabelled samples {len(idx)} < 500000")
    if side_balance_delta > 0.10:
        failures.append(f"A/B balance delta {side_balance_delta:.4f} > 0.10")
    if post_reset_count < 20000:
        failures.append(f"post-reset relabelled {post_reset_count} < 20000")
    if under_fire_count < 40000:
        failures.append(f"under-fire relabelled {under_fire_count} < 40000")
    if not (0.20 <= pos_rate <= 0.45):
        failures.append(f"shoot positive rate {pos_rate:.4f} outside [0.20, 0.45]")
    if sat95 != 0.0:
        failures.append(f"abs(look_label)>0.95 rate {sat95:.6f} != 0")
    if sat80 >= 0.05:
        failures.append(f"abs(look_label)>0.80 rate {sat80:.6f} >= 0.05")
    if min(yaw_acc, pitch_acc) < 0.80:
        failures.append(f"corrective sign accuracy too low yaw={yaw_acc:.4f} pitch={pitch_acc:.4f}")
    if not positives_respect_gate:
        failures.append("shoot positives violate target_visible/target_los gate")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "sample_count": int(len(idx)),
        "agent_side_counts": dict(side_counts),
        "side_balance_delta": float(side_balance_delta),
        "shoot_positive_count": int(np.sum(labels["shoot_intent"] > 0.5)),
        "shoot_positive_rate": pos_rate,
        "shoot_mask_count": int(np.sum(labels["shoot_mask"] > 0.5)),
        "ambiguous_count": int(np.sum(labels["ambiguous"])),
        "look_label_abs_gt_0_95_rate": sat95,
        "look_label_abs_gt_0_80_rate": sat80,
        "look_label_abs_max": float(np.max(np.abs(look))) if len(idx) else 0.0,
        "yaw_corrective_sign_accuracy": yaw_acc,
        "pitch_corrective_sign_accuracy": pitch_acc,
        "target_visible_los_respected_for_shoot_positives": positives_respect_gate,
        "post_reset_count": post_reset_count,
        "under_fire_count": under_fire_count,
        "visible_no_hit_count": visible_no_hit_count,
        "sample_weight_mean": float(np.mean(weights)) if len(weights) else 0.0,
        "sample_weight_min": float(np.min(weights)) if len(weights) else 0.0,
        "sample_weight_max": float(np.max(weights)) if len(weights) else 0.0,
        "aim_bin_counts": {
            "0_15": int(np.sum((aim >= 0.0) & (aim < 15.0))),
            "15_30": int(np.sum((aim >= 15.0) & (aim < 30.0))),
            "30_60": int(np.sum((aim >= 30.0) & (aim < 60.0))),
            "60_90": int(np.sum((aim >= 60.0) & (aim < 90.0))),
            "90_150": int(np.sum((aim >= 90.0) & (aim <= 150.0))),
        },
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GvG DAgger v2.1 Relabel Summary",
        "",
        f"Status: {summary['status']}",
        f"Sample count: {summary['sample_count']}",
        f"Agent side counts: {summary['agent_side_counts']}",
        f"Shoot positive rate: {summary['shoot_positive_rate']:.6f}",
        f"Look label abs >0.95 rate: {summary['look_label_abs_gt_0_95_rate']:.6f}",
        f"Look label abs >0.80 rate: {summary['look_label_abs_gt_0_80_rate']:.6f}",
        f"Yaw corrective sign accuracy: {summary['yaw_corrective_sign_accuracy']:.6f}",
        f"Pitch corrective sign accuracy: {summary['pitch_corrective_sign_accuracy']:.6f}",
        f"Post-reset count: {summary['post_reset_count']}",
        f"Under-fire count: {summary['under_fire_count']}",
        f"Visible/no-hit count: {summary['visible_no_hit_count']}",
        f"Aim bins: {summary['aim_bin_counts']}",
    ]
    if summary["failures"]:
        lines.extend(["", "## Failures"])
        lines.extend(f"- {failure}" for failure in summary["failures"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=9300)
    parser.add_argument("--target-positive-rate", type=float, default=0.30)
    parser.add_argument("--ambiguous-fraction", type=float, default=0.10)
    parser.add_argument("--yaw-scale-deg", type=float, default=55.0)
    parser.add_argument("--pitch-scale-deg", type=float, default=55.0)
    parser.add_argument("--max-look-label", type=float, default=0.55)
    parser.add_argument("--postreset-sample-mult", type=float, default=6.0)
    args = parser.parse_args()
    policy_root = Path(args.policy_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    data = load_policy_states(policy_root)
    side = data["agent_side"].astype(str)
    aim = data["aim_error"].astype(np.float32)
    visible = data["target_visible"].astype(np.float32) > 0.5
    los = data["target_los"].astype(np.float32) > 0.5
    positive = visible & los & (aim <= 35.0)
    negative = (~visible) | (~los) | (aim >= 75.0)
    ambiguous = ~(positive | negative)
    idx = choose_side_balanced_indices(
        rng,
        side,
        positive,
        negative,
        ambiguous,
        args.target_positive_rate,
        args.ambiguous_fraction,
        data["post_reset_flag"].astype(np.float32) > 0.5,
        args.postreset_sample_mult,
    )
    labels = make_labels(data, idx, args.yaw_scale_deg, args.pitch_scale_deg, args.max_look_label)
    weights = sample_weights(data, idx, labels)
    np.savez_compressed(
        out / "gvg_dagger_v21_relabelled_shard.npz",
        obs=labels["obs"],
        action=labels["action"],
        shoot_intent=labels["shoot_intent"],
        shoot_mask=labels["shoot_mask"],
        aim_error=labels["aim_error"],
        yaw_error_deg=labels["yaw_error_deg"],
        pitch_error_deg=labels["pitch_error_deg"],
        target_visible=labels["target_visible"],
        target_los=labels["target_los"],
        agent_side=data["agent_side"][idx].astype(str),
        sample_weight=weights.astype(np.float32),
        policy_action=data["env_applied_action"][idx].astype(np.float32),
        policy_mu=data["policy_mu"][idx].astype(np.float32),
        policy_logstd=data["policy_logstd"][idx].astype(np.float32),
        fired=data["shot_fired"][idx].astype(np.float32),
        hit=data["hit"][idx].astype(np.float32),
        damage_taken=data["damage_taken"][idx].astype(np.float32),
        post_reset_flag=data["post_reset_flag"][idx].astype(np.float32),
        source_job=data["source_job"][idx].astype(str),
        source_index=idx.astype(np.int64),
    )
    summary = summarize(data, idx, labels, weights)
    summary.update(
        {
            "policy_root": str(policy_root),
            "output_dir": str(out),
            "relabelled_shard": str(out / "gvg_dagger_v21_relabelled_shard.npz"),
            "target_positive_rate": args.target_positive_rate,
            "ambiguous_fraction": args.ambiguous_fraction,
            "yaw_scale_deg": args.yaw_scale_deg,
            "pitch_scale_deg": args.pitch_scale_deg,
            "max_look_label": args.max_look_label,
            "postreset_sample_mult": args.postreset_sample_mult,
        }
    )
    write_json(out / "GVG_DAGGER_V21_RELABEL_SUMMARY.json", summary)
    write_report(out / "GVG_DAGGER_V21_RELABEL_REPORT.md", summary)
    print(json.dumps({"status": summary["status"], "samples": summary["sample_count"], "shoot_positive_rate": summary["shoot_positive_rate"]}, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
