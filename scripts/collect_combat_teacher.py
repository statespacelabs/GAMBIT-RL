#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.combat_observations import SCHEMA_VERSION, collect_records, discover_slot_map
from scripts.combat_runtime import (
    HARD_BLOCK_REASONS,
    OBS_DIM,
    RuntimeConfig,
    collect_all_weapon_events,
    configure_unity_env,
    launch_unity_env,
    sha256_file,
    summarize_weapon_events,
    validate_specs,
    write_json,
)
from gambit.rl.online_rl.selfplay_runner import make_action_tuple, policy_action_to_buffers


ROOT = Path('experiments/phase3v2/phase3v2_c_local45_teacher_dataset')
AIM_BINS = [(0.0, 15.0), (15.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 150.0001)]
AIM_BIN_NAMES = ['0_15', '15_30', '30_60', '60_90', '90_150']
TARGET_FRACS = {'0_15': 0.15, '15_30': 0.20, '30_60': 0.25, '60_90': 0.20, '90_150': 0.20}


def aim_bin(aim: float) -> str:
    for name, (lo, hi) in zip(AIM_BIN_NAMES, AIM_BINS):
        if lo <= aim < hi:
            return name
    return 'out_of_range'


def teacher_label(obs: np.ndarray, step: int, shoot_pos: float, shoot_neg: float) -> tuple[np.ndarray, float, float, float, str]:
    yaw = float(obs[20])
    pitch = float(obs[21])
    aim = float(max(0.0, obs[22]))
    action = np.zeros(8, dtype=np.float32)
    action[0] = 0.18 * (1.0 if ((step // 160) % 2 == 0) else -1.0)
    action[1] = 0.08 * math.sin(step * 0.021)
    action[2] = np.clip(yaw / 45.0, -1.0, 1.0)
    action[3] = np.clip(-pitch / 45.0, -1.0, 1.0)
    if np.isfinite(aim) and aim <= shoot_pos:
        shoot_intent = 1.0
        shoot_mask = 1.0
    elif (not np.isfinite(aim)) or aim >= shoot_neg:
        shoot_intent = 0.0
        shoot_mask = 1.0
    else:
        shoot_intent = 0.0
        shoot_mask = 0.0
    action[4] = shoot_intent
    return action, shoot_intent, shoot_mask, aim, aim_bin(aim)


def perturb_action(rng: np.random.Generator, step: int) -> np.ndarray:
    action = np.zeros(8, dtype=np.float32)
    action[0] = 0.25 * (1.0 if ((step // 80) % 2 == 0) else -1.0)
    action[1] = 0.15 * math.sin(step * 0.11)
    action[2] = rng.uniform(-1.0, 1.0)
    action[3] = rng.uniform(-0.75, 0.75)
    action[4] = 0.0
    return action


def make_targets(total: int) -> dict[str, dict[str, int]]:
    per_side = total // 2
    targets: dict[str, dict[str, int]] = {'A': {}, 'B': {}}
    for side in ('A', 'B'):
        assigned = 0
        for name in AIM_BIN_NAMES[:-1]:
            n = int(round(per_side * TARGET_FRACS[name]))
            targets[side][name] = n
            assigned += n
        targets[side][AIM_BIN_NAMES[-1]] = per_side - assigned
    return targets


def targets_met(counts: dict[str, Counter], targets: dict[str, dict[str, int]]) -> bool:
    return all(counts[s][b] >= targets[s][b] for s in ('A', 'B') for b in AIM_BIN_NAMES)


def needs_high_bins(side: str, counts: dict[str, Counter], targets: dict[str, dict[str, int]]) -> bool:
    return counts[side]['60_90'] < targets[side]['60_90'] or counts[side]['90_150'] < targets[side]['90_150']


def record_row(
    rows: dict[str, list[Any]],
    counts: dict[str, Counter],
    targets: dict[str, dict[str, int]],
    side: str,
    area: int,
    obs: np.ndarray,
    reward: float,
    step: int,
    mode: str,
    shoot_pos: float,
    shoot_neg: float,
) -> bool:
    action, shoot, mask, aim, bin_name = teacher_label(obs, step, shoot_pos, shoot_neg)
    if side not in ('A', 'B') or bin_name not in AIM_BIN_NAMES:
        return False
    if counts[side][bin_name] >= targets[side][bin_name]:
        return False
    if action.shape != (8,) or not np.isfinite(action).all() or np.any(action[:4] < -1.0) or np.any(action[:4] > 1.0):
        raise RuntimeError(f'invalid teacher action: {action.tolist()}')
    rows['obs'].append(obs.copy())
    rows['action'].append(action.copy())
    rows['shoot_intent'].append(shoot)
    rows['shoot_mask'].append(mask)
    rows['aim_error'].append(aim)
    rows['fired'].append(float(obs[44] > 0.5))
    rows['hit'].append(float(reward > 0.05))
    rows['aim_bin'].append(bin_name)
    rows['side'].append(side)
    rows['area'].append(area)
    rows['bot_mode'].append(mode)
    counts[side][bin_name] += 1
    return True


def collect_segment(args: argparse.Namespace, name: str, game_mode: str, bot_mode: str, max_steps: int, targets: dict[str, dict[str, int]], counts: dict[str, Counter], rows: dict[str, list[Any]], port: int) -> dict[str, Any]:
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = RuntimeConfig(env_path=args.env_path, output_dir=str(out), base_port=port, seed=args.seed + port, time_scale=args.time_scale, device='cpu', game_mode=game_mode, player_b_bot_mode=bot_mode, num_areas=args.areas)
    weapon_log = configure_unity_env(out, cfg)
    rng = np.random.default_rng(args.seed + port)
    perturb_left: dict[int, int] = defaultdict(int)
    slot: dict[int, tuple[str, int]] = {}
    recorded = 0
    side_seen = Counter()
    env = None
    start = time.time()
    try:
        env = launch_unity_env(cfg)
        behavior, action_spec, _spec = validate_specs(env)
        if game_mode == 'GambitVsGambit':
            slot = discover_slot_map(env, behavior, args.areas)
        for step in range(max_steps):
            if targets_met(counts, targets):
                break
            dec, ter = env.get_steps(behavior)
            records = collect_records(dec, ter)
            for aid, obs, reward, _done in records:
                if obs.shape != (OBS_DIM,) or not np.isfinite(obs).all():
                    raise RuntimeError(f'{name}: invalid obs for agent {aid}')
                if game_mode == 'GambitVsGambit':
                    side, area = slot.get(aid, ('A' if aid % 2 == 0 else 'B', aid // 2))
                else:
                    side, area = 'A', min(args.areas - 1, aid % max(1, args.areas))
                side_seen[side] += 1
                if record_row(rows, counts, targets, side, int(area), obs, reward, step, name, args.shoot_pos_deg, args.shoot_neg_deg):
                    recorded += 1
            if len(dec):
                obs_batch = np.asarray(dec.obs[0], dtype=np.float32)
                for row, aid_raw in enumerate(dec.agent_id):
                    aid = int(aid_raw)
                    obs = obs_batch[row]
                    if game_mode == 'GambitVsGambit':
                        side, _area = slot.get(aid, ('A' if aid % 2 == 0 else 'B', aid // 2))
                    else:
                        side = 'A'
                    label, _shoot, _mask, aim, bin_name = teacher_label(obs, step, args.shoot_pos_deg, args.shoot_neg_deg)
                    need_high = side in ('A', 'B') and needs_high_bins(side, counts, targets)
                    good_bins_full = side in ('A', 'B') and bin_name in ('0_15', '15_30') and counts[side][bin_name] >= targets[side][bin_name]
                    if perturb_left[aid] <= 0 and ((need_high and aim < 95.0) or good_bins_full or (aim < 20.0 and rng.random() < args.perturb_fraction)):
                        perturb_left[aid] = int(rng.integers(args.perturb_min, args.perturb_max + 1))
                    if perturb_left[aid] > 0:
                        execute = perturb_action(rng, step)
                        perturb_left[aid] -= 1
                    else:
                        execute = label
                    continuous, discrete = policy_action_to_buffers(execute, action_spec)
                    env.set_action_for_agent(behavior, aid, make_action_tuple(continuous, discrete))
            t0 = time.monotonic()
            env.step()
            if time.monotonic() - t0 > args.deadlock_step_seconds:
                raise RuntimeError(f'{name}: Unity step exceeded {args.deadlock_step_seconds}s')
    finally:
        if env is not None:
            env.close()
    weapon = summarize_weapon_events(collect_all_weapon_events(weapon_log))
    summary = {
        'name': name,
        'game_mode': game_mode,
        'bot_mode': bot_mode,
        'max_steps': max_steps,
        'recorded': recorded,
        'side_seen': dict(side_seen),
        'counts_after': {s: dict(counts[s]) for s in ('A', 'B')},
        'slot_map': {str(k): {'side': v[0], 'area': v[1]} for k, v in sorted(slot.items())},
        'weapon_summary': weapon,
        'elapsed_sec': time.time() - start,
    }
    write_json(out / 'segment_summary.json', summary)
    return summary


def summarize_dataset(root: Path, rows: dict[str, list[Any]], segment_summaries: list[dict[str, Any]], targets: dict[str, dict[str, int]], failures: list[str]) -> dict[str, Any]:
    n = len(rows['obs'])
    obs = np.asarray(rows['obs'], dtype=np.float32).reshape(n, OBS_DIM)
    act = np.asarray(rows['action'], dtype=np.float32).reshape(n, 8)
    shoot = np.asarray(rows['shoot_intent'], dtype=np.float32)
    mask = np.asarray(rows['shoot_mask'], dtype=np.float32)
    aim = np.asarray(rows['aim_error'], dtype=np.float32)
    fired = np.asarray(rows['fired'], dtype=np.float32)
    hit = np.asarray(rows['hit'], dtype=np.float32)
    aim_bins = np.asarray(rows['aim_bin'])
    sides = np.asarray(rows['side'])
    areas = np.asarray(rows['area'], dtype=np.int32)
    modes = np.asarray(rows['bot_mode'])
    side_counts = Counter(sides.tolist())
    bin_counts = Counter(aim_bins.tolist())
    side_bin_counts = {s: dict(Counter(aim_bins[sides == s].tolist())) for s in ('A', 'B')}
    hard = Counter()
    owner_mismatch = 0
    for seg in segment_summaries:
        ws = seg.get('weapon_summary', {}) or {}
        hard.update({k: int(v) for k, v in (ws.get('hard_blocked_fire_count_by_reason') or {}).items()})
        owner_mismatch += int(ws.get('owner_mismatch_count', 0) or 0)
    structural = {k: int(v) for k, v in hard.items() if k in HARD_BLOCK_REASONS and int(v) > 0}
    if n <= 0 or not np.isfinite(obs).all() or not np.isfinite(act).all():
        failures.append('empty or non-finite dataset')
    if n < sum(sum(x.values()) for x in targets.values()):
        failures.append(f'sample_count {n} below target')
    if side_counts.get('A', 0) != side_counts.get('B', 0):
        failures.append(f'side imbalance: {dict(side_counts)}')
    shoot_rate = float(np.mean(shoot > 0.5)) if n else 0.0
    if not (0.20 <= shoot_rate <= 0.40):
        failures.append(f'shoot positive rate out of range: {shoot_rate}')
    if structural:
        failures.append(f'hard blocked-fire counts nonzero: {structural}')
    if owner_mismatch:
        failures.append(f'owner mismatch count nonzero: {owner_mismatch}')
    if np.any(act[:, :4] < -1.0) or np.any(act[:, :4] > 1.0):
        failures.append('continuous labels out of bounds')
    if not (np.any(fired[sides == 'A'] > 0.5) and np.any(fired[sides == 'B'] > 0.5)):
        failures.append('both sides did not fire in teacher rows')
    if not (np.any(hit[sides == 'A'] > 0.5) and np.any(hit[sides == 'B'] > 0.5)):
        failures.append('both sides did not hit in teacher rows')
    job = root / 'job_0_local45_mixed'
    job.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(job / 'teacher_shard_v2.npz', obs=obs, action=act, shoot_intent=shoot, shoot_mask=mask, aim_error=aim, fired=fired, hit=hit, aim_bin=aim_bins, side=sides, area=areas, bot_mode=modes)
    summary = {
        'status': 'PASS' if not failures else 'FAIL',
        'failures': failures,
        'root': str(root),
        'shard': str(job / 'teacher_shard_v2.npz'),
        'obs_schema_version': SCHEMA_VERSION,
        'obs_dim': OBS_DIM,
        'sample_count': int(n),
        'side_counts': dict(side_counts),
        'area_counts': {str(k): int(v) for k, v in sorted(Counter(areas.tolist()).items())},
        'aim_bin_counts': dict(bin_counts),
        'side_aim_bin_counts': side_bin_counts,
        'shoot_intent_positive_count': int(np.sum(shoot > 0.5)),
        'shoot_intent_negative_count': int(np.sum((shoot < 0.5) & (mask > 0.5))),
        'shoot_intent_ambiguous_count': int(np.sum(mask < 0.5)),
        'shoot_intent_positive_rate': shoot_rate,
        'fired_count': int(np.sum(fired > 0.5)),
        'hit_count': int(np.sum(hit > 0.5)),
        'fired_by_side': {s: int(np.sum(fired[sides == s] > 0.5)) for s in ('A', 'B')},
        'hit_by_side': {s: int(np.sum(hit[sides == s] > 0.5)) for s in ('A', 'B')},
        'hard_blocked_fire_count_by_reason': dict(hard),
        'owner_mismatch_count': owner_mismatch,
        'normalizer_path': 'experiments/phase3v2/normalizers/phase3v2_c_local45_no_pressure.pt',
        'normalizer_sha256': sha256_file(Path('experiments/phase3v2/normalizers/phase3v2_c_local45_no_pressure.pt')),
        'segments': segment_summaries,
    }
    write_json(job / 'teacher_summary_v2.json', summary)
    write_json(root / 'teacher_dataset_v2_summary.json', summary)
    lines = [
        '# Phase 3v2_c Local45 Teacher Dataset', '',
        f"Status: {summary['status']}",
        f"sample_count: {summary['sample_count']}",
        f"side_counts: {summary['side_counts']}",
        f"shoot_intent_positive_rate: {summary['shoot_intent_positive_rate']}",
        f"fired_by_side: {summary['fired_by_side']}",
        f"hit_by_side: {summary['hit_by_side']}",
        f"failures: {summary['failures']}",
    ]
    (root / 'teacher_dataset_v2_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def main() -> None:
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', default=str(ROOT))
    ap.add_argument('--env-path', default='artifacts/unity/combat/Builds/BotArena.x86_64')
    ap.add_argument('--areas', type=int, default=8)
    ap.add_argument('--target-samples', type=int, default=400000)
    ap.add_argument('--scripted-steps', type=int, default=4096)
    ap.add_argument('--gvg-max-steps', type=int, default=45000)
    ap.add_argument('--base-port', type=int, default=64900)
    ap.add_argument('--seed', type=int, default=454500)
    ap.add_argument('--time-scale', type=float, default=8.0)
    ap.add_argument('--shoot-pos-deg', type=float, default=30.0)
    ap.add_argument('--shoot-neg-deg', type=float, default=60.0)
    ap.add_argument('--perturb-fraction', type=float, default=0.12)
    ap.add_argument('--perturb-min', type=int, default=10)
    ap.add_argument('--perturb-max', type=int, default=40)
    ap.add_argument('--deadlock-step-seconds', type=float, default=30.0)
    args = ap.parse_args()
    ROOT = Path(args.output_root)
    ROOT.mkdir(parents=True, exist_ok=True)
    targets = make_targets(args.target_samples)
    counts: dict[str, Counter] = {'A': Counter(), 'B': Counter()}
    rows: dict[str, list[Any]] = defaultdict(list)
    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    segments = [
        ('scripted_idle', 'GambitVsScripted', 'Idle', args.scripted_steps),
        ('scripted_face', 'GambitVsScripted', 'FaceOpponent', args.scripted_steps),
        ('scripted_strafe_face', 'GambitVsScripted', 'StrafeAndFace', args.scripted_steps),
        ('gvg_mirrored_oracle', 'GambitVsGambit', 'None', args.gvg_max_steps),
    ]
    write_json(ROOT / 'target_counts.json', targets)
    for idx, (name, game_mode, bot_mode, steps) in enumerate(segments):
        summaries.append(collect_segment(args, name, game_mode, bot_mode, steps, targets, counts, rows, args.base_port + idx))
        if targets_met(counts, targets):
            break
    if not targets_met(counts, targets):
        failures.append(f'target counts not met: counts={{"A": {dict(counts["A"])}, "B": {dict(counts["B"])} }} targets={targets}')
    summary = summarize_dataset(ROOT, rows, summaries, targets, failures)
    print(json.dumps({k: summary[k] for k in ('status', 'sample_count', 'side_counts', 'shoot_intent_positive_rate', 'fired_by_side', 'hit_by_side', 'failures')}, sort_keys=True))
    if summary['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
