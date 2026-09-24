#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.combat_observations import SCHEMA_VERSION, collect_records, discover_slot_map, oracle_action
from scripts.combat_runtime import (
    HARD_BLOCK_REASONS,
    OBS_DIM,
    RuntimeConfig,
    collect_all_weapon_events,
    configure_unity_env,
    launch_unity_env,
    run_text,
    sha256_file,
    summarize_weapon_events,
    validate_specs,
    write_json,
)
from gambit.rl.online_rl.selfplay_runner import make_action_tuple, policy_action_to_buffers


NORMALIZER_PATH = Path('experiments/phase3v2/normalizers/phase3v2_c_local45_no_pressure.pt')
BUILD_DIR = Path('experiments/phase3v2/normalizers/phase3v2_c_local45_no_pressure_build')


class Welford:
    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.min = np.full(dim, np.inf, dtype=np.float64)
        self.max = np.full(dim, -np.inf, dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        arr = np.asarray(batch, dtype=np.float64).reshape(-1, self.mean.size)
        if not np.isfinite(arr).all():
            raise ValueError('normalizer collection saw NaN/Inf')
        for x in arr:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.m2 += delta * delta2
            self.min = np.minimum(self.min, x)
            self.max = np.maximum(self.max, x)

    @property
    def variance(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(self.mean)
        return self.m2 / (self.n - 1)


def scripted_action(obs: np.ndarray, step: int, shoot_threshold: float) -> np.ndarray:
    action = oracle_action(obs, shoot_threshold=shoot_threshold, force_shoot=(step % 6 in (0, 1)))
    phase = step * 0.037
    action[0] = np.clip(0.25 * math.sin(phase), -1.0, 1.0)
    action[1] = np.clip(0.20 * math.cos(phase * 0.7), -1.0, 1.0)
    return action


def choose_balanced_indices(sides: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    by_side: dict[str, list[int]] = {'A': [], 'B': []}
    for i, side in enumerate(sides):
        if side in by_side:
            by_side[side].append(i)
    target = min(len(by_side['A']), len(by_side['B']))
    if target <= 0:
        return np.zeros((0,), dtype=np.int64), {'A': len(by_side['A']), 'B': len(by_side['B'])}
    chosen: list[int] = []
    for side in ('A', 'B'):
        arr = np.asarray(by_side[side], dtype=np.int64)
        if len(arr) > target:
            take = np.linspace(0, len(arr) - 1, target, dtype=np.int64)
            arr = arr[take]
        chosen.extend(arr.tolist())
    chosen_arr = np.asarray(sorted(chosen), dtype=np.int64)
    return chosen_arr, {'A': target, 'B': target}


def collect_segment(args: argparse.Namespace, name: str, game_mode: str, bot_mode: str, steps: int, port: int) -> dict[str, Any]:
    out = BUILD_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = RuntimeConfig(
        env_path=args.env_path,
        output_dir=str(out),
        base_port=port,
        seed=args.seed + port,
        time_scale=args.time_scale,
        device='cpu',
        game_mode=game_mode,
        player_b_bot_mode=bot_mode,
        num_areas=args.areas,
    )
    weapon_log = configure_unity_env(out, cfg)
    rows: list[np.ndarray] = []
    sides: list[str] = []
    areas: list[int] = []
    obs44_values: set[int] = set()
    slot: dict[int, tuple[str, int]] = {}
    env = None
    start = time.time()
    try:
        env = launch_unity_env(cfg)
        behavior, action_spec, _spec = validate_specs(env)
        if game_mode == 'GambitVsGambit':
            slot = discover_slot_map(env, behavior, args.areas)
        for step in range(steps):
            dec, ter = env.get_steps(behavior)
            records = collect_records(dec, ter)
            for aid, obs, _rew, _done in records:
                if obs.shape != (OBS_DIM,):
                    raise RuntimeError(f'{name}: obs shape {obs.shape}')
                if not np.isfinite(obs).all():
                    raise RuntimeError(f'{name}: non-finite observation')
                if game_mode == 'GambitVsGambit':
                    side, area = slot.get(aid, ('A' if aid % 2 == 0 else 'B', aid // 2))
                else:
                    side, area = 'A', min(args.areas - 1, len(rows) % max(1, args.areas))
                rows.append(obs.copy())
                sides.append(side)
                areas.append(int(area))
                obs44_values.add(int(float(obs[44]) > 0.5))
            if len(dec):
                obs_batch = np.asarray(dec.obs[0], dtype=np.float32)
                for row, aid_raw in enumerate(dec.agent_id):
                    action = scripted_action(obs_batch[row], step, args.shoot_threshold)
                    continuous, discrete = policy_action_to_buffers(action, action_spec)
                    env.set_action_for_agent(behavior, int(aid_raw), make_action_tuple(continuous, discrete))
            t0 = time.monotonic()
            env.step()
            if time.monotonic() - t0 > args.deadlock_step_seconds:
                raise RuntimeError(f'{name}: Unity step exceeded {args.deadlock_step_seconds}s')
    finally:
        if env is not None:
            env.close()

    events = collect_all_weapon_events(weapon_log)
    weapon_summary = summarize_weapon_events(events)
    segment = {
        'name': name,
        'game_mode': game_mode,
        'bot_mode': bot_mode,
        'steps': steps,
        'sample_count': len(rows),
        'side_counts': dict(Counter(sides)),
        'areas_seen': sorted(set(areas)),
        'obs44_values': sorted(obs44_values),
        'weapon_summary': weapon_summary,
        'slot_map': {str(k): {'side': v[0], 'area': v[1]} for k, v in sorted(slot.items())},
        'elapsed_sec': time.time() - start,
    }
    write_json(out / 'segment_summary.json', segment)
    return {'summary': segment, 'rows': rows, 'sides': sides, 'areas': areas}


def main() -> None:
    global BUILD_DIR, NORMALIZER_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument('--env-path', default='artifacts/unity/combat/Builds/BotArena.x86_64')
    ap.add_argument('--output-dir', default=str(BUILD_DIR))
    ap.add_argument('--normalizer-path', default=str(NORMALIZER_PATH))
    ap.add_argument('--areas', type=int, default=8)
    ap.add_argument('--scripted-steps', type=int, default=1024)
    ap.add_argument('--gvg-steps', type=int, default=7000)
    ap.add_argument('--target-samples', type=int, default=100000)
    ap.add_argument('--base-port', type=int, default=64700)
    ap.add_argument('--seed', type=int, default=45045)
    ap.add_argument('--time-scale', type=float, default=8.0)
    ap.add_argument('--shoot-threshold', type=float, default=35.0)
    ap.add_argument('--deadlock-step-seconds', type=float, default=30.0)
    args = ap.parse_args()

    BUILD_DIR = Path(args.output_dir)
    NORMALIZER_PATH = Path(args.normalizer_path)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    NORMALIZER_PATH.parent.mkdir(parents=True, exist_ok=True)

    segments = [
        ('scripted_idle', 'GambitVsScripted', 'Idle', args.scripted_steps),
        ('scripted_face', 'GambitVsScripted', 'FaceOpponent', args.scripted_steps),
        ('scripted_strafe_face', 'GambitVsScripted', 'StrafeAndFace', args.scripted_steps),
        ('gvg_oracle', 'GambitVsGambit', 'None', args.gvg_steps),
    ]
    all_rows: list[np.ndarray] = []
    all_sides: list[str] = []
    all_areas: list[int] = []
    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    hard_blocks = Counter()
    owner_mismatch = 0

    for idx, (name, game_mode, bot_mode, steps) in enumerate(segments):
        result = collect_segment(args, name, game_mode, bot_mode, steps, args.base_port + idx)
        summaries.append(result['summary'])
        all_rows.extend(result['rows'])
        all_sides.extend(result['sides'])
        all_areas.extend(result['areas'])
        ws = result['summary'].get('weapon_summary', {}) or {}
        for reason, count in (ws.get('hard_blocked_fire_count_by_reason', {}) or {}).items():
            hard_blocks[str(reason)] += int(count)
        owner_mismatch += int(ws.get('owner_mismatch_count', 0) or 0)

    raw_count = len(all_rows)
    chosen_idx, _balanced_counts = choose_balanced_indices(all_sides)
    if raw_count == 0 or chosen_idx.size == 0:
        balanced = np.zeros((0, OBS_DIM), dtype=np.float32)
        balanced_sides: list[str] = []
        balanced_areas: list[int] = []
    else:
        raw = np.stack(all_rows).astype(np.float32)
        balanced = raw[chosen_idx]
        balanced_sides = [all_sides[int(i)] for i in chosen_idx]
        balanced_areas = [all_areas[int(i)] for i in chosen_idx]

    stats = Welford(OBS_DIM)
    if balanced.size:
        stats.update(balanced)
    variance = stats.variance
    std_floor = 1e-6
    std = np.sqrt(np.maximum(variance, 0.0))
    effective_std = np.maximum(std, std_floor)
    obs44_values = sorted({int(float(x) > 0.5) for x in balanced[:, 44]}) if balanced.size else []
    side_counts = dict(Counter(balanced_sides))
    area_counts = {str(k): int(v) for k, v in sorted(Counter(balanced_areas).items())}

    if int(stats.n) < args.target_samples:
        failures.append(f'sample_count {stats.n} < target {args.target_samples}')
    if not balanced.size or not np.isfinite(balanced).all():
        failures.append('balanced observations are empty or non-finite')
    if side_counts.get('A', 0) != side_counts.get('B', 0):
        failures.append(f'A/B side imbalance after balancing: {side_counts}')
    if sorted(obs44_values) != [0, 1]:
        failures.append(f'obs[44] did not observe both 0 and 1: {obs44_values}')
    if not np.isfinite(stats.mean).all() or not np.isfinite(variance).all() or not np.isfinite(effective_std).all():
        failures.append('normalizer statistics contain NaN/Inf')
    if np.any(effective_std <= 0):
        failures.append('normalizer effective std <= 0')
    if balanced.size and float(np.max(np.abs(balanced))) > 10000.0:
        failures.append(f'feature explosion max_abs={float(np.max(np.abs(balanced)))}')
    structural = {k: int(v) for k, v in hard_blocks.items() if k in HARD_BLOCK_REASONS and int(v) > 0}
    if structural:
        failures.append(f'hard blocked-fire counts nonzero: {structural}')
    if owner_mismatch:
        failures.append(f'owner mismatch count nonzero: {owner_mismatch}')

    repo_commit = run_text(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT)
    unity_sha = sha256_file(Path(args.env_path))
    collection_config = {
        'obs_schema_version': SCHEMA_VERSION,
        'segments': [{'name': n, 'game_mode': gm, 'bot_mode': bm, 'steps': st} for n, gm, bm, st in segments],
        'areas': args.areas,
        'time_scale': args.time_scale,
        'shoot_threshold': args.shoot_threshold,
        'target_samples': args.target_samples,
        'side_balancing': 'deterministic downsample to equal A/B counts after collection',
        'ray_correction_alpha': 0,
        'target_hurtbox_inflate': 0,
        'use_true_aim_for_reward': False,
        'policy_updates': 0,
    }
    payload = {
        'obs_schema_version': SCHEMA_VERSION,
        'obs_dim': OBS_DIM,
        'sample_count': int(stats.n),
        'mean': torch.tensor(stats.mean, dtype=torch.float32),
        'variance': torch.tensor(variance, dtype=torch.float32),
        'standard_deviation': torch.tensor(std, dtype=torch.float32),
        'effective_standard_deviation': torch.tensor(effective_std, dtype=torch.float32),
        'epsilon': 1e-8,
        'std_floor': std_floor,
        'clipping_policy': {'enabled': True, 'clip_value': 10.0},
        'collection_config': collection_config,
        'repository_commit': repo_commit,
        'unity_executable_sha256': unity_sha,
        'side_counts': side_counts,
        'area_counts': area_counts,
    }

    normalizer_sha = None
    load_back_smoke = False
    if not failures:
        tmp = NORMALIZER_PATH.with_suffix('.pt.tmp')
        torch.save(payload, tmp)
        os.replace(tmp, NORMALIZER_PATH)
        loaded = torch.load(NORMALIZER_PATH, map_location='cpu', weights_only=False)
        normalizer_sha = sha256_file(NORMALIZER_PATH)
        sample = torch.as_tensor(balanced[: min(len(balanced), 4096)], dtype=torch.float32)
        normed = torch.clamp(
            (sample - torch.as_tensor(loaded['mean'], dtype=torch.float32)) / torch.as_tensor(loaded['effective_standard_deviation'], dtype=torch.float32),
            -float(loaded['clipping_policy']['clip_value']),
            float(loaded['clipping_policy']['clip_value']),
        )
        load_back_smoke = bool(torch.isfinite(normed).all())
        if not load_back_smoke:
            failures.append('load-back normalization smoke produced non-finite values')
    else:
        tmp = NORMALIZER_PATH.with_suffix('.pt.tmp')
        if tmp.exists():
            tmp.unlink()

    raw_stats = {
        'raw_sample_count': raw_count,
        'balanced_sample_count': int(stats.n),
        'raw_side_counts': dict(Counter(all_sides)),
        'balanced_side_counts': side_counts,
        'balanced_area_counts': area_counts,
        'obs44_values': obs44_values,
        'mean': stats.mean.tolist(),
        'variance': variance.tolist(),
        'standard_deviation': std.tolist(),
        'min': stats.min.tolist() if stats.n else [],
        'max': stats.max.tolist() if stats.n else [],
    }
    summary = {
        'status': 'PASS' if not failures else 'FAIL',
        'failures': failures,
        'normalizer_path': str(NORMALIZER_PATH) if NORMALIZER_PATH.exists() else None,
        'normalizer_sha256': normalizer_sha,
        'obs_schema_version': SCHEMA_VERSION,
        'obs_dim': OBS_DIM,
        'raw_sample_count': raw_count,
        'sample_count': int(stats.n),
        'side_counts': side_counts,
        'area_counts': area_counts,
        'obs44_values': obs44_values,
        'hard_blocked_fire_count_by_reason': dict(hard_blocks),
        'owner_mismatch_count': owner_mismatch,
        'load_back_smoke_finite': load_back_smoke,
        'unity_executable_sha256': unity_sha,
        'repository_commit': repo_commit,
        'segments': summaries,
    }
    manifest = {
        'normalizer_sha256': normalizer_sha,
        'schema_version': SCHEMA_VERSION,
        'unity_executable_sha256': unity_sha,
        'repository_commit': repo_commit,
        'collection_config': collection_config,
        'summary': summary,
    }
    write_json(BUILD_DIR / 'raw_feature_statistics.json', raw_stats)
    write_json(BUILD_DIR / 'normalizer_manifest.json', manifest)
    write_json(BUILD_DIR / 'normalizer_build_summary.json', summary)
    lines = [
        '# Phase 3v2_c Local45 Normalizer Build',
        '',
        f"Status: {summary['status']}",
        f"normalizer_path: {summary['normalizer_path']}",
        f"normalizer_sha256: {summary['normalizer_sha256']}",
        f"sample_count: {summary['sample_count']}",
        f"side_counts: {summary['side_counts']}",
        f"obs44_values: {summary['obs44_values']}",
        f"load_back_smoke_finite: {summary['load_back_smoke_finite']}",
        f"failures: {summary['failures']}",
    ]
    (BUILD_DIR / 'normalizer_build_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('status', 'normalizer_path', 'normalizer_sha256', 'sample_count', 'side_counts', 'obs44_values', 'failures')}, sort_keys=True))
    if summary['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
