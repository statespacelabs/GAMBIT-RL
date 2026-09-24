#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

AREA_SPACING = 500.0
SCHEMA_VERSION = "phase3v2_c_local45"
FORMERLY_SUSPICIOUS = tuple(list(range(0, 9)) + [10] + list(range(12, 20)))


def collect_records(dec: Any, ter: Any) -> list[tuple[int, np.ndarray, float, bool]]:
    out: list[tuple[int, np.ndarray, float, bool]] = []
    if len(dec):
        obs_batch = np.asarray(dec.obs[0], dtype=np.float32)
        for row, aid_raw in enumerate(dec.agent_id):
            out.append((int(aid_raw), obs_batch[row], float(dec.reward[row]), False))
    if len(ter):
        obs_batch = np.asarray(ter.obs[0], dtype=np.float32)
        for row, aid_raw in enumerate(ter.agent_id):
            out.append((int(aid_raw), obs_batch[row], float(ter.reward[row]), True))
    return out


def build_slot_map_from_agent_ids(seed_obs: dict[int, np.ndarray], num_areas: int) -> dict[int, tuple[str, int]]:
    slot: dict[int, tuple[str, int]] = {}
    ordered = sorted(seed_obs)
    if len(ordered) >= num_areas * 2:
        for area in range(num_areas):
            pair = ordered[area * 2 : area * 2 + 2]
            if len(pair) == 2:
                slot[pair[0]] = ("A", area)
                slot[pair[1]] = ("B", area)
    for idx, aid in enumerate(ordered):
        area = min(num_areas - 1, idx // 2)
        side = "A" if idx % 2 == 0 else "B"
        slot.setdefault(aid, (side, area))
    return slot


def discover_slot_map(env: Any, behavior: str, num_areas: int, max_steps: int = 100) -> dict[int, tuple[str, int]]:
    seed_obs: dict[int, np.ndarray] = {}
    for _ in range(max_steps):
        dec, ter = env.get_steps(behavior)
        for aid, obs, _, _ in collect_records(dec, ter):
            seed_obs[aid] = obs
        if len(seed_obs) >= num_areas * 2:
            break
        env.step()
    return build_slot_map_from_agent_ids(seed_obs, num_areas)


def oracle_action(obs: np.ndarray, shoot_threshold: float = 35.0, force_shoot: bool = False) -> np.ndarray:
    action = np.zeros(8, dtype=np.float32)
    action[2] = np.clip(float(obs[20]) / 45.0, -1.0, 1.0)
    action[3] = np.clip(-float(obs[21]) / 45.0, -1.0, 1.0)
    action[4] = 1.0 if force_shoot or float(max(0.0, obs[22])) <= shoot_threshold else 0.0
    return action


def grouped_current(records: list[tuple[int, np.ndarray, float, bool]], slot: dict[int, tuple[str, int]]) -> dict[int, dict[str, tuple[int, np.ndarray, float, bool]]]:
    current: dict[int, dict[str, tuple[int, np.ndarray, float, bool]]] = defaultdict(dict)
    for aid, obs, rew, done in records:
        side, area = slot.get(aid, ("A" if aid % 2 == 0 else "B", aid // 2))
        current[area][side] = (aid, obs, rew, done)
    return current
