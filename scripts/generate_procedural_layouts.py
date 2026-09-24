#!/usr/bin/env python3
"""Deterministic primitive-collider layout generator for Phase 5."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import deque
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments/phase5_map_general_headless_hunter/02_layouts"
SCHEMA = "phase5_procedural_layout_v001"
SUITE_SCHEMA = "phase5_procedural_layout_suite_v001"
RESOLUTION = 0.5
AGENT_RADIUS = 0.65
FAMILIES = (
    "open_scattered_cover",
    "parallel_corridors",
    "rooms_doorways",
    "u_traps_dead_ends",
    "pillars_alternating_cover",
    "narrow_wide_transitions",
)
TRAIN_FAMILIES = tuple(name for name in FAMILIES if name != "u_traps_dead_ends")
HELDOUT_FAMILIES = ("u_traps_dead_ends",)
SPLIT_COUNTS = {"train": 84, "validation": 18, "heldout": 18}
SEED_BASES = {"train": 540000, "validation": 550000, "heldout": 560000}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rounded(value: float) -> float:
    return round(float(value), 6)


def box(identifier: str, x: float, z: float, sx: float, sz: float, height: float, kind: str) -> dict[str, Any]:
    return {
        "id": identifier, "primitive": "box", "kind": kind,
        "center": [rounded(x), rounded(height / 2), rounded(z)],
        "size": [rounded(sx), rounded(height), rounded(sz)],
    }


def wall_with_gap(obstacles: list[dict[str, Any]], prefix: str, axis: str, coordinate: float,
                  half_extent: float, gap_center: float, gap_width: float,
                  thickness: float, height: float) -> None:
    low_end = gap_center - gap_width / 2
    high_start = gap_center + gap_width / 2
    for suffix, lo, hi in (("lo", -half_extent, low_end), ("hi", high_start, half_extent)):
        length = hi - lo
        if length <= 0.5:
            continue
        center = (lo + hi) / 2
        if axis == "z":
            obstacles.append(box(f"{prefix}_{suffix}", coordinate, center, thickness, length, height, "wall"))
        else:
            obstacles.append(box(f"{prefix}_{suffix}", center, coordinate, length, thickness, height, "wall"))


def perimeter(width: float, depth: float, thickness: float, height: float) -> list[dict[str, Any]]:
    hw, hd = width / 2, depth / 2
    return [
        box("perimeter_n", 0, hd + thickness / 2, width + thickness * 2, thickness, height, "boundary"),
        box("perimeter_s", 0, -hd - thickness / 2, width + thickness * 2, thickness, height, "boundary"),
        box("perimeter_e", hw + thickness / 2, 0, thickness, depth, height, "boundary"),
        box("perimeter_w", -hw - thickness / 2, 0, thickness, depth, height, "boundary"),
    ]


def family_geometry(family: str, rng: random.Random, width: float, depth: float,
                    wall_height: float, wall_thickness: float, obstacle_width: float,
                    doorway_width: float) -> list[dict[str, Any]]:
    hw, hd = width / 2, depth / 2
    obstacles = perimeter(width, depth, wall_thickness, wall_height)
    if family == "open_scattered_cover":
        for i in range(rng.randint(6, 11)):
            sx = rng.uniform(1.2, obstacle_width * 1.8)
            sz = rng.uniform(1.2, obstacle_width * 1.8)
            obstacles.append(box(f"cover_{i}", rng.uniform(-hw + 3, hw - 3), rng.uniform(-hd + 3, hd - 3),
                                 sx, sz, rng.uniform(1.8, wall_height), "cover"))
    elif family == "parallel_corridors":
        count = rng.randint(2, 4)
        for i in range(count):
            x = -hw * 0.65 + (i + 1) * (hw * 1.3 / (count + 1))
            gap = rng.uniform(-hd * 0.55, hd * 0.55)
            wall_with_gap(obstacles, f"corridor_{i}", "z", x, hd - 1.5, gap, doorway_width,
                          wall_thickness, wall_height)
    elif family == "rooms_doorways":
        divider_x = rng.uniform(-2, 2)
        wall_with_gap(obstacles, "room_vertical", "z", divider_x, hd - 1.5,
                      rng.uniform(-hd * 0.45, hd * 0.45), doorway_width, wall_thickness, wall_height)
        # Offset room partitions stop short of the divider, so they form
        # door-accessible rooms without intersecting another collider.
        left_start, left_end = -hw + 1.5, divider_x - wall_thickness / 2 - 0.75
        right_start, right_end = divider_x + wall_thickness / 2 + 0.75, hw - 1.5
        obstacles.append(box("room_partition_left", (left_start + left_end) / 2, -hd * 0.35,
                             left_end - left_start, wall_thickness, wall_height, "wall"))
        obstacles.append(box("room_partition_right", (right_start + right_end) / 2, hd * 0.35,
                             right_end - right_start, wall_thickness, wall_height, "wall"))
    elif family == "u_traps_dead_ends":
        for i, sign in enumerate((-1, 1)):
            cx = sign * hw * 0.42
            cz = rng.uniform(-hd * 0.35, hd * 0.35)
            arm = rng.uniform(5.0, min(9.0, depth * 0.28))
            mouth = rng.uniform(3.5, 6.0)
            orientation = 1 if i == 0 else -1
            back_z = cz - orientation * arm / 2
            obstacles.append(box(f"u_{i}_back", cx, back_z, mouth + wall_thickness * 2, wall_thickness, wall_height, "wall"))
            obstacles.append(box(f"u_{i}_left", cx - (mouth + wall_thickness) / 2, cz,
                                 wall_thickness, arm - wall_thickness, wall_height, "wall"))
            obstacles.append(box(f"u_{i}_right", cx + (mouth + wall_thickness) / 2, cz,
                                 wall_thickness, arm - wall_thickness, wall_height, "wall"))
    elif family == "pillars_alternating_cover":
        spacing_x = width / 6
        spacing_z = depth / 5
        idx = 0
        for row in range(-2, 3):
            for col in range(-2, 3):
                if (row + col) % 2 == 0:
                    size = rng.uniform(1.1, obstacle_width)
                    obstacles.append(box(f"pillar_{idx}", col * spacing_x, row * spacing_z,
                                         size, size, rng.uniform(2.0, wall_height), "pillar"))
                    idx += 1
    elif family == "narrow_wide_transitions":
        narrow = rng.uniform(3.2, max(3.3, doorway_width + 0.8))
        wide = rng.uniform(narrow + 3.0, min(width * 0.55, narrow + 9.0))
        transition_z = rng.uniform(-2.0, 2.0)
        negative_length = transition_z + hd - 1.5
        positive_length = hd - 1.5 - transition_z
        for side in (-1, 1):
            obstacles.append(box(f"neck_{side}", side * narrow / 2, (-hd + 1.5 + transition_z) / 2,
                                 wall_thickness, negative_length, wall_height, "wall"))
            obstacles.append(box(f"wide_{side}", side * wide / 2, (hd - 1.5 + transition_z) / 2,
                                 wall_thickness, positive_length, wall_height, "wall"))
        for i in range(4):
            obstacles.append(box(f"transition_cover_{i}", rng.choice([-1, 1]) * wide * 0.32,
                                 rng.uniform(transition_z + 2, hd - 3), obstacle_width, obstacle_width,
                                 rng.uniform(1.8, wall_height), "cover"))
    else:
        raise ValueError(f"unknown family {family}")
    return obstacles


def rectangle(obstacle: dict[str, Any], inflate: float = 0.0) -> tuple[float, float, float, float]:
    x, _, z = obstacle["center"]
    sx, _, sz = obstacle["size"]
    return x - sx / 2 - inflate, x + sx / 2 + inflate, z - sz / 2 - inflate, z + sz / 2 + inflate


def obstacle_pairs_overlap(obstacles: list[dict[str, Any]], epsilon: float = 1e-6) -> bool:
    for index, first in enumerate(obstacles):
        ax0, ax1, az0, az1 = rectangle(first)
        for second in obstacles[index + 1:]:
            bx0, bx1, bz0, bz1 = rectangle(second)
            if min(ax1, bx1) - max(ax0, bx0) > epsilon and min(az1, bz1) - max(az0, bz0) > epsilon:
                return True
    return False


def inside_any(x: float, z: float, obstacles: list[dict[str, Any]], inflate: float) -> bool:
    for obstacle in obstacles:
        if obstacle["kind"] == "boundary":
            continue
        x0, x1, z0, z1 = rectangle(obstacle, inflate)
        if x0 <= x <= x1 and z0 <= z <= z1:
            return True
    return False


def segment_hits_rect(a: tuple[float, float], b: tuple[float, float], rect: tuple[float, float, float, float]) -> bool:
    # Liang-Barsky clipping in x/z.
    x0, z0 = a; x1, z1 = b
    dx, dz = x1 - x0, z1 - z0
    p = (-dx, dx, -dz, dz)
    q = (x0 - rect[0], rect[1] - x0, z0 - rect[2], rect[3] - z0)
    low, high = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                low = max(low, t)
            else:
                high = min(high, t)
            if low > high:
                return False
    return True


def line_of_sight(a: tuple[float, float], b: tuple[float, float], obstacles: list[dict[str, Any]]) -> bool:
    for obstacle in obstacles:
        if obstacle["kind"] == "boundary" or obstacle["size"][1] < 1.7:
            continue
        if segment_hits_rect(a, b, rectangle(obstacle, 0.0)):
            return False
    return True


def occupancy(width: float, depth: float, obstacles: list[dict[str, Any]]) -> tuple[list[float], list[float], list[list[bool]]]:
    hw, hd = width / 2, depth / 2
    xs = [rounded(-hw + AGENT_RADIUS + i * RESOLUTION)
          for i in range(int((width - 2 * AGENT_RADIUS) / RESOLUTION) + 1)]
    zs = [rounded(-hd + AGENT_RADIUS + i * RESOLUTION)
          for i in range(int((depth - 2 * AGENT_RADIUS) / RESOLUTION) + 1)]
    free = [[not inside_any(x, z, obstacles, AGENT_RADIUS) for z in zs] for x in xs]
    return xs, zs, free


def connected_component(free: list[list[bool]], start: tuple[int, int]) -> set[tuple[int, int]]:
    nx, nz = len(free), len(free[0])
    seen = {start}; queue = deque([start])
    while queue:
        x, z = queue.popleft()
        for nxt in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
            ix, iz = nxt
            if 0 <= ix < nx and 0 <= iz < nz and free[ix][iz] and nxt not in seen:
                seen.add(nxt); queue.append(nxt)
    return seen


def shortest_path_steps(free: list[list[bool]], start: tuple[int, int], goal: tuple[int, int]) -> int | None:
    queue = deque([(start, 0)]); seen = {start}
    nx, nz = len(free), len(free[0])
    while queue:
        node, distance = queue.popleft()
        if node == goal:
            return distance
        x, z = node
        for nxt in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
            ix, iz = nxt
            if 0 <= ix < nx and 0 <= iz < nz and free[ix][iz] and nxt not in seen:
                seen.add(nxt); queue.append((nxt, distance + 1))
    return None


def choose_spawns(rng: random.Random, xs: list[float], zs: list[float], free: list[list[bool]],
                  obstacles: list[dict[str, Any]], requested_los: bool,
                  minimum_distance: float, maximum_distance: float) -> tuple[dict[str, Any], dict[str, Any], int] | None:
    cells = [(i, j) for i in range(len(xs)) for j in range(len(zs)) if free[i][j]]
    if len(cells) < 2:
        return None
    for _ in range(30000):
        a = cells[rng.randrange(len(cells))]
        b = cells[rng.randrange(len(cells))]
        if a == b:
            continue
        pa, pb = (xs[a[0]], zs[a[1]]), (xs[b[0]], zs[b[1]])
        distance = math.dist(pa, pb)
        if not (minimum_distance <= distance <= maximum_distance):
            continue
        if line_of_sight(pa, pb, obstacles) != requested_los:
            continue
        steps = shortest_path_steps(free, a, b)
        if steps is None:
            continue
        yaw_a = math.degrees(math.atan2(pb[0] - pa[0], pb[1] - pa[1]))
        yaw_b = math.degrees(math.atan2(pa[0] - pb[0], pa[1] - pb[1]))
        return (
            {"position": [pa[0], 1.0, pa[1]], "yaw_degrees": rounded(yaw_a)},
            {"position": [pb[0], 1.0, pb[1]], "yaw_degrees": rounded(yaw_b)},
            steps,
        )
    return None


def validate_layout(layout: dict[str, Any]) -> dict[str, Any]:
    width, depth = layout["arena_size"]
    obstacles = layout["obstacles"]
    xs, zs, free = occupancy(width, depth, obstacles)
    free_cells = [(i, j) for i in range(len(xs)) for j in range(len(zs)) if free[i][j]]
    if not free_cells:
        return {"status": "FAIL", "failure": "no_free_cells"}
    component = connected_component(free, free_cells[0])
    def cell_for(spawn: dict[str, Any]) -> tuple[int, int]:
        x, _, z = spawn["position"]
        return min(range(len(xs)), key=lambda i: abs(xs[i] - x)), min(range(len(zs)), key=lambda j: abs(zs[j] - z))
    a_cell, b_cell = cell_for(layout["spawn_a"]), cell_for(layout["spawn_b"])
    a_valid = free[a_cell[0]][a_cell[1]]
    b_valid = free[b_cell[0]][b_cell[1]]
    route_steps = shortest_path_steps(free, a_cell, b_cell) if a_valid and b_valid else None
    pa = (layout["spawn_a"]["position"][0], layout["spawn_a"]["position"][2])
    pb = (layout["spawn_b"]["position"][0], layout["spawn_b"]["position"][2])
    actual_los = line_of_sight(pa, pb, obstacles)
    distance = math.dist(pa, pb)
    checks = {
        "spawn_a_valid_floor": a_valid,
        "spawn_b_valid_floor": b_valid,
        "collision_free_route_exists": route_steps is not None,
        "no_unreachable_sealed_pockets": len(component) == len(free_cells),
        "no_immediate_overlap": distance >= 3.0,
        "no_obstacle_overlap": not obstacle_pairs_overlap(obstacles),
        "requested_los_satisfied": actual_los == layout["requested_initial_los"],
        "primitive_colliders_only": all(item.get("primitive") == "box" for item in obstacles),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "free_cells": len(free_cells), "reachable_free_cells": len(component),
        "route_length_m": rounded((route_steps or 0) * RESOLUTION),
        "spawn_distance_m": rounded(distance), "actual_initial_los": actual_los,
        "occupancy_resolution_m": RESOLUTION,
        "occupancy_graph_sha256": canonical_hash({"xs": xs, "zs": zs, "free": free}),
    }


def generate_layout(seed: int, family: str, split: str) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(family)
    for attempt in range(100):
        rng = random.Random(seed * 1009 + FAMILIES.index(family) * 9176 + attempt * 104729)
        width = rounded(rng.uniform(28.0, 44.0))
        depth = rounded(rng.uniform(28.0, 44.0))
        wall_height = rounded(rng.uniform(2.2, 4.8))
        wall_thickness = rounded(rng.uniform(0.35, 0.8))
        obstacle_width = rounded(rng.uniform(1.2, 3.2))
        doorway_width = rounded(rng.uniform(2.2, 4.5))
        obstacles = family_geometry(family, rng, width, depth, wall_height, wall_thickness,
                                    obstacle_width, doorway_width)
        if obstacle_pairs_overlap(obstacles):
            continue
        xs, zs, free = occupancy(width, depth, obstacles)
        free_cells = [(i, j) for i in range(len(xs)) for j in range(len(zs)) if free[i][j]]
        if not free_cells or len(connected_component(free, free_cells[0])) != len(free_cells):
            continue
        requested_los = bool(rng.getrandbits(1))
        min_distance = rounded(rng.uniform(5.0, 10.0))
        max_distance = rounded(rng.uniform(max(14.0, min_distance + 6), min(30.0, math.hypot(width, depth) * 0.7)))
        spawns = choose_spawns(rng, xs, zs, free, obstacles, requested_los, min_distance, max_distance)
        if spawns is None:
            continue
        spawn_a, spawn_b, _ = spawns
        layout = {
            "schema_version": SCHEMA, "seed": seed, "split": split, "family": family,
            "generation_attempt": attempt, "arena_size": [width, depth],
            "floor": {"primitive": "box", "center": [0.0, -0.25, 0.0], "size": [width, 0.5, depth]},
            "obstacles": obstacles, "spawn_a": spawn_a, "spawn_b": spawn_b,
            "requested_initial_los": requested_los,
            "parameters": {
                "wall_height": wall_height, "wall_thickness": wall_thickness,
                "obstacle_width": obstacle_width, "doorway_width": doorway_width,
                "spawn_distance_range": [min_distance, max_distance],
                "spawn_bearing_degrees": rounded(spawn_a["yaw_degrees"]),
                "target_motion": rng.choice(["stationary", "strafe", "alternating_cover", "patrol"]),
                "floor_dynamic_friction": rounded(rng.uniform(0.2, 0.8)),
                "floor_static_friction": rounded(rng.uniform(0.25, 0.9)),
                "movement_speed_scale": rounded(rng.uniform(0.85, 1.15)),
            },
            "privileged_connectivity_only": {
                "kind": "occupancy_graph", "actor_visible": False,
                "may_drive": ["layout_validation", "privileged_teacher"],
            },
        }
        validation = validate_layout(layout)
        if validation["status"] != "PASS":
            continue
        layout["validation"] = validation
        payload = dict(layout)
        layout["layout_sha256"] = canonical_hash(payload)
        return layout
    raise RuntimeError(f"unable to generate valid layout seed={seed} family={family}")


def assignments(split: str) -> list[tuple[int, str]]:
    count = SPLIT_COUNTS[split]
    base = SEED_BASES[split]
    families = HELDOUT_FAMILIES if split == "heldout" else TRAIN_FAMILIES
    return [(base + index, families[index % len(families)]) for index in range(count)]


def build_manifest(split: str) -> dict[str, Any]:
    layouts = [generate_layout(seed, family, split) for seed, family in assignments(split)]
    manifest = {
        "schema_version": SUITE_SCHEMA, "split": split, "immutable": True,
        "seed_count": len(layouts), "seeds": [item["seed"] for item in layouts],
        "families": sorted({item["family"] for item in layouts}),
        "heldout_entire_families": list(HELDOUT_FAMILIES),
        "split_policy": {"train_percent": 70, "validation_percent": 15, "heldout_percent": 15,
                         "total_seed_count": sum(SPLIT_COUNTS.values())},
        "actor_access_to_occupancy_or_navmesh": False,
        "layouts": layouts,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return manifest


def write_suite(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    names = {"train": "TRAIN_LAYOUTS.json", "validation": "VALIDATION_LAYOUTS.json", "heldout": "HELDOUT_LAYOUTS.json"}
    manifests = {}
    for split, name in names.items():
        manifest = build_manifest(split)
        path = output / name
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifests[split] = manifest
        print(split, len(manifest["layouts"]), sha256_file(path))
    hashes = []
    for name in names.values():
        path = output / name
        hashes.append(f"{sha256_file(path)}  {name}")
    (output / "layout_hashes.sha256").write_text("\n".join(hashes) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    write_suite(Path(args.output_dir))


if __name__ == "__main__":
    main()
