from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.navigation_ppo import PEEKER_VARIANTS
from scripts.navigation_policy import ACTION_DIM, ACTOR_DIM, LOS_INDEX
from scripts.local_geometry_peeker import LocalGeometryPeeker, detect_local_cover


def cover_observation(*, visible: bool = False, both_sides: bool = True) -> np.ndarray:
    actor = np.zeros(ACTOR_DIM, dtype=np.float32)
    actor[LOS_INDEX] = float(visible)
    if visible:
        actor[193] = 1.0
        actor[195] = 1.0
    else:
        actor[202] = 1.0
        actor[205] = 1.0
        actor[206] = 0.0
    actor[30:94:2] = 0.50
    actor[30] = 0.10
    actor[31] = 1.0
    for ray in (7, 8, 9):
        actor[30 + ray * 2] = 0.60
    for ray in (23, 24, 25):
        actor[30 + ray * 2] = 0.60 if both_sides else 0.01
    actor[187] = 0.60
    actor[191] = 0.60 if both_sides else 0.01
    return actor


def controller(seed: int = 1) -> LocalGeometryPeeker:
    return LocalGeometryPeeker(
        seed=seed, dwell_min=1, dwell_max=1, risk=0.5, repeek_penalty=0.02,
    )


def test_local_cover_contract_uses_only_egocentric_actor_telemetry() -> None:
    actor = cover_observation()
    state = detect_local_cover(actor)
    assert state.behind_cover
    assert state.lateral_space_left and state.lateral_space_right
    shifted_or_rotated_world_with_same_egocentric_telemetry = actor.copy()
    assert detect_local_cover(shifted_or_rotated_world_with_same_egocentric_telemetry) == state


def test_peek_lifecycle_preserves_frozen_aim_and_shoot() -> None:
    peeker = controller()
    navigation = np.zeros((1, ACTION_DIM), dtype=np.float32)
    combat = np.array([[0.25, -0.1, 0.7, -0.2, 1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    rewards = []
    applied, _, reward, _ = peeker.apply(
        cover_observation()[None], navigation, combat, [7], np.array([1.0]),
    )
    rewards.extend(reward)
    assert applied[0, 0] != 0
    for health in (1.0, 0.8):
        applied, _, reward, _ = peeker.apply(
            cover_observation(visible=True)[None], navigation, combat, [7],
            np.array([health], dtype=np.float32),
        )
        rewards.extend(reward)
        np.testing.assert_array_equal(applied[0, 2:8], combat[0, 2:8])
    _, _, reward, _ = peeker.apply(
        cover_observation()[None], navigation, combat, [7], np.array([0.8]),
    )
    rewards.extend(reward)
    summary = peeker.summary()
    assert summary["completed_peeks"] == 1
    assert summary["useful_peeks"] == 1
    assert summary["damage_positive_peeks"] == 1
    assert sum(abs(float(value)) for value in rewards) <= 0.20 + 1e-7


def test_privileged_damage_label_cannot_change_tactical_actions() -> None:
    a = controller(11)
    b = controller(11)
    navigation = np.zeros((1, ACTION_DIM), dtype=np.float32)
    combat = np.array([[0.0, 0.0, 0.4, 0.1, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    observations = [cover_observation(), cover_observation(visible=True),
                    cover_observation(visible=True), cover_observation()]
    action_a, action_b = [], []
    for step, actor in enumerate(observations):
        action_a.append(a.apply(actor[None], navigation, combat, [3], np.array([1.0]))[0])
        changing_health = np.array([1.0 - 0.2 * max(0, step - 1)], dtype=np.float32)
        action_b.append(b.apply(actor[None], navigation, combat, [3], changing_health)[0])
    np.testing.assert_array_equal(np.concatenate(action_a), np.concatenate(action_b))


def test_eight_branches_cover_requested_axes() -> None:
    assert len(PEEKER_VARIANTS) == 8
    assert len({item.branch.seed for item in PEEKER_VARIANTS}) == 8
    assert len({(item.dwell_min_steps, item.dwell_max_steps) for item in PEEKER_VARIANTS}) >= 4
    assert len({item.risk for item in PEEKER_VARIANTS}) >= 4
    assert len({item.repeek_penalty for item in PEEKER_VARIANTS}) >= 3
    assert any(item.branch.gru_hidden == 256 for item in PEEKER_VARIANTS)


def test_no_per_map_cover_anchor_or_peek_point_assets() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = ("cover_anchor", "peek_point")
    for folder in (root / "BotArenaPhase5", root / "scripts"):
        for path in folder.rglob("*"):
            if path.is_file() and path.name != Path(__file__).name:
                assert not any(token in path.name.lower() for token in forbidden)
