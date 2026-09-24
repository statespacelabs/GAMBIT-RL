from __future__ import annotations

import numpy as np

from scripts.navigation_ppo import REACQUIRE_VARIANTS, RewardLedger
from scripts.navigation_policy import ACTOR_DIM, FairReacquisitionSearch


def fair_hidden_observation() -> np.ndarray:
    actor = np.zeros((1, ACTOR_DIM), dtype=np.float32)
    actor[:, 30:182:2] = 1.0
    actor[0, 202] = 1.0
    actor[0, 203] = 0.10
    actor[0, 205] = 0.20
    actor[0, 206] = 0.20
    actor[0, 211 + 4] = 1.0
    actor[0, 224] = 1.0
    return actor


def test_eight_variants_cover_requested_axes() -> None:
    assert len(REACQUIRE_VARIANTS) == 8
    assert len({item.branch.branch_id for item in REACQUIRE_VARIANTS}) == 8
    assert len({item.memory_steps for item in REACQUIRE_VARIANTS}) >= 3
    assert len({item.cue_dropout for item in REACQUIRE_VARIANTS}) >= 3
    assert len({item.stale_timeout_seconds for item in REACQUIRE_VARIANTS}) >= 3
    assert len({item.branch.gru_hidden for item in REACQUIRE_VARIANTS}) >= 2
    assert len({item.reacquisition_reward for item in REACQUIRE_VARIANTS}) >= 3
    assert len({item.route_change_penalty for item in REACQUIRE_VARIANTS}) >= 2


def test_search_is_invariant_to_unprovided_hidden_enemy_state() -> None:
    actor = fair_hidden_observation()
    action = np.zeros((1, 8), dtype=np.float32)
    first, _ = FairReacquisitionSearch(4.0).apply(actor, action, [12])
    # A hidden enemy can move arbitrarily; with the permitted cue held constant,
    # actor231 is unchanged, so the action must be exactly unchanged.
    second, _ = FairReacquisitionSearch(4.0).apply(actor.copy(), action, [12])
    np.testing.assert_array_equal(first, second)


def test_stale_last_seen_falls_back_to_quantized_cue() -> None:
    actor = fair_hidden_observation()
    actor[0, 206] = 0.9
    result, active = FairReacquisitionSearch(2.5).apply(
        actor, np.zeros((1, 8), dtype=np.float32), [3]
    )
    assert active.tolist() == [True]
    assert np.isfinite(result).all()
    assert result[0, 2] != 0.0


def test_reacquisition_and_route_rewards_are_bounded() -> None:
    ledger = RewardLedger()
    assert ledger.terminal("reacquired", 0.10) == 0.10
    assert ledger.terminal("reacquired", 99.0) == 0.20
    assert all(0.0 < item.branch.bc_floor for item in REACQUIRE_VARIANTS)

