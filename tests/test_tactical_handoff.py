from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.tactical_handoff import FairTacticalHandoff


def actor(*, visible: bool, combat_mode: bool = True) -> np.ndarray:
    value = np.zeros((1, 231), dtype=np.float32)
    value[0, 193] = float(visible)
    value[0, 195] = 1.0
    value[0, 197] = 1.0
    value[0, 9] = 1.0
    value[0, 229] = float(combat_mode)
    return value


def test_always_visible_duel_keeps_frozen_combat_action() -> None:
    handoff = FairTacticalHandoff()
    combat = np.asarray([[0.3, -0.2, 0.1, 0.2, 0, 0, 1, 0]], dtype=np.float32)
    applied, intervened = handoff.apply(actor(visible=True), combat, [7])
    np.testing.assert_array_equal(applied, combat)
    assert not intervened.any()


def test_hidden_then_visible_holds_only_locomotion() -> None:
    handoff = FairTacticalHandoff()
    combat = np.asarray([[0.8, -0.7, 0.3, -0.4, 1, 0, 0, 1]], dtype=np.float32)
    handoff.apply(actor(visible=False), combat, [7])
    observation = actor(visible=True)
    applied, intervened = handoff.apply(observation, combat, [7])
    assert intervened.tolist() == [True]
    assert applied[0, 0] == 0
    assert applied[0, 1] == 0
    assert applied[0, 2] == 0
    np.testing.assert_array_equal(applied[0, 3:], combat[0, 3:])


def test_reset_restores_exact_combat_handoff() -> None:
    handoff = FairTacticalHandoff()
    combat = np.ones((1, 8), dtype=np.float32)
    handoff.apply(actor(visible=False), combat, [11])
    handoff.reset(11)
    applied, intervened = handoff.apply(actor(visible=True), combat, [11])
    np.testing.assert_array_equal(applied, combat)
    assert not intervened.any()
