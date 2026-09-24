#!/usr/bin/env python3
"""Fair-telemetry tactical handoff used after a hidden-target hunt."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from scripts.navigation_policy import ACTOR_DIM, ACTION_DIM, LOS_INDEX


class FairTacticalHandoff:
    """Stabilize a reacquired target without privileged state or map identity.

    Always-visible encounters remain entirely under the frozen combat expert.
    Once an agent has experienced hidden LOS in the current life, a later
    actor-visible contact may use the schema's allowed relative bearing and
    elevation to hold contact while retaining the frozen expert as an
    unchanged external policy.
    """

    def __init__(self) -> None:
        self._was_hidden: set[int] = set()

    def reset(self, agent_id: int) -> None:
        self._was_hidden.discard(int(agent_id))

    def apply(
        self,
        actor_observation: np.ndarray,
        combat_composed_action: np.ndarray,
        agent_ids: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        actor = np.asarray(actor_observation, dtype=np.float32)
        output = np.asarray(combat_composed_action, dtype=np.float32).copy()
        ids = [int(value) for value in agent_ids]
        if actor.ndim != 2 or actor.shape[1] != ACTOR_DIM:
            raise ValueError("tactical handoff requires actor231")
        if output.shape != (len(actor), ACTION_DIM) or len(ids) != len(actor):
            raise ValueError("tactical handoff action/agent shape mismatch")
        intervened = np.zeros(len(actor), dtype=np.bool_)
        for row, agent_id in enumerate(ids):
            visible = actor[row, LOS_INDEX] > 0.5
            if not visible:
                self._was_hidden.add(agent_id)
                continue
            if agent_id not in self._was_hidden:
                continue
            # Tactical mode index 229 is the fair telemetry's combat bit.
            if actor[row, 229] <= 0.5:
                continue
            # Preserve the frozen expert pitch, reload and stance; correct actor-visible yaw and hold fire.
            # Only locomotion is held to avoid walking back behind cover.
            output[row, 0:2] = 0.0
            bearing = float(np.arctan2(actor[row, 194], actor[row, 195]))
            output[row, 2] = float(np.clip(bearing / (np.pi / 4.0), -1.0, 1.0))
            if actor[row, 9] > 1e-4:
                output[row, 4] = 1.0
            intervened[row] = True
        return output, intervened


def apply_fair_tactical_handoff(
    actor_observation: np.ndarray,
    action: np.ndarray,
    agent_ids: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    return FairTacticalHandoff().apply(actor_observation, action, agent_ids)
