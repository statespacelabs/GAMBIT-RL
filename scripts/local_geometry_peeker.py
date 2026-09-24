#!/usr/bin/env python3
"""Actor231-only local-cover detection and peek tactical control."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from scripts.navigation_policy import ACTOR_DIM, ACTION_DIM, LOS_INDEX

ENEMY_HEALTH_CRITIC_INDEX = 372


@dataclass(frozen=True)
class LocalCoverState:
    behind_cover: bool
    lateral_space_left: bool
    lateral_space_right: bool
    enemy_bearing_degrees: float
    obstacle_clearance: float
    left_clearance: float
    right_clearance: float


@dataclass
class AgentPeekState:
    phase: str = "idle"
    side: int = 0
    phase_ticks: int = 0
    dwell_target: int = 20
    useful: bool = False
    damage_positive: bool = False
    fired: bool = False
    was_exposed: bool = False
    cover_latched: bool = False
    last_side: int = 0
    last_end_tick: int = -10000
    shaping_abs: float = 0.0
    previous_enemy_health: float | None = None


@dataclass
class PeekMetrics:
    valid_cover_engagements: int = 0
    cover_uses: int = 0
    peek_starts: int = 0
    exposed_events: int = 0
    completed_peeks: int = 0
    useful_peeks: int = 0
    damage_positive_peeks: int = 0
    rapid_same_side_repeeks: int = 0
    edge_oscillations: int = 0
    return_after_firing: int = 0
    prolonged_useless_exposures: int = 0

    def as_dict(self) -> dict[str, float | int]:
        cover_denominator = max(1, self.valid_cover_engagements)
        peek_denominator = max(1, self.valid_cover_engagements)
        start_denominator = max(1, self.peek_starts)
        return {
            **self.__dict__,
            "cover_use_rate": min(1.0, self.cover_uses / cover_denominator),
            "completed_peek_rate": min(1.0, self.completed_peeks / peek_denominator),
            "useful_peek_rate": self.useful_peeks / max(1, self.completed_peeks),
            "damage_positive_peek_rate": self.damage_positive_peeks / max(1, self.completed_peeks),
            "rapid_same_side_repeek_rate": self.rapid_same_side_repeeks / start_denominator,
        }


def _enemy_bearing(actor: np.ndarray) -> float | None:
    if actor[LOS_INDEX] > 0.5:
        return float(np.degrees(np.arctan2(actor[194], actor[195])))
    if actor[202] > 0.5 and actor[206] < 0.8:
        return float(np.degrees(np.arctan2(actor[203], actor[205])))
    if actor[207] > 0.5 and actor[210] < 0.9:
        return float(np.degrees(np.arctan2(actor[208], actor[209])))
    if actor[224] > 0.5:
        bearing_bin = int(np.argmax(actor[211:219]))
        return -180.0 + (bearing_bin + 0.5) * 45.0
    return None


def detect_local_cover(actor_observation: np.ndarray) -> LocalCoverState:
    """Derive the complete cover contract from one fair actor observation."""
    actor = np.asarray(actor_observation, dtype=np.float32)
    if actor.shape != (ACTOR_DIM,) or not np.isfinite(actor).all():
        raise ValueError("cover detection requires one finite actor231 observation")
    bearing = _enemy_bearing(actor)
    obstacle_clearance = 1.0
    obstacle_near_bearing = False
    if bearing is not None:
        ray_index = int(round((bearing % 360.0) / 11.25)) % 32
        obstacle_clearance = float(actor[30 + ray_index * 2])
        obstacle_near_bearing = bool(
            actor[31 + ray_index * 2] > 0.5 and obstacle_clearance < 0.28
        )
    left_clearance = float(max(actor[30 + 23 * 2], actor[30 + 24 * 2], actor[30 + 25 * 2]))
    right_clearance = float(max(actor[30 + 7 * 2], actor[30 + 8 * 2], actor[30 + 9 * 2]))
    left = left_clearance >= 0.065 and actor[185 + 6] >= 0.045
    right = right_clearance >= 0.065 and actor[185 + 2] >= 0.045
    return LocalCoverState(
        behind_cover=bool(actor[LOS_INDEX] <= 0.5 and obstacle_near_bearing),
        lateral_space_left=bool(left),
        lateral_space_right=bool(right),
        enemy_bearing_degrees=float(bearing or 0.0),
        obstacle_clearance=obstacle_clearance,
        left_clearance=left_clearance,
        right_clearance=right_clearance,
    )


class LocalGeometryPeeker:
    """Stateful tactical adapter; inputs are actor231 plus reward-only health labels."""

    def __init__(
        self,
        *,
        seed: int,
        dwell_min: int,
        dwell_max: int,
        risk: float,
        repeek_penalty: float,
    ) -> None:
        if dwell_min <= 0 or dwell_max < dwell_min:
            raise ValueError("invalid dwell distribution")
        self.rng = np.random.default_rng(seed)
        self.dwell_min = int(dwell_min)
        self.dwell_max = int(dwell_max)
        self.risk = float(np.clip(risk, 0.0, 1.0))
        self.repeek_penalty = float(np.clip(repeek_penalty, 0.0, 0.10))
        self.states: dict[int, AgentPeekState] = {}
        self.metrics = PeekMetrics()
        self.tick = 0

    def reset(self, agent_id: int) -> None:
        self.states.pop(int(agent_id), None)

    @staticmethod
    def _bounded_reward(state: AgentPeekState, proposed: float) -> float:
        remaining = max(0.0, 0.20 - state.shaping_abs)
        applied = float(np.clip(proposed, -remaining, remaining))
        state.shaping_abs += abs(applied)
        return applied

    def _choose_side(self, cover: LocalCoverState, state: AgentPeekState) -> int:
        available = []
        if cover.lateral_space_left:
            available.append(-1)
        if cover.lateral_space_right:
            available.append(1)
        if not available:
            return 0
        if len(available) == 1:
            if state.last_side == available[0] and self.tick - state.last_end_tick < 75:
                # Wait for a genuinely new route/angle instead of immediately
                # bouncing over the same exposed edge.
                return 0
            return available[0]
        clearance_choice = 1 if cover.right_clearance >= cover.left_clearance else -1
        if state.last_side and self.tick - state.last_end_tick < 75:
            return -state.last_side
        return clearance_choice if self.rng.random() >= self.risk * 0.35 else -clearance_choice

    def apply(
        self,
        actor_observation: np.ndarray,
        navigation_action: np.ndarray,
        combat_action: np.ndarray,
        agent_ids: Iterable[int],
        enemy_health: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        actor = np.asarray(actor_observation, dtype=np.float32)
        navigation = np.asarray(navigation_action, dtype=np.float32)
        combat = np.asarray(combat_action, dtype=np.float32)
        ids = [int(value) for value in agent_ids]
        if actor.ndim != 2 or actor.shape[1] != ACTOR_DIM or len(ids) != len(actor):
            raise ValueError("peeker requires [batch,231] actor observations")
        if navigation.shape != (len(actor), ACTION_DIM) or combat.shape != navigation.shape:
            raise ValueError("peeker action shape mismatch")
        health = None if enemy_health is None else np.asarray(enemy_health, dtype=np.float32)
        applied = navigation.copy()
        target = navigation.copy()
        rewards = np.zeros(len(actor), dtype=np.float32)
        active = np.zeros(len(actor), dtype=np.bool_)
        self.tick += 1
        for row, agent_id in enumerate(ids):
            state = self.states.setdefault(agent_id, AgentPeekState())
            cover = detect_local_cover(actor[row])
            visible = actor[row, LOS_INDEX] > 0.5
            # The tactical adapter owns translation only.  Whenever the enemy is
            # visible, inherit the frozen expert's aim/shoot action verbatim.
            if visible:
                applied[row] = combat[row]
            current_health = float(health[row]) if health is not None else None
            damage_delta = (
                current_health is not None and state.previous_enemy_health is not None
                and current_health < state.previous_enemy_health - 1e-5
            )
            if current_health is not None:
                state.previous_enemy_health = current_health
            if cover.behind_cover and not state.cover_latched:
                state.cover_latched = True
                self.metrics.valid_cover_engagements += 1
            elif visible:
                state.cover_latched = False

            if state.phase == "idle" and cover.behind_cover:
                side = self._choose_side(cover, state)
                if side:
                    if state.last_side == side and self.tick - state.last_end_tick < 75:
                        self.metrics.rapid_same_side_repeeks += 1
                        rewards[row] += self._bounded_reward(state, -self.repeek_penalty)
                    state.phase = "peek"
                    state.side = side
                    state.phase_ticks = 0
                    state.dwell_target = int(self.rng.integers(self.dwell_min, self.dwell_max + 1))
                    state.useful = state.damage_positive = state.fired = False
                    state.was_exposed = False
                    self.metrics.cover_uses += 1
                    self.metrics.peek_starts += 1

            if state.phase == "peek":
                active[row] = True
                target[row, 0] = applied[row, 0] = 0.90 * state.side
                target[row, 1] = applied[row, 1] = 0.10
                if visible:
                    state.phase = "exposed"
                    state.phase_ticks = 0
                    state.was_exposed = True
                    self.metrics.exposed_events += 1
                elif state.phase_ticks >= 75:
                    self.metrics.edge_oscillations += 1
                    rewards[row] += self._bounded_reward(state, -0.02)
                    state.phase = "return"
                    state.phase_ticks = 0
            elif state.phase == "exposed":
                active[row] = True
                applied[row] = combat[row]
                target[row, 0] = 0.15 * state.side
                target[row, 1] = 0.05
                applied[row, 0:2] = target[row, 0:2]
                fired = bool(combat[row, 4] > 0.5 or actor[row, 20] > 0.5)
                state.fired |= fired
                # Damage is a reward/metric label only.  Tactical transitions
                # depend exclusively on fair actor telemetry and expert firing.
                state.useful |= fired
                state.damage_positive |= bool(damage_delta)
                if state.useful and state.phase_ticks >= state.dwell_target:
                    state.phase = "return"
                    state.phase_ticks = 0
                    if state.fired:
                        self.metrics.return_after_firing += 1
                        rewards[row] += self._bounded_reward(state, 0.02)
                elif state.phase_ticks >= state.dwell_target and not state.useful:
                    self.metrics.prolonged_useless_exposures += 1
                    rewards[row] += self._bounded_reward(state, -0.02)
                    state.phase = "return"
                    state.phase_ticks = 0
            elif state.phase == "return":
                active[row] = True
                if visible:
                    applied[row] = combat[row]
                target[row, 0] = applied[row, 0] = -0.95 * state.side
                target[row, 1] = applied[row, 1] = -0.15
                if cover.behind_cover or (not visible and state.phase_ticks >= 8):
                    if state.was_exposed:
                        self.metrics.completed_peeks += 1
                        if state.useful:
                            self.metrics.useful_peeks += 1
                            rewards[row] += self._bounded_reward(state, 0.05)
                        if state.damage_positive:
                            self.metrics.damage_positive_peeks += 1
                            rewards[row] += self._bounded_reward(state, 0.03)
                    state.last_side = state.side
                    state.last_end_tick = self.tick
                    state.phase = "idle"
                    state.side = 0
                    state.phase_ticks = 0
            state.phase_ticks += 1
        return applied, target, rewards, active

    def summary(self) -> dict[str, float | int]:
        return self.metrics.as_dict()
