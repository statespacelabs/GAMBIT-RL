#!/usr/bin/env python3
"""Fair, frozen minimum hunter/duelist runtime for Goal 1.

This wraps the certified D2 evaluator. The hunter consumes actor231 every
decision. The duelist's frozen local45 recurrence is updated only while the
same actor231 row declares LOS, exactly as in D2. No checkpoint is modified.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
D2_RUNNER = ROOT / "scripts/visible_combat_runtime.py"

spec = importlib.util.spec_from_file_location("phase6_d2_match_minimum", D2_RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import D2 runner: {D2_RUNNER}")
d2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = d2
spec.loader.exec_module(d2)

from scripts.collect_navigation_dagger import action_from_distribution  # noqa: E402
from scripts.navigation_policy import LOS_INDEX, compose_actions  # noqa: E402


@dataclass
class RouteState:
    mode: str = "HUNT"
    mode_ticks: int = 0
    visible_streak: int = 0
    absent_streak: int = 0


class RoutingStateMachine:
    """Hysteretic HUNT/ENGAGE/REACQUIRE routing with explicit dwell."""

    def __init__(
        self,
        *,
        minimum_hunt_dwell: int,
        minimum_engage_dwell: int,
        stable_los_steps: int,
        reacquire_absent_steps: int,
    ) -> None:
        self.minimum_hunt_dwell = int(minimum_hunt_dwell)
        self.minimum_engage_dwell = int(minimum_engage_dwell)
        self.stable_los_steps = int(stable_los_steps)
        self.reacquire_absent_steps = int(reacquire_absent_steps)
        self.states: dict[int, RouteState] = {}

    def update(self, agent_id: int, *, visible: bool, damaged: bool) -> str:
        agent_id = int(agent_id)
        state = self.states.setdefault(agent_id, RouteState())
        state.mode_ticks += 1
        state.visible_streak = state.visible_streak + 1 if visible else 0
        state.absent_streak = 0 if visible else state.absent_streak + 1

        if state.mode in {"HUNT", "REACQUIRE"}:
            if (
                state.mode_ticks >= self.minimum_hunt_dwell
                and (
                    state.visible_streak >= self.stable_los_steps
                    or damaged
                )
            ):
                state.mode = "ENGAGE"
                state.mode_ticks = 0
                state.absent_streak = 0
        elif (
            state.mode == "ENGAGE"
            and state.mode_ticks >= self.minimum_engage_dwell
            and state.absent_streak >= self.reacquire_absent_steps
        ):
            state.mode = "REACQUIRE"
            state.mode_ticks = 0
            state.visible_streak = 0
        return state.mode

    def reset(self, agent_id: int) -> None:
        self.states.pop(int(agent_id), None)


class BoundedVisibleAimResidual:
    """Delayed, smoothed actor231-visible bearing/elevation correction."""

    def __init__(
        self,
        *,
        maximum: float,
        delay_steps: int = 6,
        smoothing: float = 0.25,
    ) -> None:
        if maximum not in (0.0, 0.05, 0.10):
            raise ValueError("aim residual must be 0%, 5%, or 10%")
        self.maximum = float(maximum)
        self.delay_steps = int(delay_steps)
        self.smoothing = float(smoothing)
        self.queues: dict[int, deque[np.ndarray]] = {}
        self.smoothed: dict[int, np.ndarray] = {}

    def apply(
        self,
        actor: np.ndarray,
        actions: np.ndarray,
        ids: list[int],
        engage: list[bool],
    ) -> np.ndarray:
        output = np.asarray(actions, dtype=np.float32).copy()
        for row, agent_id in enumerate(ids):
            agent_id = int(agent_id)
            visible = bool(actor[row, LOS_INDEX] > 0.5)
            if self.maximum <= 0.0 or not visible or not engage[row]:
                self.queues.pop(agent_id, None)
                self.smoothed.pop(agent_id, None)
                continue
            bearing = math.atan2(float(actor[row, 194]), float(actor[row, 195]))
            elevation = math.atan2(float(actor[row, 196]), float(actor[row, 197]))
            desired = np.asarray(
                [
                    np.clip(math.degrees(bearing) / 55.0, -self.maximum, self.maximum),
                    np.clip(math.degrees(elevation) / 40.0, -self.maximum, self.maximum),
                ],
                dtype=np.float32,
            )
            queue = self.queues.setdefault(
                agent_id,
                deque(
                    [np.zeros(2, dtype=np.float32) for _ in range(self.delay_steps)],
                    maxlen=self.delay_steps + 1,
                ),
            )
            queue.append(desired)
            delayed = queue.popleft() if len(queue) > self.delay_steps else np.zeros(2)
            previous = self.smoothed.get(agent_id, np.zeros(2, dtype=np.float32))
            correction = (
                (1.0 - self.smoothing) * previous
                + self.smoothing * np.asarray(delayed, dtype=np.float32)
            )
            correction = np.clip(correction, -self.maximum, self.maximum)
            self.smoothed[agent_id] = correction
            output[row, 2:4] = np.clip(
                output[row, 2:4] + correction, -1.0, 1.0
            )
        return output

    def reset(self, agent_id: int) -> None:
        self.queues.pop(int(agent_id), None)
        self.smoothed.pop(int(agent_id), None)


class MinimumHunterRuntime(d2.FairPhase6Runtime):
    """Frozen actor231 hunter plus a frozen, visible-only local45 duelist."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.route = RoutingStateMachine(
            minimum_hunt_dwell=int(os.environ.get("MVH_HUNT_DWELL", "16")),
            minimum_engage_dwell=int(os.environ.get("MVH_ENGAGE_DWELL", "16")),
            stable_los_steps=int(os.environ.get("MVH_STABLE_LOS_STEPS", "3")),
            reacquire_absent_steps=int(
                os.environ.get("MVH_REACQUIRE_ABSENT_STEPS", "38")
            ),
        )
        self.aim = BoundedVisibleAimResidual(
            maximum=float(os.environ.get("MVH_AIM_RESIDUAL", "0")),
            delay_steps=int(os.environ.get("MVH_AIM_DELAY_STEPS", "6")),
            smoothing=float(os.environ.get("MVH_AIM_SMOOTHING", "0.25")),
        )

    def act(
        self,
        *,
        actor: np.ndarray,
        local45: np.ndarray,
        ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool]]:
        actor = np.asarray(actor, dtype=np.float32)
        local45 = np.asarray(local45, dtype=np.float32)

        combat_actions = np.zeros((len(ids), 8), dtype=np.float32)
        visible_rows = np.flatnonzero(actor[:, LOS_INDEX] > 0.5)
        if len(visible_rows):
            visible_ids = [ids[index] for index in visible_rows]
            combat_actions[visible_rows] = action_from_distribution(
                self.combat,
                self.normalizer,
                self.combat_device,
                local45[visible_rows],
                visible_ids,
                self.combat_hidden,
            )

        hidden_before = torch.cat(
            [
                self.nav_hidden.get(
                    agent_id, self.model.initial_hidden(1, self.device)
                )
                for agent_id in ids
            ],
            dim=1,
        )
        with torch.no_grad():
            prediction, next_hidden = self.model(
                torch.as_tensor(actor, device=self.device).unsqueeze(1),
                hidden_before,
            )
        mean = prediction["mean"][:, 0]
        raw = mean
        if self.mode == "stochastic":
            raw = mean + torch.randn(
                mean.shape,
                device=mean.device,
                dtype=mean.dtype,
                generator=self.generator,
            ) * prediction["log_std"][:, 0].exp()
        sampled = raw.clamp(-1.0, 1.0)
        navigation = np.zeros((len(ids), 8), dtype=np.float32)
        navigation[:, :4] = sampled.cpu().numpy()
        navigation, _ = self.safety.apply(actor, navigation, ids)

        engage: list[bool] = []
        for row, agent_id in enumerate(ids):
            visible = bool(actor[row, LOS_INDEX] > 0.5)
            damaged = bool(actor[row, 12] > 0.5)
            mode = self.route.update(
                agent_id, visible=visible, damaged=damaged
            )
            engage.append(mode == "ENGAGE")

        applied = navigation.copy()
        for row in range(len(ids)):
            if engage[row] and actor[row, LOS_INDEX] > 0.5:
                applied[row] = compose_actions(
                    actor[row : row + 1],
                    navigation[row : row + 1],
                    combat_actions[row : row + 1],
                )[0]
        applied = self.aim.apply(actor, applied, ids, engage)

        for index, agent_id in enumerate(ids):
            self.nav_hidden[agent_id] = next_hidden[
                :, index : index + 1
            ].detach()
        return (
            applied,
            mean.detach().cpu().numpy(),
            raw.detach().cpu().numpy(),
            [False] * len(ids),
        )

    def reset(self, agent_id: int) -> int:
        leak = super().reset(agent_id)
        self.route.reset(agent_id)
        self.aim.reset(agent_id)
        return int(
            leak
            or int(agent_id) in self.route.states
            or int(agent_id) in self.aim.queues
            or int(agent_id) in self.aim.smoothed
        )


class RuntimeFactory:
    def __new__(cls, **kwargs: Any) -> Any:
        policy_id = str(kwargs["policy_id"])
        if policy_id.startswith("phase4_"):
            return d2.FairWrappedPhase4Runtime(**kwargs)
        if policy_id.startswith("mvh_"):
            return MinimumHunterRuntime(**kwargs)
        return d2.FairPhase6Runtime(**kwargs)


def augment_minimum_summary(output: Path) -> None:
    summary_path = output / "run_summary.json"
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["minimum_viable_hunter"] = {
        "composition": os.environ.get("MVH_COMPOSITION", "0") == "1",
        "minimum_hunt_dwell_decisions": int(os.environ.get("MVH_HUNT_DWELL", "16")),
        "minimum_engage_dwell_decisions": int(
            os.environ.get("MVH_ENGAGE_DWELL", "16")
        ),
        "stable_los_decisions": int(os.environ.get("MVH_STABLE_LOS_STEPS", "3")),
        "reacquire_absent_decisions": int(
            os.environ.get("MVH_REACQUIRE_ABSENT_STEPS", "38")
        ),
        "aim_residual_maximum": float(os.environ.get("MVH_AIM_RESIDUAL", "0")),
        "aim_residual_delay_decisions": int(
            os.environ.get("MVH_AIM_DELAY_STEPS", "6")
        ),
        "aim_residual_visible_only": True,
        "automatic_fire_added": False,
        "hidden_local45_update": False,
        "opponent_identity_input": False,
        "map_identity_input": False,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    d2.release.PolicyRuntime = RuntimeFactory
    output_arg = next(
        (
            sys.argv[index + 1]
            for index, value in enumerate(sys.argv[:-1])
            if value == "--output-dir"
        ),
        "",
    )
    code = d2.main()
    if output_arg:
        augment_minimum_summary(Path(output_arg).resolve())
    return code


if __name__ == "__main__":
    raise SystemExit(main())
