#!/usr/bin/env python3
"""Goal 3 CUDA examiner-bank match adapter.

Extends the frozen terminal tournament evaluator with deterministic, fair
parametric anchors and the already-screened multi-family policy runtimes.
Anchors may use only the released actor observation and visible local45 aim
channels; hidden-target coordinates are never read.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT
os.environ.setdefault("RESEARCH_REPO_OVERLAY", str(ROOT))
os.environ.setdefault(
    "RESEARCH_HUD_STATE_PATH", "/tmp/phase6_examiner_bank_inert_hud.json"
)
TOURNAMENT_SOURCE = ROOT / "scripts/symmetric_tournament.py"
tournament_spec = importlib.util.spec_from_file_location(
    "phase6_examiner_tournament_source", TOURNAMENT_SOURCE
)
if tournament_spec is None or tournament_spec.loader is None:
    raise RuntimeError(f"cannot import tournament runner: {TOURNAMENT_SOURCE}")
tournament = importlib.util.module_from_spec(tournament_spec)
sys.modules[tournament_spec.name] = tournament
tournament_spec.loader.exec_module(tournament)
BasePolicyRuntime = tournament.PolicyRuntime

SCREEN_SOURCE = Path(__file__).resolve().with_name(
    "screen_anchor_controllers.py"
)
screen_spec = importlib.util.spec_from_file_location(
    "phase6_examiner_goal1_adapter", SCREEN_SOURCE
)
if screen_spec is None or screen_spec.loader is None:
    raise RuntimeError(f"cannot import Goal 1 screen adapter: {SCREEN_SOURCE}")
goal1 = importlib.util.module_from_spec(screen_spec)
sys.modules[screen_spec.name] = goal1
screen_spec.loader.exec_module(goal1)

screen = goal1.screen
LOS_INDEX = screen.LOS_INDEX


class ExaminerAnchorRuntime(goal1.ScriptedAnchorRuntime):
    """Frozen visible-target teacher plus fair actor-only reacquisition."""

    def __init__(
        self,
        *,
        policy_id: str,
        checkpoint: Path,
        seed: int,
        seed_offset: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            policy_id=policy_id,
            checkpoint=checkpoint,
            seed=seed,
            seed_offset=seed_offset,
            **kwargs,
        )
        payload = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
        anchors = payload.get("anchors", {})
        if policy_id not in anchors:
            raise RuntimeError(f"anchor not frozen in specification: {policy_id}")
        self.parameters = dict(anchors[policy_id])
        self.tick: dict[int, int] = {}
        self.phase_offset: dict[int, int] = {}
        self.seed = int(seed) + int(seed_offset)

    def _direction(self, agent_id: int, period: int) -> float:
        if agent_id not in self.phase_offset:
            self.phase_offset[agent_id] = (
                self.seed * 1103515245 + int(agent_id) * 12345
            ) % max(1, period * 2)
        tick = self.tick.get(agent_id, 0) + self.phase_offset[agent_id]
        return 1.0 if (tick // max(1, period)) % 2 == 0 else -1.0

    def act(
        self, *, actor: np.ndarray, local45: np.ndarray, ids: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool]]:
        actor = np.asarray(actor, dtype=np.float32)
        local45 = np.asarray(local45, dtype=np.float32)
        applied = np.zeros((len(ids), 8), dtype=np.float32)

        # Fair hidden-target behavior reads only released actor memory/rays.
        hidden_rows = np.flatnonzero(actor[:, LOS_INDEX] <= 0.5)
        if len(hidden_rows):
            hidden_ids = [ids[int(index)] for index in hidden_rows]
            hidden_actions = np.zeros((len(hidden_rows), 8), dtype=np.float32)
            hidden_actions, _ = self.search.apply(
                actor[hidden_rows], hidden_actions, hidden_ids
            )
            hidden_actions, _ = self.safety.apply(
                actor[hidden_rows], hidden_actions, hidden_ids
            )
            applied[hidden_rows] = hidden_actions

        visible_rows = np.flatnonzero(actor[:, LOS_INDEX] > 0.5)
        for raw_index in visible_rows:
            index = int(raw_index)
            agent_id = int(ids[index])
            aim = float(local45[index, 22])
            if not np.isfinite(aim) or aim < 0.0 or aim >= 149.0:
                continue
            direction = self._direction(
                agent_id, int(self.parameters["switch_period_decisions"])
            )
            applied[index, 0] = float(self.parameters["strafe"]) * direction
            applied[index, 1] = float(self.parameters["forward"])
            applied[index, 2] = np.clip(
                float(local45[index, 20]) / 45.0,
                -float(self.parameters["max_look"]),
                float(self.parameters["max_look"]),
            )
            applied[index, 3] = np.clip(
                -float(local45[index, 21]) / 45.0,
                -float(self.parameters["max_look"]),
                float(self.parameters["max_look"]),
            )
            applied[index, 4] = float(
                aim <= float(self.parameters["shoot_threshold_deg"])
            )

        # Safety is actor-only and may modify movement, never visible aiming.
        if len(visible_rows):
            visible_ids = [ids[int(index)] for index in visible_rows]
            safe, _ = self.safety.apply(
                actor[visible_rows], applied[visible_rows], visible_ids
            )
            applied[visible_rows] = safe

        for agent_id in ids:
            self.tick[int(agent_id)] = self.tick.get(int(agent_id), 0) + 1
        continuous = applied[:, :4].copy()
        return applied, continuous, continuous.copy(), [False] * len(ids)

    def reset(self, agent_id: int) -> int:
        agent_id = int(agent_id)
        self.tick.pop(agent_id, None)
        self.phase_offset.pop(agent_id, None)
        leaked = int(super().reset(agent_id))
        return int(leaked or agent_id in self.tick or agent_id in self.phase_offset)


class ExaminerRuntimeFactory:
    def __new__(cls, **kwargs: Any) -> Any:
        policy_id = str(kwargs["policy_id"])
        if policy_id.startswith("bank_anchor_"):
            return ExaminerAnchorRuntime(**kwargs)
        if policy_id == "bank_original_combat_expert":
            return screen.CombatOnlyRuntime(**kwargs)
        if policy_id.startswith("bank_phase4_"):
            return screen.ScreenPhase4Runtime(**kwargs)
        if policy_id.startswith("bank_phase5_"):
            return BasePolicyRuntime(**kwargs)
        return goal1.ScreenRuntimeFactory(**kwargs)


tournament.PolicyRuntime = ExaminerRuntimeFactory


if __name__ == "__main__":
    os.environ.setdefault("MVH_RENDERED", "0")
    raise SystemExit(tournament.main())
