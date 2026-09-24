#!/usr/bin/env python3
"""Fair repaired-hunter runtime with a 16-decision tactical mode dwell."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


BASE_RUNNER = Path(__file__).resolve().parent / "encounter_runtime.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("hard_negative_repaired_cert_base", BASE_RUNNER)

from scripts.collect_navigation_dagger import action_from_distribution  # noqa: E402
from scripts.navigation_policy import LOS_INDEX  # noqa: E402


class RepairedRuntime(base.mvh.d2.FairPhase6Runtime):
    """Selected repaired navigator plus frozen visible-only combat expert.

    A mode may change only after 16 decisions.  Duel entry additionally
    requires four consecutive visible decisions; hunt re-entry requires four
    consecutive hidden decisions.  Navigation remains available during brief
    LOS losses, while local45 combat is evaluated only in visible duel mode.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tactical_mode: dict[int, str] = {}
        self.mode_dwell: dict[int, int] = {}
        self.visible_streak: dict[int, int] = {}
        self.hidden_streak: dict[int, int] = {}

    def _advance_mode(self, agent_id: int, visible: bool) -> str:
        mode = self.tactical_mode.get(agent_id, "hunt")
        dwell = self.mode_dwell.get(agent_id, 0) + 1
        visible_count = self.visible_streak.get(agent_id, 0) + 1 if visible else 0
        hidden_count = self.hidden_streak.get(agent_id, 0) + 1 if not visible else 0
        if mode == "hunt" and dwell >= 16 and visible_count >= 4:
            mode = "duel"
            dwell = 0
        elif mode == "duel" and dwell >= 16 and hidden_count >= 4:
            mode = "hunt"
            dwell = 0
        self.tactical_mode[agent_id] = mode
        self.mode_dwell[agent_id] = dwell
        self.visible_streak[agent_id] = visible_count
        self.hidden_streak[agent_id] = hidden_count
        return mode

    def act(
        self,
        *,
        actor: np.ndarray,
        local45: np.ndarray,
        ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool]]:
        actor = np.asarray(actor, dtype=np.float32)
        local45 = np.asarray(local45, dtype=np.float32)
        modes = [
            self._advance_mode(int(agent_id), bool(actor[row, LOS_INDEX] > 0.5))
            for row, agent_id in enumerate(ids)
        ]
        combat = np.zeros((len(ids), 8), dtype=np.float32)
        duel_rows = np.asarray(
            [
                row
                for row, mode in enumerate(modes)
                if mode == "duel" and actor[row, LOS_INDEX] > 0.5
            ],
            dtype=np.int64,
        )
        if len(duel_rows):
            duel_ids = [ids[int(row)] for row in duel_rows]
            combat[duel_rows] = action_from_distribution(
                self.combat,
                self.normalizer,
                self.combat_device,
                local45[duel_rows],
                duel_ids,
                self.combat_hidden,
            )

        hidden_before = torch.cat(
            [
                self.nav_hidden.get(agent_id, self.model.initial_hidden(1, self.device))
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
        applied = navigation.copy()
        if len(duel_rows):
            applied[duel_rows] = combat[duel_rows]
        for index, agent_id in enumerate(ids):
            self.nav_hidden[agent_id] = next_hidden[:, index : index + 1].detach()
        return applied, mean.detach().cpu().numpy(), raw.detach().cpu().numpy(), [False] * len(ids)

    def reset(self, agent_id: int) -> int:
        agent_id = int(agent_id)
        leak = super().reset(agent_id)
        self.tactical_mode.pop(agent_id, None)
        self.mode_dwell.pop(agent_id, None)
        self.visible_streak.pop(agent_id, None)
        self.hidden_streak.pop(agent_id, None)
        return int(
            leak
            or agent_id in self.tactical_mode
            or agent_id in self.mode_dwell
            or agent_id in self.visible_streak
            or agent_id in self.hidden_streak
        )


class RuntimeFactory:
    def __new__(cls, **kwargs: Any) -> Any:
        if str(kwargs["policy_id"]).startswith("phase4_"):
            return base.mvh.d2.FairWrappedPhase4Runtime(**kwargs)
        return RepairedRuntime(**kwargs)


base.release.PolicyRuntime = RuntimeFactory


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
