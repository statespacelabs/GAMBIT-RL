#!/usr/bin/env python3
"""V002 screen adapter: add explicit inert state objects for scripted anchors."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


SOURCE = Path(__file__).resolve().with_name("screen_learned_controllers.py")
spec = importlib.util.spec_from_file_location("phase6_bot_screen_match_v001_source", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import V001 screen adapter: {SOURCE}")
screen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = screen
spec.loader.exec_module(screen)


class ScriptedAnchorRuntime(screen.ScriptedAnchorRuntime):
    """Anchor with explicit unique sentinels for release-runner independence audits."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = object()
        self.normalizer = object()
        self.nav_hidden: dict[int, Any] = {}
        self.combat_hidden: dict[int, Any] = {}

    def reset(self, agent_id: int) -> int:
        agent_id = int(agent_id)
        self.nav_hidden.pop(agent_id, None)
        self.combat_hidden.pop(agent_id, None)
        return int(super().reset(agent_id))


class ScreenRuntimeFactory:
    def __new__(cls, **kwargs: Any) -> Any:
        if str(kwargs["policy_id"]).startswith("screen_anchor_"):
            return ScriptedAnchorRuntime(**kwargs)
        return screen.ScreenRuntimeFactory(**kwargs)


screen.research.base.base.release.PolicyRuntime = ScreenRuntimeFactory


if __name__ == "__main__":
    raise SystemExit(screen.research.main())
