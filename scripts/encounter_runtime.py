#!/usr/bin/env python3
"""Goal 2 encounter/session evaluator with frozen close-search spawn bounds."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
TOOLS = Path(__file__).resolve().parent
MINIMUM_MATCH = TOOLS / "hunter_runtime.py"
RELEASE_SOURCE = ROOT / "scripts/dual_policy_runtime.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mvh = load_module("minimum_hunter_goal2_base", MINIMUM_MATCH)

# The certified release loop is reused verbatim except for authored multi-match
# support. This permits one Unity process to switch among three preloaded,
# independently recurrent Phase 4 reference runtimes at terminal boundaries.
source = RELEASE_SOURCE.read_text(encoding="utf-8")
old_authored = "or config.target_matches != 1"
old_boundary = "area_match_index[area_id] >= 1"
old_no_graphics = "no_graphics=True,"
old_headless_arg = '"--phase5-headless",'
if (
    source.count(old_authored) != 1
    or source.count(old_boundary) != 1
    or source.count(old_no_graphics) != 1
    or source.count(old_headless_arg) != 1
):
    raise RuntimeError("certified release source no longer matches Goal 2 patch contract")
source = source.replace(old_authored, "or config.target_matches < 1", 1)
source = source.replace(
    old_boundary,
    "area_match_index[area_id] >= config.target_matches",
    1,
)
source = source.replace(
    old_no_graphics,
    'no_graphics=os.environ.get("MVH_RENDERED", "0") != "1",',
    1,
)
source = source.replace(
    old_headless_arg,
    '*([] if os.environ.get("MVH_RENDERED", "0") == "1" '
    'else ["--phase5-headless"]),',
    1,
)
release = types.ModuleType("phase6_release_match_batch_minimum_goal2")
release.__file__ = str(RELEASE_SOURCE)
sys.modules[release.__name__] = release
exec(compile(source, str(RELEASE_SOURCE), "exec"), release.__dict__)


class SessionOpponentRuntime:
    """Preload three frozen Phase 4 runtimes and advance only after terminal reset."""

    def __init__(self, **kwargs: Any) -> None:
        roster = json.loads(os.environ["MVH_SESSION_ROSTER_JSON"])
        if len(roster) != 3:
            raise RuntimeError("continuous minimum session requires exactly three opponents")
        self.roster = roster
        self.experts = []
        for index, item in enumerate(roster):
            child = dict(kwargs)
            child.update(
                {
                    "policy_id": item["policy_id"],
                    "checkpoint": Path(item["path"]),
                    "checkpoint_sha256": item["sha256"],
                    "combat_checkpoint": Path(item["path"]),
                    "combat_sha256": item["sha256"],
                    "seed_offset": int(kwargs["seed_offset"]) + index * 100,
                }
            )
            self.experts.append(mvh.d2.FairWrappedPhase4Runtime(**child))
        self.active_index = 0
        self.policy_id = "phase4_session_roster"
        self.model = self.experts[0].model
        self.normalizer = self.experts[0].normalizer
        self.nav_hidden: dict[int, Any] = {}
        self.combat_hidden: dict[int, Any] = {}
        self.safety = self.experts[0].safety

    def act(self, **kwargs: Any) -> Any:
        return self.experts[self.active_index].act(**kwargs)

    def reset(self, agent_id: int) -> int:
        leak = self.experts[self.active_index].reset(agent_id)
        if self.active_index < len(self.experts) - 1:
            self.active_index += 1
            leak += self.experts[self.active_index].reset(agent_id)
        self.nav_hidden.clear()
        self.combat_hidden.clear()
        self.safety = self.experts[self.active_index].safety
        return int(leak)


class RuntimeFactory:
    def __new__(cls, **kwargs: Any) -> Any:
        policy_id = str(kwargs["policy_id"])
        if policy_id == "phase4_session_roster":
            return SessionOpponentRuntime(**kwargs)
        if policy_id.startswith("phase4_"):
            return mvh.d2.FairWrappedPhase4Runtime(**kwargs)
        return mvh.d2.FairPhase6Runtime(**kwargs)


original_configure = release.configure_environment


def configure_close_search(config: Any, output: Path) -> dict[str, Path]:
    paths = original_configure(config, output)
    os.environ.update(
        {
            "PHASE4_4_REQUIRE_OBSTACLE_BETWEEN": "1",
            "PHASE4_4_SPAWN_DISTANCE_MIN": os.environ.get(
                "MVH_SPAWN_DISTANCE_MIN", "10"
            ),
            "PHASE4_4_SPAWN_DISTANCE_MAX": os.environ.get(
                "MVH_SPAWN_DISTANCE_MAX", "15"
            ),
        }
    )
    return paths


release.PolicyRuntime = RuntimeFactory
release.configure_environment = configure_close_search


def augment(output: Path) -> None:
    mvh.d2.augment_outputs(output)
    summary_path = output / "run_summary.json"
    matches_path = output / "matches.jsonl"
    if not summary_path.is_file() or not matches_path.is_file():
        return
    summary = json.loads(summary_path.read_text())
    rows = [
        json.loads(line)
        for line in matches_path.read_text().splitlines()
        if line.strip()
    ]
    roster = json.loads(os.environ.get("MVH_SESSION_ROSTER_JSON", "[]"))
    if roster:
        for index, row in enumerate(rows):
            opponent = roster[min(index, len(roster) - 1)]
            old_id = row.get("policy_b_id")
            row["session_opponent_id"] = opponent["policy_id"]
            row["policy_b_id"] = opponent["policy_id"]
            row["policy_b_sha256"] = opponent["sha256"]
            if row.get("winner_policy_id") == old_id:
                row["winner_policy_id"] = opponent["policy_id"]
        matches_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    summary["minimum_certification"] = {
        "scenario": "authored_hidden_obstacle_close_search",
        "spawn_distance_min": float(os.environ.get("MVH_SPAWN_DISTANCE_MIN", "10")),
        "spawn_distance_max": float(os.environ.get("MVH_SPAWN_DISTANCE_MAX", "15")),
        "session_roster": [item["policy_id"] for item in roster],
        "opponents_preloaded": bool(roster),
        "opponent_switch_only_after_terminal_reset": bool(roster),
        "candidate_model_switch": False,
        "new_training": False,
    }
    summary["matches_sha256"] = mvh.d2.sha256_file(matches_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    output_arg = next(
        (
            sys.argv[index + 1]
            for index, value in enumerate(sys.argv[:-1])
            if value == "--output-dir"
        ),
        "",
    )
    code = release.main()
    if output_arg:
        augment(Path(output_arg).resolve())
    return code


if __name__ == "__main__":
    raise SystemExit(main())
