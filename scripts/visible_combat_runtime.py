#!/usr/bin/env python3
"""D2 fair showcase adapter around the certified Phase 6 release runner.

Phase 6 navigators consume actor231. The frozen Phase 4 reference policies are
local45 combat policies, so their actions and recurrent updates are admitted
only while actor231 reports true LOS. Hidden movement uses only actor231
last-seen/heard/coarse-hunt cues plus the map-independent safety layer.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "scripts/dual_policy_runtime.py"

spec = importlib.util.spec_from_file_location("phase6_release_match_batch_d2", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load source runner: {SOURCE}")
release = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = release
spec.loader.exec_module(release)

from scripts.collect_navigation_dagger import action_from_distribution  # noqa: E402
from scripts.navigation_policy import (  # noqa: E402
    FairReacquisitionSearch,
    LOS_INDEX,
    MapIndependentSafetyLayer,
    compose_actions,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FairPhase6Runtime(release.PolicyRuntime):
    """The certified navigator with visible-only local45 combat recurrence."""

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
        previous_phases = [
            self.peeker.states.get(agent_id).phase
            if self.peeker is not None and agent_id in self.peeker.states
            else "idle"
            for agent_id in ids
        ]
        if self.visible_tactic is not None:
            applied = compose_actions(actor, navigation, combat_actions)
            applied, _ = self.visible_tactic.apply(actor, applied, ids)
        elif self.peeker is not None:
            applied, _, _, _ = self.peeker.apply(
                actor, navigation, combat_actions, ids, None
            )
        else:
            applied = compose_actions(actor, navigation, combat_actions)
        peek_started = [
            self.peeker is not None
            and agent_id in self.peeker.states
            and previous_phases[index] != "peek"
            and self.peeker.states[agent_id].phase == "peek"
            for index, agent_id in enumerate(ids)
        ]
        for index, agent_id in enumerate(ids):
            self.nav_hidden[agent_id] = next_hidden[
                :, index : index + 1
            ].detach()
        return (
            applied,
            mean.detach().cpu().numpy(),
            raw.detach().cpu().numpy(),
            peek_started,
        )


class FairWrappedPhase4Runtime:
    """Frozen Phase 4 visible combat plus actor231-only hidden search."""

    def __init__(
        self,
        *,
        policy_id: str,
        checkpoint: Path,
        checkpoint_sha256: str,
        combat_checkpoint: Path,
        combat_sha256: str,
        normalizer_path: Path,
        normalizer_sha256: str,
        mode: str,
        seed: int,
        seed_offset: int,
        device: torch.device,
        env_path: str,
        output_dir: Path,
        base_port: int,
        time_scale: float,
    ) -> None:
        del seed, seed_offset
        from scripts.combat_runtime import (
            RuntimeConfig,
            load_policy_and_normalizer,
        )

        checkpoint = Path(checkpoint)
        combat_checkpoint = Path(combat_checkpoint)
        normalizer_path = Path(normalizer_path)
        for label, path, expected in (
            ("checkpoint", checkpoint, checkpoint_sha256),
            ("combat", combat_checkpoint, combat_sha256),
            ("normalizer", normalizer_path, normalizer_sha256),
        ):
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(
                    f"{policy_id} {label} hash mismatch: {actual} != {expected}"
                )
        if checkpoint.resolve() != combat_checkpoint.resolve():
            raise RuntimeError("Phase 4 showcase checkpoint/combat path mismatch")
        if checkpoint_sha256 != combat_sha256:
            raise RuntimeError("Phase 4 showcase checkpoint/combat SHA mismatch")
        if mode != "deterministic":
            raise RuntimeError("Phase 4 fair wrapper supports deterministic mode only")

        runtime = RuntimeConfig(
            env_path=env_path,
            normalizer_path=str(normalizer_path),
            resume_from=str(checkpoint),
            output_dir=str(output_dir / f"runtime_{policy_id}"),
            base_port=base_port,
            seed=0,
            time_scale=time_scale,
            device=str(device),
            game_mode="GambitVsGambit",
            player_b_bot_mode="Idle",
            num_areas=1,
        )
        (
            self.combat,
            self.normalizer,
            _,
            _,
            _,
            self.combat_device,
        ) = load_policy_and_normalizer(runtime)
        self.combat.set_ppo_mode(training=False)
        self.policy_id = policy_id
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = checkpoint_sha256
        self.combat_checkpoint = checkpoint
        self.combat_sha256 = checkpoint_sha256
        self.normalizer_path = normalizer_path
        self.normalizer_sha256 = normalizer_sha256
        self.mode = mode
        self.device = device
        self.model = self.combat
        self.payload: dict[str, Any] = {}
        self.peeker = None
        self.visible_tactic = None
        self.safety = MapIndependentSafetyLayer()
        self.search = FairReacquisitionSearch(stale_timeout_seconds=4.0)
        self.nav_hidden: dict[int, torch.Tensor] = {}
        self.combat_hidden: dict[int, torch.Tensor] = {}

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

        navigation = np.zeros((len(ids), 8), dtype=np.float32)
        navigation, _ = self.search.apply(actor, navigation, ids)
        navigation, _ = self.safety.apply(actor, navigation, ids)
        applied = compose_actions(actor, navigation, combat_actions)
        return applied, applied.copy(), applied.copy(), [False] * len(ids)

    def reset(self, agent_id: int) -> int:
        agent_id = int(agent_id)
        self.nav_hidden.pop(agent_id, None)
        self.combat_hidden.pop(agent_id, None)
        self.safety.reset(agent_id)
        self.search.reset(agent_id)
        return int(
            agent_id in self.nav_hidden
            or agent_id in self.combat_hidden
            or agent_id in self.safety.wall_side
            or agent_id in self.safety.clear_ticks
            or agent_id in self.safety.hidden_ticks
            or agent_id in self.search.search_side
            or agent_id in self.search.hidden_ticks
        )


class RuntimeFactory:
    def __new__(cls, **kwargs: Any) -> Any:
        if str(kwargs["policy_id"]).startswith("phase4_"):
            return FairWrappedPhase4Runtime(**kwargs)
        return FairPhase6Runtime(**kwargs)


release.PolicyRuntime = RuntimeFactory


def augment_outputs(output: Path) -> None:
    matches_path = output / "matches.jsonl"
    summary_path = output / "run_summary.json"
    if not matches_path.is_file() or not summary_path.is_file():
        return
    rows = [
        json.loads(line)
        for line in matches_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        duration = float(row["duration_seconds"])
        for side in ("a", "b"):
            command_magnitude = float(row[f"policy_{side}_movement_magnitude"])
            row[f"policy_{side}_path_progress"] = command_magnitude * duration
        row["path_progress_metric"] = (
            "movement-command magnitude integrated over terminal duration"
        )
        row["phase4_hidden_local45_action_or_state_update"] = False
        row["phase6_hidden_local45_combat_state_update"] = False
        row["manual_intervention"] = False
    matches_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "matches_sha256": sha256_file(matches_path),
            "d2_fair_visible_only_local45": True,
            "d2_phase4_hidden_local45_action_or_state_update": False,
            "d2_phase6_hidden_local45_combat_state_update": False,
            "d2_path_progress_metric": (
                "movement-command magnitude integrated over terminal duration"
            ),
        }
    )
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
        augment_outputs(Path(output_arg).resolve())
    return code


if __name__ == "__main__":
    raise SystemExit(main())
