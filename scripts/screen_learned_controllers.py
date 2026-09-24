#!/usr/bin/env python3
"""Multi-family, no-update match adapter for CUDA bot-bank screening."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "scripts/frozen_hybrid_runtime.py"
spec = importlib.util.spec_from_file_location("phase6_bot_screen_research_source", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load research runtime: {SOURCE}")
research = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = research
spec.loader.exec_module(research)

from scripts.navigation_policy import (  # noqa: E402
    FairReacquisitionSearch,
    LOS_INDEX,
    MapIndependentSafetyLayer,
    compose_actions,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ScreenPhase4Runtime(research.base.base.mvh.d2.FairWrappedPhase4Runtime):
    """Certified fair hidden search plus explicit fixed-seed hybrid sampling."""

    def __init__(self, **kwargs: Any) -> None:
        requested_mode = str(kwargs["mode"])
        seed = int(kwargs["seed"])
        seed_offset = int(kwargs["seed_offset"])
        if requested_mode not in {"deterministic", "stochastic"}:
            raise RuntimeError(f"unsupported Phase 4 screen mode: {requested_mode}")
        base_kwargs = dict(kwargs)
        base_kwargs["mode"] = "deterministic"
        super().__init__(**base_kwargs)
        self.mode = requested_mode
        self.screen_generator = torch.Generator(device=self.combat_device)
        self.screen_generator.manual_seed(seed + seed_offset)

    def _visible_actions(
        self,
        local45: np.ndarray,
        ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observations = torch.as_tensor(
            local45, dtype=torch.float32, device=self.combat_device
        ).unsqueeze(1)
        normalized = self.normalizer.normalize_tensor(observations)
        hidden = torch.cat(
            [
                self.combat_hidden.get(
                    agent_id, self.combat.init_hidden(1, self.combat_device)
                )
                for agent_id in ids
            ],
            dim=1,
        )
        with torch.no_grad():
            distribution, values, next_hidden = self.combat(normalized, hidden)
            continuous_mean = distribution.cont_mean[:, 0]
            continuous_std = distribution.cont_logstd[:, 0].exp()
            binary_logits = distribution.binary_logits[:, 0]
            if self.mode == "stochastic":
                continuous_raw = continuous_mean + torch.randn(
                    continuous_mean.shape,
                    device=continuous_mean.device,
                    dtype=continuous_mean.dtype,
                    generator=self.screen_generator,
                ) * continuous_std
                probabilities = torch.sigmoid(binary_logits)
                binary = (
                    torch.rand(
                        probabilities.shape,
                        device=probabilities.device,
                        dtype=probabilities.dtype,
                        generator=self.screen_generator,
                    )
                    < probabilities
                ).to(continuous_mean.dtype)
            else:
                continuous_raw = continuous_mean
                binary = (binary_logits > 0).to(continuous_mean.dtype)
            continuous_applied = continuous_raw.clamp(-1.0, 1.0)
            applied = torch.cat((continuous_applied, binary), dim=-1)
        if (
            not torch.isfinite(values).all()
            or not torch.isfinite(next_hidden).all()
            or not torch.isfinite(applied).all()
        ):
            raise RuntimeError("Phase 4 screening inference produced NaN/Inf")
        for index, agent_id in enumerate(ids):
            self.combat_hidden[agent_id] = next_hidden[:, index : index + 1].detach()
        return (
            applied.detach().cpu().numpy().astype(np.float32),
            continuous_mean.detach().cpu().numpy().astype(np.float32),
            continuous_raw.detach().cpu().numpy().astype(np.float32),
        )

    def act(
        self, *, actor: np.ndarray, local45: np.ndarray, ids: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool]]:
        actor = np.asarray(actor, dtype=np.float32)
        local45 = np.asarray(local45, dtype=np.float32)
        combat = np.zeros((len(ids), 8), dtype=np.float32)
        means = np.zeros((len(ids), 4), dtype=np.float32)
        raws = np.zeros((len(ids), 4), dtype=np.float32)
        visible_rows = np.flatnonzero(actor[:, LOS_INDEX] > 0.5)
        if len(visible_rows):
            visible_ids = [ids[int(index)] for index in visible_rows]
            actions, visible_means, visible_raws = self._visible_actions(
                local45[visible_rows], visible_ids
            )
            combat[visible_rows] = actions
            means[visible_rows] = visible_means
            raws[visible_rows] = visible_raws
        navigation = np.zeros((len(ids), 8), dtype=np.float32)
        navigation, _ = self.search.apply(actor, navigation, ids)
        navigation, _ = self.safety.apply(actor, navigation, ids)
        applied = compose_actions(actor, navigation, combat)
        return applied, means, raws, [False] * len(ids)


class CombatOnlyRuntime(ScreenPhase4Runtime):
    """Original visible combat expert with no hidden-target search policy."""

    def act(
        self, *, actor: np.ndarray, local45: np.ndarray, ids: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool]]:
        actor = np.asarray(actor, dtype=np.float32)
        local45 = np.asarray(local45, dtype=np.float32)
        applied = np.zeros((len(ids), 8), dtype=np.float32)
        means = np.zeros((len(ids), 4), dtype=np.float32)
        raws = np.zeros((len(ids), 4), dtype=np.float32)
        visible_rows = np.flatnonzero(actor[:, LOS_INDEX] > 0.5)
        if len(visible_rows):
            visible_ids = [ids[int(index)] for index in visible_rows]
            actions, visible_means, visible_raws = self._visible_actions(
                local45[visible_rows], visible_ids
            )
            applied[visible_rows] = actions
            means[visible_rows] = visible_means
            raws[visible_rows] = visible_raws
        return applied, means, raws, [False] * len(ids)


class ScriptedAnchorRuntime:
    """Simple actor-only diagnostic lower bounds; never reads hidden target state."""

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
        **_: Any,
    ) -> None:
        for label, raw_path, expected in (
            ("anchor specification", checkpoint, checkpoint_sha256),
            ("combat reference", combat_checkpoint, combat_sha256),
            ("normalizer reference", normalizer_path, normalizer_sha256),
        ):
            path = Path(raw_path)
            if sha256(path) != expected:
                raise RuntimeError(f"{policy_id} {label} hash mismatch")
        if mode not in {"deterministic", "stochastic"}:
            raise RuntimeError(f"unsupported anchor mode: {mode}")
        self.policy_id = policy_id
        self.mode = mode
        self.search = FairReacquisitionSearch(stale_timeout_seconds=4.0)
        self.safety = MapIndependentSafetyLayer()

    def act(
        self, *, actor: np.ndarray, local45: np.ndarray, ids: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool]]:
        del local45
        actor = np.asarray(actor, dtype=np.float32)
        applied = np.zeros((len(ids), 8), dtype=np.float32)
        if self.policy_id == "screen_anchor_search_no_fire":
            applied, _ = self.search.apply(actor, applied, ids)
            applied, _ = self.safety.apply(actor, applied, ids)
        continuous = applied[:, :4].copy()
        return applied, continuous, continuous.copy(), [False] * len(ids)

    def reset(self, agent_id: int) -> int:
        agent_id = int(agent_id)
        self.search.reset(agent_id)
        self.safety.reset(agent_id)
        return int(
            agent_id in self.search.search_side
            or agent_id in self.search.hidden_ticks
            or agent_id in self.safety.wall_side
            or agent_id in self.safety.clear_ticks
            or agent_id in self.safety.hidden_ticks
        )


class ScreenRuntimeFactory:
    def __new__(cls, **kwargs: Any) -> Any:
        policy_id = str(kwargs["policy_id"])
        if policy_id == "screen_repaired_hunter":
            return research.ResearchRepairedRuntime(**kwargs)
        if policy_id == "screen_original_combat_expert":
            return CombatOnlyRuntime(**kwargs)
        if policy_id.startswith("screen_phase4_") or policy_id.startswith("phase4_"):
            return ScreenPhase4Runtime(**kwargs)
        if policy_id.startswith("screen_anchor_"):
            return ScriptedAnchorRuntime(**kwargs)
        return research.base.base.mvh.d2.FairPhase6Runtime(**kwargs)


research.base.base.release.PolicyRuntime = ScreenRuntimeFactory


if __name__ == "__main__":
    os.environ.setdefault("MVH_RENDERED", "0")
    raise SystemExit(research.main())
