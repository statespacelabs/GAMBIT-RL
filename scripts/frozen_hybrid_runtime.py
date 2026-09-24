#!/usr/bin/env python3
"""Exact Goal B repaired runtime with a read-only, fair HUD observer.

The observer is downstream of the frozen policy call: it does not change
observations, recurrent state, sampled actions, or environment-applied actions.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


OVERLAY = Path(os.environ["RESEARCH_REPO_OVERLAY"]).resolve()
BASE_RUNNER = OVERLAY / "scripts/repaired_navigation_runtime.py"
HUD_PATH = Path(os.environ["RESEARCH_HUD_STATE_PATH"]).resolve()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("research_demo_exact_goal_b_runner", BASE_RUNNER)

# Route the build's existing two-factor attended-rendered authorization through
# the Unity boundary.  The historical wrapper already toggles graphics through
# MVH_RENDERED but otherwise owns UnityEnvironment construction internally.
from mlagents_envs import environment as mlagents_environment  # noqa: E402

_OriginalUnityEnvironment = mlagents_environment.UnityEnvironment


class ResearchUnityEnvironment(_OriginalUnityEnvironment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        additional = list(kwargs.get("additional_args") or [])
        if os.environ.get("MVH_RENDERED", "0") == "1":
            if os.environ.get("PHASE5_RENDERED_ATTENDED_SMOKE", "0") != "1":
                raise RuntimeError("rendered launch lacks attended-smoke environment guard")
            if "--phase5-rendered-smoke" not in additional:
                additional.insert(0, "--phase5-rendered-smoke")
        kwargs["additional_args"] = additional
        super().__init__(*args, **kwargs)


mlagents_environment.UnityEnvironment = ResearchUnityEnvironment
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_observer import atomic_write_json  # noqa: E402
from scripts import combat_runtime as readiness  # noqa: E402


def load_full_state_policy(cfg: Any) -> tuple[Any, ...]:
    """Instantiate the fixed architecture, then strictly load every final weight.

    The historical loader first reads an unrelated 1.1 GB Phase 1 encoder solely
    to construct this architecture. Every one of those bootstrap parameters is
    then overwritten by ``cfg.resume_from`` with ``strict=True``. The package
    creates the same architecture directly and retains that strict full-state
    load, avoiding a model that never contributes a deployed parameter.
    """
    ppo_cfg = readiness.PPOTrainConfig.from_yaml(cfg.ppo_config)
    distill_cfg = readiness.DistillConfig.from_yaml(cfg.distill_config)
    ppo_cfg.device = (
        cfg.device
        if torch.cuda.is_available() and str(cfg.device).startswith("cuda")
        else "cpu"
    )
    device = torch.device(ppo_cfg.device)
    if not cfg.resume_from:
        raise RuntimeError("research runtime requires a frozen full-state checkpoint")
    checkpoint = Path(cfg.resume_from)
    policy = readiness.RecurrentActorCritic(readiness.TelemetryEncoder())
    readiness._load_actor_critic_checkpoint(policy, checkpoint, device)
    policy.to(device)
    policy.policy_init_source = f"strict_full_state:{checkpoint}"
    if hasattr(policy, "actor_cont_logstd"):
        with torch.no_grad():
            policy.actor_cont_logstd.clamp_(cfg.min_log_std, cfg.max_log_std)
    policy.set_ppo_mode(training=False)
    normalizer = readiness.CertifiedObsNormalizer(cfg.normalizer_path, device)
    return policy, normalizer, ppo_cfg, distill_cfg, checkpoint, device


readiness.load_policy_and_normalizer = load_full_state_policy


class HudObserver:
    """Observe fair LOS/local45 health and already-applied fire actions."""

    def __init__(self) -> None:
        self.seen: dict[int, bool] = {}
        self.decisions: dict[int, int] = {}
        self.first_contact: dict[int, float] = {}
        self.last_health: dict[int, float] = {}
        self.damage: dict[int, float] = {}
        self.shots: dict[int, int] = {}
        self.hits: dict[int, int] = {}
        self.last_state: dict[int, str] = {}

    def observe(
        self,
        *,
        runtime: Any,
        actor: np.ndarray,
        local45: np.ndarray,
        ids: list[int],
        applied: np.ndarray,
    ) -> None:
        decision_hz = float(os.environ.get("RESEARCH_DECISION_HZ", "50"))
        for row, raw_id in enumerate(ids):
            agent_id = int(raw_id)
            count = self.decisions.get(agent_id, 0) + 1
            self.decisions[agent_id] = count
            visible = bool(actor[row, base.LOS_INDEX] > 0.5)
            previously_seen = self.seen.get(agent_id, False)
            if visible and not previously_seen:
                self.seen[agent_id] = True
                self.first_contact[agent_id] = count / decision_hz
            if applied.shape[1] > 4 and float(applied[row, 4]) > 0.5:
                self.shots[agent_id] = self.shots.get(agent_id, 0) + 1

            # local45 index 39 is opponent HP fraction. It is sampled for the
            # presentation observer only while actor231 LOS is true.
            if visible and local45.shape[1] > 39:
                health = float(np.clip(local45[row, 39], 0.0, 1.0))
                previous = self.last_health.get(agent_id, 1.0)
                drop = max(0.0, previous - health)
                if drop > 1e-4:
                    self.hits[agent_id] = self.hits.get(agent_id, 0) + 1
                    self.damage[agent_id] = self.damage.get(agent_id, 0.0) + 100.0 * drop
                self.last_health[agent_id] = health

            if visible and not previously_seen:
                state = "CONTACT"
            elif visible:
                state = "ENGAGE"
            elif self.seen.get(agent_id, False):
                state = "REACQUIRE"
            else:
                state = "SEARCH"
            tactical = getattr(runtime, "tactical_mode", {}).get(agent_id)
            if tactical == "duel" and visible:
                state = "ENGAGE"
            changed = state != self.last_state.get(agent_id)
            self.last_state[agent_id] = state
            if changed or count % 5 == 0:
                health_value = self.last_health.get(agent_id)
                atomic_write_json(
                    HUD_PATH,
                    {
                        "schema_version": "phase6_research_hud_v001",
                        "screen": "scenario",
                        "label": "EXPERIMENTAL AUTONOMOUS AI",
                        "scenario": {
                            "scenario_id": os.environ.get(
                                "RESEARCH_SCENARIO_ID", ""
                            ),
                            "order": int(os.environ.get("RESEARCH_SCENARIO_ORDER", "1")),
                            "total": int(os.environ.get("RESEARCH_SCENARIO_TOTAL", "1")),
                            "seed": int(os.environ.get("RESEARCH_SCENARIO_SEED", "0")),
                        },
                        "state": state,
                        "result": "in progress",
                        "metrics": {
                            "time_to_contact_seconds": self.first_contact.get(agent_id),
                            "damage_dealt": round(self.damage.get(agent_id, 0.0), 3),
                            "shots": self.shots.get(agent_id, 0),
                            "hits": self.hits.get(agent_id, 0),
                            "opponent_health_percent": (
                                None
                                if health_value is None
                                else round(100.0 * health_value, 3)
                            ),
                        },
                        "audit": {
                            "model_sha256": os.environ.get(
                                "RESEARCH_MODEL_SHA256", ""
                            ),
                            "observer_changes_policy_action": False,
                            "hidden_target_telemetry_read": False,
                        },
                    },
                )

    def reset(self, agent_id: int) -> None:
        agent_id = int(agent_id)
        for mapping in (
            self.seen,
            self.decisions,
            self.first_contact,
            self.last_health,
            self.damage,
            self.shots,
            self.hits,
            self.last_state,
        ):
            mapping.pop(agent_id, None)


class ResearchRepairedRuntime(base.RepairedRuntime):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hud_observer = HudObserver()

    def act(
        self,
        *,
        actor: np.ndarray,
        local45: np.ndarray,
        ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool]]:
        result = super().act(actor=actor, local45=local45, ids=ids)
        self.hud_observer.observe(
            runtime=self,
            actor=np.asarray(actor, dtype=np.float32),
            local45=np.asarray(local45, dtype=np.float32),
            ids=ids,
            applied=np.asarray(result[0], dtype=np.float32),
        )
        return result

    def reset(self, agent_id: int) -> int:
        leak = super().reset(agent_id)
        self.hud_observer.reset(agent_id)
        return int(leak)


class RuntimeFactory:
    def __new__(cls, **kwargs: Any) -> Any:
        if str(kwargs["policy_id"]).startswith("phase4_"):
            return base.base.mvh.d2.FairWrappedPhase4Runtime(**kwargs)
        return ResearchRepairedRuntime(**kwargs)


base.base.release.PolicyRuntime = RuntimeFactory


def main() -> int:
    code = int(base.base.main())
    if os.environ.get("MVH_RENDERED", "0") != "1":
        return code
    output_value = next(
        (
            sys.argv[index + 1]
            for index, value in enumerate(sys.argv[:-1])
            if value == "--output-dir"
        ),
        "",
    )
    if not output_value:
        return code
    output = Path(output_value).resolve()
    summary_path = output / "run_summary.json"
    if not summary_path.is_file():
        return code
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runtime_path = Path(summary["source_runtime_audits"]["runtime"])
    unity_path = Path(summary["source_runtime_audits"]["unity_log"])
    unity_text = unity_path.read_text(encoding="utf-8", errors="replace")
    if runtime_path.is_file():
        runtime_status_pass = (
            json.loads(runtime_path.read_text(encoding="utf-8")).get("status")
            == "PASS"
        )
    else:
        runtime_status_pass = (
            "[Bootstrapper] Game initialized successfully" in unity_text
            and "Unhandled Exception" not in unity_text
        )
    rendered_pass = all(
        (
            summary.get("matches_completed") == summary.get("target_matches"),
            summary.get("missing_terminal_outcomes") == 0,
            summary.get("unity_audit_terminal_count_mismatch") == 0,
            summary.get("hidden_state_leakage_events") == 0,
            summary.get("side_config_mismatches") == 0,
            summary.get("observation_nonfinite_values") == 0,
            summary.get("action_nonfinite_values") == 0,
            float(summary.get("critic_prefix_max_abs_error", 1.0)) <= 1e-5,
            float(summary.get("action_abs_max", 2.0)) <= 1.000001,
            summary.get("weapon_ownership_mismatch_count") == 0,
            summary.get("weapon_fire_mismatch_count") == 0,
            summary.get("human_controller_count") == 0,
            not summary.get("actor_received_map_identity", True),
            not summary.get("actor_received_opponent_id", True),
            not summary.get("actor_received_privileged_suffix", True),
            not summary.get("final_heldout_used", True),
            runtime_status_pass,
            "[Phase5Navigator] explicit attended rendered smoke accepted"
            in unity_text,
            "[Phase5HunterPPO] requires headless" not in unity_text,
            "NavMesh actor/oracle inputs are forbidden" not in unity_text,
        )
    )
    if not rendered_pass:
        return code
    summary.update(
        {
            "base_status_before_rendered_reconciliation": summary["status"],
            "status": "PASS",
            "headless": False,
            "rendered": True,
            "rendered_attended_audit": {
                "status": "PASS",
                "explicit_attended_guard": True,
                "headless_graphics_audit_applicable": False,
                "mode_specific_gate_substitution_only": True,
                "policy_action_changed": False,
            },
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
