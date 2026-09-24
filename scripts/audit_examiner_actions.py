#!/usr/bin/env python3
"""Read-only action audit around the frozen CUDA examiner match runtime."""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(__file__).resolve().with_name("examiner_match.py")
spec = importlib.util.spec_from_file_location(
    "phase6_human_demo_examiner_source", SOURCE
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import examiner adapter: {SOURCE}")
examiner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = examiner
spec.loader.exec_module(examiner)

BaseRuntimeFactory = examiner.tournament.PolicyRuntime
AUDIT: dict[str, dict[str, Any]] = {}


def _state(policy_id: str) -> dict[str, Any]:
    if policy_id not in AUDIT:
        AUDIT[policy_id] = {
            "action_rows": 0,
            "requested_fire_steps": 0,
            "applied_action_sha256": hashlib.sha256(),
            "reset_calls": 0,
            "reset_residual_count": 0,
        }
    return AUDIT[policy_id]


class AuditedRuntime:
    """Transparent composition wrapper; returned actions are never modified."""

    def __init__(self, **kwargs: Any) -> None:
        self.policy_id = str(kwargs["policy_id"])
        self.inner = BaseRuntimeFactory(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def act(
        self,
        *,
        actor: np.ndarray,
        local45: np.ndarray,
        ids: list[int],
    ) -> Any:
        result = self.inner.act(actor=actor, local45=local45, ids=ids)
        applied = np.asarray(result[0], dtype=np.float32)
        state = _state(self.policy_id)
        state["action_rows"] += int(applied.shape[0])
        state["requested_fire_steps"] += int(np.count_nonzero(applied[:, 4] > 0.5))
        state["applied_action_sha256"].update(
            np.ascontiguousarray(applied, dtype="<f4").tobytes()
        )
        return result

    def reset(self, agent_id: int) -> int:
        state = _state(self.policy_id)
        state["reset_calls"] += 1
        residual = int(self.inner.reset(agent_id))
        state["reset_residual_count"] += int(bool(residual))
        return residual


def _write_audit() -> None:
    raw_path = os.environ.get("PHASE6_HUMAN_DEMO_ACTION_AUDIT_PATH", "")
    if not raw_path:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "phase6_human_demo_action_audit_v001",
        "policies": {
            policy_id: {
                **{
                    key: value
                    for key, value in state.items()
                    if key != "applied_action_sha256"
                },
                "applied_action_sha256": state["applied_action_sha256"].hexdigest(),
            }
            for policy_id, state in sorted(AUDIT.items())
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


atexit.register(_write_audit)
examiner.tournament.PolicyRuntime = AuditedRuntime


if __name__ == "__main__":
    os.environ.setdefault("MVH_RENDERED", "0")
    raise SystemExit(examiner.tournament.main())
