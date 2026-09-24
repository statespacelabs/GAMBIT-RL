"""Aggregate Phase 3AC weapon-state debug JSONL into rollout metrics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def aggregate_weapon_state_metrics(log_path: str | Path) -> dict[str, float]:
    path = Path(log_path)
    if not path.is_file():
        return {}
    counts: Counter[str] = Counter()
    resets = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") == "blocked_fire":
            reason = str(rec.get("blocked_fire_reason", "UNKNOWN"))
            counts[reason] += 1
        elif rec.get("event") == "weapon_reset":
            resets += 1
    out: dict[str, float] = {
        "weapon/blocked_fire_total": float(sum(counts.values())),
        "weapon/reset_events": float(resets),
    }
    for reason, n in counts.items():
        key = f"weapon/blocked_{reason.lower()}"
        out[key] = float(n)
    if counts:
        top_reason, top_n = counts.most_common(1)[0]
        out["weapon/blocked_top_count"] = float(top_n)
    return out
