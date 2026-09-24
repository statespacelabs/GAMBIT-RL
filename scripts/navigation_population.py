#!/usr/bin/env python3
"""Deterministic anchored-league contracts shared by Goal 12 runners/tests."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from scripts.navigation_ppo import LEAGUE_LANES

OPPONENT_DISTRIBUTION = {
    "frozen_phase4_roster": 0.30,
    "scripted_camper_hold_search": 0.20,
    "kiter_cover_change": 0.15,
    "historical_phase5": 0.15,
    "rating_near_peer": 0.10,
    "specialist_exploiter": 0.05,
    "mirror": 0.05,
}


@dataclass(frozen=True)
class WindowSpec:
    index: int
    category: str
    opponent_key: str
    preset: str
    neural: bool


WINDOW_SCHEDULE: tuple[WindowSpec, ...] = (
    WindowSpec(0, "frozen_phase4_roster", "stationary_unarmed", "peek_p0", False),
    WindowSpec(1, "scripted_camper_hold_search", "slow_hold_angle", "peek_p1", False),
    WindowSpec(2, "kiter_cover_change", "cover_user_armed", "peek_p4", False),
    WindowSpec(3, "historical_phase5", "dagger_navigator", "peek_p4", True),
    WindowSpec(4, "frozen_phase4_roster", "phase4_primary_combat_parent", "peek_p2", True),
    WindowSpec(5, "scripted_camper_hold_search", "accurate_hold_angle", "peek_p2", False),
    WindowSpec(6, "historical_phase5", "hunter_champion", "peek_p4", True),
    WindowSpec(7, "kiter_cover_change", "strafe_duelist", "peek_p3", False),
    WindowSpec(8, "frozen_phase4_roster", "phase4_primary_combat_parent", "peek_p3", True),
    WindowSpec(9, "historical_phase5", "reacquirer", "peek_p4", True),
    WindowSpec(10, "rating_near_peer", "rating_near", "peek_p4", True),
    WindowSpec(11, "frozen_phase4_roster", "stationary_easy_anchor", "peek_p0", False),
    WindowSpec(12, "specialist_exploiter", "specialist", "peek_p4", True),
    WindowSpec(13, "scripted_camper_hold_search", "camper", "peek_p2", False),
    WindowSpec(14, "rating_near_peer", "rating_near", "peek_p4", True),
    WindowSpec(15, "kiter_cover_change", "cover_changer", "peek_p4", False),
    WindowSpec(16, "frozen_phase4_roster", "hold_angle_easy_anchor", "peek_p1", False),
    WindowSpec(17, "mirror", "mirror", "peek_p4", True),
    WindowSpec(18, "scripted_camper_hold_search", "search_suite", "peek_p3", False),
    WindowSpec(19, "frozen_phase4_roster", "stationary_easy_anchor", "peek_p0", False),
)


def validate_schedule() -> None:
    if len(LEAGUE_LANES) != 8 or [lane.gpu for lane in LEAGUE_LANES] != list(range(8)):
        raise RuntimeError("league requires exactly one named lane per GPU")
    if len(WINDOW_SCHEDULE) != 20:
        raise RuntimeError("league schedule must contain 20 frozen windows")
    counts = Counter(window.category for window in WINDOW_SCHEDULE)
    for category, probability in OPPONENT_DISTRIBUTION.items():
        if counts[category] != int(round(probability * len(WINDOW_SCHEDULE))):
            raise RuntimeError(f"opponent schedule mismatch for {category}")
    if any(window.index != index for index, window in enumerate(WINDOW_SCHEDULE)):
        raise RuntimeError("window indexes must be contiguous")


def adaptive_sampling_weights(
    opponents: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Double sampling mass for opponents in the informative 30–70% band."""
    rows = []
    for opponent in opponents:
        win_rate = float(opponent.get("learner_win_rate", 0.5))
        multiplier = 2.0 if 0.30 <= win_rate <= 0.70 else 1.0
        rows.append({**opponent, "adaptive_multiplier": multiplier})
    total = sum(float(row.get("base_weight", 1.0)) * row["adaptive_multiplier"] for row in rows)
    for row in rows:
        row["sampling_probability"] = (
            float(row.get("base_weight", 1.0)) * row["adaptive_multiplier"] / max(total, 1e-9)
        )
    return rows


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def update_elo(rating: float, opponent_rating: float, score: float, k: float = 24.0) -> float:
    if not (0.0 <= score <= 1.0) or not all(math.isfinite(v) for v in (rating, opponent_rating, k)):
        raise ValueError("invalid Elo update")
    return rating + k * (score - expected_score(rating, opponent_rating))


def score_window(result: dict[str, Any]) -> float:
    terminals = result.get("reward_terminal_counts") or {}
    wins = int(terminals.get("kill", 0))
    losses = int(terminals.get("death", 0))
    timeouts = max(0, int(terminals.get("timeout", 0)))
    if wins + losses + timeouts:
        return wins / float(wins + losses + timeouts)
    damage = int((result.get("peek_metrics") or {}).get("damage_positive_peeks", 0))
    return 0.55 if damage > 0 else 0.50


def hard_stop_failures(result: dict[str, Any]) -> list[str]:
    failures = []
    finite_fields = (
        "policy_mean_look_saturation_rate", "critic_prefix_max_abs_error",
        "max_navigation_shaping_abs_per_life",
    )
    if any(not math.isfinite(float(result.get(name, math.nan))) for name in finite_fields):
        failures.append("nan_or_inf")
    if int(result.get("weapon_ownership_mismatch_count", 1)):
        failures.append("weapon_ownership_mismatch")
    if int(result.get("weapon_fire_mismatch_count", 1)):
        failures.append("weapon_fire_mismatch")
    if float(result.get("policy_mean_look_saturation_rate", 1.0)) > 0.10:
        failures.append("policy_mean_look_saturation")
    if int(result.get("hidden_state_leakage_events", 1)):
        failures.append("hidden_state_leakage")
    if bool(result.get("zero_fire_collapse", True)):
        failures.append("zero_fire_collapse")
    if result.get("opponent_frozen_within_window") is not True:
        failures.append("opponent_not_frozen")
    if result.get("simultaneous_opponent_updates") is not False:
        failures.append("simultaneous_self_play")
    return failures


def lane_manifest_rows() -> list[dict[str, Any]]:
    return [asdict(lane) for lane in LEAGUE_LANES]


validate_schedule()
