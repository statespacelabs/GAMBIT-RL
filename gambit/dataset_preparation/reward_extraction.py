"""Phase 2 reward extraction: scalar rewards from raw gameplay JSON.

Rewards are computed from the interval [t, t+1]. This module only uses
structured trajectory fields (hits, kills, damage, deaths, action counts)
and does not use rendered video.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default reward weights
# ---------------------------------------------------------------------------

DEFAULT_REWARD_CONFIG = {
    "kill_weight": 10.0,
    "hit_weight": 1.0,
    "damage_weight": 0.1,
    "death_penalty": -5.0,
    "damage_taken_weight": -0.1,
    "engagement_shaping": 0.05,
    "waste_penalty": -0.01,
    "shoot_action_type": 0,
}


# ---------------------------------------------------------------------------
# Interval reward computation
# ---------------------------------------------------------------------------

def compute_interval_reward(
    frames: list[dict],
    t_start: float,
    t_end: float,
    reward_config: dict | None = None,
) -> dict:
    """Compute reward components from raw frames in [t_start, t_end].

    First version reward formula:
        + kill / target destroyed
        + hit
        + damage dealt, if available
        - damage taken, if available
        - death, if available
        - missed shots or action spam
        + targeted action / engagement shaping

    The function gracefully handles missing fields by using zeros.

    Returns:
        Dictionary with reward and component breakdown.
    """
    if reward_config is None:
        reward_config = DEFAULT_REWARD_CONFIG

    kill_w = reward_config.get("kill_weight", 10.0)
    hit_w = reward_config.get("hit_weight", 1.0)
    damage_w = reward_config.get("damage_weight", 0.1)
    death_p = reward_config.get("death_penalty", -5.0)
    damage_taken_w = reward_config.get("damage_taken_weight", -0.1)
    engagement_w = reward_config.get("engagement_shaping", 0.05)
    waste_p = reward_config.get("waste_penalty", -0.01)
    shoot_type = reward_config.get("shoot_action_type", 0)

    # Filter frames to interval
    interval_frames = [
        fr for fr in frames
        if t_start <= float(fr.get("time", -1.0)) < t_end
    ]

    # Accumulate signals
    kills = 0
    hits = 0
    damage_dealt = 0.0
    deaths = 0
    damage_taken = 0.0
    targeted_actions = 0
    shoot_count = 0
    total_actions = 0

    for fr in interval_frames:
        # Frame-level stats if available
        kills += int(fr.get("kills", 0))
        hits += int(fr.get("hits", 0))
        damage_dealt += float(fr.get("damage_dealt", 0.0))
        deaths += int(fr.get("deaths", 0))
        damage_taken += float(fr.get("damage_taken", 0.0))

        actions = fr.get("actions", [])
        total_actions += len(actions)

        for act in actions:
            atype = int(act.get("action_type", -1))
            target_id = int(act.get("target_id", -1))

            if atype == shoot_type:
                shoot_count += 1
            if target_id >= 0:
                targeted_actions += 1
                # Count hits from targeted shoot actions
                if atype == shoot_type:
                    hits += 1

    # Compute reward components
    reward_kill = kills * kill_w
    reward_hit = hits * hit_w
    reward_damage = damage_dealt * damage_w
    reward_death = deaths * death_p
    reward_damage_taken = damage_taken * damage_taken_w
    reward_engagement = targeted_actions * engagement_w

    # Waste penalty: penalize untargeted shots
    untargeted_shots = max(0, shoot_count - targeted_actions)
    reward_waste = untargeted_shots * waste_p

    total_reward = (
        reward_kill
        + reward_hit
        + reward_damage
        + reward_death
        + reward_damage_taken
        + reward_engagement
        + reward_waste
    )

    return {
        "reward": float(total_reward),
        "reward_hit": float(reward_hit),
        "reward_damage": float(reward_damage),
        "reward_kill": float(reward_kill),
        "reward_death": float(reward_death),
        "reward_engagement": float(reward_engagement),
        "reward_waste": float(reward_waste),
    }


# ---------------------------------------------------------------------------
# RewardExtractor class
# ---------------------------------------------------------------------------

class RewardExtractor:
    """Computes scalar rewards from raw gameplay analytics JSON.

    Rewards are computed from the interval [t, t+1]. This module must not use
    rendered video. It should only use structured trajectory fields such as
    hits, kills, damage, deaths, target destruction, action counts, and
    engagement indicators.
    """

    def __init__(self, reward_config: dict | None = None):
        self.reward_config = reward_config or DEFAULT_REWARD_CONFIG
        self._json_cache: dict[str, dict] = {}

    def _load_json(self, tel_json_path: str | Path) -> dict:
        key = str(tel_json_path)
        if key not in self._json_cache:
            with open(key, "r", encoding="utf-8") as f:
                self._json_cache[key] = json.load(f)
        return self._json_cache[key]

    def clear_cache(self) -> None:
        self._json_cache.clear()

    def compute_for_window(
        self,
        tel_json_path: str | Path,
        t: float,
        step_seconds: float = 1.0,
    ) -> dict:
        """Compute reward and reward components for interval [t, t+step_seconds].

        Returns:
            Dict containing scalar reward and component breakdown.
        """
        chunk = self._load_json(tel_json_path)
        frames = chunk.get("frames", [])

        return compute_interval_reward(
            frames=frames,
            t_start=t,
            t_end=t + step_seconds,
            reward_config=self.reward_config,
        )


# ---------------------------------------------------------------------------
# Reward statistics
# ---------------------------------------------------------------------------

def fit_reward_stats(
    transition_manifest_path: str | Path,
    output_stats_path: str | Path,
) -> dict:
    """Compute reward normalization statistics from the training split.

    Stores mean, std, min, max, and percentiles.

    Returns:
        Reward statistics dictionary.
    """
    df = pd.read_csv(transition_manifest_path)
    train_df = df[df["split"] == "train"]

    if len(train_df) == 0:
        raise ValueError("No training transitions for reward statistics.")

    rewards = train_df["reward"].values.astype(np.float64)

    stats = {
        "mean": float(np.mean(rewards)),
        "std": float(max(np.std(rewards), 1e-6)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "p5": float(np.percentile(rewards, 5)),
        "p25": float(np.percentile(rewards, 25)),
        "p50": float(np.percentile(rewards, 50)),
        "p75": float(np.percentile(rewards, 75)),
        "p95": float(np.percentile(rewards, 95)),
        "count": int(len(rewards)),
    }

    output_path = Path(output_stats_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json as json_mod
    with open(output_path, "w", encoding="utf-8") as f:
        json_mod.dump(stats, f, indent=2)

    return stats
