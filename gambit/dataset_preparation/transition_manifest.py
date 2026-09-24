"""Phase 2 transition manifest builder.

Links consecutive encoder windows into offline RL transitions:
    state_t → action_[t,t+1] → reward_[t,t+1] → state_{t+1}
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .reward_extraction import RewardExtractor
from .action_extraction import ActionExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_transition_id(session_id: str, t: float) -> str:
    """Create a deterministic transition ID from session ID and decision time.

    Example::

        session_abc_tr005000
    """
    return f"{session_id}_tr{int(t * 1000):06d}"


def find_next_window(
    df: pd.DataFrame,
    session_id: str,
    t_next: float,
    tol: float = 0.01,
) -> pd.Series | None:
    """Find the next-state window in the same session whose t_end equals t_next.

    Returns None if no valid next window exists.
    """
    candidates = df[
        (df["session_id"] == session_id)
        & ((df["t_end"] - t_next).abs() < tol)
    ]
    if len(candidates) == 0:
        return None
    return candidates.iloc[0]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_transition_manifest(
    window_manifest_path: str | Path,
    latent_index_path: str | Path,
    action_dir: str | Path,
    output_transition_manifest_path: str | Path,
    reward_extractor: RewardExtractor,
    action_extractor: ActionExtractor | None = None,
    step_seconds: float = 1.0,
    save_actions: bool = True,
) -> pd.DataFrame:
    """Build the offline RL transition manifest.

    For each window ending at time t:
        state      = latent for window ending at t
        action     = extracted from [t, t+1]
        reward     = extracted from [t, t+1]
        next_state = latent for window ending at t+1

    Skip invalid rows:
        missing state latent
        missing next_state latent
        missing action
        invalid reward
        session boundary crossing
        split mismatch

    Returns:
        Transition DataFrame.
    """
    window_manifest_path = Path(window_manifest_path)
    action_dir = Path(action_dir)
    output_path = Path(output_transition_manifest_path)

    # Load window manifest and latent index
    window_df = pd.read_csv(window_manifest_path)
    latent_df = pd.read_csv(latent_index_path)

    # Merge latent paths into window manifest
    window_df = window_df.merge(
        latent_df[["window_id", "latent_path"]],
        on="window_id",
        how="left",
    )

    # Sort by session and time for efficient sequential processing
    window_df = window_df.sort_values(["session_id", "t_end"]).reset_index(drop=True)

    # Build a lookup: (session_id, t_end_rounded) → row index
    window_lookup: dict[tuple[str, int], int] = {}
    for idx, row in window_df.iterrows():
        key = (str(row["session_id"]), round(row["t_end"] * 1000))
        window_lookup[key] = idx

    if action_extractor is None:
        action_extractor = ActionExtractor()

    transition_rows: list[dict] = []
    skipped = {"no_latent": 0, "no_next": 0, "no_next_latent": 0, "invalid_reward": 0}

    for idx, row in window_df.iterrows():
        session_id = str(row["session_id"])
        t_end = float(row["t_end"])
        t = t_end  # Decision time is the window end time
        split = str(row["split"])

        # Check state latent exists
        state_latent_path = row.get("latent_path")
        if pd.isna(state_latent_path) or not Path(str(state_latent_path)).exists():
            skipped["no_latent"] += 1
            continue

        # Find next window (1 second ahead)
        t_next = t_end + step_seconds
        next_key = (session_id, round(t_next * 1000))
        next_idx = window_lookup.get(next_key)

        if next_idx is None:
            skipped["no_next"] += 1
            continue

        next_row = window_df.iloc[next_idx]

        # Validate same session and split
        if str(next_row["session_id"]) != session_id:
            skipped["no_next"] += 1
            continue
        if str(next_row["split"]) != split:
            skipped["no_next"] += 1
            continue

        # Check next_state latent exists
        next_latent_path = next_row.get("latent_path")
        if pd.isna(next_latent_path) or not Path(str(next_latent_path)).exists():
            skipped["no_next_latent"] += 1
            continue

        # Extract action for [t, t + step_seconds]
        action_path = action_dir / split / f"{row['window_id']}.npz"
        if save_actions:
            action_data = action_extractor.extract_for_window(
                tel_json_path=row["tel_json_path"],
                t=t,
                step_seconds=step_seconds,
            )
            action_extractor.save_action_npz(action_path, action_data)

        if not action_path.exists() and not save_actions:
            continue

        # Extract reward for [t, t + step_seconds]
        reward_data = reward_extractor.compute_for_window(
            tel_json_path=row["tel_json_path"],
            t=t,
            step_seconds=step_seconds,
        )
        reward = reward_data["reward"]

        if not isinstance(reward, (int, float)) or not __import__("math").isfinite(reward):
            skipped["invalid_reward"] += 1
            continue

        # Determine done and timeout flags
        # Check if next_state is the last window in the session
        next_next_key = (session_id, round((t_next + step_seconds) * 1000))
        is_last = next_next_key not in window_lookup

        transition_id = make_transition_id(session_id, t)

        transition_rows.append({
            "transition_id": transition_id,
            "player_id": str(row["player_id"]),
            "session_id": session_id,
            "t": t,
            "state_window_id": str(row["window_id"]),
            "next_state_window_id": str(next_row["window_id"]),
            "state_latent_path": str(state_latent_path),
            "next_state_latent_path": str(next_latent_path),
            "action_path": str(action_path),
            "reward": reward,
            "done": is_last,
            "timeout": is_last,  # First version: all terminal = timeout
            "split": split,
            # Optional diagnostic columns
            "reward_hit": reward_data.get("reward_hit", 0.0),
            "reward_damage": reward_data.get("reward_damage", 0.0),
            "reward_kill": reward_data.get("reward_kill", 0.0),
            "reward_death": reward_data.get("reward_death", 0.0),
            "action_count": action_data.get("action_vector", [0] * 14)[13]
                if save_actions else 0,
            "shoot_count": action_data.get("action_vector", [0] * 14)[7]
                if save_actions else 0,
        })

    logger.info(
        "Built %d transitions (skipped: %s)",
        len(transition_rows),
        skipped,
    )

    if not transition_rows:
        raise ValueError(
            f"No valid transitions could be built. Skipped: {skipped}"
        )

    df = pd.DataFrame(transition_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df
