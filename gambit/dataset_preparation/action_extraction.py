"""Phase 2 action extraction: compact 1-second action vectors from raw JSON.

For a state window ending at time t, this module reads the raw frames/actions
in interval [t, t+1] and produces a fixed-size action_vector.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .schemas import ActionSchema, DEFAULT_ACTION_SCHEMA


# ---------------------------------------------------------------------------
# Interval extraction
# ---------------------------------------------------------------------------


def extract_interval_actions(
    frames: list[dict],
    t_start: float,
    t_end: float,
    action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
    shoot_action_type: int = 0,
    reload_action_type: int = 5,
    jump_action_type: int = 6,
    crouch_action_type: int = 7,
) -> np.ndarray:
    """Convert raw frame-level actions into one compact action vector.

    Filters frames to [t_start, t_end], then computes summary statistics
    across all frames in the interval.

    Returns:
        action_vector as float32 array of shape [action_schema.dim].
    """
    # Filter frames to interval
    interval_frames = [
        fr for fr in frames if t_start <= float(fr.get("time", -1.0)) < t_end
    ]

    n_frames = len(interval_frames)
    action_vector = np.zeros(action_schema.dim, dtype=np.float32)

    if n_frames == 0:
        return action_vector

    # Accumulate per-frame values
    move_x_vals: list[float] = []
    move_y_vals: list[float] = []
    look_dx_vals: list[float] = []
    look_dy_vals: list[float] = []

    shoot_count = 0
    shoot_pressed = False
    reload_pressed = False
    jump_pressed = False
    crouch_pressed = False
    targeted_action_count = 0
    total_action_count = 0

    prev_yaw: float | None = None
    prev_pitch: float | None = None

    for fr in interval_frames:
        # Movement from velocity
        vel = fr.get("velocity", {})
        move_x_vals.append(float(vel.get("x", 0.0)))
        move_y_vals.append(float(vel.get("z", 0.0)))  # z is forward in Unity

        # Look deltas from viewing angle changes
        angle = fr.get("viewing_angle", {})
        yaw = float(angle.get("y", 0.0))
        pitch = float(angle.get("x", 0.0))

        if prev_yaw is not None:
            dx = yaw - prev_yaw
            dy = pitch - prev_pitch
            # Handle angle wrapping for yaw
            if dx > 180:
                dx -= 360
            elif dx < -180:
                dx += 360
            look_dx_vals.append(dx)
            look_dy_vals.append(dy)

        prev_yaw = yaw
        prev_pitch = pitch

        # Actions
        actions = fr.get("actions", [])
        total_action_count += len(actions)
        for act in actions:
            atype = int(act.get("action_type", -1))
            target_id = int(act.get("target_id", -1))

            if atype == shoot_action_type:
                shoot_count += 1
                shoot_pressed = True
            elif atype == reload_action_type:
                reload_pressed = True
            elif atype == jump_action_type:
                jump_pressed = True
            elif atype == crouch_action_type:
                crouch_pressed = True

            if target_id >= 0:
                targeted_action_count += 1

    # Build the action vector
    # 0: move_x_mean
    action_vector[0] = float(np.mean(move_x_vals)) if move_x_vals else 0.0
    # 1: move_y_mean
    action_vector[1] = float(np.mean(move_y_vals)) if move_y_vals else 0.0
    # 2: look_dx_sum
    action_vector[2] = float(np.sum(look_dx_vals)) if look_dx_vals else 0.0
    # 3: look_dy_sum
    action_vector[3] = float(np.sum(look_dy_vals)) if look_dy_vals else 0.0
    # 4: look_dx_mean
    action_vector[4] = float(np.mean(look_dx_vals)) if look_dx_vals else 0.0
    # 5: look_dy_mean
    action_vector[5] = float(np.mean(look_dy_vals)) if look_dy_vals else 0.0
    # 6: look_speed_mean
    if look_dx_vals:
        speeds = [
            math.sqrt(dx**2 + dy**2) for dx, dy in zip(look_dx_vals, look_dy_vals)
        ]
        action_vector[6] = float(np.mean(speeds))
    # 7: shoot_count
    action_vector[7] = float(shoot_count)
    # 8: shoot_pressed
    action_vector[8] = 1.0 if shoot_pressed else 0.0
    # 9: reload_pressed
    action_vector[9] = 1.0 if reload_pressed else 0.0
    # 10: jump_pressed
    action_vector[10] = 1.0 if jump_pressed else 0.0
    # 11: crouch_pressed
    action_vector[11] = 1.0 if crouch_pressed else 0.0
    # 12: targeted_action_count
    action_vector[12] = float(targeted_action_count)
    # 13: action_count
    action_vector[13] = float(total_action_count)

    return action_vector


# ---------------------------------------------------------------------------
# ActionExtractor class
# ---------------------------------------------------------------------------


class ActionExtractor:
    """Extracts compact next-1-second action vectors from raw gameplay JSON.

    For a state window ending at time t, this class reads the raw frames/actions
    in interval [t, t+1] and produces a fixed-size action_vector.

    The first version produces a compact summary, not all 30 frame-level actions.
    """

    def __init__(
        self,
        action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
        shoot_action_type: int = 0,
        reload_action_type: int = 5,
        jump_action_type: int = 6,
        crouch_action_type: int = 7,
    ):
        self.action_schema = action_schema
        self.shoot_action_type = shoot_action_type
        self.reload_action_type = reload_action_type
        self.jump_action_type = jump_action_type
        self.crouch_action_type = crouch_action_type

        # Cache loaded JSONs to avoid re-reading for consecutive windows
        self._json_cache: dict[str, dict] = {}

    def _load_json(self, tel_json_path: str | Path) -> dict:
        """Load and cache a JSON file."""
        key = str(tel_json_path)
        if key not in self._json_cache:
            with open(key, "r", encoding="utf-8") as f:
                self._json_cache[key] = json.load(f)
        return self._json_cache[key]

    def clear_cache(self) -> None:
        """Clear the JSON cache."""
        self._json_cache.clear()

    def extract_for_window(
        self,
        tel_json_path: str | Path,
        t: float,
        step_seconds: float = 1.0,
    ) -> dict:
        """Extract the action summary for interval [t, t+step_seconds].

        Returns:
            Dict containing:
                action_vector:
                    Fixed-size float32 vector.
                action_mask:
                    Float mask indicating which dimensions are valid (always 1
                    for now, reserved for partial observations).
                raw_action_counts:
                    Diagnostic counts by action type.
        """
        chunk = self._load_json(tel_json_path)
        frames = chunk.get("frames", [])

        t_start = t
        t_end = t + step_seconds

        action_vector = extract_interval_actions(
            frames=frames,
            t_start=t_start,
            t_end=t_end,
            action_schema=self.action_schema,
            shoot_action_type=self.shoot_action_type,
            reload_action_type=self.reload_action_type,
            jump_action_type=self.jump_action_type,
            crouch_action_type=self.crouch_action_type,
        )

        action_mask = np.ones(self.action_schema.dim, dtype=np.float32)

        # Count actions by type for diagnostics
        interval_frames = [
            fr for fr in frames if t_start <= float(fr.get("time", -1.0)) < t_end
        ]
        raw_counts: dict[int, int] = {}
        for fr in interval_frames:
            for act in fr.get("actions", []):
                atype = int(act.get("action_type", -1))
                raw_counts[atype] = raw_counts.get(atype, 0) + 1

        return {
            "action_vector": action_vector,
            "action_mask": action_mask,
            "raw_action_counts": raw_counts,
            "t_start": t_start,
            "t_end": t_end,
        }

    def save_action_npz(
        self,
        output_path: str | Path,
        action_data: dict,
    ) -> None:
        """Save action data to an .npz file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            action_vector=action_data["action_vector"],
            action_mask=action_data["action_mask"],
            t_start=np.asarray(action_data.get("t_start", np.nan), dtype=np.float32),
            t_end=np.asarray(action_data.get("t_end", np.nan), dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Action scaler
# ---------------------------------------------------------------------------


def fit_action_scaler(
    action_paths: list[str | Path],
    output_scaler_path: str | Path,
    continuous_indices: list[int] | None = None,
) -> None:
    """Compute mean/std/min/max statistics for action normalization.

    Only continuous indices are normalized. Binary action dimensions
    should remain in their original 0/1 form.
    """
    if continuous_indices is None:
        continuous_indices = list(DEFAULT_ACTION_SCHEMA.continuous_indices)

    vectors: list[np.ndarray] = []
    for p in action_paths:
        with np.load(p) as data:
            vectors.append(data["action_vector"])

    if not vectors:
        raise ValueError("No action files to fit scaler from.")

    all_actions = np.stack(vectors, axis=0)  # [N, action_dim]

    mean = np.mean(all_actions, axis=0).astype(np.float32)
    std = np.std(all_actions, axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    min_val = np.min(all_actions, axis=0).astype(np.float32)
    max_val = np.max(all_actions, axis=0).astype(np.float32)

    # Mark which indices to normalize
    normalize_mask = np.zeros(all_actions.shape[1], dtype=np.float32)
    for idx in continuous_indices:
        normalize_mask[idx] = 1.0

    output_path = Path(output_scaler_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        mean=mean,
        std=std,
        min=min_val,
        max=max_val,
        normalize_mask=normalize_mask,
        continuous_indices=np.asarray(continuous_indices, dtype=np.int64),
    )
