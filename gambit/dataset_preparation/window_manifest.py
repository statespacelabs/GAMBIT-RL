"""Phase 2 sliding-window manifest builder.

Reads the existing clip/session manifest and creates one row per 5-second
observation window, shifted forward by stride_seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schemas import WindowRecord


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def make_window_id(session_id: str, t_end: float) -> str:
    """Create a deterministic window ID from session ID and window end time.

    The ID is stable across repeated preprocessing runs so cached latents
    and actions can be reused safely.

    Example::

        session_abc_t005000
    """
    return f"{session_id}_t{int(t_end * 1000):06d}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_WINDOW_COLUMNS = {
    "window_id",
    "player_id",
    "session_id",
    "t_start",
    "t_end",
    "video_path",
    "tel_json_path",
    "tel_npz_path",
    "split",
    "source_clip_id",
    "start_frame",
    "end_frame",
}


def validate_window_manifest(df: pd.DataFrame) -> None:
    """Validate the Phase 2 window manifest.

    Checks required columns, valid time ranges, positive duration, valid split
    values, duplicate window IDs, and session split leakage.

    Raises:
        ValueError if the manifest is invalid.
    """
    missing = _REQUIRED_WINDOW_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Window manifest missing columns: {sorted(missing)}")

    # Duration check
    bad_dur = df[df["t_end"] - df["t_start"] <= 0]
    if len(bad_dur) > 0:
        raise ValueError(
            f"{len(bad_dur)} windows have non-positive duration. "
            f"First: window_id={bad_dur.iloc[0]['window_id']}"
        )

    # Valid splits
    valid_splits = {"train", "val", "test"}
    actual_splits = set(df["split"].unique())
    bad_splits = actual_splits - valid_splits
    if bad_splits:
        raise ValueError(f"Invalid split values: {bad_splits}")

    # Duplicate window IDs
    dupes = df[df["window_id"].duplicated(keep=False)]
    if len(dupes) > 0:
        raise ValueError(
            f"{len(dupes)} duplicate window IDs found. "
            f"First: {dupes.iloc[0]['window_id']}"
        )

    # Session split leakage
    session_splits = df.groupby("session_id")["split"].nunique()
    leaking = session_splits[session_splits > 1]
    if len(leaking) > 0:
        raise ValueError(
            f"Session split leakage: {leaking.index.tolist()[:10]}"
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _get_session_duration(json_paths: list[str]) -> float:
    """Return the end time of the last chunk in a session."""
    max_end = 0.0
    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
            chunk = json.load(f)
        end = float(chunk.get("end_time", 0.0))
        if end > max_end:
            max_end = end
    return max_end


def build_phase2_window_manifest(
    source_manifest_path: str | Path,
    output_manifest_path: str | Path,
    window_seconds: float = 5.0,
    stride_seconds: float = 1.0,
    fps: float = 30.0,
    seq_len: int = 150,
    split_by_session: bool = True,
) -> pd.DataFrame:
    """Build the Phase 2 sliding-window manifest.

    Reads the existing clip/session manifest and creates one row per 5-second
    observation window. Windows are shifted forward by stride_seconds.

    This function preserves train/val/test split boundaries and does not
    allow windows from the same session to appear in multiple splits.

    The output CSV is consumed by latent_cache.py and by telemetry preprocessing.

    Returns:
        A DataFrame with one row per 5-second window.
    """
    source_manifest_path = Path(source_manifest_path)
    output_manifest_path = Path(output_manifest_path)

    source_df = pd.read_csv(source_manifest_path)

    # Resolve relative paths against manifest directory
    base = source_manifest_path.parent
    for col in ("video_path", "tel_json_path", "tel_npz_path"):
        if col in source_df.columns:
            resolved = []
            for v in source_df[col]:
                v_str = str(v) if not (isinstance(v, float) and __import__("math").isnan(v)) else ""
                if v_str and v_str != "nan" and not Path(v_str).is_absolute():
                    resolved.append(str((base / v_str).resolve()))
                else:
                    resolved.append(v_str)
            source_df[col] = resolved

    # Group by session to get session-level metadata
    rows: list[dict] = []

    for session_id, session_group in source_df.groupby("session_id"):
        session_group = session_group.sort_values("clip_id").reset_index(drop=True)

        player_id = str(session_group.iloc[0]["player_id"])
        split = str(session_group.iloc[0]["split"])

        # Use the first clip's video and json paths as representative for the session.
        # For sessions with multiple chunks, the video_path and tel_json_path
        # point to the per-chunk files. We store the first chunk's paths and the
        # window's start_frame / t_start control which segment to load.
        video_path = str(session_group.iloc[0]["video_path"])

        # Collect all JSON paths for the session to determine total duration
        json_paths = session_group["tel_json_path"].tolist()
        session_duration = _get_session_duration(json_paths)

        if session_duration < window_seconds:
            continue

        # Map each chunk to its time range for later lookups
        chunk_time_ranges: list[tuple[float, float, str, str, str]] = []
        for _, crow in session_group.iterrows():
            jp = str(crow["tel_json_path"])
            with open(jp, "r", encoding="utf-8") as f:
                chunk = json.load(f)
            c_start = float(chunk.get("start_time", 0.0))
            c_end = float(chunk.get("end_time", 0.0))
            chunk_time_ranges.append((
                c_start, c_end, jp,
                str(crow["video_path"]),
                str(crow.get("tel_npz_path", "")),
            ))
        chunk_time_ranges.sort(key=lambda x: x[0])

        # Generate sliding windows
        t = 0.0
        while t + window_seconds <= session_duration + 1e-6:
            t_start = t
            t_end = t + window_seconds
            start_frame = round(t_start * fps)
            end_frame = start_frame + seq_len

            wid = make_window_id(str(session_id), t_end)

            # Find which chunk contains t_start to get the right video/json
            # For simplicity, use the chunk whose time range contains t_start
            chunk_video = video_path
            chunk_json = str(json_paths[0])
            chunk_npz = ""
            source_clip_id = str(session_group.iloc[0].get("clip_id", ""))

            for c_start, c_end, cjp, cvp, cnpz in chunk_time_ranges:
                if c_start <= t_start < c_end:
                    chunk_video = cvp
                    chunk_json = cjp
                    chunk_npz = cnpz
                    # Find the source clip_id for this chunk
                    match = session_group[
                        session_group["tel_json_path"] == cjp
                    ]
                    if len(match) > 0:
                        source_clip_id = str(match.iloc[0].get("clip_id", ""))
                    break

            rows.append({
                "window_id": wid,
                "player_id": player_id,
                "session_id": str(session_id),
                "t_start": t_start,
                "t_end": t_end,
                "video_path": chunk_video,
                "tel_json_path": chunk_json,
                "tel_npz_path": chunk_npz,
                "split": split,
                "source_clip_id": source_clip_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
            })

            t += stride_seconds

    if not rows:
        raise ValueError("No valid windows could be created from the source manifest.")

    df = pd.DataFrame(rows)
    validate_window_manifest(df)

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_manifest_path, index=False)

    return df
