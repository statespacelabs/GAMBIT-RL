import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


WHERE_DIM = 10
VIEW_DIM = 13
RHYTHM_EXTRA_DIM = 8

# with generous help from gemini!

@dataclass(frozen=True)
class TelemetryStats:
    where_mean: np.ndarray
    where_std: np.ndarray
    view_mean: np.ndarray
    view_std: np.ndarray
    rhythm_mean: np.ndarray
    rhythm_std: np.ndarray

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            where_mean=self.where_mean.astype(np.float32),
            where_std=self.where_std.astype(np.float32),
            view_mean=self.view_mean.astype(np.float32),
            view_std=self.view_std.astype(np.float32),
            rhythm_mean=self.rhythm_mean.astype(np.float32),
            rhythm_std=self.rhythm_std.astype(np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TelemetryStats":
        with np.load(path) as data:
            return cls(
                where_mean=data["where_mean"].astype(np.float32),
                where_std=data["where_std"].astype(np.float32),
                view_mean=data["view_mean"].astype(np.float32),
                view_std=data["view_std"].astype(np.float32),
                rhythm_mean=data["rhythm_mean"].astype(np.float32),
                rhythm_std=data["rhythm_std"].astype(np.float32),
            )


def _vec3(obj: dict | None) -> list[float]:
    obj = obj or {}
    return [
        float(obj.get("x", 0.0)),
        float(obj.get("y", 0.0)),
        float(obj.get("z", 0.0)),
    ]


def _sin_cos_deg(angle_deg: float) -> tuple[float, float]:
    rad = math.radians(float(angle_deg))
    return math.sin(rad), math.cos(rad)


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _time_since(
    now: float,
    last_time: float | None,
    clip_seconds: float,
    normalize: bool = True,
) -> float:
    if last_time is None:
        value = clip_seconds
    else:
        value = min(max(now - last_time, 0.0), clip_seconds)

    if normalize:
        return value / clip_seconds

    return value


def _load_fixed_frames(
    json_path: str | Path,
    seq_len: int,
    fps: float,
) -> tuple[dict, list[dict]]:
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        chunk = json.load(f)

    frames = chunk.get("frames", [])
    if len(frames) == 0:
        raise ValueError(f"No frames found in {json_path}")

    if len(frames) > seq_len:
        frames = frames[:seq_len]
    elif len(frames) < seq_len:
        last = frames[-1]
        padded = []
        for j in range(seq_len - len(frames)):
            new_frame = dict(last)
            new_frame["actions"] = []
            new_frame["time"] = _safe_float(last.get("time", 0.0)) + (j + 1) / fps
            padded.append(new_frame)
        frames = frames + padded

    return chunk, frames


def max_action_type_in_json(json_path: str | Path) -> int:
    with open(json_path, "r", encoding="utf-8") as f:
        chunk = json.load(f)

    max_action = -1
    for frame in chunk.get("frames", []):
        for action in frame.get("actions", []):
            max_action = max(max_action, _safe_int(action.get("action_type", -1), -1))
    return max_action


def infer_action_vocab_size(json_paths: Iterable[str | Path]) -> int:
    max_action = -1
    for path in json_paths:
        max_action = max(max_action, max_action_type_in_json(path))
    if max_action < 0:
        raise ValueError("No non-negative action_type values found; action vocab size cannot be inferred")
    return max_action + 1


def _extract_arrays_from_frames(
    chunk: dict,
    frames: list[dict],
    action_vocab_size: int,
    seq_len: int,
    clip_seconds: float,
    fps: float,
    shoot_action_type: int,
    reload_action_type: int,
    normalize_time_since: bool,
    strict_action_vocab: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if action_vocab_size <= 0:
        raise ValueError("action_vocab_size must be positive")

    where_tel = np.zeros((seq_len, WHERE_DIM), dtype=np.float32)
    view_tel = np.zeros((seq_len, VIEW_DIM), dtype=np.float32)
    rhythm_tel = np.zeros((seq_len, action_vocab_size + RHYTHM_EXTRA_DIM), dtype=np.float32)

    start_time = _safe_float(chunk.get("start_time", 0.0), 0.0)

    last_action_time: float | None = None
    last_shot_time: float | None = None
    last_reload_time: float | None = None
    last_targeted_action_time: float | None = None
    previous_target_id: int | None = None

    for i, frame in enumerate(frames):
        t_global = _safe_float(frame.get("time", start_time + i / fps))
        t_rel = t_global - start_time

        pos = _vec3(frame.get("position"))
        vel = _vec3(frame.get("velocity"))
        acc = _vec3(frame.get("acceleration"))
        dist = _safe_float(frame.get("distance_to_opponent", 0.0))
        where_tel[i] = np.asarray(pos + vel + acc + [dist], dtype=np.float32)

        angle = frame.get("viewing_angle", {})
        pitch = _safe_float(angle.get("x", 0.0))
        yaw = _safe_float(angle.get("y", 0.0))
        sin_pitch, cos_pitch = _sin_cos_deg(pitch)
        sin_yaw, cos_yaw = _sin_cos_deg(yaw)

        view_vel = _vec3(frame.get("viewing_velocity"))
        view_acc = _vec3(frame.get("viewing_acceleration"))
        yaw_err = _safe_float(frame.get("crosshair_offset_yaw", 0.0))
        pitch_err = _safe_float(frame.get("crosshair_offset_pitch", 0.0))
        err_mag = math.sqrt(yaw_err * yaw_err + pitch_err * pitch_err)

        view_tel[i] = np.asarray(
            [
                sin_pitch,
                cos_pitch,
                sin_yaw,
                cos_yaw,
                *view_vel,
                *view_acc,
                yaw_err,
                pitch_err,
                err_mag,
            ],
            dtype=np.float32,
        )

        actions = frame.get("actions", [])
        action_count = len(actions)
        has_any_action = 1.0 if action_count > 0 else 0.0
        has_targeted_action = 0.0
        same_target_as_previous_action = 0.0

        for action in actions:
            action_type = _safe_int(action.get("action_type", -1), -1)
            target_id = _safe_int(action.get("target_id", -1), -1)
            action_t_rel = _safe_float(action.get("relative_time", t_rel), t_rel)

            if action_type >= action_vocab_size:
                if strict_action_vocab:
                    raise ValueError(
                        f"action_type={action_type} is outside action_vocab_size={action_vocab_size}"
                    )
            elif action_type >= 0:
                rhythm_tel[i, action_type] += 1.0

            last_action_time = action_t_rel
            if action_type == shoot_action_type:
                last_shot_time = action_t_rel
            if action_type == reload_action_type:
                last_reload_time = action_t_rel

            if target_id >= 0:
                has_targeted_action = 1.0
                last_targeted_action_time = action_t_rel
                if previous_target_id is not None and target_id == previous_target_id:
                    same_target_as_previous_action = 1.0
                previous_target_id = target_id

        rhythm_tel[i, action_vocab_size:] = np.asarray(
            [
                float(action_count),
                has_any_action,
                has_targeted_action,
                _time_since(t_rel, last_action_time, clip_seconds, normalize_time_since),
                _time_since(t_rel, last_shot_time, clip_seconds, normalize_time_since),
                _time_since(t_rel, last_reload_time, clip_seconds, normalize_time_since),
                _time_since(t_rel, last_targeted_action_time, clip_seconds, normalize_time_since),
                same_target_as_previous_action,
            ],
            dtype=np.float32,
        )

    return where_tel, view_tel, rhythm_tel


def _apply_stats(
    where_tel: np.ndarray,
    view_tel: np.ndarray,
    rhythm_tel: np.ndarray,
    stats: TelemetryStats | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stats is None:
        return where_tel, view_tel, rhythm_tel

    where_out = where_tel.copy()
    view_out = view_tel.copy()
    rhythm_out = rhythm_tel.copy()

    # Normalize all where dims.
    where_out = ((where_out - stats.where_mean) / stats.where_std).astype(np.float32)

    # Normalize only continuous view dims:
    # 0-3 are sin/cos pitch/yaw and should stay raw.
    view_out[:, 4:13] = (
        (view_out[:, 4:13] - stats.view_mean[4:13])
        / stats.view_std[4:13]
    )

    return (
        where_out.astype(np.float32),
        view_out.astype(np.float32),
        rhythm_out.astype(np.float32),
    )


def preprocess_chunk_json(
    json_path: str | Path,
    output_npz_path: str | Path,
    action_vocab_size: int,
    seq_len: int = 150,
    clip_seconds: float = 5.0,
    fps: float = 30.0,
    shoot_action_type: int = 0,
    reload_action_type: int = 5,
    normalize_time_since: bool = True,
    stats: TelemetryStats | str | Path | None = None,
    strict_action_vocab: bool = True,
) -> dict:
    """Convert one Unity chunk JSON into fixed model tensors and save an NPZ."""
    json_path = Path(json_path)
    output_npz_path = Path(output_npz_path)
    loaded_stats = TelemetryStats.load(stats) if isinstance(stats, (str, Path)) else stats

    chunk, frames = _load_fixed_frames(json_path, seq_len=seq_len, fps=fps)
    where_tel, view_tel, rhythm_tel = _extract_arrays_from_frames(
        chunk=chunk,
        frames=frames,
        action_vocab_size=action_vocab_size,
        seq_len=seq_len,
        clip_seconds=clip_seconds,
        fps=fps,
        shoot_action_type=shoot_action_type,
        reload_action_type=reload_action_type,
        normalize_time_since=normalize_time_since,
        strict_action_vocab=strict_action_vocab,
    )
    where_tel, view_tel, rhythm_tel = _apply_stats(where_tel, view_tel, rhythm_tel, loaded_stats)

    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = _safe_float(chunk.get("start_time", 0.0), 0.0)
    np.savez_compressed(
        output_npz_path,
        where_tel=where_tel,
        view_tel=view_tel,
        rhythm_tel=rhythm_tel,
        chunk_id=np.asarray(str(chunk.get("chunk_id", json_path.stem))),
        start_time=np.asarray(start_time, dtype=np.float32),
        end_time=np.asarray(_safe_float(chunk.get("end_time", start_time + clip_seconds)), dtype=np.float32),
        normalized=np.asarray(loaded_stats is not None),
        action_vocab_size=np.asarray(action_vocab_size, dtype=np.int64),
    )

    return {
        "chunk_id": str(chunk.get("chunk_id", json_path.stem)),
        "where_shape": where_tel.shape,
        "view_shape": view_tel.shape,
        "rhythm_shape": rhythm_tel.shape,
    }


def compute_telemetry_stats(
    json_paths: Iterable[str | Path],
    action_vocab_size: int,
    seq_len: int = 150,
    clip_seconds: float = 5.0,
    fps: float = 30.0,
    shoot_action_type: int = 0,
    reload_action_type: int = 5,
    normalize_time_since: bool = True,
    strict_action_vocab: bool = True,
    eps: float = 1e-6,
) -> TelemetryStats:
    where_chunks: list[np.ndarray] = []
    view_chunks: list[np.ndarray] = []
    rhythm_chunks: list[np.ndarray] = []

    for json_path in json_paths:
        chunk, frames = _load_fixed_frames(json_path, seq_len=seq_len, fps=fps)
        where_tel, view_tel, rhythm_tel = _extract_arrays_from_frames(
            chunk=chunk,
            frames=frames,
            action_vocab_size=action_vocab_size,
            seq_len=seq_len,
            clip_seconds=clip_seconds,
            fps=fps,
            shoot_action_type=shoot_action_type,
            reload_action_type=reload_action_type,
            normalize_time_since=normalize_time_since,
            strict_action_vocab=strict_action_vocab,
        )
        where_chunks.append(where_tel)
        view_chunks.append(view_tel)
        rhythm_chunks.append(rhythm_tel)

    if not where_chunks:
        raise ValueError("No JSON paths provided for telemetry statistics")

    def mean_std(chunks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate(chunks, axis=0).astype(np.float64)
        mean = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32)
        std = np.maximum(std, eps).astype(np.float32)
        return mean, std

    where_mean, where_std = mean_std(where_chunks)
    view_mean, view_std = mean_std(view_chunks)
    rhythm_mean, rhythm_std = mean_std(rhythm_chunks)

    return TelemetryStats(
        where_mean=where_mean,
        where_std=where_std,
        view_mean=view_mean,
        view_std=view_std,
        rhythm_mean=rhythm_mean,
        rhythm_std=rhythm_std,
    )


def preprocess_chunks(
    rows: Iterable[dict],
    action_vocab_size: int,
    stats: TelemetryStats | str | Path | None = None,
    seq_len: int = 150,
    clip_seconds: float = 5.0,
    fps: float = 30.0,
    shoot_action_type: int = 0,
    reload_action_type: int = 5,
) -> list[dict]:
    results = []
    for row in rows:
        results.append(
            preprocess_chunk_json(
                json_path=row["tel_json_path"],
                output_npz_path=row["tel_npz_path"],
                action_vocab_size=action_vocab_size,
                seq_len=seq_len,
                clip_seconds=clip_seconds,
                fps=fps,
                shoot_action_type=shoot_action_type,
                reload_action_type=reload_action_type,
                stats=stats,
            )
        )
    return results
