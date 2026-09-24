#!/usr/bin/env python3
"""Build provenance-tagged normalized telemetry NPZs for Phase 3A."""

from __future__ import annotations

import argparse
import csv
import hashlib
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gambit.datasets.telemetry_preprocess import (  # noqa: E402
    TelemetryStats,
    _apply_stats,
    _extract_arrays_from_frames,
    _safe_float,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ACTION_VOCAB_SIZE = 14
FPS = 30.0
CLIP_SECONDS = 5.0
BUILDER_VERSION = "p3.1"
REQUIRED_ARRAYS = {
    "where_tel": 10,
    "view_tel": 13,
    "rhythm_tel": 22,
}


def _stats_sha(path: str | Path) -> str:
    """Return the stable short fingerprint stored in every production NPZ."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _scalar_string(value: np.ndarray | Any) -> str:
    return str(np.asarray(value).item())


def _validate_arrays(
    window_id: str,
    where_tel: np.ndarray,
    view_tel: np.ndarray,
    rhythm_tel: np.ndarray,
) -> None:
    arrays = {
        "where_tel": where_tel,
        "view_tel": view_tel,
        "rhythm_tel": rhythm_tel,
    }
    lengths: set[int] = set()
    for name, expected_dim in REQUIRED_ARRAYS.items():
        array = arrays[name]
        if array.ndim != 2 or array.shape[1] != expected_dim:
            raise ValueError(
                f"{window_id}: {name} shape {array.shape} != (*,{expected_dim})"
            )
        if array.shape[0] == 0:
            raise ValueError(f"{window_id}: {name} has no frames")
        if not np.isfinite(array).all():
            raise ValueError(f"{window_id}: non-finite {name}")
        lengths.add(int(array.shape[0]))
    if len(lengths) != 1:
        raise ValueError(f"{window_id}: telemetry branch lengths differ: {lengths}")
    if np.allclose(where_tel, 0.0) and np.allclose(view_tel, 0.0):
        raise ValueError(f"{window_id}: all-zero where/view telemetry")


def _validate_existing_npz(
    path: Path,
    expected_stats_sha: str,
    window_id: str,
) -> tuple[bool, str, dict[str, np.ndarray] | None]:
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = set(REQUIRED_ARRAYS) - set(data.files)
            if missing:
                return False, "wrong_shape", None
            if "normalized" not in data.files or not bool(data["normalized"]):
                return False, "not_normalized", None
            if "stats_sha" not in data.files:
                return False, "wrong_stats_sha", None
            if _scalar_string(data["stats_sha"]) != expected_stats_sha:
                return False, "wrong_stats_sha", None
            arrays = {name: np.asarray(data[name]) for name in REQUIRED_ARRAYS}
            _validate_arrays(window_id, **arrays)
            return True, "valid", arrays
    except (OSError, ValueError, KeyError, EOFError) as exc:
        message = str(exc).lower()
        if "non-finite" in message:
            reason = "non_finite"
        elif "all-zero" in message:
            reason = "all_zero"
        elif "shape" in message or "frames" in message or "length" in message:
            reason = "wrong_shape"
        else:
            reason = "corrupt"
        return False, reason, None


@lru_cache(maxsize=8)
def _load_session_json(tel_json_path: str) -> dict:
    with open(tel_json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_frames_for_window(
    tel_json_path: str,
    t_start: float,
    t_end: float,
    fps: float = FPS,
) -> tuple[dict, list[dict], np.ndarray]:
    """Load a right-open window and pad short tails with the final real frame."""
    chunk = _load_session_json(tel_json_path)
    all_frames = chunk.get("frames", [])
    if not all_frames:
        raise ValueError(f"No frames in {tel_json_path}")

    epsilon = 1e-4
    window_frames = [
        frame
        for frame in all_frames
        if t_start - epsilon <= _safe_float(frame.get("time", -999.0)) < t_end
    ]
    seq_len = int(round((t_end - t_start) * fps))
    if seq_len <= 0:
        raise ValueError(f"Invalid window interval [{t_start}, {t_end})")
    if not window_frames:
        raise ValueError(
            f"No frames in [{t_start:.6f}, {t_end:.6f}) for {tel_json_path}"
        )

    window_frames = window_frames[:seq_len]
    while len(window_frames) < seq_len:
        padded = dict(window_frames[-1])
        padded["actions"] = []
        padded["time"] = t_start + len(window_frames) / fps
        window_frames.append(padded)

    timestamps = np.asarray(
        [
            _safe_float(frame.get("time", t_start + i / fps))
            for i, frame in enumerate(window_frames)
        ],
        dtype=np.float64,
    )
    return chunk, window_frames, timestamps


class _BranchStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, array: np.ndarray) -> None:
        values = np.asarray(array, dtype=np.float64)
        self.count += int(values.size)
        self.total += float(values.sum())
        self.total_sq += float(np.square(values).sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

    def result(self) -> dict[str, float | int]:
        if not self.count:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": variance**0.5,
            "min": self.minimum,
            "max": self.maximum,
        }


def _write_partial_rows(
    partial_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> tuple[Any, csv.DictWriter]:
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(partial_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    handle.flush()
    os.fsync(handle.fileno())
    return handle, writer


def build_telemetry_npz(
    window_manifest_path: str,
    telemetry_stats_path: str,
    out_dir: str,
    out_manifest_path: str,
    *,
    resume: bool = False,
    skip_existing: bool = False,
    force: bool = False,
    fail_fast: bool = True,
    report_path: str | None = None,
) -> dict[str, Any]:
    """Build all windows with resumable manifest and NPZ provenance checks."""
    stats_path = Path(telemetry_stats_path).expanduser()
    if not stats_path.exists():
        raise FileNotFoundError(f"telemetry_stats not found: {stats_path}")
    manifest_path = Path(window_manifest_path).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"window_manifest not found: {manifest_path}")

    stats = TelemetryStats.load(stats_path)
    stats_sha = _stats_sha(stats_path)
    out_dir_path = Path(out_dir).expanduser()
    out_manifest = Path(out_manifest_path).expanduser()
    partial_path = out_manifest.with_suffix(".csv.partial")

    if out_dir_path.exists() and any(out_dir_path.iterdir()):
        if not (force or resume or skip_existing):
            raise SystemExit(
                f"Output directory is non-empty: {out_dir_path}. "
                "Pass --force, --resume, or --skip-existing explicitly."
            )
    out_dir_path.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames or not rows:
        raise ValueError(f"Empty window manifest: {manifest_path}")
    for required in ("window_id", "tel_json_path", "t_start", "t_end", "session_id"):
        if required not in fieldnames:
            raise ValueError(f"Window manifest missing required column: {required}")
    if "tel_npz_path" not in fieldnames:
        fieldnames.append("tel_npz_path")

    previous_rows: dict[str, dict[str, str]] = {}
    if resume and partial_path.exists():
        with open(partial_path, newline="", encoding="utf-8") as handle:
            previous_rows = {
                row["window_id"]: row
                for row in csv.DictReader(handle)
                if row.get("window_id")
            }

    counters = {
        "total_windows_expected": len(rows),
        "processed": 0,
        "skipped_valid_existing": 0,
        "rebuilt": 0,
        "failed": 0,
        "missing_path": 0,
        "non_finite": 0,
        "wrong_shape": 0,
        "all_zero": 0,
        "wrong_stats_sha": 0,
        "corrupt": 0,
    }
    branch_stats = {name: _BranchStats() for name in REQUIRED_ARRAYS}
    sampled_windows = 0
    completed_rows: list[dict[str, str]] = []
    completed_ids: set[str] = set()

    # Keep only previously completed rows whose deterministic NPZ still validates.
    if resume:
        for source_row in rows:
            window_id = source_row["window_id"]
            previous = previous_rows.get(window_id)
            if not previous:
                continue
            target = out_dir_path / f"{window_id}.npz"
            valid, _, arrays = _validate_existing_npz(target, stats_sha, window_id)
            if not valid:
                continue
            row = dict(source_row)
            row["tel_npz_path"] = str(target.resolve())
            completed_rows.append(row)
            completed_ids.add(window_id)
            counters["skipped_valid_existing"] += 1
            if sampled_windows < 128 and arrays is not None:
                for name, array in arrays.items():
                    branch_stats[name].update(array)
                sampled_windows += 1

    partial_handle, writer = _write_partial_rows(
        partial_path, fieldnames, completed_rows
    )
    try:
        for index, source_row in enumerate(rows, start=1):
            window_id = source_row["window_id"]
            if window_id in completed_ids:
                continue
            row = dict(source_row)
            target = out_dir_path / f"{window_id}.npz"
            try:
                arrays: dict[str, np.ndarray] | None = None
                if (skip_existing or resume) and target.exists() and not force:
                    valid, reason, arrays = _validate_existing_npz(
                        target, stats_sha, window_id
                    )
                    if valid:
                        counters["skipped_valid_existing"] += 1
                    else:
                        counters[reason] = counters.get(reason, 0) + 1
                        arrays = None

                if arrays is None:
                    tel_json_path = Path(str(row["tel_json_path"])).expanduser()
                    if not tel_json_path.exists():
                        counters["missing_path"] += 1
                        raise FileNotFoundError(f"JSON not found: {tel_json_path}")
                    t_start = float(row["t_start"])
                    t_end = float(row["t_end"])
                    chunk, frames, timestamps = _load_frames_for_window(
                        str(tel_json_path), t_start, t_end, fps=FPS
                    )
                    where_tel, view_tel, rhythm_tel = _extract_arrays_from_frames(
                        chunk=chunk,
                        frames=frames,
                        action_vocab_size=ACTION_VOCAB_SIZE,
                        seq_len=len(frames),
                        clip_seconds=CLIP_SECONDS,
                        fps=FPS,
                        shoot_action_type=0,
                        reload_action_type=5,
                        normalize_time_since=True,
                        strict_action_vocab=False,
                    )
                    where_tel, view_tel, rhythm_tel = _apply_stats(
                        where_tel, view_tel, rhythm_tel, stats
                    )
                    arrays = {
                        "where_tel": where_tel,
                        "view_tel": view_tel,
                        "rhythm_tel": rhythm_tel,
                    }
                    _validate_arrays(window_id, **arrays)
                    temporary = target.with_suffix(".tmp.npz")
                    np.savez_compressed(
                        temporary,
                        **arrays,
                        timestamps=timestamps,
                        chunk_id=np.asarray(
                            str(chunk.get("chunk_id", row["session_id"]))
                        ),
                        start_time=np.asarray(t_start, dtype=np.float64),
                        end_time=np.asarray(t_end, dtype=np.float64),
                        normalized=np.asarray(True),
                        action_vocab_size=np.asarray(ACTION_VOCAB_SIZE, dtype=np.int64),
                        stats_sha=np.asarray(stats_sha),
                        builder_version=np.asarray(BUILDER_VERSION),
                    )
                    os.replace(temporary, target)
                    counters["rebuilt"] += 1

                row["tel_npz_path"] = str(target.resolve())
                writer.writerow(row)
                counters["processed"] += 1
                if sampled_windows < 128:
                    for name, array in arrays.items():
                        branch_stats[name].update(array)
                    sampled_windows += 1
                partial_handle.flush()
                if counters["processed"] % 500 == 0:
                    os.fsync(partial_handle.fileno())
                if index % 500 == 0 or index == len(rows):
                    logger.info(
                        "Progress %d/%d processed=%d skipped=%d rebuilt=%d failed=%d",
                        index,
                        len(rows),
                        counters["processed"],
                        counters["skipped_valid_existing"],
                        counters["rebuilt"],
                        counters["failed"],
                    )
            except Exception as exc:
                counters["failed"] += 1
                message = str(exc).lower()
                if "non-finite" in message:
                    counters["non_finite"] += 1
                elif "all-zero" in message:
                    counters["all_zero"] += 1
                elif "shape" in message or "frames" in message:
                    counters["wrong_shape"] += 1
                logger.exception("Failed window %s: %s", window_id, exc)
                if fail_fast:
                    partial_handle.flush()
                    os.fsync(partial_handle.fileno())
                    raise
        partial_handle.flush()
        os.fsync(partial_handle.fileno())
    finally:
        partial_handle.close()

    with open(partial_path, newline="", encoding="utf-8") as handle:
        final_rows = list(csv.DictReader(handle))
    populated = sum(bool(row.get("tel_npz_path", "").strip()) for row in final_rows)
    counters["processed"] = len(final_rows)
    counters["final_manifest_row_count"] = len(final_rows)
    counters["tel_npz_path_populated_count"] = populated
    counters["missing_tel_npz_path_count"] = len(final_rows) - populated
    if counters["failed"] or len(final_rows) != len(rows) or populated != len(rows):
        raise RuntimeError(
            "Telemetry build incomplete; partial manifest preserved at "
            f"{partial_path}. Summary: {counters}"
        )

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial_path, out_manifest)
    report = {
        **counters,
        "stats_sha": stats_sha,
        "builder_version": BUILDER_VERSION,
        "window_manifest": str(manifest_path.resolve()),
        "telemetry_stats": str(stats_path.resolve()),
        "out_dir": str(out_dir_path.resolve()),
        "out_manifest": str(out_manifest.resolve()),
        "sampled_windows": sampled_windows,
        "branch_stats": {
            name: accumulator.result() for name, accumulator in branch_stats.items()
        },
    }
    report_target = (
        Path(report_path).expanduser()
        if report_path
        else out_manifest.with_suffix(".build_report.json")
    )
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Telemetry build complete:\n%s", json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build production normalized telemetry NPZ files."
    )
    parser.add_argument("--window_manifest", required=True)
    parser.add_argument("--telemetry_stats", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_manifest", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    build_telemetry_npz(
        window_manifest_path=args.window_manifest,
        telemetry_stats_path=args.telemetry_stats,
        out_dir=args.out_dir,
        out_manifest_path=args.out_manifest,
        resume=args.resume,
        skip_existing=args.skip_existing,
        force=args.force,
        fail_fast=args.fail_fast,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
