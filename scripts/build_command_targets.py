#!/usr/bin/env python3
"""Rebuild Phase 3 command labels from raw event telemetry.

The historical Phase 2 action extractor used event IDs 0 and 5 for shoot and
reload. In the source analytics schema those IDs are session-start and target
spawn events. Fire is event 13. Reload, jump, and crouch commands are not
recorded by this dataset, so their labels are explicitly unavailable/zero.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gambit.dataset_preparation.action_extraction import extract_interval_actions  # noqa: E402


BUILDER_VERSION = "p3a.2"
FIRE_EVENT_TYPE = 13
UNAVAILABLE_EVENT_TYPE = -999
BINARY_LABEL_AVAILABLE = np.asarray([True, False, False, False])
EVENT_SCHEMA = {
    "fire": FIRE_EVENT_TYPE,
    "reload": None,
    "jump": None,
    "crouch": None,
}
EVENT_SCHEMA_SHA = hashlib.sha256(
    json.dumps(EVENT_SCHEMA, sort_keys=True).encode()
).hexdigest()[:16]


def _scalar(data: np.lib.npyio.NpzFile, name: str) -> str:
    return str(np.asarray(data[name]).item())


def _validate_existing(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "action_vector",
                "action_mask",
                "builder_version",
                "event_schema_sha",
                "binary_label_available",
                "transition_id",
                "state_window_id",
            }
            if required - set(data.files):
                return False
            action = np.asarray(data["action_vector"], dtype=np.float32)
            mask = np.asarray(data["action_mask"], dtype=np.float32)
            available = np.asarray(data["binary_label_available"], dtype=bool)
            if action.shape != (14,) or mask.shape != (14,):
                return False
            if available.shape != (4,) or not np.array_equal(
                available, BINARY_LABEL_AVAILABLE
            ):
                return False
            if not np.isfinite(action).all():
                return False
            if not np.isin(action[8:12], (0.0, 1.0)).all():
                return False
            return (
                _scalar(data, "builder_version") == BUILDER_VERSION
                and _scalar(data, "event_schema_sha") == EVENT_SCHEMA_SHA
            )
    except (OSError, ValueError, KeyError):
        return False


def _load_frames(path: Path) -> tuple[list[dict], list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Raw telemetry JSON not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        frames = json.load(handle).get("frames", [])
    if not frames:
        raise ValueError(f"No frames in raw telemetry JSON: {path}")
    times = [float(frame.get("time", -1.0)) for frame in frames]
    if any(right < left for left, right in zip(times, times[1:])):
        raise ValueError(f"Frame times are not monotonic: {path}")
    return frames, times


def _save_target(
    path: Path,
    action: np.ndarray,
    t_start: float,
    t_end: float,
    transition_id: str,
    state_window_id: str,
) -> None:
    if action.shape != (14,) or not np.isfinite(action).all():
        raise ValueError(f"Invalid action vector for {path}: {action.shape}")
    if not np.isin(action[8:12], (0.0, 1.0)).all():
        raise ValueError(f"Non-binary command targets for {path}: {action[8:12]}")
    mask = np.ones(14, dtype=np.float32)
    mask[9:12] = 0.0
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        action_vector=action.astype(np.float32),
        action_mask=mask,
        t_start=np.asarray(t_start, dtype=np.float64),
        t_end=np.asarray(t_end, dtype=np.float64),
        fire_event_type=np.asarray(FIRE_EVENT_TYPE, dtype=np.int64),
        binary_label_available=BINARY_LABEL_AVAILABLE,
        event_schema_sha=np.asarray(EVENT_SCHEMA_SHA),
        builder_version=np.asarray(BUILDER_VERSION),
        transition_id=np.asarray(transition_id),
        state_window_id=np.asarray(state_window_id),
    )
    os.replace(temporary, path)


def build_targets(
    transition_manifest: Path,
    window_manifest: Path,
    out_dir: Path,
    out_manifest: Path,
    *,
    resume: bool,
    skip_existing: bool,
    force: bool,
    report_path: Path | None,
) -> dict:
    transitions = pd.read_csv(transition_manifest)
    windows = pd.read_csv(window_manifest)
    required_transitions = {"transition_id", "session_id", "t", "state_window_id"}
    required_windows = {"window_id", "session_id", "tel_json_path"}
    if missing := required_transitions - set(transitions.columns):
        raise ValueError(f"Transition manifest missing columns: {sorted(missing)}")
    if missing := required_windows - set(windows.columns):
        raise ValueError(f"Window manifest missing columns: {sorted(missing)}")
    if windows["window_id"].astype(str).duplicated().any():
        raise ValueError("Window manifest contains duplicate window_id")

    if out_dir.exists() and any(out_dir.iterdir()) and not (
        force or resume or skip_existing
    ):
        raise SystemExit(
            f"Output directory is non-empty: {out_dir}. "
            "Pass --resume, --skip-existing, or --force."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    transition_rows = transitions.copy()
    transition_rows["_state_window_key"] = transition_rows[
        "state_window_id"
    ].astype(str)
    window_rows = windows[["window_id", "session_id", "tel_json_path"]].copy()
    window_rows["_state_window_key"] = window_rows["window_id"].astype(str)
    joined = transition_rows.merge(
        window_rows.drop(columns=["window_id"]),
        on="_state_window_key",
        how="left",
        validate="many_to_one",
        suffixes=("", "_window"),
    )
    if joined["tel_json_path"].isna().any():
        missing_id = joined.loc[
            joined["tel_json_path"].isna(), "state_window_id"
        ].iloc[0]
        raise ValueError(f"Transition references missing window: {missing_id}")
    if (
        joined["session_id"].astype(str)
        != joined["session_id_window"].astype(str)
    ).any():
        raise ValueError("Transition/window session mismatch")

    counters = {
        "total_transitions": int(len(joined)),
        "rebuilt": 0,
        "skipped_valid_existing": 0,
        "invalid_existing": 0,
        "failed": 0,
    }
    binary_positive_counts = np.zeros(4, dtype=np.int64)

    for _, session_rows in joined.groupby("session_id", sort=False):
        json_paths = session_rows["tel_json_path"].astype(str).unique()
        if len(json_paths) != 1:
            raise ValueError(
                f"Session {session_rows.iloc[0]['session_id']} maps to "
                f"{len(json_paths)} raw telemetry files"
            )
        frames, frame_times = _load_frames(Path(json_paths[0]).expanduser())
        for index, row in session_rows.iterrows():
            transition_id = str(row["transition_id"])
            target = out_dir / f"{transition_id}.npz"
            try:
                valid_existing = target.exists() and _validate_existing(target)
                if valid_existing and (resume or skip_existing) and not force:
                    counters["skipped_valid_existing"] += 1
                    with np.load(target, allow_pickle=False) as data:
                        action = np.asarray(data["action_vector"])
                else:
                    if target.exists() and not valid_existing:
                        counters["invalid_existing"] += 1
                    t_start = float(row["t"])
                    t_end = t_start + 1.0
                    start = bisect.bisect_left(frame_times, t_start)
                    end = bisect.bisect_left(frame_times, t_end)
                    action = extract_interval_actions(
                        frames[start:end],
                        t_start,
                        t_end,
                        shoot_action_type=FIRE_EVENT_TYPE,
                        reload_action_type=UNAVAILABLE_EVENT_TYPE,
                        jump_action_type=UNAVAILABLE_EVENT_TYPE,
                        crouch_action_type=UNAVAILABLE_EVENT_TYPE,
                    )
                    _save_target(
                        target,
                        action,
                        t_start,
                        t_end,
                        transition_id,
                        str(row["state_window_id"]),
                    )
                    counters["rebuilt"] += 1
                binary_positive_counts += action[8:12].astype(np.int64)
                transitions.at[index, "action_path"] = str(target.resolve())
            except Exception:
                counters["failed"] += 1
                raise

    if transitions["action_path"].isna().any():
        raise RuntimeError("Output transition manifest has missing action_path values")
    temporary_manifest = out_manifest.with_suffix(".csv.tmp")
    transitions.to_csv(temporary_manifest, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary_manifest, out_manifest)

    report = {
        **counters,
        "manifest_rows": int(len(transitions)),
        "action_path_populated": int(transitions["action_path"].notna().sum()),
        "builder_version": BUILDER_VERSION,
        "event_schema": EVENT_SCHEMA,
        "event_schema_sha": EVENT_SCHEMA_SHA,
        "binary_label_available": BINARY_LABEL_AVAILABLE.tolist(),
        "binary_positive_counts": binary_positive_counts.tolist(),
        "binary_positive_rates": (
            binary_positive_counts / max(len(transitions), 1)
        ).tolist(),
        "out_manifest": str(out_manifest.resolve()),
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition_manifest", required=True, type=Path)
    parser.add_argument("--window_manifest", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--out_manifest", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_targets(
        args.transition_manifest.expanduser(),
        args.window_manifest.expanduser(),
        args.out_dir.expanduser(),
        args.out_manifest.expanduser(),
        resume=args.resume,
        skip_existing=args.skip_existing,
        force=args.force,
        report_path=args.report.expanduser() if args.report else None,
    )


if __name__ == "__main__":
    main()
