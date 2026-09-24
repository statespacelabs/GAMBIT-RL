#!/usr/bin/env python3
"""Create leakage-resistant Phase 3 transition and window splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def session_split(session_id: object, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{session_id}".encode()).hexdigest()
    bucket = int(digest, 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def _row_split_by_window(transitions: pd.DataFrame, seed: int) -> pd.Series:
    """Split unique state windows so one window cannot cross splits."""
    window_ids = transitions["state_window_id"].astype(str).drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(window_ids)
    n_windows = len(window_ids)
    train_end = int(round(n_windows * 0.8))
    val_end = train_end + int(round(n_windows * 0.1))
    mapping = {
        window_id: (
            "train" if index < train_end else "val" if index < val_end else "test"
        )
        for index, window_id in enumerate(window_ids)
    }
    return transitions["state_window_id"].astype(str).map(mapping)


def make_splits(
    transition_manifest: str | Path,
    window_manifest: str | Path,
    out_transition_manifest: str | Path,
    out_window_manifest: str | Path,
    seed: int = 42,
    report_path: str | Path | None = None,
) -> dict:
    transition_path = Path(transition_manifest).expanduser()
    window_path = Path(window_manifest).expanduser()
    transitions = pd.read_csv(transition_path)
    windows = pd.read_csv(window_path)

    required_transition = {"state_window_id"}
    required_window = {"window_id"}
    if missing := required_transition - set(transitions.columns):
        raise ValueError(f"Transition manifest missing columns: {sorted(missing)}")
    if missing := required_window - set(windows.columns):
        raise ValueError(f"Window manifest missing columns: {sorted(missing)}")
    if transitions.empty or windows.empty:
        raise ValueError("Transition and window manifests must be non-empty")
    if windows["window_id"].astype(str).duplicated().any():
        raise ValueError("Window manifest contains duplicate window_id values")

    session_count = (
        int(transitions["session_id"].nunique())
        if "session_id" in transitions.columns
        else 0
    )
    if "session_id" in transitions.columns and session_count >= 20:
        split_mode = "session_id"
        transitions["split"] = transitions["session_id"].map(
            lambda value: session_split(value, seed)
        )
        if "session_id" not in windows.columns:
            raise ValueError(
                "Window manifest needs session_id for session-based splitting"
            )
        windows["split"] = windows["session_id"].map(
            lambda value: session_split(value, seed)
        )
    else:
        split_mode = "deterministic_row"
        transitions["split"] = _row_split_by_window(transitions, seed)
        split_by_window = dict(
            zip(
                transitions["state_window_id"].astype(str),
                transitions["split"],
            )
        )
        windows["split"] = windows["window_id"].astype(str).map(split_by_window)
        unmapped_mask = windows["split"].isna()
        if unmapped_mask.any():
            unmapped_ids = windows.loc[unmapped_mask, "window_id"].astype(str)
            windows.loc[unmapped_mask, "split"] = unmapped_ids.map(
                lambda value: session_split(value, seed)
            )

    window_split_map = windows.set_index(windows["window_id"].astype(str))["split"]
    mapped_transition_split = (
        transitions["state_window_id"].astype(str).map(window_split_map)
    )
    unmapped_transitions = int(mapped_transition_split.isna().sum())
    mismatch = mapped_transition_split.notna() & (
        mapped_transition_split != transitions["split"]
    )
    if mismatch.any():
        examples = transitions.loc[mismatch, ["state_window_id", "split"]].head()
        raise ValueError(f"Transition/window split mismatch:\n{examples}")
    if unmapped_transitions:
        raise ValueError(
            f"{unmapped_transitions} transitions reference missing windows"
        )
    if (windows.groupby("window_id")["split"].nunique() > 1).any():
        raise ValueError("A window_id appears in multiple splits")

    out_transition = Path(out_transition_manifest).expanduser()
    out_window = Path(out_window_manifest).expanduser()
    out_transition.parent.mkdir(parents=True, exist_ok=True)
    out_window.parent.mkdir(parents=True, exist_ok=True)
    transitions.to_csv(out_transition, index=False)
    windows.to_csv(out_window, index=False)

    split_names = ("train", "val", "test")
    report = {
        "split_mode": split_mode,
        "seed": seed,
        "transition_counts": {
            name: int((transitions["split"] == name).sum()) for name in split_names
        },
        "window_counts": {
            name: int((windows["split"] == name).sum()) for name in split_names
        },
        "session_counts": (
            {
                name: int(
                    transitions.loc[
                        transitions["split"] == name, "session_id"
                    ].nunique()
                )
                for name in split_names
            }
            if "session_id" in transitions.columns
            else {}
        ),
        "unmapped_windows": int(
            (
                ~windows["window_id"]
                .astype(str)
                .isin(transitions["state_window_id"].astype(str))
            ).sum()
        ),
        "unmapped_transitions": unmapped_transitions,
        "out_transition_manifest": str(out_transition.resolve()),
        "out_window_manifest": str(out_window.resolve()),
    }
    target = (
        Path(report_path).expanduser()
        if report_path
        else out_window.with_suffix(".split_report.json")
    )
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition_manifest", required=True)
    parser.add_argument("--window_manifest", required=True)
    parser.add_argument("--out_transition_manifest", required=True)
    parser.add_argument("--out_window_manifest", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    make_splits(
        transition_manifest=args.transition_manifest,
        window_manifest=args.window_manifest,
        out_transition_manifest=args.out_transition_manifest,
        out_window_manifest=args.out_window_manifest,
        seed=args.seed,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
