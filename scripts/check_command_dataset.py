#!/usr/bin/env python3
"""Validate production Phase 3 distillation manifests and sample statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gambit.rl.online_rl.distillation_dataset import DistillationDataset  # noqa: E402


def _stats(array: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def check_dataset(
    transition_manifest: str | Path,
    window_manifest: str | Path,
    sample_count: int = 100,
    seed: int = 42,
) -> dict:
    transition_path = Path(transition_manifest).expanduser()
    window_path = Path(window_manifest).expanduser()
    transitions = pd.read_csv(transition_path)
    windows = pd.read_csv(window_path)
    if sample_count < 10:
        raise ValueError("sample_count must be at least 10")

    action_paths = transitions["action_path"].astype(str).map(Path)
    telemetry_path_values = windows["tel_npz_path"].fillna("").astype(str)
    telemetry_paths = telemetry_path_values.map(Path)
    missing_actions = int(sum(not path.exists() for path in action_paths))
    missing_telemetry = int(
        sum(
            not value.strip() or not Path(value).exists()
            for value in telemetry_path_values
        )
    )

    window_split = windows.set_index(windows["window_id"].astype(str))["split"]
    mapped_split = transitions["state_window_id"].astype(str).map(window_split)
    unmapped_transitions = int(mapped_split.isna().sum())
    split_mismatches = int(
        (
            mapped_split.notna()
            & (mapped_split.astype(str) != transitions["split"].astype(str))
        ).sum()
    )
    if missing_actions or missing_telemetry or unmapped_transitions or split_mismatches:
        raise RuntimeError(
            "Dataset path/split checks failed: "
            f"missing_actions={missing_actions}, "
            f"missing_telemetry={missing_telemetry}, "
            f"unmapped_transitions={unmapped_transitions}, "
            f"split_mismatches={split_mismatches}"
        )

    rng = np.random.default_rng(seed)
    split_reports: dict[str, dict] = {}
    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    alignment_passes = 0
    alignment_failures = 0
    timestamp_checks = 0

    for split in ("train", "val"):
        dataset = DistillationDataset(
            transition_manifest_path=transition_path,
            window_manifest_path=window_path,
            mode="single",
            split=split,
            require_normalized_telemetry=True,
        )
        if len(dataset) == 0:
            raise RuntimeError(f"Split {split!r} has no transitions")
        count = min(sample_count, len(dataset))
        indices = rng.choice(len(dataset), size=count, replace=False)
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        for index in indices:
            obs, action = dataset[int(index)]
            observations.append(obs.numpy())
            actions.append(action.numpy())
        obs_array = np.stack(observations)
        action_array = np.stack(actions)
        if not np.isfinite(obs_array).all() or np.allclose(obs_array, 0.0):
            raise RuntimeError(f"Split {split!r} has invalid sampled observations")
        if not np.isfinite(action_array).all():
            raise RuntimeError(f"Split {split!r} has non-finite actions")
        if np.any(action_array[:, :4] < -1.0) or np.any(action_array[:, :4] > 1.0):
            raise RuntimeError(f"Split {split!r} has continuous actions outside [-1,1]")
        binary_unique = np.unique(action_array[:, 4:8])
        if not np.isin(binary_unique, (0.0, 1.0)).all():
            raise RuntimeError(f"Split {split!r} has non-binary action targets")

        alignment = dataset.verify_alignment(n_sessions=20)
        alignment_passes += int(bool(alignment["passed"]))
        alignment_failures += int(not alignment["passed"])
        timestamp_checks += int(alignment["timestamp_checks"])
        if not alignment["passed"]:
            raise RuntimeError(
                f"Split {split!r} alignment failed: {alignment['errors'][:5]}"
            )
        split_reports[split] = {
            "transitions": len(dataset),
            "sample_count": count,
            "obs_stats": _stats(obs_array),
            "continuous_action_stats": _stats(action_array[:, :4]),
            "binary_positive_rates": action_array[:, 4:8].mean(axis=0).tolist(),
            "binary_unique_values": binary_unique.tolist(),
            "alignment": alignment,
        }
        all_obs.append(obs_array)
        all_actions.append(action_array)

    sampled_windows = windows.sample(
        n=min(sample_count, len(windows)), random_state=seed
    )
    branch_values: dict[str, list[np.ndarray]] = {
        "where_tel": [],
        "view_tel": [],
        "rhythm_tel": [],
    }
    for path_value in sampled_windows["tel_npz_path"]:
        with np.load(Path(str(path_value)), allow_pickle=False) as data:
            for name in branch_values:
                branch_values[name].append(np.asarray(data[name]))

    obs_array = np.concatenate(all_obs)
    action_array = np.concatenate(all_actions)
    report = {
        "path_resolution": {
            "action_paths_total": len(action_paths),
            "action_paths_missing": missing_actions,
            "telemetry_paths_total": len(telemetry_paths),
            "telemetry_paths_missing": missing_telemetry,
        },
        "split_consistency": {
            "unmapped_transitions": unmapped_transitions,
            "split_mismatches": split_mismatches,
        },
        "alignment": {
            "splits_passed": alignment_passes,
            "splits_failed": alignment_failures,
            "timestamp_checks": timestamp_checks,
        },
        "telemetry_branch_stats": {
            name: _stats(np.concatenate(values))
            for name, values in branch_values.items()
        },
        "sampled_obs": {
            **_stats(obs_array),
            "finite": bool(np.isfinite(obs_array).all()),
            "nonzero": bool(not np.allclose(obs_array, 0.0)),
        },
        "continuous_actions": _stats(action_array[:, :4]),
        "binary_positive_rates": action_array[:, 4:8].mean(axis=0).tolist(),
        "binary_unique_values": np.unique(action_array[:, 4:8]).tolist(),
        "splits": split_reports,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition_manifest", required=True)
    parser.add_argument("--window_manifest", required=True)
    parser.add_argument("--sample_count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = check_dataset(
        transition_manifest=args.transition_manifest,
        window_manifest=args.window_manifest,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    target = Path(args.report).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
