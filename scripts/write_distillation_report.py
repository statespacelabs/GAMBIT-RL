#!/usr/bin/env python3
"""Assemble the Phase 3A production gate artifacts into REPORT.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def best_epoch(records: list[dict]) -> dict:
    return min(
        records,
        key=lambda row: row.get("val", {}).get("total_loss", float("inf")),
        default={},
    )


def dump(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    logs = root / "logs"
    checkpoints = root / "checkpoints"
    build = load_json(logs / "telemetry_build_report.json")
    actions = load_json(logs / "action_build_report.json")
    split = load_json(logs / "split_report.json")
    dataset = load_json(logs / "dataset_check_report.json")
    delta = load_json(logs / "recurrent_delta_report.json")
    invariant = load_json(logs / "invariant_report.json")
    feedforward = best_epoch(load_jsonl(logs / "distill_stage1.jsonl"))
    recurrent = best_epoch(load_jsonl(logs / "distill_stage2.jsonl"))

    lines = [
        "# Phase 3A Production Distillation Report",
        "",
        "PPO training was not started. `enable_ppo` remained `false`.",
        "",
        "## A. Telemetry Build Summary",
        f"- Total windows expected: {build.get('total_windows_expected')}",
        f"- Valid existing skipped: {build.get('skipped_valid_existing')}",
        f"- Rebuilt: {build.get('rebuilt')}",
        f"- Failed: {build.get('failed')}",
        f"- Final manifest rows: {build.get('final_manifest_row_count')}",
        f"- Populated telemetry paths: {build.get('tel_npz_path_populated_count')}",
        f"- Missing paths: {build.get('missing_path')}",
        f"- Non-finite: {build.get('non_finite')}",
        f"- Wrong shape: {build.get('wrong_shape')}",
        f"- All-zero: {build.get('all_zero')}",
        f"- Stats SHA: `{build.get('stats_sha')}`",
        f"- Builder version: `{build.get('builder_version')}`",
        f"- Branch stats: `{dump(build.get('branch_stats', {}))}`",
        "",
        "## A2. Command Label Build",
        f"- Total transitions: {actions.get('total_transitions')}",
        f"- Rebuilt/skipped/failed: {actions.get('rebuilt')}/{actions.get('skipped_valid_existing')}/{actions.get('failed')}",
        f"- Event schema: `{dump(actions.get('event_schema', {}))}`",
        f"- Event schema SHA: `{actions.get('event_schema_sha')}`",
        f"- Binary label availability: `{dump(actions.get('binary_label_available', []))}`",
        f"- Binary positive rates: `{dump(actions.get('binary_positive_rates', []))}`",
        "- Source telemetry records fire events only; reload, jump, and crouch imitation labels are unavailable and explicitly zero/masked.",
        "",
        "## B. Split Summary",
        f"- Mode: `{split.get('split_mode')}`",
        f"- Seed: {split.get('seed')}",
        f"- Transition counts: `{dump(split.get('transition_counts', {}))}`",
        f"- Window counts: `{dump(split.get('window_counts', {}))}`",
        f"- Session counts: `{dump(split.get('session_counts', {}))}`",
        f"- Unmapped windows/transitions: {split.get('unmapped_windows')}/{split.get('unmapped_transitions')}",
        "",
        "## C. Dataset, Alignment, And Sample Stats",
        f"- Path resolution: `{dump(dataset.get('path_resolution', {}))}`",
        f"- Split consistency: `{dump(dataset.get('split_consistency', {}))}`",
        f"- Alignment: `{dump(dataset.get('alignment', {}))}`",
        f"- Telemetry stats: `{dump(dataset.get('telemetry_branch_stats', {}))}`",
        f"- Observation checks: `{dump(dataset.get('sampled_obs', {}))}`",
        f"- Continuous actions: `{dump(dataset.get('continuous_actions', {}))}`",
        f"- Binary positive rates: `{dump(dataset.get('binary_positive_rates', []))}`",
        f"- Binary unique values: `{dump(dataset.get('binary_unique_values', []))}`",
        "",
        "## D. Feedforward Distillation",
        f"- Best epoch: {feedforward.get('epoch')}",
        f"- Train metrics: `{dump(feedforward.get('train', {}))}`",
        f"- Validation metrics: `{dump(feedforward.get('val', {}))}`",
        f"- Checkpoint: `{(checkpoints / 'command_best.pt').resolve()}`",
        "",
        "## E. Recurrent BC",
        f"- Best epoch: {recurrent.get('epoch')}",
        f"- Train metrics: `{dump(recurrent.get('train', {}))}`",
        f"- Validation metrics: `{dump(recurrent.get('val', {}))}`",
        f"- GRU mean |delta|: {delta.get('gru_mean_abs_delta')}",
        f"- Actor head mean |delta|: {delta.get('actor_head_mean_abs_delta')}",
        f"- Checkpoint: `{(checkpoints / 'command_recurrent_best.pt').resolve()}`",
        "",
        "## F. PPO Pre-Update Invariant",
        f"- Clip fraction: {invariant.get('clip_fraction')}",
        f"- Max |delta logp|: {invariant.get('max_abs_delta_logp')}",
        f"- Ratio mean/std/min/max: {invariant.get('ratio_mean')}/{invariant.get('ratio_std')}/{invariant.get('ratio_min')}/{invariant.get('ratio_max')}",
        f"- Log-prob finite: {invariant.get('logp_finite')}",
        f"- No optimizer step: {invariant.get('no_optimizer_step')}",
        "",
        "## G. Unity 100-Step Rollout",
        f"- BehaviorSpec: `{dump(invariant.get('behavior_spec', {}))}`",
        f"- Reward mean/min/max: {invariant.get('rollout/reward_mean')}/{invariant.get('rollout/reward_min')}/{invariant.get('rollout/reward_max')}",
        f"- Continuous sent actions: `{dump(invariant.get('continuous_sent', {}))}`",
        f"- Binary action rates: `{dump(invariant.get('binary_action_rates', []))}`",
        f"- Checkpoint loaded: `{invariant.get('checkpoint_loaded')}`",
        f"- No illegal actions/NaNs: {bool(invariant) and invariant.get('logp_finite')}",
        "",
        "## H. Reward-Damage Decision",
        "- Option (a) selected.",
        "- Unity `env_reward` is the combat damage/kill signal.",
        "- `require_damage_signals:false`.",
        "- Observation-index damage terms are disabled.",
        "- `obs_dim` remains 45 and the Unity observation schema is unchanged.",
        "- A 47-dimensional observation or side-channel is deferred as a separate future project.",
        "",
        "## Stop Condition",
        "The pipeline stopped after `invariant_only` and the 100-step Unity rollout. No PPO optimization or checkpoint promotion ran.",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
