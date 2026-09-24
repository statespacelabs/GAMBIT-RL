#!/usr/bin/env python3
"""Initialize or update one Goal 9 fair hunter branch from real-Unity rollouts."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.navigation_ppo import (  # noqa: E402
    CRITIC_DIM,
    branch_by_id,
    conservative_ppo_loss,
    continue_from_goal7,
    league_lane_by_branch_id,
    load_checkpoint,
    peeker_variant_by_id,
    reacquire_variant_by_id,
    save_checkpoint,
    sha256_file,
)
from scripts.navigation_policy import ACTOR_DIM, save_checkpoint as save_navigator  # noqa: E402


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal7-navigator", required=True)
    parser.add_argument("--branch-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--rollout", action="append", default=[])
    parser.add_argument("--stage", default="INIT")
    parser.add_argument("--environment-steps", type=int, required=True)
    parser.add_argument("--total-curriculum-steps", type=int, default=12288)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-id")
    parser.add_argument("--initialize-only", action="store_true")
    return parser.parse_args()


REQUIRED = {
    "actor_obs", "critic_obs", "policy_action", "old_mean", "old_log_std",
    "old_mode", "nav_hidden", "bc_target", "reward", "visible", "intervened",
}


def load_rollouts(paths: list[str]) -> dict[str, np.ndarray]:
    shards: list[dict[str, np.ndarray]] = []
    for raw in paths:
        path = Path(raw)
        with np.load(path) as source:
            if not REQUIRED.issubset(source.files):
                raise RuntimeError(f"rollout arrays missing from {path}")
            shard = {name: np.asarray(source[name]) for name in REQUIRED}
        count = len(shard["actor_obs"])
        if shard["actor_obs"].shape != (count, ACTOR_DIM):
            raise RuntimeError(f"actor shape mismatch in {path}")
        if shard["critic_obs"].shape != (count, CRITIC_DIM):
            raise RuntimeError(f"critic shape mismatch in {path}")
        if not all(np.isfinite(value).all() for value in shard.values()):
            raise RuntimeError(f"non-finite rollout {path}")
        if not np.allclose(shard["actor_obs"], shard["critic_obs"][:, :ACTOR_DIM], atol=1e-5):
            raise RuntimeError(f"critic prefix differs from actor transport in {path}")
        shards.append(shard)
    if not shards:
        raise RuntimeError("at least one rollout is required")
    return {name: np.concatenate([shard[name] for shard in shards]) for name in REQUIRED}


def main() -> int:
    config = args()
    branch = branch_by_id(config.branch_id)
    try:
        reacquire_variant = reacquire_variant_by_id(config.branch_id)
    except KeyError:
        reacquire_variant = None
    try:
        peeker_variant = peeker_variant_by_id(config.branch_id)
    except KeyError:
        peeker_variant = None
    try:
        league_lane = league_lane_by_branch_id(config.branch_id)
    except KeyError:
        league_lane = None
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested GPU is unavailable")
    torch.manual_seed(branch.seed + config.environment_steps)
    np.random.seed(branch.seed + config.environment_steps)
    random.seed(branch.seed + config.environment_steps)
    if config.resume:
        model, prior, optimizer = load_checkpoint(config.resume, device, load_optimizer=True)
        if model.branch.branch_id != branch.branch_id:
            # Reassigned GPUs keep the winning branch definition and only change candidate ID.
            branch = model.branch
        assert optimizer is not None
        start_steps = int(prior["environment_steps"])
        inherited_metadata = dict(prior.get("metadata", {}))
    else:
        model = continue_from_goal7(config.goal7_navigator, branch, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=branch.tactical_lr, weight_decay=1e-4)
        start_steps = 0
        parent_payload = torch.load(config.goal7_navigator, map_location="cpu", weights_only=False)
        inherited_metadata = dict(parent_payload.get("metadata", {}))
    if config.environment_steps < start_steps:
        raise RuntimeError("environment steps cannot go backwards")
    fraction = min(1.0, config.environment_steps / max(1, config.total_curriculum_steps))
    learning_rate = max(3e-5, branch.tactical_lr * (1.0 - 0.70 * fraction))
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    bc_weight = max(branch.bc_floor, branch.bc_start * (1.0 - 0.80 * fraction))

    totals = {name: 0.0 for name in (
        "total", "policy", "value", "bc", "anchor", "tactical", "smoothness",
        "approx_kl", "valid_fraction",
    )}
    accepted_updates = 0
    kl_stops = 0
    started = time.perf_counter()
    if not config.initialize_only:
        arrays = load_rollouts(config.rollout)
        count = len(arrays["actor_obs"])
        rng = np.random.default_rng(branch.seed + config.environment_steps)
        model.train()
        # PPO likelihoods must be recomputed against the exact transported
        # observation. Training-only cue augmentation belongs in collection,
        # not inside the new-policy log-probability used by the KL gate.
        model.navigator.training_augmentation_enabled = False
        for _ in range(config.updates):
            indexes = rng.integers(0, count, size=min(config.batch_size, count))
            batch = {
                name: torch.as_tensor(value[indexes], device=device)
                for name, value in arrays.items()
            }
            # Exactly two PPO epochs per sampled on-policy batch.
            for _epoch in range(2):
                optimizer.zero_grad(set_to_none=True)
                terms = conservative_ppo_loss(model, batch, bc_weight=bc_weight, clip=0.10)
                if not torch.isfinite(terms["total"]):
                    raise RuntimeError("non-finite PPO loss")
                if float(terms["approx_kl"].detach()) > 0.01:
                    kl_stops += 1
                    break
                terms["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                accepted_updates += 1
                for name in totals:
                    totals[name] += float(terms[name].detach())
            if kl_stops:
                break

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidate_id = config.candidate_id or branch.branch_id
    inherited_reacquire_config = inherited_metadata.get("reacquire_config")
    active_reacquire_config = (
        asdict(reacquire_variant) if reacquire_variant is not None
        else inherited_reacquire_config
    )
    inherited_peeker_config = inherited_metadata.get("peeker_config")
    active_peeker_config = (
        asdict(peeker_variant) if peeker_variant is not None
        else inherited_peeker_config
    )
    metadata = {
        "candidate_id": candidate_id,
        "stage": config.stage,
        "goal7_navigator": str(Path(config.goal7_navigator).resolve()),
        "goal7_navigator_sha256": sha256_file(config.goal7_navigator),
        "combat_expert_frozen_external": True,
        "combat_parameters_in_optimizer": False,
        "actor_input_schema": "phase5_actor_obs_v001",
        "critic_input_schema": "phase5_privileged_critic_obs_v001",
        "actor_privileged_fields": False,
        "ppo_clip": 0.10,
        "ppo_epochs": 2,
        "target_kl": 0.01,
        "max_grad_norm": 0.5,
        "bc_auxiliary_weight": bc_weight,
        "bc_auxiliary_nonzero": bc_weight > 0,
        "tactical_learning_rate": learning_rate,
        "rollout_hashes": {str(Path(path).resolve()): sha256_file(path) for path in config.rollout},
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "goal10_reacquisition_training": reacquire_variant is not None,
        "reacquire_config": active_reacquire_config,
        "parent_hunter_frozen_anchor": (
            active_reacquire_config is not None or active_peeker_config is not None
        ),
        "kl_anchor_nonzero": True,
        "goal11_local_geometry_peek_training": peeker_variant is not None,
        "peeker_config": active_peeker_config,
        "aim_shoot_expert_frozen_external": active_peeker_config is not None,
        "per_map_cover_anchors": False,
        "peek_point_files": False,
        "goal12_anchored_league_training": league_lane is not None,
        "league_lane": asdict(league_lane) if league_lane is not None else None,
        "smoothness_coefficient": branch.smoothness_coef,
        "reaction_delay_steps": branch.reaction_delay_steps,
        "opponents_updated_with_learner": False,
    }
    hunter_path = output / f"hunter_s{config.environment_steps:06d}.pt"
    save_checkpoint(hunter_path, model, optimizer, config.environment_steps, metadata)
    export_optimizer = torch.optim.AdamW(model.navigator.parameters(), lr=learning_rate)
    navigator_path = output / f"navigator_s{config.environment_steps:06d}.pt"
    save_navigator(
        navigator_path, model.navigator, export_optimizer,
        config.environment_steps, metadata,
    )
    denominator = max(1, accepted_updates)
    result = {
        "schema_version": "phase5_map_general_hunter_training_result_v001",
        "status": "PASS",
        "candidate_id": candidate_id,
        "branch_id": model.branch.branch_id,
        "stage": config.stage,
        "start_environment_steps": start_steps,
        "environment_steps": config.environment_steps,
        "fixed_environment_step_checkpoint": True,
        "hunter_checkpoint": str(hunter_path.resolve()),
        "hunter_checkpoint_sha256": sha256_file(hunter_path),
        "navigator_checkpoint": str(navigator_path.resolve()),
        "navigator_checkpoint_sha256": sha256_file(navigator_path),
        "requested_updates": 0 if config.initialize_only else config.updates * 2,
        "accepted_updates": accepted_updates,
        "early_kl_stops": kl_stops,
        "mean_approx_kl": totals["approx_kl"] / denominator,
        "mean_losses": {name: totals[name] / denominator for name in totals if name != "approx_kl"},
        "duration_seconds": time.perf_counter() - started,
        "metadata": metadata,
    }
    (output / "training_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
