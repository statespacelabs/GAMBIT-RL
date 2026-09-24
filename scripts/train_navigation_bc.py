#!/usr/bin/env python3
"""Train or extend one Phase 5 recurrent navigator candidate."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.navigation_policy import (  # noqa: E402
    LOCAL45_NORMALIZER_SHA256,
    PARENT_SHA256,
    Phase5Navigator,
    SequenceCorpus,
    branch_by_id,
    evaluate_corpus,
    load_checkpoint,
    loss_terms,
    manifest_npz_paths,
    save_checkpoint,
    sha256_file,
    supplemental_npz_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--branch-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-updates", type=int, required=True)
    parser.add_argument("--resume")
    parser.add_argument("--seed-override", type=int)
    parser.add_argument("--candidate-id")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--actor-normalizer", required=True)
    parser.add_argument("--supplemental-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-batches", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sha256_file(args.checkpoint) != PARENT_SHA256:
        raise RuntimeError("frozen combat checkpoint hash mismatch")
    if sha256_file(args.normalizer) != LOCAL45_NORMALIZER_SHA256:
        raise RuntimeError("frozen local45 normalizer hash mismatch")
    actor_normalizer = json.loads(Path(args.actor_normalizer).read_text())
    if actor_normalizer.get("schema_id") != "phase5_actor_obs_v001":
        raise RuntimeError("actor normalizer schema mismatch")

    config = branch_by_id(args.branch_id)
    if args.seed_override is not None:
        config = replace(config, seed=args.seed_override)
    candidate_id = args.candidate_id or config.branch_id
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested GPU is unavailable")

    train_paths = manifest_npz_paths(args.manifest, validation=False)
    validation_paths = manifest_npz_paths(args.manifest, validation=True)
    train = SequenceCorpus(train_paths, args.sequence_length)
    supplemental_train = None
    if args.supplemental_dir:
        supplemental_train = SequenceCorpus(
            supplemental_npz_paths(args.supplemental_dir, validation=False),
            args.sequence_length,
        )
        validation = SequenceCorpus(
            supplemental_npz_paths(args.supplemental_dir, validation=True),
            args.sequence_length,
        )
    else:
        validation = SequenceCorpus(validation_paths, args.sequence_length)
    if any("gpu7_validation" in str(path) for path in train_paths):
        raise RuntimeError("heldback validation data entered the training corpus")

    if args.resume:
        model, payload, optimizer = load_checkpoint(args.resume, device, load_optimizer=True)
        if model.config.branch_id != config.branch_id:
            # Reuse stages may change only seed/candidate identity, never architecture.
            expected = asdict(config)
            actual = asdict(model.config)
            for key in expected:
                if key not in {"branch_id", "seed"} and expected[key] != actual[key]:
                    raise RuntimeError(f"resume architecture mismatch: {key}")
        start_update = int(payload["update"])
        assert optimizer is not None
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
    else:
        model = Phase5Navigator(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
        start_update = 0
    if args.target_updates <= start_update:
        raise RuntimeError("target updates must exceed resume update")

    rng = random.Random(config.seed + start_update)
    model.train()
    train_accumulator = {
        key: 0.0 for key in ("total", "navigation", "mode", "blend", "corner")
    }
    started = time.perf_counter()
    for update in range(start_update + 1, args.target_updates + 1):
        if supplemental_train is None:
            batch = train.sample(args.batch_size, rng, device)
        else:
            old_count = max(1, args.batch_size // 4)
            old_batch = train.sample(old_count, rng, device)
            new_batch = supplemental_train.sample(
                args.batch_size - old_count, rng, device
            )
            batch = {
                name: torch.cat((old_batch[name], new_batch[name]), dim=0)
                for name in old_batch
            }
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(batch["actor_obs"].float())
        terms = loss_terms(prediction, batch, model.config)
        if not torch.isfinite(terms["total"]):
            raise RuntimeError(f"non-finite loss at update {update}")
        terms["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for key in train_accumulator:
            train_accumulator[key] += float(terms[key].detach())

    validation_metrics = evaluate_corpus(
        model,
        validation,
        device,
        batches=args.validation_batches,
        batch_size=args.batch_size,
        seed=57999,
    )
    duration = time.perf_counter() - started
    updates_run = args.target_updates - start_update
    train_metrics = {
        f"train_{key}_loss": value / updates_run
        for key, value in train_accumulator.items()
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / f"checkpoint_u{args.target_updates:04d}.pt"
    metadata = {
        "candidate_id": candidate_id,
        "branch_id": config.branch_id,
        "training_split": "train_plus_authored_excluding_gpu7_validation",
        "selection_split": "gpu7_validation_heldback",
        "train_samples": train.sample_count,
        "validation_samples": validation.sample_count,
        "supplemental_train_samples": (
            supplemental_train.sample_count if supplemental_train is not None else 0
        ),
        "supplemental_collection_sha256": (
            sha256_file(Path(args.supplemental_dir) / "COLLECTION_RUN.json")
            if args.supplemental_dir else ""
        ),
        "goal6_batch_fraction": 0.25 if supplemental_train is not None else 1.0,
        "dataset_manifest_sha256": sha256_file(args.manifest),
        "actor_normalizer_sha256": sha256_file(args.actor_normalizer),
        "combat_parent_sha256": sha256_file(args.checkpoint),
        "local45_normalizer_sha256": sha256_file(args.normalizer),
        "frozen_combat_parent": True,
        "frozen_local45_normalizer": True,
        "old_encoder_branches_in_optimizer": False,
    }
    save_checkpoint(
        checkpoint_path, model, optimizer, args.target_updates, metadata
    )
    result = {
        "schema_version": "phase5_navigator_training_result_v001",
        "status": "PASS",
        "candidate_id": candidate_id,
        "branch_id": config.branch_id,
        "config": asdict(config),
        "start_update": start_update,
        "target_update": args.target_updates,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "duration_seconds": duration,
        "updates_per_second": updates_run / duration,
        **train_metrics,
        **validation_metrics,
        # Persistence is evaluated by the sweep across three distinct milestones.
        "persistent_mean_saturation_hard_stop": False,
        "device": str(device),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "metadata": metadata,
    }
    (output / "training_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
