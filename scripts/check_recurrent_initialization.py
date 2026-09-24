#!/usr/bin/env python3
"""Verify recurrent BC changed GRU and actor parameters from Stage 1 init."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gambit.dataset_preparation.configs import DistillConfig  # noqa: E402
from gambit.rl.online_rl.actor_critic import RecurrentActorCritic  # noqa: E402
from gambit.rl.online_rl.core_action_projector import CoreActionProjector  # noqa: E402
from gambit.rl.online_rl.telemetry_encoder import TelemetryEncoder  # noqa: E402


def mean_abs_delta(
    initial: dict[str, torch.Tensor],
    trained: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> float:
    deltas = [
        (trained[name].cpu() - value.cpu()).abs().reshape(-1)
        for name, value in initial.items()
        if name.startswith(prefixes)
    ]
    if not deltas:
        raise RuntimeError(f"No parameters matched prefixes {prefixes}")
    return float(torch.cat(deltas).mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distill_config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    config = DistillConfig.from_yaml(args.distill_config)
    device = torch.device("cpu")
    stage1_path = Path(config.checkpoint_dir) / "command_best.pt"
    recurrent_path = Path(config.checkpoint_dir) / "command_recurrent_best.pt"
    if not stage1_path.exists() or not recurrent_path.exists():
        raise FileNotFoundError(
            f"Required checkpoints missing: {stage1_path}, {recurrent_path}"
        )

    encoder = TelemetryEncoder.from_encoder_checkpoint(
        config.phase1_ckpt, device=device
    )
    projector = CoreActionProjector(encoder)
    stage1 = torch.load(stage1_path, map_location=device, weights_only=False)
    projector.load_state_dict(stage1["projector_state"], strict=True)
    initialized = RecurrentActorCritic.from_distilled(projector)
    initial_state = initialized.state_dict()

    recurrent = torch.load(recurrent_path, map_location=device, weights_only=False)
    trained_state = recurrent["actor_critic_state"]
    report = {
        "stage1_checkpoint": str(stage1_path.resolve()),
        "recurrent_checkpoint": str(recurrent_path.resolve()),
        "gru_mean_abs_delta": mean_abs_delta(initial_state, trained_state, ("gru.",)),
        "actor_head_mean_abs_delta": mean_abs_delta(
            initial_state,
            trained_state,
            ("actor_cont_mean.", "actor_binary_logits."),
        ),
    }
    if report["gru_mean_abs_delta"] <= 0.0:
        raise RuntimeError("Recurrent checkpoint did not change GRU parameters")
    if report["actor_head_mean_abs_delta"] <= 0.0:
        raise RuntimeError("Recurrent checkpoint did not change actor heads")
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
