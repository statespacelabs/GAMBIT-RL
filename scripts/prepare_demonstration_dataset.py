#!/usr/bin/env python3
"""Phase 2 full dataset pipeline: build source manifest -> sliding window -> latent cache -> transitions.

Runs the Phase 2 data pipeline on the complete rendered dataset on gpu-30.
Uses the best Phase 1 encoder checkpoint.

Usage:
    python3 scripts/prepare_demonstration_dataset.py \
        --rendered-dir artifacts/demonstrations/rendered \
        --output-dir artifacts/demonstrations/transitions \
        --checkpoint checkpoints_v2_large/best_checkpoint.pt \
        --telemetry-stats artifacts/demonstrations/packed/telemetry_stats.npz \
        --joint-manifest artifacts/demonstrations/joint_manifest.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("phase2_full")


def build_source_manifest(
    rendered_dir: Path,
    joint_manifest_path: Path,
    output_source_manifest: Path
) -> None:
    """Scan rendered directory and build a source manifest mapping sessions to webm/json files."""
    logger.info("=" * 60)
    logger.info("STEP 0: Building source manifest from rendered sessions")
    logger.info("=" * 60)

    session_to_split = {}
    if joint_manifest_path.exists() and joint_manifest_path.is_file() and joint_manifest_path.stat().st_size > 0:
        try:
            joint_df = pd.read_csv(joint_manifest_path)
            session_to_split = dict(zip(joint_df["session_id"].astype(str), joint_df["split"].astype(str)))
            logger.info("Loaded split lookup for %d sessions from joint manifest", len(session_to_split))
        except Exception as e:
            logger.warning("Failed to parse joint manifest at %s: %s. Defaulting to 'train'.", joint_manifest_path, e)
    else:
        logger.warning("Joint manifest not found or empty at %s. New sessions will default to 'train' split.", joint_manifest_path)

    rows = []
    # Search recursively for JSON files
    for json_path in sorted(rendered_dir.rglob("*.json")):
        # Skip chunked JSON files if they happen to be in the folder
        if json_path.name.startswith("ID_") or json_path.name.startswith("toy_"):
            continue

        if json_path.name == "SESSION.json":
            session_id = json_path.parent.name
            video_path = json_path.with_name("SESSION.webm")
        else:
            session_id = json_path.stem
            video_path = json_path.with_suffix(".webm")

        if not video_path.exists():
            continue

        split = session_to_split.get(session_id, "train")

        # Read player_id from JSON if possible, skip if invalid JSON
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            player_id = str(data.get("player_id", session_id))
        except Exception as e:
            logger.warning("Skipping session %s because JSON could not be loaded: %s", session_id, e)
            continue

        rows.append({
            "clip_id": session_id,
            "player_id": player_id,
            "session_id": session_id,
            "video_path": str(video_path.resolve()),
            "tel_json_path": str(json_path.resolve()),
            "tel_npz_path": "",
            "split": split,
        })

    if not rows:
        raise ValueError(f"No valid SESSION.json + SESSION.webm pairs found under {rendered_dir}!")

    df = pd.DataFrame(rows)
    output_source_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_source_manifest, index=False)
    logger.info("Source manifest built with %d sessions and saved to %s", len(df), output_source_manifest)
    logger.info("Split distribution: %s", df["split"].value_counts().to_dict())


def step1_window_manifest(
    source_manifest: Path,
    window_manifest: Path,
    window_seconds: float = 5.0,
    stride_seconds: float = 1.0,
    max_windows_per_session: int | None = None
) -> None:
    """Build sliding-window manifest from source sessions."""
    logger.info("=" * 60)
    logger.info("STEP 1: Building sliding-window manifest")
    logger.info("=" * 60)

    from gambit.dataset_preparation.window_manifest import build_phase2_window_manifest

    df = build_phase2_window_manifest(
        source_manifest_path=source_manifest,
        output_manifest_path=window_manifest,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        fps=30.0,
        seq_len=150,
    )

    if max_windows_per_session is not None and max_windows_per_session > 0:
        df = df.groupby("session_id").head(max_windows_per_session).reset_index(drop=True)
        df.to_csv(window_manifest, index=False)
        logger.info("Filtered windows to max %d per session", max_windows_per_session)

    logger.info("Created %d windows", len(df))
    logger.info("Sessions: %d unique sessions", df["session_id"].nunique())
    logger.info("Split distribution: %s", df["split"].value_counts().to_dict())
    logger.info("t_start range: [%.1f, %.1f]", df["t_start"].min(), df["t_start"].max())
    logger.info("Window manifest saved to: %s", window_manifest)


def step2_latent_cache(
    checkpoint: Path,
    window_manifest: Path,
    latent_dir: Path,
    latent_index: Path,
    telemetry_stats: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    amp: bool
) -> None:
    """Run frozen encoder over windows and cache latents on GPU."""
    logger.info("=" * 60)
    logger.info("STEP 2: Caching encoder latents")
    logger.info("=" * 60)

    from gambit.dataset_preparation.latent_cache import EncoderLatentCacheWriter

    writer = EncoderLatentCacheWriter(
        encoder_checkpoint_path=checkpoint,
        window_manifest_path=window_manifest,
        output_latent_dir=latent_dir,
        image_size=(160, 240),
        seq_len=150,
        fps=30.0,
        action_vocab_size=14,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        amp=amp,
        telemetry_stats_path=telemetry_stats,
    )

    t0 = time.time()
    index_df = writer.run()
    elapsed = time.time() - t0

    # Save latent index
    latent_index.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(latent_index, index=False)

    logger.info("Cached %d latents in %.1fs (%.2f s/window)",
                len(index_df), elapsed, elapsed / max(1, len(index_df)))
    logger.info("Latent index saved to: %s", latent_index)


def step3_transitions(
    window_manifest: Path,
    latent_index: Path,
    latent_dir: Path,
    action_dir: Path,
    transition_manifest: Path
) -> None:
    """Build transition manifest with actions and rewards."""
    logger.info("=" * 60)
    logger.info("STEP 3: Building transitions + extracting actions/rewards")
    logger.info("=" * 60)

    from gambit.dataset_preparation.transition_manifest import build_transition_manifest
    from gambit.dataset_preparation.reward_extraction import RewardExtractor
    from gambit.dataset_preparation.action_extraction import ActionExtractor
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np

    reward_extractor = RewardExtractor()
    action_extractor = ActionExtractor()

    t0 = time.time()
    df = build_transition_manifest(
        window_manifest_path=window_manifest,
        latent_index_path=latent_index,
        action_dir=action_dir,
        output_transition_manifest_path=transition_manifest,
        reward_extractor=reward_extractor,
        action_extractor=action_extractor,
        step_seconds=1.0,
        save_actions=True,
    )
    elapsed = time.time() - t0

    logger.info("Built %d transitions in %.1fs", len(df), elapsed)

    # Scan latent files for NaNs in parallel to clean the dataset
    logger.info("Scanning cached latent files for NaNs/Infs...")
    latent_paths = list(latent_dir.rglob("*.npz"))
    
    def check_nan(p):
        try:
            with np.load(p) as d:
                if not np.isfinite(d["z_raw"]).all():
                    return p.name
        except:
            pass
        return None

    t_scan = time.time()
    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(check_nan, latent_paths))
    nan_filenames = {r for r in results if r is not None}
    logger.info("Scanned %d latent files in %.1fs, found %d with NaNs", 
                len(latent_paths), time.time() - t_scan, len(nan_filenames))

    if len(nan_filenames) > 0:
        logger.info("Filtering NaN states from transition manifest...")
        initial_len = len(df)
        
        state_names = df["state_latent_path"].apply(lambda x: Path(str(x)).name)
        next_names = df["next_state_latent_path"].apply(lambda x: Path(str(x)).name)
        nan_mask = state_names.isin(nan_filenames) | next_names.isin(nan_filenames)
        
        df = df[~nan_mask].reset_index(drop=True)
        df.to_csv(transition_manifest, index=False)
        logger.info("Filtered out %d transitions containing NaNs. Cleaned manifest has %d transitions.", 
                    initial_len - len(df), len(df))

    logger.info("Transition manifest saved to: %s", transition_manifest)
    logger.info("Reward stats: mean=%.4f std=%.4f min=%.4f max=%.4f",
                df["reward"].mean(), df["reward"].std(),
                df["reward"].min(), df["reward"].max())
    logger.info("Done count: %d, Timeout count: %d",
                df["done"].sum(), df["timeout"].sum())


def step4_scalers_and_stats(action_dir: Path, transition_manifest: Path, action_scaler: Path, reward_stats: Path) -> None:
    """Fit action scaler and reward stats on the full dataset."""
    logger.info("=" * 60)
    logger.info("STEP 4: Fitting scalers and reward stats")
    logger.info("=" * 60)

    from gambit.dataset_preparation.action_extraction import fit_action_scaler
    from gambit.dataset_preparation.reward_extraction import fit_reward_stats

    # Collect all action .npz paths
    action_paths = sorted(action_dir.rglob("*.npz"))
    logger.info("Found %d action files", len(action_paths))

    fit_action_scaler(
        action_paths=[str(p) for p in action_paths],
        output_scaler_path=action_scaler,
    )
    logger.info("Action scaler saved to: %s", action_scaler)

    reward_stats_dict = fit_reward_stats(
        transition_manifest_path=transition_manifest,
        output_stats_path=reward_stats,
    )
    logger.info("Reward stats saved to: %s", reward_stats)
    logger.info("  Reward stats summary: %s", reward_stats_dict)


def step5_validate(transition_manifest: Path) -> None:
    """Run full dataset validation checks."""
    logger.info("=" * 60)
    logger.info("STEP 5: Validating full dataset")
    logger.info("=" * 60)

    from gambit.dataset_preparation.validate_dataset import validate_phase2_dataset, summarize_transition_manifest

    summary = summarize_transition_manifest(transition_manifest)
    logger.info("High-level summary completed.")

    stats = validate_phase2_dataset(
        transition_manifest_path=transition_manifest,
        strict=True,  # Enforce strict validation checks for production run
    )
    logger.info("Validation completed successfully: %s", stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Full Pipeline")
    parser.add_argument("--rendered-dir", type=str, default="artifacts/demonstrations/rendered",
                        help="Path to rendered sessions directory")
    parser.add_argument("--output-dir", type=str, default="artifacts/demonstrations/transitions",
                        help="Path to output packed directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v2_large/best_checkpoint.pt",
                        help="Path to encoder checkpoint")
    parser.add_argument("--telemetry-stats", type=str, default="artifacts/demonstrations/packed/telemetry_stats.npz",
                        help="Path to telemetry normalization stats")
    parser.add_argument("--joint-manifest", type=str, default="artifacts/demonstrations/joint_manifest.csv",
                        help="Path to joint manifest for split mapping")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for latent cache inference")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (cuda or cpu)")
    parser.add_argument("--amp", type=bool, default=True, help="Use Automatic Mixed Precision")
    parser.add_argument("--window-seconds", type=float, default=5.0, help="Sliding window duration in seconds")
    parser.add_argument("--stride-seconds", type=float, default=1.0, help="Sliding window stride in seconds")
    parser.add_argument("--max-windows-per-session", type=int, default=None,
                        help="Limit number of windows per session (useful for quick testing/debugging)")
    parser.add_argument("--skip-steps", type=str, default="",
                        help="Comma-separated list of step names or numbers to skip (e.g. '0,1,2')")
    args = parser.parse_args()

    rendered_dir = Path(args.rendered_dir)
    output_dir = Path(args.output_dir)
    checkpoint = Path(args.checkpoint)
    telemetry_stats = Path(args.telemetry_stats)
    joint_manifest = Path(args.joint_manifest)

    logger.info("Starting Phase 2 Dataset Generation Pipeline")
    logger.info("  Rendered dir:    %s", rendered_dir)
    logger.info("  Output dir:      %s", output_dir)
    logger.info("  Checkpoint:      %s", checkpoint)
    logger.info("  Telemetry stats: %s", telemetry_stats)

    # Validate inputs
    assert rendered_dir.exists(), f"Rendered directory does not exist: {rendered_dir}"
    assert checkpoint.exists(), f"Encoder checkpoint does not exist: {checkpoint}"
    assert telemetry_stats.exists(), f"Telemetry stats do not exist: {telemetry_stats}"

    # Setup outputs
    manifests_dir = output_dir / "manifests"
    source_manifest = manifests_dir / "source_manifest.csv"
    window_manifest = manifests_dir / "phase2_window_manifest.csv"
    latent_dir = output_dir / "latents"
    latent_index = manifests_dir / "latent_index.csv"
    action_dir = output_dir / "actions"
    transition_manifest = manifests_dir / "phase2_transition_manifest.csv"
    action_scaler = output_dir / "scalers" / "action_scaler.npz"
    reward_stats = output_dir / "rewards" / "reward_stats.json"

    skip_set = {s.strip() for s in args.skip_steps.split(",") if s.strip()}
    t_total = time.time()

    if "0" not in skip_set and "step0" not in skip_set:
        build_source_manifest(rendered_dir, joint_manifest, source_manifest)
    else:
        logger.info("Skipping STEP 0 per skip-steps config.")

    if "1" not in skip_set and "step1" not in skip_set:
        step1_window_manifest(source_manifest, window_manifest, args.window_seconds, args.stride_seconds, args.max_windows_per_session)
    else:
        logger.info("Skipping STEP 1 per skip-steps config.")

    if "2" not in skip_set and "step2" not in skip_set:
        step2_latent_cache(checkpoint, window_manifest, latent_dir, latent_index, telemetry_stats,
                           args.batch_size, args.num_workers, args.device, args.amp)
    else:
        logger.info("Skipping STEP 2 per skip-steps config.")

    if "3" not in skip_set and "step3" not in skip_set:
        step3_transitions(window_manifest, latent_index, latent_dir, action_dir, transition_manifest)
    else:
        logger.info("Skipping STEP 3 per skip-steps config.")

    if "4" not in skip_set and "step4" not in skip_set:
        step4_scalers_and_stats(action_dir, transition_manifest, action_scaler, reward_stats)
    else:
        logger.info("Skipping STEP 4 per skip-steps config.")

    if "5" not in skip_set and "step5" not in skip_set:
        step5_validate(transition_manifest)
    else:
        logger.info("Skipping STEP 5 per skip-steps config.")

    total_time = time.time() - t_total
    logger.info("=" * 60)
    logger.info("DONE! Phase 2 Full Pipeline successfully completed in %.1f seconds", total_time)
    logger.info("=" * 60)
    logger.info("Outputs location: %s", output_dir)
    logger.info("  Source manifest:     %s", source_manifest)
    logger.info("  Window manifest:     %s", window_manifest)
    logger.info("  Latent index:        %s", latent_index)
    logger.info("  Transition manifest: %s", transition_manifest)
    logger.info("  Action scaler:       %s", action_scaler)
    logger.info("  Reward stats:        %s", reward_stats)


if __name__ == "__main__":
    main()
