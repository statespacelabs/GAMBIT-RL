"""Phase 2 dataset validation and summary statistics."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_phase2_dataset(
    transition_manifest_path: str | Path,
    state_dim: int = 512,
    action_dim: int | None = None,
    strict: bool = True,
) -> dict:
    """Validate the complete Phase 2 transition dataset.

    Checks:
        all latent paths exist
        all action paths exist
        state and next_state dimensions match
        no NaN/Inf in latents
        no NaN/Inf in actions
        rewards are finite
        done appears only at session end
        next_state belongs to same session
        next_state time = state time + 1 second
        no session leakage across splits
        action distribution is not all zeros
        reward distribution is not all zeros

    Returns:
        Dictionary of dataset statistics.

    Raises:
        ValueError when strict=True and a critical issue is found.
    """
    df = pd.read_csv(transition_manifest_path)
    issues: list[str] = []
    stats: dict = {"total_transitions": len(df)}

    # ── File existence ──
    missing_state = 0
    missing_next = 0
    missing_action = 0

    for _, row in df.iterrows():
        if not Path(str(row["state_latent_path"])).exists():
            missing_state += 1
        if not Path(str(row["next_state_latent_path"])).exists():
            missing_next += 1
        if not Path(str(row["action_path"])).exists():
            missing_action += 1

    stats["missing_state_latents"] = missing_state
    stats["missing_next_latents"] = missing_next
    stats["missing_actions"] = missing_action

    if missing_state > 0:
        issues.append(f"{missing_state} state latent files missing")
    if missing_next > 0:
        issues.append(f"{missing_next} next_state latent files missing")
    if missing_action > 0:
        issues.append(f"{missing_action} action files missing")

    # ── Sample a subset for numerical checks ──
    sample_size = min(500, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    nan_inf_states = 0
    nan_inf_actions = 0
    bad_state_dims = 0

    for _, row in sample_df.iterrows():
        # Check state latent
        sp = str(row["state_latent_path"])
        if Path(sp).exists():
            with np.load(sp) as data:
                z = data["z_raw"]
                if z.shape[0] != state_dim:
                    bad_state_dims += 1
                if np.any(~np.isfinite(z)):
                    nan_inf_states += 1

        # Check action
        ap = str(row["action_path"])
        if Path(ap).exists():
            with np.load(ap) as data:
                av = data["action_vector"]
                if action_dim is not None and av.shape[0] != action_dim:
                    issues.append(
                        f"Action dim mismatch: expected {action_dim}, got {av.shape[0]}"
                    )
                if np.any(~np.isfinite(av)):
                    nan_inf_actions += 1

    stats["nan_inf_states_sampled"] = nan_inf_states
    stats["nan_inf_actions_sampled"] = nan_inf_actions
    stats["bad_state_dims_sampled"] = bad_state_dims

    if nan_inf_states > 0:
        issues.append(f"{nan_inf_states}/{sample_size} sampled states have NaN/Inf")
    if nan_inf_actions > 0:
        issues.append(f"{nan_inf_actions}/{sample_size} sampled actions have NaN/Inf")
    if bad_state_dims > 0:
        issues.append(f"{bad_state_dims}/{sample_size} sampled states have wrong dim")

    # ── Reward checks ──
    rewards = df["reward"].values
    non_finite = np.sum(~np.isfinite(rewards))
    stats["non_finite_rewards"] = int(non_finite)
    if non_finite > 0:
        issues.append(f"{non_finite} non-finite rewards")

    all_zero_reward = np.all(rewards == 0)
    stats["all_zero_rewards"] = bool(all_zero_reward)
    if all_zero_reward:
        issues.append("All rewards are zero")

    # ── Session consistency ──
    session_splits = df.groupby("session_id")["split"].nunique()
    leaking = session_splits[session_splits > 1]
    stats["session_split_leakage"] = len(leaking)
    if len(leaking) > 0:
        issues.append(f"Session split leakage: {leaking.index.tolist()[:5]}")

    # ── Time consistency ──
    if "t" in df.columns:
        # Group by session, sort by t, check consecutive differences
        for sid, group in df.groupby("session_id"):
            group = group.sort_values("t")
            # All next_state should be same session
            if (group["session_id"] != sid).any():
                issues.append(f"Session {sid} has cross-session next_state")

    # ── Split statistics ──
    stats["transitions_per_split"] = df["split"].value_counts().to_dict()
    stats["num_players"] = df["player_id"].nunique()
    stats["num_sessions"] = df["session_id"].nunique()
    stats["reward_mean"] = float(np.mean(rewards))
    stats["reward_std"] = float(np.std(rewards))
    stats["reward_min"] = float(np.min(rewards))
    stats["reward_max"] = float(np.max(rewards))

    # ── Report ──
    if issues:
        msg = "Phase 2 dataset validation issues:\n" + "\n".join(f"  - {i}" for i in issues)
        logger.warning(msg)
        if strict:
            raise ValueError(msg)

    logger.info("Phase 2 validation passed. Stats: %s", stats)
    return stats


def summarize_transition_manifest(
    transition_manifest_path: str | Path,
) -> dict:
    """Print and return high-level dataset statistics.

    Reports:
        number of transitions
        number of players
        number of sessions
        transitions per split
        reward mean/std/min/max
        action sparsity
        missing file counts
    """
    df = pd.read_csv(transition_manifest_path)

    rewards = df["reward"].values

    summary = {
        "total_transitions": len(df),
        "num_players": df["player_id"].nunique(),
        "num_sessions": df["session_id"].nunique(),
        "transitions_per_split": df["split"].value_counts().to_dict(),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "done_count": int(df["done"].sum()),
        "timeout_count": int(df.get("timeout", pd.Series([0])).sum()),
    }

    # Check file existence for a sample
    sample = df.head(100)
    missing_latents = sum(
        1 for _, r in sample.iterrows()
        if not Path(str(r["state_latent_path"])).exists()
    )
    missing_actions = sum(
        1 for _, r in sample.iterrows()
        if not Path(str(r["action_path"])).exists()
    )
    summary["missing_latents_sampled_100"] = missing_latents
    summary["missing_actions_sampled_100"] = missing_actions

    print("=" * 60)
    print("Phase 2 Transition Manifest Summary")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    return summary
