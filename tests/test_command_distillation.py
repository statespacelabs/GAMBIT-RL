from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from scripts.split_demonstration_sessions import make_splits
from gambit.dataset_preparation.configs import DistillConfig, PPOTrainConfig
from gambit.rl.online_rl.build_telemetry_npz import (
    BUILDER_VERSION,
    _stats_sha,
    _validate_arrays,
    _validate_existing_npz,
)
from gambit.rl.online_rl.core_action_projector import _compute_distillation_loss
from gambit.rl.online_rl.invariant_check import run_invariant_only
from gambit.rl.online_rl.reward_builder import RewardBuilder


class Phase3ProductionTests(unittest.TestCase):
    def test_stats_fingerprint_and_existing_npz_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stats = root / "stats.npz"
            stats.write_bytes(b"production-stats")
            fingerprint = _stats_sha(stats)
            path = root / "window.npz"
            arrays = {
                "where_tel": np.ones((3, 10), dtype=np.float32),
                "view_tel": np.ones((3, 13), dtype=np.float32),
                "rhythm_tel": np.ones((3, 22), dtype=np.float32),
            }
            np.savez(
                path,
                **arrays,
                normalized=np.asarray(True),
                stats_sha=np.asarray(fingerprint),
                builder_version=np.asarray(BUILDER_VERSION),
            )
            valid, reason, loaded = _validate_existing_npz(path, fingerprint, "window")
            self.assertTrue(valid)
            self.assertEqual(reason, "valid")
            self.assertEqual(loaded["where_tel"].shape, (3, 10))

            valid, reason, _ = _validate_existing_npz(
                path, "wrong-fingerprint", "window"
            )
            self.assertFalse(valid)
            self.assertEqual(reason, "wrong_stats_sha")

    def test_builder_validation_rejects_bad_and_zero_arrays(self):
        good = (
            np.ones((2, 10), dtype=np.float32),
            np.ones((2, 13), dtype=np.float32),
            np.ones((2, 22), dtype=np.float32),
        )
        _validate_arrays("window", *good)
        with self.assertRaises(ValueError):
            _validate_arrays("window", good[0][:, :9], good[1], good[2])
        with self.assertRaises(ValueError):
            _validate_arrays(
                "window",
                np.zeros_like(good[0]),
                np.zeros_like(good[1]),
                good[2],
            )

    def test_session_split_propagates_to_windows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transitions = []
            windows = []
            for index in range(30):
                session = f"session-{index:02d}"
                window = f"window-{index:02d}"
                transitions.append(
                    {
                        "transition_id": f"transition-{index}",
                        "session_id": session,
                        "state_window_id": window,
                    }
                )
                windows.append(
                    {
                        "window_id": window,
                        "session_id": session,
                        "tel_npz_path": f"/tmp/{window}.npz",
                    }
                )
            transition_path = root / "transitions.csv"
            window_path = root / "windows.csv"
            out_transition = root / "transitions_split.csv"
            out_window = root / "windows_split.csv"
            pd.DataFrame(transitions).to_csv(transition_path, index=False)
            pd.DataFrame(windows).to_csv(window_path, index=False)
            report = make_splits(
                transition_path,
                window_path,
                out_transition,
                out_window,
                seed=42,
            )
            self.assertEqual(report["split_mode"], "session_id")
            split_transitions = pd.read_csv(out_transition)
            split_windows = pd.read_csv(out_window)
            mapping = split_windows.set_index("window_id")["split"]
            self.assertTrue(
                (
                    split_transitions["state_window_id"].map(mapping)
                    == split_transitions["split"]
                ).all()
            )

    def test_weighted_binary_loss_reports_per_head_metrics(self):
        prediction = torch.zeros(2, 8, requires_grad=True)
        target = torch.tensor(
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1],
            ],
            dtype=torch.float32,
        )
        loss, metrics = _compute_distillation_loss(
            prediction,
            target,
            pos_weight=torch.tensor([3.0, 8.0, 8.0, 8.0]),
        )
        loss.backward()
        self.assertEqual(len(metrics["binary_accuracy_per_head"]), 4)
        self.assertEqual(len(metrics["binary_f1_per_head"]), 4)
        self.assertEqual(len(metrics["predicted_positive_rate_per_head"]), 4)
        self.assertEqual(metrics["target_positive_rate_per_head"], [0.5, 0.5, 0.0, 0.5])
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_env_reward_is_used_without_observation_damage(self):
        builder = RewardBuilder(
            dmg_dealt_idx=None,
            dmg_taken_idx=None,
            env_reward_coef=1.0,
            damage_dealt_coef=0.0,
            damage_taken_coef=0.0,
            kill_coef=0.0,
            death_coef=0.0,
            jerk_coef=0.0,
            spam_coef=0.0,
        )
        obs = np.zeros(45, dtype=np.float32)
        action = np.zeros(8, dtype=np.float32)
        reward, components = builder.compute(
            obs, obs, action, None, env_reward=2.5, is_terminal=False
        )
        self.assertEqual(reward, 2.5)
        self.assertEqual(components["reward/env"], 2.5)
        self.assertFalse(builder.has_damage_signals)

    def test_invariant_stage_rejects_enabled_ppo_before_io(self):
        ppo = PPOTrainConfig(enable_ppo=True)
        distill = DistillConfig()
        with self.assertRaisesRegex(ValueError, "enable_ppo:false"):
            run_invariant_only(ppo, distill)

    def test_production_configs_keep_ppo_off_and_obs_45(self):
        distill = DistillConfig.from_yaml("configs/distill_production.yaml")
        ppo = PPOTrainConfig.from_yaml("configs/ppo_production.yaml")
        self.assertEqual(distill.obs_dim, 45)
        self.assertEqual(distill.action_dim, 8)
        self.assertEqual(distill.binary_pos_weight, [3.0, 8.0, 8.0, 8.0])
        self.assertFalse(ppo.enable_ppo)
        self.assertFalse(ppo.require_damage_signals)
        self.assertIsNone(ppo.damage_dealt_obs_index)
        self.assertIsNone(ppo.damage_taken_obs_index)


if __name__ == "__main__":
    unittest.main()
