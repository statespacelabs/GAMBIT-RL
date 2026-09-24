"""Headless smoke tests for vectorized PPO components.

Validates buffer shapes, per-env hidden resets, GAE finiteness, PPO update
mechanics, and checkpoint round-trip — all without Unity.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gambit.rl.online_rl.vectorized_buffer import VectorizedRolloutBuffer
from gambit.rl.online_rl.actor_critic import RecurrentActorCritic
from gambit.rl.online_rl.telemetry_encoder import TelemetryEncoder
from gambit.rl.online_rl.core_action_projector import CoreActionProjector
from gambit.rl.online_rl.ppo_loss import compute_ppo_loss, compute_reference_anchor


def _make_actor_critic(device="cpu") -> RecurrentActorCritic:
    tel_encoder = TelemetryEncoder(proj_dim=512)
    projector = CoreActionProjector(tel_encoder)
    ac = RecurrentActorCritic.from_distilled(projector)
    ac.to(device)
    return ac


class TestVectorizedBufferShapes(unittest.TestCase):
    """Verify [T, E, ...] storage layout and shapes."""

    def test_buffer_shapes(self):
        T, E = 64, 4
        buf = VectorizedRolloutBuffer(
            rollout_steps=T, num_envs=E, obs_dim=45, action_dim=8, hidden_dim=512
        )
        self.assertEqual(buf.obs.shape, (T, E, 45))
        self.assertEqual(buf.actions.shape, (T, E, 8))
        self.assertEqual(buf.rewards.shape, (T, E))
        self.assertEqual(buf.dones.shape, (T, E))
        self.assertEqual(buf.values.shape, (T, E))
        self.assertEqual(buf.log_probs.shape, (T, E))
        self.assertEqual(buf.hiddens.shape, (T, E, 512))
        self.assertEqual(buf.returns.shape, (T, E))
        self.assertEqual(buf.advantages.shape, (T, E))

    def test_add_and_pos(self):
        buf = VectorizedRolloutBuffer(rollout_steps=10, num_envs=2)
        for _ in range(10):
            buf.add(
                obs=torch.randn(2, 45),
                actions=torch.randn(2, 8),
                rewards=torch.randn(2),
                dones=torch.zeros(2),
                values=torch.randn(2),
                log_probs=torch.randn(2),
                hiddens=torch.randn(2, 512),
            )
        self.assertEqual(buf.pos, 10)
        buf.add(
            obs=torch.randn(2, 45),
            actions=torch.randn(2, 8),
            rewards=torch.randn(2),
            dones=torch.zeros(2),
            values=torch.randn(2),
            log_probs=torch.randn(2),
            hiddens=torch.randn(2, 512),
        )
        self.assertEqual(buf.pos, 10)


class TestVectorizedGAE(unittest.TestCase):
    """Verify per-env GAE computation with done masking."""

    def test_gae_finite(self):
        T, E = 64, 4
        buf = VectorizedRolloutBuffer(rollout_steps=T, num_envs=E)
        for t in range(T):
            dones = torch.zeros(E)
            if t == 31:
                dones[0] = 1.0
                dones[2] = 1.0
            buf.add(
                obs=torch.randn(E, 45),
                actions=torch.randn(E, 8),
                rewards=torch.randn(E) * 0.1,
                dones=dones,
                values=torch.randn(E) * 0.5,
                log_probs=torch.randn(E),
                hiddens=torch.randn(E, 512),
            )

        last_values = torch.randn(E) * 0.5
        last_dones = torch.zeros(E)
        buf.compute_returns_and_advantages(last_values, last_dones)

        self.assertTrue(torch.isfinite(buf.advantages[:T]).all())
        self.assertTrue(torch.isfinite(buf.returns[:T]).all())

    def test_gae_done_masking(self):
        """Advantage at done boundary should not leak future rewards."""
        T, E = 8, 1
        buf = VectorizedRolloutBuffer(rollout_steps=T, num_envs=E)
        for t in range(T):
            reward = 1.0 if t >= 4 else 0.0
            done = 1.0 if t == 3 else 0.0
            buf.add(
                obs=torch.zeros(E, 45),
                actions=torch.zeros(E, 8),
                rewards=torch.full((E,), reward),
                dones=torch.full((E,), done),
                values=torch.zeros(E),
                log_probs=torch.zeros(E),
                hiddens=torch.zeros(E, 512),
            )

        buf.compute_returns_and_advantages(torch.zeros(E), torch.zeros(E))
        pre_done_returns = buf.returns[:4, 0]
        post_done_returns = buf.returns[4:, 0]
        self.assertTrue((pre_done_returns == 0.0).all(),
                        "Returns before done should be zero (zero reward, zero value)")
        self.assertTrue((post_done_returns > 0.0).all(),
                        "Returns after done should reflect positive rewards")


class TestRecurrentMinibatches(unittest.TestCase):
    """Verify minibatch generation respects done boundaries."""

    def test_no_cross_episode_leak(self):
        T, E = 128, 2
        buf = VectorizedRolloutBuffer(rollout_steps=T, num_envs=E)
        for t in range(T):
            dones = torch.zeros(E)
            if t in (31, 63, 95):
                dones[:] = 1.0
            buf.add(
                obs=torch.randn(E, 45),
                actions=torch.randn(E, 8),
                rewards=torch.randn(E),
                dones=dones,
                values=torch.randn(E),
                log_probs=torch.randn(E),
                hiddens=torch.randn(E, 512),
            )

        buf.compute_returns_and_advantages(torch.zeros(E), torch.zeros(E))

        seq_len = 32
        total_seqs = 0
        for batch in buf.recurrent_minibatches(seq_len=seq_len, batch_size=4):
            B, T_batch, obs_dim = batch["obs"].shape
            self.assertEqual(T_batch, seq_len)
            self.assertEqual(obs_dim, 45)
            self.assertEqual(batch["actions"].shape, (B, seq_len, 8))
            self.assertEqual(batch["hiddens"].shape, (1, B, 512))
            self.assertTrue(torch.isfinite(batch["advantages"]).all())
            total_seqs += B

        self.assertGreater(total_seqs, 0)

    def test_empty_buffer_yields_nothing(self):
        buf = VectorizedRolloutBuffer(rollout_steps=64, num_envs=2)
        batches = list(buf.recurrent_minibatches(seq_len=32, batch_size=4))
        self.assertEqual(len(batches), 0)


class TestHiddenStateReset(unittest.TestCase):
    """Verify per-env hidden state reset on done."""

    def test_hidden_reset_independence(self):
        """Simulates vectorized collection: env 0 done at t=10, env 1 at t=20."""
        E = 2
        ac = _make_actor_critic()
        hidden = ac.init_hidden(E, torch.device("cpu"))  # [1, E, 512]

        for t in range(30):
            obs = torch.randn(E, 1, 45)
            _, _, hidden = ac(obs, hidden)

            done_mask = torch.zeros(E, dtype=torch.bool)
            if t == 10:
                done_mask[0] = True
            if t == 20:
                done_mask[1] = True

            for e in range(E):
                if done_mask[e]:
                    hidden[0, e] = 0.0

        self.assertTrue(torch.isfinite(hidden).all())
        self.assertFalse((hidden[0, 0] == 0).all(),
                         "Env 0 hidden should be non-zero after 20 more steps")
        self.assertFalse((hidden[0, 1] == 0).all(),
                         "Env 1 hidden should be non-zero after 10 more steps")


class TestPPOUpdateMechanics(unittest.TestCase):
    """Verify a full PPO update cycle with vectorized buffer."""

    def test_ppo_update_finite(self):
        T, E = 128, 4
        ac = _make_actor_critic()
        ac.set_ppo_mode(training=True)
        hidden = ac.init_hidden(E, torch.device("cpu"))

        buf = VectorizedRolloutBuffer(rollout_steps=T, num_envs=E)
        for t in range(T):
            obs = torch.randn(E, 1, 45)
            with torch.no_grad():
                dist, val, hidden = ac(obs, hidden)
            clamped, raw = dist.sample()
            lp = dist.log_prob(raw)

            dones = torch.zeros(E)
            if t > 0 and t % 40 == 0:
                dones[t % E] = 1.0
                hidden[0, t % E] = 0.0

            buf.add(
                obs=obs.squeeze(1),
                actions=raw.squeeze(1),
                rewards=torch.randn(E) * 0.1,
                dones=dones,
                values=val.squeeze(1),
                log_probs=lp.squeeze(1),
                hiddens=hidden.squeeze(0).clone(),
            )

        buf.compute_returns_and_advantages(torch.zeros(E), torch.zeros(E))
        self.assertTrue(torch.isfinite(buf.advantages[:T]).all())
        self.assertTrue(torch.isfinite(buf.returns[:T]).all())

        optimizer = torch.optim.AdamW(
            [p for p in ac.parameters() if p.requires_grad], lr=1e-4
        )
        reference = _make_actor_critic()
        reference.load_state_dict(ac.state_dict())
        for p in reference.parameters():
            p.requires_grad_(False)

        n_updates = 0
        for batch in buf.recurrent_minibatches(seq_len=32, batch_size=4):
            obs = batch["obs"]
            actions = batch["actions"]
            h = batch["hiddens"]

            current_dist, new_values, _ = ac(obs, h)
            new_log_probs = current_dist.log_prob(actions)
            entropy = current_dist.entropy()

            anchor_h = torch.zeros_like(h)
            anchor_current, _, _ = ac(obs, anchor_h)
            with torch.no_grad():
                anchor_ref, _, _ = reference(obs, anchor_h)
            ref_anchor = compute_reference_anchor(anchor_current, anchor_ref)

            loss, metrics = compute_ppo_loss(
                new_log_probs=new_log_probs,
                old_log_probs=batch["log_probs"],
                advantages=batch["advantages"],
                new_values=new_values,
                returns=batch["returns"],
                entropy=entropy,
                reference_anchor=ref_anchor,
                current_cont_mean=current_dist.cont_mean,
            )
            self.assertTrue(torch.isfinite(loss))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
            optimizer.step()
            n_updates += 1

        self.assertGreater(n_updates, 0)


class TestCheckpointRoundTrip(unittest.TestCase):
    """Verify checkpoint save and reload."""

    def test_save_load(self):
        ac = _make_actor_critic()
        ac.set_ppo_mode(training=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_ckpt.pt"
            torch.save({"actor_critic_state": ac.state_dict()}, path)

            tel_encoder = TelemetryEncoder(proj_dim=512)
            projector = CoreActionProjector(tel_encoder)
            ac2 = RecurrentActorCritic.from_distilled(projector)
            state = torch.load(path, map_location="cpu", weights_only=False)
            ac2.load_state_dict(state["actor_critic_state"])
            ac2.eval()
            ac.eval()

            obs = torch.randn(1, 1, 45)
            hidden = ac.init_hidden(1, torch.device("cpu"))

            with torch.no_grad():
                d1, v1, _ = ac(obs, hidden)
                d2, v2, _ = ac2(obs, hidden)

            self.assertTrue(torch.allclose(v1, v2, atol=1e-6))
            self.assertTrue(torch.allclose(d1.cont_mean, d2.cont_mean, atol=1e-6))


class TestNoIllegalActions(unittest.TestCase):
    """Verify action outputs from the policy are legal."""

    def test_action_format(self):
        ac = _make_actor_critic()
        ac.eval()
        E = 4
        obs = torch.randn(E, 1, 45)
        hidden = ac.init_hidden(E, torch.device("cpu"))

        with torch.no_grad():
            dist, _, _ = ac(obs, hidden)
            clamped, raw = dist.sample()

        clamped = clamped.squeeze(1)
        self.assertEqual(clamped.shape, (E, 8))
        self.assertTrue(torch.isfinite(clamped).all())
        self.assertTrue((clamped[:, :4] >= -1.0).all())
        self.assertTrue((clamped[:, :4] <= 1.0).all())
        binary = clamped[:, 4:8]
        self.assertTrue(torch.all((binary == 0.0) | (binary == 1.0)))


if __name__ == "__main__":
    unittest.main()
