"""Vectorized Rollout Buffer for multi-env PPO.

Stores step transitions in [T, E, ...] layout where T = rollout_steps and
E = num_envs.  Computes per-env GAE with proper done masking and yields
recurrent minibatches that never leak across episode boundaries.
"""

from __future__ import annotations

import numpy as np
import torch


class VectorizedRolloutBuffer:
    """Stores on-policy rollouts from E parallel environments."""

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_dim: int = 45,
        action_dim: int = 8,
        hidden_dim: int = 512,
        device: str = "cpu",
    ):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)

        T, E = rollout_steps, num_envs
        self.obs = torch.zeros((T, E, obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((T, E, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((T, E), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((T, E), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((T, E), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros((T, E), dtype=torch.float32, device=self.device)
        self.hiddens = torch.zeros((T, E, hidden_dim), dtype=torch.float32, device=self.device)

        self.returns = torch.zeros((T, E), dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros((T, E), dtype=torch.float32, device=self.device)

        self.pos = 0

    def reset(self) -> None:
        self.pos = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        hiddens: torch.Tensor,
    ) -> None:
        """Add one timestep of data from all envs.

        Args:
            obs: [E, obs_dim]
            actions: [E, action_dim]
            rewards: [E]
            dones: [E]
            values: [E]
            log_probs: [E]
            hiddens: [E, hidden_dim]
        """
        if self.pos >= self.rollout_steps:
            return
        t = self.pos
        self.obs[t].copy_(obs)
        self.actions[t].copy_(actions)
        self.rewards[t].copy_(rewards)
        self.dones[t].copy_(dones)
        self.values[t].copy_(values)
        self.log_probs[t].copy_(log_probs)
        self.hiddens[t].copy_(hiddens)
        self.pos += 1

    @torch.no_grad()
    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        """Compute GAE per-env with done masking.

        Args:
            last_values: [E] value estimates for the state after the final step.
            last_dones: [E] whether each env was done at the final step.
        """
        T = self.pos
        last_gae_lam = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam

        self.returns[:T] = self.advantages[:T] + self.values[:T]

        adv = self.advantages[:T]
        self.advantages[:T] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def recurrent_minibatches(self, seq_len: int, batch_size: int):
        """Yield minibatches of non-overlapping sequences respecting done boundaries.

        Sequences are built independently per env, never crossing episode
        boundaries (done=1 at step t means the episode ending at t is the
        last step included in a chunk starting before t).

        Args:
            seq_len: Length of sequence chunks (e.g. 32).
            batch_size: Number of sequences per minibatch.

        Yields:
            Dictionary with batched sequences.
        """
        T = self.pos
        starts = []  # list of (env_idx, time_start)

        for e in range(self.num_envs):
            episode_start = 0
            for t in range(T):
                episode_ended = bool(self.dones[t, e].item()) or t == T - 1
                if not episode_ended:
                    continue
                episode_end = t + 1
                for s in range(episode_start, episode_end - seq_len + 1, seq_len):
                    starts.append((e, s))
                episode_start = episode_end

        if not starts:
            return

        indices = np.arange(len(starts))
        np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i: i + batch_size]
            batch_starts = [starts[j] for j in batch_idx]

            b_obs = torch.stack([self.obs[s: s + seq_len, e] for e, s in batch_starts])
            b_actions = torch.stack([self.actions[s: s + seq_len, e] for e, s in batch_starts])
            b_values = torch.stack([self.values[s: s + seq_len, e] for e, s in batch_starts])
            b_returns = torch.stack([self.returns[s: s + seq_len, e] for e, s in batch_starts])
            b_log_probs = torch.stack([self.log_probs[s: s + seq_len, e] for e, s in batch_starts])
            b_advantages = torch.stack([self.advantages[s: s + seq_len, e] for e, s in batch_starts])
            b_hiddens = torch.stack([self.hiddens[s, e] for e, s in batch_starts]).unsqueeze(0)

            yield {
                "obs": b_obs,           # [B, T, 45]
                "actions": b_actions,    # [B, T, 8]
                "values": b_values,      # [B, T]
                "returns": b_returns,    # [B, T]
                "log_probs": b_log_probs,  # [B, T]
                "advantages": b_advantages,  # [B, T]
                "hiddens": b_hiddens,    # [1, B, 512]
            }
