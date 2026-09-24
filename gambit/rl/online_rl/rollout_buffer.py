"""Recurrent Rollout Buffer for PPO.

Stores step transitions (obs_45, action_8, reward, done, value, log_prob, hidden_512).
Yields contiguous sequence chunks for recurrent training.
"""

from __future__ import annotations

import numpy as np
import torch


class RecurrentRolloutBuffer:
    """Stores on-policy rollouts and generates minibatches of sequences."""

    def __init__(
        self,
        size: int,
        obs_dim: int = 45,
        action_dim: int = 8,
        hidden_dim: int = 512,
        device: str = "cpu",
    ):
        self.size = size
        self.device = torch.device(device)

        # Preallocate memory
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(
            (size, action_dim), dtype=torch.float32, device=self.device
        )
        self.rewards = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros(size, dtype=torch.float32, device=self.device)
        # Store initial hidden states for recurrent training
        self.hiddens = torch.zeros(
            (size, hidden_dim), dtype=torch.float32, device=self.device
        )

        self.returns = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros(size, dtype=torch.float32, device=self.device)

        self.pos = 0
        self.full = False

    def reset(self) -> None:
        """Reset the buffer."""
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        hidden: torch.Tensor,
    ) -> None:
        """Add a transition to the buffer."""
        if self.pos >= self.size:
            self.full = True
            return

        self.obs[self.pos].copy_(obs.squeeze())
        self.actions[self.pos].copy_(action.squeeze())
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = 1.0 if done else 0.0
        self.values[self.pos].copy_(value.squeeze())
        self.log_probs[self.pos].copy_(log_prob.squeeze())

        # Store the hidden state *before* taking the action
        # hidden is typically [1, B, H], we want [H]
        self.hiddens[self.pos].copy_(hidden.squeeze())

        self.pos += 1
        if self.pos == self.size:
            self.full = True

    def compute_returns_and_advantages(
        self, last_value: torch.Tensor, gamma: float = 0.99, lam: float = 0.95
    ) -> None:
        """Compute GAE returns and advantages.

        Args:
            last_value: Value estimate for the state after the final step.
        """
        last_gae_lam = 0.0
        T = self.pos

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            self.advantages[t] = last_gae_lam = (
                delta + gamma * lam * next_non_terminal * last_gae_lam
            )

        self.returns[:T] = self.advantages[:T] + self.values[:T]

        # Normalize advantages at the rollout level
        adv = self.advantages[:T]
        self.advantages[:T] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def recurrent_minibatches(self, seq_len: int, batch_size: int):
        """Yield minibatches of non-overlapping sequences.

        Args:
            seq_len: Length of sequence chunks (e.g. 32)
            batch_size: Number of sequences per minibatch (e.g. 16 sequences = 512 steps)

        Yields:
            Dictionary containing batched sequences.
        """
        T = self.pos

        # Build non-overlapping chunks independently inside each episode.
        starts = []
        episode_start = 0
        for step in range(T):
            episode_ended = bool(self.dones[step].item()) or step == T - 1
            if not episode_ended:
                continue
            episode_end = step + 1
            starts.extend(range(episode_start, episode_end - seq_len + 1, seq_len))
            episode_start = episode_end

        starts = np.array(starts)
        np.random.shuffle(starts)

        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i : i + batch_size]

            # If we don't have a full batch, skip or yield smaller
            # For simplicity, we just yield the remaining

            # Gather sequences
            b_obs = torch.stack([self.obs[s : s + seq_len] for s in batch_starts])
            b_actions = torch.stack(
                [self.actions[s : s + seq_len] for s in batch_starts]
            )
            b_values = torch.stack([self.values[s : s + seq_len] for s in batch_starts])
            b_returns = torch.stack(
                [self.returns[s : s + seq_len] for s in batch_starts]
            )
            b_log_probs = torch.stack(
                [self.log_probs[s : s + seq_len] for s in batch_starts]
            )
            b_advantages = torch.stack(
                [self.advantages[s : s + seq_len] for s in batch_starts]
            )

            # Initial hidden states for each sequence: [1, B, H]
            b_hiddens = torch.stack([self.hiddens[s] for s in batch_starts]).unsqueeze(
                0
            )

            yield {
                "obs": b_obs,  # [B, T, 45]
                "actions": b_actions,  # [B, T, 8]
                "values": b_values,  # [B, T]
                "returns": b_returns,  # [B, T]
                "log_probs": b_log_probs,  # [B, T]
                "advantages": b_advantages,  # [B, T]
                "hiddens": b_hiddens,  # [1, B, 512]
            }
