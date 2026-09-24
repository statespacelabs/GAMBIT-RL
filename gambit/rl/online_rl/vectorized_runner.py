"""Vectorized Unity rollout collection for multi-env PPO.

Manages N independent headless Unity processes, collects observations in
parallel, runs one batched GPU inference, and dispatches actions back.
Each env gets a unique worker_id and base_port.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .reward_builder import RewardBuilder
from .selfplay_runner import (
    make_action_tuple,
    policy_action_to_buffers,
    validate_unity_action_spec,
)
from .vectorized_buffer import VectorizedRolloutBuffer

logger = logging.getLogger(__name__)


def _launch_unity_env(
    env_path: str,
    worker_id: int,
    base_port: int,
    seed: int,
    time_scale: float,
    no_graphics: bool = True,
):
    """Launch a single Unity environment instance."""
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.engine_configuration_channel import (
        EngineConfigurationChannel,
    )

    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(time_scale=time_scale)
    env = UnityEnvironment(
        file_name=env_path if env_path else None,
        worker_id=worker_id,
        base_port=base_port,
        seed=seed,
        side_channels=[engine_channel],
        no_graphics=no_graphics,
    )
    env.reset()
    return env


class VectorizedUnityRunner:
    """Collect rollouts from N independent Unity environments with batched inference."""

    def __init__(
        self,
        env_path: str,
        num_envs: int,
        reward_builder: RewardBuilder,
        device: str = "cpu",
        base_port: int = 5005,
        base_seed: int = 42,
        time_scale: float = 20.0,
        normalizer=None,
        parallel_step: bool = False,
    ):
        if normalizer is None:
            raise ValueError(
                "VectorizedUnityRunner requires an ObservationNormalizer; online "
                "observation normalization must not be disabled during PPO."
            )
        self.num_envs = num_envs
        self.reward_builder = reward_builder
        self.normalizer = normalizer
        self.device = torch.device(device)
        # Serial stepping is the DEFAULT and recommended path. ThreadPool stepping
        # (parallel_step=True) is EXPERIMENTAL and was measured to REGRESS throughput
        # (mlagents per-step protobuf (de)serialization is GIL-bound, so worker
        # threads serialize on the GIL). Use multi-area Unity / multiprocessing for
        # real throughput, not threads.
        self.parallel_step = bool(parallel_step)
        self.envs = []

        logger.info(
            "Launching %d Unity environments (base_port=%d, time_scale=%.1f)...",
            num_envs, base_port, time_scale,
        )
        for i in range(num_envs):
            env = _launch_unity_env(
                env_path=env_path,
                worker_id=i + 1,
                base_port=base_port,
                seed=base_seed + i,
                time_scale=time_scale,
            )
            self.envs.append(env)
            logger.info("  env[%d] launched (worker_id=%d, port=%d, seed=%d)",
                        i, i + 1, base_port, base_seed + i)

        self.behavior_name = list(self.envs[0].behavior_specs.keys())[0]
        self.action_spec = self.envs[0].behavior_specs[self.behavior_name].action_spec
        self.action_contract = validate_unity_action_spec(self.action_spec)
        # Executor only created for the experimental parallel path.
        self.executor = (
            ThreadPoolExecutor(max_workers=max(num_envs, 1))
            if self.parallel_step else None
        )
        logger.info(
            "VectorizedUnityRunner: %d envs, behavior=%s, contract=%s, step_mode=%s",
            num_envs, self.behavior_name, self.action_contract,
            "threadpool(EXPERIMENTAL)" if self.parallel_step else "serial",
        )

    def close(self) -> None:
        if self.executor is not None:
            try:
                self.executor.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass

    def _get_obs_for_env(self, env_idx: int) -> tuple[np.ndarray, int, bool]:
        """Get observation from env, handling decision/terminal steps.

        Returns (obs_45, agent_id, is_waiting). is_waiting=True means
        no decision agent is available yet (need to step again).
        """
        env = self.envs[env_idx]
        decision_steps, terminal_steps = env.get_steps(self.behavior_name)
        if len(decision_steps) > 0:
            agent_id = int(decision_steps.agent_id[0])
            obs = np.asarray(decision_steps[agent_id].obs[0], dtype=np.float32)
            return obs, agent_id, False
        return np.zeros(45, dtype=np.float32), -1, True

    def _step_one_env(self, e, action_env, aid, obs_cur, force_done=False):
        """Run ONE env's set-action + step + get-steps (+ reset on done) in a worker.

        Touches only self.envs[e] (no shared Unity object across threads). Unity
        socket I/O releases the GIL, so concurrent calls overlap. Exceptions
        propagate to the caller as the future result (fail hard with env_id).
        """
        env = self.envs[e]
        continuous, discrete = policy_action_to_buffers(action_env, self.action_spec)
        env.set_action_for_agent(
            self.behavior_name, int(aid), make_action_tuple(continuous, discrete)
        )
        env.step()
        decision_steps, terminal_steps = env.get_steps(self.behavior_name)

        done = False
        if aid in terminal_steps:
            env_reward = float(terminal_steps[aid].reward)
            obs_next = np.asarray(terminal_steps[aid].obs[0], dtype=np.float32)
            done = True
        elif len(decision_steps) > 0 and aid in decision_steps:
            env_reward = float(decision_steps[aid].reward)
            obs_next = np.asarray(decision_steps[aid].obs[0], dtype=np.float32)
        else:
            env_reward = 0.0
            obs_next = obs_cur.copy()
            done = True

        if force_done:
            done = True  # test-mode forced episode boundary (Part C)

        if done:
            env.reset()
            n_obs, n_aid, waiting = self._get_obs_for_env(e)
            guard = 0
            while waiting and guard < 1000:
                env.step()
                n_obs, n_aid, waiting = self._get_obs_for_env(e)
                guard += 1
            if waiting:
                raise RuntimeError(f"env {e} produced no decision agent after reset")
            next_obs, next_aid = n_obs, n_aid
        else:
            next_obs = obs_next
            next_aid = int(decision_steps.agent_id[0]) if len(decision_steps) > 0 else aid

        return {
            "e": e, "env_reward": env_reward, "obs_next": obs_next,
            "done": done, "next_obs": next_obs, "next_aid": next_aid,
        }

    @torch.no_grad()
    def collect_rollouts(
        self,
        policy,
        buffer: VectorizedRolloutBuffer,
        n_steps: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        force_done_every_n: int | None = None,
    ) -> dict[str, float]:
        """Collect n_steps from each env with batched policy inference.

        Returns rollout metrics dict.
        """
        t_start = time.monotonic()
        policy.eval()
        buffer.reset()
        E = self.num_envs

        hidden_states = policy.init_hidden(E, self.device)  # [1, E, 512]
        previous_actions: list[np.ndarray | None] = [None] * E

        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        current_ep_reward = np.zeros(E, dtype=np.float64)
        current_ep_length = np.zeros(E, dtype=np.int64)

        component_sums: defaultdict[str, float] = defaultdict(float)
        component_count = 0

        # Per-env diagnostics: raw obs (pre-normalization), env_reward, dones.
        raw_obs_log = np.zeros((n_steps, E, 45), dtype=np.float32)
        env_reward_log = np.zeros((n_steps, E), dtype=np.float32)
        done_counts = np.zeros(E, dtype=np.int64)

        for env in self.envs:
            env.reset()

        obs_np = np.zeros((E, 45), dtype=np.float32)
        agent_ids = np.zeros(E, dtype=np.int64)
        for e in range(E):
            obs_np[e], agent_ids[e], waiting = self._get_obs_for_env(e)
            while waiting:
                self.envs[e].step()
                obs_np[e], agent_ids[e], waiting = self._get_obs_for_env(e)

        t_infer_total = t_step_total = t_collate_total = 0.0
        force_n = int(force_done_every_n) if force_done_every_n else 0

        for step in range(n_steps):
            raw_obs_log[step] = obs_np  # raw obs for diagnostics (pre-normalization)
            obs_norm_np = self.normalizer.normalize(obs_np)  # [E,45]

            # ---- Batched policy inference (main thread, GPU) ----
            t0 = time.monotonic()
            obs_tensor = torch.from_numpy(obs_norm_np).unsqueeze(1).to(self.device)
            dist, values, new_hidden = policy(obs_tensor, hidden_states)
            action_clamped, action_raw = dist.sample()
            log_probs = dist.log_prob(action_raw)
            action_clamped_np = action_clamped.squeeze(1).cpu().numpy()  # [E,8]
            action_raw_sq = action_raw.squeeze(1)
            values_sq = values.squeeze(1)
            log_probs_sq = log_probs.squeeze(1)
            hidden_before = hidden_states.squeeze(0)
            t_infer_total += time.monotonic() - t0

            buffer.add(
                obs=torch.from_numpy(obs_norm_np).to(self.device),
                actions=action_raw_sq,
                rewards=torch.zeros(E, device=self.device),  # filled below
                dones=torch.zeros(E, device=self.device),
                values=values_sq,
                log_probs=log_probs_sq,
                hiddens=hidden_before,
            )
            hidden_states = new_hidden

            force_done = force_n > 0 and ((step + 1) % force_n == 0)

            # ---- Env step (serial default; threadpool only if explicitly enabled) ----
            t0 = time.monotonic()
            results: list = [None] * E
            if self.parallel_step:
                futures = [
                    self.executor.submit(
                        self._step_one_env, e, action_clamped_np[e],
                        int(agent_ids[e]), obs_np[e], force_done,
                    )
                    for e in range(E)
                ]
                for fut in futures:
                    res = fut.result()  # propagate any env failure (fail hard)
                    results[res["e"]] = res
            else:
                for e in range(E):
                    results[e] = self._step_one_env(
                        e, action_clamped_np[e], int(agent_ids[e]),
                        obs_np[e], force_done,
                    )
            t_step_total += time.monotonic() - t0

            # ---- Collate + reward + buffer fill (main thread) ----
            t0 = time.monotonic()
            rewards_step = np.zeros(E, dtype=np.float32)
            dones_step = np.zeros(E, dtype=np.float32)
            next_obs_np = np.zeros_like(obs_np)
            for e in range(E):
                res = results[e]
                if res is None:
                    raise RuntimeError(f"env {e} produced no step result")
                env_reward = res["env_reward"]
                done = res["done"]
                action_env = action_clamped_np[e]
                shaped_reward, components = self.reward_builder.compute(
                    obs_current=obs_np[e],
                    obs_next=res["obs_next"],
                    action=action_env,
                    previous_action=previous_actions[e],
                    env_reward=env_reward,
                    is_terminal=done,
                )
                for name, val in components.items():
                    component_sums[name] += float(val)
                component_count += 1

                rewards_step[e] = shaped_reward
                dones_step[e] = 1.0 if done else 0.0
                env_reward_log[step, e] = env_reward
                previous_actions[e] = action_env.copy()
                current_ep_reward[e] += env_reward
                current_ep_length[e] += 1

                if done:
                    done_counts[e] += 1
                    episode_rewards.append(float(current_ep_reward[e]))
                    episode_lengths.append(int(current_ep_length[e]))
                    current_ep_reward[e] = 0.0
                    current_ep_length[e] = 0
                    previous_actions[e] = None
                    hidden_states[0, e] = 0.0  # per-env hidden reset only
                next_obs_np[e] = res["next_obs"]
                agent_ids[e] = res["next_aid"]

            t = buffer.pos - 1
            buffer.rewards[t].copy_(torch.from_numpy(rewards_step).to(self.device))
            buffer.dones[t].copy_(torch.from_numpy(dones_step).to(self.device))
            obs_np = next_obs_np
            t_collate_total += time.monotonic() - t0

        last_obs = torch.from_numpy(self.normalizer.normalize(obs_np)).unsqueeze(1).to(self.device)
        last_values = policy.get_value(last_obs, hidden_states).squeeze(1)
        last_dones = torch.from_numpy(dones_step).to(self.device)

        buffer.compute_returns_and_advantages(
            last_values=last_values,
            last_dones=last_dones,
            gamma=gamma,
            lam=gae_lambda,
        )

        t_end = time.monotonic()
        rollout_time = t_end - t_start
        total_steps = n_steps * E
        steps_per_sec = total_steps / max(rollout_time, 1e-6)

        metrics = {
            "rollout/avg_env_reward": float(
                np.mean(episode_rewards) if episode_rewards else 0.0
            ),
            "rollout/avg_length": float(
                np.mean(episode_lengths) if episode_lengths else 0.0
            ),
            "rollout/episodes_completed": float(len(episode_rewards)),
            "rollout/total_env_steps": float(total_steps),
            "rollout/steps_per_sec": steps_per_sec,
            "rollout/env_steps_per_sec": steps_per_sec,
            "rollout/rollout_time_sec": rollout_time,
            "timing/action_inference_sec": t_infer_total,
            "timing/env_step_wall_sec": t_step_total,
            "timing/collate_sec": t_collate_total,
        }
        if component_count:
            metrics.update(
                {
                    f"rollout/{name}": total / component_count
                    for name, total in component_sums.items()
                }
            )

        # Per-env diagnostics for the gate (raw obs, env_reward, dones==hidden resets).
        self.last_diag = {
            "raw_obs_log": raw_obs_log,        # [n_steps, E, 45]
            "env_reward_log": env_reward_log,  # [n_steps, E]
            "done_counts": done_counts,        # [E]
        }
        return metrics
