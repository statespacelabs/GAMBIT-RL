"""Visual Multi-Area Unity Runner.

Extends MultiAreaUnityRunner to capture visual observations (camera frames)
in addition to telemetry and pass them to a VisualRecurrentActorCritic.

Requires:
    - Unity build compiled with ENABLE_VISUAL_OBS=1 (adds CameraSensorComponent)
    - Policy is a VisualRecurrentActorCritic (or compatible)

Visual obs arrives as obs[1] from ML-Agents: shape [H, W, 3] uint8 per agent.
We resize to the policy's expected input size and normalize to [0, 1] float32.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .multi_area_runner import MultiAreaUnityRunner, _launch_unity_env
from .reward_builder import RewardBuilder
from .selfplay_runner import (
    make_action_tuple,
    policy_action_to_buffers,
    validate_unity_action_spec,
)
from .vectorized_buffer import VectorizedRolloutBuffer

logger = logging.getLogger(__name__)


class VisualMultiAreaUnityRunner(MultiAreaUnityRunner):
    """MultiAreaUnityRunner with visual observation capture."""

    def __init__(
        self,
        env_path: str,
        num_areas: int,
        reward_builder: RewardBuilder,
        device: str = "cpu",
        base_port: int = 5005,
        seed: int = 42,
        time_scale: float = 20.0,
        normalizer=None,
        visual_size: tuple[int, int] = (120, 160),
    ):
        """
        Args:
            visual_size: (H, W) for the visual observation after resizing.
                         Should match the CameraSensor output (120x160 default).
        """
        import os
        self.visual_size = visual_size

        # Override no_graphics for visual mode — need rendering for camera sensor.
        # Requires xvfb-run or a real display.
        if normalizer is None:
            raise ValueError(
                "VisualMultiAreaUnityRunner requires an ObservationNormalizer."
            )
        self.num_areas = num_areas
        self.num_envs = num_areas
        self.reward_builder = reward_builder
        self.normalizer = normalizer
        self.device = torch.device(device)

        logger.info(
            "Launching VISUAL multi-area Unity (num_areas=%d, no_graphics=False)...",
            num_areas,
        )
        os.environ["NUM_AREAS"] = str(num_areas)

        self.env = _launch_unity_env(
            env_path=env_path,
            worker_id=1,
            base_port=base_port,
            seed=seed,
            time_scale=time_scale,
            num_areas=num_areas,
            no_graphics=False,
        )

        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.action_spec = self.env.behavior_specs[self.behavior_name].action_spec
        self.action_contract = validate_unity_action_spec(self.action_spec)

        decision_steps, _ = self.env.get_steps(self.behavior_name)
        discovered = sorted(decision_steps.agent_id.tolist())
        if len(discovered) < num_areas:
            for _ in range(50):
                self.env.step()
                decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
                all_ids = set(decision_steps.agent_id.tolist())
                all_ids.update(terminal_steps.agent_id.tolist())
                if len(all_ids) >= num_areas:
                    discovered = sorted(all_ids)
                    break

        if len(discovered) < num_areas:
            raise RuntimeError(
                f"VisualMultiAreaUnityRunner expected {num_areas} agents but "
                f"found {len(discovered)}"
            )

        self.agent_ids = discovered[:num_areas]
        self.agent_to_area = {aid: i for i, aid in enumerate(self.agent_ids)}

        self._area_pos_offset = np.zeros((num_areas, 45), dtype=np.float32)
        for i in range(num_areas):
            self._area_pos_offset[i, 0] = i * 500.0

        self.envs = [self.env]
        self.last_diag = {}

        obs_spec = self.env.behavior_specs[self.behavior_name].observation_specs
        if len(obs_spec) < 2:
            raise RuntimeError(
                "VisualMultiAreaUnityRunner requires at least 2 observation specs "
                "(telemetry + visual). Got %d. Ensure ENABLE_VISUAL_OBS=1 is set." % len(obs_spec)
            )
        vis_shape = obs_spec[1].shape
        logger.info(
            "VisualMultiAreaUnityRunner: visual obs spec shape=%s, resize_to=%s",
            vis_shape, self.visual_size,
        )

    def _gather_all_obs_visual(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Gather both telemetry and visual observations.

        Returns:
            tel_obs: [N, 45] float32
            vis_obs: [N, 3, H, W] float32 in [0, 1]
            agent_ids: [N] int64
            info: per-agent reward/done dict
        """
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        E = self.num_areas
        H, W = self.visual_size
        tel_obs = np.zeros((E, 45), dtype=np.float32)
        vis_obs = np.zeros((E, 3, H, W), dtype=np.float32)
        info = {}

        for aid in self.agent_ids:
            area = self.agent_to_area[aid]
            if aid in decision_steps:
                tel_obs[area] = np.asarray(decision_steps[aid].obs[0], dtype=np.float32)
                raw_vis = np.asarray(decision_steps[aid].obs[1], dtype=np.float32)
                vis_obs[area] = self._process_visual(raw_vis, H, W)
                info[aid] = {"reward": float(decision_steps[aid].reward), "done": False}
            elif aid in terminal_steps:
                tel_obs[area] = np.asarray(terminal_steps[aid].obs[0], dtype=np.float32)
                raw_vis = np.asarray(terminal_steps[aid].obs[1], dtype=np.float32)
                vis_obs[area] = self._process_visual(raw_vis, H, W)
                info[aid] = {"reward": float(terminal_steps[aid].reward), "done": True}
            else:
                info[aid] = {"reward": 0.0, "done": False}

        return tel_obs, vis_obs, np.array(self.agent_ids, dtype=np.int64), info

    @staticmethod
    def _process_visual(raw: np.ndarray, H: int, W: int) -> np.ndarray:
        """Process raw visual obs from Unity to [3, H, W] float32 in [0, 1].

        Unity ML-Agents CameraSensor outputs [H, W, 3] float32 already in [0,1].
        We just transpose to channel-first format.
        """
        if raw.ndim == 3 and raw.shape[2] == 3:
            # [H_raw, W_raw, 3] -> [3, H_raw, W_raw]
            frame = raw.transpose(2, 0, 1)
        elif raw.ndim == 3 and raw.shape[0] == 3:
            frame = raw
        else:
            frame = np.zeros((3, H, W), dtype=np.float32)
            return frame

        if frame.shape[1] != H or frame.shape[2] != W:
            frame_t = torch.from_numpy(frame).unsqueeze(0)
            frame_t = F.interpolate(frame_t, size=(H, W), mode="bilinear", align_corners=False)
            frame = frame_t.squeeze(0).numpy()

        return frame

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
        """Collect rollouts with visual observations passed to policy."""
        t_start = time.monotonic()
        policy.eval()
        buffer.reset()
        E = self.num_areas
        H, W = self.visual_size

        hidden_states = policy.init_hidden(E, self.device)
        previous_actions: list[np.ndarray | None] = [None] * E

        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        current_ep_reward = np.zeros(E, dtype=np.float64)
        current_ep_length = np.zeros(E, dtype=np.int64)

        component_sums: defaultdict[str, float] = defaultdict(float)
        component_count = 0

        raw_obs_log = np.zeros((n_steps, E, 45), dtype=np.float32)
        env_reward_log = np.zeros((n_steps, E), dtype=np.float32)
        done_counts = np.zeros(E, dtype=np.int64)
        aim_err_log = np.zeros((n_steps, E), dtype=np.float32)
        action_log = np.zeros((n_steps, E, 8), dtype=np.float32)
        shot_counts = np.zeros(E, dtype=np.int64)
        hit_counts = np.zeros(E, dtype=np.int64)
        kill_counts = np.zeros(E, dtype=np.int64)
        pos_reward_counts = np.zeros(E, dtype=np.int64)

        # Visual frame buffer for logging gate stats
        gate_means: list[float] = []

        self.env.reset()

        for _ in range(50):
            decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
            all_ids = set(decision_steps.agent_id.tolist())
            if len(all_ids) >= E:
                self.agent_ids = sorted(all_ids)[:E]
                self.agent_to_area = {aid: i for i, aid in enumerate(self.agent_ids)}
                break
            self.env.step()

        tel_obs_np, vis_obs_np, _, _ = self._gather_all_obs_visual()

        t_infer_total = t_step_total = t_collate_total = 0.0
        force_n = int(force_done_every_n) if force_done_every_n else 0

        for step in range(n_steps):
            raw_obs_log[step] = tel_obs_np
            obs_local = tel_obs_np - self._area_pos_offset
            obs_norm_np = self.normalizer.normalize(obs_local)

            # ---- Batched policy inference with visual frames ----
            t0 = time.monotonic()
            obs_tensor = torch.from_numpy(obs_norm_np).unsqueeze(1).to(self.device)
            vis_tensor = torch.from_numpy(vis_obs_np).unsqueeze(1).to(self.device)

            dist, values, new_hidden = policy(
                obs_tensor, hidden_states, visual_frames=vis_tensor
            )
            action_clamped, action_raw = dist.sample()
            log_probs = dist.log_prob(action_raw)
            action_clamped_np = action_clamped.squeeze(1).cpu().numpy()
            action_raw_sq = action_raw.squeeze(1)
            values_sq = values.squeeze(1)
            log_probs_sq = log_probs.squeeze(1)
            hidden_before = hidden_states.squeeze(0)
            t_infer_total += time.monotonic() - t0

            # Track gate activation
            if hasattr(policy, "visual_fusion") and policy.visual_fusion.last_gate is not None:
                gate_means.append(policy.visual_fusion.gate_mean())

            buffer.add(
                obs=torch.from_numpy(obs_norm_np).to(self.device),
                actions=action_raw_sq,
                rewards=torch.zeros(E, device=self.device),
                dones=torch.zeros(E, device=self.device),
                values=values_sq,
                log_probs=log_probs_sq,
                hiddens=hidden_before,
            )
            hidden_states = new_hidden

            action_log[step] = action_clamped_np
            # ---- Set actions then step ----
            t0 = time.monotonic()
            decision_steps, _ = self.env.get_steps(self.behavior_name)
            for aid in self.agent_ids:
                area = self.agent_to_area[aid]
                if aid in decision_steps:
                    continuous, discrete = policy_action_to_buffers(
                        action_clamped_np[area], self.action_spec,
                    )
                    self.env.set_action_for_agent(
                        self.behavior_name, aid,
                        make_action_tuple(continuous, discrete),
                    )
            step_start = time.monotonic()
            self.env.step()
            step_dur = time.monotonic() - step_start
            t_step_total += step_dur

            if step_dur > 30.0:
                logger.warning(
                    "SLOW_STEP step=%d took %.1fs", step, step_dur,
                )

            # ---- Gather next obs ----
            t0 = time.monotonic()
            next_tel_np, next_vis_np, _, step_info = self._gather_all_obs_visual()

            force_done = force_n > 0 and ((step + 1) % force_n == 0)
            rewards_step = np.zeros(E, dtype=np.float32)
            dones_step = np.zeros(E, dtype=np.float32)

            for aid in self.agent_ids:
                e = self.agent_to_area[aid]
                info = step_info.get(aid, {"reward": 0.0, "done": False})
                env_reward = info["reward"]
                done = info["done"] or force_done

                shaped_reward, components = self.reward_builder.compute(
                    obs_current=tel_obs_np[e],
                    obs_next=next_tel_np[e],
                    action=action_clamped_np[e],
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
                aim_err_log[step, e] = max(0.0, float(tel_obs_np[e, 22]))
                is_shooting = bool(action_clamped_np[e, 4] > 0.5)
                if is_shooting:
                    shot_counts[e] += 1
                if env_reward > 0.05:
                    hit_counts[e] += 1
                if env_reward > 0.5:
                    kill_counts[e] += 1
                if env_reward > 0.0:
                    pos_reward_counts[e] += 1
                previous_actions[e] = action_clamped_np[e].copy()
                current_ep_reward[e] += env_reward
                current_ep_length[e] += 1

                if done:
                    done_counts[e] += 1
                    episode_rewards.append(float(current_ep_reward[e]))
                    episode_lengths.append(int(current_ep_length[e]))
                    current_ep_reward[e] = 0.0
                    current_ep_length[e] = 0
                    previous_actions[e] = None
                    hidden_states[0, e] = 0.0

            t = buffer.pos - 1
            buffer.rewards[t].copy_(torch.from_numpy(rewards_step).to(self.device))
            buffer.dones[t].copy_(torch.from_numpy(dones_step).to(self.device))
            tel_obs_np = next_tel_np
            vis_obs_np = next_vis_np
            t_collate_total += time.monotonic() - t0

            any_done = dones_step.any()
            if any_done or step % 100 == 99:
                self._refresh_agent_ids()

            if step % 500 == 0:
                elapsed = time.monotonic() - t_start
                gm = gate_means[-1] if gate_means else 0.0
                logger.info(
                    "VISUAL_ROLLOUT step=%d/%d elapsed=%.1fs gate_mean=%.4f",
                    step, n_steps, elapsed, gm,
                )

        # Bootstrap value
        last_obs = torch.from_numpy(
            self.normalizer.normalize(tel_obs_np - self._area_pos_offset)
        ).unsqueeze(1).to(self.device)
        last_vis = torch.from_numpy(vis_obs_np).unsqueeze(1).to(self.device)
        last_values = policy.get_value(last_obs, hidden_states, visual_frames=last_vis).squeeze(1)
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
            "rollout/avg_env_reward": float(np.mean(episode_rewards) if episode_rewards else 0.0),
            "rollout/avg_length": float(np.mean(episode_lengths) if episode_lengths else 0.0),
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
            metrics.update({
                f"rollout/{name}": total / component_count
                for name, total in component_sums.items()
            })

        # Visual-specific metrics
        metrics["visual/gate_mean"] = float(np.mean(gate_means)) if gate_means else 0.0
        metrics["visual/gate_max"] = float(np.max(gate_means)) if gate_means else 0.0
        metrics["visual/gate_min"] = float(np.min(gate_means)) if gate_means else 0.0

        # Standard action/aim metrics (same as base class)
        action_names = ["move_x", "move_y", "look_dx", "look_dy", "shoot", "reload", "jump", "crouch"]
        for i, name in enumerate(action_names):
            col = action_log[:, :, i].flatten()
            metrics[f"action/{name}_mean"] = float(col.mean())
            metrics[f"action/{name}_std"] = float(col.std())

        aim_flat = aim_err_log.flatten()
        metrics["aim/err_mean"] = float(aim_flat.mean())
        metrics["aim/err_median"] = float(np.median(aim_flat))
        metrics["aim/total_shots"] = float(shot_counts.sum())
        metrics["aim/total_hits"] = float(hit_counts.sum())
        metrics["aim/total_kills"] = float(kill_counts.sum())
        total_shots = int(shot_counts.sum())
        metrics["aim/hit_rate"] = float(hit_counts.sum()) / max(total_shots, 1)
        metrics["aim/shoot_rate"] = float(total_shots) / max(n_steps * E, 1)

        shoot_mask = action_log[:, :, 4] > 0.5
        aim_when_shoot = aim_err_log[shoot_mask]
        boot_thresh = self.reward_builder.shoot_bootstrap_aim_threshold_deg
        aligned_shots = int((aim_when_shoot < boot_thresh).sum()) if len(aim_when_shoot) > 0 else 0
        metrics["shoot/fired_steps"] = float(total_shots)
        metrics["shoot/aligned_fired"] = float(aligned_shots)
        metrics["shoot/aligned_fired_rate"] = float(aligned_shots) / max(total_shots, 1)

        self.last_diag = {
            "raw_obs_log": raw_obs_log,
            "env_reward_log": env_reward_log,
            "done_counts": done_counts,
            "aim_err_log": aim_err_log,
            "shot_counts": shot_counts,
            "hit_counts": hit_counts,
            "kill_counts": kill_counts,
        }
        return metrics
