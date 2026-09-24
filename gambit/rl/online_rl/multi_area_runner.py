"""Multi-area Unity rollout collection for single-process multi-agent PPO.

Instead of spawning N separate Unity processes (VectorizedUnityRunner), this
runner launches ONE Unity process that contains N independent arenas internally.
ML-Agents batches all agents into a single decision_steps / terminal_steps
call, giving:
  - ONE env.step() per timestep (not N)
  - zero inter-process communication overhead
  - lower memory footprint
  - better throughput scaling

The Unity build must be compiled with NUM_AREAS=N support (see
GameModeBootstrapper.cs multi-area mode).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .reward_builder import RewardBuilder
from .selfplay_runner import (
    make_action_tuple,
    policy_action_to_buffers,
    validate_unity_action_spec,
)
from .vectorized_buffer import VectorizedRolloutBuffer
from .weapon_state_summary import aggregate_weapon_state_metrics

logger = logging.getLogger(__name__)


def _launch_unity_env(
    env_path: str,
    worker_id: int,
    base_port: int,
    seed: int,
    time_scale: float,
    num_areas: int,
    no_graphics: bool = True,
):
    """Launch a single Unity environment with NUM_AREAS arenas."""
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.engine_configuration_channel import (
        EngineConfigurationChannel,
    )

    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(time_scale=time_scale)

    env_vars = os.environ.copy()
    env_vars["NUM_AREAS"] = str(num_areas)

    env = UnityEnvironment(
        file_name=env_path if env_path else None,
        worker_id=worker_id,
        base_port=base_port,
        seed=seed,
        side_channels=[engine_channel],
        no_graphics=no_graphics,
        additional_args=["--num-areas", str(num_areas)],
    )
    env.reset()
    return env


class MultiAreaUnityRunner:
    """Collect rollouts from N arenas inside ONE Unity process."""

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
        shot_debug_enabled: bool = False,
        shot_debug_sample_limit_per_area: int = 20,
        shot_debug_log_path: str | None = None,
        shot_debug_true_unity: bool = False,
        shot_debug_unity_log_path: str | None = None,
        opponent_hp_frac_obs_index: int | None = 39,
        target_hurtbox_inflate: float = 0.0,
        learner_shoot_ray_mode: str = "current",
        ray_correction_alpha: float = 0.0,
        ray_correction_require_los: bool = True,
        ray_correction_max_angle_deg: float = 30.0,
        log_weapon_state_debug: bool = False,
        weapon_state_debug_log_path: str | None = None,
    ):
        if normalizer is None:
            raise ValueError(
                "MultiAreaUnityRunner requires an ObservationNormalizer."
            )
        self.num_areas = num_areas
        self.num_envs = num_areas
        self.reward_builder = reward_builder
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.shot_debug_enabled = shot_debug_enabled
        self.shot_debug_sample_limit_per_area = shot_debug_sample_limit_per_area
        self.shot_debug_log_path = shot_debug_log_path
        self.shot_debug_true_unity = shot_debug_true_unity
        self.shot_debug_unity_log_path = shot_debug_unity_log_path
        self.opponent_hp_frac_obs_index = opponent_hp_frac_obs_index
        self.target_hurtbox_inflate = target_hurtbox_inflate
        self.learner_shoot_ray_mode = learner_shoot_ray_mode or "current"
        self.ray_correction_alpha = ray_correction_alpha
        self.ray_correction_require_los = ray_correction_require_los
        self.ray_correction_max_angle_deg = ray_correction_max_angle_deg
        self.log_weapon_state_debug = log_weapon_state_debug
        self.weapon_state_debug_log_path = weapon_state_debug_log_path
        self._env_path = env_path
        self._base_port = base_port
        self._seed = seed
        self._time_scale = time_scale

        logger.info(
            "Launching multi-area Unity (num_areas=%d, base_port=%d, time_scale=%.1f)...",
            num_areas, base_port, time_scale,
        )

        # Set Unity env vars before launching
        os.environ["NUM_AREAS"] = str(num_areas)
        os.environ["TARGET_HURTBOX_INFLATE"] = str(target_hurtbox_inflate)
        os.environ["LEARNER_SHOOT_RAY_MODE"] = self.learner_shoot_ray_mode
        os.environ["RAY_CORRECTION_ALPHA"] = str(ray_correction_alpha)
        os.environ["RAY_CORRECTION_REQUIRE_LOS"] = (
            "1" if ray_correction_require_los else "0"
        )
        os.environ["RAY_CORRECTION_MAX_ANGLE_DEG"] = str(ray_correction_max_angle_deg)
        if shot_debug_true_unity and shot_debug_unity_log_path:
            unity_log = Path(shot_debug_unity_log_path).resolve()
            unity_log.parent.mkdir(parents=True, exist_ok=True)
            if unity_log.exists():
                unity_log.unlink()
            config_path = unity_log.parent / "shot_debug_unity_config.json"
            config_data = {
                "enabled": True,
                "path": str(unity_log),
                "sample_limit": shot_debug_sample_limit_per_area,
            }
            config_path.write_text(json.dumps(config_data), encoding="utf-8")
            os.environ["SHOT_DEBUG_UNITY"] = "1"
            os.environ["SHOT_DEBUG_UNITY_PATH"] = str(unity_log)
            os.environ["SHOT_DEBUG_SAMPLE_LIMIT"] = str(shot_debug_sample_limit_per_area)
            os.environ["SHOT_DEBUG_UNITY_CONFIG"] = str(config_path.resolve())
            logger.info(
                "Shot debug Unity config: path=%s config=%s",
                unity_log,
                config_path,
            )
        else:
            os.environ.pop("SHOT_DEBUG_UNITY", None)
            os.environ.pop("SHOT_DEBUG_UNITY_PATH", None)
            os.environ.pop("SHOT_DEBUG_UNITY_CONFIG", None)
            os.environ.pop("SHOT_DEBUG_SAMPLE_LIMIT", None)

        if log_weapon_state_debug and weapon_state_debug_log_path:
            wlog = Path(weapon_state_debug_log_path).resolve()
            wlog.parent.mkdir(parents=True, exist_ok=True)
            if wlog.exists():
                wlog.unlink()
            os.environ["WEAPON_STATE_DEBUG"] = "1"
            os.environ["WEAPON_STATE_DEBUG_PATH"] = str(wlog)
            logger.info("Weapon state debug log: %s", wlog)
        else:
            os.environ.pop("WEAPON_STATE_DEBUG", None)
            os.environ.pop("WEAPON_STATE_DEBUG_PATH", None)

        self._launch_unity()

        self._area_pos_offset = np.zeros((num_areas, 45), dtype=np.float32)
        for i in range(num_areas):
            self._area_pos_offset[i, 0] = i * 500.0

        self.envs = [self.env]

    def _launch_unity(self) -> None:
        self.env = _launch_unity_env(
            env_path=self._env_path,
            worker_id=1,
            base_port=self._base_port,
            seed=self._seed,
            time_scale=self._time_scale,
            num_areas=self.num_areas,
        )
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.action_spec = self.env.behavior_specs[self.behavior_name].action_spec
        self.action_contract = validate_unity_action_spec(self.action_spec)

        decision_steps, _ = self.env.get_steps(self.behavior_name)
        discovered = sorted(decision_steps.agent_id.tolist())
        if len(discovered) < self.num_areas:
            for _ in range(50):
                self.env.step()
                decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
                all_ids = set(decision_steps.agent_id.tolist())
                all_ids.update(terminal_steps.agent_id.tolist())
                if len(all_ids) >= self.num_areas:
                    discovered = sorted(all_ids)
                    break

        if len(discovered) < self.num_areas:
            raise RuntimeError(
                f"MultiAreaUnityRunner expected {self.num_areas} agents but "
                f"found {len(discovered)}: {discovered}."
            )

        self.agent_ids = discovered[:self.num_areas]
        self.agent_to_area = {aid: i for i, aid in enumerate(self.agent_ids)}

        logger.info(
            "MultiAreaUnityRunner: %d areas, behavior=%s, contract=%s, "
            "agent_ids=%s",
            self.num_areas, self.behavior_name, self.action_contract,
            self.agent_ids,
        )

    def hard_restart(self) -> None:
        logger.info("Hard-restarting Unity process (num_areas=%d)", self.num_areas)
        try:
            self.env.close()
        except Exception:
            pass
        if self.weapon_state_debug_log_path:
            wlog = Path(self.weapon_state_debug_log_path)
            if wlog.exists():
                wlog.unlink()
        self._launch_unity()
        self.envs = [self.env]

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def _gather_all_obs(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Gather observations for all agents from one get_steps call.

        Returns (obs_array[N,45], agent_id_array[N], info_dict).
        info_dict contains per-agent reward and done status from terminal_steps.
        """
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        obs = np.zeros((self.num_areas, 45), dtype=np.float32)
        aids = np.array(self.agent_ids, dtype=np.int64)
        info = {}

        for aid in self.agent_ids:
            area = self.agent_to_area[aid]
            if aid in decision_steps:
                obs[area] = np.asarray(decision_steps[aid].obs[0], dtype=np.float32)
                info[aid] = {
                    "reward": float(decision_steps[aid].reward),
                    "done": False,
                }
            elif aid in terminal_steps:
                obs[area] = np.asarray(terminal_steps[aid].obs[0], dtype=np.float32)
                info[aid] = {
                    "reward": float(terminal_steps[aid].reward),
                    "done": True,
                }
            else:
                info[aid] = {"reward": 0.0, "done": False}

        return obs, aids, info

    @torch.no_grad()
    def collect_rollouts(
        self,
        policy,
        buffer: VectorizedRolloutBuffer,
        n_steps: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        force_done_every_n: int | None = None,
        deterministic_policy_actions: bool = False,
        action_seed: int | None = None,
    ) -> dict[str, float]:
        """Collect n_steps from all areas with batched policy inference."""
        t_start = time.monotonic()
        policy.eval()
        buffer.reset()
        E = self.num_areas

        if action_seed is not None:
            torch.manual_seed(action_seed)

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
        pressed_counts = np.zeros(E, dtype=np.int64)
        fired_counts = np.zeros(E, dtype=np.int64)
        aligned_fired_counts = np.zeros(E, dtype=np.int64)
        hit_counts = np.zeros(E, dtype=np.int64)
        kill_counts = np.zeros(E, dtype=np.int64)
        pos_reward_counts = np.zeros(E, dtype=np.int64)
        learner_damage = np.zeros(E, dtype=np.float64)
        consecutive_hits = np.zeros(E, dtype=np.int64)
        max_consecutive_hits = np.zeros(E, dtype=np.int64)
        shot_debug_counts = np.zeros(E, dtype=np.int64)
        target_hp_min = np.full(E, 100.0, dtype=np.float32)
        opponent_hp_frac_min = 1.0
        hp_idx = self.opponent_hp_frac_obs_index

        self.env.reset()

        # Re-discover agent IDs after reset (they may change)
        for _ in range(50):
            decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
            all_ids = set(decision_steps.agent_id.tolist())
            if len(all_ids) >= E:
                self.agent_ids = sorted(all_ids)[:E]
                self.agent_to_area = {aid: i for i, aid in enumerate(self.agent_ids)}
                break
            self.env.step()

        obs_np, _, _ = self._gather_all_obs()

        t_infer_total = t_step_total = t_collate_total = 0.0
        force_n = int(force_done_every_n) if force_done_every_n else 0

        for step in range(n_steps):
            raw_obs_log[step] = obs_np
            obs_local = obs_np - self._area_pos_offset
            obs_norm_np = self.normalizer.normalize(obs_local)

            # ---- Batched policy inference ----
            t0 = time.monotonic()
            obs_tensor = torch.from_numpy(obs_norm_np).unsqueeze(1).to(self.device)
            dist, values, new_hidden = policy(obs_tensor, hidden_states)
            if deterministic_policy_actions:
                action_clamped = dist.mode
                action_raw = dist.mode
            else:
                action_clamped, action_raw = dist.sample()
            log_probs = dist.log_prob(action_raw)
            action_clamped_np = action_clamped.squeeze(1).cpu().numpy()
            action_raw_sq = action_raw.squeeze(1)
            values_sq = values.squeeze(1)
            log_probs_sq = log_probs.squeeze(1)
            hidden_before = hidden_states.squeeze(0)
            t_infer_total += time.monotonic() - t0

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
            # ---- Set actions for ALL agents, then step ONCE ----
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
                    "SLOW_STEP step=%d took %.1fs — possible Unity hang. "
                    "agent_ids=%s",
                    step, step_dur, self.agent_ids,
                )

            # ---- Gather results from all agents ----
            t0 = time.monotonic()
            next_obs_np, _, step_info = self._gather_all_obs()

            force_done = force_n > 0 and ((step + 1) % force_n == 0)

            rewards_step = np.zeros(E, dtype=np.float32)
            dones_step = np.zeros(E, dtype=np.float32)

            for aid in self.agent_ids:
                e = self.agent_to_area[aid]
                info = step_info.get(aid, {"reward": 0.0, "done": False})
                env_reward = info["reward"]
                done = info["done"] or force_done

                shaped_reward, components = self.reward_builder.compute(
                    obs_current=obs_np[e],
                    obs_next=next_obs_np[e],
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
                aim_err_log[step, e] = max(0.0, float(obs_np[e, 22]))
                is_pressed = bool(action_clamped_np[e, 4] > 0.5)
                fired_idx = self.reward_builder.shot_fired_obs_index
                is_fired = (
                    fired_idx is not None
                    and fired_idx < len(next_obs_np[e])
                    and float(next_obs_np[e, fired_idx]) > 0.5
                )
                if is_pressed:
                    pressed_counts[e] += 1
                if is_fired:
                    fired_counts[e] += 1
                    shot_counts[e] += 1
                    if self.reward_builder.aligned_at_fire(obs_np[e]):
                        aligned_fired_counts[e] += 1
                elif is_pressed and not self.reward_builder.shoot_bootstrap_require_fired:
                    shot_counts[e] += 1
                if hp_idx is not None and hp_idx < len(next_obs_np[e]):
                    hp_frac = float(next_obs_np[e, hp_idx])
                    target_hp_min[e] = min(target_hp_min[e], hp_frac * 100.0)
                    opponent_hp_frac_min = min(opponent_hp_frac_min, hp_frac)
                if (
                    self.shot_debug_enabled
                    and self.shot_debug_log_path
                    and is_fired
                    and shot_debug_counts[e] < self.shot_debug_sample_limit_per_area
                ):
                    aim_at_fire = max(0.0, float(obs_np[e, 22]))
                    true_aim_at_fire = max(0.0, float(obs_np[e, 38])) if len(obs_np[e]) > 38 else 999.0
                    dist_at_fire = (
                        float(next_obs_np[e, 9]) if len(next_obs_np[e]) > 9 else 0.0
                    )
                    aligned = self.reward_builder.aligned_at_fire(obs_np[e])
                    raycast_hit = (
                        env_reward
                        > self.reward_builder.learner_hit_env_reward_threshold
                    )
                    debug_entry = {
                        "step": step,
                        "area": int(e),
                        "aim_err_at_fire": aim_at_fire,
                        "true_aim_err_at_fire": true_aim_at_fire,
                        "target_distance_at_fire": dist_at_fire,
                        "aligned_at_fire": aligned,
                        "line_of_sight_at_fire": dist_at_fire < 150.0,
                        "raycast_hit_at_fire": raycast_hit,
                        "opponent_hp_frac_at_fire": (
                            float(next_obs_np[e, hp_idx]) if hp_idx is not None else 1.0
                        ),
                        "env_reward": float(env_reward),
                    }
                    with open(self.shot_debug_log_path, "a", encoding="utf-8") as dbg:
                        dbg.write(json.dumps(debug_entry) + "\n")
                    shot_debug_counts[e] += 1
                if env_reward > self.reward_builder.learner_hit_env_reward_threshold:
                    hit_counts[e] += 1
                    learner_damage[e] += env_reward
                    consecutive_hits[e] += 1
                    max_consecutive_hits[e] = max(
                        max_consecutive_hits[e], consecutive_hits[e],
                    )
                else:
                    consecutive_hits[e] = 0
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

                    # After terminal step, the agent may re-appear in next
                    # decision_steps with a new ID. Re-discover if needed.

            t = buffer.pos - 1
            buffer.rewards[t].copy_(torch.from_numpy(rewards_step).to(self.device))
            buffer.dones[t].copy_(torch.from_numpy(dones_step).to(self.device))
            obs_np = next_obs_np
            t_collate_total += time.monotonic() - t0

            # Refresh agent IDs immediately after any done/terminal event
            any_done = dones_step.any()
            if any_done or step % 100 == 99:
                self._refresh_agent_ids()

            # Heartbeat every 500 steps
            if step % 500 == 0:
                elapsed = time.monotonic() - t_start
                logger.info(
                    "ROLLOUT_HEARTBEAT step=%d/%d elapsed=%.1fs areas=%d "
                    "dones_this_step=%d resets_total=%s",
                    step, n_steps, elapsed, E,
                    int(dones_step.sum()),
                    done_counts.tolist(),
                )

        last_obs = torch.from_numpy(
            self.normalizer.normalize(obs_np - self._area_pos_offset)
        ).unsqueeze(1).to(self.device)
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
            total_reward = component_sums.get("reward/total", 0.0) / component_count
            boot_sum = component_sums.get("reward/shoot_bootstrap", 0.0)
            chain_sum = component_sums.get("reward/damage_chain_bonus", 0.0)
            hp_sum = component_sums.get("reward/hp_delta_reward", 0.0)
            metrics["shoot/bootstrap_reward_mean"] = boot_sum / component_count
            metrics["shoot/bootstrap_reward_sum"] = boot_sum
            if total_reward > 1e-8:
                metrics["shoot/bootstrap_reward_fraction_of_total"] = boot_sum / total_reward
            metrics["damage/chain_bonus_mean"] = chain_sum / component_count
            metrics["damage/chain_bonus_sum"] = chain_sum
            metrics["damage/hp_delta_reward_mean"] = hp_sum / component_count
            metrics["damage/hp_delta_reward_sum"] = hp_sum
            if total_reward > 1e-8:
                metrics["damage/chain_bonus_fraction_of_total"] = chain_sum / total_reward
                metrics["damage/hp_delta_reward_fraction_of_total"] = hp_sum / total_reward

        action_names = ["move_x", "move_y", "look_dx", "look_dy", "shoot", "reload", "jump", "crouch"]
        for i, name in enumerate(action_names):
            col = action_log[:, :, i].flatten()
            metrics[f"action/{name}_mean"] = float(col.mean())
            metrics[f"action/{name}_std"] = float(col.std())

        aim_flat = aim_err_log.flatten()
        metrics["aim/err_mean"] = float(aim_flat.mean())
        metrics["aim/err_median"] = float(np.median(aim_flat))
        metrics["aim/err_p10"] = float(np.percentile(aim_flat, 10))
        metrics["aim/err_p90"] = float(np.percentile(aim_flat, 90))
        total_pressed = int(pressed_counts.sum())
        total_fired = int(fired_counts.sum())
        total_shots = int(shot_counts.sum())
        total_hits = int(hit_counts.sum())
        metrics["aim/total_shots"] = float(total_shots)
        metrics["aim/total_hits"] = float(total_hits)
        metrics["aim/total_kills"] = float(kill_counts.sum())
        metrics["aim/total_pos_rewards"] = float(pos_reward_counts.sum())
        metrics["aim/hit_rate"] = float(total_hits) / max(total_shots, 1)
        metrics["aim/shoot_rate"] = float(total_shots) / max(n_steps * E, 1)

        metrics["shoot/pressed_steps"] = float(total_pressed)
        metrics["shoot/pressed_rate"] = float(total_pressed) / max(n_steps * E, 1)
        metrics["shoot/fired_steps"] = float(total_fired)
        metrics["shoot/fired_rate"] = float(total_fired) / max(n_steps * E, 1)
        metrics["shoot/miss_rate"] = float(max(total_fired - total_hits, 0)) / max(total_fired, 1)
        metrics["shoot/hit_per_fired_shot"] = float(total_hits) / max(total_fired, 1)

        shoot_mask = action_log[:, :, 4] > 0.5
        aim_when_shoot = aim_err_log[shoot_mask]
        boot_thresh = self.reward_builder.shoot_bootstrap_aim_threshold_deg
        aligned_pressed = int((aim_when_shoot < boot_thresh).sum()) if len(aim_when_shoot) > 0 else 0
        aligned_fired = int(aligned_fired_counts.sum())
        metrics["shoot/aligned_fired"] = float(aligned_fired)
        metrics["shoot/aligned_fired_rate"] = float(aligned_fired) / max(total_fired, 1)
        metrics["shoot/hit_per_aligned_shot"] = float(total_hits) / max(aligned_fired, 1)
        metrics["shoot/aim_err_when_fired"] = float(aim_when_shoot.mean()) if len(aim_when_shoot) > 0 else 0.0
        metrics["config/shoot_bootstrap_aim_threshold_deg"] = float(boot_thresh)
        metrics["config/shoot_bootstrap_metric"] = self.reward_builder.shoot_bootstrap_metric
        metrics["config/true_shoot_bootstrap_threshold_deg"] = float(
            self.reward_builder.true_shoot_bootstrap_threshold_deg
        )

        metrics["config/target_hurtbox_inflate"] = float(self.target_hurtbox_inflate)
        metrics["config/learner_shoot_ray_mode"] = self.learner_shoot_ray_mode
        metrics["config/ray_correction_alpha"] = float(self.ray_correction_alpha)
        metrics["config/ray_correction_require_los"] = float(
            1.0 if self.ray_correction_require_los else 0.0
        )
        metrics["config/ray_correction_max_angle_deg"] = float(
            self.ray_correction_max_angle_deg
        )

        metrics["damage/learner_hits"] = float(total_hits)
        metrics["damage/learner_damage_dealt"] = float(learner_damage.sum())
        metrics["damage/same_target_consecutive_hits"] = float(max_consecutive_hits.max())
        metrics["damage/kills"] = float(kill_counts.sum())
        metrics["damage/target_hp_min"] = float(target_hp_min.min())
        metrics["damage/opponent_hp_frac_min"] = float(opponent_hp_frac_min)
        metrics["behavior/aim_error_deg"] = metrics["aim/err_mean"]
        metrics["combat/hits"] = float(total_hits)

        for area in range(E):
            metrics[f"per_area/area{area}_shots"] = float(shot_counts[area])
            metrics[f"per_area/area{area}_hits"] = float(hit_counts[area])
            metrics[f"per_area/area{area}_kills"] = float(kill_counts[area])
            metrics[f"per_area/area{area}_aim_err"] = float(aim_err_log[:, area].mean())

        raw_obs_flat = raw_obs_log.reshape(-1, 45)
        local_sample = raw_obs_log[0] - self._area_pos_offset
        norm_obs_sample = self.normalizer.normalize(local_sample)
        metrics["obs/raw_min"] = float(raw_obs_flat.min())
        metrics["obs/raw_max"] = float(raw_obs_flat.max())
        metrics["obs/raw_mean"] = float(raw_obs_flat.mean())
        metrics["obs/raw_std"] = float(raw_obs_flat.std())
        metrics["obs/local_pos_x_max"] = float(np.abs(local_sample[:, 0]).max())
        metrics["obs/norm_min"] = float(norm_obs_sample.min())
        metrics["obs/norm_max"] = float(norm_obs_sample.max())

        for area in range(E):
            nz = int(np.count_nonzero(env_reward_log[:, area]))
            metrics[f"per_area/area{area}_env_reward_nz"] = float(nz)
            metrics[f"per_area/area{area}_env_reward_sum"] = float(
                env_reward_log[:, area].sum()
            )
            metrics[f"per_area/area{area}_resets"] = float(done_counts[area])

        self.last_diag = {
            "raw_obs_log": raw_obs_log,
            "env_reward_log": env_reward_log,
            "done_counts": done_counts,
            "aim_err_log": aim_err_log,
            "shot_counts": shot_counts,
            "hit_counts": hit_counts,
            "kill_counts": kill_counts,
        }
        if self.log_weapon_state_debug and self.weapon_state_debug_log_path:
            metrics.update(
                aggregate_weapon_state_metrics(self.weapon_state_debug_log_path)
            )
        return metrics

    def _refresh_agent_ids(self) -> None:
        """Re-discover agent IDs if they changed after episode resets."""
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        current_ids = sorted(decision_steps.agent_id.tolist())
        if len(current_ids) >= self.num_areas:
            new_ids = current_ids[:self.num_areas]
            if new_ids != self.agent_ids:
                logger.debug(
                    "Agent IDs changed: %s -> %s", self.agent_ids, new_ids,
                )
                self.agent_ids = new_ids
                self.agent_to_area = {
                    aid: i for i, aid in enumerate(self.agent_ids)
                }
