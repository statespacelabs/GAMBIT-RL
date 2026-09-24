"""Combat-focused online reward construction."""

from __future__ import annotations

import math
import numpy as np


AIM_ERR_MAG_IDX = 22
TRUE_AIM_ERR_IDX = 38
DIST_TO_OPP_IDX = 9
OPPONENT_HP_FRAC_IDX = 39


class RewardBuilder:
    """Compute reward from post-action combat deltas and action changes."""

    def __init__(
        self,
        dmg_dealt_idx: int | None = None,
        dmg_taken_idx: int | None = None,
        env_reward_coef: float = 1.0,
        damage_dealt_coef: float = 1.0,
        damage_taken_coef: float = 1.0,
        kill_coef: float = 0.5,
        death_coef: float = 0.5,
        miss_coef: float = 0.05,
        jerk_coef: float = 0.01,
        spam_coef: float = 0.01,
        aim_coef: float = 0.0,
        aim_err_max: float = 45.0,
        aim_reward_mode: str = "proximity_next",
        bad_aim_threshold_deg: float = 90.0,
        bad_aim_coef: float = 0.0,
        bad_aim_scale_deg: float = 90.0,
        aim_sigma_deg: float = 60.0,
        use_true_aim_for_reward: bool = False,
        true_aim_err_max: float = 30.0,
        true_aim_obs_index: int = TRUE_AIM_ERR_IDX,
        shoot_bootstrap_metric: str = "old_aim",
        true_shoot_bootstrap_threshold_deg: float = 5.0,
        shoot_bootstrap_coef: float = 0.0,
        shoot_bootstrap_aim_threshold_deg: float = 30.0,
        shoot_bootstrap_require_fired: bool = False,
        shoot_bootstrap_require_los: bool = False,
        shot_fired_obs_index: int | None = 44,
        damage_chain_coef: float = 0.0,
        damage_chain_window_sec: float = 3.0,
        damage_chain_window_steps: int = 45,
        learner_hit_env_reward_threshold: float = 0.05,
        hp_delta_reward_coef: float = 0.0,
        hp_delta_reward_cap: float = 0.05,
        opponent_hp_frac_obs_index: int | None = 39,
        hp_frac_to_damage_scale: float = 100.0,
        env_hit_reward_scale: float = 0.2,
        env_hit_damage_points: float = 20.0,
    ):
        self.dmg_dealt_idx = dmg_dealt_idx
        self.dmg_taken_idx = dmg_taken_idx
        self.env_reward_coef = env_reward_coef
        self.damage_dealt_coef = damage_dealt_coef
        self.damage_taken_coef = damage_taken_coef
        self.kill_coef = kill_coef
        self.death_coef = death_coef
        self.miss_coef = miss_coef
        self.jerk_coef = jerk_coef
        self.spam_coef = spam_coef
        self.aim_coef = aim_coef
        self.aim_err_max = aim_err_max
        self.aim_reward_mode = aim_reward_mode
        self.bad_aim_threshold_deg = bad_aim_threshold_deg
        self.bad_aim_coef = bad_aim_coef
        self.bad_aim_scale_deg = bad_aim_scale_deg
        self.aim_sigma_deg = aim_sigma_deg
        self.use_true_aim_for_reward = use_true_aim_for_reward
        self.true_aim_err_max = true_aim_err_max
        self.true_aim_obs_index = true_aim_obs_index
        self.shoot_bootstrap_metric = shoot_bootstrap_metric or "old_aim"
        self.true_shoot_bootstrap_threshold_deg = true_shoot_bootstrap_threshold_deg
        self.shoot_bootstrap_coef = shoot_bootstrap_coef
        self.shoot_bootstrap_aim_threshold_deg = shoot_bootstrap_aim_threshold_deg
        self.shoot_bootstrap_require_fired = shoot_bootstrap_require_fired
        self.shoot_bootstrap_require_los = shoot_bootstrap_require_los
        self.shot_fired_obs_index = shot_fired_obs_index
        self.damage_chain_coef = damage_chain_coef
        self.damage_chain_window_sec = damage_chain_window_sec
        self.damage_chain_window_steps = max(
            1,
            damage_chain_window_steps
            if damage_chain_window_steps > 0
            else int(damage_chain_window_sec * 15),
        )
        self.learner_hit_env_reward_threshold = learner_hit_env_reward_threshold
        self.hp_delta_reward_coef = hp_delta_reward_coef
        self.hp_delta_reward_cap = hp_delta_reward_cap
        self.opponent_hp_frac_obs_index = opponent_hp_frac_obs_index
        self.hp_frac_to_damage_scale = hp_frac_to_damage_scale
        self.env_hit_reward_scale = env_hit_reward_scale
        self.env_hit_damage_points = env_hit_damage_points
        self._steps_since_last_hit = self.damage_chain_window_steps + 1

    def _compute_aim_reward(self, err_deg: float) -> float:
        """Compute aim reward given error in degrees."""
        mode = self.aim_reward_mode

        if mode == "proximity_next":
            return self.aim_coef * max(0.0, 1.0 - err_deg / self.aim_err_max)

        if mode == "proximity_bad_aim":
            good = max(0.0, 1.0 - err_deg / self.aim_err_max)
            bad = max(0.0, (err_deg - self.bad_aim_threshold_deg) / self.bad_aim_scale_deg)
            return self.aim_coef * good - self.bad_aim_coef * bad

        if mode == "gaussian":
            return self.aim_coef * math.exp(-((err_deg / self.aim_sigma_deg) ** 2))

        if mode == "true_ray_proximity":
            return self.aim_coef * max(0.0, 1.0 - err_deg / self.true_aim_err_max)

        raise ValueError(f"Unknown aim_reward_mode: {mode!r}")

    def _read_true_aim_error(self, obs: np.ndarray) -> float:
        idx = self.true_aim_obs_index
        if idx < 0 or idx >= len(obs):
            return 999.0
        return max(0.0, float(obs[idx]))

    def _read_old_aim_error(self, obs: np.ndarray) -> float:
        if len(obs) <= AIM_ERR_MAG_IDX:
            return 999.0
        return max(0.0, float(obs[AIM_ERR_MAG_IDX]))

    def _aim_error_for_reward(self, obs_next: np.ndarray) -> float:
        if self.aim_reward_mode == "true_ray_proximity" or self.use_true_aim_for_reward:
            return self._read_true_aim_error(obs_next)
        return self._read_old_aim_error(obs_next)

    def _bootstrap_good_shot(self, obs_next: np.ndarray, old_err: float) -> bool:
        if self.shoot_bootstrap_metric == "true_ray":
            return (
                self._read_true_aim_error(obs_next)
                < self.true_shoot_bootstrap_threshold_deg
            )
        return old_err < self.shoot_bootstrap_aim_threshold_deg

    def aligned_at_fire(self, obs: np.ndarray) -> bool:
        """Whether a fired shot counts as aligned for bootstrap metrics."""
        old_err = self._read_old_aim_error(obs)
        return self._bootstrap_good_shot(obs, old_err)

    @property
    def has_damage_signals(self) -> bool:
        return self.dmg_dealt_idx is not None and self.dmg_taken_idx is not None

    @staticmethod
    def _positive_delta(
        obs_current: np.ndarray,
        obs_next: np.ndarray,
        index: int | None,
    ) -> float:
        if index is None:
            return 0.0
        if index < 0 or index >= len(obs_current) or index >= len(obs_next):
            raise IndexError(f"Reward observation index out of bounds: {index}")
        return max(0.0, float(obs_next[index] - obs_current[index]))

    def _shot_fired_this_step(
        self,
        obs_next: np.ndarray,
        shoot_pressed: bool,
    ) -> bool:
        if self.shoot_bootstrap_require_fired:
            if self.shot_fired_obs_index is None:
                return shoot_pressed
            if self.shot_fired_obs_index < 0 or self.shot_fired_obs_index >= len(obs_next):
                return shoot_pressed
            return float(obs_next[self.shot_fired_obs_index]) > 0.5
        return shoot_pressed

    def compute(
        self,
        obs_current: np.ndarray,
        obs_next: np.ndarray,
        action: np.ndarray,
        previous_action: np.ndarray | None,
        env_reward: float,
        is_terminal: bool,
    ) -> tuple[float, dict[str, float]]:
        """Compute the reward for the transition ``obs_current -> obs_next``."""
        damage_dealt = self._positive_delta(obs_current, obs_next, self.dmg_dealt_idx)
        damage_taken = self._positive_delta(obs_current, obs_next, self.dmg_taken_idx)
        virtual_kill = float(is_terminal and env_reward > 0.0)
        virtual_death = float(is_terminal and env_reward < 0.0)

        shoot_pressed = bool(action[4] > 0.5)
        shot_fired = self._shot_fired_this_step(obs_next, shoot_pressed)
        missed_shot = float(
            self.has_damage_signals
            and shot_fired
            and damage_dealt <= 0.0
            and virtual_kill == 0.0
        )

        if previous_action is None:
            aim_jerk = 0.0
            action_spam = 0.0
        else:
            look_delta = action[2:4] - previous_action[2:4]
            aim_jerk = float(np.square(look_delta).sum())
            repeated_binary = (action[4:8] > 0.5) & (previous_action[4:8] > 0.5)
            action_spam = float(repeated_binary.sum())

        # Aim reward: multiple modes to shape the error→reward curve
        err_nxt = self._read_old_aim_error(obs_next)
        true_err_nxt = self._read_true_aim_error(obs_next)
        if self.aim_coef > 0:
            reward_err = self._aim_error_for_reward(obs_next)
            aim_reward = self._compute_aim_reward(reward_err)
        else:
            aim_reward = 0.0

        shoot_bootstrap = 0.0
        if self.shoot_bootstrap_coef > 0.0 and shot_fired:
            good_shot = self._bootstrap_good_shot(obs_next, err_nxt)
            if self.shoot_bootstrap_require_los:
                # Line-of-sight not in telemetry yet; keep hook for future use.
                good_shot = good_shot and False
            if good_shot:
                shoot_bootstrap = self.shoot_bootstrap_coef

        learner_hit = (
            damage_dealt > 0.0
            or virtual_kill > 0.0
            or env_reward > self.learner_hit_env_reward_threshold
        )
        damage_chain_bonus = 0.0
        if self.damage_chain_coef > 0.0:
            if learner_hit:
                if self._steps_since_last_hit <= self.damage_chain_window_steps:
                    damage_chain_bonus = self.damage_chain_coef
                self._steps_since_last_hit = 0
            else:
                self._steps_since_last_hit += 1

        if is_terminal:
            self._steps_since_last_hit = self.damage_chain_window_steps + 1

        hp_delta_reward = 0.0
        if self.hp_delta_reward_coef > 0.0:
            hp_decrease = 0.0
            idx = self.opponent_hp_frac_obs_index
            if (
                idx is not None
                and idx < len(obs_current)
                and idx < len(obs_next)
            ):
                hp_decrease = max(
                    0.0,
                    float(obs_current[idx]) - float(obs_next[idx]),
                ) * self.hp_frac_to_damage_scale
            elif env_reward > 0.0 and self.env_hit_reward_scale > 0.0:
                hp_decrease = (
                    float(env_reward)
                    / self.env_hit_reward_scale
                    * self.env_hit_damage_points
                )
            if hp_decrease > 0.0:
                hp_delta_reward = min(
                    self.hp_delta_reward_coef * hp_decrease,
                    self.hp_delta_reward_cap,
                )

        components = {
            "reward/env": self.env_reward_coef * float(env_reward),
            "reward/damage_dealt": self.damage_dealt_coef * damage_dealt,
            "reward/damage_taken": -self.damage_taken_coef * damage_taken,
            "reward/kill": self.kill_coef * virtual_kill,
            "reward/death": -self.death_coef * virtual_death,
            "reward/miss_penalty": -self.miss_coef * missed_shot,
            "reward/jerk_penalty": -self.jerk_coef * aim_jerk,
            "reward/spam_penalty": -self.spam_coef * action_spam,
            "reward/aim": aim_reward,
            "reward/shoot_bootstrap": shoot_bootstrap,
            "metric/old_aim_err": err_nxt,
            "metric/true_aim_err": true_err_nxt,
            "reward/damage_chain_bonus": damage_chain_bonus,
            "reward/hp_delta_reward": hp_delta_reward,
        }
        total = float(sum(components.values()))
        components["reward/total"] = total
        return total, components
