"""Unity rollout collection for the Phase 3 recurrent policy."""

from __future__ import annotations

from collections import defaultdict
import logging

import numpy as np
import torch

from .reward_builder import RewardBuilder
from .rollout_buffer import RecurrentRolloutBuffer

logger = logging.getLogger(__name__)


def validate_unity_action_spec(action_spec) -> str:
    """Validate and identify the supported Unity action contract."""
    continuous_size = int(getattr(action_spec, "continuous_size", 0))
    branches = tuple(getattr(action_spec, "discrete_branches", ()) or ())

    if continuous_size >= 8 and not branches:
        return "continuous8"
    if (
        continuous_size == 4
        and len(branches) >= 4
        and all(int(size) >= 2 for size in branches[:4])
    ):
        return "continuous4_binary4"
    raise ValueError(
        "Unsupported Unity ActionSpec. Phase 3 requires either 8 continuous "
        "actions or exactly 4 continuous move/look actions plus at least four "
        f"binary discrete branches; got continuous_size={continuous_size}, "
        f"discrete_branches={branches}."
    )


def policy_action_to_buffers(
    action_8: np.ndarray,
    action_spec,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a command action to legal Unity continuous/discrete buffers."""
    action_8 = np.asarray(action_8, dtype=np.float32)
    if action_8.shape != (8,):
        raise ValueError(f"Expected action shape (8,), got {action_8.shape}")
    if not np.isfinite(action_8).all():
        raise ValueError("Policy action contains NaN or Inf")

    contract = validate_unity_action_spec(action_spec)
    continuous_size = int(action_spec.continuous_size)
    branches = tuple(action_spec.discrete_branches or ())
    continuous = np.zeros((1, continuous_size), dtype=np.float32)

    if contract == "continuous8":
        continuous[0, :4] = np.clip(action_8[:4], -1.0, 1.0)
        continuous[0, 4:8] = (action_8[4:8] > 0.5).astype(np.float32)
        discrete = np.empty((1, 0), dtype=np.int32)
    else:
        continuous[0, :4] = np.clip(action_8[:4], -1.0, 1.0)
        discrete = np.zeros((1, len(branches)), dtype=np.int32)
        discrete[0, :4] = (action_8[4:8] > 0.5).astype(np.int32)

    return continuous, discrete


def make_action_tuple(continuous: np.ndarray, discrete: np.ndarray):
    """Construct the ML-Agents action object without making imports optional."""
    try:
        from mlagents_envs.base_env import ActionTuple
    except ImportError as exc:
        raise RuntimeError(
            "mlagents_envs is required for Unity rollout collection"
        ) from exc
    return ActionTuple(continuous=continuous, discrete=discrete)


def _agent_observation(steps, agent_id: int) -> np.ndarray:
    step = steps[agent_id]
    if not step.obs:
        raise ValueError(f"Agent {agent_id} has no observations")
    obs = np.asarray(step.obs[0], dtype=np.float32)
    if obs.shape != (45,):
        raise ValueError(
            f"Phase 3 expects one 45-dim vector observation, got {obs.shape}"
        )
    if not np.isfinite(obs).all():
        raise ValueError(f"Agent {agent_id} observation contains NaN or Inf")
    if np.allclose(obs, 0.0):
        raise ValueError(
            f"Agent {agent_id} observation is all-zero; Unity likely padded "
            "missing CollectObservations output"
        )
    return obs


class SelfPlayRunner:
    """Collect single-learner recurrent rollouts against Unity-controlled bots."""

    def __init__(
        self,
        env,
        reward_builder: RewardBuilder,
        device: str = "cpu",
        normalizer=None,
    ):
        if normalizer is None:
            raise ValueError(
                "SelfPlayRunner requires an ObservationNormalizer; online "
                "observation normalization must not be disabled during PPO."
            )
        self.env = env
        self.reward_builder = reward_builder
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.action_spec = self.env.behavior_specs[self.behavior_name].action_spec
        self.action_contract = validate_unity_action_spec(self.action_spec)
        logger.info(
            "Unity BehaviorSpec %s: contract=%s continuous_size=%s "
            "discrete_branches=%s",
            self.behavior_name,
            self.action_contract,
            self.action_spec.continuous_size,
            self.action_spec.discrete_branches,
        )

    @torch.no_grad()
    def collect_rollouts(
        self,
        policy,
        buffer: RecurrentRolloutBuffer,
        n_steps: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> dict[str, float]:
        """Collect a rollout and compute GAE."""
        policy.eval()
        buffer.reset()
        self.env.reset()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        hidden_states: dict[int, torch.Tensor] = {}
        previous_actions: dict[int, np.ndarray] = {}
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        shaped_rewards: list[float] = []
        component_sums: defaultdict[str, float] = defaultdict(float)
        component_count = 0
        current_episode_reward = 0.0
        current_episode_length = 0
        steps_collected = 0

        while steps_collected < n_steps:
            for agent_id in terminal_steps:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0.0
                current_episode_length = 0
                hidden_states.pop(agent_id, None)
                previous_actions.pop(agent_id, None)

            if len(decision_steps) == 0:
                self.env.step()
                decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
                continue

            # The current Phase 3 environment has one Python-controlled learner.
            agent_id = int(decision_steps.agent_id[0])
            obs_current = _agent_observation(decision_steps, agent_id)  # raw (logging/reward)
            obs_norm = self.normalizer.normalize(obs_current)           # policy/buffer input
            obs_tensor = (
                torch.from_numpy(obs_norm).unsqueeze(0).unsqueeze(0).to(self.device)
            )
            current_hidden = hidden_states.get(
                agent_id, policy.init_hidden(1, self.device)
            )

            dist, value, next_hidden = policy(obs_tensor, current_hidden)
            action_clamped, action_raw = dist.sample()
            log_prob = dist.log_prob(action_raw)
            action_env = action_clamped.squeeze(0).squeeze(0).cpu().numpy()

            continuous, discrete = policy_action_to_buffers(
                action_env, self.action_spec
            )
            self.env.set_action_for_agent(
                self.behavior_name,
                agent_id,
                make_action_tuple(continuous, discrete),
            )
            self.env.step()
            next_decision_steps, next_terminal_steps = self.env.get_steps(
                self.behavior_name
            )

            if agent_id in next_terminal_steps:
                env_reward = float(next_terminal_steps[agent_id].reward)
                obs_next = _agent_observation(next_terminal_steps, agent_id)
                done = True
            elif agent_id in next_decision_steps:
                env_reward = float(next_decision_steps[agent_id].reward)
                obs_next = _agent_observation(next_decision_steps, agent_id)
                done = False
            else:
                raise RuntimeError(
                    f"Agent {agent_id} disappeared after an environment step"
                )

            shaped_reward, components = self.reward_builder.compute(
                obs_current=obs_current,
                obs_next=obs_next,
                action=action_env,
                previous_action=previous_actions.get(agent_id),
                env_reward=env_reward,
                is_terminal=done,
            )
            for name, component in components.items():
                component_sums[name] += float(component)
            component_count += 1
            shaped_rewards.append(shaped_reward)

            buffer.add(
                obs=obs_tensor,
                action=action_raw,
                reward=shaped_reward,
                done=done,
                value=value,
                log_prob=log_prob,
                hidden=current_hidden,
            )

            hidden_states[agent_id] = next_hidden
            previous_actions[agent_id] = action_env.copy()
            current_episode_reward += env_reward
            current_episode_length += 1
            steps_collected += 1
            decision_steps = next_decision_steps
            terminal_steps = next_terminal_steps

        last_value = 0.0
        if len(decision_steps) > 0:
            agent_id = int(decision_steps.agent_id[0])
            obs = _agent_observation(decision_steps, agent_id)
            obs_norm = self.normalizer.normalize(obs)
            obs_tensor = torch.from_numpy(obs_norm).unsqueeze(0).unsqueeze(0).to(self.device)
            hidden = hidden_states.get(agent_id, policy.init_hidden(1, self.device))
            last_value = policy.get_value(obs_tensor, hidden).item()

        buffer.compute_returns_and_advantages(
            last_value=torch.as_tensor(last_value, device=buffer.device),
            gamma=gamma,
            lam=gae_lambda,
        )

        reward_array = np.asarray(shaped_rewards, dtype=np.float32)
        metrics = {
            "rollout/avg_env_reward": float(
                np.mean(episode_rewards) if episode_rewards else 0.0
            ),
            "rollout/avg_length": float(
                np.mean(episode_lengths) if episode_lengths else 0.0
            ),
            "rollout/episodes_completed": float(len(episode_rewards)),
            "rollout/reward_mean": float(reward_array.mean()),
            "rollout/reward_std": float(reward_array.std()),
            "rollout/reward_min": float(reward_array.min()),
            "rollout/reward_max": float(reward_array.max()),
        }
        if component_count:
            metrics.update(
                {
                    f"rollout/{name}": total / component_count
                    for name, total in component_sums.items()
                }
            )
        return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli_main() -> None:
    """CLI for Unity behavior spec inspection and smoke rollouts."""
    import argparse
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Inspect Unity ML-Agents behavior spec and run smoke rollouts."
    )
    parser.add_argument(
        "--unity_build",
        required=True,
        help="Path to the Unity build (or empty for Editor).",
    )
    parser.add_argument(
        "--print_behavior_spec",
        action="store_true",
        help="Print all behavior specs from the environment.",
    )
    parser.add_argument(
        "--validate_action_spec",
        action="store_true",
        help="Validate action spec matches Phase 3 requirements.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to recurrent distilled checkpoint for rollout.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of steps for the rollout smoke test.",
    )
    parser.add_argument(
        "--log_reward_components",
        action="store_true",
        help="Log shaped reward components during rollout.",
    )
    parser.add_argument(
        "--telemetry_stats",
        default="artifacts/demonstrations/packed/telemetry_stats.npz",
        help="Telemetry stats NPZ for online observation normalization.",
    )
    args = parser.parse_args()

    try:
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.engine_configuration_channel import (
            EngineConfigurationChannel,
        )
    except ImportError:
        logger.error(
            "mlagents_envs is required. Install with: pip install mlagents_envs"
        )
        sys.exit(1)

    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(time_scale=1.0)

    # Check if unity_build is a file that exists or empty
    env_path = (
        args.unity_build if args.unity_build and args.unity_build != '""' else None
    )

    env = UnityEnvironment(
        file_name=env_path,
        seed=42,
        side_channels=[engine_channel],
        no_graphics=True,
    )
    env.reset()

    if args.print_behavior_spec or args.validate_action_spec:
        logger.info("=" * 60)
        logger.info("UNITY BEHAVIOR SPECS")
        logger.info("=" * 60)

        for behavior_name, spec in env.behavior_specs.items():
            if args.print_behavior_spec:
                logger.info("")
                logger.info("Behavior: %s", behavior_name)
                logger.info("  Observation specs:")
                for i, obs_spec in enumerate(spec.observation_specs):
                    logger.info(
                        "    [%d] shape=%s name=%s", i, obs_spec.shape, obs_spec.name
                    )
                logger.info("  Action spec:")
                logger.info("    continuous_size=%d", spec.action_spec.continuous_size)
                logger.info(
                    "    discrete_branches=%s",
                    tuple(spec.action_spec.discrete_branches)
                    if spec.action_spec.discrete_branches
                    else "()",
                )

            if args.validate_action_spec:
                try:
                    contract = validate_unity_action_spec(spec.action_spec)
                    logger.info("  ✓ Phase 3 contract: %s", contract)
                except ValueError as exc:
                    logger.error("  ✗ Phase 3 validation FAILED: %s", exc)

        logger.info("")
        logger.info("=" * 60)
        logger.info("SPEC CHECK COMPLETE")
        logger.info("=" * 60)

    if args.checkpoint:
        logger.info("=" * 60)
        logger.info("RUNNING ROLLOUT SMOKE TEST")
        logger.info("=" * 60)

        from .core_action_projector import CoreActionProjector
        from .telemetry_encoder import TelemetryEncoder
        from .actor_critic import RecurrentActorCritic
        from .reward_builder import RewardBuilder

        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            logger.error("Checkpoint not found: %s", ckpt_path)
            env.close()
            sys.exit(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Build dummy encoder/projector just to create ActorCritic, then load state
        tel_encoder = TelemetryEncoder(proj_dim=512)
        projector = CoreActionProjector(tel_encoder)
        actor_critic = RecurrentActorCritic.from_distilled(projector)

        actor_critic.load_state_dict(state["actor_critic_state"])
        actor_critic.to(device)
        logger.info("Loaded checkpoint: %s", ckpt_path)

        reward_builder = RewardBuilder(
            dmg_dealt_idx=None,
            dmg_taken_idx=None,
            damage_dealt_coef=1.0,
            damage_taken_coef=1.0,
            kill_coef=0.5,
            death_coef=0.5,
            miss_coef=0.05,
            jerk_coef=0.01,
            spam_coef=0.01,
        )

        from .obs_normalizer import ObservationNormalizer
        normalizer = ObservationNormalizer(args.telemetry_stats)
        runner = SelfPlayRunner(
            env=env, reward_builder=reward_builder, device=device.type,
            normalizer=normalizer,
        )
        buffer = RecurrentRolloutBuffer(size=args.steps, device=device.type)

        metrics = runner.collect_rollouts(
            policy=actor_critic,
            buffer=buffer,
            n_steps=args.steps,
            gamma=0.99,
            gae_lambda=0.95,
        )

        logger.info("Rollout Metrics:")
        for k, v in metrics.items():
            if args.log_reward_components or not k.startswith(
                "rollout/reward_component_"
            ):
                logger.info("  %s: %.4f", k, v)

        logger.info("=" * 60)
        logger.info("ROLLOUT SMOKE TEST COMPLETE")
        logger.info("=" * 60)

    env.close()


if __name__ == "__main__":
    _cli_main()
