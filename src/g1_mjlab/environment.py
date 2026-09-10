"""Construction of the versioned standing mjlab environment and PPO runner config."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import PpoProfile, ResolvedRunConfig, StandingRewardProfile


def build_standing_train_config(
    config: ResolvedRunConfig,
    log_root: Path,
    *,
    randomized_reset: bool = True,
    reward_profile: StandingRewardProfile | None = None,
    ppo_profile: PpoProfile | None = None,
) -> Any:
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import mdp as envs_mdp
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.scripts.train import TrainConfig

    from .standing_task import TASK_ID, configure_stance, register_standing_task

    register_standing_task()

    cfg = TrainConfig.from_task(config.task_id)
    cfg.env.scene.num_envs = config.num_envs
    cfg.env.episode_length_s = config.episode_length_s
    cfg.env.sim.mujoco.timestep = config.physics_dt
    cfg.env.decimation = config.decimation
    cfg.env.seed = config.seed
    configure_stance(cfg.env, randomized_reset=randomized_reset)
    if config.task_id == TASK_ID:
        cfg.env.rewards["termination"].weight = -5.0 / config.control_dt
    if reward_profile is not None:
        if reward_profile.alive_reward_rate:
            cfg.env.rewards["alive"] = RewardTermCfg(
                func=envs_mdp.is_alive,
                weight=reward_profile.alive_reward_rate,
            )
        if reward_profile.termination_penalty:
            cfg.env.rewards["termination"] = RewardTermCfg(
                func=envs_mdp.is_terminated,
                weight=reward_profile.termination_weight(config.control_dt),
            )
    cfg.agent.seed = config.seed
    cfg.agent.num_steps_per_env = config.rollout_steps
    cfg.agent.max_iterations = config.max_iterations
    cfg.agent.save_interval = config.save_interval
    cfg.agent.clip_actions = config.action_clip
    if ppo_profile is not None:
        distribution_cfg = cfg.agent.actor.distribution_cfg
        if distribution_cfg is None:
            raise ValueError("standing actor must define an action distribution")
        distribution_cfg["init_std"] = ppo_profile.initial_action_std
    cfg.agent.experiment_name = "g1_standing"
    cfg.agent.run_name = config.run_name
    cfg.agent.logger = "tensorboard"
    cfg.agent.upload_model = False
    return replace(cfg, log_root=str(log_root), gpu_ids=[0])
