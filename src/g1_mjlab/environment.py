"""Construction of versioned mjlab environments and PPO runner configs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import (
    PpoProfile,
    ResolvedRunConfig,
    StandingRewardProfile,
    WalkingTrainingProfile,
)
from .tasks import STANDING_TASK_ID, TaskCapability, get_task


def build_train_config(
    config: ResolvedRunConfig,
    log_root: Path,
    *,
    randomized_reset: bool = True,
    reward_profile: StandingRewardProfile | None = None,
    ppo_profile: PpoProfile | None = None,
    walking_profile: WalkingTrainingProfile | None = None,
) -> Any:
    """Build a supported task after validating capability without simulator imports."""
    task = get_task(config.task_id).require(TaskCapability.TRAIN)

    import mjlab.tasks  # noqa: F401
    from mjlab.envs import mdp as envs_mdp
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.scripts.train import TrainConfig

    task.register()

    cfg = TrainConfig.from_task(config.task_id)
    cfg.env.scene.num_envs = config.num_envs
    cfg.env.episode_length_s = config.episode_length_s
    cfg.env.sim.mujoco.timestep = config.physics_dt
    cfg.env.decimation = config.decimation
    cfg.env.seed = config.seed
    task.configure_environment(
        cfg.env, randomized_reset=randomized_reset, task_profile=walking_profile
    )
    if task.termination_penalty is not None:
        cfg.env.rewards["termination"].weight = task.termination_penalty / config.control_dt
    if reward_profile is not None:
        if config.task_id != STANDING_TASK_ID:
            raise ValueError("StandingRewardProfile can only configure standing-v1")
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
            raise ValueError(f"{task.task_id} actor must define an action distribution")
        distribution_cfg["init_std"] = ppo_profile.initial_action_std
        if ppo_profile.entropy_coef is not None:
            cfg.agent.algorithm.entropy_coef = ppo_profile.entropy_coef
    cfg.agent.experiment_name = task.experiment_name
    cfg.agent.run_name = config.run_name
    cfg.agent.logger = "tensorboard"
    cfg.agent.upload_model = False
    return replace(cfg, log_root=str(log_root), gpu_ids=[0])


def build_standing_train_config(
    config: ResolvedRunConfig,
    log_root: Path,
    *,
    randomized_reset: bool = True,
    reward_profile: StandingRewardProfile | None = None,
    ppo_profile: PpoProfile | None = None,
) -> Any:
    if config.task_id != STANDING_TASK_ID:
        raise ValueError(
            f"build_standing_train_config requires {STANDING_TASK_ID!r}, got {config.task_id!r}"
        )
    return build_train_config(
        config,
        log_root,
        randomized_reset=randomized_reset,
        reward_profile=reward_profile,
        ppo_profile=ppo_profile,
    )
