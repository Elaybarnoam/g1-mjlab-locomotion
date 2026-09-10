"""Versioned standing MDP, registered separately from upstream locomotion."""

from __future__ import annotations

from typing import Any

TASK_ID = "G1-Standing-Flat-v1"


def configure_stance(env: Any, *, randomized_reset: bool = True) -> None:
    """Own standing commands, reset distribution and baseline randomization."""
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

    env.curriculum = {}
    env.observations["actor"].enable_corruption = False
    for event in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
        env.events.pop(event, None)
    command = env.commands["twist"]
    if not isinstance(command, UniformVelocityCommandCfg):
        raise TypeError("standing task requires UniformVelocityCommandCfg")
    command.rel_standing_envs = 1.0
    command.rel_heading_envs = 0.0
    command.rel_forward_envs = 0.0
    command.heading_command = False
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.ranges.heading = None
    reset = env.events["reset_base"]
    reset.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.02) if randomized_reset else (0.0, 0.0),
        "yaw": (-0.05, 0.05) if randomized_reset else (0.0, 0.0),
    }
    joint_reset = env.events["reset_robot_joints"]
    joint_reset.params["position_range"] = (-0.02, 0.02) if randomized_reset else (0.0, 0.0)
    joint_reset.params["velocity_range"] = (-0.05, 0.05) if randomized_reset else (0.0, 0.0)


def healthy_standing(env: Any, min_height: float = 0.60, max_tilt: float = 0.5235987756) -> Any:
    import torch

    robot = env.scene["robot"]
    quat = robot.data.body_link_quat_w[:, robot.body_names.index("torso_link")]
    upright = 1 - 2 * (quat[:, 1].square() + quat[:, 2].square())
    support = (env.scene["feet_ground_contact"].data.found > 0).flatten(1).any(dim=1)
    return (
        (robot.data.root_link_pos_w[:, 2] >= min_height)
        & (upright >= torch.cos(torch.tensor(max_tilt, device=env.device)))
        & support
        & ~env.termination_manager.terminated
    ).float()


def base_motion_cost(env: Any) -> Any:
    return env.scene["robot"].data.root_link_lin_vel_b[:, :2].square().sum(dim=1)


def mean_squared_effort_cost(env: Any) -> Any:
    """Mean squared generalized joint force in N²·m², not a normalized fraction."""
    robot = env.scene["robot"]
    return robot.data.qfrc_actuator.square().mean(dim=1)


def register_standing_task() -> None:
    from mjlab.envs import mdp
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.tasks.registry import list_tasks, register_mjlab_task
    from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
    from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

    if TASK_ID in list_tasks():
        return
    env = unitree_g1_flat_env_cfg()
    configure_stance(env)
    weights = {
        "upright": 1.0,
        "pose": 0.5,
        "body_ang_vel": -0.05,
        "dof_pos_limits": -0.5,
        "action_rate_l2": -0.01,
    }
    env.rewards = {name: env.rewards[name] for name in weights}
    for name, weight in weights.items():
        env.rewards[name].weight = weight
    env.rewards["pose"].params["std_standing"] = {".*": 0.3}
    env.rewards["healthy_standing"] = RewardTermCfg(func=healthy_standing, weight=2.0)
    env.rewards["base_motion"] = RewardTermCfg(func=base_motion_cost, weight=-0.5)
    env.rewards["effort"] = RewardTermCfg(func=mean_squared_effort_cost, weight=-0.001)
    # The adapter resolves this event's dt-scaled weight for the selected control_dt.
    env.rewards["termination"] = RewardTermCfg(func=mdp.is_terminated, weight=-250.0)
    register_mjlab_task(TASK_ID, env, env, unitree_g1_ppo_runner_cfg())
