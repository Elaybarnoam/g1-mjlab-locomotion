"""mjlab command, observation, reward, reset, and termination terms for walking-v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from ..motion.gait import CommandProfile, step_gait_torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


@dataclass(kw_only=True)
class WalkingCommandCfg(CommandTermCfg):  # type: ignore[misc]
    motion_file: str
    reference_speed_m_s: float
    cycle_duration_s: float
    acceleration_m_s2: float
    deceleration_m_s2: float
    blend_rate_s: float
    stand_threshold_m_s: float
    walk_threshold_m_s: float
    standing_fraction: float = 0.2
    randomize_phase: bool = True
    reference_initialization: bool = True

    def build(self, env: ManagerBasedRlEnv) -> WalkingCommand:
        return WalkingCommand(self, env)


class WalkingCommand(CommandTerm):  # type: ignore[misc]
    """GPU-resident command and cyclic reference state shared by all MDP terms."""

    cfg: WalkingCommandCfg

    def __init__(self, cfg: WalkingCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.profile = CommandProfile(
            cfg.reference_speed_m_s,
            cfg.cycle_duration_s,
            cfg.acceleration_m_s2,
            cfg.deceleration_m_s2,
            cfg.blend_rate_s,
            cfg.stand_threshold_m_s,
            cfg.walk_threshold_m_s,
        )
        self.profile.validate()
        if not 0 <= cfg.standing_fraction <= 1:
            raise ValueError("standing_fraction must be in [0, 1]")
        with np.load(cfg.motion_file, allow_pickle=False) as motion:
            fps = float(motion["fps"][0])
            if not np.isclose(fps, 1.0 / env.step_dt):
                raise ValueError(f"motion fps {fps} does not match policy rate {1.0 / env.step_dt}")
            self.reference_joint_position = torch.as_tensor(
                motion["joint_pos"], dtype=torch.float32, device=self.device
            )
            self.reference_joint_velocity = torch.as_tensor(
                motion["joint_vel"], dtype=torch.float32, device=self.device
            )
            self.reference_body_position = torch.as_tensor(
                motion["body_pos_w"], dtype=torch.float32, device=self.device
            )
            self.reference_body_quaternion = torch.as_tensor(
                motion["body_quat_w"], dtype=torch.float32, device=self.device
            )
            self.reference_body_linear_velocity = torch.as_tensor(
                motion["body_lin_vel_w"], dtype=torch.float32, device=self.device
            )
            self.reference_body_angular_velocity = torch.as_tensor(
                motion["body_ang_vel_w"], dtype=torch.float32, device=self.device
            )
            self.reference_contact = torch.as_tensor(
                motion["foot_contact"], dtype=torch.bool, device=self.device
            )
            self.reference_body_names = tuple(motion["body_names"].tolist())
            reference_joint_names = tuple(motion["joint_names"].tolist())
        if self.reference_joint_position.shape[1] != 29:
            raise ValueError("walking reference must contain exactly 29 joints")
        robot: Entity = env.scene["robot"]
        if reference_joint_names != tuple(robot.joint_names):
            raise ValueError("walking reference and simulator joint order differ")
        self._requested = torch.zeros((self.num_envs, 3), device=self.device)
        self._applied = torch.zeros_like(self._requested)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.blend = torch.zeros(self.num_envs, device=self.device)
        self.walking = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reference_distance_m = torch.zeros(self.num_envs, device=self.device)
        self._command_dt: float | torch.Tensor = 0.0
        self.metrics["command_filter_error_m_s"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._applied

    @property
    def requested_command(self) -> torch.Tensor:
        return self._requested

    def set_requested_forward_speed(self, speed_m_s: float) -> None:
        """Set one validated evaluation command for every resident environment."""
        validated = self.profile.validate_requested(np.asarray([[speed_m_s, 0.0, 0.0]]))
        self._requested.zero_()
        self._requested[:, 0] = float(validated[0, 0])

    @property
    def phase_features(self) -> torch.Tensor:
        angle = 2 * torch.pi * self.phase
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=1)

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        assert isinstance(env_ids, torch.Tensor)
        self._applied[env_ids] = 0
        self.blend[env_ids] = 0
        self.walking[env_ids] = False
        self.reference_distance_m[env_ids] = 0
        if self.cfg.randomize_phase:
            self.phase[env_ids].uniform_(0.0, 1.0)
        else:
            self.phase[env_ids] = 0
        extras = cast(dict[str, float], super().reset(env_ids))
        if self.cfg.reference_initialization:
            _write_walking_state(self._env, env_ids, self, reference_initialization=True)
        return extras

    def _update_metrics(self) -> None:
        self.metrics["command_filter_error_m_s"] = torch.abs(
            self._requested[:, 0] - self._applied[:, 0]
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        standing = torch.rand(len(env_ids), device=self.device) < self.cfg.standing_fraction
        self._requested[env_ids] = 0
        self._requested[env_ids, 0] = torch.where(
            standing,
            0.0,
            self.cfg.reference_speed_m_s,
        )

    def compute(
        self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        """Advance using the manager-provided dt, including zero-dt reset worlds."""
        self._command_dt = dt
        super().compute(dt, env_ids)

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        ids: Any = slice(None) if env_ids is None else env_ids
        dt = self._command_dt
        if isinstance(dt, torch.Tensor) and env_ids is not None and dt.ndim == 1:
            dt = dt[env_ids]
        values = step_gait_torch(
            self._applied[ids],
            self.phase[ids],
            self.blend[ids],
            self.walking[ids],
            self.reference_distance_m[ids],
            self._requested[ids],
            self.profile,
            dt=dt,
        )
        (
            self._applied[ids],
            self.phase[ids],
            self.blend[ids],
            self.walking[ids],
            self.reference_distance_m[ids],
        ) = values

    def _frame_blend(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intervals = self.reference_joint_position.shape[0] - 1
        coordinate = self.phase * intervals
        lower = torch.floor(coordinate).long()
        upper = torch.minimum(lower + 1, torch.full_like(lower, intervals))
        return lower, upper, coordinate - lower

    def interpolate(self, values: torch.Tensor) -> torch.Tensor:
        lower, upper, blend = self._frame_blend()
        shape = (self.num_envs,) + (1,) * (values.ndim - 1)
        alpha = blend.reshape(shape)
        return values[lower] * (1 - alpha) + values[upper] * alpha

    @property
    def joint_position(self) -> torch.Tensor:
        return self.interpolate(self.reference_joint_position)

    @property
    def joint_velocity(self) -> torch.Tensor:
        return self.interpolate(self.reference_joint_velocity)

    @property
    def foot_contact(self) -> torch.Tensor:
        lower, _, _ = self._frame_blend()
        return self.reference_contact[lower]


@dataclass(kw_only=True)
class ReferenceResidualJointPositionActionCfg(JointPositionActionCfg):  # type: ignore[misc]
    """Joint targets centered on the current stand/walk reference pose."""

    command_name: str

    def build(self, env: ManagerBasedRlEnv) -> ReferenceResidualJointPositionAction:
        return ReferenceResidualJointPositionAction(self, env)


class ReferenceResidualJointPositionAction(JointPositionAction):  # type: ignore[misc]
    """Apply normalized policy residuals around a phase-aligned reference."""

    cfg: ReferenceResidualJointPositionActionCfg

    def __init__(self, cfg: ReferenceResidualJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command = _command(env, cfg.command_name)

    @property
    def joint_position_target(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        default = self._entity.data.default_joint_pos[:, self._target_ids]
        reference = self._command.joint_position[:, self._target_ids]
        blend = self._command.blend[:, None]
        center = default * (1 - blend) + reference * blend
        self._processed_actions = center + self._raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


def _command(env: ManagerBasedRlEnv, command_name: str) -> WalkingCommand:
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, WalkingCommand):
        raise TypeError(f"command {command_name!r} is not WalkingCommand")
    return command


def phase_sin(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return _command(env, command_name).phase_features[:, 0:1]


def phase_cos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return _command(env, command_name).phase_features[:, 1:2]


def walk_blend(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return _command(env, command_name).blend[:, None]


def track_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Match upstream velocity reward while logging the physical tracking error."""
    robot: Entity = env.scene[asset_cfg.name]
    command = _command(env, command_name).command
    actual = robot.data.root_link_lin_vel_b
    error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    error += torch.square(actual[:, 2])
    env.extras["log"]["Errors/base_linear_velocity_rms_m_s"] = torch.sqrt(error).mean()
    return torch.exp(-error / std**2)


def commanded_forward_progress(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Provide bounded, non-saturating progress only while forward motion is requested."""
    command = _command(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    reference_speed = command.cfg.reference_speed_m_s
    command_fraction = torch.clamp(command.command[:, 0] / reference_speed, 0.0, 1.0)
    normalized_velocity = torch.clamp(
        robot.data.root_link_lin_vel_b[:, 0] / reference_speed,
        min=-1.0,
        max=1.5,
    )
    progress = command_fraction * normalized_velocity
    env.extras["log"]["Metrics/commanded_forward_progress"] = progress.mean()
    return progress


def reference_joint_pose(
    env: ManagerBasedRlEnv,
    command_name: str,
    std_rad: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    command = _command(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    alpha = command.blend[:, None]
    target = robot.data.default_joint_pos * (1 - alpha) + command.joint_position * alpha
    error = torch.mean(torch.square(robot.data.joint_pos - target), dim=1)
    env.extras["log"]["Errors/reference_joint_pose_rms_rad"] = torch.sqrt(error).mean()
    return torch.exp(-error / std_rad**2)


def reference_joint_velocity(
    env: ManagerBasedRlEnv,
    command_name: str,
    std_rad_s: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    command = _command(env, command_name)
    target = command.joint_velocity * command.blend[:, None]
    robot: Entity = env.scene[asset_cfg.name]
    error = torch.mean(torch.square(robot.data.joint_vel - target), dim=1)
    env.extras["log"]["Errors/reference_joint_velocity_rms_rad_s"] = torch.sqrt(error).mean()
    return torch.exp(-error / std_rad_s**2)


def heading_frame_delta(
    world_delta: torch.Tensor, root_quaternion_wxyz: torch.Tensor
) -> torch.Tensor:
    """Rotate batched world-frame vectors into each root's yaw-only frame."""
    quaternion = torch.nn.functional.normalize(root_quaternion_wxyz, dim=-1)
    heading = yaw_quat(quaternion)
    repeated = heading[:, None, :].expand(-1, world_delta.shape[1], -1)
    return quat_apply_inverse(repeated, world_delta)


class reference_foot_position:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        command = _command(env, cfg.params["command_name"])
        robot: Entity = env.scene["robot"]
        names = ("left_ankle_roll_link", "right_ankle_roll_link")
        self._robot_ids = torch.as_tensor(
            robot.find_bodies(names, preserve_order=True)[0], device=env.device
        )
        self._reference_ids = torch.as_tensor(
            [command.reference_body_names.index(name) for name in names], device=env.device
        )
        self._env = env

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std_m: float,
    ) -> torch.Tensor:
        command = _command(env, command_name)
        robot: Entity = env.scene["robot"]
        root = robot.data.root_link_pos_w
        simulated_delta_w = robot.data.body_link_pos_w[:, self._robot_ids] - root[:, None, :]
        simulated_b = heading_frame_delta(simulated_delta_w, robot.data.root_link_quat_w)
        reference_body = command.interpolate(command.reference_body_position)
        reference_delta_w = reference_body[:, self._reference_ids] - reference_body[:, 0:1]
        reference_quaternion = command.interpolate(command.reference_body_quaternion)[:, 0]
        reference_b = heading_frame_delta(reference_delta_w, reference_quaternion)
        error = torch.mean(torch.sum(torch.square(simulated_b - reference_b), dim=2), dim=1)
        env.extras["log"]["Errors/reference_foot_position_rms_m"] = torch.sqrt(error).mean()
        return torch.exp(-error / std_m**2) * command.blend


def reference_contact_timing(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    command = _command(env, command_name)
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    actual = sensor.data.found > 0
    agreement = (actual == command.foot_contact).float().mean(dim=1)
    env.extras["log"]["Metrics/reference_contact_agreement"] = agreement.mean()
    return agreement * command.blend


def crouch_cost(
    env: ManagerBasedRlEnv,
    minimum_height_m: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    return torch.square(torch.relu(minimum_height_m - robot.data.root_link_pos_w[:, 2]))


def low_height(
    env: ManagerBasedRlEnv,
    minimum_height_m: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    return robot.data.root_link_pos_w[:, 2] < minimum_height_m


def nonfinite_state(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    finite = (
        torch.isfinite(robot.data.root_link_pose_w).all(dim=1)
        & torch.isfinite(robot.data.root_link_vel_w).all(dim=1)
        & torch.isfinite(robot.data.joint_pos).all(dim=1)
        & torch.isfinite(robot.data.joint_vel).all(dim=1)
    )
    return ~finite


def reset_walking_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    reference_initialization: bool,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    command = _command(env, command_name)
    _write_walking_state(env, env_ids, command, reference_initialization)


def _write_walking_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command: WalkingCommand,
    reference_initialization: bool,
) -> None:
    robot: Entity = env.scene["robot"]
    default_root = robot.data.default_root_state[env_ids].clone()
    default_root[:, :3] += env.scene.env_origins[env_ids]
    default_joint_position = robot.data.default_joint_pos[env_ids].clone()
    default_joint_velocity = robot.data.default_joint_vel[env_ids].clone()
    if reference_initialization:
        moving = command.requested_command[env_ids, 0] > command.cfg.walk_threshold_m_s
        reference_body_position = command.interpolate(command.reference_body_position)[env_ids]
        reference_body_quaternion = command.interpolate(command.reference_body_quaternion)[env_ids]
        reference_body_linear_velocity = command.interpolate(
            command.reference_body_linear_velocity
        )[env_ids]
        reference_body_angular_velocity = command.interpolate(
            command.reference_body_angular_velocity
        )[env_ids]
        reference_joint_position = command.joint_position[env_ids]
        reference_joint_velocity = command.joint_velocity[env_ids]
        default_root[moving, :3] = reference_body_position[moving, 0]
        default_root[moving, :2] += env.scene.env_origins[env_ids[moving], :2]
        default_root[moving, 3:7] = reference_body_quaternion[moving, 0]
        default_root[moving, 7:10] = reference_body_linear_velocity[moving, 0]
        default_root[moving, 10:13] = reference_body_angular_velocity[moving, 0]
        default_joint_position[moving] = reference_joint_position[moving]
        default_joint_velocity[moving] = reference_joint_velocity[moving]
        command._applied[env_ids[moving]] = command.requested_command[env_ids[moving]]
        command.blend[env_ids[moving]] = 1.0
        command.walking[env_ids[moving]] = True
    robot.write_root_state_to_sim(default_root, env_ids=env_ids)
    robot.write_joint_state_to_sim(default_joint_position, default_joint_velocity, env_ids=env_ids)
