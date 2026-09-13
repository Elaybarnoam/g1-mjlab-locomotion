"""mjlab adapter for the walking-v2 reference-residual MDP."""

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

from ..motion.reference_bank import ReferenceBank
from ..motion.walking_target_state import (
    WalkingTargetProfile,
    predict_walking_target_torch,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


@dataclass(kw_only=True)
class WalkingV2CommandCfg(CommandTermCfg):
    bank_file: str
    standing_fraction: float = 0.2
    forward_speed_range_m_s: tuple[float, float] = (0.4, 0.8)
    randomize_phase: bool = True
    reference_initialization: bool = True

    def build(self, env: ManagerBasedRlEnv) -> WalkingV2Command:
        return WalkingV2Command(self, env)


class WalkingV2Command(CommandTerm):
    """GPU-resident host state plus one immutable speed-indexed bank."""

    cfg: WalkingV2CommandCfg

    def __init__(self, cfg: WalkingV2CommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        from pathlib import Path

        self.profile = WalkingTargetProfile()
        self.profile.validate()
        if not np.isclose(env.step_dt, self.profile.policy_dt_s):
            raise ValueError("walking-v2 requires a 0.020 s policy interval")
        if not 0.0 <= cfg.standing_fraction <= 1.0:
            raise ValueError("standing_fraction must be in [0, 1]")
        low, high = cfg.forward_speed_range_m_s
        if not 0.0 <= low <= high <= self.profile.maximum_forward_speed_m_s:
            raise ValueError("forward_speed_range_m_s must be ordered within [0, 0.8]")
        self.bank = ReferenceBank.load(Path(cfg.bank_file))
        robot: Entity = env.scene["robot"]
        if self.bank.metadata.joint_names != tuple(robot.joint_names):
            raise ValueError("walking-v2 bank and robot joint order differ")
        self._requested = torch.zeros((self.num_envs, 3), device=self.device)
        self._applied = torch.zeros_like(self._requested)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.blend = torch.zeros(self.num_envs, device=self.device)
        self.nominal_joint_position = torch.as_tensor(
            self.bank.nominal_joint_position,
            device=self.device,
            dtype=self._applied.dtype,
        )
        self.target_joint_position = self.nominal_joint_position.repeat(self.num_envs, 1)
        self.target_joint_velocity = torch.zeros_like(self.target_joint_position)
        self.expected_foot_position = torch.zeros((self.num_envs, 2, 3), device=self.device)
        self.expected_contact = torch.ones((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.command_rate_m_s2 = torch.zeros(self.num_envs, device=self.device)
        self.blend_rate_s = torch.zeros(self.num_envs, device=self.device)
        self.phase_rate_hz = torch.zeros(self.num_envs, device=self.device)
        self._command_dt: float | torch.Tensor = 0.0
        self.metrics["command_filter_error_m_s"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._applied

    @property
    def requested_command(self) -> torch.Tensor:
        return self._requested

    @property
    def phase_features(self) -> torch.Tensor:
        angle = 2.0 * torch.pi * self.phase
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=1)

    def set_requested_forward_speed(self, speed_m_s: float) -> None:
        validated = self.profile.validate_requested([[speed_m_s, 0.0, 0.0]])
        self._requested.zero_()
        self._requested[:, 0] = float(validated[0, 0])

    def _refresh_reference(self, ids: Any = slice(None)) -> None:
        sample = self.bank.sample_torch(self.phase[ids], self._applied[ids, 0])
        nominal = self.nominal_joint_position
        self.target_joint_position[ids] = nominal + self.blend[ids, None] * (
            sample.joint_position - nominal
        )
        self.target_joint_velocity[ids] = self.blend[ids, None] * (
            sample.joint_partial_phase * sample.phase_rate_hz[:, None]
        )
        self.expected_foot_position[ids] = sample.local_foot_position
        self.expected_contact[ids] = sample.contact
        self.phase_rate_hz[ids] = sample.phase_rate_hz

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        assert isinstance(env_ids, torch.Tensor)
        self._applied[env_ids] = 0.0
        self.blend[env_ids] = 0.0
        self.command_rate_m_s2[env_ids] = 0.0
        self.blend_rate_s[env_ids] = 0.0
        self.phase_rate_hz[env_ids] = 0.0
        if self.cfg.randomize_phase:
            self.phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        else:
            self.phase[env_ids] = 0.0
        extras = super().reset(env_ids)
        if self.cfg.reference_initialization:
            moving = self._requested[env_ids, 0] > 0.0
            moving_ids = env_ids[moving]
            self._applied[moving_ids] = self._requested[moving_ids]
            speed_fraction = torch.clamp(
                self._applied[moving_ids, 0] / self.profile.full_walk_blend_speed_m_s,
                min=0.0,
                max=1.0,
            )
            self.blend[moving_ids] = speed_fraction.square() * (3.0 - 2.0 * speed_fraction)
        self._refresh_reference(env_ids)
        if self.cfg.reference_initialization:
            _write_reference_initial_state(self._env, env_ids, self)
        return extras

    def _update_metrics(self) -> None:
        self.metrics["command_filter_error_m_s"] = torch.abs(
            self._requested[:, 0] - self._applied[:, 0]
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        standing = torch.rand(len(env_ids), device=self.device) < self.cfg.standing_fraction
        moving_speed = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.forward_speed_range_m_s
        )
        self._requested[env_ids] = 0.0
        self._requested[env_ids, 0] = torch.where(standing, 0.0, moving_speed)

    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        self._command_dt = dt
        super().compute(dt, env_ids)

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        ids: Any = slice(None) if env_ids is None else env_ids
        dt = self._command_dt
        if env_ids is not None and not isinstance(dt, torch.Tensor) and dt == 0.0:
            self._refresh_reference(ids)
            return
        previous = (
            self._applied[ids].clone(),
            self.phase[ids].clone(),
            self.blend[ids].clone(),
            self.target_joint_position[ids].clone(),
            self.target_joint_velocity[ids].clone(),
            self.expected_foot_position[ids].clone(),
            self.expected_contact[ids].clone(),
        )
        transition = predict_walking_target_torch(
            self._applied[ids],
            self.phase[ids],
            self.blend[ids],
            self._requested[ids],
            self.bank,
            self.profile,
        )
        self._applied[ids] = transition.next_applied_command
        self.phase[ids] = transition.next_phase
        self.blend[ids] = transition.next_blend
        self.target_joint_position[ids] = transition.target_joint_position
        self.target_joint_velocity[ids] = transition.target_joint_velocity
        self.command_rate_m_s2[ids] = transition.command_rate_m_s2
        self.blend_rate_s[ids] = transition.blend_rate_s
        self.phase_rate_hz[ids] = transition.phase_rate_hz
        sample = self.bank.sample_torch(self.phase[ids], self._applied[ids, 0])
        self.expected_foot_position[ids] = sample.local_foot_position
        self.expected_contact[ids] = sample.contact
        if isinstance(dt, torch.Tensor):
            reset_mask = dt == 0.0
            if bool(reset_mask.any()):
                old = previous
                reset_ids = (
                    torch.nonzero(reset_mask).flatten() if env_ids is None else env_ids[reset_mask]
                )
                self._applied[reset_ids] = old[0][reset_mask]
                self.phase[reset_ids] = old[1][reset_mask]
                self.blend[reset_ids] = old[2][reset_mask]
                self.target_joint_position[reset_ids] = old[3][reset_mask]
                self.target_joint_velocity[reset_ids] = old[4][reset_mask]
                self.expected_foot_position[reset_ids] = old[5][reset_mask]
                self.expected_contact[reset_ids] = old[6][reset_mask]
                self.command_rate_m_s2[reset_ids] = 0.0
                self.blend_rate_s[reset_ids] = 0.0


@dataclass(kw_only=True)
class ReferenceResidualActionCfg(JointPositionActionCfg):
    command_name: str

    def build(self, env: ManagerBasedRlEnv) -> ReferenceResidualAction:
        return ReferenceResidualAction(self, env)


class ReferenceResidualAction(JointPositionAction):
    """Center unfiltered residual actions on the cached interval-end reference."""

    cfg: ReferenceResidualActionCfg

    @property
    def joint_target(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        command = _command(self._env, self.cfg.command_name)
        if command.target_joint_position.shape != self._processed_actions.shape:
            raise ValueError("reference target and action shapes differ")
        self._processed_actions = command.target_joint_position + self._processed_actions


def _command(env: ManagerBasedRlEnv, command_name: str) -> WalkingV2Command:
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, WalkingV2Command):
        raise TypeError(f"command {command_name!r} is not WalkingV2Command")
    return command


def phase_sin(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return _command(env, command_name).phase_features[:, 0:1]


def phase_cos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return _command(env, command_name).phase_features[:, 1:2]


def walk_blend(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return _command(env, command_name).blend[:, None]


def reference_offset(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    return command.target_joint_position - command.nominal_joint_position


def reference_velocity(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return _command(env, command_name).target_joint_velocity


def raw_foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.force is not None
    return sensor.data.force.reshape(env.num_envs, -1)[:, :6]


def _heading_frame_delta(
    world_delta: torch.Tensor, root_quaternion_wxyz: torch.Tensor
) -> torch.Tensor:
    heading = yaw_quat(torch.nn.functional.normalize(root_quaternion_wxyz, dim=-1))
    repeated = heading[:, None, :].expand(-1, world_delta.shape[1], -1)
    return cast(torch.Tensor, quat_apply_inverse(repeated, world_delta))


def _effort_limits(robot: Entity, device: str) -> torch.Tensor:
    values: list[float] = []
    for name in robot.joint_names:
        if any(token in name for token in ("wrist_pitch", "wrist_yaw")):
            values.append(5.0)
        elif any(
            token in name
            for token in ("elbow", "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "wrist_roll")
        ):
            values.append(25.0)
        elif any(token in name for token in ("hip_roll", "knee")):
            values.append(139.0)
        elif any(token in name for token in ("hip_pitch", "hip_yaw", "waist_yaw")):
            values.append(88.0)
        else:
            values.append(50.0)
    return torch.tensor(values, device=device)


class walking_v2_rate:
    """Single auditable reward term that logs every raw and weighted component."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        robot: Entity = env.scene["robot"]
        self._site_ids = torch.as_tensor(
            robot.find_sites(("left_foot", "right_foot"), preserve_order=True)[0],
            device=env.device,
        )
        self._effort_limit = _effort_limits(robot, env.device)

    def __call__(self, env: ManagerBasedRlEnv, command_name: str, sensor_name: str) -> torch.Tensor:
        robot: Entity = env.scene["robot"]
        command = _command(env, command_name)
        contact_sensor: ContactSensor = env.scene[sensor_name]
        assert contact_sensor.data.found is not None
        actual_contact = contact_sensor.data.found.reshape(env.num_envs, -1)[:, :2] > 0
        actual_foot = _heading_frame_delta(
            robot.data.site_pos_w[:, self._site_ids] - robot.data.root_link_pos_w[:, None, :],
            robot.data.root_link_quat_w,
        )
        actual_velocity = torch.stack(
            (
                robot.data.root_link_lin_vel_b[:, 0],
                robot.data.root_link_lin_vel_b[:, 1],
                robot.data.root_link_ang_vel_b[:, 2],
            ),
            dim=1,
        )
        joint_error = robot.data.joint_pos - command.target_joint_position
        velocity_error = robot.data.joint_vel - command.target_joint_velocity
        foot_error = actual_foot - command.expected_foot_position
        orientation_error = torch.acos(
            torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
        )
        agreement = (actual_contact == command.expected_contact).float().mean(dim=1)
        raw = {
            "imitation_joint_pose": torch.exp(-joint_error.square().mean(dim=1) / 0.30**2),
            "imitation_joint_velocity": torch.exp(-velocity_error.square().mean(dim=1) / 3.0**2),
            "imitation_local_feet": torch.exp(-foot_error.square().mean(dim=(1, 2)) / 0.10**2),
            "imitation_orientation": torch.exp(-orientation_error.square() / 0.25**2),
            "imitation_contact": agreement,
            "task_forward_velocity": torch.exp(
                -(actual_velocity[:, 0] - command.command[:, 0]).square() / 0.25**2
            ),
            "task_lateral_velocity": torch.exp(
                -(actual_velocity[:, 1] - command.command[:, 1]).square() / 0.15**2
            ),
            "task_yaw_rate": torch.exp(
                -(actual_velocity[:, 2] - command.command[:, 2]).square() / 0.25**2
            ),
            "stand_pose": torch.exp(
                -(robot.data.joint_pos - robot.data.default_joint_pos).square().mean(dim=1)
                / 0.30**2
            ),
            "stand_horizontal_velocity": torch.exp(
                -actual_velocity[:, :2].square().sum(dim=1) / 0.10**2
            ),
            "action_rate": (env.action_manager.action - env.action_manager.prev_action)
            .square()
            .mean(dim=1),
            "normalized_torque": (robot.data.qfrc_actuator / self._effort_limit)
            .square()
            .mean(dim=1),
        }
        limits = robot.data.soft_joint_pos_limits
        assert limits is not None
        violation = torch.relu(limits[:, :, 0] - robot.data.joint_pos) + torch.relu(
            robot.data.joint_pos - limits[:, :, 1]
        )
        raw["soft_joint_limit"] = violation.square().sum(dim=1)
        imitation = (
            0.40 * raw["imitation_joint_pose"]
            + 0.10 * raw["imitation_joint_velocity"]
            + 0.25 * raw["imitation_local_feet"]
            + 0.15 * raw["imitation_orientation"]
            + 0.10 * raw["imitation_contact"]
        )
        task = (
            raw["task_forward_velocity"]
            + 0.25 * raw["task_lateral_velocity"]
            + 0.25 * raw["task_yaw_rate"]
        )
        stand = raw["stand_pose"] + raw["stand_horizontal_velocity"]
        rate = (
            2.0 * command.blend * imitation
            + task
            + (1.0 - command.blend) * stand
            - 0.10 * raw["action_rate"]
            - 0.05 * raw["normalized_torque"]
            - 0.50 * raw["soft_joint_limit"]
        )
        weights = {
            "imitation_joint_pose": 0.80 * command.blend,
            "imitation_joint_velocity": 0.20 * command.blend,
            "imitation_local_feet": 0.50 * command.blend,
            "imitation_orientation": 0.30 * command.blend,
            "imitation_contact": 0.20 * command.blend,
            "task_forward_velocity": torch.ones_like(command.blend),
            "task_lateral_velocity": torch.full_like(command.blend, 0.25),
            "task_yaw_rate": torch.full_like(command.blend, 0.25),
            "stand_pose": 1.0 - command.blend,
            "stand_horizontal_velocity": 1.0 - command.blend,
            "action_rate": torch.full_like(command.blend, -0.10),
            "normalized_torque": torch.full_like(command.blend, -0.05),
            "soft_joint_limit": torch.full_like(command.blend, -0.50),
        }
        for name, value in raw.items():
            env.extras["log"][f"WalkingV2Raw/{name}"] = value.mean()
            env.extras["log"][f"WalkingV2Weighted/{name}"] = (value * weights[name]).mean()
        env.extras["log"]["WalkingV2/imitation"] = imitation.mean()
        env.extras["log"]["WalkingV2/task"] = task.mean()
        env.extras["log"]["WalkingV2/stand"] = stand.mean()
        env.extras["log"]["WalkingV2/rate"] = rate.mean()
        return rate


def nonfinite_state(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    finite = (
        torch.isfinite(robot.data.root_link_pose_w).all(dim=1)
        & torch.isfinite(robot.data.root_link_vel_w).all(dim=1)
        & torch.isfinite(robot.data.joint_pos).all(dim=1)
        & torch.isfinite(robot.data.joint_vel).all(dim=1)
    )
    return ~finite


def fell_over(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
    return (tilt > 1.0471975512) | (robot.data.root_link_pos_w[:, 2] < 0.50)


def true_fall_event(env: ManagerBasedRlEnv) -> torch.Tensor:
    return fell_over(env).float()


def reference_deviation(
    env: ManagerBasedRlEnv,
    command_name: str,
    maximum_rms_rad: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    command = _command(env, command_name)
    rms = torch.sqrt(
        torch.mean((robot.data.joint_pos - command.target_joint_position).square(), dim=1)
    )
    return (command.blend > 0.05) & (rms > maximum_rms_rad)


def reset_walking_v2(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, command_name: str
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    command = _command(env, command_name)
    _write_reference_initial_state(env, env_ids, command)


def _write_reference_initial_state(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor, command: WalkingV2Command
) -> None:
    robot: Entity = env.scene["robot"]
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    root[:, 7] = command.command[env_ids, 0]
    joint_position = command.target_joint_position[env_ids].clone()
    joint_velocity = command.target_joint_velocity[env_ids].clone()
    robot.write_root_state_to_sim(root, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
