"""Walking-v1 static contract and, after W04, simulator task hooks."""

from __future__ import annotations

from typing import Any

from ..contracts import PolicyContract, fields_from_sizes
from ..motion import G1_JOINT_NAMES

TASK_ID = "G1-Walking-Flat-v1"


def walking_policy_contract() -> PolicyContract:
    """Return the deployable 102-input actor and privileged 114-input critic contract."""
    common = (
        ("base_lin_vel", 3, "m/s", "body", "pelvis linear-velocity estimate"),
        ("base_ang_vel", 3, "rad/s", "body", "pelvis angular-velocity estimate"),
        ("projected_gravity", 3, "1", "body", "gravity projected through pelvis orientation"),
        ("joint_pos", 29, "rad", "joint", "encoder position relative to nominal pose"),
        ("joint_vel", 29, "rad/s", "joint", "encoder velocity"),
        ("actions", 29, "1", "joint", "previous applied policy action"),
        ("command", 3, "m/s,m/s,rad/s", "yaw-aligned body", "applied velocity command"),
    )
    gait = (
        ("phase_sin", 1, "1", "gait", "sin(2*pi*phase)"),
        ("phase_cos", 1, "1", "gait", "cos(2*pi*phase)"),
        ("walk_blend", 1, "1", "gait", "rate-limited stand/walk blend"),
    )
    privileged = (
        ("foot_height", 2, "m", "world", "simulator foot-height ray sensors"),
        ("foot_air_time", 2, "s", "world", "simulator ground-contact history"),
        ("foot_contact", 2, "bool", "world", "simulator ground-contact state"),
        ("foot_contact_forces", 6, "N", "world", "simulator ground-contact force vectors"),
    )
    return PolicyContract(
        schema_version=1,
        robot="Unitree G1 29-DOF",
        model_revision="mjlab@8ee51fbcf806a7419189f706d9e394cbeb7790fa",
        joint_names=G1_JOINT_NAMES,
        actor_fields=fields_from_sizes((*common, *gait)),
        critic_fields=fields_from_sizes((*common, *privileged, *gait)),
        action_names=G1_JOINT_NAMES,
        physics_dt=0.005,
        control_dt=0.02,
        action_semantics="29 normalized position offsets around the nominal G1 pose",
        actuator_semantics="one built-in position actuator; PD is applied exactly once",
        task_id=TASK_ID,
        layout_id="g1-walking-actor-v1",
        command_semantics="[vx, vy, yaw_rate] in yaw-aligned body frame; v1 supports vx=0 or 1.163811593 m/s and vy=yaw_rate=0",
        phase_semantics="phase in [0,1), continuous wrap; rate follows applied vx; retained on stop",
    )


def configure_environment(env: Any, *, randomized_reset: bool = True) -> None:
    del env, randomized_reset
    raise RuntimeError("walking-v1 MDP construction is implemented by ticket W04")


def register_task() -> None:
    raise RuntimeError("walking-v1 simulator registration is implemented by ticket W04")
