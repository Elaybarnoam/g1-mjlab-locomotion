"""Walking-v1 policy contract and simulator task factory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts import PolicyContract, fields_from_sizes
from ..motion import G1_JOINT_NAMES
from ..motion.reference import MotionManifest

TASK_ID = "G1-Walking-Flat-v1"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _motion_file() -> Path:
    config_root = _repository_root() / "configs" / "walking-v1"
    manifest = MotionManifest.load(config_root / "motion-manifest.json")
    manifest.verify_assets(config_root)
    return config_root / manifest.assets[0].path


def _reward_weights() -> dict[str, float]:
    path = _repository_root() / "configs" / "walking-v1" / "rewards.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or not isinstance(raw.get("terms"), dict):
        raise ValueError("invalid walking-v1 reward profile")
    return {name: float(term["weight"]) for name, term in raw["terms"].items()}


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
    from .walking_mdp import WalkingCommandCfg

    command = env.commands["twist"]
    if not isinstance(command, WalkingCommandCfg):
        raise TypeError("walking-v1 requires WalkingCommandCfg")
    command.randomize_phase = randomized_reset
    env.events["reset_walking_state"].params["reference_initialization"] = randomized_reset


def register_task() -> None:
    from mjlab.envs import mdp as envs_mdp
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.tasks.registry import list_tasks, register_mjlab_task
    from mjlab.tasks.velocity import mdp
    from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
    from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

    from ..standing_task import mean_squared_effort_cost
    from . import walking_mdp

    if TASK_ID in list_tasks():
        return
    motion_file = _motion_file()
    weight = _reward_weights()
    env = unitree_g1_flat_env_cfg()
    env.commands["twist"] = walking_mdp.WalkingCommandCfg(
        motion_file=str(motion_file),
        reference_speed_m_s=1.16381159304071,
        cycle_duration_s=1.06,
        acceleration_m_s2=0.6,
        deceleration_m_s2=0.8,
        blend_rate_s=1.0,
        stand_threshold_m_s=0.05,
        walk_threshold_m_s=0.15,
        standing_fraction=0.2,
        randomize_phase=True,
        resampling_time_range=(4.0, 10.0),
        debug_vis=False,
    )
    for group in env.observations.values():
        group.terms["phase_sin"] = ObservationTermCfg(
            func=walking_mdp.phase_sin, params={"command_name": "twist"}
        )
        group.terms["phase_cos"] = ObservationTermCfg(
            func=walking_mdp.phase_cos, params={"command_name": "twist"}
        )
        group.terms["walk_blend"] = ObservationTermCfg(
            func=walking_mdp.walk_blend, params={"command_name": "twist"}
        )
    env.observations["actor"].enable_corruption = False

    forbidden_ground = ContactSensorCfg(
        name="forbidden_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(?!left_ankle_roll_link$|right_ankle_roll_link$).+$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        history_length=env.decimation,
    )
    env.scene.sensors = (env.scene.sensors or ()) + (forbidden_ground,)

    env.events = {
        "reset_walking_state": EventTermCfg(
            func=walking_mdp.reset_walking_state,
            mode="reset",
            params={"command_name": "twist", "reference_initialization": True},
        )
    }
    env.curriculum = {}
    env.rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=weight["track_linear_velocity"],
            params={"command_name": "twist", "std": 0.35},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=weight["track_angular_velocity"],
            params={"command_name": "twist", "std": 0.35},
        ),
        "upright": RewardTermCfg(
            func=mdp.upright,
            weight=weight["upright"],
            params={
                "std": 0.35,
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            },
        ),
        "reference_joint_pose": RewardTermCfg(
            func=walking_mdp.reference_joint_pose,
            weight=weight["reference_joint_pose"],
            params={"command_name": "twist", "std_rad": 0.20},
        ),
        "reference_joint_velocity": RewardTermCfg(
            func=walking_mdp.reference_joint_velocity,
            weight=weight["reference_joint_velocity"],
            params={"command_name": "twist", "std_rad_s": 1.5},
        ),
        "reference_foot_position": RewardTermCfg(
            func=walking_mdp.reference_foot_position,
            weight=weight["reference_foot_position"],
            params={"command_name": "twist", "std_m": 0.12},
        ),
        "reference_contact_timing": RewardTermCfg(
            func=walking_mdp.reference_contact_timing,
            weight=weight["reference_contact_timing"],
            params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
        ),
        "foot_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=weight["foot_slip"],
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            },
        ),
        "crouch": RewardTermCfg(
            func=walking_mdp.crouch_cost,
            weight=weight["crouch"],
            params={"minimum_height_m": 0.62},
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=weight["dof_pos_limits"]),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=weight["action_rate_l2"]),
        "effort": RewardTermCfg(func=mean_squared_effort_cost, weight=weight["effort"]),
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=weight["self_collisions"],
            params={"sensor_name": "self_collision", "force_threshold": 10.0},
        ),
        "termination": RewardTermCfg(
            func=envs_mdp.is_terminated,
            weight=weight["termination"] / (env.sim.mujoco.timestep * env.decimation),
        ),
    }
    env.terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": 1.0471975512}
        ),
        "low_height": TerminationTermCfg(
            func=walking_mdp.low_height, params={"minimum_height_m": 0.50}
        ),
        "nonfinite_state": TerminationTermCfg(func=walking_mdp.nonfinite_state),
        "forbidden_ground_contact": TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_name": "forbidden_ground_contact", "force_threshold": 20.0},
        ),
    }
    register_mjlab_task(TASK_ID, env, env, unitree_g1_ppo_runner_cfg())
