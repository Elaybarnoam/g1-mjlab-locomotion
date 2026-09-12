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
        action_semantics=(
            "29 normalized joint-position offsets around the robot nominal pose: "
            "q_target = q_nominal + per_joint_scale * action"
        ),
        actuator_semantics="one built-in position actuator; PD is applied exactly once",
        task_id=TASK_ID,
        layout_id="g1-walking-actor-v1",
        command_semantics=(
            "[vx, vy, yaw_rate] in yaw-aligned body frame; v1 supports "
            "0 <= vx <= 1.163811593 m/s and vy=yaw_rate=0"
        ),
        phase_semantics="phase in [0,1), continuous wrap; rate follows applied vx; retained on stop",
    )


def configure_environment(
    env: Any,
    *,
    randomized_reset: bool = True,
    task_profile: Any = None,
    stage19_reward_profile: Any = None,
) -> None:
    from ..config import Stage19RewardProfile, WalkingTrainingProfile
    from .walking_mdp import WalkingCommandCfg

    command = env.commands["twist"]
    if not isinstance(command, WalkingCommandCfg):
        raise TypeError("walking-v1 requires WalkingCommandCfg")
    if task_profile is not None and not isinstance(task_profile, WalkingTrainingProfile):
        raise TypeError("walking-v1 requires WalkingTrainingProfile")
    if stage19_reward_profile is not None and not isinstance(
        stage19_reward_profile, Stage19RewardProfile
    ):
        raise TypeError("walking-v1 requires Stage19RewardProfile")
    command.randomize_phase = (
        task_profile.randomize_phase if task_profile is not None else randomized_reset
    )
    domain_randomization = (
        task_profile.domain_randomization
        if task_profile is not None and task_profile.domain_randomization is not None
        else randomized_reset
    )
    observation_noise = (
        task_profile.observation_noise
        if task_profile is not None and task_profile.observation_noise is not None
        else randomized_reset
    )
    env.observations["actor"].enable_corruption = observation_noise
    if not domain_randomization:
        env.events = {}
    if task_profile is not None:
        command.standing_fraction = task_profile.standing_fraction
        if task_profile.forward_speed_range_m_s is not None:
            command.moving_speed_min_m_s, command.moving_speed_max_m_s = (
                task_profile.forward_speed_range_m_s
            )
        if task_profile.objective == "locomotion_bootstrap":
            _configure_locomotion_bootstrap(env)
        else:
            _configure_reference_style(env)
        if task_profile.velocity_tracking_std_m_s is not None:
            env.rewards["track_linear_velocity"].params["std"] = (
                task_profile.velocity_tracking_std_m_s
            )
        if task_profile.forward_progress_weight is not None:
            env.rewards["commanded_forward_progress"].weight = task_profile.forward_progress_weight
        if task_profile.reference_foot_position_std_m is not None:
            env.rewards["reference_foot_position"].params["std_m"] = (
                task_profile.reference_foot_position_std_m
            )
    command.reference_initialization = (
        task_profile.reference_initialization if task_profile is not None else randomized_reset
    )
    command.host_semantics_version = (
        task_profile.host_semantics_version if task_profile is not None else 1
    )
    command.reference_ground_offset_m = (
        task_profile.reference_ground_offset_m if task_profile is not None else 0.0
    )
    if stage19_reward_profile is not None:
        configure_stage19_rewards(env, stage19_reward_profile)


def configure_stage19_rewards(env: Any, profile: Any) -> None:
    """Apply a complete allowlisted profile; undeclared inherited weights cannot survive."""
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    from ..config import Stage19RewardProfile
    from ..motion.contact import load_stage19_contact_profile
    from . import walking_mdp

    if not isinstance(profile, Stage19RewardProfile):
        raise TypeError("profile must be Stage19RewardProfile")
    terms = profile.by_name()
    contact_profile = load_stage19_contact_profile(
        _repository_root() / "configs/walking-v1/stage19/contact-profile.json"
    )
    command = env.commands["twist"]
    if not isinstance(command, walking_mdp.WalkingCommandCfg):
        raise TypeError("Stage 19 requires WalkingCommandCfg")
    command.stage19_contact_parameters = contact_profile.runtime_parameters()
    expected_parameters = {
        "phase_contact_error": {"walk_threshold_m_s": command.walk_threshold_m_s},
        "extra_contact_event": {
            "phase_tolerance_cycle": contact_profile.phase_tolerance_cycle,
            "transition_grace_s": contact_profile.transition_grace_s,
        },
        "short_stance": {
            "minimum_duration_s": contact_profile.minimum_stance_duration_s,
            "transition_grace_s": contact_profile.transition_grace_s,
        },
        "short_swing": {
            "minimum_duration_s": contact_profile.minimum_swing_duration_s,
            "transition_grace_s": contact_profile.transition_grace_s,
        },
        "physical_stance_slip": {
            "slip_scale_m_s": contact_profile.slip_scale_m_s,
            "squared_error_clip": contact_profile.squared_error_clip,
        },
        "swing_clearance_error": {
            "reference_clearance_m": contact_profile.swing_clearance_reference_m,
            "clearance_scale_m": contact_profile.clearance_scale_m,
            "squared_error_clip": contact_profile.squared_error_clip,
        },
        "touchdown_placement": {
            "minimum_root_progress_fraction": contact_profile.minimum_root_progress_fraction,
            "minimum_step_reference_m": contact_profile.minimum_step_reference_m,
            "minimum_swing_clearance_m": contact_profile.minimum_swing_clearance_m,
            "placement_scale_m": contact_profile.placement_scale_m,
            "step_reference_m": contact_profile.step_reference_m,
        },
        "bilateral_flight": {
            "minimum_duration_s": contact_profile.bilateral_flight_duration_s,
        },
    }
    for name, expected in expected_parameters.items():
        if terms[name].parameter_dict != expected:
            raise ValueError(f"{name} parameters differ from the frozen contact profile")
    sensor = ContactSensorCfg(
        name="stage19_feet_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "pos", "normal", "tangent"),
        reduce="none",
        num_slots=contact_profile.contact_slots,
        global_frame=True,
    )
    env.scene.sensors = (env.scene.sensors or ()) + (sensor,)
    stateful = {
        "phase_contact_error": walking_mdp.phase_contact_error,
        "extra_contact_event": walking_mdp.extra_contact_event,
        "short_stance": walking_mdp.short_stance,
        "short_swing": walking_mdp.short_swing,
        "physical_stance_slip": walking_mdp.physical_stance_slip,
        "swing_clearance_error": walking_mdp.swing_clearance_error,
        "touchdown_placement": walking_mdp.touchdown_placement,
        "bilateral_flight": walking_mdp.bilateral_flight,
    }
    for name, function in stateful.items():
        term = terms[name]
        env.rewards[name] = RewardTermCfg(
            func=function,
            weight=term.manager_weight(env.sim.mujoco.timestep * env.decimation),
            params={"command_name": "twist"},
        )
    missing = set(terms) - set(env.rewards)
    if missing:
        raise ValueError(f"Stage 19 profile names missing from task: {sorted(missing)}")
    control_dt = env.sim.mujoco.timestep * env.decimation
    for name, reward in env.rewards.items():
        term = terms[name]
        reward.weight = term.manager_weight(control_dt) if term.enabled else 0.0


def _configure_locomotion_bootstrap(env: Any) -> None:
    """Resolve the pinned mjlab G1 flat locomotion objective without style imitation."""
    weights = {
        "track_linear_velocity": 2.0,
        "track_angular_velocity": 2.0,
        "upright": 1.0,
        "pose": 1.0,
        "body_ang_vel": -0.05,
        "angular_momentum": -0.02,
        "dof_pos_limits": -1.0,
        "action_rate_l2": -0.1,
        "air_time": 0.0,
        "foot_clearance": -2.0,
        "foot_swing_height": -0.25,
        "foot_slip": -0.1,
        "soft_landing": -1e-5,
        "self_collisions": -1.0,
        "termination": 0.0,
    }
    for name, value in weights.items():
        env.rewards[name].weight = value
    env.rewards["track_linear_velocity"].params["std"] = 0.5
    env.rewards["track_angular_velocity"].params["std"] = 0.5**0.5
    env.rewards["upright"].params["std"] = 0.2**0.5
    for name in (
        "commanded_forward_progress",
        "reference_joint_pose",
        "reference_joint_velocity",
        "reference_foot_position",
        "reference_contact_timing",
        "crouch",
        "effort",
    ):
        env.rewards[name].weight = 0.0
    # Contact/height gates are evaluation criteria, not bootstrap terminations. Early random
    # policies routinely touch a knee or dip below the final height threshold; resetting there
    # collapses rollouts before PPO can learn recovery. Retain non-finite termination as a safety
    # guard in addition to the pinned timeout/fall conditions.
    env.terminations.pop("low_height", None)
    env.terminations.pop("forbidden_ground_contact", None)


def _configure_reference_style(env: Any) -> None:
    """Enable the versioned imitation terms while retaining locomotion regularizers."""
    weights = _reward_weights()
    for name, value in weights.items():
        env.rewards[name].weight = value
    control_dt = env.sim.mujoco.timestep * env.decimation
    env.rewards["termination"].weight = weights["termination"] / control_dt


def register_task() -> None:
    from mjlab.envs import mdp as envs_mdp
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
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
        reference_initialization=True,
        host_semantics_version=1,
        reference_ground_offset_m=0.0,
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

    # The upstream velocity curriculum assumes UniformVelocityCommand internals. Walking-v1 owns
    # its command schedule, so it intentionally keeps the upstream reset/randomization events but
    # resolves curriculum stages through versioned WalkingTrainingProfile files.
    env.curriculum = {}
    env.rewards["track_linear_velocity"] = RewardTermCfg(
        func=walking_mdp.track_linear_velocity,
        weight=weight["track_linear_velocity"],
        params={"command_name": "twist", "std": 0.35},
    )
    env.rewards.update(
        {
            "commanded_forward_progress": RewardTermCfg(
                func=walking_mdp.commanded_forward_progress,
                weight=weight["commanded_forward_progress"],
                params={"command_name": "twist"},
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
            "crouch": RewardTermCfg(
                func=walking_mdp.crouch_cost,
                weight=weight["crouch"],
                params={"minimum_height_m": 0.62},
            ),
            "effort": RewardTermCfg(func=mean_squared_effort_cost, weight=weight["effort"]),
            "termination": RewardTermCfg(
                func=envs_mdp.is_terminated,
                weight=weight["termination"] / (env.sim.mujoco.timestep * env.decimation),
            ),
        }
    )
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
