"""Walking-v2 policy contract and lazy mjlab task factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import PolicyContract, fields_from_sizes
from ..motion import G1_JOINT_NAMES

TASK_ID = "G1-Walking-Flat-v2"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def walking_v2_policy_contract() -> PolicyContract:
    """Return the frozen 160-input actor and 172-input privileged critic contract."""
    physical = (
        ("base_lin_vel", 3, "m/s", "body", "pelvis linear velocity"),
        ("base_ang_vel", 3, "rad/s", "body", "pelvis angular velocity"),
        ("projected_gravity", 3, "1", "body", "gravity projected through pelvis"),
        ("joint_pos", 29, "rad", "joint", "joint position minus nominal"),
        ("joint_vel", 29, "rad/s", "joint", "joint velocity"),
        ("previous_action", 29, "1", "joint", "previous applied residual action"),
        ("command", 3, "m/s,m/s,rad/s", "body", "applied velocity command"),
    )
    phase_and_reference = (
        ("phase_sin", 1, "1", "gait", "sin(2*pi*phase)"),
        ("phase_cos", 1, "1", "gait", "cos(2*pi*phase)"),
        ("walk_blend", 1, "1", "gait", "actual rate-limited walk blend"),
        ("reference_offset", 29, "rad", "joint", "target position minus nominal"),
        ("reference_velocity", 29, "rad/s", "joint", "composed target velocity"),
    )
    privileged = (
        ("foot_height", 2, "m", "world", "simulator foot heights"),
        ("foot_air_time", 2, "s", "world", "simulator foot airtimes"),
        ("foot_contact", 2, "bool", "world", "debounced foot contacts"),
        ("foot_contact_forces", 6, "N", "world", "foot contact-force vectors"),
    )
    return PolicyContract(
        schema_version=1,
        robot="Unitree G1 29-DOF",
        model_revision="mjlab@8ee51fbcf806a7419189f706d9e394cbeb7790fa",
        joint_names=G1_JOINT_NAMES,
        actor_fields=fields_from_sizes((*physical, *phase_and_reference)),
        critic_fields=fields_from_sizes((*physical, *privileged, *phase_and_reference)),
        action_names=G1_JOINT_NAMES,
        physics_dt=0.005,
        control_dt=0.020,
        action_semantics=(
            "29 normalized reference residuals: q_target = q_nominal + "
            "blend*(q_reference-q_nominal) + per_joint_scale*residual"
        ),
        actuator_semantics="existing MuJoCo position actuator; PD applied exactly once",
        task_id=TASK_ID,
        layout_id="g1-walking-reference-actor-v2",
        command_semantics="[vx, vy, yaw_rate] body frame; acquisition supports 0..0.8, 0, 0",
        phase_semantics=(
            "phase in [0,1); rate comes from the speed-indexed bank; interval-end target at k+1"
        ),
    )


def validate_walking_v2_checkpoint_metadata(metadata: object) -> None:
    """Reject checkpoints whose vector/action contract predates walking-v2."""
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be an object")
    contract = walking_v2_policy_contract()
    expected = {
        "task_id": TASK_ID,
        "layout_id": contract.layout_id,
        "actor_observation_size": 160,
        "critic_observation_size": 172,
        "action_size": 29,
        "policy_contract_sha256": contract.sha256,
    }
    mismatched = [name for name, value in expected.items() if metadata.get(name) != value]
    if mismatched:
        raise ValueError(f"checkpoint is incompatible with walking-v2: {mismatched}")


def configure_environment(
    env: Any,
    *,
    randomized_reset: bool = True,
    task_profile: Any = None,
    stage19_reward_profile: Any = None,
) -> None:
    """Apply only supported v2 switches; reject v1 profile leakage."""
    if task_profile is not None or stage19_reward_profile is not None:
        raise ValueError("walking-v2 does not accept walking-v1 training or Stage 19 profiles")
    env.observations["actor"].enable_corruption = randomized_reset
    if not randomized_reset:
        env.events = {}
    command = env.commands["twist"]
    command.randomize_phase = randomized_reset
    command.reference_initialization = randomized_reset


def register_task() -> None:
    """Register the v2 task without importing simulator dependencies at package import."""
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.tasks.registry import list_tasks, register_mjlab_task
    from mjlab.tasks.velocity import mdp
    from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
    from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

    from . import walking_v2_mdp

    if TASK_ID in list_tasks():
        return
    root = _repository_root()
    bank_path = root / "configs" / "walking-v2" / "reference" / "reference-bank-v2.json"
    env = unitree_g1_flat_env_cfg()
    env.commands["twist"] = walking_v2_mdp.WalkingV2CommandCfg(
        bank_file=str(bank_path),
        standing_fraction=0.2,
        randomize_phase=True,
        reference_initialization=True,
        resampling_time_range=(4.0, 10.0),
        debug_vis=False,
    )
    source_action: Any = env.actions["joint_pos"]
    env.actions["joint_pos"] = walking_v2_mdp.ReferenceResidualActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=source_action.scale,
        offset=0.0,
        preserve_order=False,
        use_default_offset=False,
        command_name="twist",
    )
    actor_source = env.observations["actor"].terms
    critic_source = env.observations["critic"].terms
    physical_names = (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
    )
    privileged_names = (
        "foot_height",
        "foot_air_time",
        "foot_contact",
        "foot_contact_forces",
    )
    critic_source["foot_contact_forces"] = ObservationTermCfg(
        func=walking_v2_mdp.raw_foot_contact_forces,
        params={"sensor_name": "feet_ground_contact"},
    )
    reference_terms = {
        "phase_sin": ObservationTermCfg(
            func=walking_v2_mdp.phase_sin, params={"command_name": "twist"}
        ),
        "phase_cos": ObservationTermCfg(
            func=walking_v2_mdp.phase_cos, params={"command_name": "twist"}
        ),
        "walk_blend": ObservationTermCfg(
            func=walking_v2_mdp.walk_blend, params={"command_name": "twist"}
        ),
        "reference_offset": ObservationTermCfg(
            func=walking_v2_mdp.reference_offset, params={"command_name": "twist"}
        ),
        "reference_velocity": ObservationTermCfg(
            func=walking_v2_mdp.reference_velocity, params={"command_name": "twist"}
        ),
    }
    env.observations["actor"].terms = {
        **{name: actor_source[name] for name in physical_names},
        **reference_terms,
    }
    env.observations["critic"].terms = {
        **{name: critic_source[name] for name in physical_names},
        **{name: critic_source[name] for name in privileged_names},
        **reference_terms,
    }
    forbidden_ground = ContactSensorCfg(
        name="forbidden_ground_contact_v2",
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
    env.curriculum = {}
    env.rewards = {
        "walking_v2_rate": RewardTermCfg(
            func=walking_v2_mdp.walking_v2_rate,
            weight=1.0,
            params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
        ),
        "true_fall_event": RewardTermCfg(
            func=walking_v2_mdp.true_fall_event,
            weight=-2.0 / (env.sim.mujoco.timestep * env.decimation),
        ),
    }
    env.terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nonfinite_state": TerminationTermCfg(func=walking_v2_mdp.nonfinite_state),
        "forbidden_ground_contact": TerminationTermCfg(
            func=mdp.illegal_contact,
            params={
                "sensor_name": "forbidden_ground_contact_v2",
                "force_threshold": 20.0,
            },
        ),
        "fell_over": TerminationTermCfg(func=walking_v2_mdp.fell_over),
        "reference_deviation": TerminationTermCfg(
            func=walking_v2_mdp.reference_deviation,
            params={"command_name": "twist", "maximum_rms_rad": 1.0},
        ),
    }
    runner = unitree_g1_ppo_runner_cfg()
    runner.actor.hidden_dims = (512, 256, 128)
    runner.critic.hidden_dims = (512, 256, 128)
    runner.actor.distribution_cfg = {
        **(runner.actor.distribution_cfg or {}),
        "init_std": 0.2,
    }
    register_mjlab_task(TASK_ID, env, env, runner)
