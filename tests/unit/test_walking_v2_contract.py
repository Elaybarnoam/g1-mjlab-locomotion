from __future__ import annotations

import pytest

from g1_mjlab.config import WalkingV2CurriculumProfile, load_walking_v2_curriculum_profile
from g1_mjlab.contracts import validate_contract
from g1_mjlab.tasks.registry import TaskCapability, get_task
from g1_mjlab.tasks.walking_v2 import (
    TASK_ID,
    configure_environment,
    validate_walking_v2_checkpoint_metadata,
    walking_v2_policy_contract,
)


class _Value:
    pass


class _Environment:
    def __init__(self) -> None:
        actor = _Value()
        actor.enable_corruption = True
        command = _Value()
        command.randomize_phase = False
        command.reference_initialization = False
        self.observations = {"actor": actor}
        self.commands = {"twist": command}
        self.actions = {"joint_pos": _Value()}
        walking_reward = _Value()
        walking_reward.params = {"command_name": "twist", "sensor_name": "feet"}
        fall_reward = _Value()
        fall_reward.weight = -100.0
        self.rewards = {"walking_v2_rate": walking_reward, "true_fall_event": fall_reward}
        self.decimation = 4
        self.sim = _Value()
        self.sim.mujoco = _Value()
        self.sim.mujoco.timestep = 0.005
        self.terminations = {"reference_deviation": object(), "fell_over": object()}
        self.events = {
            "reset_base": object(),
            "reset_robot_joints": object(),
            "push_robot": object(),
            "foot_friction": object(),
            "encoder_bias": object(),
            "base_com": object(),
        }


def test_walking_v2_contract_freezes_actor_critic_and_action_layouts() -> None:
    contract = walking_v2_policy_contract()

    validate_contract(contract)
    assert sum(field.size for field in contract.actor_fields) == 160
    assert sum(field.size for field in contract.critic_fields) == 172
    assert len(contract.action_names) == 29
    assert [(field.name, field.offset, field.size) for field in contract.actor_fields] == [
        ("base_lin_vel", 0, 3),
        ("base_ang_vel", 3, 3),
        ("projected_gravity", 6, 3),
        ("joint_pos", 9, 29),
        ("joint_vel", 38, 29),
        ("previous_action", 67, 29),
        ("command", 96, 3),
        ("phase_sin", 99, 1),
        ("phase_cos", 100, 1),
        ("walk_blend", 101, 1),
        ("reference_offset", 102, 29),
        ("reference_velocity", 131, 29),
    ]
    assert [(field.name, field.offset, field.size) for field in contract.critic_fields[-7:]] == [
        ("foot_contact", 103, 2),
        ("foot_contact_forces", 105, 6),
        ("phase_sin", 111, 1),
        ("phase_cos", 112, 1),
        ("walk_blend", 113, 1),
        ("reference_offset", 114, 29),
        ("reference_velocity", 143, 29),
    ]
    assert contract.task_id == TASK_ID
    assert contract.layout_id == "g1-walking-reference-actor-v2"
    assert "reference residuals" in contract.action_semantics


def test_walking_v2_is_separate_and_trainable_in_cpu_registry() -> None:
    task = get_task(TASK_ID)

    assert task.actor_size == 160
    assert task.critic_size == 172
    assert task.supports(TaskCapability.TRAIN)
    assert task.config_directory == "walking-v2"


def test_walking_v2_checkpoint_metadata_rejects_old_layout() -> None:
    contract = walking_v2_policy_contract()
    valid = {
        "task_id": TASK_ID,
        "layout_id": contract.layout_id,
        "actor_observation_size": 160,
        "critic_observation_size": 172,
        "action_size": 29,
        "policy_contract_sha256": contract.sha256,
    }

    validate_walking_v2_checkpoint_metadata(valid)
    with pytest.raises(ValueError, match="incompatible"):
        validate_walking_v2_checkpoint_metadata({**valid, "layout_id": "g1-walking-actor-v1"})


def test_acquisition_mode_excludes_robustness_randomization() -> None:
    environment = _Environment()

    configure_environment(environment)

    assert environment.observations["actor"].enable_corruption is False
    assert set(environment.events) == {"reset_base", "reset_robot_joints"}
    assert environment.commands["twist"].randomize_phase is True
    assert environment.commands["twist"].reference_initialization is True


def test_walking_v2_curriculum_profile_is_strict_and_hashable(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        """{
  "schema_version": 2,
  "name": "transitions-v2",
  "stage": "transitions",
  "standing_fraction": 0.4,
  "forward_speed_range_m_s": [0.4, 0.8],
  "resampling_time_range_s": [1.5, 4.0],
  "reference_initialization": false,
  "observation_noise": false,
  "startup_domain_randomization": false,
  "push_disturbance": false,
  "terminate_reference_deviation": false,
  "gait_contact_weight": 1.0,
  "gait_foot_trajectory_weight": 2.0,
  "gait_alternation_event_weight": 1.0,
  "gait_contact_chatter_weight": 0.5,
  "gait_forward_progress_weight": 1.0,
  "gait_action_rate_weight": 0.1,
  "residual_action_filter_alpha": 0.25,
  "fall_penalty": -10.0
}\n""",
        encoding="utf-8",
    )

    profile = load_walking_v2_curriculum_profile(path)

    assert profile.stage == "transitions"
    assert profile.forward_speed_range_m_s == (0.4, 0.8)
    assert profile.resampling_time_range_s == (1.5, 4.0)
    assert len(profile.sha256) == 64


def test_walking_v2_curriculum_profile_controls_only_declared_stage_features() -> None:
    environment = _Environment()
    profile = WalkingV2CurriculumProfile(
        schema_version=2,
        name="robustness-v2",
        stage="robustness",
        standing_fraction=0.4,
        forward_speed_range_m_s=(0.4, 0.8),
        resampling_time_range_s=(1.5, 4.0),
        reference_initialization=False,
        observation_noise=True,
        startup_domain_randomization=True,
        push_disturbance=True,
        terminate_reference_deviation=False,
        gait_contact_weight=1.0,
        gait_foot_trajectory_weight=2.0,
        gait_alternation_event_weight=1.0,
        gait_contact_chatter_weight=0.5,
        gait_forward_progress_weight=1.0,
        gait_action_rate_weight=0.1,
        residual_action_filter_alpha=0.25,
        fall_penalty=-10.0,
    )

    configure_environment(environment, curriculum_profile=profile)

    assert environment.observations["actor"].enable_corruption is True
    assert set(environment.events) == {
        "base_com",
        "encoder_bias",
        "foot_friction",
        "push_robot",
        "reset_base",
        "reset_robot_joints",
    }
    command = environment.commands["twist"]
    assert command.standing_fraction == 0.4
    assert command.forward_speed_range_m_s == (0.4, 0.8)
    assert command.resampling_time_range == (1.5, 4.0)
    assert command.reference_initialization is False
    assert "reference_deviation" not in environment.terminations
    assert environment.rewards["walking_v2_rate"].params["gait_contact_weight"] == 1.0
    assert environment.rewards["walking_v2_rate"].params["gait_foot_trajectory_weight"] == 2.0
    assert environment.rewards["walking_v2_rate"].params["gait_alternation_event_weight"] == 1.0
    assert environment.rewards["walking_v2_rate"].params["gait_contact_chatter_weight"] == 0.5
    assert environment.rewards["walking_v2_rate"].params["gait_forward_progress_weight"] == 1.0
    assert environment.rewards["walking_v2_rate"].params["gait_action_rate_weight"] == 0.1
    assert environment.actions["joint_pos"].filter_alpha == 0.25
    assert environment.rewards["true_fall_event"].weight == -500.0
