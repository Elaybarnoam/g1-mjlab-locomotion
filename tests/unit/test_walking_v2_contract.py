from __future__ import annotations

import pytest

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
