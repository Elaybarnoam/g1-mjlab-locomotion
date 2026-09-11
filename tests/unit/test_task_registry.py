from __future__ import annotations

import pytest

from g1_mjlab.tasks import (
    STANDING_TASK_ID,
    WALKING_TASK_ID,
    TaskCapability,
    TaskCapabilityError,
    UnknownTaskError,
    get_task,
)


def test_standing_descriptor_declares_stable_runtime_identity() -> None:
    task = get_task(STANDING_TASK_ID)

    assert task.config_directory == "standing-v1"
    assert task.experiment_name == "g1_standing"
    assert task.supports(TaskCapability.TRAIN)
    assert task.supports(TaskCapability.STANDING_EVALUATION)
    assert task.supports(TaskCapability.NATIVE_INFERENCE)


def test_walking_descriptor_exposes_training_but_not_unimplemented_workflows() -> None:
    task = get_task(WALKING_TASK_ID)

    assert task.config_directory == "walking-v1"
    assert task.experiment_name == "g1_walking"
    assert task.actor_size == 102
    assert task.critic_size == 114
    assert task.layout_id == "g1-walking-actor-v1"
    task.require(TaskCapability.TRAIN)
    task.require(TaskCapability.WALKING_EVALUATION)
    with pytest.raises(TaskCapabilityError, match="native inference"):
        task.require(TaskCapability.NATIVE_INFERENCE)


def test_unknown_task_is_rejected_without_loading_the_simulator() -> None:
    with pytest.raises(UnknownTaskError, match="Unknown task"):
        get_task("G1-Unknown-v1")


def test_unavailable_walking_evaluation_fails_without_simulator_import() -> None:
    with pytest.raises(TaskCapabilityError, match="native inference"):
        get_task(WALKING_TASK_ID).require(TaskCapability.NATIVE_INFERENCE)
