from __future__ import annotations

import pytest

from g1_mjlab.config import load_config
from g1_mjlab.environment import build_train_config
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


def test_walking_descriptor_fails_closed_until_motion_contract_exists() -> None:
    task = get_task(WALKING_TASK_ID)

    assert task.config_directory == "walking-v1"
    assert task.experiment_name == "g1_walking"
    with pytest.raises(TaskCapabilityError, match="motion reference"):
        task.require(TaskCapability.TRAIN)


def test_unknown_task_is_rejected_without_loading_the_simulator() -> None:
    with pytest.raises(UnknownTaskError, match="Unknown task"):
        get_task("G1-Unknown-v1")


def test_unavailable_training_fails_before_simulator_import(tmp_path) -> None:
    source = tmp_path / "walking.json"
    source.write_text(
        """{
  "schema_version": 1,
  "task_id": "G1-Walking-Flat-v1",
  "run_name": "walking-test",
  "seed": 42,
  "num_envs": 16,
  "max_iterations": 2,
  "rollout_steps": 24,
  "save_interval": 1,
  "physics_dt": 0.005,
  "decimation": 4,
  "episode_length_s": 20.0,
  "action_clip": 1.0,
  "device": "cuda:0",
  "logger": "tensorboard",
  "video": false
}
""",
        encoding="utf-8",
    )

    with pytest.raises(TaskCapabilityError, match="motion reference"):
        build_train_config(load_config(source), tmp_path / "logs")
