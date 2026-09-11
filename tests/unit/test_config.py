from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_mjlab.config import (
    load_config,
    load_ppo_profile,
    load_reward_profile,
    load_walking_training_profile,
)


def _data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "task_id": "Mjlab-Velocity-Flat-Unitree-G1",
        "run_name": "test-run",
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
        "video": False,
    }


def _write(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_config_derives_batch_and_stable_hash(tmp_path: Path) -> None:
    path = _write(tmp_path, _data())
    first = load_config(path)
    second = load_config(path)
    assert first.control_dt == pytest.approx(0.02)
    assert first.transitions_per_update == 384
    assert first.sha256 == second.sha256


def test_config_rejects_unknown_key(tmp_path: Path) -> None:
    data = _data()
    data["mystery"] = 1
    with pytest.raises(ValueError, match="unknown"):
        load_config(_write(tmp_path, data))


def test_config_rejects_invalid_minibatch_size(tmp_path: Path) -> None:
    data = _data()
    data["num_envs"] = 3
    data["rollout_steps"] = 3
    with pytest.raises(ValueError, match="divisible"):
        load_config(_write(tmp_path, data))


def test_reward_profile_preserves_per_event_penalty(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "termination-probe",
                "alive_reward_rate": 0.0,
                "termination_penalty": -5.0,
            }
        ),
        encoding="utf-8",
    )
    profile = load_reward_profile(path)
    assert profile.termination_weight(0.02) == -250.0
    assert profile.termination_weight(0.02) * 0.02 == -5.0


def test_ppo_profile_is_strict_and_positive(tmp_path: Path) -> None:
    path = tmp_path / "ppo.json"
    path.write_text(
        json.dumps({"schema_version": 1, "name": "low-std", "initial_action_std": 0.2}),
        encoding="utf-8",
    )
    assert load_ppo_profile(path).initial_action_std == 0.2
    assert load_ppo_profile(path).entropy_coef is None
    assert load_ppo_profile(path).reset_action_std_on_transfer is True


def test_ppo_profile_can_preserve_transferred_action_std(tmp_path: Path) -> None:
    path = tmp_path / "ppo.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "preserve-std",
                "initial_action_std": 0.2,
                "entropy_coef": 0.0,
                "reset_action_std_on_transfer": False,
            }
        ),
        encoding="utf-8",
    )

    profile = load_ppo_profile(path)

    assert profile.reset_action_std_on_transfer is False


def test_walking_training_profile_is_strict_and_hashable(tmp_path: Path) -> None:
    path = tmp_path / "walking.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "reference-only",
                "standing_fraction": 0.0,
                "reference_initialization": True,
                "randomize_phase": True,
                "forward_progress_weight": 1.0,
                "reference_foot_position_std_m": 0.3,
            }
        ),
        encoding="utf-8",
    )
    profile = load_walking_training_profile(path)
    assert profile.standing_fraction == 0
    assert profile.forward_progress_weight == 1.0
    assert profile.reference_foot_position_std_m == 0.3
    assert len(profile.sha256) == 64
