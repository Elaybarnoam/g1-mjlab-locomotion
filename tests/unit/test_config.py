from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_mjlab.config import (
    STAGE19_EVENT_REWARDS,
    STAGE19_REWARD_NAMES,
    STAGE19_REWARD_PARAMETERS,
    load_config,
    load_ppo_profile,
    load_reward_profile,
    load_stage19_reward_profile,
    load_walking_training_profile,
)


def _stage19_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "arm-a-v1",
        "terms": [
            {
                "name": name,
                "enabled": False,
                "weight": 0.0,
                "integration_kind": "per_event" if name in STAGE19_EVENT_REWARDS else "rate",
                "parameters": {parameter: 1.0 for parameter in STAGE19_REWARD_PARAMETERS[name]},
                "physical_unit": "1",
                "mask_id": "always",
                "formula_id": name,
            }
            for name in sorted(STAGE19_REWARD_NAMES)
        ],
    }


def test_stage19_reward_profile_is_complete_hashed_and_event_scaled(tmp_path: Path) -> None:
    payload = _stage19_payload()
    path = _write(tmp_path, payload)
    profile = load_stage19_reward_profile(path)
    assert len(profile.terms) == len(STAGE19_REWARD_NAMES)
    assert len(profile.sha256) == 64
    termination = profile.by_name()["termination"]
    assert termination.manager_weight(0.02) == 0


@pytest.mark.parametrize(
    "mutation", ["missing", "unknown", "unsafe_formula", "bad_kind", "bad_parameter"]
)
def test_stage19_reward_profile_fails_closed(tmp_path: Path, mutation: str) -> None:
    payload = _stage19_payload()
    terms = payload["terms"]
    assert isinstance(terms, list)
    if mutation == "missing":
        terms.pop()
    elif mutation == "unknown":
        terms[0]["unknown"] = 1
    elif mutation == "unsafe_formula":
        terms[0]["formula_id"] = "package.module:function"
    elif mutation == "bad_kind":
        target = next(term for term in terms if term["name"] == "termination")
        target["integration_kind"] = "rate"
    else:
        target = next(term for term in terms if term["name"] == "physical_stance_slip")
        target["parameters"]["unreviewed_scale"] = 1.0
    with pytest.raises(ValueError):
        load_stage19_reward_profile(_write(tmp_path, payload))


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
                "host_semantics_version": 2,
                "domain_randomization": False,
                "observation_noise": True,
                "reference_ground_offset_m": 0.03,
            }
        ),
        encoding="utf-8",
    )
    profile = load_walking_training_profile(path)
    assert profile.standing_fraction == 0
    assert profile.forward_progress_weight == 1.0
    assert profile.reference_foot_position_std_m == 0.3
    assert profile.host_semantics_version == 2
    assert profile.domain_randomization is False
    assert profile.observation_noise is True
    assert profile.reference_ground_offset_m == 0.03
    assert len(profile.sha256) == 64


def test_walking_bootstrap_profile_is_explicit_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "bootstrap.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "locomotion-bootstrap",
                "objective": "locomotion_bootstrap",
                "standing_fraction": 0.1,
                "reference_initialization": False,
                "randomize_phase": True,
            }
        ),
        encoding="utf-8",
    )

    profile = load_walking_training_profile(path)

    assert profile.objective == "locomotion_bootstrap"

    path.write_text(path.read_text().replace("locomotion_bootstrap", "unknown"))
    with pytest.raises(ValueError, match="objective"):
        load_walking_training_profile(path)
