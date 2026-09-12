from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from g1_mjlab.inference.walking import (
    WalkingPolicySession,
    WalkingSensorState,
    compose_walking_actor_observation,
    walking_joint_targets,
)
from g1_mjlab.motion.gait import CommandProfile, GaitState, step_gait_numpy, step_gait_torch


def _contract() -> dict[str, object]:
    sizes = (
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("joint_pos", 29),
        ("joint_vel", 29),
        ("actions", 29),
        ("command", 3),
        ("phase_sin", 1),
        ("phase_cos", 1),
        ("walk_blend", 1),
    )
    offset = 0
    fields = []
    for name, size in sizes:
        fields.append(
            {
                "name": name,
                "offset": offset,
                "size": size,
                "resolved_term": {
                    "scale": None,
                    "clip": None,
                    "history_length": 0,
                    "delay_max_lag": 0,
                },
            }
        )
        offset += size
    return {
        "schema_version": 2,
        "task_id": "G1-Walking-Flat-v1",
        "layout_id": "g1-walking-actor-v1",
        "actor_fields": fields,
        "nominal_joint_position": [0.1] * 29,
        "encoder_bias": [0.01] * 29,
        "action_scale": [0.25] * 29,
        "action_clip": None,
        "control_dt": 0.02,
    }


def _sensor() -> WalkingSensorState:
    return WalkingSensorState(
        base_linear_velocity=np.asarray([1.0, 2.0, 3.0]),
        base_angular_velocity=np.asarray([4.0, 5.0, 6.0]),
        projected_gravity=np.asarray([0.0, 0.0, -1.0]),
        joint_position=np.full(29, 0.2),
        joint_velocity=np.full(29, 0.3),
    )


def test_walking_actor_vector_is_exact_102_field_layout() -> None:
    gait = GaitState.zeros(1)
    gait.phase[0] = 0.25
    gait.blend[0] = 0.4
    gait.applied_command[0, 0] = 0.2
    previous = np.full(29, -0.5)

    result = compose_walking_actor_observation(_contract(), _sensor(), previous, gait)

    assert result.shape == (102,)
    np.testing.assert_allclose(result[:9], [1, 2, 3, 4, 5, 6, 0, 0, -1])
    np.testing.assert_allclose(result[9:38], 0.11)
    np.testing.assert_allclose(result[38:67], 0.3)
    np.testing.assert_allclose(result[67:96], -0.5)
    np.testing.assert_allclose(result[96:99], [0.2, 0.0, 0.0])
    np.testing.assert_allclose(result[99:], [1.0, 0.0, 0.4], atol=1e-7)


class _FakeOnnx:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="observation", shape=[None, 102], type="tensor(float)")]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(shape=[None, 29], type="tensor(float)")]

    def run(self, _: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.observations.append(feeds["observation"].copy())
        return [np.full((1, 29), len(self.observations), dtype=np.float32)]


def test_policy_step_observes_old_host_state_then_advances_for_next_sample() -> None:
    runtime = _FakeOnnx()
    profile = CommandProfile(1.16381159304071, 1.06, 0.6, 0.8, 1.0, 0.05, 0.15)
    policy = WalkingPolicySession.from_components(
        _contract(),
        {"status": "unqualified_development", "development_command_domain_m_s": [0.0, 0.6]},
        profile,
        runtime,
        allow_unqualified=True,
    )
    policy.reset(_sensor(), phase=0.25)

    first = policy.step(0.6, _sensor())
    second = policy.step(0.6, _sensor())

    np.testing.assert_allclose(runtime.observations[0][0, 96:99], 0.0)
    np.testing.assert_allclose(runtime.observations[0][0, 99:], [1.0, 0.0, 0.0], atol=1e-7)
    assert runtime.observations[1][0, 96] == pytest.approx(0.012)
    np.testing.assert_allclose(runtime.observations[1][0, 67:96], 1.0)
    assert first.command.next_applied_forward_speed_m_s == pytest.approx(0.012)
    assert second.action.shape == (29,)


def test_unqualified_policy_is_fail_closed_and_command_domain_is_enforced() -> None:
    profile = CommandProfile(1.16381159304071, 1.06, 0.6, 0.8, 1.0, 0.05, 0.15)
    with pytest.raises(ValueError, match="unqualified"):
        WalkingPolicySession.from_components(
            _contract(),
            {"status": "unqualified_development", "development_command_domain_m_s": [0.0, 0.6]},
            profile,
            _FakeOnnx(),
        )
    policy = WalkingPolicySession.from_components(
        _contract(),
        {"status": "unqualified_development", "development_command_domain_m_s": [0.0, 0.6]},
        profile,
        _FakeOnnx(),
        allow_unqualified=True,
    )
    policy.reset(_sensor())
    with pytest.raises(ValueError, match="domain"):
        policy.step(0.8, _sensor())


def test_action_mapping_applies_clip_scale_nominal_and_bias_once() -> None:
    contract = _contract()
    contract["action_clip"] = 1.0
    action = np.linspace(-2.0, 2.0, 29)

    targets = walking_joint_targets(contract, action)

    expected = 0.1 + 0.25 * np.clip(action, -1.0, 1.0) - 0.01
    np.testing.assert_allclose(targets, expected, atol=1e-7)


def test_numpy_and_torch_gait_match_over_random_dt_and_more_than_ten_cycles() -> None:
    import torch

    rng = np.random.default_rng(90409)
    profile = CommandProfile(1.16381159304071, 1.06, 0.6, 0.8, 1.0, 0.05, 0.15)
    numpy_state = GaitState.zeros(3)
    torch_state = (
        torch.zeros((3, 3), dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.bool),
        torch.zeros(3, dtype=torch.float64),
    )
    elapsed = 0.0
    while elapsed < 14.0:
        dt = float(rng.uniform(0.004, 0.024))
        speeds = rng.choice([0.0, 0.4, 0.6, 1.0], size=3)
        command = np.column_stack((speeds, np.zeros((3, 2))))
        numpy_state = step_gait_numpy(numpy_state, command, profile, dt=dt)
        torch_state = step_gait_torch(
            *torch_state,
            torch.as_tensor(command),
            profile,
            dt=dt,
        )
        elapsed += dt
    np.testing.assert_allclose(torch_state[0].numpy(), numpy_state.applied_command, atol=1e-6)
    np.testing.assert_allclose(torch_state[1].numpy(), numpy_state.phase, atol=1e-6)
    np.testing.assert_allclose(torch_state[2].numpy(), numpy_state.blend, atol=1e-6)
    np.testing.assert_array_equal(torch_state[3].numpy(), numpy_state.walking)
    np.testing.assert_allclose(torch_state[4].numpy(), numpy_state.reference_distance_m, atol=1e-6)
