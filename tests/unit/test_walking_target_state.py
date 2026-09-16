from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from g1_mjlab.motion.reference_bank import ReferenceBank, build_reference_bank
from g1_mjlab.motion.walking_target_state import (
    WalkingTargetProfile,
    WalkingTargetState,
    predict_walking_target_numpy,
    predict_walking_target_torch,
    reset_walking_target_numpy,
    reset_walking_target_torch,
    smoothstep_walk_blend_numpy,
)


def _bank(tmp_path: Path) -> ReferenceBank:
    frames = 17
    phase = np.linspace(0.0, 2.0 * np.pi, frames)
    base = np.sin(phase)[:, None] * np.linspace(0.02, 0.08, 29)
    sources = tuple(tmp_path / f"source-{index}.npz" for index in range(3))
    for index, source in enumerate(sources):
        joint = base * (1.0 + 0.1 * index)
        np.savez_compressed(
            source,
            joint_pos=joint,
            joint_vel=np.gradient(joint, 0.02, axis=0),
            foot_contact=np.ones((frames, 2), dtype=np.uint8),
            fps=np.asarray([50.0]),
        )
    metadata = tmp_path / "bank.json"
    build_reference_bank(
        sources,
        metadata,
        reference_id="walking-target-test",
        source_license="test-only",
        model_sha256="a" * 64,
        controller_sha256="b" * 64,
        joint_names=tuple(f"joint-{index}" for index in range(29)),
        nominal_joint_position=np.zeros(29),
        local_foot_position=tuple(np.zeros((frames, 2, 3)) for _ in range(3)),
    )
    return ReferenceBank.load(metadata)


def test_smoothstep_blend_is_c1_and_saturates() -> None:
    speed = np.asarray([0.0, 0.2, 0.4, 0.8])

    blend = smoothstep_walk_blend_numpy(speed)

    np.testing.assert_allclose(blend, [0.0, 0.5, 1.0, 1.0])
    epsilon = 1e-6
    start_slope = (
        smoothstep_walk_blend_numpy([epsilon])[0] - smoothstep_walk_blend_numpy([0.0])[0]
    ) / epsilon
    end_slope = (
        smoothstep_walk_blend_numpy([0.4])[0] - smoothstep_walk_blend_numpy([0.4 - epsilon])[0]
    ) / epsilon
    assert start_slope < 1e-4
    assert end_slope < 1e-4


def test_prediction_is_interval_end_target_and_does_not_mutate_state(
    tmp_path: Path,
) -> None:
    bank = _bank(tmp_path)
    state = WalkingTargetState.zeros(2)
    requested = np.asarray([[0.8, 0.0, 0.0], [0.4, 0.0, 0.0]])

    transition = predict_walking_target_numpy(state, requested, bank)

    np.testing.assert_allclose(state.applied_command, 0.0)
    np.testing.assert_allclose(transition.next_state.applied_command[:, 0], 0.012)
    recomposed = bank.compose_targets_numpy(
        transition.next_state.phase,
        transition.next_state.applied_command[:, 0],
        blend=transition.next_state.blend,
        blend_rate_s=transition.blend_rate_s,
        speed_rate_m_s2=transition.command_rate_m_s2,
    )
    np.testing.assert_allclose(transition.targets.joint_position, recomposed.joint_position)
    np.testing.assert_allclose(transition.targets.joint_velocity, recomposed.joint_velocity)


def test_numpy_and_torch_transition_are_equivalent(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    bank = _bank(tmp_path)
    state = WalkingTargetState(
        applied_command=np.asarray([[0.3, 0.0, 0.0], [0.7, 0.0, 0.0]]),
        phase=np.asarray([0.99, 0.25]),
        blend=np.asarray([0.4, 1.0]),
    )
    requested = np.asarray([[0.8, 0.0, 0.0], [0.0, 0.0, 0.0]])

    numpy_result = predict_walking_target_numpy(state, requested, bank)
    torch_result = predict_walking_target_torch(
        torch.tensor(state.applied_command),
        torch.tensor(state.phase),
        torch.tensor(state.blend),
        torch.tensor(requested),
        bank,
    )

    np.testing.assert_allclose(
        torch_result.next_applied_command.numpy(),
        numpy_result.next_state.applied_command,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        torch_result.next_phase.numpy(), numpy_result.next_state.phase, atol=1e-12
    )
    np.testing.assert_allclose(
        torch_result.next_blend.numpy(), numpy_result.next_state.blend, atol=1e-12
    )
    np.testing.assert_allclose(
        torch_result.target_joint_position.numpy(),
        numpy_result.targets.joint_position,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        torch_result.target_joint_velocity.numpy(),
        numpy_result.targets.joint_velocity,
        atol=1e-12,
    )


def test_partial_resets_are_isolated_in_numpy_and_torch() -> None:
    torch = pytest.importorskip("torch")
    state = WalkingTargetState(
        applied_command=np.asarray([[0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0]]),
        phase=np.asarray([0.1, 0.2, 0.3]),
        blend=np.asarray([0.2, 0.4, 0.6]),
    )

    numpy_result = reset_walking_target_numpy(state, [1], phase=[0.75])
    torch_result = reset_walking_target_torch(
        torch.tensor(state.applied_command),
        torch.tensor(state.phase),
        torch.tensor(state.blend),
        torch.tensor([1]),
        reset_phase=torch.tensor([0.75]),
    )

    np.testing.assert_allclose(numpy_result.applied_command[[0, 2]], state.applied_command[[0, 2]])
    np.testing.assert_allclose(numpy_result.phase, [0.1, 0.75, 0.3])
    np.testing.assert_allclose(numpy_result.blend, [0.2, 0.0, 0.6])
    np.testing.assert_allclose(torch_result[0].numpy(), numpy_result.applied_command)
    np.testing.assert_allclose(torch_result[1].numpy(), numpy_result.phase)
    np.testing.assert_allclose(torch_result[2].numpy(), numpy_result.blend)


def test_stop_freezes_phase_after_rate_limited_transition(tmp_path: Path) -> None:
    bank = _bank(tmp_path)
    state = WalkingTargetState(
        applied_command=np.asarray([[0.4, 0.0, 0.0]]),
        phase=np.asarray([0.95]),
        blend=np.asarray([1.0]),
    )
    for _ in range(100):
        state = predict_walking_target_numpy(state, [[0.0, 0.0, 0.0]], bank).next_state

    assert state.applied_command[0, 0] == pytest.approx(0.0)
    assert state.blend[0] == pytest.approx(0.0)
    frozen_phase = state.phase.copy()
    state = predict_walking_target_numpy(state, [[0.0, 0.0, 0.0]], bank).next_state
    np.testing.assert_allclose(state.phase, frozen_phase)


@pytest.mark.parametrize(
    "requested",
    (
        [[-0.1, 0.0, 0.0]],
        [[0.81, 0.0, 0.0]],
        [[0.4, 0.1, 0.0]],
    ),
)
def test_invalid_commands_fail_before_transition(
    tmp_path: Path, requested: list[list[float]]
) -> None:
    with pytest.raises(ValueError):
        predict_walking_target_numpy(WalkingTargetState.zeros(1), requested, _bank(tmp_path))


def test_profile_rejects_invalid_rate_contract() -> None:
    with pytest.raises(ValueError):
        WalkingTargetProfile(full_walk_blend_speed_m_s=0.9).validate()
