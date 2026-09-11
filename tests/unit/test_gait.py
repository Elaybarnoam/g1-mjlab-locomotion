from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from g1_mjlab.motion.gait import (
    CommandProfile,
    GaitState,
    WalkingInitialState,
    load_command_profile,
    load_command_schedule,
    step_gait_numpy,
)


@pytest.fixture
def profile() -> CommandProfile:
    return CommandProfile(
        reference_speed_m_s=1.0,
        cycle_duration_s=1.0,
        acceleration_m_s2=0.5,
        deceleration_m_s2=1.0,
        blend_rate_s=2.0,
        stand_threshold_m_s=0.05,
        walk_threshold_m_s=0.15,
    )


def test_command_profile_accepts_supported_forward_range_only(profile: CommandProfile) -> None:
    profile.validate_requested(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    profile.validate_requested(np.array([[0.5, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="supported forward range"):
        profile.validate_requested(np.array([[1.1, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="lateral and yaw"):
        profile.validate_requested(np.array([[1.0, 0.1, 0.0]]))


def test_gait_step_rate_limits_command_blend_and_phase(profile: CommandProfile) -> None:
    state = GaitState.zeros(1)

    state = step_gait_numpy(state, np.array([[1.0, 0.0, 0.0]]), profile, dt=0.1)

    assert state.applied_command[0, 0] == pytest.approx(0.05)
    assert state.blend[0] == pytest.approx(0.0)
    assert state.phase[0] == pytest.approx(0.005)

    for _ in range(3):
        state = step_gait_numpy(state, np.array([[1.0, 0.0, 0.0]]), profile, dt=0.1)
    assert state.applied_command[0, 0] == pytest.approx(0.2)
    assert state.blend[0] == pytest.approx(0.4)
    assert state.phase[0] == pytest.approx(0.05)


def test_stop_preserves_phase_and_blends_to_standing(profile: CommandProfile) -> None:
    state = GaitState(
        applied_command=np.array([[1.0, 0.0, 0.0]]),
        phase=np.array([0.95]),
        blend=np.array([1.0]),
        walking=np.array([True]),
        reference_distance_m=np.array([2.0]),
    )
    for _ in range(20):
        state = step_gait_numpy(state, np.zeros((1, 3)), profile, dt=0.1)

    assert state.applied_command[0, 0] == pytest.approx(0.0)
    assert state.blend[0] == pytest.approx(0.0)
    assert 0.0 <= state.phase[0] < 1.0
    frozen_phase = state.phase.copy()
    frozen_distance = state.reference_distance_m.copy()
    state = step_gait_numpy(state, np.zeros((1, 3)), profile, dt=0.1)
    np.testing.assert_allclose(state.phase, frozen_phase)
    np.testing.assert_allclose(state.reference_distance_m, frozen_distance)


def test_batched_timeline_matches_independent_scalar_states(profile: CommandProfile) -> None:
    requested = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    ]
    batch = GaitState.zeros(2)
    scalars = [GaitState.zeros(1), GaitState.zeros(1)]
    for command in requested:
        batch = step_gait_numpy(batch, command, profile, dt=0.02)
        scalars = [
            step_gait_numpy(state, command[index : index + 1], profile, dt=0.02)
            for index, state in enumerate(scalars)
        ]

    np.testing.assert_allclose(
        batch.applied_command, np.vstack([s.applied_command for s in scalars])
    )
    np.testing.assert_allclose(batch.phase, np.concatenate([s.phase for s in scalars]))
    np.testing.assert_allclose(batch.blend, np.concatenate([s.blend for s in scalars]))
    np.testing.assert_allclose(
        batch.reference_distance_m, np.concatenate([s.reference_distance_m for s in scalars])
    )


def test_phase_wraps_without_resetting_accumulated_reference_distance(
    profile: CommandProfile,
) -> None:
    state = GaitState.zeros(1)
    wraps = 0
    previous_phase = 0.0
    for _ in range(700):
        state = step_gait_numpy(state, [[1.0, 0.0, 0.0]], profile, dt=0.02)
        wraps += int(state.phase[0] < previous_phase)
        previous_phase = float(state.phase[0])

    assert wraps >= 10
    assert state.reference_distance_m[0] > 10.0


def test_initial_state_requires_full_floating_base_velocity() -> None:
    state = WalkingInitialState(
        root_position_w=(0.0, 0.0, 0.75),
        root_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        root_linear_velocity_w=(1.0, 0.0, 0.0),
        root_angular_velocity_w=(0.0, 0.0, 0.0),
        joint_position=tuple(0.0 for _ in range(29)),
        joint_velocity=tuple(0.0 for _ in range(29)),
        phase=0.25,
        blend=1.0,
    )

    state.validate()
    with pytest.raises(ValueError, match="root_linear_velocity_w"):
        WalkingInitialState(**{**state.to_dict(), "root_linear_velocity_w": (0.0, 0.0)}).validate()


def test_public_command_profile_and_schedule_are_reference_bound() -> None:
    root = Path(__file__).parents[2] / "configs" / "walking-v1"

    loaded_profile = load_command_profile(root / "commands.json")
    schedule = load_command_schedule(root / "development-schedule.json", loaded_profile)

    assert loaded_profile.reference_speed_m_s == pytest.approx(1.16381159304071)
    assert [segment.forward_speed_m_s for segment in schedule.segments] == [
        0.0,
        loaded_profile.reference_speed_m_s,
        0.0,
    ]
    assert schedule.reference_id == "nvidia-soma-g1-neutral-walk-a057-cycle-v1"
