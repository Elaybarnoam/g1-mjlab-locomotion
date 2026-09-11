from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from g1_mjlab.motion.gait import (  # noqa: E402
    CommandProfile,
    GaitState,
    step_gait_numpy,
    step_gait_torch,
)


@pytest.mark.simulator
def test_torch_gait_step_matches_numpy_without_host_state_updates() -> None:
    profile = CommandProfile(1.0, 1.0, 0.5, 1.0, 2.0, 0.05, 0.15)
    numpy_state = GaitState.zeros(3)
    applied = torch.zeros((3, 3), dtype=torch.float64)
    phase = torch.zeros(3, dtype=torch.float64)
    blend = torch.zeros(3, dtype=torch.float64)
    walking = torch.zeros(3, dtype=torch.bool)
    reference_distance = torch.zeros(3, dtype=torch.float64)
    sequence = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
    for values in sequence:
        command = np.asarray(values)
        numpy_state = step_gait_numpy(numpy_state, command, profile, dt=0.02)
        applied, phase, blend, walking, reference_distance = step_gait_torch(
            applied,
            phase,
            blend,
            walking,
            reference_distance,
            torch.as_tensor(command),
            profile,
            dt=0.02,
        )

    np.testing.assert_allclose(applied.numpy(), numpy_state.applied_command)
    np.testing.assert_allclose(phase.numpy(), numpy_state.phase)
    np.testing.assert_allclose(blend.numpy(), numpy_state.blend)
    np.testing.assert_array_equal(walking.numpy(), numpy_state.walking)
    np.testing.assert_allclose(reference_distance.numpy(), numpy_state.reference_distance_m)
