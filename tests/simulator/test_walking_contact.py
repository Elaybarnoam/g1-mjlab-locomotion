from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from g1_mjlab.motion.contact import ContactProfile  # noqa: E402
from g1_mjlab.tasks.walking_contact import (  # noqa: E402
    TorchContactState,
    contact_point_velocity,
    tangential_slip_terms,
)


@pytest.mark.gpu
def test_torch_contact_state_handles_indexed_reset_and_hysteresis() -> None:
    state = TorchContactState(3, ContactProfile(), "cuda:0")
    state.reset(torch.full((3, 2), 20.0, device="cuda:0"))
    state.reset(torch.zeros((1, 2), device="cuda:0"), torch.tensor([1], device="cuda:0"))
    assert state.stable_contact.cpu().tolist() == [
        [True, True],
        [False, False],
        [True, True],
    ]
    result = None
    for _ in range(3):
        result = state.update(torch.zeros((3, 2), device="cuda:0"), 0.02)
    assert result is not None
    assert result.liftoff.cpu().tolist() == [[True, True], [False, False], [True, True]]


@pytest.mark.gpu
def test_contact_point_velocity_and_tangent_projection_have_known_values() -> None:
    origin = torch.tensor([[[0.0, 0.0, 0.0]]], device="cuda:0")
    linear = torch.tensor([[[0.1, 0.0, 0.0]]], device="cuda:0")
    angular = torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda:0")
    point = torch.tensor([[[0.0, 0.2, 0.0]]], device="cuda:0")
    velocity = contact_point_velocity(origin, linear, angular, point)
    torch.testing.assert_close(velocity, torch.tensor([[[-0.1, 0.0, 0.0]]], device="cuda:0"))

    numerator, denominator = tangential_slip_terms(
        velocity,
        torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda:0"),
        torch.tensor([[100.0]], device="cuda:0"),
    )
    torch.testing.assert_close(numerator, torch.tensor([[1.0]], device="cuda:0"))
    torch.testing.assert_close(denominator, torch.tensor([[100.0]], device="cuda:0"))
