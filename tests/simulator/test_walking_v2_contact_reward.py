from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from g1_mjlab.tasks.walking_v2_mdp import (  # noqa: E402
    gait_touchdown_event_signal,
    update_debounced_contact,
)


def test_debounced_contact_rejects_flicker_and_confirms_persistent_change() -> None:
    stable = torch.tensor([[True, True]])
    candidate = stable.clone()
    age = torch.zeros((1, 2))

    stable, candidate, age, flicker = update_debounced_contact(
        torch.tensor([[False, True]]), stable, candidate, age, dt=0.02
    )
    assert stable.tolist() == [[True, True]]
    assert not flicker.any()
    stable, candidate, age, flicker = update_debounced_contact(
        torch.tensor([[True, True]]), stable, candidate, age, dt=0.02
    )
    assert stable.tolist() == [[True, True]]
    assert flicker.tolist() == [[True, False]]

    for _ in range(3):
        stable, candidate, age, flicker = update_debounced_contact(
            torch.tensor([[False, True]]), stable, candidate, age, dt=0.02
        )
        assert not flicker.any()
    assert stable.tolist() == [[False, True]]


def test_touchdown_reward_requires_alternation_and_reference_stance() -> None:
    single = torch.tensor([True, True, True, False])
    touchdown_foot = torch.tensor([1, 1, 0, 1])
    last_touchdown = torch.tensor([0, 0, 0, 0])
    expected_contact = torch.tensor(
        [[False, True], [True, False], [True, False], [False, True]]
    )

    signal = gait_touchdown_event_signal(
        single, touchdown_foot, last_touchdown, expected_contact
    )

    torch.testing.assert_close(signal, torch.tensor([1.0, 0.0, -1.0, 0.0]))
