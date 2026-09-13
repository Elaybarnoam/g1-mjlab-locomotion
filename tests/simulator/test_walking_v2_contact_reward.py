from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from g1_mjlab.tasks.walking_v2_mdp import update_debounced_contact  # noqa: E402


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
