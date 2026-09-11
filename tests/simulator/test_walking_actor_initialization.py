from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from g1_mjlab.training import (  # noqa: E402
    configure_transferred_action_std,
    initialize_zero_residual_actor,
)


@pytest.mark.simulator
def test_zero_residual_initialization_changes_only_actor_output_layer() -> None:
    actor = torch.nn.Sequential(
        torch.nn.Linear(3, 8),
        torch.nn.ELU(),
        torch.nn.Linear(8, 2),
    )
    hidden_weight = actor[0].weight.detach().clone()

    metadata = initialize_zero_residual_actor(actor)

    torch.testing.assert_close(actor[0].weight, hidden_weight)
    assert torch.count_nonzero(actor[2].weight) == 0
    assert torch.count_nonzero(actor[2].bias) == 0
    assert metadata == {
        "mode": "zero-residual-output-layer",
        "output_features": 2,
        "weight_nonzero": 0,
        "bias_nonzero": 0,
    }


@pytest.mark.simulator
def test_transferred_action_std_can_be_preserved_or_reset() -> None:
    class Distribution:
        def __init__(self) -> None:
            self.std_param = torch.nn.Parameter(torch.tensor([0.11, 0.17]))

    class Actor:
        def __init__(self) -> None:
            self.distribution = Distribution()

    actor = Actor()
    assert configure_transferred_action_std(actor, 0.2, reset=False) is None
    torch.testing.assert_close(actor.distribution.std_param, torch.tensor([0.11, 0.17]))

    metadata = configure_transferred_action_std(actor, 0.2, reset=True)
    torch.testing.assert_close(actor.distribution.std_param, torch.tensor([0.2, 0.2]))
    assert metadata == "actor action distribution standard deviation"
