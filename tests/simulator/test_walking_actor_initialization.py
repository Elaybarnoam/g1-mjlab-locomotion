from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from g1_mjlab.training import (  # noqa: E402
    configure_transferred_action_std,
)


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
