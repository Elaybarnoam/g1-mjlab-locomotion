from pathlib import Path

import pytest

from g1_mjlab.checkpoints import checkpoint_iteration, ordered_checkpoints


def test_numeric_ordering(tmp_path: Path) -> None:
    for index in (99, 1000, 75, 100):
        (tmp_path / f"model_{index}.pt").touch()
    assert [checkpoint_iteration(p) for p in ordered_checkpoints(tmp_path)] == [75, 99, 100, 1000]


def test_malformed_counter_rejected() -> None:
    with pytest.raises(ValueError):
        checkpoint_iteration(Path("model_latest.pt"))
