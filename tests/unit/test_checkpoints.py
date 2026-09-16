import json
from pathlib import Path

import pytest

from g1_mjlab.checkpoints import (
    checkpoint_index,
    checkpoint_iteration,
    ordered_checkpoints,
    publish_checkpoint,
)


def test_numeric_ordering(tmp_path: Path) -> None:
    for index in (99, 1000, 75, 100):
        (tmp_path / f"model_{index}.pt").touch()
    assert [checkpoint_iteration(p) for p in ordered_checkpoints(tmp_path)] == [75, 99, 100, 1000]


def test_malformed_counter_rejected() -> None:
    with pytest.raises(ValueError):
        checkpoint_iteration(Path("model_latest.pt"))


def test_publish_checkpoint_is_atomic_and_records_lineage(tmp_path: Path) -> None:
    source = tmp_path / "upstream" / "model_99.pt"
    source.parent.mkdir()
    source.write_bytes(b"complete checkpoint")

    target = publish_checkpoint(
        source,
        tmp_path / "published",
        transitions_per_update=1536,
        source_lineage={"checkpoint_sha256": "a" * 64},
    )

    assert target.read_bytes() == b"complete checkpoint"
    index = json.loads((target.parent / "index.json").read_text(encoding="utf-8"))
    record = index["checkpoints"][0]
    assert record["completed_iteration"] == 99
    assert record["completed_updates"] == 100
    assert record["transition_count"] == 153_600
    assert record["validity"] == "complete"
    assert not list(target.parent.glob("*.tmp"))


def test_incomplete_temporary_checkpoint_is_never_indexed(tmp_path: Path) -> None:
    (tmp_path / ".model_7.pt.123.tmp").write_bytes(b"partial")
    assert checkpoint_index(tmp_path)["latest"] is None
