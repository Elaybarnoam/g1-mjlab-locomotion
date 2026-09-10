from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_mjlab.artifacts import RunStore, read_jsonl


def test_run_lifecycle_and_metrics(tmp_path: Path) -> None:
    store = RunStore.create(tmp_path / "run", {"run_id": "r1"})
    store.transition("starting")
    store.transition("running")
    assert (
        store.append_metrics(
            [
                {"update": 1, "metric": "loss", "value": 1.0},
                {"update": 2, "metric": "loss", "value": 0.5},
            ]
        )
        == 2
    )
    store.transition("completed")
    assert store.manifest()["status"] == "completed"
    records, truncated = read_jsonl(store.root / "metrics" / "metrics.jsonl")
    assert records[0]["run_id"] == "r1"
    assert records[1]["value"] == 0.5
    assert not truncated


def test_invalid_transition_is_rejected(tmp_path: Path) -> None:
    store = RunStore.create(tmp_path / "run", {"run_id": "r1"})
    with pytest.raises(ValueError, match="invalid"):
        store.transition("completed")


def test_only_incomplete_final_line_is_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"metric":"ok"}\n{"metric":', encoding="utf-8")
    records, truncated = read_jsonl(path)
    assert records == [{"metric": "ok"}]
    assert truncated
    path.write_text('{bad}\n{"metric":"ok"}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)
