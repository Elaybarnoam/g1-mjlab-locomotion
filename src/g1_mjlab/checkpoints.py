"""Explicit numeric ordering and integrity metadata for upstream checkpoints."""

import os
import re
import shutil
from pathlib import Path
from typing import Any

from .artifacts import sha256_file, write_atomic_json


def checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"unexpected checkpoint name: {path.name}")
    return int(match.group(1))


def ordered_checkpoints(directory: Path) -> list[Path]:
    return sorted(directory.glob("model_*.pt"), key=checkpoint_iteration)


def checkpoint_index(
    directory: Path,
    *,
    transitions_per_update: int | None = None,
    source_lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = ordered_checkpoints(directory)
    return {
        "schema_version": 1,
        "iteration_semantics": "upstream filename counter; model_0 is after the first update",
        "latest": paths[-1].name if paths else None,
        "best_development": None,
        "selection_rule": "not selected; requires development evaluation",
        "checkpoints": [
            {
                "name": path.name,
                "iteration": checkpoint_iteration(path),
                "completed_iteration": checkpoint_iteration(path),
                "completed_updates": checkpoint_iteration(path) + 1,
                "transition_count": (
                    (checkpoint_iteration(path) + 1) * transitions_per_update
                    if transitions_per_update is not None
                    else None
                ),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "source_lineage": source_lineage,
                "validity": "complete",
            }
            for path in paths
        ],
    }


def publish_checkpoint(
    source: Path,
    directory: Path,
    *,
    transitions_per_update: int,
    source_lineage: dict[str, Any] | None = None,
) -> Path:
    """Atomically expose one completed checkpoint and refresh its integrity index."""
    checkpoint_iteration(source)
    if transitions_per_update <= 0:
        raise ValueError("transitions_per_update must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / source.name
    temporary = directory / f".{source.name}.{os.getpid()}.tmp"
    with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    temporary.replace(target)
    write_atomic_json(
        directory / "index.json",
        checkpoint_index(
            directory,
            transitions_per_update=transitions_per_update,
            source_lineage=source_lineage,
        ),
    )
    return target
