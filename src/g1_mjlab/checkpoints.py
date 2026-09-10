"""Explicit numeric ordering and integrity metadata for upstream checkpoints."""

import re
from pathlib import Path
from typing import Any

from .artifacts import sha256_file


def checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"unexpected checkpoint name: {path.name}")
    return int(match.group(1))


def ordered_checkpoints(directory: Path) -> list[Path]:
    return sorted(directory.glob("model_*.pt"), key=checkpoint_iteration)


def checkpoint_index(directory: Path) -> dict[str, Any]:
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
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }
