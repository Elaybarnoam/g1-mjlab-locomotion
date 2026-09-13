"""Aggregate deterministic acquisition evaluations and emit a promotion decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from g1_mjlab.artifacts import write_atomic_json
from g1_mjlab.walking_v2_promotion import acquisition_decision, summarize_checkpoint


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        nargs=6,
        action="append",
        metavar=("UPDATE", "SHA256", "STAND", "SPEED_040", "SPEED_060", "SPEED_080"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        summarize_checkpoint(
            int(group[0]),
            group[1],
            [_load(Path(path))["acquisition_metrics"] for path in group[2:]],
        )
        for group in args.checkpoint
    ]
    result = acquisition_decision(rows)
    write_atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "promoted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
