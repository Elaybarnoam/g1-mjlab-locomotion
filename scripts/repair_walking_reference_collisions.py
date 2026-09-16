"""Apply the frozen Plan 06 collision repair to one walking reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_mjlab.motion.reference_repair import repair_reference_collisions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adaptation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    result = repair_reference_collisions(
        args.reference,
        args.model,
        args.adaptation,
        args.output,
        args.report,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
