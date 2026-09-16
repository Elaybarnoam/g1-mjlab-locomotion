"""Freeze the Plan 06 resource profile from ordered probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_mjlab.walking_v2_resources import qualify_resource_profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-64", required=True, type=Path)
    parser.add_argument("--eval-64", required=True, type=Path)
    parser.add_argument("--run-128", required=True, type=Path)
    parser.add_argument("--eval-128", required=True, type=Path)
    parser.add_argument("--run-256", required=True, type=Path)
    parser.add_argument("--eval-256", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = qualify_resource_profile(
        [
            (64, args.run_64, args.eval_64),
            (128, args.run_128, args.eval_128),
            (256, args.run_256, args.eval_256),
        ],
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
