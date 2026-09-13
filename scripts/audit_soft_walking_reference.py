"""Audit one walking reference for Plan 06 soft-reference admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_mjlab.motion.reference_kinematics import audit_reference_kinematics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adaptation", required=True, type=Path)
    parser.add_argument("--speed", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_reference_kinematics(
        args.reference,
        args.model,
        args.adaptation,
        args.output,
        speed_m_s=args.speed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
