"""Create the hash-bound Plan 06 soft-reference admission decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_mjlab.motion.soft_reference_admission import (
    SoftReferenceCandidate,
    decide_soft_reference_admission,
)


def _candidate(value: str) -> SoftReferenceCandidate:
    fields = value.split(",")
    if len(fields) != 6:
        raise argparse.ArgumentTypeError(
            "candidate must be ID,SPEED,REFERENCE,ADAPTATION,KINEMATICS,DYNAMICS"
        )
    identifier, speed, reference, adaptation, kinematics, dynamics = fields
    return SoftReferenceCandidate(
        identifier,
        float(speed),
        Path(reference),
        Path(adaptation),
        Path(kinematics),
        None if dynamics == "-" else Path(dynamics),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True, type=_candidate)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = decide_soft_reference_admission(
        tuple(args.candidate),
        args.output,
        source_license=args.source_license,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "admitted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
