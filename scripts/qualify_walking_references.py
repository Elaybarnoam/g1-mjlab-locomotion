"""Freeze the P05-04 reference decision from immutable candidate evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_mjlab.motion.reference_qualification import (
    ReferenceCandidateEvidence,
    qualify_reference_candidates,
)


def _candidate(value: str) -> ReferenceCandidateEvidence:
    fields = value.split(",")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "candidate must be ID,SPEED,ATTEMPT,ADAPTATION_JSON,DYNAMICS_JSON_OR_-"
        )
    identifier, speed, attempt, adaptation, dynamics = fields
    return ReferenceCandidateEvidence(
        identifier,
        float(speed),
        int(attempt),
        Path(adaptation),
        None if dynamics == "-" else Path(dynamics),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True, type=_candidate)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = qualify_reference_candidates(tuple(args.candidate), args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "qualified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
