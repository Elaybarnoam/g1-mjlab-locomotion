"""Run the constrained inverse-dynamics gate for the walking-v2 reference bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_mjlab.motion.reference_dynamics import audit_reference_dynamics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--speed", action="append", dest="speeds", type=float)
    parser.add_argument("--source-speed", type=float, default=1.16381159304071)
    args = parser.parse_args()
    result = audit_reference_dynamics(
        args.reference,
        args.model,
        args.output,
        speeds_m_s=tuple(args.speeds) if args.speeds else (0.4, 0.6, 0.8),
        source_speed_m_s=args.source_speed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
