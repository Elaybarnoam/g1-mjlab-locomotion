"""Create one documented speed-specific G1 walking reference adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_mjlab.motion.reference_adaptation import AdaptationSettings, adapt_reference_for_speed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--speed", required=True, type=float)
    parser.add_argument("--period", required=True, type=float)
    parser.add_argument("--attempt", choices=(1, 2), required=True, type=int)
    args = parser.parse_args()
    settings = AdaptationSettings(
        speed_m_s=args.speed,
        cycle_period_s=args.period,
        joint_harmonics=6 if args.attempt == 1 else 4,
        arm_harmonics=3 if args.attempt == 1 else 2,
        root_harmonics=4 if args.attempt == 1 else 3,
    )
    result = adapt_reference_for_speed(args.source, args.model, args.output, args.report, settings)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
