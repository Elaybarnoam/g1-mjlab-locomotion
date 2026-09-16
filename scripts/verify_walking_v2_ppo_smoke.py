"""Verify all P06-05 runtime evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_mjlab.walking_v2_smoke import verify_walking_v2_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--resume-run", required=True, type=Path)
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify_walking_v2_smoke(
        campaign_path=args.campaign,
        run=args.run,
        resume_run=args.resume_run,
        rollout=args.rollout,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
