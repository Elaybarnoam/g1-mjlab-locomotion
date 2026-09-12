from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_mjlab.motion.reference_feasibility import (
    audit_reference_speed_map,
    write_reference_speed_map,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the walking reference against a pinned G1 model"
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--controller", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_reference_speed_map(
        args.reference.resolve(strict=True),
        args.model.resolve(strict=True),
        args.controller.resolve(strict=True),
        reference_id="nvidia-soma-g1-neutral-walk-a057-cycle-v1",
        source_speed_m_s=1.16381159304071,
        cycle_duration_s=1.06,
    )
    write_reference_speed_map(result, args.output)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
