"""Evaluate one walking-v2 curriculum checkpoint against its frozen stage gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from g1_mjlab.artifacts import sha256_file, write_atomic_json
from g1_mjlab.walking_v2_curriculum import assess_fixed_speed_stage, assess_transition_stage

_FIXED_SPEEDS = {
    "stand": (0.0,),
    "stand-walk-040": (0.0, 0.4),
    "add-060": (0.0, 0.4, 0.6),
    "add-080": (0.0, 0.4, 0.6, 0.8),
}


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _run(command: list[str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"stage evaluation subprocess failed; inspect {log}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=(*_FIXED_SPEEDS, "transitions", "robustness")
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon-seconds", type=float, default=10.0)
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("configs/walking-v2/development-scenarios-v2.json"),
    )
    parser.add_argument(
        "--criteria",
        type=Path,
        default=Path("configs/walking-v2/development-criteria-v2.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"stage evaluation output already exists: {args.output}")
    args.output.mkdir(parents=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    config = args.config.resolve(strict=True)
    if args.stage in _FIXED_SPEEDS:
        rows: list[dict[str, Any]] = []
        recorder = Path(__file__).with_name("record_walking_v2_rollout.py")
        for speed in _FIXED_SPEEDS[args.stage]:
            trial = args.output / f"speed-{speed:.1f}"
            _run(
                [
                    sys.executable,
                    str(recorder),
                    "--config",
                    str(config),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(trial),
                    "--speed",
                    str(speed),
                    "--horizon-seconds",
                    str(args.horizon_seconds),
                    "--seed",
                    str(31_000 + round(speed * 10)),
                ],
                args.output / f"speed-{speed:.1f}.log",
            )
            rows.append(_object(trial / "summary.json")["acquisition_metrics"])
        decision = assess_fixed_speed_stage(args.stage, rows, horizon_s=args.horizon_seconds)
        decision["metrics"] = rows
    else:
        diagnostic = args.output / "diagnostic"
        _run(
            [
                sys.executable,
                "-m",
                "g1_mjlab",
                "diagnose-walking",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--scenarios",
                str(args.scenarios.resolve(strict=True)),
                "--criteria",
                str(args.criteria.resolve(strict=True)),
                "--output",
                str(diagnostic),
                "--physics-trace",
            ]
            + (["--robustness"] if args.stage == "robustness" else []),
            args.output / "diagnostic.log",
        )
        decision = assess_transition_stage(args.stage, _object(diagnostic / "summary.json"))
    decision["checkpoint"] = checkpoint.name
    decision["checkpoint_sha256"] = sha256_file(checkpoint)
    decision["config_sha256"] = sha256_file(config)
    write_atomic_json(args.output / "decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if decision["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
