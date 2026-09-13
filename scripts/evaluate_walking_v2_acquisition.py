"""Evaluate every bounded acquisition checkpoint in fresh deterministic processes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from g1_mjlab.artifacts import sha256_file, write_atomic_json
from g1_mjlab.walking_v2_promotion import acquisition_decision, summarize_checkpoint


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"unrecognized checkpoint name: {path.name}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon-seconds", default=5.0, type=float)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    manifest = _object(run / "manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError("acquisition run must be completed before sequential evaluation")
    config = run / "config.json"
    checkpoints = sorted((run / "checkpoints").glob("model_*.pt"), key=_iteration)
    if not checkpoints or _iteration(checkpoints[-1]) != int(manifest["max_iterations"]) - 1:
        raise ValueError("run does not contain its declared final checkpoint")
    args.output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    recorder = Path(__file__).with_name("record_walking_v2_rollout.py")
    for checkpoint in checkpoints:
        update = _iteration(checkpoint) + 1
        evaluations: list[dict[str, Any]] = []
        for speed in (0.0, 0.4, 0.6, 0.8):
            evaluation = args.output / f"update-{update:04d}" / f"speed-{speed:.1f}"
            command = [
                sys.executable,
                str(recorder),
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(evaluation),
                "--speed",
                str(speed),
                "--horizon-seconds",
                str(args.horizon_seconds),
                "--seed",
                str(20_000 + update + round(speed * 10)),
            ]
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            log = args.output / f"update-{update:04d}" / f"speed-{speed:.1f}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
            if completed.returncode:
                raise RuntimeError(f"evaluation failed for update {update}, speed {speed}: {log}")
            summary_path = evaluation / "summary.json"
            summary = _object(summary_path)
            evaluations.append(summary["acquisition_metrics"])
            jobs.append(
                {
                    "update": update,
                    "speed_m_s": speed,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "summary": str(summary_path.relative_to(args.output)),
                    "summary_sha256": sha256_file(summary_path),
                }
            )
        rows.append(summarize_checkpoint(update, sha256_file(checkpoint), evaluations))
    result = acquisition_decision(rows)
    result["run_manifest_sha256"] = sha256_file(run / "manifest.json")
    result["evaluation_horizon_seconds"] = args.horizon_seconds
    result["jobs"] = jobs
    write_atomic_json(args.output / "decision.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "promoted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
