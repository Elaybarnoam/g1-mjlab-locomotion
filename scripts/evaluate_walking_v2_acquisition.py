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
    parser.add_argument("--run", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon-seconds", default=5.0, type=float)
    parser.add_argument("--previous-decision", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    runs = [path.resolve(strict=True) for path in args.run]
    candidates: dict[int, tuple[Path, Path]] = {}
    manifest_hashes: list[str] = []
    for run in runs:
        manifest = _object(run / "manifest.json")
        if manifest.get("status") not in {"completed", "interrupted"}:
            raise ValueError("acquisition sources must be completed or preserved interruptions")
        checkpoints = sorted((run / "checkpoints").glob("model_*.pt"), key=_iteration)
        if not checkpoints:
            raise ValueError("acquisition source contains no complete checkpoints")
        if manifest["status"] == "completed":
            resume_path = run / "resume.json"
            restored = (
                int(_object(resume_path)["restored_iteration"]) if resume_path.exists() else -1
            )
            expected_final = restored + int(manifest["max_iterations"])
            if _iteration(checkpoints[-1]) != expected_final:
                raise ValueError("completed run does not contain its declared final checkpoint")
        for checkpoint in checkpoints:
            iteration = _iteration(checkpoint)
            candidates[iteration] = (run / "config.json", checkpoint)
        manifest_hashes.append(sha256_file(run / "manifest.json"))
    previous_rows: list[dict[str, Any]] = []
    previous_update = 0
    if args.previous_decision is not None:
        previous = _object(args.previous_decision.resolve(strict=True))
        previous_rows = previous["checkpoints"]
        previous_update = int(previous_rows[-1]["update"])
    final_iteration = max(candidates)
    selected_iterations = (
        sorted(candidates)
        if not previous_rows
        else [
            iteration
            for iteration in sorted(candidates)
            if iteration + 1 > previous_update
            and (iteration + 1 - previous_update >= 100 or iteration == final_iteration)
        ]
    )
    if not selected_iterations:
        raise ValueError("no new acquisition checkpoints satisfy the evaluation cadence")
    cadence_updates = [int(row["update"]) for row in previous_rows] + [
        iteration + 1 for iteration in selected_iterations
    ]
    if (not previous_rows and selected_iterations[0] != 0) or any(
        right - left not in {99, 100, 101}
        for left, right in zip(cadence_updates, cadence_updates[1:], strict=False)
    ):
        raise ValueError(f"acquisition checkpoint cadence differs: {cadence_updates}")
    args.output.mkdir(parents=True)
    rows: list[dict[str, Any]] = list(previous_rows)
    jobs: list[dict[str, Any]] = []
    recorder = Path(__file__).with_name("record_walking_v2_rollout.py")
    for iteration in selected_iterations:
        config, checkpoint = candidates[iteration]
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
        previous_update = update
    result = acquisition_decision(rows)
    result["run_manifest_sha256"] = manifest_hashes
    result["previous_decision_sha256"] = (
        sha256_file(args.previous_decision.resolve(strict=True))
        if args.previous_decision is not None
        else None
    )
    result["evaluation_horizon_seconds"] = args.horizon_seconds
    result["jobs"] = jobs
    write_atomic_json(args.output / "decision.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "promoted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
