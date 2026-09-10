"""Process-isolated evaluation campaigns resilient to native simulator crashes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def evaluate_checkpoints(
    *,
    config: Path,
    checkpoints: list[Path],
    run: Path,
    trials: int,
    horizon_s: float,
    retries: int = 3,
) -> dict[str, Any]:
    """Retry whole isolated processes; never reuse a possibly corrupted CUDA process."""
    if not checkpoints or not 1 <= retries <= 3:
        raise ValueError("requires checkpoints and 1..3 attempts")
    evaluation = run / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "isolation": "one fresh process per attempt",
        "attempt_limit": retries,
        "jobs": [],
    }
    for checkpoint in checkpoints:
        job: dict[str, Any] = {"checkpoint": str(checkpoint), "attempts": []}
        result["jobs"].append(job)
        succeeded = False
        for attempt in range(1, retries + 1):
            output = evaluation / f"development-{checkpoint.stem}-attempt-{attempt}"
            while output.exists():
                attempt += 1
                output = evaluation / f"development-{checkpoint.stem}-attempt-{attempt}"
            log = evaluation / f"{output.name}.log"
            command = [
                sys.executable,
                "-m",
                "g1_mjlab.cli",
                "evaluate",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--trials",
                str(trials),
                "--horizon-seconds",
                str(horizon_s),
            ]
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            attempt_record = {
                "attempt": attempt,
                "output": str(output.relative_to(run)),
                "log": str(log.relative_to(run)),
                "return_code": completed.returncode,
                "native_signal": -completed.returncode if completed.returncode < 0 else None,
            }
            job["attempts"].append(attempt_record)
            if completed.returncode == 0 and (output / "summary.json").exists():
                succeeded = True
                break
        job["succeeded"] = succeeded
        (run / "evaluation-campaign.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        if not succeeded:
            raise RuntimeError(f"evaluation failed after {retries} attempts: {checkpoint}")
    return result
