"""Run one frozen walking-v2 curriculum seed with fail-closed stage promotion."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from g1_mjlab.artifacts import sha256_file, write_atomic_json
from g1_mjlab.config import load_config
from g1_mjlab.walking_v2_curriculum import load_curriculum_method


def _run(command: list[str], log: Path) -> int:
    with log.open("w", encoding="utf-8") as stream:
        return subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=False
        ).returncode


def _status(path: Path, value: dict[str, Any]) -> None:
    write_atomic_json(path, {"schema_version": 1, **value})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    method = load_curriculum_method(args.method.resolve(strict=True))
    if args.seed not in method.seeds:
        raise ValueError(f"seed must be one of {method.seeds}")
    if args.output.exists():
        raise FileExistsError(f"curriculum output already exists: {args.output}")
    source = args.source_checkpoint.resolve(strict=True)
    base = load_config(args.base_config.resolve(strict=True))
    if base.task_id != "G1-Walking-Flat-v2":
        raise ValueError("curriculum requires G1-Walking-Flat-v2")
    args.output.mkdir(parents=True)
    status_path = args.output / "status.json"
    state: dict[str, Any] = {
        "status": "running",
        "seed": args.seed,
        "method": method.name,
        "method_sha256": sha256_file(args.method),
        "source_checkpoint_sha256": sha256_file(source),
        "stages": [],
    }
    _status(status_path, state)
    checkpoint = source
    config_path = source.parent.parent / "config.json"
    evaluator = Path(__file__).with_name("evaluate_walking_v2_stage.py")
    for index, stage in enumerate(method.stages, start=1):
        stage_root = args.output / f"stage-{index:02d}-{stage.stage}"
        stage_root.mkdir()
        if stage.updates:
            run_name = f"walking-v2-s{args.seed}-p{index:02d}-{stage.stage}"
            config = replace(
                base,
                seed=args.seed,
                num_envs=method.num_envs,
                rollout_steps=method.rollout_steps,
                save_interval=method.save_interval_updates,
                max_iterations=stage.updates,
                run_name=run_name,
            )
            config_path = stage_root / "config.json"
            write_atomic_json(config_path, config.to_dict())
            run = stage_root / "train"
            returncode = _run(
                [
                    sys.executable,
                    "-m",
                    "g1_mjlab",
                    "train",
                    "--config",
                    str(config_path),
                    "--walking-v2-curriculum-profile",
                    str(stage.profile),
                    "--fine-tune",
                    str(checkpoint),
                    "--output",
                    str(run),
                ],
                stage_root / "train.log",
            )
            if returncode:
                state.update(status="failed", failed_stage=stage.stage, reason="training_failed")
                _status(status_path, state)
                return returncode
            checkpoint = run / "checkpoints" / f"model_{stage.updates - 1}.pt"
            checkpoint.resolve(strict=True)
        gate = stage_root / "gate"
        command = [
            sys.executable,
            str(evaluator),
            "--stage",
            stage.stage,
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(gate),
            "--horizon-seconds",
            str(method.fixed_speed_evaluation_horizon_s),
            "--scenarios",
            str(method.development_scenarios),
            "--criteria",
            str(method.development_criteria),
        ]
        returncode = _run(command, stage_root / "gate.log")
        decision_path = gate / "decision.json"
        stage_result = {
            "stage": stage.stage,
            "updates": stage.updates,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "decision": str(decision_path),
            "decision_sha256": sha256_file(decision_path),
            "passed": returncode == 0,
        }
        state["stages"].append(stage_result)
        if returncode:
            state.update(status="failed", failed_stage=stage.stage, reason="promotion_failed")
            _status(status_path, state)
            return 2
        _status(status_path, state)
    state.update(
        status="completed",
        selected_checkpoint=str(checkpoint),
        selected_checkpoint_sha256=sha256_file(checkpoint),
        qualification_claim=False,
    )
    _status(status_path, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
