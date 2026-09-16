"""Select walking-v2 parallelism from complete, measured GPU probes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from .artifacts import read_jsonl, sha256_file, write_atomic_json


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def qualify_resource_profile(probes: list[tuple[int, Path, Path]], output: Path) -> dict[str, Any]:
    """Require ordered successful probes and choose maximum sustained useful throughput."""
    if [count for count, _, _ in probes] != [64, 128, 256]:
        raise ValueError("resource probes must be supplied in frozen 64, 128, 256 order")
    results: list[dict[str, Any]] = []
    for count, run, evaluation in probes:
        manifest = _object(run / "manifest.json")
        config = _object(run / "config.json")
        optimization = _object(run / "optimization.json")
        memory = _object(run / "memory.json")
        eval_summary = _object(evaluation / "summary.json")
        metrics, truncated = read_jsonl(run / "metrics/metrics.jsonl")
        fps = [
            float(item["value"])
            for item in metrics
            if item.get("metric") == "Perf/total_fps" and int(item["update"]) > 0
        ]
        total_memory = int(memory["device_total_bytes"])
        free_ratio = int(memory["device_free_at_end_bytes"]) / total_memory
        valid = (
            not truncated
            and manifest.get("status") == "completed"
            and config.get("num_envs") == count
            and optimization.get("all_kl_samples_finite") is True
            and optimization.get("all_recorded_gradients_finite") is True
            and len(fps) == 2
            and free_ratio >= 0.20
            and eval_summary.get("recorded_steps") == eval_summary.get("planned_steps")
        )
        transitions = int(optimization["total_transitions"])
        combined_wall = float(manifest["wall_seconds"]) + float(
            eval_summary["wall_seconds_including_startup"]
        )
        results.append(
            {
                "num_envs": count,
                "valid": valid,
                "sustained_transitions_per_second": mean(fps),
                "sustained_samples": fps,
                "end_to_end_transitions_per_second": transitions / combined_wall,
                "training_wall_seconds": manifest["wall_seconds"],
                "evaluation_wall_seconds": eval_summary["wall_seconds_including_startup"],
                "torch_peak_allocated_bytes": memory["torch_peak_allocated_bytes"],
                "torch_peak_reserved_bytes": memory["torch_peak_reserved_bytes"],
                "device_free_ratio_at_end": free_ratio,
                "run_manifest_sha256": sha256_file(run / "manifest.json"),
                "evaluation_summary_sha256": sha256_file(evaluation / "summary.json"),
            }
        )
    candidates = [item for item in results if item["valid"]]
    if not candidates:
        raise RuntimeError("no resource candidate passed the frozen validity checks")
    selected = max(candidates, key=lambda item: item["sustained_transitions_per_second"])
    updates, cadence = 4000, 100
    total_transitions = updates * int(selected["num_envs"]) * 24
    sustained = float(selected["sustained_transitions_per_second"])
    eval_wall = float(selected["evaluation_wall_seconds"])
    lower_seconds = total_transitions / sustained
    upper_seconds = total_transitions / (0.8 * sustained) + math.ceil(updates / cadence) * eval_wall
    profile = {
        "schema_version": 2,
        "task_id": "G1-Walking-Flat-v2",
        "selection_rule": "maximum valid sustained transitions per second after warmup",
        "minimum_device_free_ratio": 0.20,
        "probes": results,
        "selected_num_envs": selected["num_envs"],
        "rollout_steps": 24,
        "transitions_per_update": int(selected["num_envs"]) * 24,
        "estimated_4000_update_wall_seconds": {
            "lower": lower_seconds,
            "upper": upper_seconds,
            "method": "measured sustained rate; upper uses 20% slowdown plus 40 measured evaluations",
        },
        "limitations": [
            "Torch memory excludes allocations owned by MuJoCo Warp.",
            "The estimate is local-laptop evidence, not a service-level guarantee.",
        ],
    }
    write_atomic_json(output, profile)
    return profile
