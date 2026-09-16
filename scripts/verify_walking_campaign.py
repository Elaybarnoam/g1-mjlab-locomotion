"""Verify a completed local walking campaign without launching a simulator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from g1_mjlab.artifacts import read_jsonl, sha256_file, write_atomic_json
from g1_mjlab.checkpoints import checkpoint_iteration


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def verify(campaign: Path) -> dict[str, Any]:
    state = _json(campaign / "state.json")
    manifest = _json(campaign / "resolved-manifest.json")
    run_config = _json(Path(manifest["run_config"]["path"]))
    events, truncated = read_jsonl(campaign / "events.jsonl")
    checkpoint_events = [event for event in events if event["event"] == "checkpoint_completed"]
    lifecycle = [event["to"] for event in events if event["event"] == "state_transition"]
    segment_count = len(checkpoint_events)
    expected_lifecycle = ["smoke_running", "smoke_passed", "training"]
    for segment_index in range(segment_count):
        expected_lifecycle.append("evaluating")
        expected_lifecycle.append("completed" if segment_index == segment_count - 1 else "training")
    checks: dict[str, bool] = {
        "campaign_completed": state["status"] == "completed",
        "budget_completed": state["completed_updates"] == manifest["max_updates"],
        "transition_count_exact": state["transition_count"]
        == manifest["max_updates"] * manifest["environment_count"] * run_config["rollout_steps"],
        "journal_complete": not truncated and bool(events),
        "lifecycle_ordered": lifecycle == expected_lifecycle,
        "gpu_process_lifecycle_sequential": lifecycle == expected_lifecycle,
        "criteria_hash_matches": state["evaluation_criteria_sha256"]
        == sha256_file(Path(state["evaluation_criteria"])),
    }
    segment_results: list[dict[str, Any]] = []
    expected_first_iteration = 0
    for checkpoint_event in checkpoint_events:
        checkpoint_path = Path(checkpoint_event["checkpoint"])
        segment = checkpoint_path.parent.parent
        run_manifest = _json(segment / "manifest.json")
        learning = _json(segment / "learning-summary.json")
        optimization = _json(segment / "optimization.json")
        normalizers = _json(segment / "normalizer-state.json")
        memory = _json(segment / "memory.json")
        index = _json(segment / "checkpoints/index.json")
        latest = segment / "checkpoints" / index["latest"]
        records = {record["name"]: record for record in index["checkpoints"]}
        latest_record = records[index["latest"]]
        segment_checks = {
            "run_completed": run_manifest["status"] == "completed",
            "losses_finite": learning["all_losses_finite"],
            "parameters_finite": all(
                value["all_parameters_finite"] for value in optimization["components"].values()
            ),
            "parameters_changed": all(
                value["parameters_changed"] for value in optimization["components"].values()
            ),
            "gradients_finite": optimization["all_recorded_gradients_finite"],
            "gradient_norm_finite": math.isfinite(optimization["final_gradient_global_norm"]),
            "kl_complete_and_finite": optimization["all_kl_samples_finite"]
            and len(optimization["kl_divergence_samples"])
            == learning["updates"] * optimization["kl_samples_per_update"],
            "normalizers_finite": all(
                tensor["finite"]
                for model in (normalizers["actor"], normalizers["critic"])
                for tensor in model.values()
            ),
            "memory_finite": all(
                math.isfinite(value) for value in memory.values() if isinstance(value, (int, float))
            ),
            "checkpoint_hash_matches": latest_record["sha256"] == sha256_file(latest),
            "checkpoint_size_matches": latest_record["size_bytes"] == latest.stat().st_size,
            "checkpoint_complete": latest_record["validity"] == "complete",
            "iteration_contiguous": checkpoint_iteration(latest)
            == expected_first_iteration + learning["updates"] - 1,
        }
        evaluation = _json(
            campaign
            / "evaluations"
            / f"segment-{int(checkpoint_event['segment']):03d}"
            / "summary.json"
        )
        segment_checks["evaluation_complete"] = (
            evaluation["schema_version"] == 2
            and evaluation["completed"] == evaluation["planned"]
            and evaluation["checkpoint_sha256"] == latest_record["sha256"]
        )
        if expected_first_iteration:
            resume = _json(segment / "resume.json")
            segment_checks["resume_contiguous"] = resume[
                "first_new_iteration"
            ] == expected_first_iteration and set(resume["verified_restored_components"]) == {
                "actor_state_dict",
                "critic_state_dict",
                "optimizer_state_dict",
            }
        expected_first_iteration = checkpoint_iteration(latest) + 1
        segment_results.append(
            {
                "segment": segment.name,
                "passed": all(segment_checks.values()),
                "checks": segment_checks,
                "checkpoint": str(latest),
                "checkpoint_sha256": latest_record["sha256"],
            }
        )
    checks["segments_present"] = bool(segment_results)
    checks["segments_pass"] = all(result["passed"] for result in segment_results)
    checks["final_checkpoint_matches_state"] = (
        segment_results[-1]["checkpoint_sha256"] == state["latest_checkpoint_sha256"]
        if segment_results
        else False
    )
    return {
        "schema_version": 1,
        "passed": all(checks.values()),
        "checks": checks,
        "segments": segment_results,
        "event_count": len(events),
        "recovery_attempts": state["recovery_attempts"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.campaign)
    write_atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
