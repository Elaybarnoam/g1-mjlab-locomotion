"""Verify the frozen Stage 19 A/B/C experiment matrix and decision."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from g1_mjlab.artifacts import read_jsonl, sha256_file, write_atomic_json
from g1_mjlab.gait_evaluation.scenarios import load_scenario_set


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _normalizers_finite(value: dict[str, Any]) -> bool:
    return set(value) == {"schema_version", "actor", "critic"} and all(
        tensor["finite"] for name in ("actor", "critic") for tensor in value[name].values()
    )


def verify(
    source_evaluation: Path,
    scenarios_path: Path,
    scenario_verification_path: Path,
    table_path: Path,
    arms: dict[str, Path],
) -> dict[str, Any]:
    scenarios = load_scenario_set(scenarios_path, control_dt=0.02)
    scenario_verification = _json(scenario_verification_path)
    source = _json(source_evaluation)
    table = _json(table_path)
    categories = Counter(scenario.category for scenario in scenarios.scenarios)
    phases = [scenario.initial_phase for scenario in scenarios.scenarios]
    checks: dict[str, bool] = {
        "sixteen_explicit_scenarios": len(scenarios.scenarios) == 16
        and all(scenario.initial_qpos is not None for scenario in scenarios.scenarios),
        "four_scenarios_per_category": categories
        == {"nominal": 4, "yaw": 4, "joint_position": 4, "root_velocity": 4},
        "phase_cycle_exact": phases == [0.0, 0.25, 0.5, 0.75] * 4,
        "scenario_byte_hash_verified": scenario_verification["scenario_sha256"]
        == sha256_file(scenarios_path),
        "scenario_simulator_validation_passed": all(
            (
                scenario_verification["validation"]["all_finite"],
                scenario_verification["validation"]["termination_count"] == 0,
                scenario_verification["validation"]["self_collision_count"] == 0,
            )
        ),
        "source_evaluated_once_on_full_cohort": source["planned"] == 16
        and source["completed"] == 16,
        "table_hashes_source": table["source_evaluation_sha256"] == sha256_file(source_evaluation),
        "failed_hypothesis_recorded": table["outcome"] == "failed_hypothesis"
        and table["selected_candidate"] is None,
        "no_unjustified_extension": not table["extension_authorized"],
        "no_unjustified_arm_d": not table["arm_d_authorized"],
        "method_decision_recorded": table["method_decision_required"]
        and not table["method_decision"]["further_ppo_reward_training_authorized"],
    }
    arm_results: dict[str, Any] = {}
    source_checkpoint_hashes: set[str] = set()
    total_updates = 0
    for arm, campaign in sorted(arms.items()):
        state = _json(campaign / "state.json")
        manifest = _json(campaign / "resolved-manifest.json")
        source_checkpoint_hashes.add(manifest["source_checkpoint"]["sha256"])
        events, truncated = read_jsonl(campaign / "events.jsonl")
        checkpoint_events = [event for event in events if event["event"] == "checkpoint_completed"]
        evaluations = sorted((campaign / "evaluations").glob("segment-*"))
        segment_checks: list[bool] = []
        for event, evaluation in zip(checkpoint_events, evaluations, strict=True):
            segment = Path(event["checkpoint"]).parent.parent
            optimization = _json(segment / "optimization.json")
            normalizers = _json(segment / "normalizer-state.json")
            summary = _json(evaluation / "summary.json")
            comparison = _json(evaluation / "development-comparison.json")
            segment_checks.append(
                all(
                    (
                        summary["planned"] == 16,
                        summary["completed"] == 16,
                        comparison["checkpoint_sha256"] == event["checkpoint_sha256"],
                        sha256_file(Path(event["checkpoint"])) == event["checkpoint_sha256"],
                        optimization["all_recorded_gradients_finite"],
                        optimization["all_kl_samples_finite"],
                        _normalizers_finite(normalizers),
                    )
                )
            )
        updates = int(state["completed_updates"])
        total_updates += updates
        expected_updates = 200 if arm == "a" else 300
        expected_status = "stopped" if arm == "a" else "completed"
        arm_passed = all(
            (
                not truncated,
                state["status"] == expected_status,
                updates == expected_updates,
                state["transition_count"] == updates * 64 * 24,
                state["recovery_attempts"] == 0,
                len(evaluations) == expected_updates // 100,
                all(segment_checks),
                not any(
                    _json(path / "development-comparison.json")["extension_eligible"]
                    for path in evaluations
                ),
            )
        )
        arm_results[arm] = {
            "passed": arm_passed,
            "status": state["status"],
            "updates": updates,
            "transitions": state["transition_count"],
            "checkpoint_count": len(checkpoint_events),
        }
    checks["independent_common_source"] = len(source_checkpoint_hashes) == 1
    checks["arm_budgets_and_evidence_pass"] = all(
        result["passed"] for result in arm_results.values()
    )
    checks["actual_update_total_exact"] = total_updates == 800
    return {
        "schema_version": 1,
        "passed": all(checks.values()),
        "checks": checks,
        "arms": arm_results,
        "actual_updates": total_updates,
        "actual_training_transitions": total_updates * 64 * 24,
        "experiment_table_sha256": sha256_file(table_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-evaluation", required=True, type=Path)
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument("--scenario-verification", required=True, type=Path)
    parser.add_argument("--experiment-table", required=True, type=Path)
    parser.add_argument("--arm-a", required=True, type=Path)
    parser.add_argument("--arm-b", required=True, type=Path)
    parser.add_argument("--arm-c", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(
        args.source_evaluation,
        args.scenarios,
        args.scenario_verification,
        args.experiment_table,
        {"a": args.arm_a, "b": args.arm_b, "c": args.arm_c},
    )
    write_atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
