from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from g1_mjlab.walking_experiments import (
    Stage19DecisionRule,
    build_development_scenario_payload,
    build_stage19_experiment_table,
    compare_development_evaluation,
    stop_decision,
)


def test_development_scenarios_are_explicit_bounded_and_reproducible() -> None:
    qpos = np.asarray([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, *([0.0] * 29)])
    qvel = np.zeros(35)
    limits = np.column_stack((np.full(29, -0.2), np.full(29, 0.2)))

    first = build_development_scenario_payload(qpos, qvel, limits, seed=10042)
    second = build_development_scenario_payload(qpos, qvel, limits, seed=10042)

    assert first == second
    assert first["schema_version"] == 3
    scenarios = first["scenarios"]
    assert len(scenarios) == 16
    assert [item["initial_phase"] for item in scenarios] == [0.0, 0.25, 0.5, 0.75] * 4
    assert [item["category"] for item in scenarios] == [
        *("nominal" for _ in range(4)),
        *("yaw" for _ in range(4)),
        *("joint_position" for _ in range(4)),
        *("root_velocity" for _ in range(4)),
    ]
    for item in scenarios:
        assert len(item["initial_qpos"]) == 36
        assert len(item["initial_qvel"]) == 35
        assert sum(segment["duration_s"] for segment in item["segments"]) == 15.0
    for item in scenarios[4:8]:
        yaw = 2 * math.atan2(item["initial_qpos"][6], item["initial_qpos"][3])
        assert math.degrees(yaw) in {-5.0, -2.5, 2.5, 5.0}
    for item in scenarios[8:12]:
        assert np.max(np.abs(np.asarray(item["initial_qpos"])[7:])) <= 0.015
    for item in scenarios[12:16]:
        assert np.linalg.norm(np.asarray(item["initial_qvel"])[0:2]) <= 0.03 + 1e-12


def _summary(*, scale: float, functional: int = 16, style: int = 0) -> dict[str, object]:
    trials = []
    for index in range(16):
        trials.append(
            {
                "trial_id": index,
                "scenario_name": f"{index:02d}-nominal",
                "functional_passed": index < functional,
                "style_passed": index < style,
                "completed_horizon_s": 15.0,
                "planned_horizon_s": 15.0,
                "command_rms_m_s": 0.1,
                "cadence_steps_s": 0.9727 * (1 - 0.4 * scale),
                "step_length_median_m": 0.6064 * (1 - 0.4 * scale),
                "physical_slip_rms_m_s": 0.12 * (1 + scale),
                "same_side_repeat_count": 3 * scale,
                "raw_transition_rate_s": 2.0,
                "stop_segments": [{"passed": True, "eligible_samples": 50}],
                "completed_steps": 8,
                "insufficient_evidence": False,
            }
        )
    return {"schema_version": 2, "planned": 16, "completed": 16, "trials": trials}


def test_development_comparator_applies_function_and_style_progress_rule() -> None:
    rule = Stage19DecisionRule()
    source = _summary(scale=1.0)
    improved = _summary(scale=0.5)

    result = compare_development_evaluation(source, improved, rule)

    assert result["function_gate_passed"]
    assert result["improved_metric_count"] >= 2
    assert result["extension_eligible"]

    failed_function = compare_development_evaluation(
        source, _summary(scale=0.4, functional=12), rule
    )
    assert not failed_function["extension_eligible"]


def test_stop_decision_honors_low_function_and_no_progress_patience() -> None:
    low_function = [
        {"function_pass_count": 12, "extension_eligible": False},
        {"function_pass_count": 11, "extension_eligible": False},
    ]
    assert stop_decision(low_function, Stage19DecisionRule())["stop"]
    no_progress = [
        {"function_pass_count": 16, "extension_eligible": False},
        {"function_pass_count": 16, "extension_eligible": False},
        {"function_pass_count": 16, "extension_eligible": False},
    ]
    assert stop_decision(no_progress, Stage19DecisionRule())["reason"] == "three_no_progress"


def test_experiment_table_records_failed_hypothesis_without_inventing_candidate(
    tmp_path: Path,
) -> None:
    rule = Stage19DecisionRule()
    source = _summary(scale=1.0)
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    rule_path = tmp_path / "rule.json"
    rule_path.write_text(json.dumps(asdict(rule)), encoding="utf-8")
    campaigns: dict[str, Path] = {}
    for arm, scale in (("a", 0.9), ("b", 0.7), ("c", 0.8)):
        campaign = tmp_path / arm
        evaluation = campaign / "evaluations/segment-000"
        evaluation.mkdir(parents=True)
        summary = _summary(scale=scale)
        (evaluation / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        comparison = compare_development_evaluation(source, summary, rule)
        comparison.update(
            {
                "completed_updates": 100,
                "checkpoint": str(campaign / "model_99.pt"),
                "checkpoint_sha256": arm * 64,
            }
        )
        (evaluation / "development-comparison.json").write_text(
            json.dumps(comparison), encoding="utf-8"
        )
        (campaign / "state.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        campaigns[arm] = campaign

    result = build_stage19_experiment_table(
        source_path, campaigns, rule_path, tmp_path / "experiment-table.json"
    )

    assert result["outcome"] == "failed_hypothesis"
    assert result["selected_candidate"] is None
    assert result["method_decision_required"]
    assert not result["arm_d_authorized"]
