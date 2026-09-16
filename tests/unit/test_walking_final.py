from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_mjlab.gait_evaluation.acceptance import FinalAcceptanceCriteria
from g1_mjlab.walking_final import (
    assess_final_prerequisite,
    build_final_scenario_payload,
    qualify_final_walking,
    qualify_final_walking_v2,
)


def _state() -> tuple[list[float], list[float]]:
    return [0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0, *([0.0] * 29)], [0.0] * 35


def test_final_scenarios_are_deterministic_balanced_and_exactly_sixty_seconds() -> None:
    qpos, qvel = _state()

    first = build_final_scenario_payload(qpos, qvel)
    second = build_final_scenario_payload(qpos, qvel)

    assert first == second
    assert len(first["scenarios"]) == 100
    assert [item["seed"] for item in first["scenarios"]] == list(range(20000, 20100))
    for category in ("slow", "medium", "fast", "start_stop"):
        selected = [item for item in first["scenarios"] if item["category"] == category]
        assert len(selected) == 25
        assert all(
            sum(segment["duration_s"] for segment in item["segments"]) == 60 for item in selected
        )


def test_final_prerequisite_rejects_unqualified_or_narrow_policy(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "status": "unqualified_development",
                "qualified_command_domain_m_s": [],
                "checkpoint_sha256": "checkpoint",
            }
        ),
        encoding="utf-8",
    )

    result = assess_final_prerequisite(bundle, None)

    assert result["final_execution_authorized"] is False
    assert "development_qualified_policy" in result["missing"]
    assert "qualified_speed_domain_0_to_0_8_m_s" in result["missing"]
    assert "owner_visual_acceptance" in result["missing"]


def test_qualification_requires_95_overall_and_23_per_stratum(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "acceptance": {"minimum_overall_passes": 95, "minimum_stratum_passes": 23},
                "hashes": {"checkpoint": "abc", "onnx": "def"},
            }
        ),
        encoding="utf-8",
    )
    trials = [
        {
            "trial_id": index,
            "category": ("slow", "medium", "fast", "start_stop")[index // 25],
            "function_passed": index != 0,
            "style_passed": index != 0,
        }
        for index in range(100)
    ]
    mjlab = tmp_path / "mjlab.json"
    native = tmp_path / "native.json"
    for path in (mjlab, native):
        path.write_text(
            json.dumps({"trials": trials, "checkpoint_sha256": "abc"}), encoding="utf-8"
        )

    result = qualify_final_walking(freeze, mjlab, native, tmp_path / "qualification.json")

    assert result["qualified"] is True
    assert result["backends"]["mjlab"]["overall_passes"] == 99

    trials[25]["style_passed"] = False
    trials[26]["style_passed"] = False
    trials[27]["style_passed"] = False
    for path in (mjlab, native):
        path.write_text(
            json.dumps({"trials": trials, "checkpoint_sha256": "abc"}), encoding="utf-8"
        )
    with pytest.raises(FileExistsError):
        qualify_final_walking(freeze, mjlab, native, tmp_path / "qualification.json")
    result = qualify_final_walking(freeze, mjlab, native, tmp_path / "failed.json")
    assert result["qualified"] is False


def _final_measurements() -> dict[str, object]:
    return {
        "completed_horizon_s": 60.0,
        "planned_horizon_s": 60.0,
        "finite": True,
        "terminated": False,
        "termination_reason": None,
        "forbidden_ground_contact": False,
        "steady_walk_duration_s": 40.0,
        "completed_steps": 20,
        "steady_forward_tracking_rms_m_s": 0.05,
        "lateral_tracking_rms_m_s": 0.04,
        "heading_error_max_deg": 5.0,
        "pelvis_height_min_m": 0.7,
        "tilt_over_limit_max_duration_s": 0.0,
        "absolute_tilt_peak_deg": 12.0,
        "stopped_speed_rms_max_m_s": 0.05,
        "stopped_horizontal_drift_max_m": 0.1,
        "cadence_reference_error_max_fraction": 0.1,
        "step_length_reference_error_max_fraction": 0.1,
        "step_asymmetry_fraction": 0.1,
        "alternation_fraction": 0.95,
        "tiny_step_fraction": 0.1,
        "physical_slip_rms_m_s": 0.08,
        "physical_slip_p95_m_s": 0.1,
        "physical_slip_samples": 1000,
        "physical_slip_force_coverage": 1.0,
        "unmatched_event_fraction": 0.05,
        "bilateral_flight_fraction": 0.0,
    }


def test_v2_qualification_recomputes_acceptance_and_binds_exact_trial_identities(
    tmp_path: Path,
) -> None:
    criteria = FinalAcceptanceCriteria().to_dict()
    identities = {
        name: f"{index:x}" * 64
        for index, name in enumerate(
            (
                "checkpoint",
                "onnx",
                "model",
                "controller",
                "contract",
                "reference",
                "host_profile",
                "evaluator",
                "acceptance",
                "scenarios",
            ),
            start=1,
        )
    }
    expected = [
        {
            "trial_id": index,
            "seed": 20000 + index,
            "category": ("slow", "medium", "fast", "start_stop")[index // 25],
            "initial_state_sha256": f"{index:064x}",
        }
        for index in range(100)
    ]
    freeze = tmp_path / "freeze-v2.json"
    freeze.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "criteria": criteria,
                "identities": identities,
                "trials": expected,
            }
        ),
        encoding="utf-8",
    )
    trials = [
        {**item, "measurements": _final_measurements(), "accepted": True} for item in expected
    ]
    # A supplied pass flag cannot override a failing raw measurement.
    trials[0]["measurements"]["physical_slip_rms_m_s"] = 0.5  # type: ignore[index]
    for backend in ("mjlab", "native"):
        (tmp_path / f"{backend}.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "backend": backend,
                    "identities": identities,
                    "trials": trials,
                }
            ),
            encoding="utf-8",
        )

    result = qualify_final_walking_v2(
        freeze, tmp_path / "mjlab.json", tmp_path / "native.json", tmp_path / "result.json"
    )

    assert result["qualified"] is True
    assert result["backends"]["mjlab"]["overall_passes"] == 99
    assert "slip_rms" in result["backends"]["mjlab"]["failures"]["0"]

    trials[1]["trial_id"] = 0
    (tmp_path / "native-duplicate.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "backend": "native",
                "identities": identities,
                "trials": trials,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact frozen trial identities"):
        qualify_final_walking_v2(
            freeze,
            tmp_path / "mjlab.json",
            tmp_path / "native-duplicate.json",
            tmp_path / "duplicate-result.json",
        )
