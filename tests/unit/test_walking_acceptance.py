from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from g1_mjlab.gait_evaluation.acceptance import (
    FinalAcceptanceCriteria,
    FinalTrialMeasurements,
    assess_final_trial,
    load_final_acceptance_criteria,
)


def _passing() -> FinalTrialMeasurements:
    return FinalTrialMeasurements(
        completed_horizon_s=60.0,
        planned_horizon_s=60.0,
        finite=True,
        terminated=False,
        termination_reason=None,
        forbidden_ground_contact=False,
        steady_walk_duration_s=40.0,
        completed_steps=20,
        steady_forward_tracking_rms_m_s=0.10,
        lateral_tracking_rms_m_s=0.08,
        heading_error_max_deg=10.0,
        pelvis_height_min_m=0.62,
        tilt_over_limit_max_duration_s=0.19,
        absolute_tilt_peak_deg=20.0,
        stopped_speed_rms_max_m_s=0.10,
        stopped_horizontal_drift_max_m=0.20,
        cadence_reference_error_max_fraction=0.20,
        step_length_reference_error_max_fraction=0.20,
        step_asymmetry_fraction=0.15,
        alternation_fraction=0.90,
        tiny_step_fraction=0.20,
        physical_slip_rms_m_s=0.12,
        physical_slip_p95_m_s=0.15,
        physical_slip_samples=100,
        physical_slip_force_coverage=0.95,
        unmatched_event_fraction=0.10,
        bilateral_flight_fraction=0.01,
    )


def test_final_acceptance_applies_every_frozen_threshold_at_the_boundary() -> None:
    result = assess_final_trial(_passing(), FinalAcceptanceCriteria())

    assert result.accepted is True
    assert result.violations == ()
    assert set(result.criteria) == set(FinalAcceptanceCriteria.__dataclass_fields__)


@pytest.mark.parametrize(
    ("field", "value", "violation"),
    [
        ("steady_forward_tracking_rms_m_s", 0.1001, "forward_tracking"),
        ("lateral_tracking_rms_m_s", 0.0801, "lateral_tracking"),
        ("heading_error_max_deg", 10.01, "heading"),
        ("pelvis_height_min_m", 0.619, "pelvis_height"),
        ("absolute_tilt_peak_deg", 20.01, "peak_tilt"),
        ("stopped_speed_rms_max_m_s", 0.1001, "stop_speed"),
        ("stopped_horizontal_drift_max_m", 0.201, "stop_drift"),
        ("cadence_reference_error_max_fraction", 0.201, "cadence"),
        ("step_length_reference_error_max_fraction", 0.201, "step_length"),
        ("step_asymmetry_fraction", 0.151, "step_asymmetry"),
        ("alternation_fraction", 0.899, "alternation"),
        ("tiny_step_fraction", 0.201, "tiny_steps"),
        ("physical_slip_rms_m_s", 0.121, "slip_rms"),
        ("physical_slip_p95_m_s", 0.151, "slip_p95"),
        ("unmatched_event_fraction", 0.101, "contact_events"),
        ("bilateral_flight_fraction", 0.011, "bilateral_flight"),
    ],
)
def test_final_acceptance_fails_each_numerical_limit(
    field: str, value: float, violation: str
) -> None:
    result = assess_final_trial(replace(_passing(), **{field: value}), FinalAcceptanceCriteria())

    assert result.accepted is False
    assert violation in result.violations


def test_final_acceptance_fails_tilt_above_limit_for_the_sustained_interval() -> None:
    measurements = replace(_passing(), tilt_over_limit_max_duration_s=0.20)

    result = assess_final_trial(measurements, FinalAcceptanceCriteria())

    assert result.accepted is False
    assert "sustained_tilt" in result.violations


@pytest.mark.parametrize(
    "change",
    [
        {"finite": False},
        {"terminated": True, "termination_reason": "fell_over"},
        {"forbidden_ground_contact": True},
        {"steady_walk_duration_s": 3.99},
        {"completed_steps": 5},
        {"completed_horizon_s": 59.98},
        {"physical_slip_p95_m_s": None},
        {"physical_slip_samples": 99},
        {"physical_slip_force_coverage": 0.949},
    ],
)
def test_final_acceptance_fails_unsafe_or_insufficient_trials(change: dict[str, object]) -> None:
    result = assess_final_trial(replace(_passing(), **change), FinalAcceptanceCriteria())

    assert result.accepted is False


def test_final_criteria_loader_rejects_unknown_or_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "criteria.json"
    payload = {"schema_version": 2, **FinalAcceptanceCriteria().to_dict()}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_final_acceptance_criteria(path) == FinalAcceptanceCriteria()

    payload["surprise"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        load_final_acceptance_criteria(path)
