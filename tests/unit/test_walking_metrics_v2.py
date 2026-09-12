from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from g1_mjlab.gait_evaluation.diagnostic_plot import render_measurement_svg
from g1_mjlab.gait_evaluation.walking_v2 import (
    PhysicsTraceV2,
    TraceMetadataV2,
    WalkingCriteriaV2,
    WalkingTraceV2,
    evaluate_walking_v2,
    load_trace_v2,
    load_walking_criteria_v2,
    save_trace_v2,
)


def synthetic_trace(*, frames: int = 500, yaw_rad: float = 0.0) -> WalkingTraceV2:
    dt = 0.02
    time = np.arange(frames) * dt
    moving = (time >= 1.5) & (time < 8.0)
    requested = np.zeros((frames, 3))
    applied = np.zeros((frames, 3))
    requested[moving, 0] = 0.6
    applied[moving, 0] = 0.6
    phase = np.mod(np.maximum(time - 1.5, 0.0), 1.0)
    left = moving & (phase < 0.45)
    right = moving & (phase >= 0.5) & (phase < 0.95)
    sole = np.zeros((frames, 2, 3))
    heading = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
    for index in range(frames):
        half_cycle = int(max(time[index] - 1.5, 0.0) // 0.5)
        sole[index, 0] = heading * (0.6 * ((half_cycle + 1) // 2))
        sole[index, 1] = heading * (0.3 + 0.6 * (half_cycle // 2))
    quaternion = np.tile([np.cos(yaw_rad / 2), 0.0, 0.0, np.sin(yaw_rad / 2)], (frames, 1))
    arrays: dict[str, np.ndarray] = {
        "time_s": time,
        "requested_command": requested,
        "applied_command": applied,
        "root_position_w": np.column_stack((0.6 * time, np.zeros(frames), np.full(frames, 0.75))),
        "root_quaternion_wxyz": quaternion,
        "root_lin_vel_w": np.column_stack(
            (0.6 * np.cos(yaw_rad) * moving, 0.6 * np.sin(yaw_rad) * moving, np.zeros(frames))
        ),
        "root_ang_vel_w": np.zeros((frames, 3)),
        "phase": phase,
        "blend": moving.astype(float),
        "expected_contact": np.column_stack((left, right)),
        "joint_pos": np.zeros((frames, 29)),
        "joint_vel": np.zeros((frames, 29)),
        "raw_action": np.zeros((frames, 29)),
        "applied_action": np.zeros((frames, 29)),
        "q_target": np.zeros((frames, 29)),
        "actuator_torque": np.zeros((frames, 29)),
        "ankle_position_w": sole.copy(),
        "sole_position_w": sole,
        "sole_quaternion_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (frames, 2, 1)),
        "raw_contact": np.column_stack((left, right)),
        "debounced_contact": np.column_stack((left, right)),
        "normal_force_n": 100.0 * np.column_stack((left, right)),
        "stance_age_s": np.zeros((frames, 2)),
        "swing_age_s": np.zeros((frames, 2)),
        "expected_foot_position_heading": np.zeros((frames, 2, 3)),
        "swing_peak_height_m": np.full((frames, 2), 0.04),
        "finite": np.ones(frames, dtype=bool),
        "terminated": np.zeros(frames, dtype=bool),
        "truncated": np.zeros(frames, dtype=bool),
        "reset_counter": np.zeros(frames, dtype=np.int64),
    }
    for foot in range(2):
        age = 0.0
        prior = False
        for index, contact in enumerate(arrays["debounced_contact"][:, foot]):
            age = age + dt if bool(contact) == prior else 0.0
            arrays["stance_age_s"][index, foot] = age if contact else 0.0
            arrays["swing_age_s"][index, foot] = age if not contact else 0.0
            prior = bool(contact)
    metadata = TraceMetadataV2(
        schema_version=2,
        checkpoint_sha256="a" * 64,
        source_sha256="b" * 64,
        controller_sha256="c" * 64,
        reference_sha256="d" * 64,
        scenario_sha256="e" * 64,
        control_dt=dt,
        physics_dt=0.005,
        seed=10042,
        initialization="standing",
        terminated=False,
        reason=None,
        completed_horizon_s=frames * dt,
        field_definitions={name: "synthetic test field" for name in arrays},
        initial_state_sha256="f" * 64,
    )
    return WalkingTraceV2(metadata, arrays)


def physics_trace(speed_m_s: float, frames: int = 2000) -> PhysicsTraceV2:
    denominator = np.full((frames, 2), 100.0)
    numerator = denominator * speed_m_s**2
    return PhysicsTraceV2(
        supported=True,
        reason=None,
        arrays={
            "time_s": np.arange(frames) * 0.005,
            "contact": np.ones((frames, 2), dtype=bool),
            "normal_force_n": denominator,
            "tangential_speed_square_numerator": numerator,
            "contact_force_denominator": denominator,
            "contact_count": np.ones((frames, 2), dtype=np.int64),
            "lowest_sole_clearance_m": np.zeros((frames, 2)),
            "ankle_position_w": np.zeros((frames, 2, 3)),
            "sole_position_w": np.zeros((frames, 2, 3)),
        },
    )


def test_trace_v2_round_trip_rejects_object_and_unknown_arrays(tmp_path: Path) -> None:
    trace = synthetic_trace()
    trace_path = tmp_path / "trace-000.npz"
    metadata_path = tmp_path / "trace-000.metadata.json"
    save_trace_v2(trace, trace_path, metadata_path)
    loaded = load_trace_v2(trace_path, metadata_path)
    np.testing.assert_array_equal(loaded.arrays["joint_pos"], trace.arrays["joint_pos"])

    broken = dict(trace.arrays)
    broken["unknown"] = np.zeros(500)
    with pytest.raises(ValueError, match="fields"):
        WalkingTraceV2(trace.metadata, broken).validate()
    broken = dict(trace.arrays)
    broken["joint_pos"] = broken["joint_pos"].astype(object)
    with pytest.raises(ValueError, match="object"):
        WalkingTraceV2(trace.metadata, broken).validate()


def test_repository_evaluation_v2_profile_is_strict() -> None:
    criteria = load_walking_criteria_v2(Path("configs/walking-v1/evaluation-v2.json"))
    assert criteria.minimum_swing_duration_s == 0.12
    assert criteria.maximum_physical_slip_rms_m_s == 0.12


def test_ideal_alternating_gait_has_distinct_step_and_stride_metrics() -> None:
    expected = json.loads(Path("tests/fixtures/walking-v2-ideal.json").read_text(encoding="utf-8"))
    result = evaluate_walking_v2(synthetic_trace(), WalkingCriteriaV2(), physics_trace(0.0))
    assert result.insufficient_evidence is False
    assert result.step_length_median_m == pytest.approx(expected["step_length_m"], abs=1e-6)
    assert result.stride_length_median_m == pytest.approx(expected["stride_length_m"], abs=1e-6)
    assert result.alternation_ratio == expected["alternation_ratio"]
    assert result.physical_slip_rms_m_s == expected["physical_slip_rms_m_s"]


def test_force_weighted_contact_slip_distinguishes_rolling_ankle_proxy() -> None:
    trace = synthetic_trace()
    rolling = dict(trace.arrays)
    rolling["ankle_position_w"] = rolling["ankle_position_w"].copy()
    rolling["ankle_position_w"][:, :, 0] += np.arange(trace.frame_count)[:, None] * 0.004
    result = evaluate_walking_v2(
        WalkingTraceV2(trace.metadata, rolling), WalkingCriteriaV2(), physics_trace(0.0)
    )
    assert result.physical_slip_rms_m_s == 0.0
    assert result.ankle_speed_proxy_rms_m_s is not None
    assert result.ankle_speed_proxy_rms_m_s > 0.15

    sliding = evaluate_walking_v2(trace, WalkingCriteriaV2(), physics_trace(0.2))
    assert sliding.physical_slip_rms_m_s == pytest.approx(0.2)

    unequal = physics_trace(0.0)
    unequal.arrays["contact_force_denominator"][:, 0] = 300.0
    unequal.arrays["contact_force_denominator"][:, 1] = 100.0
    unequal.arrays["tangential_speed_square_numerator"][:, 0] = 300.0 * 0.1**2
    unequal.arrays["tangential_speed_square_numerator"][:, 1] = 100.0 * 0.3**2
    weighted = evaluate_walking_v2(trace, WalkingCriteriaV2(), unequal)
    assert weighted.physical_slip_rms_m_s == pytest.approx(np.sqrt(0.03), abs=1e-9)


def test_heading_projection_handles_ninety_degree_yaw() -> None:
    result = evaluate_walking_v2(
        synthetic_trace(yaw_rad=np.pi / 2), WalkingCriteriaV2(), physics_trace(0.0)
    )
    assert result.step_length_median_m == pytest.approx(0.3, abs=1e-6)


def test_same_side_and_simultaneous_touchdowns_are_not_alternating_steps() -> None:
    trace = synthetic_trace()
    arrays = dict(trace.arrays)
    contacts = np.zeros_like(arrays["debounced_contact"])
    for index in (100, 130, 160):
        contacts[index : index + 10, 0] = True
    contacts[190:200] = True
    arrays["debounced_contact"] = contacts
    arrays["raw_contact"] = contacts
    arrays["swing_age_s"] = np.full_like(arrays["swing_age_s"], 0.2)
    result = evaluate_walking_v2(
        WalkingTraceV2(trace.metadata, arrays), WalkingCriteriaV2(), physics_trace(0.0)
    )
    assert result.same_side_repeat_count >= 1
    assert result.hop_event_count == 1
    assert result.alternation_ratio == 0.0


def test_missing_physics_or_final_stop_is_insufficient_evidence() -> None:
    trace = synthetic_trace(frames=300)
    result = evaluate_walking_v2(trace, WalkingCriteriaV2(), PhysicsTraceV2.unsupported("api"))
    assert result.physical_slip_rms_m_s is None
    assert result.insufficient_evidence
    assert "physical_slip_unsupported" in result.violations
    assert "missing_final_stop" in result.violations


def test_incomplete_horizon_and_too_few_steps_fail_function_gate() -> None:
    trace = synthetic_trace(frames=200)
    result = evaluate_walking_v2(
        trace, WalkingCriteriaV2(), physics_trace(0.0, frames=800), planned_horizon_s=10.0
    )
    assert not result.functional_passed
    assert result.insufficient_evidence
    assert "incomplete_horizon" in result.violations
    assert "insufficient_steps" in result.violations


def test_root_static_foot_cycling_does_not_hide_step_measurement() -> None:
    trace = synthetic_trace()
    arrays = dict(trace.arrays)
    arrays["root_position_w"] = arrays["root_position_w"].copy()
    arrays["root_position_w"][:, :2] = 0.0
    arrays["root_lin_vel_w"] = np.zeros_like(arrays["root_lin_vel_w"])
    result = evaluate_walking_v2(
        WalkingTraceV2(trace.metadata, arrays), WalkingCriteriaV2(), physics_trace(0.0)
    )
    assert result.step_length_median_m == pytest.approx(0.3, abs=1e-6)
    assert "command_tracking" in result.violations


def test_nonfinite_required_data_is_rejected() -> None:
    trace = synthetic_trace()
    arrays = dict(trace.arrays)
    arrays["root_position_w"] = arrays["root_position_w"].copy()
    arrays["root_position_w"][10, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_walking_v2(WalkingTraceV2(trace.metadata, arrays), WalkingCriteriaV2())


def test_measurement_plot_is_dependency_free_and_labels_physical_slip(tmp_path: Path) -> None:
    output = tmp_path / "diagnostic-000.svg"
    render_measurement_svg(synthetic_trace(), physics_trace(0.2), output)
    content = output.read_text(encoding="utf-8")
    assert "Contact and normal force" in content
    assert "Physical contact slip" in content
    assert "Target tracking" in content
