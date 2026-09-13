"""Strict, versioned final walking acceptance independent of reported pass flags."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Self

import numpy as np

from .walking_v2 import PhysicsTraceV2, WalkingTraceV2


@dataclass(frozen=True, slots=True)
class FinalAcceptanceCriteria:
    minimum_overall_passes: int = 95
    minimum_stratum_passes: int = 23
    trials_per_stratum: int = 25
    horizon_seconds: float = 60.0
    minimum_steady_walk_duration_s: float = 4.0
    minimum_completed_steps: int = 6
    command_threshold_m_s: float = 0.15
    command_settle_delta_m_s: float = 0.01
    command_settle_duration_s: float = 1.0
    stop_settle_duration_s: float = 0.5
    minimum_stop_samples_s: float = 1.0
    minimum_swing_duration_s: float = 0.12
    minimum_swing_clearance_m: float = 0.025
    tiny_step_threshold_m: float = 0.18
    contact_event_tolerance_s: float = 0.08
    minimum_physical_slip_samples: int = 100
    minimum_physical_slip_force_coverage: float = 0.95
    steady_forward_tracking_rms_max_m_s: float = 0.10
    lateral_tracking_rms_max_m_s: float = 0.08
    heading_error_max_deg: float = 10.0
    pelvis_height_min_m: float = 0.62
    sustained_tilt_max_deg: float = 15.0
    sustained_tilt_interval_s: float = 0.20
    absolute_tilt_peak_max_deg: float = 20.0
    stopped_speed_rms_max_m_s: float = 0.10
    stopped_horizontal_drift_max_m: float = 0.20
    cadence_and_step_length_reference_tolerance_fraction: float = 0.20
    step_asymmetry_max_fraction: float = 0.15
    alternation_min_fraction: float = 0.90
    tiny_step_max_fraction: float = 0.20
    physical_slip_rms_max_m_s: float = 0.12
    physical_slip_p95_max_m_s: float = 0.15
    unmatched_event_max_fraction: float = 0.10
    bilateral_flight_max_fraction: float = 0.01

    def __post_init__(self) -> None:
        numeric = asdict(self)
        if any(
            isinstance(value, float) and (not math.isfinite(value) or value < 0)
            for value in numeric.values()
        ):
            raise ValueError("final acceptance criteria must be finite and non-negative")
        if (
            self.minimum_overall_passes <= 0
            or self.minimum_stratum_passes <= 0
            or self.trials_per_stratum <= 0
            or self.minimum_completed_steps <= 0
            or self.minimum_stratum_passes > self.trials_per_stratum
            or self.minimum_overall_passes > 4 * self.trials_per_stratum
        ):
            raise ValueError("final acceptance counts are invalid")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def load_final_acceptance_criteria(path: Path) -> FinalAcceptanceCriteria:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise ValueError("final acceptance criteria must use schema 2")
    values = {key: value for key, value in raw.items() if key != "schema_version"}
    expected = {field.name for field in fields(FinalAcceptanceCriteria)}
    if set(values) != expected:
        raise ValueError("final acceptance criteria fields do not match schema")
    return FinalAcceptanceCriteria(**values)


@dataclass(frozen=True, slots=True)
class FinalTrialMeasurements:
    completed_horizon_s: float
    planned_horizon_s: float
    finite: bool
    terminated: bool
    termination_reason: str | None
    forbidden_ground_contact: bool
    steady_walk_duration_s: float
    completed_steps: int
    steady_forward_tracking_rms_m_s: float | None
    lateral_tracking_rms_m_s: float | None
    heading_error_max_deg: float | None
    pelvis_height_min_m: float | None
    tilt_over_limit_max_duration_s: float | None
    absolute_tilt_peak_deg: float | None
    stopped_speed_rms_max_m_s: float | None
    stopped_horizontal_drift_max_m: float | None
    cadence_reference_error_max_fraction: float | None
    step_length_reference_error_max_fraction: float | None
    step_asymmetry_fraction: float | None
    alternation_fraction: float | None
    tiny_step_fraction: float | None
    physical_slip_rms_m_s: float | None
    physical_slip_p95_m_s: float | None
    physical_slip_samples: int
    physical_slip_force_coverage: float | None
    unmatched_event_fraction: float | None
    bilateral_flight_fraction: float | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        expected = {field.name for field in fields(cls)}
        if set(raw) != expected:
            raise ValueError("final trial measurement fields do not match schema")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FinalTrialAcceptance:
    accepted: bool
    violations: tuple[str, ...]
    measurements: dict[str, Any]
    criteria: dict[str, int | float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "violations": list(self.violations),
            "measurements": self.measurements,
            "criteria": self.criteria,
        }


@dataclass(frozen=True, slots=True)
class GaitTarget:
    cadence_steps_s: float
    step_length_m: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.cadence_steps_s)
            or not math.isfinite(self.step_length_m)
            or self.cadence_steps_s <= 0
            or self.step_length_m <= 0
        ):
            raise ValueError("gait targets must be finite and positive")


def _consecutive_mask(values: np.ndarray, frames: int) -> np.ndarray:
    result = np.zeros(len(values), dtype=bool)
    run = 0
    for index, value in enumerate(values):
        run = run + 1 if bool(value) else 0
        result[index] = run >= frames
    return result


def _yaw(quaternions: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _events(contact: np.ndarray, eligible: np.ndarray) -> list[tuple[int, int, bool]]:
    changes = contact[1:] != contact[:-1]
    return [
        (int(index + 1), int(foot), bool(contact[index + 1, foot]))
        for index, foot in np.argwhere(changes)
        if eligible[index] and eligible[index + 1]
    ]


def _unmatched_fraction(
    expected: list[tuple[int, int, bool]],
    actual: list[tuple[int, int, bool]],
    tolerance_frames: int,
) -> float:
    used: set[int] = set()
    matched = 0
    for expected_index, expected_foot, expected_edge in expected:
        candidates = [
            (abs(actual_index - expected_index), actual_index, index)
            for index, (actual_index, actual_foot, actual_edge) in enumerate(actual)
            if index not in used
            and actual_foot == expected_foot
            and actual_edge == expected_edge
            and abs(actual_index - expected_index) <= tolerance_frames
        ]
        if candidates:
            _, _, selected = min(candidates)
            used.add(selected)
            matched += 1
    total = len(expected) + len(actual)
    return (total - 2 * matched) / total if total else 0.0


def _maximum_true_duration(values: np.ndarray, dt: float) -> float:
    longest = 0
    run = 0
    for value in values:
        run = run + 1 if bool(value) else 0
        longest = max(longest, run)
    return longest * dt


def measure_final_trial(
    trace: WalkingTraceV2,
    physics: PhysicsTraceV2,
    criteria: FinalAcceptanceCriteria,
    targets: dict[float, GaitTarget],
    *,
    forbidden_ground_contact: bool,
    planned_horizon_s: float,
) -> FinalTrialMeasurements:
    """Derive final measurements from pre-reset traces using the frozen final windows."""
    trace.validate()
    physics.validate()
    if not physics.supported:
        raise ValueError("final acceptance requires physical contact-point slip telemetry")
    if not targets:
        raise ValueError("final acceptance requires at least one speed-dependent gait target")
    arrays = trace.arrays
    dt = trace.metadata.control_dt
    requested = np.asarray(arrays["requested_command"], dtype=float)
    applied = np.asarray(arrays["applied_command"], dtype=float)
    tracking_settled = (
        np.linalg.norm(requested - applied, axis=1) <= criteria.command_settle_delta_m_s
    )
    settled = _consecutive_mask(
        tracking_settled, max(1, math.ceil(criteria.command_settle_duration_s / dt))
    )
    eligible = (
        settled
        & (applied[:, 0] >= criteria.command_threshold_m_s)
        & (np.asarray(arrays["blend"], dtype=float) >= 0.9)
    )
    velocity = np.asarray(arrays["root_lin_vel_w"], dtype=float)
    yaw = _yaw(np.asarray(arrays["root_quaternion_wxyz"], dtype=float))
    heading = np.column_stack((np.cos(yaw), np.sin(yaw)))
    lateral = np.column_stack((-np.sin(yaw), np.cos(yaw)))
    forward_speed = np.sum(velocity[:, :2] * heading, axis=1)
    lateral_speed = np.sum(velocity[:, :2] * lateral, axis=1)

    forward_rms = (
        float(np.sqrt(np.mean(np.square(forward_speed[eligible] - applied[eligible, 0]))))
        if np.any(eligible)
        else None
    )
    lateral_rms = (
        float(np.sqrt(np.mean(np.square(lateral_speed[eligible] - applied[eligible, 1]))))
        if np.any(eligible)
        else None
    )
    desired_yaw = yaw[0] + np.concatenate(
        (np.zeros(1, dtype=float), np.cumsum(applied[:-1, 2]) * dt)
    )
    heading_error = float(np.max(np.abs(np.rad2deg(yaw - desired_yaw))))

    quaternion = np.asarray(arrays["root_quaternion_wxyz"], dtype=float)
    upright_z = 1 - 2 * (np.square(quaternion[:, 1]) + np.square(quaternion[:, 2]))
    tilt_deg = np.rad2deg(np.arccos(np.clip(upright_z, -1.0, 1.0)))
    tilt_duration = _maximum_true_duration(tilt_deg > criteria.sustained_tilt_max_deg, dt)
    root_position = np.asarray(arrays["root_position_w"], dtype=float)

    stopped = np.linalg.norm(applied, axis=1) <= 1e-9
    stop_starts = np.flatnonzero(stopped & ~np.concatenate(([False], stopped[:-1])))
    stop_speed_rms: list[float] = []
    stop_drift: list[float] = []
    settle_frames = math.ceil(criteria.stop_settle_duration_s / dt)
    minimum_stop_frames = math.ceil(criteria.minimum_stop_samples_s / dt)
    previously_moving = applied[:, 0] >= criteria.command_threshold_m_s
    for start in stop_starts:
        if not np.any(previously_moving[:start]):
            continue
        end = int(start)
        while end < len(stopped) and stopped[end]:
            end += 1
        first = int(start) + settle_frames
        if end - first < minimum_stop_frames:
            continue
        speeds = np.linalg.norm(velocity[first:end, :2], axis=1)
        stop_speed_rms.append(float(np.sqrt(np.mean(np.square(speeds)))))
        displacement = np.linalg.norm(
            root_position[first:end, :2] - root_position[first, :2], axis=1
        )
        stop_drift.append(float(np.max(displacement)))

    contact = np.asarray(arrays["debounced_contact"], dtype=bool)
    prior = np.vstack((np.zeros((1, 2), dtype=bool), contact[:-1]))
    touchdown_candidates = np.argwhere(contact & ~prior)
    swing_age = np.asarray(arrays["swing_age_s"], dtype=float)
    swing_peak = np.asarray(arrays["swing_peak_height_m"], dtype=float)
    touchdowns = [
        (int(index), int(foot))
        for index, foot in touchdown_candidates
        if eligible[index]
        and swing_age[max(0, int(index) - 1), foot] + 1e-9 >= criteria.minimum_swing_duration_s
        and swing_peak[index, foot] >= criteria.minimum_swing_clearance_m
    ]
    sole = np.asarray(arrays["sole_position_w"], dtype=float)
    step_records: list[tuple[int, float, int]] = []
    alternating = 0
    for previous, current in zip(touchdowns, touchdowns[1:], strict=False):
        if previous[1] == current[1]:
            continue
        alternating += 1
        index, foot = current
        length = float(
            np.dot(sole[index, foot, :2] - sole[previous[0], previous[1], :2], heading[index])
        )
        step_records.append((index, length, foot))
    pair_count = max(0, len(touchdowns) - 1)
    alternation = alternating / pair_count if pair_count else None
    tiny_fraction = (
        sum(length < criteria.tiny_step_threshold_m for _, length, _ in step_records)
        / len(step_records)
        if step_records
        else None
    )
    side_means = [
        np.mean([length for _, length, side in step_records if side == foot])
        for foot in range(2)
        if sum(side == foot for _, _, side in step_records) >= 3
    ]
    asymmetry = (
        float(abs(side_means[0] - side_means[1]) / max(0.5 * sum(side_means), 1e-6))
        if len(side_means) == 2
        else None
    )

    cadence_errors: list[float] = []
    step_errors: list[float] = []
    for speed, target in sorted(targets.items()):
        speed_mask = eligible & (np.abs(applied[:, 0] - speed) <= criteria.command_settle_delta_m_s)
        duration = float(np.count_nonzero(speed_mask) * dt)
        selected_steps = [record for record in step_records if speed_mask[record[0]]]
        if duration <= 0 or len(selected_steps) < 2:
            cadence_errors.append(math.inf)
            step_errors.append(math.inf)
            continue
        cadence = len(selected_steps) / duration
        step_median = float(np.median([record[1] for record in selected_steps]))
        cadence_errors.append(abs(cadence - target.cadence_steps_s) / target.cadence_steps_s)
        step_errors.append(abs(step_median - target.step_length_m) / target.step_length_m)

    expected_events = _events(np.asarray(arrays["expected_contact"], dtype=bool), eligible)
    actual_events = _events(contact, eligible)
    unmatched = _unmatched_fraction(
        expected_events,
        actual_events,
        max(0, round(criteria.contact_event_tolerance_s / dt)),
    )
    flight = ~contact[:, 0] & ~contact[:, 1] & eligible
    flight_fraction = (
        float(np.count_nonzero(flight) / np.count_nonzero(eligible)) if np.any(eligible) else None
    )

    numerator = np.asarray(physics.arrays["tangential_speed_square_numerator"], dtype=float)
    denominator = np.asarray(physics.arrays["contact_force_denominator"], dtype=float)
    physics_time = np.asarray(physics.arrays["time_s"], dtype=float)
    control_index = np.minimum((physics_time / dt).astype(int), len(eligible) - 1)
    sample_numerator = np.sum(np.maximum(numerator, 0.0), axis=1)
    sample_denominator = np.sum(np.maximum(denominator, 0.0), axis=1)
    valid_slip = eligible[control_index] & (sample_denominator > 0)
    speed_squares = np.maximum(sample_numerator[valid_slip] / sample_denominator[valid_slip], 0.0)
    slip_speeds = np.sqrt(speed_squares)
    eligible_force = float(np.sum(sample_denominator[eligible[control_index]]))
    slip_force_coverage = (
        float(np.sum(sample_denominator[valid_slip]) / eligible_force)
        if eligible_force > 0
        else None
    )
    return FinalTrialMeasurements(
        completed_horizon_s=trace.metadata.completed_horizon_s,
        planned_horizon_s=planned_horizon_s,
        finite=bool(np.asarray(arrays["finite"], dtype=bool).all()),
        terminated=bool(trace.metadata.terminated or np.asarray(arrays["terminated"]).any()),
        termination_reason=trace.metadata.reason,
        forbidden_ground_contact=forbidden_ground_contact,
        steady_walk_duration_s=float(np.count_nonzero(eligible) * dt),
        completed_steps=alternating,
        steady_forward_tracking_rms_m_s=forward_rms,
        lateral_tracking_rms_m_s=lateral_rms,
        heading_error_max_deg=heading_error,
        pelvis_height_min_m=float(np.min(root_position[:, 2])),
        tilt_over_limit_max_duration_s=tilt_duration,
        absolute_tilt_peak_deg=float(np.max(tilt_deg)),
        stopped_speed_rms_max_m_s=max(stop_speed_rms) if stop_speed_rms else None,
        stopped_horizontal_drift_max_m=max(stop_drift) if stop_drift else None,
        cadence_reference_error_max_fraction=max(cadence_errors) if cadence_errors else None,
        step_length_reference_error_max_fraction=max(step_errors) if step_errors else None,
        step_asymmetry_fraction=asymmetry,
        alternation_fraction=alternation,
        tiny_step_fraction=tiny_fraction,
        physical_slip_rms_m_s=(
            float(np.sqrt(np.mean(speed_squares))) if len(speed_squares) else None
        ),
        physical_slip_p95_m_s=(float(np.percentile(slip_speeds, 95)) if len(slip_speeds) else None),
        physical_slip_samples=len(slip_speeds),
        physical_slip_force_coverage=slip_force_coverage,
        unmatched_event_fraction=unmatched,
        bilateral_flight_fraction=flight_fraction,
    )


def assess_final_trial(
    measurement: FinalTrialMeasurements, criteria: FinalAcceptanceCriteria
) -> FinalTrialAcceptance:
    """Apply every final threshold; missing or non-finite measurements fail closed."""
    violations: set[str] = set()
    if not measurement.finite:
        violations.add("nonfinite")
    if measurement.terminated:
        violations.add("termination")
    if measurement.forbidden_ground_contact:
        violations.add("forbidden_ground_contact")
    if measurement.completed_horizon_s + 1e-9 < measurement.planned_horizon_s:
        violations.add("incomplete_horizon")
    if measurement.planned_horizon_s + 1e-9 < criteria.horizon_seconds:
        violations.add("short_planned_horizon")
    if measurement.steady_walk_duration_s < criteria.minimum_steady_walk_duration_s:
        violations.add("insufficient_walk_duration")
    if measurement.completed_steps < criteria.minimum_completed_steps:
        violations.add("insufficient_steps")
    if measurement.physical_slip_samples < criteria.minimum_physical_slip_samples:
        violations.add("insufficient_slip_samples")
    if (
        measurement.physical_slip_force_coverage is None
        or not math.isfinite(measurement.physical_slip_force_coverage)
        or measurement.physical_slip_force_coverage < criteria.minimum_physical_slip_force_coverage
    ):
        violations.add("insufficient_slip_force_coverage")

    upper = (
        (
            "steady_forward_tracking_rms_m_s",
            "forward_tracking",
            criteria.steady_forward_tracking_rms_max_m_s,
        ),
        ("lateral_tracking_rms_m_s", "lateral_tracking", criteria.lateral_tracking_rms_max_m_s),
        ("heading_error_max_deg", "heading", criteria.heading_error_max_deg),
        ("absolute_tilt_peak_deg", "peak_tilt", criteria.absolute_tilt_peak_max_deg),
        ("stopped_speed_rms_max_m_s", "stop_speed", criteria.stopped_speed_rms_max_m_s),
        ("stopped_horizontal_drift_max_m", "stop_drift", criteria.stopped_horizontal_drift_max_m),
        (
            "cadence_reference_error_max_fraction",
            "cadence",
            criteria.cadence_and_step_length_reference_tolerance_fraction,
        ),
        (
            "step_length_reference_error_max_fraction",
            "step_length",
            criteria.cadence_and_step_length_reference_tolerance_fraction,
        ),
        ("step_asymmetry_fraction", "step_asymmetry", criteria.step_asymmetry_max_fraction),
        ("tiny_step_fraction", "tiny_steps", criteria.tiny_step_max_fraction),
        ("physical_slip_rms_m_s", "slip_rms", criteria.physical_slip_rms_max_m_s),
        ("physical_slip_p95_m_s", "slip_p95", criteria.physical_slip_p95_max_m_s),
        ("unmatched_event_fraction", "contact_events", criteria.unmatched_event_max_fraction),
        ("bilateral_flight_fraction", "bilateral_flight", criteria.bilateral_flight_max_fraction),
    )
    for field_name, violation, maximum in upper:
        value = getattr(measurement, field_name)
        if value is None or not math.isfinite(value) or value > maximum:
            violations.add(violation)
    if (
        measurement.pelvis_height_min_m is None
        or not math.isfinite(measurement.pelvis_height_min_m)
        or measurement.pelvis_height_min_m < criteria.pelvis_height_min_m
    ):
        violations.add("pelvis_height")
    if (
        measurement.alternation_fraction is None
        or not math.isfinite(measurement.alternation_fraction)
        or measurement.alternation_fraction < criteria.alternation_min_fraction
    ):
        violations.add("alternation")
    if measurement.tilt_over_limit_max_duration_s is None or not math.isfinite(
        measurement.tilt_over_limit_max_duration_s
    ):
        violations.add("sustained_tilt")
    elif measurement.tilt_over_limit_max_duration_s + 1e-9 >= criteria.sustained_tilt_interval_s:
        violations.add("sustained_tilt")
    return FinalTrialAcceptance(
        accepted=not violations,
        violations=tuple(sorted(violations)),
        measurements=measurement.to_dict(),
        criteria=criteria.to_dict(),
    )
