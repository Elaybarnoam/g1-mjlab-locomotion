"""Walking evaluation schema 2 with physical-contact gait metrics."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.generic]

CONTROL_FIELDS: dict[str, tuple[int, ...]] = {
    "time_s": (),
    "requested_command": (3,),
    "applied_command": (3,),
    "root_position_w": (3,),
    "root_quaternion_wxyz": (4,),
    "root_lin_vel_w": (3,),
    "root_ang_vel_w": (3,),
    "phase": (),
    "blend": (),
    "expected_contact": (2,),
    "joint_pos": (29,),
    "joint_vel": (29,),
    "raw_action": (29,),
    "applied_action": (29,),
    "q_target": (29,),
    "actuator_torque": (29,),
    "ankle_position_w": (2, 3),
    "sole_position_w": (2, 3),
    "sole_quaternion_wxyz": (2, 4),
    "raw_contact": (2,),
    "debounced_contact": (2,),
    "normal_force_n": (2,),
    "stance_age_s": (2,),
    "swing_age_s": (2,),
    "expected_foot_position_heading": (2, 3),
    "swing_peak_height_m": (2,),
    "finite": (),
    "terminated": (),
    "truncated": (),
    "reset_counter": (),
}

PHYSICS_FIELDS: dict[str, tuple[int, ...]] = {
    "time_s": (),
    "contact": (2,),
    "normal_force_n": (2,),
    "tangential_speed_square_numerator": (2,),
    "contact_force_denominator": (2,),
    "contact_count": (2,),
    "lowest_sole_clearance_m": (2,),
    "ankle_position_w": (2, 3),
    "sole_position_w": (2, 3),
}


def _strict_dataclass(cls: type[Any], raw: dict[str, Any]) -> Any:
    expected = {field.name for field in fields(cls)}
    if set(raw) != expected:
        raise ValueError(f"{cls.__name__} fields do not match schema")
    return cls(**raw)


@dataclass(frozen=True)
class TraceMetadataV2:
    schema_version: int
    checkpoint_sha256: str
    source_sha256: str
    controller_sha256: str
    reference_sha256: str
    scenario_sha256: str
    control_dt: float
    physics_dt: float
    seed: int
    initialization: str
    terminated: bool
    reason: str | None
    completed_horizon_s: float
    field_definitions: dict[str, str]
    initial_state_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("unsupported trace metadata schema")
        for value in (
            self.checkpoint_sha256,
            self.source_sha256,
            self.controller_sha256,
            self.reference_sha256,
            self.scenario_sha256,
            self.initial_state_sha256,
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("trace identities must be lowercase SHA-256 values")
        if (
            not math.isfinite(self.control_dt)
            or not math.isfinite(self.physics_dt)
            or self.control_dt <= 0
            or self.physics_dt <= 0
            or self.control_dt < self.physics_dt
        ):
            raise ValueError("trace time steps must be finite, positive, and ordered")
        if self.seed < 0 or self.initialization not in {"standing", "reference", "reference-fixed"}:
            raise ValueError("invalid trace seed or initialization")
        if not math.isfinite(self.completed_horizon_s) or self.completed_horizon_s <= 0:
            raise ValueError("completed horizon must be finite and positive")
        if set(self.field_definitions) != set(CONTROL_FIELDS):
            raise ValueError("field definitions do not match trace schema 2")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TraceMetadataV2:
        return cast(TraceMetadataV2, _strict_dataclass(cls, raw))


def _validate_arrays(arrays: dict[str, Array], schema: dict[str, tuple[int, ...]]) -> int:
    if set(arrays) != set(schema):
        raise ValueError("array fields do not match schema")
    count: int | None = None
    for name, suffix in schema.items():
        value = np.asarray(arrays[name])
        if value.dtype.hasobject:
            raise ValueError(f"{name} must not use object dtype")
        if value.ndim != len(suffix) + 1 or tuple(value.shape[1:]) != suffix:
            raise ValueError(f"{name} has invalid shape {value.shape}")
        if count is None:
            count = len(value)
        elif len(value) != count:
            raise ValueError("all arrays must have the same leading dimension")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    if count is None or count == 0:
        raise ValueError("trace arrays must not be empty")
    return count


@dataclass(frozen=True)
class WalkingTraceV2:
    metadata: TraceMetadataV2
    arrays: dict[str, Array]

    @property
    def frame_count(self) -> int:
        return len(self.arrays["time_s"])

    def validate(self) -> None:
        count = _validate_arrays(self.arrays, CONTROL_FIELDS)
        time = np.asarray(self.arrays["time_s"], dtype=float)
        if count > 1 and not np.allclose(np.diff(time), self.metadata.control_dt, atol=1e-8):
            raise ValueError("control trace time does not match control_dt")
        expected_horizon = count * self.metadata.control_dt
        if not math.isclose(
            self.metadata.completed_horizon_s, expected_horizon, rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError("completed horizon does not match the trace")


@dataclass(frozen=True)
class PhysicsTraceV2:
    supported: bool
    reason: str | None
    arrays: dict[str, Array]

    @classmethod
    def unsupported(cls, reason: str) -> PhysicsTraceV2:
        if not reason:
            raise ValueError("unsupported physics trace requires a reason")
        return cls(False, reason, {})

    def validate(self) -> None:
        if not self.supported:
            if not self.reason or self.arrays:
                raise ValueError("unsupported physics trace must contain only a reason")
            return
        if self.reason is not None:
            raise ValueError("supported physics trace cannot contain an unsupported reason")
        _validate_arrays(self.arrays, PHYSICS_FIELDS)


def save_trace_v2(trace: WalkingTraceV2, path: Path, metadata_path: Path) -> None:
    trace.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **trace.arrays)  # type: ignore[arg-type]
    temporary.replace(path)
    metadata_path.write_text(
        json.dumps(asdict(trace.metadata), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_trace_v2(path: Path, metadata_path: Path) -> WalkingTraceV2:
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("trace metadata must be an object")
    metadata = TraceMetadataV2.from_dict(raw)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    result = WalkingTraceV2(metadata, arrays)
    result.validate()
    return result


@dataclass(frozen=True)
class WalkingCriteriaV2:
    command_threshold_m_s: float = 0.15
    command_settle_delta_m_s: float = 0.01
    command_settle_duration_s: float = 1.0
    stop_settle_duration_s: float = 0.5
    minimum_stop_samples_s: float = 1.0
    minimum_swing_duration_s: float = 0.12
    minimum_swing_clearance_m: float = 0.025
    minimum_walk_duration_s: float = 4.0
    minimum_completed_steps: int = 6
    maximum_command_rms_m_s: float = 0.25
    maximum_stop_rms_m_s: float = 0.10
    maximum_physical_slip_rms_m_s: float = 0.12
    minimum_step_length_m: float = 0.18
    maximum_step_length_m: float = 0.65
    maximum_tiny_step_fraction: float = 0.20
    minimum_cadence_steps_s: float = 0.60
    maximum_cadence_steps_s: float = 3.0
    minimum_alternation_ratio: float = 0.80
    maximum_raw_transition_rate_s: float = 6.0
    minimum_pelvis_height_m: float = 0.62
    maximum_torso_tilt_rad: float = 0.35
    maximum_step_asymmetry: float = 0.20
    maximum_hopping_fraction: float = 0.05

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value < 0 for value in vars(self).values()):
            raise ValueError("walking criteria must be finite and nonnegative")


def load_walking_criteria_v2(path: Path) -> WalkingCriteriaV2:
    raw_value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict) or raw_value.pop("schema_version", None) != 2:
        raise ValueError("unsupported walking criteria schema")
    expected = {field.name for field in fields(WalkingCriteriaV2)}
    if set(raw_value) != expected:
        raise ValueError("walking criteria fields do not match schema")
    return WalkingCriteriaV2(**raw_value)


@dataclass(frozen=True)
class StopSegmentResult:
    start_time_s: float
    eligible_samples: int
    speed_rms_m_s: float | None
    passed: bool


@dataclass(frozen=True)
class WalkingSummaryV2:
    schema_version: int
    functional_passed: bool
    style_passed: bool
    insufficient_evidence: bool
    violations: tuple[str, ...]
    completed_horizon_s: float
    planned_horizon_s: float
    initial_state_sha256: str
    command_rms_m_s: float | None
    stop_segments: tuple[StopSegmentResult, ...]
    completed_steps: int
    step_length_median_m: float | None
    stride_length_median_m: float | None
    tiny_step_fraction: float | None
    step_asymmetry: float | None
    cadence_steps_s: float | None
    alternation_ratio: float | None
    same_side_repeat_count: int
    hop_event_count: int
    raw_transition_rate_s: float | None
    physical_slip_rms_m_s: float | None
    physical_slip_p95_m_s: float | None
    physical_slip_samples: int
    physical_slip_force_coverage: float | None
    ankle_speed_proxy_rms_m_s: float | None
    hopping_fraction: float | None
    minimum_pelvis_height_m: float | None
    maximum_torso_tilt_rad: float | None


def _yaw_heading(quaternion: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.column_stack((np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)))


def _settled_walk_mask(
    command: npt.NDArray[np.float64],
    blend: npt.NDArray[np.float64],
    criteria: WalkingCriteriaV2,
    dt: float,
) -> npt.NDArray[np.bool_]:
    changes = np.linalg.norm(np.diff(command, axis=0, prepend=command[:1]), axis=1)
    stable = changes <= criteria.command_settle_delta_m_s
    frames = max(1, math.ceil(criteria.command_settle_duration_s / dt))
    run = 0
    settled = np.zeros(len(command), dtype=bool)
    for index, value in enumerate(stable):
        run = run + 1 if value else 0
        settled[index] = run >= frames
    return settled & (command[:, 0] >= criteria.command_threshold_m_s) & (blend >= 0.9)


def _touchdowns(
    contact: npt.NDArray[np.bool_],
    swing_age: npt.NDArray[np.float64],
    swing_peak: npt.NDArray[np.float64],
    eligible: npt.NDArray[np.bool_],
    criteria: WalkingCriteriaV2,
) -> list[tuple[int, int]]:
    prior = np.vstack((np.zeros((1, 2), dtype=bool), contact[:-1]))
    candidates = np.argwhere(contact & ~prior)
    result: list[tuple[int, int]] = []
    for index, foot in candidates:
        before = max(0, int(index) - 1)
        if (
            eligible[index]
            and swing_age[before, foot] + 1e-6 >= criteria.minimum_swing_duration_s
            and swing_peak[index, foot] >= criteria.minimum_swing_clearance_m
        ):
            result.append((int(index), int(foot)))
    return result


def _stop_segments(
    applied: npt.NDArray[np.float64],
    speed: npt.NDArray[np.float64],
    dt: float,
    criteria: WalkingCriteriaV2,
) -> tuple[StopSegmentResult, ...]:
    moving = applied[:, 0] >= criteria.command_threshold_m_s
    stopped = np.linalg.norm(applied, axis=1) <= 1e-9
    starts = np.flatnonzero(stopped & ~np.concatenate(([False], stopped[:-1])))
    results: list[StopSegmentResult] = []
    settle = math.ceil(criteria.stop_settle_duration_s / dt)
    minimum = math.ceil(criteria.minimum_stop_samples_s / dt)
    for start in starts:
        if not np.any(moving[:start]):
            continue
        end = int(start)
        while end < len(stopped) and stopped[end]:
            end += 1
        sample = speed[int(start) + settle : end]
        rms = float(np.sqrt(np.mean(np.square(sample)))) if len(sample) >= minimum else None
        results.append(
            StopSegmentResult(
                start_time_s=float(start * dt),
                eligible_samples=len(sample),
                speed_rms_m_s=rms,
                passed=rms is not None and rms <= criteria.maximum_stop_rms_m_s,
            )
        )
    return tuple(results)


def evaluate_walking_v2(
    trace: WalkingTraceV2,
    criteria: WalkingCriteriaV2,
    physics: PhysicsTraceV2 | None = None,
    *,
    planned_horizon_s: float | None = None,
) -> WalkingSummaryV2:
    """Reduce one pre-reset trace without substituting proxy measurements."""
    trace.validate()
    if physics is not None:
        physics.validate()
    arrays = trace.arrays
    dt = trace.metadata.control_dt
    applied = np.asarray(arrays["applied_command"], dtype=float)
    heading = _yaw_heading(np.asarray(arrays["root_quaternion_wxyz"], dtype=float))
    velocity = np.asarray(arrays["root_lin_vel_w"], dtype=float)
    forward_speed = np.sum(velocity * heading, axis=1)
    eligible = _settled_walk_mask(applied, np.asarray(arrays["blend"], dtype=float), criteria, dt)
    command_rms = (
        float(np.sqrt(np.mean(np.square(forward_speed[eligible] - applied[eligible, 0]))))
        if np.any(eligible)
        else None
    )
    contact = np.asarray(arrays["debounced_contact"], dtype=bool)
    touchdowns = _touchdowns(
        contact,
        np.asarray(arrays["swing_age_s"], dtype=float),
        np.asarray(arrays["swing_peak_height_m"], dtype=float),
        eligible,
        criteria,
    )
    grouped: list[list[tuple[int, int]]] = []
    for event in touchdowns:
        if grouped and event[0] == grouped[-1][0][0]:
            grouped[-1].append(event)
        else:
            grouped.append([event])
    hop_events = sum(len(group) == 2 for group in grouped)
    singles = [group[0] for group in grouped if len(group) == 1]
    sole = np.asarray(arrays["sole_position_w"], dtype=float)
    step_lengths: list[float] = []
    stride_lengths: list[float] = []
    side_steps: dict[int, list[float]] = {0: [], 1: []}
    same_side = 0
    alternating = 0
    for previous, current in zip(singles, singles[1:], strict=False):
        previous_index, previous_foot = previous
        current_index, current_foot = current
        if previous_foot == current_foot:
            same_side += 1
            continue
        alternating += 1
        length = float(
            np.dot(
                sole[current_index, current_foot] - sole[previous_index, previous_foot],
                heading[previous_index],
            )
        )
        step_lengths.append(length)
        side_steps[current_foot].append(length)
    for foot in range(2):
        foot_events = [(index, side) for index, side in singles if side == foot]
        for previous, current in zip(foot_events, foot_events[1:], strict=False):
            stride_lengths.append(
                float(
                    np.dot(sole[current[0], foot] - sole[previous[0], foot], heading[previous[0]])
                )
            )
    pair_count = max(0, len(singles) - 1)
    alternation_ratio = alternating / pair_count if pair_count else None
    steady_seconds = float(np.count_nonzero(eligible) * dt)
    cadence = alternating / steady_seconds if steady_seconds else None
    means = [float(np.mean(side_steps[foot])) if side_steps[foot] else None for foot in range(2)]
    asymmetry = None
    if all(len(side_steps[foot]) >= 3 for foot in range(2)):
        assert means[0] is not None and means[1] is not None
        asymmetry = abs(means[0] - means[1]) / max(0.5 * (means[0] + means[1]), 1e-6)
    tiny_fraction = (
        sum(length < criteria.minimum_step_length_m for length in step_lengths) / len(step_lengths)
        if step_lengths
        else None
    )

    raw = np.asarray(arrays["raw_contact"], dtype=bool)
    transition_eligible = eligible[1:] & eligible[:-1]
    raw_count = int(np.count_nonzero(np.diff(raw.astype(np.int8), axis=0)[transition_eligible]))
    transition_rate = raw_count / steady_seconds if steady_seconds else None
    ankle = np.asarray(arrays["ankle_position_w"], dtype=float)
    ankle_velocity = np.diff(ankle, axis=0) / dt
    stance = contact[1:] & contact[:-1] & eligible[1:, None]
    ankle_squares = np.sum(np.square(ankle_velocity[:, :, :2]), axis=2)[stance]
    ankle_proxy = float(np.sqrt(np.mean(ankle_squares))) if len(ankle_squares) else None

    slip_rms: float | None = None
    slip_p95: float | None = None
    slip_samples = 0
    force_coverage: float | None = None
    if physics is not None and physics.supported:
        numerator = np.asarray(physics.arrays["tangential_speed_square_numerator"], dtype=float)
        denominator = np.asarray(physics.arrays["contact_force_denominator"], dtype=float)
        physics_time = np.asarray(physics.arrays["time_s"], dtype=float)
        control_index = np.minimum((physics_time / dt).astype(int), len(eligible) - 1)
        physics_eligible = eligible[control_index]
        sample_numerator = np.sum(np.maximum(numerator, 0.0), axis=1)
        sample_denominator = np.sum(np.maximum(denominator, 0.0), axis=1)
        valid = (sample_denominator > 0) & physics_eligible
        speed_squares = np.maximum(sample_numerator[valid] / sample_denominator[valid], 0.0)
        speeds = np.sqrt(speed_squares)
        slip_samples = len(speeds)
        if slip_samples:
            slip_rms = float(np.sqrt(np.mean(speed_squares)))
            slip_p95 = float(np.percentile(speeds, 95))
            eligible_force = float(np.sum(sample_denominator[physics_eligible]))
            force_coverage = float(np.sum(sample_denominator[valid]) / eligible_force)

    both_air = ~contact[:, 0] & ~contact[:, 1] & eligible
    minimum_air = math.ceil(0.06 / dt)
    qualifying_air = np.zeros_like(both_air)
    start = 0
    while start < len(both_air):
        end = start + 1
        while end < len(both_air) and both_air[end] == both_air[start]:
            end += 1
        if both_air[start] and end - start >= minimum_air:
            qualifying_air[start:end] = True
        start = end
    hopping_fraction = (
        float(np.count_nonzero(qualifying_air) / np.count_nonzero(eligible))
        if np.any(eligible)
        else None
    )
    stops = _stop_segments(applied, forward_speed, dt, criteria)

    violations: set[str] = set()
    insufficient = False
    resolved_horizon_s = (
        trace.metadata.completed_horizon_s if planned_horizon_s is None else planned_horizon_s
    )
    if trace.metadata.completed_horizon_s + 0.5 * dt < resolved_horizon_s:
        violations.add("incomplete_horizon")
        insufficient = True
    if not np.any(eligible):
        violations.add("insufficient_steady_walk")
        insufficient = True
    if steady_seconds < criteria.minimum_walk_duration_s:
        violations.add("insufficient_walk_duration")
        insufficient = True
    if physics is None or not physics.supported or slip_rms is None:
        violations.add("physical_slip_unsupported")
        insufficient = True
    if not stops:
        violations.add("missing_final_stop")
        insufficient = True
    elif any(not stop.passed for stop in stops):
        violations.add("failed_stop")
        insufficient = insufficient or any(stop.speed_rms_m_s is None for stop in stops)
    if trace.metadata.terminated or np.any(arrays["terminated"]):
        violations.add("fall")
    if command_rms is not None and command_rms > criteria.maximum_command_rms_m_s:
        violations.add("command_tracking")
    step_median = float(np.median(step_lengths)) if step_lengths else None
    stride_median = float(np.median(stride_lengths)) if stride_lengths else None
    if alternating < criteria.minimum_completed_steps:
        violations.add("insufficient_steps")
    if step_median is None or not (
        criteria.minimum_step_length_m <= step_median <= criteria.maximum_step_length_m
    ):
        violations.add("step_length")
    if tiny_fraction is None or tiny_fraction > criteria.maximum_tiny_step_fraction:
        violations.add("shuffling")
    if (
        cadence is None
        or not criteria.minimum_cadence_steps_s <= cadence <= criteria.maximum_cadence_steps_s
    ):
        violations.add("cadence")
    if alternation_ratio is None or alternation_ratio < criteria.minimum_alternation_ratio:
        violations.add("nonalternating_steps")
    if transition_rate is not None and transition_rate > criteria.maximum_raw_transition_rate_s:
        violations.add("contact_chatter")
    if slip_rms is not None and slip_rms > criteria.maximum_physical_slip_rms_m_s:
        violations.add("physical_slip")
    pelvis_height = np.asarray(arrays["root_position_w"], dtype=float)[:, 2]
    walk_heights = pelvis_height[eligible]
    minimum_height = float(np.min(walk_heights)) if len(walk_heights) else None
    quaternion = np.asarray(arrays["root_quaternion_wxyz"], dtype=float)
    upright_z = 1 - 2 * (np.square(quaternion[:, 1]) + np.square(quaternion[:, 2]))
    walk_tilts = np.arccos(np.clip(upright_z[eligible], -1.0, 1.0))
    maximum_tilt = float(np.max(walk_tilts)) if len(walk_tilts) else None
    if minimum_height is not None and minimum_height < criteria.minimum_pelvis_height_m:
        violations.add("crouch")
    if maximum_tilt is not None and maximum_tilt > criteria.maximum_torso_tilt_rad:
        violations.add("torso_tilt")
    if asymmetry is not None and asymmetry > criteria.maximum_step_asymmetry:
        violations.add("step_asymmetry")
    if hopping_fraction is not None and hopping_fraction > criteria.maximum_hopping_fraction:
        violations.add("hopping")
    functional_terms = {
        "fall",
        "failed_stop",
        "command_tracking",
        "insufficient_steady_walk",
        "insufficient_walk_duration",
        "incomplete_horizon",
        "insufficient_steps",
        "crouch",
    }
    style_terms = {
        "step_length",
        "cadence",
        "nonalternating_steps",
        "contact_chatter",
        "physical_slip",
        "shuffling",
        "torso_tilt",
        "step_asymmetry",
        "hopping",
    }
    return WalkingSummaryV2(
        schema_version=2,
        functional_passed=not bool(violations & functional_terms) and not insufficient,
        style_passed=not bool(violations & style_terms) and not insufficient,
        insufficient_evidence=insufficient,
        violations=tuple(sorted(violations)),
        completed_horizon_s=trace.metadata.completed_horizon_s,
        planned_horizon_s=resolved_horizon_s,
        initial_state_sha256=trace.metadata.initial_state_sha256,
        command_rms_m_s=command_rms,
        stop_segments=stops,
        completed_steps=alternating,
        step_length_median_m=step_median,
        stride_length_median_m=stride_median,
        tiny_step_fraction=tiny_fraction,
        step_asymmetry=asymmetry,
        cadence_steps_s=cadence,
        alternation_ratio=alternation_ratio,
        same_side_repeat_count=same_side,
        hop_event_count=hop_events,
        raw_transition_rate_s=transition_rate,
        physical_slip_rms_m_s=slip_rms,
        physical_slip_p95_m_s=slip_p95,
        physical_slip_samples=slip_samples,
        physical_slip_force_coverage=force_coverage,
        ankle_speed_proxy_rms_m_s=ankle_proxy,
        hopping_fraction=hopping_fraction,
        minimum_pelvis_height_m=minimum_height,
        maximum_torso_tilt_rad=maximum_tilt,
    )
