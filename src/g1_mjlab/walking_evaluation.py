"""Simulator-independent functional and gait-style evaluation for walking-v1."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WalkingCriteria:
    """Frozen W07 development thresholds; final thresholds require owner review."""

    command_threshold_m_s: float = 0.15
    max_command_rms_m_s: float = 0.25
    minimum_walk_seconds: float = 4.0
    minimum_completed_steps: int = 6
    minimum_step_length_m: float = 0.18
    maximum_step_length_m: float = 0.65
    maximum_tiny_step_fraction: float = 0.20
    minimum_cadence_steps_s: float = 1.0
    maximum_cadence_steps_s: float = 3.0
    minimum_alternation_ratio: float = 0.80
    maximum_stance_slip_rms_m_s: float = 0.12
    minimum_pelvis_height_m: float = 0.62
    maximum_torso_tilt_rad: float = 0.35
    maximum_contact_transitions_s: float = 6.0
    contact_minimum_duration_s: float = 0.06
    stop_settling_seconds: float = 0.50
    maximum_stop_speed_rms_m_s: float = 0.10

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def load_walking_criteria(path: Path) -> WalkingCriteria:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.pop("schema_version", None) != 1:
        raise ValueError("unsupported walking criteria schema")
    expected = set(WalkingCriteria.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("walking criteria fields do not match schema")
    return WalkingCriteria(**raw)


@dataclass(frozen=True)
class WalkingTrace:
    dt: float
    command_speed: np.ndarray
    forward_speed: np.ndarray
    pelvis_height: np.ndarray
    torso_tilt_rad: np.ndarray
    left_foot_position: np.ndarray
    right_foot_position: np.ndarray
    left_contact: np.ndarray
    right_contact: np.ndarray
    finite: np.ndarray
    terminated: bool

    @property
    def frame_count(self) -> int:
        return len(self.command_speed)

    def validate(self) -> None:
        if not math.isfinite(self.dt) or self.dt <= 0 or self.frame_count == 0:
            raise ValueError("walking trace requires a finite positive dt and frames")
        vectors = (
            self.forward_speed,
            self.pelvis_height,
            self.torso_tilt_rad,
            self.left_contact,
            self.right_contact,
            self.finite,
        )
        if any(len(value) != self.frame_count for value in vectors):
            raise ValueError("walking trace vectors must have equal length")
        if self.left_foot_position.shape != (
            self.frame_count,
            3,
        ) or self.right_foot_position.shape != (
            self.frame_count,
            3,
        ):
            raise ValueError("foot positions must have shape (frames, 3)")


@dataclass(frozen=True)
class WalkingResult:
    functional_passed: bool
    style_passed: bool
    classification: str
    violations: tuple[str, ...]
    walk_seconds: float
    completed_steps: int
    command_rms_m_s: float | None
    step_length_median_m: float | None
    cadence_steps_s: float | None
    alternation_ratio: float | None
    tiny_step_fraction: float | None
    stance_slip_rms_m_s: float | None
    minimum_pelvis_height_m: float | None
    maximum_torso_tilt_rad: float | None
    contact_transitions_s: float
    stop_samples: int
    stop_speed_rms_m_s: float | None


def _debounce(contact: np.ndarray, minimum_frames: int) -> np.ndarray:
    source = np.asarray(contact, dtype=bool)
    result = np.zeros_like(source)
    start = 0
    while start < len(source):
        end = start + 1
        while end < len(source) and source[end] == source[start]:
            end += 1
        if source[start] and end - start >= minimum_frames:
            result[start:end] = True
        start = end
    return result


def _strike_indices(contact: np.ndarray) -> np.ndarray:
    return np.flatnonzero(contact & ~np.concatenate(([False], contact[:-1])))


def _stop_mask(command: np.ndarray, dt: float, settling_seconds: float) -> np.ndarray:
    stopped = np.abs(command) < 1e-9
    result = np.zeros_like(stopped)
    settle_frames = round(settling_seconds / dt)
    run_start: int | None = None
    for index, value in enumerate(stopped):
        if value and run_start is None:
            run_start = index
        elif not value:
            run_start = None
        if run_start is not None and index - run_start >= settle_frames:
            result[index] = True
    return result


def evaluate_walking_trace(trace: WalkingTrace, criteria: WalkingCriteria) -> WalkingResult:
    """Evaluate one pre-reset trace without treating survival as natural walking."""
    trace.validate()
    command = np.asarray(trace.command_speed, dtype=float)
    speed = np.asarray(trace.forward_speed, dtype=float)
    walk = command >= criteria.command_threshold_m_s
    walk_seconds = float(np.count_nonzero(walk) * trace.dt)
    minimum_frames = max(1, math.ceil(criteria.contact_minimum_duration_s / trace.dt))
    left = _debounce(trace.left_contact, minimum_frames)
    right = _debounce(trace.right_contact, minimum_frames)
    transition_mask = walk[1:] & walk[:-1]
    raw_transitions = np.count_nonzero(
        np.diff(trace.left_contact.astype(np.int8))[transition_mask]
    ) + np.count_nonzero(np.diff(trace.right_contact.astype(np.int8))[transition_mask])
    contact_transitions_s = float(raw_transitions / walk_seconds) if walk_seconds else 0.0

    events = [(int(index), "left") for index in _strike_indices(left) if walk[index]]
    events += [(int(index), "right") for index in _strike_indices(right) if walk[index]]
    events.sort()
    completed_steps = max(0, len(events) - 1)
    step_lengths: list[float] = []
    alternating = 0
    for previous, current in zip(events, events[1:], strict=False):
        previous_index, previous_side = previous
        current_index, current_side = current
        if previous_side != current_side:
            alternating += 1
        previous_position = (
            trace.left_foot_position[previous_index]
            if previous_side == "left"
            else trace.right_foot_position[previous_index]
        )
        current_position = (
            trace.left_foot_position[current_index]
            if current_side == "left"
            else trace.right_foot_position[current_index]
        )
        step_lengths.append(float(current_position[0] - previous_position[0]))

    command_rms = (
        float(np.sqrt(np.mean(np.square(speed[walk] - command[walk])))) if np.any(walk) else None
    )
    step_median = float(np.median(step_lengths)) if step_lengths else None
    cadence = completed_steps / walk_seconds if walk_seconds else None
    alternation = alternating / completed_steps if completed_steps else None
    tiny_fraction = (
        sum(length < criteria.minimum_step_length_m for length in step_lengths) / len(step_lengths)
        if step_lengths
        else None
    )

    slip_squares: list[np.ndarray] = []
    for position, contact in (
        (trace.left_foot_position, left),
        (trace.right_foot_position, right),
    ):
        velocity = np.diff(position[:, :2], axis=0) / trace.dt
        stance = contact[1:] & contact[:-1] & walk[1:]
        if np.any(stance):
            slip_squares.append(np.sum(np.square(velocity[stance]), axis=1))
    stance_slip = float(np.sqrt(np.mean(np.concatenate(slip_squares)))) if slip_squares else None
    walk_heights = trace.pelvis_height[walk]
    walk_tilts = trace.torso_tilt_rad[walk]
    minimum_height = float(np.min(walk_heights)) if len(walk_heights) else None
    maximum_tilt = float(np.max(walk_tilts)) if len(walk_tilts) else None
    stop = _stop_mask(command, trace.dt, criteria.stop_settling_seconds)
    stop_rms = float(np.sqrt(np.mean(np.square(speed[stop])))) if np.any(stop) else None

    violations: set[str] = set()
    if trace.terminated:
        violations.add("fall")
    if not bool(np.asarray(trace.finite, dtype=bool).all()) or not all(
        np.isfinite(value).all()
        for value in (
            command,
            speed,
            trace.pelvis_height,
            trace.torso_tilt_rad,
            trace.left_foot_position,
            trace.right_foot_position,
        )
    ):
        violations.add("nonfinite_state")
    if walk_seconds < criteria.minimum_walk_seconds:
        violations.add("insufficient_walk_duration")
    if command_rms is not None and command_rms > criteria.max_command_rms_m_s:
        violations.add("command_tracking")
    if completed_steps < criteria.minimum_completed_steps:
        violations.add("insufficient_steps")
    if contact_transitions_s > criteria.maximum_contact_transitions_s:
        violations.add("contact_chatter")
    if alternation is not None and alternation < criteria.minimum_alternation_ratio:
        violations.add("nonalternating_steps")
    if step_median is not None and (
        step_median < criteria.minimum_step_length_m
        or step_median > criteria.maximum_step_length_m
        or tiny_fraction is not None
        and tiny_fraction > criteria.maximum_tiny_step_fraction
    ):
        violations.add("shuffling")
    if cadence is not None and not (
        criteria.minimum_cadence_steps_s <= cadence <= criteria.maximum_cadence_steps_s
    ):
        violations.add("cadence")
    if stance_slip is not None and stance_slip > criteria.maximum_stance_slip_rms_m_s:
        violations.add("sliding")
    if minimum_height is not None and minimum_height < criteria.minimum_pelvis_height_m:
        violations.add("crouch")
    if maximum_tilt is not None and maximum_tilt > criteria.maximum_torso_tilt_rad:
        violations.add("torso_tilt")
    if stop_rms is not None and stop_rms > criteria.maximum_stop_speed_rms_m_s:
        violations.add("failed_stop")

    functional_terms = {
        "fall",
        "nonfinite_state",
        "insufficient_walk_duration",
        "command_tracking",
        "insufficient_steps",
        "failed_stop",
    }
    style_terms = {
        "insufficient_walk_duration",
        "insufficient_steps",
        "contact_chatter",
        "nonalternating_steps",
        "shuffling",
        "cadence",
        "sliding",
        "crouch",
        "torso_tilt",
    }
    functional_passed = not (violations & functional_terms)
    style_passed = not (violations & style_terms)
    precedence = (
        "fall",
        "nonfinite_state",
        "contact_chatter",
        "insufficient_steps",
        "command_tracking",
        "sliding",
        "shuffling",
        "nonalternating_steps",
        "cadence",
        "crouch",
        "torso_tilt",
        "failed_stop",
        "insufficient_walk_duration",
    )
    classification = next((name for name in precedence if name in violations), "qualified_trace")
    return WalkingResult(
        functional_passed=functional_passed,
        style_passed=style_passed,
        classification=classification,
        violations=tuple(sorted(violations)),
        walk_seconds=walk_seconds,
        completed_steps=completed_steps,
        command_rms_m_s=command_rms,
        step_length_median_m=step_median,
        cadence_steps_s=cadence,
        alternation_ratio=alternation,
        tiny_step_fraction=tiny_fraction,
        stance_slip_rms_m_s=stance_slip,
        minimum_pelvis_height_m=minimum_height,
        maximum_torso_tilt_rad=maximum_tilt,
        contact_transitions_s=contact_transitions_s,
        stop_samples=int(np.count_nonzero(command == 0)),
        stop_speed_rms_m_s=stop_rms,
    )
