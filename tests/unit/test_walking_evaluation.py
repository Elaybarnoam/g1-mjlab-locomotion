from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from g1_mjlab.selection import score_walking
from g1_mjlab.walking_evaluation import (
    WalkingCriteria,
    WalkingTrace,
    evaluate_walking_trace,
)


def gait_trace(*, steps: int = 500, dt: float = 0.02) -> WalkingTrace:
    time = np.arange(steps) * dt
    phase = np.mod(time / 1.0, 1.0)
    left_contact = phase < 0.48
    right_contact = (phase >= 0.5) & (phase < 0.98)
    strikes = np.floor(time / 0.5).astype(int)
    left_x = np.floor((strikes + 1) / 2) * 0.58
    right_x = np.floor(strikes / 2) * 0.58 + 0.29
    return WalkingTrace(
        dt=dt,
        command_speed=np.full(steps, 1.16),
        forward_speed=np.full(steps, 1.14),
        pelvis_height=np.full(steps, 0.74),
        torso_tilt_rad=np.full(steps, 0.08),
        left_foot_position=np.column_stack((left_x, np.full(steps, 0.1), np.zeros(steps))),
        right_foot_position=np.column_stack((right_x, np.full(steps, -0.1), np.zeros(steps))),
        left_contact=left_contact,
        right_contact=right_contact,
        finite=np.ones(steps, dtype=bool),
        terminated=False,
    )


def test_ideal_alternating_gait_passes() -> None:
    result = evaluate_walking_trace(gait_trace(), WalkingCriteria())
    assert result.functional_passed
    assert result.style_passed
    assert result.classification == "qualified_trace"
    assert result.completed_steps >= 8


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("shuffle", "shuffling"),
        ("slide", "sliding"),
        ("no_steps", "insufficient_steps"),
        ("fall", "fall"),
        ("stand_at_command", "command_tracking"),
        ("chatter", "contact_chatter"),
    ],
)
def test_failure_modes_are_not_misclassified_as_walking(mutation: str, expected: str) -> None:
    trace = gait_trace()
    if mutation == "shuffle":
        trace = replace(
            trace,
            left_foot_position=trace.left_foot_position * np.array([0.1, 1, 1]),
            right_foot_position=trace.right_foot_position * np.array([0.1, 1, 1]),
        )
    elif mutation == "slide":
        drift = np.arange(trace.frame_count)[:, None] * np.array([0.01, 0, 0])
        trace = replace(
            trace,
            left_foot_position=trace.left_foot_position + drift,
            right_foot_position=trace.right_foot_position + drift,
        )
    elif mutation == "no_steps":
        trace = replace(
            trace,
            left_contact=np.ones(trace.frame_count, dtype=bool),
            right_contact=np.ones(trace.frame_count, dtype=bool),
        )
    elif mutation == "fall":
        trace = replace(trace, terminated=True)
    elif mutation == "stand_at_command":
        trace = replace(trace, forward_speed=np.zeros(trace.frame_count))
    elif mutation == "chatter":
        contact = np.arange(trace.frame_count) % 2 == 0
        trace = replace(trace, left_contact=contact, right_contact=~contact)
    result = evaluate_walking_trace(trace, WalkingCriteria())
    assert not (result.functional_passed and result.style_passed)
    assert expected in result.violations


def test_zero_command_stop_is_scored_separately_from_walking() -> None:
    trace = gait_trace()
    command = trace.command_speed.copy()
    speed = trace.forward_speed.copy()
    command[-100:] = 0
    speed[-75:] = 0.02
    result = evaluate_walking_trace(
        replace(trace, command_speed=command, forward_speed=speed), WalkingCriteria()
    )
    assert result.stop_samples == 100
    assert result.stop_speed_rms_m_s < 0.1


def test_walking_selection_prefers_gates_then_command_error_not_reward() -> None:
    def summary(passed: int, functional: int, command_error: float) -> dict[str, object]:
        return {
            "phase": "development",
            "checkpoint": "model_10.pt",
            "passed_both": passed,
            "trials": [
                {
                    "functional_passed": index < functional,
                    "style_passed": index < passed,
                    "command_rms_m_s": command_error,
                    "alternation_ratio": 1.0,
                    "tiny_step_fraction": 0.0,
                    "stance_slip_rms_m_s": 0.01,
                }
                for index in range(4)
            ],
        }

    gate_pass = summary(3, 4, 0.2)
    low_error_but_style_fail = summary(0, 4, 0.01)
    assert score_walking(gate_pass) > score_walking(low_error_but_style_fail)
