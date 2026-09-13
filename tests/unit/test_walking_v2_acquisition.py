from __future__ import annotations

import numpy as np

from g1_mjlab.walking_v2_acquisition import effort_limits_nm, summarize_acquisition_trace


def test_effort_limits_cover_frozen_joint_order() -> None:
    limits = effort_limits_nm()

    assert limits.shape == (29,)
    assert limits.min() == 5.0
    assert limits.max() == 139.0


def test_acquisition_summary_detects_stable_alternating_walk() -> None:
    count = 150
    contact = np.ones((count, 2), dtype=bool)
    contact[60:70, 0] = False
    contact[90:100, 1] = False
    contact[120:130, 0] = False
    root = np.zeros((count, 3))
    root[:, 0] = np.arange(count) * 0.02 * 0.4
    root[:, 2] = 0.75
    arrays = {
        "reward": np.ones(count),
        "root_linear_velocity_body": np.tile([0.4, 0.0, 0.0], (count, 1)),
        "command": np.tile([0.4, 0.0, 0.0], (count, 1)),
        "joint_position": np.zeros((count, 29)),
        "joint_target": np.zeros((count, 29)),
        "joint_velocity": np.zeros((count, 29)),
        "reference_joint_velocity": np.zeros((count, 29)),
        "actuator_torque": np.zeros((count, 29)),
        "contact": contact,
        "expected_contact": contact.copy(),
        "root_position_w": root,
        "root_quaternion_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
        "terminated": np.zeros(count, dtype=bool),
    }

    result = summarize_acquisition_trace(arrays, control_dt=0.02, requested_speed_m_s=0.4)

    assert result["development_functional_passed"] is True
    assert result["touchdown_count"] == 3
    assert result["alternation_fraction"] == 1.0
    assert result["settled_forward_command_rms_m_s"] == 0.0


def test_acquisition_summary_fails_fall() -> None:
    count = 60
    arrays = {
        "reward": np.zeros(count),
        "root_linear_velocity_body": np.zeros((count, 3)),
        "command": np.zeros((count, 3)),
        "joint_position": np.zeros((count, 29)),
        "joint_target": np.zeros((count, 29)),
        "joint_velocity": np.zeros((count, 29)),
        "reference_joint_velocity": np.zeros((count, 29)),
        "actuator_torque": np.zeros((count, 29)),
        "contact": np.ones((count, 2), dtype=bool),
        "expected_contact": np.ones((count, 2), dtype=bool),
        "root_position_w": np.tile([0.0, 0.0, 0.4], (count, 1)),
        "root_quaternion_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
        "terminated": np.arange(count) == count - 1,
    }

    result = summarize_acquisition_trace(arrays, control_dt=0.02, requested_speed_m_s=0.4)

    assert result["terminated"] is True
    assert result["development_functional_passed"] is False
