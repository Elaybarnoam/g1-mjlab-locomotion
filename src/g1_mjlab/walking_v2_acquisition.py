"""Development-only metrics for bounded walking-v2 acquisition checkpoints."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from .motion import G1_JOINT_NAMES

FloatArray = npt.NDArray[np.floating[Any]]


def _debounce_contact(contact: npt.NDArray[np.bool_], minimum_frames: int) -> npt.NDArray[np.bool_]:
    """Require a state change to persist before accepting it for gait events."""
    result = np.empty_like(contact)
    for foot in range(contact.shape[1]):
        state = bool(contact[0, foot])
        candidate_frames = 0
        for frame, measured in enumerate(contact[:, foot]):
            if bool(measured) == state:
                candidate_frames = 0
            else:
                candidate_frames += 1
                if candidate_frames >= minimum_frames:
                    state = bool(measured)
                    candidate_frames = 0
            result[frame, foot] = state
    return result


def effort_limits_nm(joint_names: tuple[str, ...] = G1_JOINT_NAMES) -> FloatArray:
    """Return the actuator-limit convention used by the walking-v2 reward."""
    values: list[float] = []
    for name in joint_names:
        if any(token in name for token in ("wrist_pitch", "wrist_yaw")):
            values.append(5.0)
        elif any(
            token in name
            for token in ("elbow", "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "wrist_roll")
        ):
            values.append(25.0)
        elif any(token in name for token in ("hip_roll", "knee")):
            values.append(139.0)
        elif any(token in name for token in ("hip_pitch", "hip_yaw", "waist_yaw")):
            values.append(88.0)
        else:
            values.append(50.0)
    return np.asarray(values, dtype=np.float64)


def summarize_acquisition_trace(
    arrays: dict[str, npt.NDArray[Any]], *, control_dt: float, requested_speed_m_s: float
) -> dict[str, Any]:
    """Reduce a deterministic pre-reset rollout into promotion trend metrics."""
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be finite and positive")
    sample_count = len(arrays["reward"])
    if sample_count == 0 or any(len(value) != sample_count for value in arrays.values()):
        raise ValueError("trace arrays must be non-empty and aligned")
    settle = min(sample_count - 1, math.ceil(1.0 / control_dt))
    eligible = slice(settle, None)
    velocity = np.asarray(arrays["root_linear_velocity_body"], dtype=float)
    command = np.asarray(arrays["command"], dtype=float)
    joint_position = np.asarray(arrays["joint_position"], dtype=float)
    joint_target = np.asarray(arrays["joint_target"], dtype=float)
    joint_velocity = np.asarray(arrays["joint_velocity"], dtype=float)
    reference_velocity = np.asarray(arrays["reference_joint_velocity"], dtype=float)
    torque = np.asarray(arrays["actuator_torque"], dtype=float)
    torque_ratio = np.abs(torque) / effort_limits_nm()[None, :]
    contact = np.asarray(arrays["contact"], dtype=bool)
    expected = np.asarray(arrays["expected_contact"], dtype=bool)
    transitions = np.count_nonzero(np.diff(contact.astype(np.int8), axis=0))
    stable_contact = _debounce_contact(contact, max(1, math.ceil(0.06 / control_dt)))
    duration = sample_count * control_dt
    touchdowns = np.argwhere(stable_contact[1:] & ~stable_contact[:-1])
    touchdown_sides = touchdowns[:, 1] if len(touchdowns) else np.empty(0, dtype=int)
    alternating = np.count_nonzero(touchdown_sides[1:] != touchdown_sides[:-1])
    alternation = alternating / max(1, len(touchdown_sides) - 1)
    root = np.asarray(arrays["root_position_w"], dtype=float)
    quaternion = np.asarray(arrays["root_quaternion_wxyz"], dtype=float)
    upright_z = 1.0 - 2.0 * (quaternion[:, 1] ** 2 + quaternion[:, 2] ** 2)
    tilt = np.arccos(np.clip(upright_z, -1.0, 1.0))
    termination = np.asarray(arrays["terminated"], dtype=bool)
    moving = requested_speed_m_s >= 0.15
    return {
        "schema_version": 1,
        "requested_speed_m_s": requested_speed_m_s,
        "sample_count": sample_count,
        "survived_seconds": duration,
        "terminated": bool(np.any(termination)),
        "mean_reward": float(np.mean(arrays["reward"])),
        "settled_forward_command_rms_m_s": float(
            np.sqrt(np.mean(np.square(velocity[eligible, 0] - command[eligible, 0])))
        ),
        "reference_joint_position_rms_rad": float(
            np.sqrt(np.mean(np.square(joint_position[eligible] - joint_target[eligible])))
        ),
        "reference_joint_velocity_rms_rad_s": float(
            np.sqrt(np.mean(np.square(joint_velocity[eligible] - reference_velocity[eligible])))
        ),
        "torque_ratio_p95": float(np.percentile(torque_ratio, 95)),
        "torque_ratio_peak": float(np.max(torque_ratio)),
        "contact_mismatch_fraction": float(np.mean(stable_contact != expected)),
        "contact_transition_rate_s": float(transitions / duration),
        "touchdown_count": int(len(touchdowns)),
        "alternation_fraction": float(alternation),
        "cadence_steps_s": float(len(touchdowns) / duration),
        "forward_progress_m": float(root[-1, 0] - root[0, 0]),
        "minimum_pelvis_height_m": float(np.min(root[:, 2])),
        "maximum_torso_tilt_rad": float(np.max(tilt)),
        "development_functional_passed": bool(
            not np.any(termination)
            and np.min(root[:, 2]) >= 0.55
            and np.max(tilt) <= 0.52
            and (
                (moving and len(touchdowns) >= 2 and alternation >= 0.5)
                or (not moving and np.sqrt(np.mean(np.square(velocity[eligible, :2]))) <= 0.2)
            )
        ),
        "qualification_claim": False,
    }
