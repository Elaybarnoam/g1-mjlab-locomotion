"""Pure walking-v2 reward and acquisition-termination semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class WalkingRewardInputs:
    joint_error_rad: FloatArray
    joint_velocity_error_rad_s: FloatArray
    local_foot_error_m: FloatArray
    orientation_error_rad: FloatArray
    contact_agreement: FloatArray
    actual_velocity_body_m_s: FloatArray
    command_body_m_s: FloatArray
    joint_from_nominal_rad: FloatArray
    action: FloatArray
    previous_action: FloatArray
    normalized_torque: FloatArray
    soft_joint_limit_violation_rad: FloatArray
    blend: FloatArray
    true_fall_event: BoolArray

    def validate(self) -> None:
        batch = self.blend.shape
        expected = {
            "joint_error_rad": (*batch, 29),
            "joint_velocity_error_rad_s": (*batch, 29),
            "local_foot_error_m": (*batch, 2, 3),
            "orientation_error_rad": batch,
            "contact_agreement": batch,
            "actual_velocity_body_m_s": (*batch, 3),
            "command_body_m_s": (*batch, 3),
            "joint_from_nominal_rad": (*batch, 29),
            "action": (*batch, 29),
            "previous_action": (*batch, 29),
            "normalized_torque": (*batch, 29),
            "soft_joint_limit_violation_rad": (*batch, 29),
            "true_fall_event": batch,
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value.dtype.kind != "b" and not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if np.any((self.blend < 0) | (self.blend > 1)):
            raise ValueError("blend must be in [0, 1]")
        if np.any((self.contact_agreement < 0) | (self.contact_agreement > 1)):
            raise ValueError("contact agreement must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class WalkingRewardBreakdown:
    raw_terms: dict[str, FloatArray]
    weighted_terms: dict[str, FloatArray]
    imitation: FloatArray
    task: FloatArray
    stand: FloatArray
    rate: FloatArray
    step: FloatArray


def compute_walking_reward(
    inputs: WalkingRewardInputs, *, policy_dt_s: float = 0.020
) -> WalkingRewardBreakdown:
    """Evaluate the frozen Plan 06 reward after one action interval."""
    if not np.isfinite(policy_dt_s) or policy_dt_s <= 0.0:
        raise ValueError("policy_dt_s must be finite and positive")
    inputs.validate()
    raw = {
        "imitation_joint_pose": np.exp(
            -np.mean(np.square(inputs.joint_error_rad), axis=-1) / 0.30**2
        ),
        "imitation_joint_velocity": np.exp(
            -np.mean(np.square(inputs.joint_velocity_error_rad_s), axis=-1) / 3.0**2
        ),
        "imitation_local_feet": np.exp(
            -np.mean(np.square(inputs.local_foot_error_m), axis=(-2, -1)) / 0.10**2
        ),
        "imitation_orientation": np.exp(-np.square(inputs.orientation_error_rad) / 0.25**2),
        "imitation_contact": inputs.contact_agreement,
        "task_forward_velocity": np.exp(
            -np.square(inputs.actual_velocity_body_m_s[..., 0] - inputs.command_body_m_s[..., 0])
            / 0.25**2
        ),
        "task_lateral_velocity": np.exp(
            -np.square(inputs.actual_velocity_body_m_s[..., 1] - inputs.command_body_m_s[..., 1])
            / 0.15**2
        ),
        "task_yaw_rate": np.exp(
            -np.square(inputs.actual_velocity_body_m_s[..., 2] - inputs.command_body_m_s[..., 2])
            / 0.25**2
        ),
        "stand_pose": np.exp(-np.mean(np.square(inputs.joint_from_nominal_rad), axis=-1) / 0.30**2),
        "stand_horizontal_velocity": np.exp(
            -np.sum(np.square(inputs.actual_velocity_body_m_s[..., :2]), axis=-1) / 0.10**2
        ),
        "action_rate": np.mean(np.square(inputs.action - inputs.previous_action), axis=-1),
        "normalized_torque": np.mean(np.square(inputs.normalized_torque), axis=-1),
        "soft_joint_limit": np.sum(np.square(inputs.soft_joint_limit_violation_rad), axis=-1),
    }
    imitation_weights = {
        "imitation_joint_pose": 0.40,
        "imitation_joint_velocity": 0.10,
        "imitation_local_feet": 0.25,
        "imitation_orientation": 0.15,
        "imitation_contact": 0.10,
    }
    weighted = {name: raw[name] * weight for name, weight in imitation_weights.items()}
    imitation = np.zeros_like(inputs.blend, dtype=np.float64)
    for value in weighted.values():
        imitation += value
    weighted.update(
        {
            "task_forward_velocity": raw["task_forward_velocity"],
            "task_lateral_velocity": 0.25 * raw["task_lateral_velocity"],
            "task_yaw_rate": 0.25 * raw["task_yaw_rate"],
            "stand_pose": raw["stand_pose"],
            "stand_horizontal_velocity": raw["stand_horizontal_velocity"],
            "action_rate": -0.10 * raw["action_rate"],
            "normalized_torque": -0.05 * raw["normalized_torque"],
            "soft_joint_limit": -0.50 * raw["soft_joint_limit"],
        }
    )
    task = (
        weighted["task_forward_velocity"]
        + weighted["task_lateral_velocity"]
        + weighted["task_yaw_rate"]
    )
    stand = weighted["stand_pose"] + weighted["stand_horizontal_velocity"]
    penalties = (
        weighted["action_rate"] + weighted["normalized_torque"] + weighted["soft_joint_limit"]
    )
    rate = 2.0 * inputs.blend * imitation + task + (1.0 - inputs.blend) * stand + penalties
    step = policy_dt_s * rate - 2.0 * inputs.true_fall_event.astype(np.float64)
    return WalkingRewardBreakdown(raw, weighted, imitation, task, stand, rate, step)


class AcquisitionTermination(StrEnum):
    RUNNING = "running"
    NONFINITE = "nonfinite"
    FORBIDDEN_CONTACT = "forbidden_contact"
    FALL = "fall"
    REFERENCE_DEVIATION = "reference_deviation"


def classify_acquisition_termination(
    *,
    nonfinite: npt.ArrayLike,
    forbidden_contact: npt.ArrayLike,
    fall: npt.ArrayLike,
    reference_deviation: npt.ArrayLike,
) -> npt.NDArray[np.str_]:
    """Classify mutually exclusive termination causes using frozen safety precedence."""
    arrays = np.broadcast_arrays(
        np.asarray(nonfinite, dtype=bool),
        np.asarray(forbidden_contact, dtype=bool),
        np.asarray(fall, dtype=bool),
        np.asarray(reference_deviation, dtype=bool),
    )
    result = np.full(arrays[0].shape, AcquisitionTermination.RUNNING.value, dtype="<U20")
    for mask, cause in reversed(
        tuple(
            zip(
                arrays,
                (
                    AcquisitionTermination.NONFINITE,
                    AcquisitionTermination.FORBIDDEN_CONTACT,
                    AcquisitionTermination.FALL,
                    AcquisitionTermination.REFERENCE_DEVIATION,
                ),
                strict=True,
            )
        )
    ):
        result[mask] = cause.value
    return result
