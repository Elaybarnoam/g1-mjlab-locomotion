"""Shared command, gait-phase, and stand/walk transition state machine."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class CommandProfile:
    """V1 supports stop and forward speeds up to the audited reference speed."""

    reference_speed_m_s: float
    cycle_duration_s: float
    acceleration_m_s2: float
    deceleration_m_s2: float
    blend_rate_s: float
    stand_threshold_m_s: float
    walk_threshold_m_s: float
    reference_id: str = ""

    def validate(self) -> None:
        values = (
            self.reference_speed_m_s,
            self.cycle_duration_s,
            self.acceleration_m_s2,
            self.deceleration_m_s2,
            self.blend_rate_s,
            self.stand_threshold_m_s,
            self.walk_threshold_m_s,
        )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("command profile values must be finite and positive")
        if self.stand_threshold_m_s >= self.walk_threshold_m_s:
            raise ValueError("stand threshold must be lower than walk threshold")
        if self.walk_threshold_m_s >= self.reference_speed_m_s:
            raise ValueError("walk threshold must be lower than reference speed")

    def validate_requested(self, command: npt.ArrayLike) -> FloatArray:
        self.validate()
        requested = np.asarray(command, dtype=np.float64)
        if requested.ndim != 2 or requested.shape[1] != 3 or not np.all(np.isfinite(requested)):
            raise ValueError("requested command must have finite shape (num_envs, 3)")
        if not np.allclose(requested[:, 1:], 0.0, atol=1e-12):
            raise ValueError("walking-v1 does not support lateral and yaw commands")
        valid_forward = (requested[:, 0] >= 0.0) & (requested[:, 0] <= self.reference_speed_m_s)
        if not np.all(valid_forward):
            raise ValueError(
                "walking-v1 requested forward speed must be within the supported forward range"
            )
        return requested


@dataclass(frozen=True, slots=True)
class GaitState:
    applied_command: FloatArray
    phase: FloatArray
    blend: FloatArray
    walking: BoolArray
    reference_distance_m: FloatArray

    @classmethod
    def zeros(cls, num_envs: int) -> Self:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        return cls(
            applied_command=np.zeros((num_envs, 3), dtype=np.float64),
            phase=np.zeros(num_envs, dtype=np.float64),
            blend=np.zeros(num_envs, dtype=np.float64),
            walking=np.zeros(num_envs, dtype=np.bool_),
            reference_distance_m=np.zeros(num_envs, dtype=np.float64),
        )

    def validate(self) -> None:
        count = self.phase.shape[0]
        if (
            self.applied_command.shape != (count, 3)
            or self.blend.shape != (count,)
            or self.walking.shape != (count,)
            or self.reference_distance_m.shape != (count,)
        ):
            raise ValueError("gait state arrays have inconsistent batch dimensions")
        if (
            not np.all(np.isfinite(self.applied_command))
            or not np.all(np.isfinite(self.phase))
            or not np.all(np.isfinite(self.reference_distance_m))
        ):
            raise ValueError("gait state contains non-finite values")
        if np.any((self.phase < 0) | (self.phase >= 1)) or np.any(
            (self.blend < 0) | (self.blend > 1)
        ):
            raise ValueError("phase and blend must be in their declared ranges")

    @property
    def policy_features(self) -> FloatArray:
        return np.column_stack(
            (np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase), self.blend)
        )


def step_gait_numpy(
    state: GaitState,
    requested_command: npt.ArrayLike,
    profile: CommandProfile,
    *,
    dt: float,
) -> GaitState:
    """Advance once in the canonical order: command, mode, blend, then phase."""
    state.validate()
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    requested = profile.validate_requested(requested_command)
    if requested.shape != state.applied_command.shape:
        raise ValueError("requested command batch does not match gait state")
    delta = requested - state.applied_command
    limit = np.where(delta >= 0, profile.acceleration_m_s2, profile.deceleration_m_s2) * dt
    applied = state.applied_command + np.clip(delta, -limit, limit)
    walking = np.where(
        state.walking,
        applied[:, 0] > profile.stand_threshold_m_s,
        applied[:, 0] >= profile.walk_threshold_m_s,
    )
    blend_target = walking.astype(np.float64)
    blend = state.blend + np.clip(
        blend_target - state.blend, -profile.blend_rate_s * dt, profile.blend_rate_s * dt
    )
    phase_rate = applied[:, 0] / (profile.reference_speed_m_s * profile.cycle_duration_s)
    phase = np.mod(state.phase + dt * phase_rate, 1.0)
    reference_distance = state.reference_distance_m + applied[:, 0] * dt
    result = GaitState(applied, phase, blend, walking, reference_distance)
    result.validate()
    return result


def step_gait_torch(
    applied_command: Any,
    phase: Any,
    blend: Any,
    walking: Any,
    reference_distance_m: Any,
    requested_command: Any,
    profile: CommandProfile,
    *,
    dt: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Vectorized Torch equivalent; intentionally imports Torch only inside GPU runtime."""
    import torch

    profile.validate()
    if requested_command.ndim != 2 or requested_command.shape[1] != 3:
        raise ValueError("requested command must have shape (num_envs, 3)")
    # Requested values are validated when the GPU-resident schedule is created. Do not add a
    # tensor-to-host truth conversion here: this function is called in the per-control-step path.
    dt_vector = torch.as_tensor(dt, dtype=applied_command.dtype, device=applied_command.device)
    if dt_vector.ndim > 1 or (dt_vector.ndim == 1 and dt_vector.shape[0] != phase.shape[0]):
        raise ValueError("dt must be scalar or contain one value per environment")
    dt_command = dt_vector[:, None] if dt_vector.ndim == 1 else dt_vector
    delta = requested_command - applied_command
    rate = torch.where(
        delta >= 0,
        profile.acceleration_m_s2,
        profile.deceleration_m_s2,
    )
    applied = applied_command + torch.clamp(delta, min=-rate * dt_command, max=rate * dt_command)
    walking_next = torch.where(
        walking,
        applied[:, 0] > profile.stand_threshold_m_s,
        applied[:, 0] >= profile.walk_threshold_m_s,
    )
    target = walking_next.to(dtype=blend.dtype)
    blend_next = blend + torch.clamp(
        target - blend,
        min=-profile.blend_rate_s * dt_vector,
        max=profile.blend_rate_s * dt_vector,
    )
    phase_rate = applied[:, 0] / (profile.reference_speed_m_s * profile.cycle_duration_s)
    phase_next = torch.remainder(phase + dt_vector * phase_rate, 1.0)
    reference_distance_next = reference_distance_m + applied[:, 0] * dt_vector
    return applied, phase_next, blend_next, walking_next, reference_distance_next


@dataclass(frozen=True, slots=True)
class WalkingInitialState:
    """Serializable physical state sufficient to reconstruct a nonstationary trial."""

    root_position_w: tuple[float, ...]
    root_quaternion_wxyz: tuple[float, ...]
    root_linear_velocity_w: tuple[float, ...]
    root_angular_velocity_w: tuple[float, ...]
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    phase: float
    blend: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        expected = {
            "root_position_w": 3,
            "root_quaternion_wxyz": 4,
            "root_linear_velocity_w": 3,
            "root_angular_velocity_w": 3,
            "joint_position": 29,
            "joint_velocity": 29,
        }
        for name, size in expected.items():
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain {size} finite values")
        norm = np.linalg.norm(self.root_quaternion_wxyz)
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError("root_quaternion_wxyz must be normalized")
        if not math.isfinite(self.phase) or not 0 <= self.phase < 1:
            raise ValueError("phase must be finite in [0, 1)")
        if not math.isfinite(self.blend) or not 0 <= self.blend <= 1:
            raise ValueError("blend must be finite in [0, 1]")


@dataclass(frozen=True, slots=True)
class CommandSegment:
    duration_s: float
    forward_speed_m_s: float


@dataclass(frozen=True, slots=True)
class CommandSchedule:
    schema_version: int
    name: str
    reference_id: str
    segments: tuple[CommandSegment, ...]


def load_command_profile(path: Path) -> CommandProfile:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("unsupported command profile schema")
    expected_root = {"schema_version", "reference_id", "command_frame", "profile"}
    if set(raw) != expected_root or not isinstance(raw["profile"], dict):
        raise ValueError("command profile fields do not match schema")
    expected_profile = set(CommandProfile.__dataclass_fields__) - {"reference_id"}
    if set(raw["profile"]) != expected_profile:
        raise ValueError("command profile numeric fields do not match schema")
    profile = CommandProfile(**raw["profile"], reference_id=str(raw["reference_id"]))
    profile.validate()
    return profile


def load_command_schedule(path: Path, profile: CommandProfile) -> CommandSchedule:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("unsupported command schedule schema")
    if set(raw) != {"schema_version", "name", "reference_id", "segments"}:
        raise ValueError("command schedule fields do not match schema")
    segments = tuple(CommandSegment(**item) for item in raw["segments"])
    if not segments:
        raise ValueError("command schedule must contain at least one segment")
    for segment in segments:
        if not math.isfinite(segment.duration_s) or segment.duration_s <= 0:
            raise ValueError("command segment duration must be finite and positive")
        profile.validate_requested(np.asarray([[segment.forward_speed_m_s, 0.0, 0.0]]))
    if profile.reference_id and raw["reference_id"] != profile.reference_id:
        raise ValueError("command schedule and profile reference identities do not match")
    return CommandSchedule(
        schema_version=1,
        name=str(raw["name"]),
        reference_id=str(raw["reference_id"]),
        segments=segments,
    )
