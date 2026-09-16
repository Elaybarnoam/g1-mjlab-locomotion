"""Batched state transition for walking commands and interval-end references."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from .reference_bank import ReferenceBank, ReferenceTargets

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WalkingTargetProfile:
    """Frozen host-side rates for the walking-v2 target state."""

    policy_dt_s: float = 0.020
    maximum_forward_speed_m_s: float = 0.8
    full_walk_blend_speed_m_s: float = 0.4
    acceleration_m_s2: float = 0.6
    deceleration_m_s2: float = 0.8
    blend_rate_s: float = 1.0

    def validate(self) -> None:
        values = (
            self.policy_dt_s,
            self.maximum_forward_speed_m_s,
            self.full_walk_blend_speed_m_s,
            self.acceleration_m_s2,
            self.deceleration_m_s2,
            self.blend_rate_s,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("walking target profile values must be finite and positive")
        if self.full_walk_blend_speed_m_s > self.maximum_forward_speed_m_s:
            raise ValueError("full-walk blend speed must be inside the command domain")

    def validate_requested(self, requested_command: npt.ArrayLike) -> FloatArray:
        self.validate()
        requested = np.asarray(requested_command, dtype=np.float64)
        if requested.ndim != 2 or requested.shape[1] != 3:
            raise ValueError("requested command must have shape (num_envs, 3)")
        if not np.isfinite(requested).all():
            raise ValueError("requested command must be finite")
        if not np.allclose(requested[:, 1:], 0.0, atol=1e-12):
            raise ValueError("walking-v2 first acquisition supports forward commands only")
        if np.any(requested[:, 0] < 0) or np.any(requested[:, 0] > self.maximum_forward_speed_m_s):
            raise ValueError("forward command must be in [0, 0.8] m/s")
        return requested


@dataclass(frozen=True, slots=True)
class WalkingTargetState:
    """Committed state at one policy boundary."""

    applied_command: FloatArray
    phase: FloatArray
    blend: FloatArray

    @classmethod
    def zeros(cls, num_envs: int) -> Self:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        return cls(
            applied_command=np.zeros((num_envs, 3), dtype=np.float64),
            phase=np.zeros(num_envs, dtype=np.float64),
            blend=np.zeros(num_envs, dtype=np.float64),
        )

    def validate(self) -> None:
        count = len(self.phase)
        if self.applied_command.shape != (count, 3) or self.blend.shape != (count,):
            raise ValueError("walking target state has inconsistent batch dimensions")
        if not all(
            np.isfinite(value).all() for value in (self.applied_command, self.phase, self.blend)
        ):
            raise ValueError("walking target state must be finite")
        if np.any((self.phase < 0) | (self.phase >= 1)):
            raise ValueError("phase must be in [0, 1)")
        if np.any((self.blend < 0) | (self.blend > 1)):
            raise ValueError("blend must be in [0, 1]")

    @property
    def policy_features(self) -> FloatArray:
        return np.column_stack(
            (
                np.sin(2 * np.pi * self.phase),
                np.cos(2 * np.pi * self.phase),
                self.blend,
            )
        )


@dataclass(frozen=True, slots=True)
class WalkingTargetTransition:
    """Cached interval-end prediction consumed by action, reward and commit."""

    next_state: WalkingTargetState
    targets: ReferenceTargets
    command_rate_m_s2: FloatArray
    blend_rate_s: FloatArray
    phase_rate_hz: FloatArray


@dataclass(frozen=True, slots=True)
class TorchWalkingTargetTransition:
    next_applied_command: Any
    next_phase: Any
    next_blend: Any
    target_joint_position: Any
    target_joint_velocity: Any
    command_rate_m_s2: Any
    blend_rate_s: Any
    phase_rate_hz: Any


def smoothstep_walk_blend_numpy(
    forward_speed_m_s: npt.ArrayLike, full_walk_speed_m_s: float = 0.4
) -> FloatArray:
    """Map forward speed to the desired C1 stand/walk blend."""
    speed = np.asarray(forward_speed_m_s, dtype=np.float64)
    if not math.isfinite(full_walk_speed_m_s) or full_walk_speed_m_s <= 0:
        raise ValueError("full walk speed must be finite and positive")
    if not np.isfinite(speed).all() or np.any(speed < 0):
        raise ValueError("forward speed must be finite and nonnegative")
    fraction = np.clip(speed / full_walk_speed_m_s, 0.0, 1.0)
    return np.asarray(fraction * fraction * (3.0 - 2.0 * fraction), dtype=np.float64)


def predict_walking_target_numpy(
    state: WalkingTargetState,
    requested_command: npt.ArrayLike,
    bank: ReferenceBank,
    profile: WalkingTargetProfile | None = None,
) -> WalkingTargetTransition:
    """Predict exactly one policy interval without mutating committed state."""
    profile = profile or WalkingTargetProfile()
    state.validate()
    requested = profile.validate_requested(requested_command)
    if requested.shape != state.applied_command.shape:
        raise ValueError("requested command batch does not match target state")
    dt = profile.policy_dt_s
    delta = requested[:, 0] - state.applied_command[:, 0]
    rate_limit = np.where(delta >= 0.0, profile.acceleration_m_s2, profile.deceleration_m_s2)
    next_forward = state.applied_command[:, 0] + np.clip(delta, -rate_limit * dt, rate_limit * dt)
    next_command = np.zeros_like(state.applied_command)
    next_command[:, 0] = next_forward
    command_rate = (next_forward - state.applied_command[:, 0]) / dt
    desired_blend = smoothstep_walk_blend_numpy(next_forward, profile.full_walk_blend_speed_m_s)
    next_blend = state.blend + np.clip(
        desired_blend - state.blend,
        -profile.blend_rate_s * dt,
        profile.blend_rate_s * dt,
    )
    blend_rate = (next_blend - state.blend) / dt
    phase_rate = bank.sample_numpy(state.phase, next_forward).phase_rate_hz
    next_phase = np.mod(state.phase + dt * phase_rate, 1.0)
    next_state = WalkingTargetState(next_command, next_phase, next_blend)
    next_state.validate()
    targets = bank.compose_targets_numpy(
        next_phase,
        next_forward,
        blend=next_blend,
        blend_rate_s=blend_rate,
        speed_rate_m_s2=command_rate,
    )
    return WalkingTargetTransition(
        next_state,
        targets,
        command_rate,
        blend_rate,
        phase_rate,
    )


def reset_walking_target_numpy(
    state: WalkingTargetState,
    indices: npt.ArrayLike,
    *,
    phase: npt.ArrayLike | float = 0.0,
) -> WalkingTargetState:
    """Return a copy with only selected environments reset."""
    state.validate()
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= len(state.phase)):
        raise ValueError("reset indices must be a valid one-dimensional selection")
    reset_phase = np.broadcast_to(np.asarray(phase, dtype=np.float64), selected.shape)
    if not np.isfinite(reset_phase).all() or np.any((reset_phase < 0) | (reset_phase >= 1)):
        raise ValueError("reset phase must be finite in [0, 1)")
    command = state.applied_command.copy()
    phases = state.phase.copy()
    blend = state.blend.copy()
    command[selected] = 0.0
    phases[selected] = reset_phase
    blend[selected] = 0.0
    result = WalkingTargetState(command, phases, blend)
    result.validate()
    return result


def predict_walking_target_torch(
    applied_command: Any,
    phase: Any,
    blend: Any,
    requested_command: Any,
    bank: ReferenceBank,
    profile: WalkingTargetProfile | None = None,
) -> TorchWalkingTargetTransition:
    """GPU-resident equivalent; command-domain validation belongs at schedule creation."""
    import torch

    profile = profile or WalkingTargetProfile()
    profile.validate()
    if requested_command.ndim != 2 or requested_command.shape[1] != 3:
        raise ValueError("requested command must have shape (num_envs, 3)")
    if applied_command.shape != requested_command.shape or phase.shape != blend.shape:
        raise ValueError("torch target-state tensors have inconsistent batch dimensions")
    if phase.ndim != 1 or phase.shape[0] != applied_command.shape[0]:
        raise ValueError("torch target-state tensors require one flat environment dimension")
    dt = profile.policy_dt_s
    delta = requested_command[:, 0] - applied_command[:, 0]
    rate_limit = torch.where(
        delta >= 0.0,
        torch.full_like(delta, profile.acceleration_m_s2),
        torch.full_like(delta, profile.deceleration_m_s2),
    )
    next_forward = applied_command[:, 0] + torch.clamp(
        delta, min=-rate_limit * dt, max=rate_limit * dt
    )
    next_command = torch.zeros_like(applied_command)
    next_command[:, 0] = next_forward
    command_rate = (next_forward - applied_command[:, 0]) / dt
    fraction = torch.clamp(next_forward / profile.full_walk_blend_speed_m_s, min=0.0, max=1.0)
    desired_blend = fraction * fraction * (3.0 - 2.0 * fraction)
    next_blend = blend + torch.clamp(
        desired_blend - blend,
        min=-profile.blend_rate_s * dt,
        max=profile.blend_rate_s * dt,
    )
    blend_rate = (next_blend - blend) / dt
    phase_rate = bank.sample_torch(phase, next_forward).phase_rate_hz
    next_phase = torch.remainder(phase + dt * phase_rate, 1.0)
    sample = bank.sample_torch(next_phase, next_forward)
    nominal = torch.as_tensor(
        bank.nominal_joint_position,
        dtype=applied_command.dtype,
        device=applied_command.device,
    )
    target_position = nominal + next_blend[:, None] * (sample.joint_position - nominal)
    target_velocity = blend_rate[:, None] * (sample.joint_position - nominal) + next_blend[
        :, None
    ] * (
        sample.joint_partial_phase * phase_rate[:, None]
        + sample.joint_partial_speed * command_rate[:, None]
    )
    return TorchWalkingTargetTransition(
        next_command,
        next_phase,
        next_blend,
        target_position,
        target_velocity,
        command_rate,
        blend_rate,
        phase_rate,
    )


def reset_walking_target_torch(
    applied_command: Any,
    phase: Any,
    blend: Any,
    indices: Any,
    *,
    reset_phase: Any = 0.0,
) -> tuple[Any, Any, Any]:
    """Clone and partially reset GPU-resident target-state tensors."""
    import torch

    command_next = applied_command.clone()
    phase_next = phase.clone()
    blend_next = blend.clone()
    command_next[indices] = 0.0
    phase_next[indices] = torch.as_tensor(reset_phase, dtype=phase.dtype, device=phase.device)
    blend_next[indices] = 0.0
    return command_next, phase_next, blend_next
