"""Simulator-independent incentive probes for walking reward design."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class WalkingProbe:
    command_speed_m_s: float
    achieved_speed_m_s: float
    joint_pose_rms_rad: float
    joint_velocity_rms_rad_s: float
    contact_agreement: float
    stance_slip_m_s: float
    pelvis_height_m: float
    torso_tilt_rad: float
    step_length_m: float
    cadence_steps_min: float
    non_foot_contact: bool
    finite: bool


@dataclass(frozen=True, slots=True)
class ProbeClassification:
    passed: bool
    failures: tuple[str, ...]


def classify_walking_probe(probe: WalkingProbe) -> ProbeClassification:
    failures: list[str] = []
    numeric = (
        probe.command_speed_m_s,
        probe.achieved_speed_m_s,
        probe.joint_pose_rms_rad,
        probe.joint_velocity_rms_rad_s,
        probe.contact_agreement,
        probe.stance_slip_m_s,
        probe.pelvis_height_m,
        probe.torso_tilt_rad,
        probe.step_length_m,
        probe.cadence_steps_min,
    )
    if not probe.finite or not all(math.isfinite(value) for value in numeric):
        failures.append("nonfinite")
        return ProbeClassification(False, tuple(failures))
    if abs(probe.achieved_speed_m_s - probe.command_speed_m_s) > 0.2:
        failures.append("command_tracking")
    if probe.joint_pose_rms_rad > 0.2 or probe.joint_velocity_rms_rad_s > 1.5:
        failures.append("reference_tracking")
    if probe.contact_agreement < 0.7:
        failures.append("contact_timing")
    if probe.stance_slip_m_s > 0.15:
        failures.append("stance_slip")
    if probe.pelvis_height_m < 0.62:
        failures.append("crouching")
    if probe.torso_tilt_rad > 0.6:
        failures.append("fallen")
    if probe.step_length_m < 0.18 and probe.command_speed_m_s > 0.15:
        failures.append("shuffling")
    if probe.cadence_steps_min > 170.0:
        failures.append("shuffling")
    if probe.non_foot_contact:
        failures.append("non_foot_contact")
    return ProbeClassification(not failures, tuple(dict.fromkeys(failures)))


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    contributions: Mapping[str, float]
    total_rate: float

    def integrated_step_reward(self, dt: float) -> float:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        return self.total_rate * dt


def weighted_reward_rate(
    raw_terms: Mapping[str, float], weights: Mapping[str, float]
) -> RewardBreakdown:
    if set(raw_terms) != set(weights):
        raise ValueError("raw reward terms and weights must have identical keys")
    contributions = {name: float(raw_terms[name] * weights[name]) for name in sorted(raw_terms)}
    if not all(math.isfinite(value) for value in contributions.values()):
        raise ValueError("reward contribution is non-finite")
    return RewardBreakdown(MappingProxyType(contributions), sum(contributions.values()))
