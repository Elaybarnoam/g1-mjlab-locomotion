"""Reproducible feasibility audit for time-warped walking references."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    """Hash repository text independently of checkout line-ending conversion."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@dataclass(frozen=True, slots=True)
class ReferenceAuditCriteria:
    maximum_pose_seam_rms_rad: float = 0.05
    maximum_velocity_seam_rms_rad_s: float = 1.5
    maximum_interpolation_derivative_rms_rad_s: float = 0.5
    minimum_joint_limit_margin_rad: float = 0.019
    maximum_actuator_demand_ratio: float = 0.8
    minimum_corrected_sole_clearance_m: float = 0.0
    maximum_contact_fraction_asymmetry: float = 0.30

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value < 0 for value in asdict(self).values()):
            raise ValueError("reference audit criteria must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class ReferenceSpeedEntry:
    speed_m_s: float
    time_scale: float
    cycle_period_s: float
    expected_cadence_steps_s: float
    step_length_median_m: float
    stride_length_median_m: float
    maximum_joint_velocity_rad_s: float
    maximum_joint_acceleration_rad_s2: float
    minimum_joint_limit_margin_rad: float
    maximum_actuator_demand_ratio: float
    sole_clearance_range_m: tuple[float, float]
    contact_fraction: tuple[float, float]
    double_support_fraction: float
    flight_fraction: float
    reachable_sole_forward_range_m: tuple[float, float]
    passed: bool
    violations: tuple[str, ...]

    def __post_init__(self) -> None:
        numeric = (
            self.speed_m_s,
            self.time_scale,
            self.cycle_period_s,
            self.expected_cadence_steps_s,
            self.step_length_median_m,
            self.stride_length_median_m,
            self.maximum_joint_velocity_rad_s,
            self.maximum_joint_acceleration_rad_s2,
            self.minimum_joint_limit_margin_rad,
            self.maximum_actuator_demand_ratio,
            *self.sole_clearance_range_m,
            *self.contact_fraction,
            self.double_support_fraction,
            self.flight_fraction,
            *self.reachable_sole_forward_range_m,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("reference speed entry values must be finite")
        if self.speed_m_s <= 0 or self.time_scale <= 0 or self.cycle_period_s <= 0:
            raise ValueError("reference speed entry timing must be positive")
        if any(
            not 0 <= value <= 1
            for value in (
                *self.contact_fraction,
                self.double_support_fraction,
                self.flight_fraction,
            )
        ):
            raise ValueError("reference contact fractions must be in [0, 1]")
        for bounds in (self.sole_clearance_range_m, self.reachable_sole_forward_range_m):
            if bounds[0] > bounds[1]:
                raise ValueError("reference ranges must be ordered")


@dataclass(frozen=True, slots=True)
class ReferenceSpeedMap:
    schema_version: int
    reference_id: str
    reference_sha256: str
    model_sha256: str
    controller_sha256: str
    criteria: ReferenceAuditCriteria
    transformation: str
    source_cycle_duration_s: float
    source_speed_m_s: float
    pose_seam_rms_rad: float
    velocity_seam_rms_rad_s: float
    interpolation_derivative_rms_rad_s: float
    reference_ground_offset_m: float
    supported_speed_range_m_s: tuple[float, float]
    entries: tuple[ReferenceSpeedEntry, ...]
    passed: bool
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.schema_version != 2 or not self.reference_id or not self.entries:
            raise ValueError("invalid reference speed map identity")
        for identity in (self.reference_sha256, self.model_sha256, self.controller_sha256):
            if len(identity) != 64 or any(
                character not in "0123456789abcdef" for character in identity
            ):
                raise ValueError("reference speed map identities must be lowercase SHA-256")
        if self.supported_speed_range_m_s[0] > self.supported_speed_range_m_s[1]:
            raise ValueError("supported speed range must be ordered")
        scalar = (
            self.source_cycle_duration_s,
            self.source_speed_m_s,
            self.pose_seam_rms_rad,
            self.velocity_seam_rms_rad_s,
            self.interpolation_derivative_rms_rad_s,
            self.reference_ground_offset_m,
            *self.supported_speed_range_m_s,
        )
        if any(not math.isfinite(value) or value < 0 for value in scalar):
            raise ValueError("reference speed map values must be finite and nonnegative")


def _touchdowns(contact: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    prior = np.vstack((contact[-1:], contact[:-1]))
    return [(int(frame), int(foot)) for frame, foot in np.argwhere(contact & ~prior)]


def _step_geometry(
    sole_position: npt.NDArray[np.float64], contact: npt.NDArray[np.bool_]
) -> tuple[float, float]:
    events = sorted(_touchdowns(contact))
    cycle_displacement = float(np.mean(sole_position[-1, :, 0] - sole_position[0, :, 0]))
    steps: list[float] = []
    strides: list[float] = []
    for previous, current in zip(events, events[1:], strict=False):
        if previous[1] != current[1]:
            steps.append(
                abs(
                    float(
                        sole_position[current[0], current[1], 0]
                        - sole_position[previous[0], previous[1], 0]
                    )
                )
            )
    if len(events) > 1 and events[-1][1] != events[0][1]:
        steps.append(
            abs(
                float(
                    sole_position[events[0][0], events[0][1], 0]
                    + cycle_displacement
                    - sole_position[events[-1][0], events[-1][1], 0]
                )
            )
        )
    for foot in range(2):
        indices = [frame for frame, side in events if side == foot]
        for previous_frame, current_frame in zip(indices, indices[1:], strict=False):
            strides.append(
                abs(
                    float(
                        sole_position[current_frame, foot, 0]
                        - sole_position[previous_frame, foot, 0]
                    )
                )
            )
        if indices:
            strides.append(
                abs(
                    float(
                        sole_position[indices[0], foot, 0]
                        + cycle_displacement
                        - sole_position[indices[-1], foot, 0]
                    )
                )
            )
    if not steps or not strides:
        raise ValueError("reference must contain alternating touchdown and same-foot stride events")
    return float(np.median(steps)), float(np.median(strides))


def _sole_kinematics(
    model_path: Path,
    root_position: npt.NDArray[np.float64],
    root_quaternion: npt.NDArray[np.float64],
    joint_names: tuple[str, ...],
    joint_position: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in ("left_foot", "right_foot")
    ]
    if any(site_id < 0 for site_id in site_ids):
        raise ValueError("model must expose left_foot and right_foot sites")
    qpos_addresses = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"model does not contain reference joint {name!r}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
    positions = np.empty((len(joint_position), 2, 3), dtype=np.float64)
    for frame in range(len(joint_position)):
        data.qpos[:3] = root_position[frame]
        data.qpos[3:7] = root_quaternion[frame]
        data.qpos[qpos_addresses] = joint_position[frame]
        mujoco.mj_forward(model, data)
        positions[frame] = data.site_xpos[site_ids]
    return positions, model


def audit_reference_speed_map(
    reference_path: Path,
    model_path: Path,
    controller_path: Path,
    *,
    reference_id: str,
    source_speed_m_s: float,
    cycle_duration_s: float,
    speeds_m_s: tuple[float, ...] = (0.4, 0.6, 0.8),
    reference_ground_offset_m: float = 0.03,
    criteria: ReferenceAuditCriteria | None = None,
) -> ReferenceSpeedMap:
    """Audit a spatially fixed, time-warped reference against the pinned robot."""
    if not speeds_m_s or any(not math.isfinite(speed) or speed <= 0 for speed in speeds_m_s):
        raise ValueError("audit speeds must be finite and positive")
    criteria = criteria or ReferenceAuditCriteria()
    if not math.isfinite(reference_ground_offset_m) or reference_ground_offset_m < 0:
        raise ValueError("reference ground offset must be finite and nonnegative")
    with np.load(reference_path, allow_pickle=False) as archive:
        joint_position = np.asarray(archive["joint_pos"], dtype=np.float64)
        source_joint_velocity = np.asarray(archive["joint_vel"], dtype=np.float64)
        body_position = np.asarray(archive["body_pos_w"], dtype=np.float64)
        body_quaternion = np.asarray(archive["body_quat_w"], dtype=np.float64)
        contact = np.asarray(archive["foot_contact"], dtype=bool)
        joint_names = tuple(str(name) for name in archive["joint_names"].tolist())
        body_names = tuple(str(name) for name in archive["body_names"].tolist())
        fps = float(archive["fps"][0])
    if joint_position.shape != source_joint_velocity.shape or joint_position.shape[1] != 29:
        raise ValueError("reference joint arrays must have matching (frames, 29) shapes")
    if not np.isclose(fps * cycle_duration_s, len(joint_position) - 1, atol=1e-6):
        raise ValueError("cycle duration does not match reference frame intervals")
    pelvis = body_names.index("pelvis")
    sole_position, model = _sole_kinematics(
        model_path,
        body_position[:, pelvis],
        body_quaternion[:, pelvis],
        joint_names,
        joint_position,
    )
    step_length, stride_length = _step_geometry(sole_position, contact)
    dt = 1.0 / fps
    finite_difference = np.diff(joint_position, axis=0) / dt
    midpoint_velocity = 0.5 * (source_joint_velocity[:-1] + source_joint_velocity[1:])
    interpolation_error = float(np.sqrt(np.mean(np.square(finite_difference - midpoint_velocity))))
    pose_seam = float(np.sqrt(np.mean(np.square(joint_position[-1] - joint_position[0]))))
    velocity_seam = float(
        np.sqrt(np.mean(np.square(source_joint_velocity[-1] - source_joint_velocity[0])))
    )
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    damping = -np.asarray(controller["actuator_bias_parameters"], dtype=float)[:, 2]
    armature = np.asarray(controller["armature"], dtype=float)
    force_limits = np.max(
        np.abs(np.asarray(controller["actuator_force_limits"], dtype=float)), axis=1
    )
    if np.any(force_limits <= 0):
        raise ValueError("controller actuator force limits must be positive")
    hard_limits = np.asarray(controller["hard_joint_limits"], dtype=float)
    controller_identity = str(controller.get("sha256") or _sha256(controller_path))
    acceleration = np.gradient(source_joint_velocity, dt, axis=0)
    joint_margin = np.minimum(
        joint_position - hard_limits[:, 0], hard_limits[:, 1] - joint_position
    )
    sole_relative = sole_position - body_position[:, pelvis, None, :]
    corrected_clearance = sole_position[:, :, 2] + reference_ground_offset_m
    entries: list[ReferenceSpeedEntry] = []
    for speed in speeds_m_s:
        scale = speed / source_speed_m_s
        scaled_velocity = source_joint_velocity * scale
        scaled_acceleration = acceleration * scale**2
        actuator_proxy = damping * scaled_velocity + armature * scaled_acceleration
        demand_ratio = float(np.max(np.abs(actuator_proxy) / force_limits))
        violations: list[str] = []
        if float(np.min(joint_margin)) < criteria.minimum_joint_limit_margin_rad:
            violations.append("joint_limit_margin")
        if demand_ratio > criteria.maximum_actuator_demand_ratio:
            violations.append("actuator_demand")
        if float(np.min(corrected_clearance)) < criteria.minimum_corrected_sole_clearance_m:
            violations.append("terrain_penetration")
        entries.append(
            ReferenceSpeedEntry(
                speed_m_s=speed,
                time_scale=scale,
                cycle_period_s=cycle_duration_s / scale,
                expected_cadence_steps_s=2.0 * scale / cycle_duration_s,
                step_length_median_m=step_length,
                stride_length_median_m=stride_length,
                maximum_joint_velocity_rad_s=float(np.max(np.abs(scaled_velocity))),
                maximum_joint_acceleration_rad_s2=float(np.max(np.abs(scaled_acceleration))),
                minimum_joint_limit_margin_rad=float(np.min(joint_margin)),
                maximum_actuator_demand_ratio=demand_ratio,
                sole_clearance_range_m=(
                    float(np.min(corrected_clearance)),
                    float(np.max(corrected_clearance)),
                ),
                contact_fraction=(float(np.mean(contact[:, 0])), float(np.mean(contact[:, 1]))),
                double_support_fraction=float(np.mean(np.all(contact, axis=1))),
                flight_fraction=float(np.mean(~np.any(contact, axis=1))),
                reachable_sole_forward_range_m=(
                    float(np.min(sole_relative[:, :, 0])),
                    float(np.max(sole_relative[:, :, 0])),
                ),
                passed=not violations,
                violations=tuple(violations),
            )
        )
    supported = [entry.speed_m_s for entry in entries if entry.passed]
    if not supported:
        raise ValueError("no audited speed passes reference feasibility")
    global_violations: list[str] = []
    if pose_seam > criteria.maximum_pose_seam_rms_rad:
        global_violations.append("pose_seam")
    if velocity_seam > criteria.maximum_velocity_seam_rms_rad_s:
        global_violations.append("velocity_seam")
    if interpolation_error > criteria.maximum_interpolation_derivative_rms_rad_s:
        global_violations.append("interpolation_derivative")
    if (
        abs(float(np.mean(contact[:, 0]) - np.mean(contact[:, 1])))
        > criteria.maximum_contact_fraction_asymmetry
    ):
        global_violations.append("contact_fraction_asymmetry")
    return ReferenceSpeedMap(
        schema_version=2,
        reference_id=reference_id,
        reference_sha256=_sha256(reference_path),
        model_sha256=_normalized_text_sha256(model_path),
        controller_sha256=controller_identity,
        criteria=criteria,
        transformation="retain spatial trajectory; scale phase rate, velocities by s, and accelerations by s^2",
        source_cycle_duration_s=cycle_duration_s,
        source_speed_m_s=source_speed_m_s,
        pose_seam_rms_rad=pose_seam,
        velocity_seam_rms_rad_s=velocity_seam,
        interpolation_derivative_rms_rad_s=interpolation_error,
        reference_ground_offset_m=reference_ground_offset_m,
        supported_speed_range_m_s=(min(supported), max(supported)),
        entries=tuple(entries),
        passed=not global_violations and all(entry.passed for entry in entries),
        violations=tuple(global_violations),
    )


def write_reference_speed_map(value: ReferenceSpeedMap, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output)


def load_reference_speed_map(path: Path) -> ReferenceSpeedMap:
    """Strictly load the frozen map; unknown or missing fields fail."""
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {field.name for field in fields(ReferenceSpeedMap)}:
        raise ValueError("reference speed map fields do not match schema")
    raw_entries = raw["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("reference speed map entries must be a nonempty list")
    expected_entry = {field.name for field in fields(ReferenceSpeedEntry)}
    entries: list[ReferenceSpeedEntry] = []
    tuple_fields = {
        "sole_clearance_range_m",
        "contact_fraction",
        "reachable_sole_forward_range_m",
        "violations",
    }
    for value in raw_entries:
        if not isinstance(value, dict) or set(value) != expected_entry:
            raise ValueError("reference speed entry fields do not match schema")
        converted: dict[str, Any] = dict(value)
        for key in tuple_fields:
            converted[key] = tuple(converted[key])
        entries.append(ReferenceSpeedEntry(**converted))
    converted_root = dict(raw)
    raw_criteria = raw["criteria"]
    if not isinstance(raw_criteria, dict) or set(raw_criteria) != {
        field.name for field in fields(ReferenceAuditCriteria)
    }:
        raise ValueError("reference audit criteria fields do not match schema")
    converted_root["criteria"] = ReferenceAuditCriteria(**raw_criteria)
    converted_root["entries"] = tuple(entries)
    converted_root["supported_speed_range_m_s"] = tuple(raw["supported_speed_range_m_s"])
    converted_root["violations"] = tuple(raw["violations"])
    return ReferenceSpeedMap(**converted_root)
