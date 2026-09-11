"""Strict local ingestion and auditing for cyclic G1 motion references."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_ROOT_COLUMNS = (
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
)


class MotionValidationError(ValueError):
    """Raised when motion bytes do not satisfy their declared contract."""


@dataclass(frozen=True, slots=True)
class MotionSeries:
    """Floating-base joint motion in SI units and MuJoCo G1 joint order."""

    fps: float
    root_position_m: FloatArray
    root_quaternion_wxyz: FloatArray
    joint_position_rad: FloatArray
    joint_names: tuple[str, ...] = G1_JOINT_NAMES

    @property
    def frame_count(self) -> int:
        return int(self.joint_position_rad.shape[0])

    @property
    def duration_s(self) -> float:
        return (self.frame_count - 1) / self.fps

    def validate(self) -> None:
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise MotionValidationError("fps must be finite and positive")
        frame_count = self.frame_count
        expected = ((frame_count, 3), (frame_count, 4), (frame_count, len(self.joint_names)))
        actual = (
            self.root_position_m.shape,
            self.root_quaternion_wxyz.shape,
            self.joint_position_rad.shape,
        )
        if frame_count < 2 or actual != expected:
            raise MotionValidationError(
                f"invalid motion array shapes: {actual}; expected {expected}"
            )
        for name, values in (
            ("root position", self.root_position_m),
            ("root quaternion", self.root_quaternion_wxyz),
            ("joint position", self.joint_position_rad),
        ):
            if not np.all(np.isfinite(values)):
                raise MotionValidationError(f"{name} contains non-finite values")
        norms = np.linalg.norm(self.root_quaternion_wxyz, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            raise MotionValidationError("root quaternions are not normalized")


@dataclass(frozen=True, slots=True)
class MotionAsset:
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class MotionManifest:
    schema_version: int
    reference_id: str
    redistribution: str
    assets: tuple[MotionAsset, ...]
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Self:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            assets = tuple(MotionAsset(**item) for item in raw["assets"])
            manifest = cls(
                schema_version=int(raw["schema_version"]),
                reference_id=str(raw["reference_id"]),
                redistribution=str(raw["redistribution"]),
                assets=assets,
                raw=raw,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise MotionValidationError(f"invalid motion manifest {path}: {error}") from error
        if manifest.schema_version != 1 or not manifest.assets:
            raise MotionValidationError("unsupported or empty motion manifest")
        return manifest

    def verify_assets(self, root: Path) -> None:
        root = root.resolve()
        for asset in self.assets:
            path = (root / asset.path).resolve()
            if root not in path.parents:
                raise MotionValidationError(f"asset escapes manifest directory: {asset.path}")
            if not path.is_file() or path.stat().st_size != asset.bytes:
                raise MotionValidationError(f"asset size mismatch: {asset.path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest.lower() != asset.sha256.lower():
                raise MotionValidationError(f"asset SHA-256 mismatch: {asset.path}")


@dataclass(frozen=True, slots=True)
class MotionAudit:
    duration_s: float
    forward_speed_m_s: float
    joint_pose_seam_rms_rad: float
    joint_velocity_seam_rms_rad_s: float
    root_orientation_seam_rad: float
    contact_seam_matches: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "duration_s": self.duration_s,
            "forward_speed_m_s": self.forward_speed_m_s,
            "joint_pose_seam_rms_rad": self.joint_pose_seam_rms_rad,
            "joint_velocity_seam_rms_rad_s": self.joint_velocity_seam_rms_rad_s,
            "root_orientation_seam_rad": self.root_orientation_seam_rad,
            "contact_seam_matches": self.contact_seam_matches,
        }


def _intrinsic_xyz_degrees_to_wxyz(euler_degrees: FloatArray) -> FloatArray:
    half = np.deg2rad(euler_degrees) / 2.0
    cx, cy, cz = np.cos(half).T
    sx, sy, sz = np.sin(half).T
    return np.column_stack(
        (
            cx * cy * cz - sx * sy * sz,
            sx * cy * cz + cx * sy * sz,
            cx * sy * cz - sx * cy * sz,
            cx * cy * sz + sx * sy * cz,
        )
    )


def load_soma_csv(path: Path, *, fps: float) -> MotionSeries:
    """Load NVIDIA SOMA's flat G1 CSV format into a strict SI representation."""
    expected_columns = (*_ROOT_COLUMNS, *(f"{name}_dof" for name in G1_JOINT_NAMES))
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            columns = tuple(next(reader))
        except StopIteration as error:
            raise MotionValidationError("motion CSV is empty") from error
        if columns != expected_columns:
            raise MotionValidationError("motion CSV joint column order does not match G1 contract")
        try:
            values = np.asarray(
                [[float(value) for value in row] for row in reader], dtype=np.float64
            )
        except ValueError as error:
            raise MotionValidationError(f"motion CSV contains non-numeric data: {error}") from error
    if values.ndim != 2 or values.shape[1] != len(expected_columns):
        raise MotionValidationError("motion CSV has an invalid row width")
    frame_ids = values[:, 0]
    if not np.array_equal(frame_ids, np.arange(len(frame_ids))):
        raise MotionValidationError("motion CSV frame identifiers must be contiguous from zero")
    series = MotionSeries(
        fps=fps,
        root_position_m=values[:, 1:4] / 100.0,
        root_quaternion_wxyz=_intrinsic_xyz_degrees_to_wxyz(values[:, 4:7]),
        joint_position_rad=np.deg2rad(values[:, 7:]),
    )
    series.validate()
    return series


def _slerp(q0: FloatArray, q1: FloatArray, blend: FloatArray) -> FloatArray:
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where((dot < 0)[:, None], -q1, q1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    near = sine < 1e-8
    left = np.divide(np.sin((1.0 - blend) * angle), sine, out=1.0 - blend, where=~near)
    right = np.divide(np.sin(blend * angle), sine, out=blend, where=~near)
    result = left[:, None] * q0 + right[:, None] * q1
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def resample_motion(source: MotionSeries, *, output_fps: float) -> MotionSeries:
    source.validate()
    if not np.isfinite(output_fps) or output_fps <= 0:
        raise MotionValidationError("output fps must be finite and positive")
    count = int(round(source.duration_s * output_fps)) + 1
    times = np.linspace(0.0, source.duration_s, count)
    coordinates = times * source.fps
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, source.frame_count - 1)
    blend = coordinates - lower

    def lerp(values: FloatArray) -> FloatArray:
        return values[lower] * (1.0 - blend[:, None]) + values[upper] * blend[:, None]

    result = MotionSeries(
        fps=output_fps,
        root_position_m=lerp(source.root_position_m),
        root_quaternion_wxyz=_slerp(
            source.root_quaternion_wxyz[lower], source.root_quaternion_wxyz[upper], blend
        ),
        joint_position_rad=lerp(source.joint_position_rad),
        joint_names=source.joint_names,
    )
    result.validate()
    return result


def derive_foot_contacts(
    height_m: npt.ArrayLike,
    horizontal_speed_m_s: npt.ArrayLike,
    *,
    enter_height_m: float,
    exit_height_m: float,
    max_stance_speed_m_s: float | None,
    minimum_frames: int,
) -> BoolArray:
    """Derive one foot's contact state using hysteresis and debounce duration."""
    height = np.asarray(height_m, dtype=np.float64)
    speed = np.asarray(horizontal_speed_m_s, dtype=np.float64)
    if height.shape != speed.shape or height.ndim != 1 or minimum_frames < 1:
        raise MotionValidationError("contact inputs must be equal one-dimensional arrays")
    if enter_height_m > exit_height_m:
        raise MotionValidationError("contact enter height must not exceed exit height")
    output = np.empty(height.shape, dtype=np.bool_)
    speed_allowed = (
        np.ones_like(speed, dtype=np.bool_)
        if max_stance_speed_m_s is None
        else speed <= max_stance_speed_m_s
    )
    state = bool(height[0] <= enter_height_m and speed_allowed[0])
    pending = state
    count = 0
    for index, (sample_height, sample_speed_allowed) in enumerate(
        zip(height, speed_allowed, strict=True)
    ):
        candidate = state
        if state and (sample_height >= exit_height_m or not sample_speed_allowed):
            candidate = False
        elif not state and sample_height <= enter_height_m and sample_speed_allowed:
            candidate = True
        if candidate == state:
            count = 0
        elif candidate != pending:
            pending, count = candidate, 1
        else:
            count += 1
            if count >= minimum_frames:
                state, count = candidate, 0
        output[index] = state
    return output


def _quaternion_distance(q0: FloatArray, q1: FloatArray) -> float:
    dot = float(np.clip(abs(np.dot(q0, q1)), 0.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def audit_periodic_motion(
    joint_position_rad: npt.ArrayLike,
    root_position_m: npt.ArrayLike,
    root_quaternion_wxyz: npt.ArrayLike,
    contacts: npt.ArrayLike,
    *,
    fps: float,
) -> MotionAudit:
    joint = np.asarray(joint_position_rad, dtype=np.float64)
    root = np.asarray(root_position_m, dtype=np.float64)
    quaternion = np.asarray(root_quaternion_wxyz, dtype=np.float64)
    contact = np.asarray(contacts, dtype=np.bool_)
    frame_count = joint.shape[0]
    if (
        frame_count < 3
        or root.shape != (frame_count, 3)
        or quaternion.shape != (frame_count, 4)
        or contact.shape != (frame_count, 2)
        or not all(np.all(np.isfinite(array)) for array in (joint, root, quaternion))
    ):
        raise MotionValidationError("periodic audit received invalid arrays")
    velocity = np.gradient(joint, 1.0 / fps, axis=0)
    duration = (frame_count - 1) / fps
    displacement = root[-1, :2] - root[0, :2]
    return MotionAudit(
        duration_s=duration,
        forward_speed_m_s=float(np.linalg.norm(displacement) / duration),
        joint_pose_seam_rms_rad=float(np.sqrt(np.mean(np.square(joint[-1] - joint[0])))),
        joint_velocity_seam_rms_rad_s=float(
            np.sqrt(np.mean(np.square(velocity[-1] - velocity[0])))
        ),
        root_orientation_seam_rad=_quaternion_distance(quaternion[0], quaternion[-1]),
        contact_seam_matches=bool(np.array_equal(contact[0], contact[-1])),
    )
