"""Bounded, auditable speed adaptation for retargeted periodic walking motion."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..artifacts import sha256_file, write_atomic_json

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AdaptationSettings:
    speed_m_s: float
    cycle_period_s: float
    output_frames: int = 64
    joint_harmonics: int = 6
    arm_harmonics: int = 3
    root_harmonics: int = 4
    joint_limit_margin_rad: float = 0.019
    foot_position_weight: float = 100.0
    root_regularization_weight: float = 2.0
    root_temporal_regularization_weight: float = 10.0
    pose_regularization_weight: float = 0.10
    temporal_regularization_weight: float = 0.05
    maximum_foot_error_m: float = 0.01

    def validate(self) -> None:
        if not math.isfinite(self.speed_m_s) or not 0 < self.speed_m_s <= 0.8:
            raise ValueError("adaptation speed must be in (0, 0.8]")
        if not math.isfinite(self.cycle_period_s) or self.cycle_period_s <= 0:
            raise ValueError("cycle period must be finite and positive")
        if self.output_frames < 16:
            raise ValueError("adaptation requires at least 16 output frames")
        if min(self.joint_harmonics, self.arm_harmonics, self.root_harmonics) < 1:
            raise ValueError("harmonic counts must be positive")


def periodic_fourier_resample(
    values: npt.ArrayLike,
    output_frames: int,
    maximum_harmonic: int,
) -> FloatArray:
    """Least-squares periodic resampling with an explicit frequency cutoff."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim < 1 or len(source) < 4 or output_frames < 4:
        raise ValueError("periodic resampling requires at least four frames")
    if not 1 <= maximum_harmonic < len(source) // 2 + 1:
        raise ValueError("maximum harmonic is outside the supported source range")
    source_phase = np.arange(len(source), dtype=np.float64) / len(source)
    target_phase = np.arange(output_frames, dtype=np.float64) / output_frames

    def basis(phase: FloatArray) -> FloatArray:
        columns = [np.ones_like(phase)]
        for harmonic in range(1, maximum_harmonic + 1):
            columns.extend(
                [
                    np.sin(2 * np.pi * harmonic * phase),
                    np.cos(2 * np.pi * harmonic * phase),
                ]
            )
        return np.stack(columns, axis=1)

    flattened = source.reshape(len(source), -1)
    coefficients, *_ = np.linalg.lstsq(basis(source_phase), flattened, rcond=None)
    return (basis(target_phase) @ coefficients).reshape((output_frames, *source.shape[1:]))


def _normalized_linear_resample(values: FloatArray, output_frames: int) -> FloatArray:
    source_phase = np.arange(len(values), dtype=np.float64) / len(values)
    target_phase = np.arange(output_frames, dtype=np.float64) / output_frames
    flat = values.reshape(len(values), -1)
    extended_phase = np.append(source_phase, 1.0)
    extended = np.concatenate((flat, flat[:1]), axis=0)
    result = np.stack(
        [
            np.interp(target_phase, extended_phase, extended[:, column])
            for column in range(flat.shape[1])
        ],
        axis=1,
    ).reshape((output_frames, *values.shape[1:]))
    return np.asarray(result, dtype=np.float64)


def _model_id(mujoco: Any, model: Any, kind: Any, name: str) -> int:
    identifier = int(mujoco.mj_name2id(model, kind, name))
    return identifier if identifier >= 0 else int(mujoco.mj_name2id(model, kind, f"robot/{name}"))


def _foot_ik_residual(
    candidate: FloatArray,
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    target_root_position: FloatArray,
    root_quaternion: FloatArray,
    complete_pose: FloatArray,
    qpos_addresses: npt.NDArray[np.int64],
    leg_qpos: npt.NDArray[np.int64],
    site_ids: list[int],
    target_foot: FloatArray,
    target_pose: FloatArray,
    previous_pose: FloatArray,
    previous_root_position: FloatArray,
    settings: AdaptationSettings,
) -> FloatArray:
    root_position = candidate[:3]
    leg_position = candidate[3:]
    data.qpos[:] = model.qpos0
    data.qpos[:3] = root_position
    data.qpos[3:7] = root_quaternion
    data.qpos[qpos_addresses] = complete_pose
    data.qpos[leg_qpos] = leg_position
    mujoco.mj_forward(model, data)
    foot_error = (data.site_xpos[site_ids] - target_foot).reshape(-1)
    return np.concatenate(
        (
            settings.foot_position_weight * foot_error,
            settings.root_regularization_weight * (root_position - target_root_position),
            settings.root_temporal_regularization_weight * (root_position - previous_root_position),
            settings.pose_regularization_weight * (leg_position - target_pose),
            settings.temporal_regularization_weight * (leg_position - previous_pose),
        )
    )


def adapt_reference_for_speed(
    source_path: Path,
    model_path: Path,
    output_path: Path,
    report_path: Path,
    settings: AdaptationSettings,
) -> dict[str, Any]:  # pragma: no cover - native MuJoCo evidence path
    """Fit a periodic speed-specific reference while preserving world foot paths with IK."""
    import mujoco
    from scipy.optimize import least_squares  # type: ignore[import-untyped]

    settings.validate()
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    joint_names = tuple(str(name) for name in arrays["joint_names"].tolist())
    body_names = tuple(str(name) for name in arrays["body_names"].tolist())
    source_joint = np.asarray(arrays["joint_pos"], dtype=np.float64)
    source_body_position = np.asarray(arrays["body_pos_w"], dtype=np.float64)
    source_body_quaternion = np.asarray(arrays["body_quat_w"], dtype=np.float64)
    source_contact = np.asarray(arrays["foot_contact"], dtype=np.uint8)
    pelvis = body_names.index("pelvis")
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    joint_ids = [_model_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    qpos_addresses = np.asarray([model.jnt_qposadr[joint] for joint in joint_ids], dtype=np.int64)
    site_ids = [
        _model_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in ("left_foot", "right_foot")
    ]
    if any(identifier < 0 for identifier in (*joint_ids, *site_ids)):
        raise ValueError("model does not satisfy adaptation joint/site contract")
    leg_indices = np.asarray(
        [
            index
            for index, name in enumerate(joint_names)
            if any(token in name for token in ("hip_", "knee_", "ankle_"))
        ],
        dtype=np.int64,
    )
    arm_indices = np.asarray(
        [
            index
            for index, name in enumerate(joint_names)
            if "shoulder_" in name or "elbow_" in name or "wrist_" in name
        ],
        dtype=np.int64,
    )
    leg_qpos = qpos_addresses[leg_indices]
    source_frames = len(source_joint)
    output_frames = settings.output_frames
    source_pose = periodic_fourier_resample(source_joint, output_frames, settings.joint_harmonics)
    source_pose[:, arm_indices] = periodic_fourier_resample(
        source_joint[:, arm_indices], output_frames, settings.arm_harmonics
    )
    root = source_body_position[:, pelvis]
    source_phase = np.arange(source_frames, dtype=np.float64) / source_frames
    target_phase = np.arange(output_frames, dtype=np.float64) / output_frames
    source_stride = float((root[-1, 0] - root[0, 0]) / source_phase[-1])
    if source_stride <= 0:
        raise ValueError("source reference must progress forward")
    detrended_root = root.copy()
    detrended_root[:, 0] -= source_phase * source_stride
    target_stride = settings.speed_m_s * settings.cycle_period_s
    root_quaternion = _normalized_linear_resample(source_body_quaternion[:, pelvis], output_frames)
    root_quaternion /= np.linalg.norm(root_quaternion, axis=1, keepdims=True)

    source_resampled_root = periodic_fourier_resample(
        detrended_root, output_frames, settings.root_harmonics
    )
    source_resampled_root[:, 0] += target_phase * source_stride
    adapted_root = source_resampled_root.copy()
    adapted_root[:, 0] = source_resampled_root[0, 0] + (
        source_resampled_root[:, 0] - source_resampled_root[0, 0]
    ) * (target_stride / source_stride)
    source_resampled_quaternion = _normalized_linear_resample(
        source_body_quaternion[:, pelvis], output_frames
    )
    source_resampled_quaternion /= np.linalg.norm(
        source_resampled_quaternion, axis=1, keepdims=True
    )
    target_foot = np.empty((output_frames, 2, 3), dtype=np.float64)
    for frame in range(output_frames):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = source_resampled_root[frame]
        data.qpos[3:7] = source_resampled_quaternion[frame]
        data.qpos[qpos_addresses] = source_pose[frame]
        mujoco.mj_forward(model, data)
        target_foot[frame] = data.site_xpos[site_ids]
    origin_x = float(target_foot[0, 0, 0])
    target_foot[:, :, 0] = origin_x + (target_foot[:, :, 0] - origin_x) * (
        target_stride / source_stride
    )

    lower = np.asarray([model.jnt_range[joint, 0] for joint in np.asarray(joint_ids)[leg_indices]])
    upper = np.asarray([model.jnt_range[joint, 1] for joint in np.asarray(joint_ids)[leg_indices]])
    lower += settings.joint_limit_margin_rad
    upper -= settings.joint_limit_margin_rad
    adapted_joint = source_pose.copy()
    convergence: list[dict[str, Any]] = []
    previous = np.clip(source_pose[0, leg_indices], lower, upper)
    previous_root = adapted_root[0].copy()
    for frame in range(output_frames):
        target_pose = np.clip(source_pose[frame, leg_indices], lower, upper)

        root_target = adapted_root[frame].copy()
        candidate_lower = np.concatenate((root_target - [0.03, 0.05, 0.08], lower))
        candidate_upper = np.concatenate((root_target + [0.03, 0.05, 0.08], upper))
        optimized = least_squares(
            partial(
                _foot_ik_residual,
                mujoco=mujoco,
                model=model,
                data=data,
                target_root_position=root_target,
                root_quaternion=root_quaternion[frame],
                complete_pose=source_pose[frame],
                qpos_addresses=qpos_addresses,
                leg_qpos=leg_qpos,
                site_ids=site_ids,
                target_foot=target_foot[frame],
                target_pose=target_pose,
                previous_pose=previous,
                previous_root_position=previous_root,
                settings=settings,
            ),
            np.concatenate((root_target, previous)),
            bounds=(candidate_lower, candidate_upper),
            max_nfev=300,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        adapted_root[frame] = optimized.x[:3]
        adapted_joint[frame, leg_indices] = optimized.x[3:]
        previous = optimized.x[3:]
        previous_root = optimized.x[:3]
        data.qpos[:] = model.qpos0
        data.qpos[:3] = adapted_root[frame]
        data.qpos[3:7] = root_quaternion[frame]
        data.qpos[qpos_addresses] = adapted_joint[frame]
        mujoco.mj_forward(model, data)
        foot_error = float(
            np.max(np.linalg.norm(data.site_xpos[site_ids] - target_foot[frame], axis=1))
        )
        convergence.append(
            {
                "frame": frame,
                "success": bool(optimized.success),
                "evaluations": int(optimized.nfev),
                "maximum_foot_error_m": foot_error,
            }
        )

    closed_joint = np.concatenate((adapted_joint, adapted_joint[:1]), axis=0)
    body_position = _normalized_linear_resample(source_body_position, output_frames)
    body_quaternion = _normalized_linear_resample(source_body_quaternion, output_frames)
    body_position[:, pelvis] = adapted_root
    body_quaternion[:, pelvis] = root_quaternion
    closed_body_position = np.concatenate((body_position, body_position[:1]), axis=0)
    closed_body_position[-1, :, 0] += target_stride
    closed_body_quaternion = np.concatenate((body_quaternion, body_quaternion[:1]), axis=0)
    dt = settings.cycle_period_s / output_frames
    joint_velocity = np.gradient(closed_joint, dt, axis=0)
    if settings.joint_harmonics <= 4:
        foot_velocity = np.asarray(np.gradient(target_foot, dt, axis=0), dtype=np.float64)
        height = target_foot[:, :, 2] - np.min(target_foot[:, :, 2], axis=0)
        support_score = height + 0.20 * np.linalg.norm(foot_velocity[:, :, :2], axis=2)
        contact = np.zeros((output_frames, 2), dtype=np.uint8)
        support_count = int(math.ceil(0.60 * output_frames))
        for foot in range(2):
            contact[np.argsort(support_score[:, foot])[:support_count], foot] = 1
        flight = np.flatnonzero(np.sum(contact, axis=1) == 0)
        contact[flight, np.argmin(support_score[flight], axis=1)] = 1
    else:
        contact_indices = np.floor(np.arange(output_frames) / output_frames * source_frames).astype(
            np.int64
        )
        contact = source_contact[contact_indices]
    closed_contact = np.concatenate((contact, contact[:1]), axis=0)
    body_linear_velocity = np.gradient(closed_body_position, dt, axis=0)
    body_angular_velocity = np.zeros_like(body_linear_velocity)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        body_ang_vel_w=body_angular_velocity,
        body_lin_vel_w=body_linear_velocity,
        body_names=arrays["body_names"],
        body_pos_w=closed_body_position,
        body_quat_w=closed_body_quaternion,
        foot_contact=closed_contact,
        fps=np.asarray([1.0 / dt]),
        joint_names=arrays["joint_names"],
        joint_pos=closed_joint,
        joint_vel=joint_velocity,
    )
    maximum_error = max(float(item["maximum_foot_error_m"]) for item in convergence)
    report = {
        "schema_version": 1,
        "source_sha256": sha256_file(source_path),
        "model_sha256": sha256_file(model_path),
        "output_sha256": sha256_file(output_path),
        "settings": asdict(settings),
        "source_stride_m": source_stride,
        "target_stride_m": target_stride,
        "objective": {
            "foot_position": "preserve spatially scaled world path, including stance stationarity",
            "root_progression": "speed times declared cycle period",
            "joint_limits": "hard bounds with declared margin",
            "pose_and_temporal_regularization": "least-squares terms with declared weights",
            "arm_motion": "periodic Fourier acceleration smoothing; no global angle shrink",
        },
        "contact_label_rule": (
            "lowest foot-height-plus-horizontal-speed score; 60% duty per foot; no flight"
            if settings.joint_harmonics <= 4
            else "source labels resampled by nearest phase"
        ),
        "preprocessing_revision": "contact-constrained-root-foot-ik-v2",
        "converged_frames": sum(bool(item["success"]) for item in convergence),
        "frame_count": output_frames,
        "maximum_foot_error_m": maximum_error,
        "passed": all(bool(item["success"]) for item in convergence)
        and maximum_error <= settings.maximum_foot_error_m,
        "frames": convergence,
    }
    write_atomic_json(report_path, report)
    return report
