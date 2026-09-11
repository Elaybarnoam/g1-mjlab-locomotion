"""Local-only conversion of a G1 joint motion into mjlab-compatible kinematics."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .reference import (
    G1_JOINT_NAMES,
    MotionSeries,
    MotionValidationError,
    audit_periodic_motion,
    derive_foot_contacts,
    load_soma_csv,
    resample_motion,
)

FloatArray = npt.NDArray[np.float64]


def write_deterministic_npz(path: Path, arrays: dict[str, Any]) -> None:
    """Write unpickled NumPy members with stable ordering, timestamps and permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o644 << 16
            archive.writestr(
                member, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
            )


def _multiply_quaternions(left: FloatArray, right: FloatArray) -> FloatArray:
    lw, lx, ly, lz = left.T
    rw, rx, ry, rz = right.T
    return np.column_stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def normalize_forward_cycle(series: MotionSeries, *, first: int, last: int) -> MotionSeries:
    """Select an inclusive cycle and align its horizontal displacement with +X."""
    if first < 0 or last >= series.frame_count or last - first < 2:
        raise MotionValidationError("cycle frame range is outside the source motion")
    root = series.root_position_m[first : last + 1].copy()
    quaternion = series.root_quaternion_wxyz[first : last + 1].copy()
    displacement = root[-1, :2] - root[0, :2]
    if np.linalg.norm(displacement) < 1e-6:
        raise MotionValidationError("cycle has no measurable horizontal displacement")
    heading = np.arctan2(displacement[1], displacement[0])
    cosine, sine = np.cos(-heading), np.sin(-heading)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    origin = root[0, :2].copy()
    root[:, :2] = (root[:, :2] - origin) @ rotation.T
    yaw = np.zeros_like(quaternion)
    yaw[:, 0] = np.cos(-heading / 2.0)
    yaw[:, 3] = np.sin(-heading / 2.0)
    quaternion = _multiply_quaternions(yaw, quaternion)
    result = replace(
        series,
        root_position_m=root,
        root_quaternion_wxyz=quaternion,
        joint_position_rad=series.joint_position_rad[first : last + 1].copy(),
    )
    result.validate()
    return result


def _quaternion_angular_velocity(quaternion: FloatArray, fps: float) -> FloatArray:
    relative = _multiply_quaternions(
        quaternion[1:], quaternion[:-1] * np.asarray([1.0, -1.0, -1.0, -1.0])
    )
    relative = np.where((relative[:, :1] < 0), -relative, relative)
    vector_norm = np.linalg.norm(relative[:, 1:], axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[:, 0], -1.0, 1.0))
    axis = np.divide(
        relative[:, 1:],
        vector_norm[:, None],
        out=np.zeros_like(relative[:, 1:]),
        where=vector_norm[:, None] > 1e-10,
    )
    interval = axis * angle[:, None] * fps
    return np.vstack((interval[:1], (interval[:-1] + interval[1:]) / 2.0, interval[-1:]))


def compute_mujoco_kinematics(series: MotionSeries, *, model_xml: Path) -> dict[str, Any]:
    """Run deterministic MuJoCo forward kinematics; no simulation steps or network I/O."""
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError("motion preparation requires the 'native' package extra") from error

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in G1_JOINT_NAMES
    ]
    if any(identifier < 0 for identifier in joint_ids):
        raise MotionValidationError("MuJoCo model does not expose the complete G1 joint contract")
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joints) != 1:
        raise MotionValidationError("G1 model must have exactly one floating-base joint")
    root_address = int(model.jnt_qposadr[free_joints[0]])
    body_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        for body_id in range(1, model.nbody)
    )
    body_position = np.empty((series.frame_count, len(body_names), 3), dtype=np.float64)
    body_quaternion = np.empty((series.frame_count, len(body_names), 4), dtype=np.float64)
    qpos_addresses = [int(model.jnt_qposadr[identifier]) for identifier in joint_ids]
    for frame in range(series.frame_count):
        data.qpos[root_address : root_address + 3] = series.root_position_m[frame]
        data.qpos[root_address + 3 : root_address + 7] = series.root_quaternion_wxyz[frame]
        data.qpos[qpos_addresses] = series.joint_position_rad[frame]
        mujoco.mj_forward(model, data)
        body_position[frame] = data.xpos[1:]
        body_quaternion[frame] = data.xquat[1:]
    dt = 1.0 / series.fps
    return {
        "fps": np.asarray([series.fps], dtype=np.float32),
        "joint_pos": series.joint_position_rad.astype(np.float32),
        "joint_vel": np.gradient(series.joint_position_rad, dt, axis=0).astype(np.float32),
        "body_pos_w": body_position.astype(np.float32),
        "body_quat_w": body_quaternion.astype(np.float32),
        "body_lin_vel_w": np.asarray(np.gradient(body_position, dt, axis=0)).astype(np.float32),
        "body_ang_vel_w": np.stack(
            [
                _quaternion_angular_velocity(body_quaternion[:, index], series.fps)
                for index in range(len(body_names))
            ],
            axis=1,
        ).astype(np.float32),
        "body_names": np.asarray(body_names),
        "joint_names": np.asarray(G1_JOINT_NAMES),
    }


def prepare_reference(
    *,
    source_csv: Path,
    model_xml: Path,
    output_npz: Path,
    output_audit: Path,
    source_fps: float,
    output_fps: float,
    first_frame: int,
    last_frame: int,
) -> dict[str, Any]:
    """Prepare, hash and audit one selected gait cycle as local deterministic artifacts."""
    source = load_soma_csv(source_csv, fps=source_fps)
    selected = normalize_forward_cycle(source, first=first_frame, last=last_frame)
    sampled = resample_motion(selected, output_fps=output_fps)
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in G1_JOINT_NAMES]
    )
    joint_range = model.jnt_range[joint_ids]
    safety_margin_rad = 0.02
    constrained_joint_position = np.clip(
        sampled.joint_position_rad,
        joint_range[None, :, 0] + safety_margin_rad,
        joint_range[None, :, 1] - safety_margin_rad,
    )
    maximum_constraint_adjustment = float(
        np.max(np.abs(constrained_joint_position - sampled.joint_position_rad))
    )
    constrained_samples = int(
        np.count_nonzero(np.abs(constrained_joint_position - sampled.joint_position_rad) > 1e-12)
    )
    sampled = replace(sampled, joint_position_rad=constrained_joint_position)
    arrays = compute_mujoco_kinematics(sampled, model_xml=model_xml)
    body_names = arrays["body_names"].tolist()
    try:
        feet = [
            body_names.index(name) for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]
    except ValueError as error:
        raise MotionValidationError("G1 model is missing an ankle body") from error
    foot_position = arrays["body_pos_w"][:, feet]
    floor = float(np.min(foot_position[:, :, 2]))
    foot_height = foot_position[:, :, 2] - floor
    foot_speed = np.linalg.norm(
        np.gradient(foot_position[:, :, :2], 1.0 / output_fps, axis=0), axis=2
    )
    contacts = np.column_stack(
        [
            derive_foot_contacts(
                foot_height[:, side],
                foot_speed[:, side],
                enter_height_m=0.04,
                exit_height_m=0.065,
                max_stance_speed_m_s=0.8,
                minimum_frames=max(2, round(output_fps * 0.04)),
            )
            for side in range(2)
        ]
    )
    arrays["foot_contact"] = contacts
    write_deterministic_npz(output_npz, arrays)
    audit = audit_periodic_motion(
        arrays["joint_pos"],
        sampled.root_position_m,
        sampled.root_quaternion_wxyz,
        contacts,
        fps=output_fps,
    )
    lower_margin = arrays["joint_pos"] - joint_range[None, :, 0]
    upper_margin = joint_range[None, :, 1] - arrays["joint_pos"]
    all_finite = all(
        bool(np.all(np.isfinite(value)))
        for value in arrays.values()
        if np.issubdtype(np.asarray(value).dtype, np.number)
    )
    result: dict[str, Any] = {
        **audit.to_dict(),
        "schema_version": 1,
        "source_frames_inclusive": [first_frame, last_frame],
        "source_fps": source_fps,
        "output_fps": output_fps,
        "output_frames": sampled.frame_count,
        "contact_frames": {"left": int(contacts[:, 0].sum()), "right": int(contacts[:, 1].sum())},
        "double_support_frames": int(np.all(contacts, axis=1).sum()),
        "flight_frames": int((~np.any(contacts, axis=1)).sum()),
        "all_values_finite": all_finite,
        "joint_limits_valid": bool(np.min(lower_margin) >= 0 and np.min(upper_margin) >= 0),
        "joint_limit_safety_margin_rad": safety_margin_rad,
        "joint_limit_constrained_samples": constrained_samples,
        "max_joint_limit_adjustment_rad": maximum_constraint_adjustment,
        "minimum_joint_limit_margin_rad": float(min(np.min(lower_margin), np.min(upper_margin))),
        "max_absolute_joint_velocity_rad_s": float(np.max(np.abs(arrays["joint_vel"]))),
        "root_height_range_m": [
            float(np.min(arrays["body_pos_w"][:, 0, 2])),
            float(np.max(arrays["body_pos_w"][:, 0, 2])),
        ],
        "acceptance": {
            "joint_pose_seam_rms_rad_max": 0.05,
            "joint_velocity_seam_rms_rad_s_max": 1.5,
            "root_orientation_seam_rad_max": 0.05,
            "contact_seam_required": True,
            "passed": bool(
                all_finite
                and np.min(lower_margin) >= 0
                and np.min(upper_margin) >= 0
                and audit.joint_pose_seam_rms_rad <= 0.05
                and audit.joint_velocity_seam_rms_rad_s <= 1.5
                and audit.root_orientation_seam_rad <= 0.05
                and audit.contact_seam_matches
                and np.all(contacts.sum(axis=0) > 0)
            ),
        },
        "npz_sha256": hashlib.sha256(output_npz.read_bytes()).hexdigest(),
        "npz_bytes": output_npz.stat().st_size,
    }
    output_audit.parent.mkdir(parents=True, exist_ok=True)
    output_audit.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def render_reference_preview(
    *,
    reference_npz: Path,
    model_xml: Path,
    output_mp4: Path,
    width: int = 640,
    height: int = 480,
) -> dict[str, Any]:
    """Render the exact reference at its declared real-time frame rate."""
    try:
        import imageio.v2 as imageio
        import mujoco
    except ImportError as error:
        raise RuntimeError("preview rendering requires the 'native' and 'video' extras") from error
    with np.load(reference_npz, allow_pickle=False) as reference:
        fps = float(reference["fps"][0])
        joint_position = reference["joint_pos"]
        body_position = reference["body_pos_w"]
        body_quaternion = reference["body_quat_w"]
        joint_names = tuple(reference["joint_names"].tolist())
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    root_joint = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
    root_address = int(model.jnt_qposadr[root_joint])
    addresses = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)])
        for joint_name in joint_names
    ]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [float(body_position[:, 0, 0].mean()), 0.0, 0.8]
    camera.distance = 2.8
    camera.azimuth = 90.0
    camera.elevation = -8.0
    renderer = mujoco.Renderer(model, width=width, height=height)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    if output_mp4.exists():
        raise FileExistsError(f"preview already exists: {output_mp4}")
    try:
        with imageio.get_writer(
            output_mp4,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:
            for frame in range(joint_position.shape[0]):
                data.qpos[root_address : root_address + 3] = body_position[frame, 0]
                data.qpos[root_address + 3 : root_address + 7] = body_quaternion[frame, 0]
                data.qpos[addresses] = joint_position[frame]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())  # type: ignore[attr-defined]
    finally:
        renderer.close()
    return {
        "fps": fps,
        "frames": int(joint_position.shape[0]),
        "duration_s": (joint_position.shape[0] - 1) / fps,
        "sha256": hashlib.sha256(output_mp4.read_bytes()).hexdigest(),
        "bytes": output_mp4.stat().st_size,
    }
