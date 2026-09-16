"""Native kinematic admission audit for soft walking references."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import sha256_file, write_atomic_json


@dataclass(frozen=True, slots=True)
class KinematicAdmissionCriteria:
    minimum_joint_margin_rad: float = 0.019
    maximum_foot_ik_error_m: float = 0.010
    maximum_ground_penetration_m: float = 0.005
    maximum_self_penetration_m: float = 0.002
    maximum_stride_error_m: float = 0.020
    maximum_joint_seam_error_rad: float = 0.050
    maximum_quaternion_norm_error: float = 1e-4


def _model_id(mujoco: Any, model: Any, kind: Any, name: str) -> int:
    identifier = int(mujoco.mj_name2id(model, kind, name))
    if identifier < 0:
        identifier = int(mujoco.mj_name2id(model, kind, f"robot/{name}"))
    return identifier


def audit_reference_kinematics(
    reference_path: Path,
    model_path: Path,
    adaptation_path: Path,
    output_path: Path,
    *,
    speed_m_s: float,
    criteria: KinematicAdmissionCriteria | None = None,
) -> dict[str, Any]:  # pragma: no cover - native MuJoCo evidence path
    """Inspect arrays and collision geometry without advancing simulation dynamics."""
    import json

    import mujoco

    criteria = criteria or KinematicAdmissionCriteria()
    adaptation = json.loads(adaptation_path.read_text(encoding="utf-8"))
    if not isinstance(adaptation, dict):
        raise ValueError("adaptation report must contain an object")
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    with np.load(reference_path, allow_pickle=False) as archive:
        required = {
            "joint_pos",
            "joint_vel",
            "joint_names",
            "body_pos_w",
            "body_quat_w",
            "body_names",
            "foot_contact",
            "fps",
        }
        if not required.issubset(archive.files):
            raise ValueError("reference archive is missing required arrays")
        joint_position = np.asarray(archive["joint_pos"], dtype=np.float64)
        joint_velocity = np.asarray(archive["joint_vel"], dtype=np.float64)
        body_position = np.asarray(archive["body_pos_w"], dtype=np.float64)
        body_quaternion = np.asarray(archive["body_quat_w"], dtype=np.float64)
        contact = np.asarray(archive["foot_contact"], dtype=np.uint8)
        fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
        joint_names = tuple(str(value) for value in archive["joint_names"].tolist())
        body_names = tuple(str(value) for value in archive["body_names"].tolist())
    arrays = (joint_position, joint_velocity, body_position, body_quaternion, contact)
    finite = all(np.isfinite(value).all() for value in arrays) and math.isfinite(fps) and fps > 0
    if (
        len(joint_names) != 29
        or joint_position.shape != joint_velocity.shape
        or joint_position.ndim != 2
        or joint_position.shape[1] != 29
        or len(joint_position) < 17
        or body_position.shape[:2] != body_quaternion.shape[:2]
        or len(body_position) != len(joint_position)
        or contact.shape != (len(joint_position), 2)
    ):
        raise ValueError("reference array shapes do not satisfy the G1 cycle contract")
    pelvis = body_names.index("pelvis")
    joint_ids = [_model_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    if any(identifier < 0 for identifier in joint_ids):
        raise ValueError("reference joint is absent from the model")
    qpos_addresses = [int(model.jnt_qposadr[identifier]) for identifier in joint_ids]
    robot_root_id = int(model.body_rootid[int(model.jnt_bodyid[joint_ids[0]])])
    ranges = np.asarray(model.jnt_range[joint_ids], dtype=np.float64)
    joint_margin = float(
        np.min(
            np.minimum(
                joint_position - ranges[:, 0],
                ranges[:, 1] - joint_position,
            )
        )
    )
    joint_seam_error = float(np.max(np.abs(joint_position[-1] - joint_position[0])))
    quaternion_norm_error = float(
        np.max(np.abs(np.linalg.norm(body_quaternion[:, pelvis], axis=1) - 1.0))
    )
    period_s = (len(joint_position) - 1) / fps
    displacement_m = float(body_position[-1, pelvis, 0] - body_position[0, pelvis, 0])
    stride_error_m = abs(displacement_m - speed_m_s * period_s)
    flight_frame_count = int(np.count_nonzero(np.sum(contact[:-1], axis=1) == 0))
    maximum_ground_penetration = 0.0
    maximum_self_penetration = 0.0
    forbidden_ground_contacts = 0
    collision_depth_by_pair: dict[tuple[str, str], float] = {}
    for frame in range(len(joint_position) - 1):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = body_position[frame, pelvis]
        data.qpos[3:7] = body_quaternion[frame, pelvis]
        data.qpos[qpos_addresses] = joint_position[frame]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        for collision in data.contact:
            if collision.dist >= 0:
                continue
            first, second = int(collision.geom[0]), int(collision.geom[1])
            first_body = int(model.geom_bodyid[first])
            second_body = int(model.geom_bodyid[second])
            first_is_robot = int(model.body_rootid[first_body]) == robot_root_id
            second_is_robot = int(model.body_rootid[second_body]) == robot_root_id
            first_name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, first))
            second_name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, second))
            depth = -float(collision.dist)
            pair = (first_name, second_name)
            collision_depth_by_pair[pair] = max(collision_depth_by_pair.get(pair, 0.0), depth)
            if first_is_robot != second_is_robot:
                maximum_ground_penetration = max(maximum_ground_penetration, depth)
                robot_geom = first_name if first_is_robot else second_name
                if "foot" not in robot_geom.lower() and depth > 1e-4:
                    forbidden_ground_contacts += 1
            elif first_is_robot and second_is_robot:
                maximum_self_penetration = max(maximum_self_penetration, depth)
    foot_ik_error = float(adaptation.get("maximum_foot_error_m", math.inf))
    reasons: list[str] = []
    checks = {
        "finite": finite,
        "joint_margin": joint_margin + 1e-12 >= criteria.minimum_joint_margin_rad,
        "foot_ik": foot_ik_error <= criteria.maximum_foot_ik_error_m,
        "ground_penetration": maximum_ground_penetration <= criteria.maximum_ground_penetration_m,
        "self_penetration": maximum_self_penetration <= criteria.maximum_self_penetration_m,
        "forbidden_ground_contacts": forbidden_ground_contacts == 0,
        "stride": stride_error_m <= criteria.maximum_stride_error_m,
        "joint_seam": joint_seam_error <= criteria.maximum_joint_seam_error_rad,
        "quaternion_norm": quaternion_norm_error <= criteria.maximum_quaternion_norm_error,
        "no_flight_gap": flight_frame_count == 0,
    }
    reasons.extend(name for name, passed in checks.items() if not passed)
    collisions = [
        {"geoms": list(pair), "maximum_depth_m": depth}
        for pair, depth in sorted(
            collision_depth_by_pair.items(), key=lambda item: item[1], reverse=True
        )[:20]
    ]
    result = {
        "schema_version": 2,
        "semantics": "soft-reference-kinematic-admission",
        "reference_sha256": sha256_file(reference_path),
        "model_sha256": sha256_file(model_path),
        "adaptation_sha256": sha256_file(adaptation_path),
        "speed_m_s": speed_m_s,
        "criteria": asdict(criteria),
        "measurements": {
            "finite": finite,
            "minimum_joint_margin_rad": joint_margin,
            "maximum_foot_ik_error_m": foot_ik_error,
            "maximum_ground_penetration_m": maximum_ground_penetration,
            "maximum_self_penetration_m": maximum_self_penetration,
            "forbidden_ground_contact_count": forbidden_ground_contacts,
            "joint_seam_error_rad": joint_seam_error,
            "quaternion_norm_error": quaternion_norm_error,
            "cycle_period_s": period_s,
            "forward_displacement_m": displacement_m,
            "stride_error_m": stride_error_m,
            "flight_frame_count": flight_frame_count,
        },
        "checks": checks,
        "largest_collision_penetrations": collisions,
        "reasons": reasons,
        "passed": not reasons,
    }
    write_atomic_json(output_path, result)
    return result
