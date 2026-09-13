"""Bounded collision repair for periodic G1 walking references."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..artifacts import sha256_file, write_atomic_json

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CollisionRepairSettings:
    """Frozen bounds for the deterministic collision-repair search."""

    target_ground_penetration_m: float = 0.001
    target_self_penetration_m: float = 0.001
    maximum_root_lift_m: float = 0.080
    arm_blend_increment: float = 0.01
    source_joint_margin_rad: float = 0.023

    def validate(self) -> None:
        values = (
            self.target_ground_penetration_m,
            self.target_self_penetration_m,
            self.maximum_root_lift_m,
            self.arm_blend_increment,
            self.source_joint_margin_rad,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("collision repair settings must be finite and positive")
        if self.target_ground_penetration_m > 0.005:
            raise ValueError("ground target cannot weaken the admission threshold")
        if self.target_self_penetration_m > 0.002:
            raise ValueError("self-collision target cannot weaken the admission threshold")
        if self.maximum_root_lift_m > 0.080:
            raise ValueError("root repair is bounded to 0.080 m")
        if self.arm_blend_increment > 0.05:
            raise ValueError("arm search increment must be at most 0.05")
        if not 0.019 <= self.source_joint_margin_rad <= 0.05:
            raise ValueError("source joint margin must reserve interpolation headroom")


def _model_id(mujoco: Any, model: Any, kind: Any, name: str) -> int:
    identifier = int(mujoco.mj_name2id(model, kind, name))
    if identifier < 0:
        identifier = int(mujoco.mj_name2id(model, kind, f"robot/{name}"))
    return identifier


def repair_reference_collisions(
    reference_path: Path,
    model_path: Path,
    adaptation_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    settings: CollisionRepairSettings | None = None,
) -> dict[str, Any]:  # pragma: no cover - native MuJoCo evidence path
    """Lift the root and minimally blend arms toward nominal until contacts clear."""
    import mujoco

    settings = settings or CollisionRepairSettings()
    settings.validate()
    inherited = json.loads(adaptation_path.read_text(encoding="utf-8"))
    if not isinstance(inherited, dict) or inherited.get("passed") is not True:
        raise ValueError("collision repair requires a passing adaptation report")
    with np.load(reference_path, allow_pickle=False) as archive:
        arrays: dict[str, Any] = {name: np.asarray(archive[name]) for name in archive.files}
    joint_position = np.asarray(arrays["joint_pos"], dtype=np.float64).copy()
    body_position = np.asarray(arrays["body_pos_w"], dtype=np.float64).copy()
    body_quaternion = np.asarray(arrays["body_quat_w"], dtype=np.float64).copy()
    joint_names = tuple(str(value) for value in arrays["joint_names"].tolist())
    body_names = tuple(str(value) for value in arrays["body_names"].tolist())
    pelvis = body_names.index("pelvis")
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    joint_ids = [_model_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    body_ids = [_model_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names]
    if any(identifier < 0 for identifier in (*joint_ids, *body_ids)):
        raise ValueError("reference names do not match the repair model")
    qpos_addresses = np.asarray(
        [model.jnt_qposadr[identifier] for identifier in joint_ids], dtype=np.int64
    )
    robot_root_id = int(model.body_rootid[int(model.jnt_bodyid[joint_ids[0]])])
    arm_indices = np.asarray(
        [
            index
            for index, name in enumerate(joint_names)
            if any(token in name for token in ("shoulder_", "elbow_", "wrist_"))
        ],
        dtype=np.int64,
    )
    nominal = np.asarray(model.qpos0[qpos_addresses], dtype=np.float64)
    joint_ranges = np.asarray(model.jnt_range[joint_ids], dtype=np.float64)

    def collision_metrics(root_lift_m: float, arm_blend: float) -> dict[str, Any]:
        maximum_ground = 0.0
        maximum_self = 0.0
        forbidden_ground = 0
        for frame in range(len(joint_position) - 1):
            pose = joint_position[frame].copy()
            pose[arm_indices] = (1.0 - arm_blend) * pose[arm_indices] + arm_blend * nominal[
                arm_indices
            ]
            data.qpos[:] = model.qpos0
            data.qpos[:3] = body_position[frame, pelvis]
            data.qpos[2] += root_lift_m
            data.qpos[3:7] = body_quaternion[frame, pelvis]
            data.qpos[qpos_addresses] = pose
            mujoco.mj_forward(model, data)
            for collision in data.contact:
                if collision.dist >= 0:
                    continue
                first, second = int(collision.geom[0]), int(collision.geom[1])
                first_body = int(model.geom_bodyid[first])
                second_body = int(model.geom_bodyid[second])
                first_robot = int(model.body_rootid[first_body]) == robot_root_id
                second_robot = int(model.body_rootid[second_body]) == robot_root_id
                depth = -float(collision.dist)
                if first_robot != second_robot:
                    maximum_ground = max(maximum_ground, depth)
                    robot_geom = first if first_robot else second
                    name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom))
                    if "foot" not in name.lower() and depth > 1e-4:
                        forbidden_ground += 1
                elif first_robot and second_robot:
                    maximum_self = max(maximum_self, depth)
        return {
            "maximum_ground_penetration_m": maximum_ground,
            "maximum_self_penetration_m": maximum_self,
            "forbidden_ground_contact_count": forbidden_ground,
        }

    before = collision_metrics(0.0, 0.0)
    root_lift = max(
        0.0,
        float(before["maximum_ground_penetration_m"]) - settings.target_ground_penetration_m + 1e-6,
    )
    if root_lift > settings.maximum_root_lift_m:
        raise ValueError("required root lift exceeds the frozen repair bound")
    arm_blend = math.nan
    after: dict[str, Any] | None = None
    steps = int(round(1.0 / settings.arm_blend_increment))
    for step in range(steps + 1):
        candidate_blend = min(1.0, step * settings.arm_blend_increment)
        candidate_metrics = collision_metrics(root_lift, candidate_blend)
        if (
            candidate_metrics["maximum_self_penetration_m"] <= settings.target_self_penetration_m
            and candidate_metrics["maximum_ground_penetration_m"]
            <= settings.target_ground_penetration_m
            and candidate_metrics["forbidden_ground_contact_count"] == 0
        ):
            arm_blend = candidate_blend
            after = candidate_metrics
            break
    if after is None:
        raise ValueError("bounded arm-to-nominal search did not remove collisions")

    joint_position[:, arm_indices] = (1.0 - arm_blend) * joint_position[
        :, arm_indices
    ] + arm_blend * nominal[arm_indices]
    body_position[:, pelvis, 2] += root_lift
    pre_margin_joint_position = joint_position.copy()
    joint_position = np.clip(
        joint_position,
        joint_ranges[:, 0] + settings.source_joint_margin_rad,
        joint_ranges[:, 1] - settings.source_joint_margin_rad,
    )
    site_ids = [
        _model_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in ("left_foot", "right_foot")
    ]
    if any(identifier < 0 for identifier in site_ids):
        raise ValueError("repair model is missing foot sites")
    maximum_foot_change = 0.0
    repaired_body_position = np.empty_like(body_position)
    repaired_body_quaternion = np.empty_like(body_quaternion)
    for frame in range(len(joint_position)):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = body_position[frame, pelvis]
        data.qpos[3:7] = body_quaternion[frame, pelvis]
        data.qpos[qpos_addresses] = pre_margin_joint_position[frame]
        mujoco.mj_forward(model, data)
        foot_before = data.site_xpos[site_ids].copy()
        data.qpos[qpos_addresses] = joint_position[frame]
        mujoco.mj_forward(model, data)
        maximum_foot_change = max(
            maximum_foot_change,
            float(np.max(np.linalg.norm(data.site_xpos[site_ids] - foot_before, axis=1))),
        )
        repaired_body_position[frame] = data.xpos[body_ids]
        repaired_body_quaternion[frame] = data.xquat[body_ids]
    after = collision_metrics(root_lift, 0.0)
    maximum_foot_error = float(inherited["maximum_foot_error_m"]) + maximum_foot_change
    passed = (
        after["maximum_ground_penetration_m"] <= 0.005
        and after["maximum_self_penetration_m"] <= 0.002
        and after["forbidden_ground_contact_count"] == 0
        and maximum_foot_error <= 0.010
    )
    fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
    joint_velocity = np.gradient(joint_position, 1.0 / fps, axis=0)
    body_linear_velocity = np.gradient(repaired_body_position, 1.0 / fps, axis=0)
    arrays.update(
        {
            "joint_pos": joint_position,
            "joint_vel": joint_velocity,
            "body_pos_w": repaired_body_position,
            "body_quat_w": repaired_body_quaternion,
            "body_lin_vel_w": body_linear_velocity,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    report = {
        "schema_version": 2,
        "semantics": "bounded-reference-collision-repair",
        "source_sha256": sha256_file(reference_path),
        "model_sha256": sha256_file(model_path),
        "inherited_adaptation_sha256": sha256_file(adaptation_path),
        "output_sha256": sha256_file(output_path),
        "settings": asdict(settings),
        "root_lift_m": root_lift,
        "arm_blend_to_model_nominal": arm_blend,
        "before": before,
        "after": after,
        "maximum_foot_error_m": maximum_foot_error,
        "maximum_foot_path_change_from_margin_m": maximum_foot_change,
        "foot_path_transform": "uniform root/target vertical translation; local path unchanged",
        "passed": passed,
    }
    write_atomic_json(report_path, report)
    return report
