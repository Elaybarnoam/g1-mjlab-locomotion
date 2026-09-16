"""Constrained contact-wrench inverse-dynamics audit for walking references."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..artifacts import sha256_file, write_atomic_json

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class WrenchSolution:
    success: bool
    wrenches: FloatArray
    normalized_base_residual: float
    optimizer_message: str


@dataclass(frozen=True, slots=True)
class DynamicsCriteria:
    minimum_feasible_fraction: float = 0.95
    maximum_normalized_base_residual: float = 0.05
    maximum_torque_p95_ratio: float = 0.80
    maximum_torque_peak_ratio: float = 1.00
    friction_coefficient: float = 0.7
    foot_half_length_m: float = 0.10
    foot_half_width_m: float = 0.04


def solve_contact_wrenches(
    required_base_wrench: npt.ArrayLike,
    contact_jacobians: npt.ArrayLike,
    active_contacts: npt.ArrayLike,
    *,
    friction_coefficient: float,
    foot_half_length_m: float,
    foot_half_width_m: float,
    total_weight_n: float,
    robot_height_m: float,
    required_actuator_torque: npt.ArrayLike | None = None,
    actuator_contact_jacobians: npt.ArrayLike | None = None,
    actuator_force_limits: npt.ArrayLike | None = None,
) -> WrenchSolution:
    """Fit two foot wrenches under unilateral, friction-cone and support-polygon bounds."""
    from scipy.optimize import LinearConstraint, minimize  # type: ignore[import-untyped]

    required = np.asarray(required_base_wrench, dtype=np.float64)
    jacobians = np.asarray(contact_jacobians, dtype=np.float64)
    active = np.asarray(active_contacts, dtype=bool)
    if required.shape != (6,) or jacobians.shape != (2, 6, 6) or active.shape != (2,):
        raise ValueError("wrench solver inputs have invalid shapes")
    scalars = (
        friction_coefficient,
        foot_half_length_m,
        foot_half_width_m,
        total_weight_n,
        robot_height_m,
    )
    if any(not math.isfinite(value) or value <= 0 for value in scalars):
        raise ValueError("wrench solver scales must be finite and positive")
    mapping = np.concatenate([jacobians[foot].T for foot in range(2)], axis=1)
    actuator_required = (
        None
        if required_actuator_torque is None
        else np.asarray(required_actuator_torque, dtype=np.float64)
    )
    actuator_jacobians = (
        None
        if actuator_contact_jacobians is None
        else np.asarray(actuator_contact_jacobians, dtype=np.float64)
    )
    force_limits = (
        None
        if actuator_force_limits is None
        else np.asarray(actuator_force_limits, dtype=np.float64)
    )
    actuator_inputs = (actuator_required, actuator_jacobians, force_limits)
    if any(value is None for value in actuator_inputs) and not all(
        value is None for value in actuator_inputs
    ):
        raise ValueError("actuator wrench constraints must be supplied together")
    actuator_mapping: FloatArray | None = None
    if (
        actuator_required is not None
        and actuator_jacobians is not None
        and force_limits is not None
    ):
        if (
            actuator_required.ndim != 1
            or actuator_jacobians.shape != (2, 6, len(actuator_required))
            or force_limits.shape != actuator_required.shape
            or np.any(force_limits <= 0)
        ):
            raise ValueError("actuator wrench constraint shapes are invalid")
        actuator_mapping = np.concatenate([actuator_jacobians[foot].T for foot in range(2)], axis=1)
    normalization = np.array(
        [total_weight_n] * 3 + [total_weight_n * robot_height_m] * 3,
        dtype=np.float64,
    )

    def objective(flat: FloatArray) -> float:
        residual = (mapping @ flat - required) / normalization
        actuator_cost = 0.0
        if (
            actuator_mapping is not None
            and actuator_required is not None
            and force_limits is not None
        ):
            torque_ratio = (actuator_required - actuator_mapping @ flat) / force_limits
            actuator_cost = 1e-4 * float(torque_ratio @ torque_ratio)
        return float(
            residual @ residual + actuator_cost + 1e-10 * (flat @ flat) / total_weight_n**2
        )

    constraint_rows: list[FloatArray] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []

    def add_inequality(row: FloatArray, lower: float, upper: float = math.inf) -> None:
        constraint_rows.append(row)
        constraint_lower.append(lower)
        constraint_upper.append(upper)

    for foot in range(2):
        offset = foot * 6
        if not active[foot]:
            continue
        normal = np.zeros(12)
        normal[offset + 2] = 1
        add_inequality(normal, 0)
        for component, scale in (
            (0, friction_coefficient),
            (1, friction_coefficient),
            (3, foot_half_width_m),
            (4, foot_half_length_m),
            (5, 0.02),
        ):
            for sign in (-1.0, 1.0):
                row = np.zeros(12)
                row[offset + 2] = scale
                row[offset + component] = sign
                add_inequality(row, 0)
    if actuator_mapping is not None and actuator_required is not None and force_limits is not None:
        for actuator in range(len(actuator_required)):
            add_inequality(
                actuator_mapping[actuator],
                actuator_required[actuator] - force_limits[actuator],
                actuator_required[actuator] + force_limits[actuator],
            )
    linear_constraint = LinearConstraint(
        np.stack(constraint_rows),
        np.asarray(constraint_lower),
        np.asarray(constraint_upper),
    )
    bounds = [(None, None) if active[foot] else (0.0, 0.0) for foot in range(2) for _ in range(6)]
    initial = np.zeros(12, dtype=np.float64)
    if np.any(active):
        initial.reshape(2, 6)[active, 2] = max(required[2], total_weight_n) / np.count_nonzero(
            active
        )
    optimized = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=[linear_constraint],
        options={"ftol": 1e-10, "maxiter": 200},
    )
    wrenches = np.asarray(optimized.x, dtype=np.float64).reshape(2, 6)
    residual = (mapping @ optimized.x - required) / normalization
    return WrenchSolution(
        bool(optimized.success),
        wrenches,
        float(np.linalg.norm(residual)),
        str(optimized.message),
    )


def _periodic_qvel(model: Any, qpos: FloatArray, dt: float) -> FloatArray:
    import mujoco

    count = len(qpos)
    qvel = np.empty((count, model.nv), dtype=np.float64)
    for frame in range(count):
        previous = qpos[(frame - 1) % count].copy()
        following = qpos[(frame + 1) % count].copy()
        displacement = qpos[-1, 0] - qpos[0, 0]
        if frame == 0:
            previous[0] -= displacement
        if frame == count - 1:
            following[0] += displacement
        mujoco.mj_differentiatePos(model, qvel[frame], 2 * dt, previous, following)
    return qvel


def actuator_force_limits_by_joint(model: Any, joint_ids: list[int]) -> FloatArray:
    """Return actuator force limits in the requested joint order, never declaration order."""
    actuator_by_joint = {
        int(model.actuator_trnid[actuator, 0]): actuator
        for actuator in range(model.nu)
        if int(model.actuator_trnid[actuator, 0]) >= 0
    }
    if len(actuator_by_joint) != model.nu or any(
        joint not in actuator_by_joint for joint in joint_ids
    ):
        raise ValueError("model must map exactly one actuator to every reference joint")
    limits = np.asarray(
        [
            np.max(np.abs(model.actuator_forcerange[actuator_by_joint[joint]]))
            for joint in joint_ids
        ],
        dtype=np.float64,
    )
    if np.any(limits <= 0):
        raise ValueError("reference joint actuators must have positive force limits")
    return limits


def audit_reference_dynamics(
    reference_path: Path,
    model_path: Path,
    output: Path,
    *,
    speeds_m_s: tuple[float, ...] = (0.4, 0.6, 0.8),
    source_speed_m_s: float = 1.16381159304071,
    criteria: DynamicsCriteria | None = None,
) -> dict[str, Any]:  # pragma: no cover - native MuJoCo evidence path
    import mujoco

    criteria = criteria or DynamicsCriteria()
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    with np.load(reference_path, allow_pickle=False) as reference:
        joint_position = np.asarray(reference["joint_pos"], dtype=np.float64)[:-1]
        body_position = np.asarray(reference["body_pos_w"], dtype=np.float64)[:-1]
        body_quaternion = np.asarray(reference["body_quat_w"], dtype=np.float64)[:-1]
        contact = np.asarray(reference["foot_contact"], dtype=bool)[:-1]
        joint_names = tuple(str(name) for name in reference["joint_names"].tolist())
        body_names = tuple(str(name) for name in reference["body_names"].tolist())
        fps = float(np.asarray(reference["fps"]).reshape(-1)[0])
    pelvis = body_names.index("pelvis")

    def model_id(kind: Any, name: str) -> int:
        result = int(mujoco.mj_name2id(model, kind, name))
        return result if result >= 0 else int(mujoco.mj_name2id(model, kind, f"robot/{name}"))

    joint_ids = [model_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    if any(joint_id < 0 for joint_id in joint_ids):
        raise ValueError("reference joint is absent from model")
    qpos_addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    dof_addresses = [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
    site_ids = [model_id(mujoco.mjtObj.mjOBJ_SITE, name) for name in ("left_foot", "right_foot")]
    if any(site_id < 0 for site_id in site_ids):
        raise ValueError("model must expose left_foot and right_foot sites")
    qpos = np.zeros((len(joint_position), model.nq), dtype=np.float64)
    qpos[:] = model.qpos0
    qpos[:, :3] = body_position[:, pelvis]
    qpos[:, 3:7] = body_quaternion[:, pelvis]
    qpos[:, qpos_addresses] = joint_position
    source_qvel = _periodic_qvel(model, qpos, 1 / fps)
    source_qacc = np.gradient(source_qvel, 1 / fps, axis=0)
    gravity = abs(float(model.opt.gravity[2]))
    total_weight = float(np.sum(model.body_mass) * gravity)
    robot_height = float(np.max(model.body_pos[:, 2]) - np.min(model.body_pos[:, 2]))
    force_limits = actuator_force_limits_by_joint(model, joint_ids)
    speed_results: list[dict[str, Any]] = []
    for speed in speeds_m_s:
        scale = speed / source_speed_m_s
        residuals: list[float] = []
        torque_ratios: list[float] = []
        torque_ratio_by_joint: list[FloatArray] = []
        gravity_ratio_by_joint: list[FloatArray] = []
        inertia_ratio_by_joint: list[FloatArray] = []
        velocity_passive_ratio_by_joint: list[FloatArray] = []
        contact_ratio_by_joint: list[FloatArray] = []
        successes = 0
        optimizer_failures = 0
        for frame in range(len(qpos)):
            data.qpos[:] = qpos[frame]
            data.qvel[:] = source_qvel[frame] * scale
            data.qacc[:] = source_qacc[frame] * scale**2
            # Build the unconstrained generalized demand explicitly. ``mj_inverse`` also
            # solves contacts already present in the model and folds those solver forces
            # into qfrc_inverse, which would double-count contact in this audit.
            mujoco.mj_forward(model, data)
            inertia_force = np.empty(model.nv, dtype=np.float64)
            mujoco.mj_mulM(model, data, inertia_force, data.qacc)
            bias_force = np.asarray(data.qfrc_bias, dtype=np.float64).copy()
            passive_force = np.asarray(data.qfrc_passive, dtype=np.float64).copy()
            saved_velocity = data.qvel.copy()
            data.qvel[:] = 0
            data.qacc[:] = 0
            mujoco.mj_forward(model, data)
            gravity_force = np.asarray(data.qfrc_bias, dtype=np.float64).copy()
            data.qvel[:] = saved_velocity
            data.qacc[:] = source_qacc[frame] * scale**2
            required = inertia_force + bias_force - passive_force
            jacobians = np.zeros((2, 6, model.nv), dtype=np.float64)
            for foot, site_id in enumerate(site_ids):
                mujoco.mj_jacSite(model, data, jacobians[foot, :3], jacobians[foot, 3:], site_id)
            solution = solve_contact_wrenches(
                required[:6],
                jacobians[:, :, :6],
                contact[frame],
                friction_coefficient=criteria.friction_coefficient,
                foot_half_length_m=criteria.foot_half_length_m,
                foot_half_width_m=criteria.foot_half_width_m,
                total_weight_n=total_weight,
                robot_height_m=robot_height,
                required_actuator_torque=required[dof_addresses],
                actuator_contact_jacobians=jacobians[:, :, dof_addresses],
                actuator_force_limits=force_limits,
            )
            residuals.append(solution.normalized_base_residual)
            optimizer_failures += int(not solution.success)
            contact_generalized = sum(
                jacobians[foot].T @ solution.wrenches[foot] for foot in range(2)
            )
            actuator_required = required[dof_addresses] - contact_generalized[dof_addresses]
            ratios = np.abs(actuator_required) / force_limits
            torque_ratios.extend(ratios.tolist())
            torque_ratio_by_joint.append(ratios)
            gravity_ratio_by_joint.append(np.abs(gravity_force[dof_addresses]) / force_limits)
            inertia_ratio_by_joint.append(np.abs(inertia_force[dof_addresses]) / force_limits)
            velocity_passive_ratio_by_joint.append(
                np.abs((bias_force - gravity_force - passive_force)[dof_addresses]) / force_limits
            )
            contact_ratio_by_joint.append(np.abs(contact_generalized[dof_addresses]) / force_limits)
            successes += int(
                solution.success
                and solution.normalized_base_residual <= criteria.maximum_normalized_base_residual
            )
        feasible_fraction = successes / len(qpos)
        torque_p95 = float(np.percentile(torque_ratios, 95))
        torque_peak = float(np.max(torque_ratios))
        torque_matrix = np.stack(torque_ratio_by_joint)

        def component_summary(values: list[FloatArray]) -> dict[str, float]:
            matrix = np.stack(values)
            return {
                "rms_ratio": float(np.sqrt(np.mean(matrix**2))),
                "p95_ratio": float(np.percentile(matrix, 95)),
                "peak_ratio": float(np.max(matrix)),
            }

        per_joint = [
            {
                "joint": name,
                "torque_ratio_p95": float(np.percentile(torque_matrix[:, index], 95)),
                "torque_ratio_peak": float(np.max(torque_matrix[:, index])),
            }
            for index in np.argsort(np.max(torque_matrix, axis=0))[::-1]
            for name in (joint_names[int(index)],)
        ]
        speed_results.append(
            {
                "speed_m_s": speed,
                "feasible_wrench_fraction": feasible_fraction,
                "normalized_base_residual_p95": float(np.percentile(residuals, 95)),
                "normalized_base_residual_peak": float(np.max(residuals)),
                "torque_ratio_p95": torque_p95,
                "torque_ratio_peak": torque_peak,
                "torque_components": {
                    "gravity": component_summary(gravity_ratio_by_joint),
                    "inertia": component_summary(inertia_ratio_by_joint),
                    "velocity_and_passive": component_summary(velocity_passive_ratio_by_joint),
                    "contact": component_summary(contact_ratio_by_joint),
                },
                "per_joint_torque_ratios": per_joint,
                "optimizer_failure_count": optimizer_failures,
                "passed": feasible_fraction >= criteria.minimum_feasible_fraction
                and torque_p95 <= criteria.maximum_torque_p95_ratio
                and torque_peak <= criteria.maximum_torque_peak_ratio,
            }
        )
    result = {
        "schema_version": 1,
        "reference_sha256": sha256_file(reference_path),
        "model_sha256": sha256_file(model_path),
        "criteria": asdict(criteria),
        "normalization": {
            "force": "total robot weight",
            "moment": "total robot weight times robot height",
            "total_weight_n": total_weight,
            "robot_height_m": robot_height,
        },
        "contact_assumptions": {
            "friction": "linear pyramid",
            "normal_force": "nonnegative",
            "center_of_pressure": "inside rectangular foot support polygon",
            "torsional_limit": "0.02 m times normal force",
            "contact_labels": "frozen reference foot_contact",
        },
        "speeds": speed_results,
        "passed": all(item["passed"] for item in speed_results),
    }
    write_atomic_json(output, result)
    return result
