from __future__ import annotations

import numpy as np

from g1_mjlab.motion.reference_dynamics import (
    actuator_force_limits_by_joint,
    solve_contact_wrenches,
)


def test_constrained_wrench_solver_balances_weight_inside_friction_and_support() -> None:
    required = np.array([0.0, 0.0, 300.0, 0.0, 0.0, 0.0])
    jacobians = np.zeros((2, 6, 6))
    jacobians[:, :, :] = np.eye(6)

    result = solve_contact_wrenches(
        required,
        jacobians,
        np.array([True, True]),
        friction_coefficient=0.7,
        foot_half_length_m=0.10,
        foot_half_width_m=0.04,
        total_weight_n=300.0,
        robot_height_m=1.3,
    )

    assert result.success
    assert result.normalized_base_residual < 1e-6
    assert np.all(result.wrenches[:, 2] >= 0)
    assert np.all(np.abs(result.wrenches[:, 0]) <= 0.7 * result.wrenches[:, 2] + 1e-8)
    assert np.all(np.abs(result.wrenches[:, 3]) <= 0.04 * result.wrenches[:, 2] + 1e-8)
    assert np.all(np.abs(result.wrenches[:, 4]) <= 0.10 * result.wrenches[:, 2] + 1e-8)


def test_inactive_contact_is_exactly_zero() -> None:
    required = np.array([20.0, 0.0, 300.0, 0.0, 0.0, 0.0])
    jacobians = np.stack((np.eye(6), np.eye(6)))

    result = solve_contact_wrenches(
        required,
        jacobians,
        np.array([True, False]),
        friction_coefficient=0.7,
        foot_half_length_m=0.10,
        foot_half_width_m=0.04,
        total_weight_n=300.0,
        robot_height_m=1.3,
    )

    assert result.success
    np.testing.assert_allclose(result.wrenches[1], 0.0, atol=1e-10)


def test_actuator_constraints_distribute_contact_without_exceeding_limit() -> None:
    required = np.array([0.0, 0.0, 300.0, 0.0, 0.0, 0.0])
    base_jacobians = np.stack((np.eye(6), np.eye(6)))
    actuator_jacobians = np.zeros((2, 6, 1))
    actuator_jacobians[:, 2, 0] = 0.5

    result = solve_contact_wrenches(
        required,
        base_jacobians,
        np.array([True, True]),
        friction_coefficient=0.7,
        foot_half_length_m=0.10,
        foot_half_width_m=0.04,
        total_weight_n=300.0,
        robot_height_m=1.3,
        required_actuator_torque=np.array([150.0]),
        actuator_contact_jacobians=actuator_jacobians,
        actuator_force_limits=np.array([1.0]),
    )

    assert result.success
    contact_torque = sum(
        actuator_jacobians[foot].T @ result.wrenches[foot] for foot in range(2)
    )
    assert abs(150.0 - contact_torque[0]) <= 1.0 + 1e-7


def test_actuator_limits_are_reordered_by_transmission_joint() -> None:
    class Model:
        nu = 3
        actuator_trnid = np.array([[7, -1], [3, -1], [5, -1]])
        actuator_forcerange = np.array([[-70.0, 70.0], [-30.0, 30.0], [-50.0, 50.0]])

    limits = actuator_force_limits_by_joint(Model(), [3, 5, 7])

    np.testing.assert_array_equal(limits, [30.0, 50.0, 70.0])
