from __future__ import annotations

from g1_mjlab.motion.reference_kinematics import KinematicAdmissionCriteria


def test_kinematic_admission_criteria_are_frozen() -> None:
    criteria = KinematicAdmissionCriteria()

    assert criteria.minimum_joint_margin_rad == 0.019
    assert criteria.maximum_foot_ik_error_m == 0.010
    assert criteria.maximum_ground_penetration_m == 0.005
    assert criteria.maximum_self_penetration_m == 0.002
    assert criteria.maximum_stride_error_m == 0.020
