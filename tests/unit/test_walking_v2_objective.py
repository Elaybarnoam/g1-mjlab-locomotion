from __future__ import annotations

import numpy as np
import pytest

from g1_mjlab.walking_v2_objective import (
    WalkingRewardInputs,
    classify_acquisition_termination,
    compute_walking_reward,
)


def _inputs(batch: int = 2) -> WalkingRewardInputs:
    return WalkingRewardInputs(
        joint_error_rad=np.zeros((batch, 29)),
        joint_velocity_error_rad_s=np.zeros((batch, 29)),
        local_foot_error_m=np.zeros((batch, 2, 3)),
        orientation_error_rad=np.zeros(batch),
        contact_agreement=np.ones(batch),
        actual_velocity_body_m_s=np.zeros((batch, 3)),
        command_body_m_s=np.zeros((batch, 3)),
        joint_from_nominal_rad=np.zeros((batch, 29)),
        action=np.zeros((batch, 29)),
        previous_action=np.zeros((batch, 29)),
        normalized_torque=np.zeros((batch, 29)),
        soft_joint_limit_violation_rad=np.zeros((batch, 29)),
        blend=np.asarray([0.0, 1.0])[:batch],
        true_fall_event=np.zeros(batch, dtype=bool),
    )


def test_perfect_reward_matches_frozen_equation() -> None:
    result = compute_walking_reward(_inputs())

    np.testing.assert_allclose(result.imitation, 1.0)
    np.testing.assert_allclose(result.task, 1.5)
    np.testing.assert_allclose(result.stand, 2.0)
    np.testing.assert_allclose(result.rate, [3.5, 3.5])
    np.testing.assert_allclose(result.step, [0.07, 0.07])
    assert set(result.raw_terms) == set(result.weighted_terms)


def test_fall_is_one_event_penalty_and_errors_reduce_reward() -> None:
    inputs = _inputs(1)
    degraded = WalkingRewardInputs(
        **{
            **vars_without_slots(inputs),
            "joint_error_rad": np.full((1, 29), 0.3),
            "true_fall_event": np.ones(1, dtype=bool),
        }
    )

    baseline = compute_walking_reward(inputs)
    result = compute_walking_reward(degraded)

    assert result.imitation[0] < baseline.imitation[0]
    assert result.step[0] == pytest.approx(0.02 * result.rate[0] - 2.0)


def vars_without_slots(inputs: WalkingRewardInputs) -> dict[str, np.ndarray]:
    return {field: getattr(inputs, field) for field in WalkingRewardInputs.__dataclass_fields__}


def test_reward_input_shapes_are_fail_closed() -> None:
    inputs = _inputs(1)
    invalid = WalkingRewardInputs(**{**vars_without_slots(inputs), "action": np.zeros((1, 28))})

    with pytest.raises(ValueError, match="action"):
        compute_walking_reward(invalid)

    with pytest.raises(ValueError, match="policy_dt_s"):
        compute_walking_reward(inputs, policy_dt_s=0.0)


def test_termination_classification_has_safety_precedence() -> None:
    result = classify_acquisition_termination(
        nonfinite=[False, False, False, True],
        forbidden_contact=[False, False, True, True],
        fall=[False, True, True, True],
        reference_deviation=[True, True, True, True],
    )

    assert result.tolist() == [
        "reference_deviation",
        "fall",
        "forbidden_contact",
        "nonfinite",
    ]
