from __future__ import annotations

import pytest

from g1_mjlab.walking_objectives import (
    WalkingProbe,
    classify_walking_probe,
    weighted_reward_rate,
)


def _probe(**overrides: float | bool) -> WalkingProbe:
    values: dict[str, float | bool] = {
        "command_speed_m_s": 1.16,
        "achieved_speed_m_s": 1.12,
        "joint_pose_rms_rad": 0.08,
        "joint_velocity_rms_rad_s": 0.4,
        "contact_agreement": 0.9,
        "stance_slip_m_s": 0.03,
        "pelvis_height_m": 0.74,
        "torso_tilt_rad": 0.05,
        "step_length_m": 0.50,
        "cadence_steps_min": 112.0,
        "non_foot_contact": False,
        "finite": True,
    }
    values.update(overrides)
    return WalkingProbe(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "probe", "reason"),
    [
        ("ideal", _probe(), None),
        ("shuffling", _probe(step_length_m=0.08, cadence_steps_min=190.0), "shuffling"),
        ("sliding", _probe(stance_slip_m_s=0.35), "stance_slip"),
        ("crouching", _probe(pelvis_height_m=0.54), "crouching"),
        ("falling", _probe(torso_tilt_rad=1.0), "fallen"),
        ("standing", _probe(achieved_speed_m_s=0.01, step_length_m=0.0), "command_tracking"),
        ("knee_support", _probe(non_foot_contact=True), "non_foot_contact"),
    ],
)
def test_behavior_probe_distinguishes_failure_modes(
    name: str, probe: WalkingProbe, reason: str | None
) -> None:
    del name
    result = classify_walking_probe(probe)

    assert result.passed is (reason is None)
    if reason is not None:
        assert reason in result.failures


def test_weighted_reward_sum_matches_reported_contributions() -> None:
    raw = {"velocity": 0.8, "reference_pose": 0.5, "slip": 0.04}
    weights = {"velocity": 2.0, "reference_pose": 1.5, "slip": -0.5}

    result = weighted_reward_rate(raw, weights)

    assert result.contributions == {
        "velocity": pytest.approx(1.6),
        "reference_pose": pytest.approx(0.75),
        "slip": pytest.approx(-0.02),
    }
    assert result.total_rate == pytest.approx(2.33)
    assert result.integrated_step_reward(0.02) == pytest.approx(0.0466)
