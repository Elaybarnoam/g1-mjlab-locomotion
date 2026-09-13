from __future__ import annotations

from g1_mjlab.walking_v2_promotion import acquisition_decision, summarize_checkpoint


def _evaluation(
    speed: float, *, command_rms: float = 0.1, passed: bool = True
) -> dict[str, object]:
    return {
        "requested_speed_m_s": speed,
        "development_functional_passed": passed,
        "terminated": False,
        "survived_seconds": 5.0,
        "settled_forward_command_rms_m_s": command_rms,
        "reference_joint_position_rms_rad": 0.1,
        "reference_joint_velocity_rms_rad_s": 0.5,
        "torque_ratio_p95": 0.4,
        "torque_ratio_peak": 0.8,
        "contact_transition_rate_s": 2.0,
        "touchdown_count": 8,
    }


def _checkpoint(update: int, command_rms: float = 0.1) -> dict[str, object]:
    evaluations = [_evaluation(speed, command_rms=command_rms) for speed in (0.0, 0.4, 0.6, 0.8)]
    return summarize_checkpoint(update, f"{update:064x}", evaluations)


def test_selects_eligible_checkpoint() -> None:
    result = acquisition_decision([_checkpoint(100, 0.2), _checkpoint(200, 0.1)])

    assert result["status"] == "promoted"
    assert result["selected_update"] == 200


def test_authorizes_continuation_only_for_positive_safe_trend() -> None:
    rows = [_checkpoint(100, 0.6), _checkpoint(200, 0.5), _checkpoint(500, 0.4)]

    result = acquisition_decision(rows)

    assert result["status"] == "not_promoted"
    assert result["positive_trend"] is True
    assert result["continuation_authorized"] is True
    assert result["extension_authorized"] is False


def test_rejects_checkpoint_order_drift() -> None:
    try:
        acquisition_decision([_checkpoint(200), _checkpoint(100)])
    except ValueError as error:
        assert "increasing" in str(error)
    else:
        raise AssertionError("reordered checkpoints must fail")
