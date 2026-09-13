"""Fail-closed promotion decisions for walking-v2 acquisition checkpoints."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def summarize_checkpoint(
    update: int, checkpoint_sha256: str, evaluations: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate the four frozen deterministic speeds for one checkpoint."""
    rows = list(evaluations)
    speeds = [float(row["requested_speed_m_s"]) for row in rows]
    if speeds != [0.0, 0.4, 0.6, 0.8]:
        raise ValueError("acquisition evaluations must be ordered 0.0, 0.4, 0.6, 0.8 m/s")
    if len(checkpoint_sha256) != 64:
        raise ValueError("checkpoint identity must be SHA-256")
    moving = rows[1:]
    return {
        "update": update,
        "checkpoint_sha256": checkpoint_sha256,
        "all_finite": all(
            all(not isinstance(value, float) or math.isfinite(value) for value in row.values())
            for row in rows
        ),
        "functional_pass_count": sum(bool(row["development_functional_passed"]) for row in rows),
        "terminated_count": sum(bool(row["terminated"]) for row in rows),
        "minimum_survival_s": min(float(row["survived_seconds"]) for row in rows),
        "moving_command_rms_mean_m_s": sum(
            float(row["settled_forward_command_rms_m_s"]) for row in moving
        )
        / len(moving),
        "reference_position_rms_mean_rad": sum(
            float(row["reference_joint_position_rms_rad"]) for row in moving
        )
        / len(moving),
        "reference_velocity_rms_mean_rad_s": sum(
            float(row["reference_joint_velocity_rms_rad_s"]) for row in moving
        )
        / len(moving),
        "torque_ratio_p95_max": max(float(row["torque_ratio_p95"]) for row in rows),
        "torque_ratio_peak_max": max(float(row["torque_ratio_peak"]) for row in rows),
        "contact_transition_rate_max_s": max(
            float(row["contact_transition_rate_s"]) for row in moving
        ),
        "moving_touchdown_count_min": min(int(row["touchdown_count"]) for row in moving),
    }


def acquisition_decision(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    """Select a passing checkpoint or declare whether one bounded extension is justified."""
    if not checkpoints or any(
        int(right["update"]) <= int(left["update"])
        for left, right in zip(checkpoints, checkpoints[1:], strict=False)
    ):
        raise ValueError("checkpoints must be in strictly increasing update order")
    eligible = [
        row
        for row in checkpoints
        if row["all_finite"]
        and row["functional_pass_count"] == 4
        and row["terminated_count"] == 0
        and row["minimum_survival_s"] >= 5.0
        and row["moving_command_rms_mean_m_s"] <= 0.25
        and row["torque_ratio_p95_max"] <= 0.80
        and row["torque_ratio_peak_max"] <= 1.00
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["moving_command_rms_mean_m_s"],
            row["reference_position_rms_mean_rad"],
            row["update"],
        ),
        default=None,
    )
    first, last = checkpoints[0], checkpoints[-1]
    safety_valid = bool(
        last["all_finite"]
        and last["torque_ratio_p95_max"] <= 0.80
        and last["torque_ratio_peak_max"] <= 1.00
    )
    positive_trend = bool(
        len(checkpoints) >= 3
        and safety_valid
        and last["minimum_survival_s"] >= first["minimum_survival_s"]
        and last["terminated_count"] <= first["terminated_count"]
        and last["moving_command_rms_mean_m_s"] < first["moving_command_rms_mean_m_s"]
        and last["functional_pass_count"] >= first["functional_pass_count"]
    )
    return {
        "schema_version": 1,
        "status": "promoted" if selected is not None else "not_promoted",
        "selected_checkpoint_sha256": selected["checkpoint_sha256"] if selected else None,
        "selected_update": selected["update"] if selected else None,
        "positive_trend": positive_trend,
        "safety_valid": safety_valid,
        "continuation_authorized": bool(
            selected is None and positive_trend and int(last["update"]) < 4000
        ),
        "extension_authorized": bool(
            selected is None and positive_trend and int(last["update"]) >= 4000
        ),
        "checkpoints": checkpoints,
        "qualification_claim": False,
    }
