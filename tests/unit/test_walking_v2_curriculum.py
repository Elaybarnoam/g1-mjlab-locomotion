from __future__ import annotations

import json

import pytest

from g1_mjlab.artifacts import sha256_file
from g1_mjlab.walking_v2_curriculum import (
    assess_fixed_speed_stage,
    assess_transition_stage,
    load_curriculum_method,
)


def _row(speed: float) -> dict[str, object]:
    return {
        "requested_speed_m_s": speed,
        "development_functional_passed": True,
        "terminated": False,
        "survived_seconds": 10.0,
        "settled_forward_command_rms_m_s": 0.10 if speed else 0.01,
        "minimum_pelvis_height_m": 0.70,
        "maximum_torso_tilt_rad": 0.20,
        "torque_ratio_p95": 0.30,
        "torque_ratio_peak": 0.80,
        "forward_progress_m": speed * 5.0,
        "cadence_steps_s": 1.8 if speed else 0.0,
        "alternation_fraction": 0.9 if speed else 1.0,
        "contact_transition_rate_s": 3.6 if speed else 0.0,
        "touchdown_count": 12 if speed else 0,
    }


def test_fixed_speed_stage_requires_exact_speed_frontier_and_gait() -> None:
    rows = [_row(speed) for speed in (0.0, 0.4, 0.6)]

    result = assess_fixed_speed_stage("add-060", rows, horizon_s=10.0)

    assert result["passed"] is True
    assert result["required_speeds_m_s"] == [0.0, 0.4, 0.6]
    rows[1]["alternation_fraction"] = 0.79
    assert assess_fixed_speed_stage("add-060", rows, horizon_s=10.0)["passed"] is False
    rows[1]["alternation_fraction"] = 0.9
    rows[1]["forward_progress_m"] = 0.99
    assert assess_fixed_speed_stage("add-060", rows, horizon_s=10.0)["passed"] is False


def test_fixed_speed_stage_rejects_missing_or_extra_speeds() -> None:
    with pytest.raises(ValueError, match="speed frontier"):
        assess_fixed_speed_stage("stand-walk-040", [_row(0.0)], horizon_s=10.0)


def test_transition_stage_requires_every_functional_and_style_gate() -> None:
    summary = {
        "trials": [
            {"functional_passed": True, "style_passed": True, "insufficient_evidence": False}
            for _ in range(4)
        ]
    }

    assert assess_transition_stage("transitions", summary)["passed"] is True
    summary["trials"][2]["style_passed"] = False
    assert assess_transition_stage("transitions", summary)["passed"] is False


def test_frozen_curriculum_method_binds_every_stage() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    method = load_curriculum_method(root / "configs/walking-v2/curriculum-method-v2.json")

    assert method.seeds == (42, 43, 44)
    assert [stage.stage for stage in method.stages] == [
        "stand",
        "stand-walk-040",
        "add-060",
        "add-080",
        "transitions",
        "robustness",
    ]
    assert sum(stage.updates for stage in method.stages) == 2500


def test_failed_curriculum_result_is_fail_closed_and_binds_method() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    result = json.loads(
        (root / "configs/walking-v2/curriculum-result-v2.json").read_text(encoding="utf-8")
    )

    assert result["status"] == "failed"
    assert result["selected_checkpoint"] is None
    assert result["qualification_claim"] is False
    assert result["release_authorized"] is False
    assert result["method_sha256"] == sha256_file(
        root / "configs/walking-v2/curriculum-method-v2.json"
    )
