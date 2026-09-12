from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_mjlab.walking_final import (
    assess_final_prerequisite,
    build_final_scenario_payload,
    qualify_final_walking,
)


def _state() -> tuple[list[float], list[float]]:
    return [0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0, *([0.0] * 29)], [0.0] * 35


def test_final_scenarios_are_deterministic_balanced_and_exactly_sixty_seconds() -> None:
    qpos, qvel = _state()

    first = build_final_scenario_payload(qpos, qvel)
    second = build_final_scenario_payload(qpos, qvel)

    assert first == second
    assert len(first["scenarios"]) == 100
    assert [item["seed"] for item in first["scenarios"]] == list(range(20000, 20100))
    for category in ("slow", "medium", "fast", "start_stop"):
        selected = [item for item in first["scenarios"] if item["category"] == category]
        assert len(selected) == 25
        assert all(
            sum(segment["duration_s"] for segment in item["segments"]) == 60 for item in selected
        )


def test_final_prerequisite_rejects_unqualified_or_narrow_policy(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "status": "unqualified_development",
                "qualified_command_domain_m_s": [],
                "checkpoint_sha256": "checkpoint",
            }
        ),
        encoding="utf-8",
    )

    result = assess_final_prerequisite(bundle, None)

    assert result["final_execution_authorized"] is False
    assert "development_qualified_policy" in result["missing"]
    assert "qualified_speed_domain_0_to_0_8_m_s" in result["missing"]
    assert "owner_visual_acceptance" in result["missing"]


def test_qualification_requires_95_overall_and_23_per_stratum(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "acceptance": {"minimum_overall_passes": 95, "minimum_stratum_passes": 23},
                "hashes": {"checkpoint": "abc", "onnx": "def"},
            }
        ),
        encoding="utf-8",
    )
    trials = [
        {
            "trial_id": index,
            "category": ("slow", "medium", "fast", "start_stop")[index // 25],
            "function_passed": index != 0,
            "style_passed": index != 0,
        }
        for index in range(100)
    ]
    mjlab = tmp_path / "mjlab.json"
    native = tmp_path / "native.json"
    for path in (mjlab, native):
        path.write_text(
            json.dumps({"trials": trials, "checkpoint_sha256": "abc"}), encoding="utf-8"
        )

    result = qualify_final_walking(freeze, mjlab, native, tmp_path / "qualification.json")

    assert result["qualified"] is True
    assert result["backends"]["mjlab"]["overall_passes"] == 99

    trials[25]["style_passed"] = False
    trials[26]["style_passed"] = False
    trials[27]["style_passed"] = False
    for path in (mjlab, native):
        path.write_text(
            json.dumps({"trials": trials, "checkpoint_sha256": "abc"}), encoding="utf-8"
        )
    with pytest.raises(FileExistsError):
        qualify_final_walking(freeze, mjlab, native, tmp_path / "qualification.json")
    result = qualify_final_walking(freeze, mjlab, native, tmp_path / "failed.json")
    assert result["qualified"] is False
