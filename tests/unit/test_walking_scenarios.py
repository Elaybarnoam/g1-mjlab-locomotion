from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_mjlab.gait_evaluation.scenarios import load_scenario_set


def scenario_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "name": "measurement-audit-v2",
        "scenarios": [
            {
                "name": "stand-walk-stop",
                "seed": 10042,
                "initialization": "standing",
                "segments": [
                    {"duration_s": 2.0, "command": [0.0, 0.0, 0.0]},
                    {"duration_s": 6.0, "command": [0.6, 0.0, 0.0]},
                    {"duration_s": 2.0, "command": [0.0, 0.0, 0.0]},
                ],
            }
        ],
    }


def test_scenario_set_is_strict_hashed_and_control_aligned(tmp_path: Path) -> None:
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(scenario_payload()), encoding="utf-8")
    result = load_scenario_set(path, control_dt=0.02)
    assert result.name == "measurement-audit-v2"
    assert result.scenarios[0].horizon_s == 10.0
    assert len(result.sha256) == 64


def test_scenario_set_rejects_unknown_fields_and_misalignment(tmp_path: Path) -> None:
    payload = scenario_payload()
    payload["unknown"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        load_scenario_set(path, control_dt=0.02)

    payload = scenario_payload()
    scenarios = payload["scenarios"]
    assert isinstance(scenarios, list)
    scenarios[0]["segments"][0]["duration_s"] = 2.001  # type: ignore[index]
    path = tmp_path / "unaligned.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="align"):
        load_scenario_set(path, control_dt=0.02)


def test_scenario_set_rejects_lateral_commands_for_v1(tmp_path: Path) -> None:
    payload = scenario_payload()
    scenarios = payload["scenarios"]
    assert isinstance(scenarios, list)
    scenarios[0]["segments"][1]["command"] = [0.6, 0.1, 0.0]  # type: ignore[index]
    path = tmp_path / "lateral.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="forward-only"):
        load_scenario_set(path, control_dt=0.02)
