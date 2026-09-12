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


def test_explicit_scenario_state_is_exact_and_hashed(tmp_path: Path) -> None:
    payload = scenario_payload()
    payload["schema_version"] = 3
    scenario = payload["scenarios"][0]  # type: ignore[index]
    scenario["category"] = "nominal"
    scenario["initial_phase"] = 0.25
    scenario["initial_qpos"] = [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, *([0.0] * 29)]
    scenario["initial_qvel"] = [0.0] * 35
    path = tmp_path / "explicit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = load_scenario_set(path, control_dt=0.02)

    assert result.schema_version == 3
    assert result.scenarios[0].initial_phase == 0.25
    assert len(result.scenarios[0].initial_qpos or ()) == 36
    assert len(result.scenarios[0].initial_qvel or ()) == 35


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("initial_phase", 1.0, "phase"),
        ("initial_qpos", [0.0] * 35, "qpos"),
        ("initial_qvel", [0.0] * 34, "qvel"),
    ],
)
def test_explicit_scenario_rejects_invalid_state(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = scenario_payload()
    payload["schema_version"] = 3
    scenario = payload["scenarios"][0]  # type: ignore[index]
    scenario.update(
        {
            "category": "nominal",
            "initial_phase": 0.0,
            "initial_qpos": [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, *([0.0] * 29)],
            "initial_qvel": [0.0] * 35,
        }
    )
    scenario[field] = value
    path = tmp_path / "invalid-explicit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_scenario_set(path, control_dt=0.02)
