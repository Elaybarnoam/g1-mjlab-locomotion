from __future__ import annotations

import json
from pathlib import Path

from g1_mjlab.reporting.walking import _script_json, render_walking_research_report


def test_walking_report_script_json_escapes_imported_html() -> None:
    encoded = _script_json({"name": "</script><script>alert(1)</script>"})
    assert "</script>" not in encoded
    assert "\\u003c/script>" in encoded


def test_walking_report_is_offline_and_labels_failed_hypothesis(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    objects = {
        "manifest.json": {"status": "completed"},
        "config.json": {"task_id": "G1-Walking-Flat-v1"},
        "algorithm.json": {},
        "mdp.json": {"rewards": {"upright": {"weight": 1, "func": "safe"}}},
        "optimization.json": {},
        "learning-summary.json": {"losses": [{"metric": "Loss/value", "update": 0, "value": 1.0}]},
    }
    for name, value in objects.items():
        (run / name).write_text(json.dumps(value), encoding="utf-8")
    experiment = tmp_path / "experiment.json"
    experiment.write_text(
        json.dumps(
            {
                "best_observed": {
                    "arm": "c",
                    "completed_updates": 100,
                    "checkpoint_sha256": "abc",
                    "selection": {"function_pass_count": 16, "function_style_pass_count": 0},
                },
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "validation": {"actor_dimension": 102},
                "networks": {"actor": {"parameter_count": 1}},
            }
        ),
        encoding="utf-8",
    )
    native = tmp_path / "native.json"
    native.write_text(
        json.dumps({"passed_finite_rollouts": 1, "planned_rollouts": 1}), encoding="utf-8"
    )
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({"samples": 100, "passed": True}), encoding="utf-8")
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "time_s": 0.02,
                "requested_speed_m_s": 0.0,
                "applied_speed_m_s": 0.0,
                "pelvis_height_m": 0.79,
                "action": [0.0, 0.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    index = render_walking_research_report(
        run, experiment, policy, native, trace, parity, tmp_path / "report"
    )

    content = index.read_text(encoding="utf-8")
    assert "failed hypothesis" in content
    assert "Unavailable means not recorded" in content
    assert "https://" not in content
    assert (index.parent / "report-data.json").is_file()
