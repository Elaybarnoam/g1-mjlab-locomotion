import json
from pathlib import Path

from g1_mjlab.selection import score, select_checkpoint


def summary(name: str, passed: int, survival: int, drift: float) -> dict:
    return {
        "schema_version": 2,
        "phase": "development",
        "checkpoint": name,
        "passed": passed,
        "survival_passed": survival,
        "trials": [{"max_drift_m": drift, "max_torso_tilt_deg": 5.0}],
    }


def test_strict_success_precedes_survival_and_drift() -> None:
    assert score(summary("model_10.pt", 1, 1, 0.1)) > score(summary("model_20.pt", 0, 20, 0.01))


def test_selection_updates_canonical_evaluation_and_index(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "evaluation" / "a").mkdir(parents=True)
    (run / "evaluation" / "b").mkdir()
    for folder, value in (
        ("a", summary("model_9.pt", 0, 20, 0.3)),
        ("b", summary("model_10.pt", 1, 1, 0.1)),
    ):
        (run / "evaluation" / folder / "summary.json").write_text(json.dumps(value))
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "index.json").write_text("{}")
    selected = select_checkpoint(run)
    assert selected["selected_checkpoint"] == "model_10.pt"
    assert (
        json.loads((run / "evaluation" / "summary.json").read_text())["checkpoint"] == "model_10.pt"
    )


def test_selection_dispatches_to_walking_gates(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for folder, passed in (("a", 0), ("b", 1)):
        path = run / "evaluation" / folder
        path.mkdir(parents=True)
        payload = {
            "schema_version": 1,
            "task_id": "G1-Walking-Flat-v1",
            "phase": "development",
            "checkpoint": f"model_{10 + passed}.pt",
            "passed_both": passed,
            "trials": [
                {
                    "functional_passed": True,
                    "style_passed": bool(passed),
                    "command_rms_m_s": 0.1,
                    "alternation_ratio": 1.0,
                    "tiny_step_fraction": 0.0,
                    "stance_slip_rms_m_s": 0.01,
                }
            ],
        }
        (path / "summary.json").write_text(json.dumps(payload))
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "index.json").write_text("{}")
    selected = select_checkpoint(run)
    assert selected["selected_checkpoint"] == "model_11.pt"
    assert "gait" in selected["rule"]
