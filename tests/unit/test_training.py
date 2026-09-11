import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from g1_mjlab.artifacts import snapshot_source
from g1_mjlab.config import WalkingTrainingProfile, load_config
from g1_mjlab.runtime import summarize_training_metrics
from g1_mjlab.training import (
    validate_actor_initialization,
    validate_fine_tune_initialization,
    validate_resume,
)


def test_source_snapshot_includes_untracked_implementation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "src" / "g1_mjlab"
    source.mkdir(parents=True)
    (source / "untracked.py").write_text("value = 1\n", encoding="utf-8")
    snapshot = snapshot_source(project, tmp_path / "source.zip")
    assert snapshot["files"][0]["path"] == "src/g1_mjlab/untracked.py"
    assert len(snapshot["sha256"]) == 64


def test_source_snapshot_selects_only_the_active_task_config(tmp_path: Path) -> None:
    project = tmp_path / "project"
    for task in ("standing-v1", "walking-v1"):
        directory = project / "configs" / task
        directory.mkdir(parents=True)
        (directory / "train.json").write_text(task, encoding="utf-8")

    destination = tmp_path / "source.zip"
    snapshot_source(project, destination, config_directory="walking-v1")

    with zipfile.ZipFile(destination) as archive:
        names = set(archive.namelist())
    assert "configs/walking-v1/train.json" in names
    assert "configs/standing-v1/train.json" not in names


def test_resume_rejects_changed_action_semantics(tmp_path: Path) -> None:
    config = load_config(Path(__file__).resolve().parents[2] / "configs/standing-v1/smoke.json")
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints" / "model_1.pt"
    checkpoint.touch()
    (run / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    validate_resume(replace(config, max_iterations=20), checkpoint, None, None)
    with pytest.raises(ValueError, match="action_clip"):
        validate_resume(replace(config, action_clip=None), checkpoint, None, None)


def test_resume_compares_json_normalized_walking_profile(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/walking-v1/bootstrap-smoke.json")
    profile = WalkingTrainingProfile(
        schema_version=1,
        name="bootstrap",
        standing_fraction=0.1,
        reference_initialization=False,
        randomize_phase=True,
        objective="locomotion_bootstrap",
        forward_speed_range_m_s=(0.4, 0.8),
    )
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints" / "model_400.pt"
    checkpoint.touch()
    (run / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    (run / "walking-profile.json").write_text(
        json.dumps(profile.to_dict()), encoding="utf-8"
    )

    validate_resume(config, checkpoint, None, None, profile)


def test_actor_initialization_requires_same_task_and_is_not_resume(tmp_path: Path) -> None:
    config = load_config(Path(__file__).resolve().parents[2] / "configs/standing-v1/smoke.json")
    run = tmp_path / "source"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints" / "model_1.pt"
    checkpoint.touch()
    (run / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")

    metadata = validate_actor_initialization(config, checkpoint)

    assert metadata["mode"] == "actor-and-actor-normalizer-only"
    with pytest.raises(ValueError, match="same task"):
        validate_actor_initialization(replace(config, task_id="G1-Walking-Flat-v1"), checkpoint)


def test_fine_tune_declares_full_learner_state_with_fresh_iteration(tmp_path: Path) -> None:
    config = load_config(Path(__file__).resolve().parents[2] / "configs/standing-v1/smoke.json")
    run = tmp_path / "source"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints" / "model_7.pt"
    checkpoint.touch()
    (run / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")

    metadata = validate_fine_tune_initialization(config, checkpoint)

    assert metadata["mode"] == "full-learner-state-fine-tune"
    assert metadata["source_iteration"] == 7
    assert metadata["fresh_components"] == ["iteration", "environment_state"]


def test_training_metric_summary_requires_finite_losses_for_every_update() -> None:
    records = [
        {"update": update, "metric": metric, "value": value}
        for update in (0, 1)
        for metric, value in (
            ("Loss/value", 0.5),
            ("Loss/surrogate", -0.1),
            ("Loss/entropy", 4.0),
        )
    ]
    assert summarize_training_metrics(records, expected_updates=2)["all_losses_finite"]
    records[-1]["value"] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        summarize_training_metrics(records, expected_updates=2)
