import json
from dataclasses import replace
from pathlib import Path

import pytest

from g1_mjlab.artifacts import snapshot_source
from g1_mjlab.config import load_config
from g1_mjlab.training import validate_resume


def test_source_snapshot_includes_untracked_implementation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "src" / "g1_mjlab"
    source.mkdir(parents=True)
    (source / "untracked.py").write_text("value = 1\n", encoding="utf-8")
    snapshot = snapshot_source(project, tmp_path / "source.zip")
    assert snapshot["files"][0]["path"] == "src/g1_mjlab/untracked.py"
    assert len(snapshot["sha256"]) == 64


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
