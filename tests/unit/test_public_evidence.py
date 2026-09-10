from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qualified_summary_is_internally_consistent() -> None:
    summary = json.loads((ROOT / "evidence/standing-v1/summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "qualified_for_declared_simulation_scope"
    assert summary["results"]["mjlab_mujoco_warp"]["passed"] == 100
    assert summary["results"]["mjlab_mujoco_warp"]["planned"] == 100
    assert summary["results"]["native_mujoco_onnx"]["passed"] == 100
    assert summary["results"]["native_mujoco_onnx"]["planned"] == 100
    assert summary["training"]["transitions_per_seed"] == (
        summary["training"]["environments"]
        * summary["training"]["rollout_steps_per_environment"]
        * summary["training"]["updates_per_seed"]
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert summary["qualification_date"] in readme
    assert f"**{summary['results']['mjlab_mujoco_warp']['passed']}/100**" in readme
    assert f"{summary['training']['transitions_per_seed']:,}" in readme
    assert summary["hashes"]["checkpoint_sha256"] in (
        ROOT / "docs/evaluation-and-evidence.md"
    ).read_text(encoding="utf-8")


def test_curated_media_matches_manifest() -> None:
    media = ROOT / "docs/assets/standing-v1"
    manifest = json.loads((media / "media-manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "deterministic_inference_only"
    assert manifest["learning_enabled"] is False
    for name, expected in manifest["files"].items():
        path = media / name
        assert path.stat().st_size == expected["bytes"]
        assert sha256(path) == expected["sha256"]
