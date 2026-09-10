import json
from pathlib import Path

import pytest

import g1_mjlab.deployment as deployment
from g1_mjlab.deployment import qualify_final_bundle


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_final_qualification_fails_closed_below_gate(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoint_hash = "checkpoint"
    onnx_hash = "onnx"
    write_json(
        run / "policy-bundle.json",
        {
            "checkpoint_sha256": checkpoint_hash,
            "onnx_sha256": onnx_hash,
        },
    )
    write_json(
        run / "evaluation/final-100x60/summary.json",
        {
            "phase": "final",
            "planned": 100,
            "horizon_seconds": 60.0,
            "passed": 94,
            "checkpoint_sha256": checkpoint_hash,
        },
    )
    write_json(
        run / "native-final-100x60/summary.json",
        {
            "phase": "final",
            "planned": 100,
            "horizon_seconds": 60.0,
            "passed": 100,
            "policy_bundle": {"onnx_sha256": onnx_hash},
        },
    )
    with pytest.raises(ValueError, match="mjlab"):
        qualify_final_bundle(run)


def test_final_qualification_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = tmp_path / "run"
    checkpoint_hash = "checkpoint"
    onnx_hash = "onnx"
    write_json(
        run / "policy-bundle.json",
        {"checkpoint_sha256": checkpoint_hash, "onnx_sha256": onnx_hash},
    )
    write_json(
        run / "evaluation/final-100x60/summary.json",
        {
            "phase": "final",
            "planned": 100,
            "horizon_seconds": 60.0,
            "passed": 100,
            "checkpoint_sha256": checkpoint_hash,
        },
    )
    write_json(
        run / "native-final-100x60/summary.json",
        {
            "phase": "final",
            "planned": 100,
            "horizon_seconds": 60.0,
            "passed": 100,
            "policy_bundle": {"onnx_sha256": onnx_hash},
        },
    )

    def fake_snapshot_source(_root: Path, candidate: Path) -> dict[str, str]:
        candidate.write_bytes(b"stable source")
        return {"archive": candidate.name, "sha256": deployment.sha256_file(candidate)}

    monkeypatch.setattr(deployment, "snapshot_source", fake_snapshot_source)
    first = qualify_final_bundle(run)
    second = qualify_final_bundle(run)

    assert second == first
    assert not (run / "source-qualified-candidate.zip").exists()
