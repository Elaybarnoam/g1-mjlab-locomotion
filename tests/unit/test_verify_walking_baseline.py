from __future__ import annotations

import hashlib
import json
from pathlib import Path

from g1_mjlab.walking_baseline import (
    EXPECTED_C100_CHECKPOINT_SHA256,
    verify_baseline,
    verify_fresh_evaluation,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_hashed(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def test_verifier_inventories_plan04_c100_and_required_gaps(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    segment = repository / ".runtime/plan04-p07-arm-c-001/segments/segment-000"
    evaluation = repository / ".runtime/plan04-p07-arm-c-001/evaluations/segment-000"
    checkpoint = segment / "checkpoints/model_99.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trusted checkpoint fixture")
    model_sha256 = _write_hashed(segment / "model.mjb", b"model")
    reference_sha256 = _write_hashed(
        repository / "configs/walking-v1/reference-map-v2.json", b"reference"
    )
    _write_json(
        segment / "policy-bundle.json",
        {
            "checkpoint": "model_99.pt",
            "checkpoint_sha256": EXPECTED_C100_CHECKPOINT_SHA256,
            "model_sha256": model_sha256,
        },
    )
    _write_json(
        segment / "manifest.json",
        {
            "status": "completed",
            "run_id": "campaign-stage19-segment-000",
            "source_commit": "67075b40ea7cd158e1891d718c51e046dad17daa",
        },
    )
    _write_json(segment / "runtime.json", {"packages": {"torch": "2.13.0"}})
    _write_json(
        segment / "source.json",
        {
            "files": [
                {
                    "path": "configs/walking-v1/reference-map-v2.json",
                    "sha256": reference_sha256,
                }
            ]
        },
    )
    trials = []
    for trial_id in range(16):
        trials.append(
            {
                "trial_id": trial_id,
                "scenario_name": f"scenario-{trial_id:02d}",
                "functional_passed": True,
                "style_passed": False,
                "command_rms_m_s": 0.08,
                "raw_transition_rate_s": 10.0,
                "physical_slip_rms_m_s": float(trial_id),
                "step_length_median_m": float(trial_id) / 10,
                "trace": f"trace-{trial_id:03d}.npz",
                "physics_trace": f"physics-trace-{trial_id:03d}.npz",
            }
        )
        _write_json(
            evaluation / f"trace-{trial_id:03d}.metadata.json",
            {
                "schema_version": 2,
                "checkpoint_sha256": EXPECTED_C100_CHECKPOINT_SHA256,
                "controller_sha256": "controller-identity",
                "reference_sha256": "reference-identity",
            },
        )
    _write_json(
        evaluation / "summary.json",
        {
            "schema_version": 2,
            "checkpoint": "model_99.pt",
            "checkpoint_sha256": EXPECTED_C100_CHECKPOINT_SHA256,
            "trials": trials,
        },
    )

    baseline, gaps = verify_baseline(
        repository=repository,
        segment=segment,
        evaluation=evaluation,
        source={"commit": "current-head", "branch": "shared", "changes": []},
        host={"python": "3.12.10"},
        checkpoint_file_sha256=EXPECTED_C100_CHECKPOINT_SHA256,
    )

    assert baseline["candidate"] == {
        "arm": "C",
        "completed_updates": 100,
        "checkpoint": "model_99.pt",
        "checkpoint_sha256": EXPECTED_C100_CHECKPOINT_SHA256,
    }
    assert baseline["artifacts"]["model"]["sha256"] == model_sha256
    assert baseline["artifacts"]["reference_map"]["sha256"] == reference_sha256
    assert baseline["artifacts"]["controller"]["sha256"] == "controller-identity"
    assert baseline["evaluation"]["counts"] == {
        "trials": 16,
        "functional_passes": 16,
        "style_passes": 0,
        "combined_passes": 0,
    }
    assert baseline["evaluation"]["retained_trajectory_trials"] == {
        "nominal": 0,
        "worst_slip": 15,
        "worst_step": 0,
    }
    assert baseline["interpretation"]["current"] == "unqualified"
    assert gaps["status"] == "gaps_present"
    gap_by_id = {item["id"]: item for item in gaps["gaps"]}
    assert gap_by_id["fresh_gpu_reproduction"]["status"] == "not_run"
    assert gap_by_id["natural_walking_candidate"]["status"] == "missing"


def test_fresh_reproduction_requires_same_checkpoint_and_preserves_failed_style(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.json"
    _write_json(
        summary,
        {
            "schema_version": 2,
            "checkpoint_sha256": EXPECTED_C100_CHECKPOINT_SHA256,
            "completed": 16,
            "trials": [
                {
                    "functional_passed": True,
                    "style_passed": False,
                    "command_rms_m_s": 0.08,
                    "physical_slip_rms_m_s": 0.25,
                    "step_length_median_m": 0.15,
                    "raw_transition_rate_s": 10.0,
                }
                for _ in range(16)
            ],
        },
    )

    result = verify_fresh_evaluation(summary)

    assert result["reproduced_decision"] is True
    assert result["counts"] == {
        "trials": 16,
        "functional_passes": 16,
        "style_passes": 0,
        "combined_passes": 0,
    }
    assert result["median_physical_slip_rms_m_s"] == 0.25
