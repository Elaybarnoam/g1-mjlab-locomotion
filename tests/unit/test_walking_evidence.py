from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from g1_mjlab.walking_evidence import (
    CandidateEvidence,
    VisualReviewV2,
    assess_candidate_prerequisites,
)


def _candidate_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "method_id": "reference-conditioned-ppo-v1",
        "task_id": "G1-Walking-Flat-v2",
        "layout_id": "g1-walking-reference-actor-v2",
        "selected_checkpoint_sha256": "1" * 64,
        "source_sha256": "2" * 64,
        "reference_sha256": "3" * 64,
        "controller_sha256": "4" * 64,
        "model_sha256": "5" * 64,
        "contract_sha256": "6" * 64,
        "evaluator_sha256": "7" * 64,
        "criteria_sha256": "8" * 64,
        "scenario_set_sha256": "9" * 64,
        "trial_results": [
            {"path": "evaluation/seed-000.json", "sha256": "a" * 64},
            {"path": "evaluation/seed-001.json", "sha256": "b" * 64},
        ],
        "speed_coverage_m_s": [0.0, 0.8],
        "numeric_status": "development_qualified",
        "visual_status": "pending",
        "final_status": "not_run",
    }


def _review_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "decision": "accepted",
        "checkpoint_sha256": "1" * 64,
        "model_sha256": "5" * 64,
        "reference_sha256": "3" * 64,
        "contract_sha256": "6" * 64,
        "video_sha256s": ["c" * 64, "d" * 64],
        "reviewer": "walking-policy-owner",
        "reviewed_at": "2026-09-13T08:00:00+00:00",
        "comments": "Natural gait and stable stops observed.",
    }


def test_candidate_round_trips_as_immutable_schema_two() -> None:
    payload = _candidate_payload()

    candidate = CandidateEvidence.from_dict(payload)

    assert candidate.to_dict() == payload
    assert isinstance(candidate.trial_results, tuple)
    with pytest.raises(FrozenInstanceError):
        candidate.method_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 1),
        ("schema_version", 2.0),
        ("method_id", ""),
        ("selected_checkpoint_sha256", "not-a-sha256"),
        ("speed_coverage_m_s", [0.8, 0.0]),
        ("speed_coverage_m_s", [False, 0.8]),
        ("numeric_status", "yes"),
        ("visual_status", True),
        ("final_status", "accepted"),
        ("trial_results", []),
    ],
)
def test_candidate_rejects_invalid_values(field: str, value: object) -> None:
    payload = _candidate_payload()
    payload[field] = value

    with pytest.raises(ValueError):
        CandidateEvidence.from_dict(payload)


def test_candidate_constructor_cannot_bypass_schema_validation() -> None:
    candidate = CandidateEvidence.from_dict(_candidate_payload())

    with pytest.raises(ValueError, match="method_id"):
        replace(candidate, method_id="")


def test_candidate_load_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = _candidate_payload()
    payload["unexpected"] = "ignored by permissive decoders"
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fields"):
        CandidateEvidence.load(path)


def test_candidate_verifies_trial_result_references_under_declared_root(tmp_path: Path) -> None:
    result = tmp_path / "evaluation/seed-000.json"
    result.parent.mkdir()
    result.write_bytes(b"immutable result")
    payload = _candidate_payload()
    payload["trial_results"] = [
        {
            "path": "evaluation/seed-000.json",
            "sha256": hashlib.sha256(b"immutable result").hexdigest(),
        }
    ]
    candidate = CandidateEvidence.from_dict(payload)

    candidate.verify_references(tmp_path)
    result.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        candidate.verify_references(tmp_path)


def test_candidate_rejects_duplicate_or_escaping_trial_references(tmp_path: Path) -> None:
    payload = _candidate_payload()
    payload["trial_results"] = [
        {"path": "../outside.json", "sha256": "a" * 64},
        {"path": "../outside.json", "sha256": "a" * 64},
    ]

    with pytest.raises(ValueError, match="path"):
        CandidateEvidence.from_dict(payload)


def test_visual_review_round_trips_and_validates_exact_candidate_binding(tmp_path: Path) -> None:
    path = tmp_path / "review.json"
    path.write_text(json.dumps(_review_payload()), encoding="utf-8")

    review = VisualReviewV2.load(path)

    assert review.to_dict() == _review_payload()
    assert isinstance(review.video_sha256s, tuple)
    review.validate_candidate(CandidateEvidence.from_dict(_candidate_payload()))
    with pytest.raises(FrozenInstanceError):
        review.decision = "rejected"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 1),
        ("schema_version", 2.0),
        ("decision", "approved"),
        ("decision", True),
        ("checkpoint_sha256", "x" * 64),
        ("video_sha256s", []),
        ("video_sha256s", ["c" * 64, "c" * 64]),
        ("reviewer", ""),
        ("reviewed_at", "  "),
        ("reviewed_at", "2026-09-13"),
        ("reviewed_at", "not-a-timestamp"),
        ("comments", 1),
    ],
)
def test_visual_review_rejects_invalid_values(field: str, value: object) -> None:
    payload = _review_payload()
    payload[field] = value

    with pytest.raises(ValueError):
        VisualReviewV2.from_dict(payload)


def test_visual_review_constructor_cannot_bypass_schema_validation() -> None:
    review = VisualReviewV2.from_dict(_review_payload())

    with pytest.raises(ValueError, match="reviewer"):
        replace(review, reviewer="")


@pytest.mark.parametrize(
    "field",
    ["selected_checkpoint_sha256", "model_sha256", "reference_sha256", "contract_sha256"],
)
def test_visual_review_rejects_substituted_candidate_identity(field: str) -> None:
    candidate_payload = _candidate_payload()
    candidate_payload[field] = "e" * 64

    with pytest.raises(ValueError, match="candidate"):
        VisualReviewV2.from_dict(_review_payload()).validate_candidate(
            CandidateEvidence.from_dict(candidate_payload)
        )


def test_explicit_legacy_adapter_maps_only_approved_to_accepted() -> None:
    candidate = CandidateEvidence.from_dict(_candidate_payload())
    legacy = {
        "schema_version": 1,
        "decision": "approved",
        "checkpoint_sha256": candidate.selected_checkpoint_sha256,
        "reviewer": "owner",
        "reviewed_at": "2026-09-12T00:00:00+00:00",
    }

    review = VisualReviewV2.from_legacy_dict(
        legacy,
        candidate=candidate,
        video_sha256s=("c" * 64,),
    )

    assert review.decision == "accepted"
    assert review.schema_version == 2
    assert review.comments == "Adapted from visual review schema 1."


def test_candidate_prerequisite_uses_one_review_decision_for_curriculum_and_final() -> None:
    candidate = CandidateEvidence.from_dict(_candidate_payload())
    review = VisualReviewV2.from_dict(_review_payload())

    result = assess_candidate_prerequisites(candidate, review)

    assert result["curriculum_authorized"] is True
    assert result["final_freeze_authorized"] is True
    assert all(result["checks"].values())

    rejected = replace(review, decision="rejected")
    result = assess_candidate_prerequisites(candidate, rejected)
    assert result["curriculum_authorized"] is True
    assert result["final_freeze_authorized"] is False


@pytest.mark.parametrize(
    ("schema_version", "decision"),
    [(1, True), (1, 1), (1, "yes"), (1, "accepted"), (True, "approved")],
)
def test_legacy_adapter_rejects_arbitrary_truthy_decisions(
    schema_version: object, decision: object
) -> None:
    candidate = CandidateEvidence.from_dict(_candidate_payload())
    legacy = {
        "schema_version": schema_version,
        "decision": decision,
        "checkpoint_sha256": candidate.selected_checkpoint_sha256,
        "reviewer": "owner",
        "reviewed_at": "2026-09-12T00:00:00+00:00",
    }

    with pytest.raises(ValueError, match="schema|decision"):
        VisualReviewV2.from_legacy_dict(
            legacy,
            candidate=candidate,
            video_sha256s=("c" * 64,),
        )
