"""Strict immutable schemas for walking candidate and human-review evidence."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import sha256_file

_SHA256 = re.compile(r"[0-9a-f]{64}")
_NUMERIC_STATUSES = frozenset(
    {"pending", "acquisition_qualified", "development_qualified", "rejected"}
)
_VISUAL_STATUSES = frozenset({"pending", "accepted", "rejected"})
_FINAL_STATUSES = frozenset({"not_run", "qualified", "failed"})


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _require_timestamp(value: object, field: str) -> str:
    timestamp = _require_identifier(value, field)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a UTC offset")
    return timestamp


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise ValueError("artifact reference path must be a string")
        parsed = PurePosixPath(self.path)
        if (
            not self.path
            or parsed.is_absolute()
            or ".." in parsed.parts
            or parsed.as_posix() != self.path
        ):
            raise ValueError("artifact reference path must be a normalized relative POSIX path")
        _require_sha256(self.sha256, "artifact sha256")

    @classmethod
    def from_dict(cls, value: object) -> ArtifactReference:
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ValueError("artifact reference fields do not match schema")
        path = value["path"]
        return cls(path=path, sha256=_require_sha256(value["sha256"], "artifact sha256"))

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    schema_version: int = field(default=2, init=False)
    method_id: str
    task_id: str
    layout_id: str
    selected_checkpoint_sha256: str
    source_sha256: str
    reference_sha256: str
    controller_sha256: str
    model_sha256: str
    contract_sha256: str
    evaluator_sha256: str
    criteria_sha256: str
    scenario_set_sha256: str
    trial_results: tuple[ArtifactReference, ...]
    speed_coverage_m_s: tuple[float, float]
    numeric_status: str
    visual_status: str
    final_status: str

    def __post_init__(self) -> None:
        for name in ("method_id", "task_id", "layout_id"):
            _require_identifier(getattr(self, name), name)
        for name in (
            "selected_checkpoint_sha256",
            "source_sha256",
            "reference_sha256",
            "controller_sha256",
            "model_sha256",
            "contract_sha256",
            "evaluator_sha256",
            "criteria_sha256",
            "scenario_set_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.trial_results, tuple) or not self.trial_results:
            raise ValueError("trial_results must be a nonempty tuple")
        if any(not isinstance(item, ArtifactReference) for item in self.trial_results):
            raise ValueError("trial_results must contain artifact references")
        paths = [item.path for item in self.trial_results]
        if len(paths) != len(set(paths)):
            raise ValueError("trial result paths must be unique")
        if (
            not isinstance(self.speed_coverage_m_s, tuple)
            or len(self.speed_coverage_m_s) != 2
            or any(type(item) not in {int, float} for item in self.speed_coverage_m_s)
        ):
            raise ValueError("speed_coverage_m_s must be a pair of numbers")
        if (
            not all(math.isfinite(item) for item in self.speed_coverage_m_s)
            or self.speed_coverage_m_s[0] < 0
            or self.speed_coverage_m_s[0] > self.speed_coverage_m_s[1]
        ):
            raise ValueError("speed_coverage_m_s must be finite, nonnegative, and ordered")
        if self.numeric_status not in _NUMERIC_STATUSES:
            raise ValueError("numeric_status is not supported")
        if self.visual_status not in _VISUAL_STATUSES:
            raise ValueError("visual_status is not supported")
        if self.final_status not in _FINAL_STATUSES:
            raise ValueError("final_status is not supported")

    @classmethod
    def load(cls, path: Path) -> CandidateEvidence:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value: object) -> CandidateEvidence:
        fields = {
            "schema_version",
            "method_id",
            "task_id",
            "layout_id",
            "selected_checkpoint_sha256",
            "source_sha256",
            "reference_sha256",
            "controller_sha256",
            "model_sha256",
            "contract_sha256",
            "evaluator_sha256",
            "criteria_sha256",
            "scenario_set_sha256",
            "trial_results",
            "speed_coverage_m_s",
            "numeric_status",
            "visual_status",
            "final_status",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("candidate evidence fields do not match schema 2")
        if type(value["schema_version"]) is not int or value["schema_version"] != 2:
            raise ValueError("candidate evidence requires schema version 2")
        trial_results = value["trial_results"]
        coverage = value["speed_coverage_m_s"]
        if not isinstance(trial_results, list) or not trial_results:
            raise ValueError("trial_results must be a nonempty list")
        if not isinstance(coverage, list) or len(coverage) != 2:
            raise ValueError("speed_coverage_m_s must contain two values")
        if any(type(item) not in {int, float} for item in coverage):
            raise ValueError("speed_coverage_m_s values must be numbers")
        typed_coverage = (float(coverage[0]), float(coverage[1]))
        if (
            not all(math.isfinite(item) for item in typed_coverage)
            or typed_coverage[0] < 0
            or typed_coverage[0] > typed_coverage[1]
        ):
            raise ValueError("speed_coverage_m_s must be finite, nonnegative, and ordered")
        identifiers = ("method_id", "task_id", "layout_id")
        hashes = (
            "selected_checkpoint_sha256",
            "source_sha256",
            "reference_sha256",
            "controller_sha256",
            "model_sha256",
            "contract_sha256",
            "evaluator_sha256",
            "criteria_sha256",
            "scenario_set_sha256",
        )
        parsed_references = tuple(ArtifactReference.from_dict(item) for item in trial_results)
        paths = [item.path for item in parsed_references]
        if len(paths) != len(set(paths)):
            raise ValueError("trial result paths must be unique")
        for name in identifiers:
            _require_identifier(value[name], name)
        for name in hashes:
            _require_sha256(value[name], name)
        if value["numeric_status"] not in _NUMERIC_STATUSES:
            raise ValueError("numeric_status is not supported")
        if value["visual_status"] not in _VISUAL_STATUSES:
            raise ValueError("visual_status is not supported")
        if value["final_status"] not in _FINAL_STATUSES:
            raise ValueError("final_status is not supported")
        return cls(
            method_id=value["method_id"],
            task_id=value["task_id"],
            layout_id=value["layout_id"],
            selected_checkpoint_sha256=value["selected_checkpoint_sha256"],
            source_sha256=value["source_sha256"],
            reference_sha256=value["reference_sha256"],
            controller_sha256=value["controller_sha256"],
            model_sha256=value["model_sha256"],
            contract_sha256=value["contract_sha256"],
            evaluator_sha256=value["evaluator_sha256"],
            criteria_sha256=value["criteria_sha256"],
            scenario_set_sha256=value["scenario_set_sha256"],
            trial_results=parsed_references,
            speed_coverage_m_s=typed_coverage,
            numeric_status=value["numeric_status"],
            visual_status=value["visual_status"],
            final_status=value["final_status"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "method_id": self.method_id,
            "task_id": self.task_id,
            "layout_id": self.layout_id,
            "selected_checkpoint_sha256": self.selected_checkpoint_sha256,
            "source_sha256": self.source_sha256,
            "reference_sha256": self.reference_sha256,
            "controller_sha256": self.controller_sha256,
            "model_sha256": self.model_sha256,
            "contract_sha256": self.contract_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "criteria_sha256": self.criteria_sha256,
            "scenario_set_sha256": self.scenario_set_sha256,
            "trial_results": [item.to_dict() for item in self.trial_results],
            "speed_coverage_m_s": list(self.speed_coverage_m_s),
            "numeric_status": self.numeric_status,
            "visual_status": self.visual_status,
            "final_status": self.final_status,
        }

    def verify_references(self, root: Path) -> None:
        resolved_root = root.resolve(strict=True)
        for reference in self.trial_results:
            path = (resolved_root / reference.path).resolve(strict=True)
            if not path.is_relative_to(resolved_root):
                raise ValueError(f"artifact reference escapes root: {reference.path}")
            if sha256_file(path) != reference.sha256:
                raise ValueError(f"artifact hash does not match reference: {reference.path}")


@dataclass(frozen=True, slots=True)
class VisualReviewV2:
    decision: str
    checkpoint_sha256: str
    model_sha256: str
    reference_sha256: str
    contract_sha256: str
    video_sha256s: tuple[str, ...]
    reviewer: str
    reviewed_at: str
    comments: str
    schema_version: int = field(default=2, init=False)

    def __post_init__(self) -> None:
        if self.decision not in {"accepted", "rejected"} or not isinstance(self.decision, str):
            raise ValueError("visual review decision must be accepted or rejected")
        for name in (
            "checkpoint_sha256",
            "model_sha256",
            "reference_sha256",
            "contract_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.video_sha256s, tuple) or not self.video_sha256s:
            raise ValueError("video_sha256s must be a nonempty ordered tuple")
        for index, digest in enumerate(self.video_sha256s):
            _require_sha256(digest, f"video_sha256s[{index}]")
        if len(self.video_sha256s) != len(set(self.video_sha256s)):
            raise ValueError("video_sha256s must not contain duplicates")
        _require_identifier(self.reviewer, "reviewer")
        _require_timestamp(self.reviewed_at, "reviewed_at")
        if not isinstance(self.comments, str):
            raise ValueError("comments must be a string")

    @classmethod
    def load(cls, path: Path) -> VisualReviewV2:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value: object) -> VisualReviewV2:
        fields = {
            "schema_version",
            "decision",
            "checkpoint_sha256",
            "model_sha256",
            "reference_sha256",
            "contract_sha256",
            "video_sha256s",
            "reviewer",
            "reviewed_at",
            "comments",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("visual review fields do not match schema 2")
        if type(value["schema_version"]) is not int or value["schema_version"] != 2:
            raise ValueError("visual review requires schema version 2")
        if value["decision"] not in {"accepted", "rejected"}:
            raise ValueError("visual review decision must be accepted or rejected")
        video_sha256s = value["video_sha256s"]
        if not isinstance(video_sha256s, list) or not video_sha256s:
            raise ValueError("video_sha256s must be a nonempty ordered list")
        parsed_videos = tuple(
            _require_sha256(item, f"video_sha256s[{index}]")
            for index, item in enumerate(video_sha256s)
        )
        if len(parsed_videos) != len(set(parsed_videos)):
            raise ValueError("video_sha256s must not contain duplicates")
        reviewer = _require_identifier(value["reviewer"], "reviewer")
        reviewed_at = _require_timestamp(value["reviewed_at"], "reviewed_at")
        comments = value["comments"]
        if not isinstance(comments, str):
            raise ValueError("comments must be a string")
        decision = value["decision"]
        if not isinstance(decision, str):
            raise ValueError("visual review decision must be a string")
        return cls(
            decision=decision,
            checkpoint_sha256=_require_sha256(value["checkpoint_sha256"], "checkpoint_sha256"),
            model_sha256=_require_sha256(value["model_sha256"], "model_sha256"),
            reference_sha256=_require_sha256(value["reference_sha256"], "reference_sha256"),
            contract_sha256=_require_sha256(value["contract_sha256"], "contract_sha256"),
            video_sha256s=parsed_videos,
            reviewer=reviewer,
            reviewed_at=reviewed_at,
            comments=comments,
        )

    @classmethod
    def from_legacy_dict(
        cls,
        value: object,
        *,
        candidate: CandidateEvidence,
        video_sha256s: tuple[str, ...],
    ) -> VisualReviewV2:
        fields = {
            "schema_version",
            "decision",
            "checkpoint_sha256",
            "reviewer",
            "reviewed_at",
        }
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
        ):
            raise ValueError("legacy visual review fields do not match schema 1")
        decision = value["decision"]
        if decision not in {"approved", "rejected"} or not isinstance(decision, str):
            raise ValueError("legacy visual review decision must be approved or rejected")
        if value["checkpoint_sha256"] != candidate.selected_checkpoint_sha256:
            raise ValueError("legacy visual review checkpoint does not match candidate")
        return cls.from_dict(
            {
                "schema_version": 2,
                "decision": "accepted" if decision == "approved" else "rejected",
                "checkpoint_sha256": candidate.selected_checkpoint_sha256,
                "model_sha256": candidate.model_sha256,
                "reference_sha256": candidate.reference_sha256,
                "contract_sha256": candidate.contract_sha256,
                "video_sha256s": list(video_sha256s),
                "reviewer": value["reviewer"],
                "reviewed_at": value["reviewed_at"],
                "comments": "Adapted from visual review schema 1.",
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision": self.decision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_sha256": self.model_sha256,
            "reference_sha256": self.reference_sha256,
            "contract_sha256": self.contract_sha256,
            "video_sha256s": list(self.video_sha256s),
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "comments": self.comments,
        }

    def validate_candidate(self, candidate: CandidateEvidence) -> None:
        expected = (
            candidate.selected_checkpoint_sha256,
            candidate.model_sha256,
            candidate.reference_sha256,
            candidate.contract_sha256,
        )
        actual = (
            self.checkpoint_sha256,
            self.model_sha256,
            self.reference_sha256,
            self.contract_sha256,
        )
        if actual != expected:
            raise ValueError("visual review does not match candidate identity")


def assess_candidate_prerequisites(
    candidate: CandidateEvidence, review: VisualReviewV2 | None
) -> dict[str, object]:
    """Use one checkpoint-bound review schema for curriculum and final freeze."""
    if review is not None:
        review.validate_candidate(candidate)
    checks = {
        "development_qualified": candidate.numeric_status == "development_qualified",
        "speed_domain_0_to_0_8_m_s": candidate.speed_coverage_m_s[0] <= 0.0
        and candidate.speed_coverage_m_s[1] >= 0.8,
        "visual_review_accepted": review is not None and review.decision == "accepted",
        "final_suite_not_previously_run": candidate.final_status == "not_run",
    }
    final_common = all(
        checks[name]
        for name in (
            "development_qualified",
            "speed_domain_0_to_0_8_m_s",
            "visual_review_accepted",
        )
    )
    return {
        "schema_version": 2,
        "checks": checks,
        "curriculum_authorized": candidate.numeric_status
        in {"acquisition_qualified", "development_qualified"}
        and candidate.speed_coverage_m_s[0] <= 0.6 <= candidate.speed_coverage_m_s[1],
        "final_freeze_authorized": final_common and checks["final_suite_not_previously_run"],
        "checkpoint_sha256": candidate.selected_checkpoint_sha256,
        "blockers": [name for name, passed in checks.items() if not passed],
    }
