"""Fail-closed prerequisite gate for walking speed-curriculum training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import sha256_file, write_atomic_json


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def assess_curriculum_prerequisite(
    experiment_table_path: Path,
    visual_approval_path: Path | None,
    output: Path,
) -> dict[str, Any]:
    """Authorize curriculum compute only for a qualified, visually approved exact checkpoint."""
    if output.exists():
        raise FileExistsError(f"curriculum gate output already exists: {output}")
    table = _object(experiment_table_path.resolve(strict=True))
    if table.get("schema_version") != 1 or table.get("outcome") not in {
        "candidate",
        "failed_hypothesis",
    }:
        raise ValueError("unsupported Stage 19 experiment table")
    candidate = table.get("selected_candidate")
    qualified = bool(table.get("development_qualified")) and isinstance(candidate, dict)
    checkpoint_sha256 = candidate.get("checkpoint_sha256") if isinstance(candidate, dict) else None
    approved = False
    approval_sha256: str | None = None
    if visual_approval_path is not None:
        approval_path = visual_approval_path.resolve(strict=True)
        approval = _object(approval_path)
        expected_fields = {
            "schema_version",
            "decision",
            "checkpoint_sha256",
            "reviewer",
            "reviewed_at",
        }
        if set(approval) != expected_fields or approval["schema_version"] != 1:
            raise ValueError("visual approval fields do not match schema")
        if checkpoint_sha256 is None or approval["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("visual approval checkpoint does not match selected candidate")
        approved = approval["decision"] == "approved" and bool(approval["reviewer"])
        approval_sha256 = sha256_file(approval_path)
    blockers: list[str] = []
    if not qualified:
        blockers.append("missing_development_qualified_0.6_m_s_policy")
    if not approved:
        blockers.append("missing_owner_visual_approval")
    result = {
        "schema_version": 1,
        "status": "authorized" if not blockers else "blocked",
        "training_authorized": not blockers,
        "blockers": blockers,
        "experiment_table": str(experiment_table_path.resolve()),
        "experiment_table_sha256": sha256_file(experiment_table_path),
        "selected_checkpoint_sha256": checkpoint_sha256,
        "visual_approval": str(visual_approval_path.resolve())
        if visual_approval_path is not None
        else None,
        "visual_approval_sha256": approval_sha256,
        "forbidden_actions_when_blocked": [
            "speed_curriculum_training",
            "seed_replication_training",
            "candidate_freeze",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic_json(output, result)
    return result
