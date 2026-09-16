from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_mjlab.walking_curriculum import assess_curriculum_prerequisite


def test_curriculum_is_blocked_without_qualified_candidate_or_visual_approval(
    tmp_path: Path,
) -> None:
    table = tmp_path / "experiment-table.json"
    table.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "development_qualified": False,
                "selected_candidate": None,
                "outcome": "failed_hypothesis",
            }
        ),
        encoding="utf-8",
    )

    result = assess_curriculum_prerequisite(table, None, tmp_path / "gate.json")

    assert result["status"] == "blocked"
    assert not result["training_authorized"]
    assert result["blockers"] == [
        "missing_development_qualified_0.6_m_s_policy",
        "missing_owner_visual_approval",
    ]


def test_curriculum_requires_hash_matching_visual_approval(tmp_path: Path) -> None:
    table = tmp_path / "experiment-table.json"
    table.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "development_qualified": True,
                "selected_candidate": {"checkpoint_sha256": "a" * 64},
                "outcome": "candidate",
            }
        ),
        encoding="utf-8",
    )
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "decision": "approved",
                "checkpoint_sha256": "b" * 64,
                "reviewer": "owner",
                "reviewed_at": "2026-09-12T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint"):
        assess_curriculum_prerequisite(table, approval, tmp_path / "gate.json")
