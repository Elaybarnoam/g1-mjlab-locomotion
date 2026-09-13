from __future__ import annotations

import json
from pathlib import Path

from g1_mjlab.motion.reference_qualification import (
    ReferenceCandidateEvidence,
    qualify_reference_candidates,
)


def _report(path: Path, passed: bool) -> Path:
    path.write_text(json.dumps({"passed": passed}), encoding="utf-8")
    return path


def test_qualification_is_fail_closed_for_missing_or_failed_speed(tmp_path: Path) -> None:
    candidates = (
        ReferenceCandidateEvidence(
            "040-a1", 0.4, 1, _report(tmp_path / "a.json", True), _report(tmp_path / "d.json", True)
        ),
        ReferenceCandidateEvidence(
            "060-a1", 0.6, 1, _report(tmp_path / "b.json", True), _report(tmp_path / "e.json", False)
        ),
    )

    result = qualify_reference_candidates(candidates, tmp_path / "decision.json")

    assert result["status"] == "blocked"
    assert result["unqualified_speeds_m_s"] == [0.6, 0.8]
    assert result["downstream_training_authorized"] is False


def test_qualification_requires_kinematics_and_dynamics_at_every_speed(tmp_path: Path) -> None:
    candidates = tuple(
        ReferenceCandidateEvidence(
            f"{speed}-a1",
            speed,
            1,
            _report(tmp_path / f"a-{speed}.json", True),
            _report(tmp_path / f"d-{speed}.json", True),
        )
        for speed in (0.4, 0.6, 0.8)
    )

    result = qualify_reference_candidates(candidates, tmp_path / "decision.json")

    assert result["status"] == "qualified"
    assert result["reference_bank_frozen"] is True
    assert result["downstream_training_authorized"] is True
