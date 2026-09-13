from __future__ import annotations

import json
from pathlib import Path

from g1_mjlab.motion.soft_reference_admission import (
    SoftReferenceCandidate,
    decide_soft_reference_admission,
)


def _json(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _candidate(tmp_path: Path, speed: float, *, kinematics_pass: bool = True) -> SoftReferenceCandidate:
    code = str(speed).replace(".", "")
    reference = tmp_path / f"reference-{code}.npz"
    reference.write_bytes(f"reference-{speed}".encode())
    import hashlib

    reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    return SoftReferenceCandidate(
        candidate_id=f"candidate-{code}",
        speed_m_s=speed,
        reference_path=reference,
        adaptation_path=_json(
            tmp_path / f"adaptation-{code}.json",
            {"passed": True, "output_sha256": reference_hash},
        ),
        kinematics_path=_json(
            tmp_path / f"kinematics-{code}.json",
            {"passed": kinematics_pass, "reference_sha256": reference_hash},
        ),
        dynamics_path=_json(
            tmp_path / f"dynamics-{code}.json",
            {"passed": False, "reference_sha256": reference_hash},
        ),
    )


def test_soft_reference_can_be_admitted_with_failed_dynamics_diagnostic(tmp_path: Path) -> None:
    candidates = tuple(_candidate(tmp_path, speed) for speed in (0.4, 0.6, 0.8))

    result = decide_soft_reference_admission(candidates, tmp_path / "decision.json")

    assert result["status"] == "admitted"
    assert result["training_authorized"] is True
    assert result["policy_rollout_dynamics_gate_required"] is True
    assert all(item["dynamics_diagnostic"] == "failed" for item in result["candidates"])


def test_soft_reference_rejects_failed_kinematics(tmp_path: Path) -> None:
    candidates = (
        _candidate(tmp_path, 0.4),
        _candidate(tmp_path, 0.6, kinematics_pass=False),
        _candidate(tmp_path, 0.8),
    )

    result = decide_soft_reference_admission(candidates, tmp_path / "decision.json")

    assert result["status"] == "rejected"
    assert result["training_authorized"] is False
    assert result["unadmitted_speeds_m_s"] == [0.6]


def test_soft_reference_rejects_missing_dynamics_or_hash_mismatch(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, 0.4)
    mismatched = _json(
        tmp_path / "mismatched.json",
        {"passed": True, "reference_sha256": "0" * 64},
    )

    result = decide_soft_reference_admission(
        (
            SoftReferenceCandidate(
                candidate.candidate_id,
                candidate.speed_m_s,
                candidate.reference_path,
                candidate.adaptation_path,
                mismatched,
                None,
            ),
        ),
        tmp_path / "decision.json",
        required_speeds_m_s=(0.4,),
    )

    assert result["status"] == "rejected"
    reasons = result["candidates"][0]["reasons"]
    assert "kinematics_reference_hash_mismatch" in reasons
    assert "missing_dynamics_diagnostic" in reasons
