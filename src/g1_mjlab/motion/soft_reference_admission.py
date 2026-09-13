"""Fail-closed admission of kinematic references for bounded policy learning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import sha256_file, write_atomic_json


@dataclass(frozen=True, slots=True)
class SoftReferenceCandidate:
    candidate_id: str
    speed_m_s: float
    reference_path: Path
    adaptation_path: Path
    kinematics_path: Path
    dynamics_path: Path | None


def decide_soft_reference_admission(
    candidates: tuple[SoftReferenceCandidate, ...],
    output_path: Path,
    *,
    required_speeds_m_s: tuple[float, ...] = (0.4, 0.6, 0.8),
    source_license: str = "test-only",
) -> dict[str, Any]:
    """Admit one hash-bound kinematic reference at each required speed."""
    results: list[dict[str, Any]] = []
    accepted: dict[float, str] = {}
    for candidate in candidates:
        reference_hash = sha256_file(candidate.reference_path)
        adaptation = _read_object(candidate.adaptation_path)
        kinematics = _read_object(candidate.kinematics_path)
        dynamics_path = candidate.dynamics_path
        dynamics = None if dynamics_path is None else _read_object(dynamics_path)
        reasons: list[str] = []
        if adaptation.get("output_sha256") != reference_hash:
            reasons.append("adaptation_reference_hash_mismatch")
        if kinematics.get("reference_sha256") != reference_hash:
            reasons.append("kinematics_reference_hash_mismatch")
        if adaptation.get("passed") is not True:
            reasons.append("adaptation_failed")
        if kinematics.get("passed") is not True:
            reasons.append("kinematics_failed")
        if dynamics_path is None:
            reasons.append("missing_dynamics_diagnostic")
            dynamics_state = "missing"
            dynamics_hash = None
        else:
            assert dynamics is not None
            if dynamics.get("reference_sha256") != reference_hash:
                reasons.append("dynamics_reference_hash_mismatch")
            dynamics_state = "passed" if dynamics.get("passed") is True else "failed"
            dynamics_hash = sha256_file(dynamics_path)
        admitted = not reasons
        if admitted and candidate.speed_m_s not in accepted:
            accepted[candidate.speed_m_s] = candidate.candidate_id
        results.append(
            {
                "candidate_id": candidate.candidate_id,
                "speed_m_s": candidate.speed_m_s,
                "reference_sha256": reference_hash,
                "adaptation_sha256": sha256_file(candidate.adaptation_path),
                "kinematics_sha256": sha256_file(candidate.kinematics_path),
                "dynamics_sha256": dynamics_hash,
                "dynamics_diagnostic": dynamics_state,
                "reasons": reasons,
                "admitted": admitted,
            }
        )
    missing = [speed for speed in required_speeds_m_s if speed not in accepted]
    status = "admitted" if not missing else "rejected"
    decision = {
        "schema_version": 2,
        "semantics": "soft-reference-training-admission-not-policy-qualification",
        "status": status,
        "source_license": source_license,
        "required_speeds_m_s": list(required_speeds_m_s),
        "accepted_candidates": {str(speed): accepted[speed] for speed in sorted(accepted)},
        "unadmitted_speeds_m_s": missing,
        "candidates": results,
        "training_authorized": not missing,
        "policy_rollout_dynamics_gate_required": True,
        "reference_dynamics_claim": "diagnostic-only; no feasibility claim",
    }
    write_atomic_json(output_path, decision)
    return decision


def authorized_reference_hashes(
    decision: dict[str, Any], required_speeds_m_s: tuple[float, ...]
) -> tuple[str, ...]:
    """Resolve selected hashes only from an explicitly training-authorized decision."""
    legacy_qualified = (
        decision.get("schema_version") == 1
        and decision.get("status") == "qualified"
        and decision.get("downstream_training_authorized") is True
    )
    soft_admitted = (
        decision.get("schema_version") == 2
        and decision.get("semantics")
        == "soft-reference-training-admission-not-policy-qualification"
        and decision.get("status") == "admitted"
        and decision.get("training_authorized") is True
    )
    if not (legacy_qualified or soft_admitted):
        raise ValueError("reference decision is not training-authorized")
    accepted = decision.get("accepted_candidates")
    candidate_values = decision.get("candidates")
    if not isinstance(accepted, dict) or not isinstance(candidate_values, list):
        raise ValueError("reference decision has malformed candidate selection")
    candidates = {
        value.get("candidate_id"): value
        for value in candidate_values
        if isinstance(value, dict) and isinstance(value.get("candidate_id"), str)
    }
    hashes: list[str] = []
    for speed in required_speeds_m_s:
        candidate_id = accepted.get(str(speed))
        candidate = candidates.get(candidate_id)
        reference_hash = None if candidate is None else candidate.get("reference_sha256")
        if not isinstance(reference_hash, str) or len(reference_hash) != 64:
            raise ValueError(f"reference decision has no valid hash for {speed} m/s")
        hashes.append(reference_hash)
    return tuple(hashes)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value
