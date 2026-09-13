"""Fail-closed qualification of speed-specific walking references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import sha256_file, write_atomic_json


@dataclass(frozen=True, slots=True)
class ReferenceCandidateEvidence:
    candidate_id: str
    speed_m_s: float
    attempt: int
    adaptation_path: Path
    dynamics_path: Path | None


def qualify_reference_candidates(
    candidates: tuple[ReferenceCandidateEvidence, ...],
    output_path: Path,
    *,
    required_speeds_m_s: tuple[float, ...] = (0.4, 0.6, 0.8),
) -> dict[str, Any]:
    """Accept only one fully passing candidate at every required speed."""
    results: list[dict[str, Any]] = []
    accepted: dict[float, str] = {}
    for candidate in candidates:
        adaptation = _read_object(candidate.adaptation_path)
        dynamics = (
            None if candidate.dynamics_path is None else _read_object(candidate.dynamics_path)
        )
        kinematic_pass = adaptation.get("passed") is True
        dynamics_pass = dynamics is not None and dynamics.get("passed") is True
        passed = kinematic_pass and dynamics_pass
        result = {
            "candidate_id": candidate.candidate_id,
            "speed_m_s": candidate.speed_m_s,
            "attempt": candidate.attempt,
            "adaptation_sha256": sha256_file(candidate.adaptation_path),
            "reference_sha256": adaptation.get("output_sha256"),
            "dynamics_sha256": (
                None
                if candidate.dynamics_path is None
                else sha256_file(candidate.dynamics_path)
            ),
            "kinematic_pass": kinematic_pass,
            "dynamics_pass": dynamics_pass,
            "passed": passed,
        }
        if passed and candidate.speed_m_s not in accepted:
            accepted[candidate.speed_m_s] = candidate.candidate_id
        results.append(result)
    missing = [speed for speed in required_speeds_m_s if speed not in accepted]
    decision = {
        "schema_version": 1,
        "status": "qualified" if not missing else "blocked",
        "required_speeds_m_s": list(required_speeds_m_s),
        "accepted_candidates": {str(speed): accepted[speed] for speed in sorted(accepted)},
        "unqualified_speeds_m_s": missing,
        "candidates": results,
        "reference_bank_frozen": not missing,
        "acceptance_targets_frozen": not missing,
        "downstream_training_authorized": not missing,
        "required_change": (
            None
            if not missing
            else "Provide licensed speed-specific G1 trajectories with dynamically feasible "
            "support, or revise the measured controller contract from hardware evidence; then "
            "version and rerun P05-04 without relaxing its gates."
        ),
    }
    write_atomic_json(output_path, decision)
    return decision


def _read_object(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value
