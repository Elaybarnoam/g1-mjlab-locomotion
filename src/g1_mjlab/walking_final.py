"""Frozen walking-v1 final scenarios, prerequisite gate, and paired qualification."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import sha256_file
from .gait_evaluation.acceptance import (
    FinalAcceptanceCriteria,
    FinalTrialMeasurements,
    assess_final_trial,
)

_CATEGORIES = ("slow", "medium", "fast", "start_stop")


def _canonical(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _segments(category: str) -> list[dict[str, Any]]:
    values: tuple[tuple[int, float], ...]
    if category == "start_stop":
        values = ((5, 0.0), (10, 0.4), (5, 0.0), (10, 0.6), (5, 0.0), (10, 0.8), (15, 0.0))
    else:
        speed = {"slow": 0.4, "medium": 0.6, "fast": 0.8}[category]
        values = ((5, 0.0), (45, speed), (10, 0.0))
    return [
        {"duration_s": float(duration), "command": [speed, 0.0, 0.0]} for duration, speed in values
    ]


def _yaw_quaternion(base: np.ndarray, yaw: float) -> list[float]:
    yaw_q = np.asarray([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    w1, x1, y1, z1 = yaw_q
    w2, x2, y2, z2 = base
    result = np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    result /= np.linalg.norm(result)
    return [float(value) for value in result]


def build_final_scenario_payload(base_qpos: list[float], base_qvel: list[float]) -> dict[str, Any]:
    """Generate the untouched 100-case suite using only reserved seeds 20000–20099."""
    if len(base_qpos) != 36 or len(base_qvel) != 35:
        raise ValueError("final base state must contain qpos[36] and qvel[35]")
    qpos_source = np.asarray(base_qpos, dtype=float)
    qvel_source = np.asarray(base_qvel, dtype=float)
    if not np.isfinite(qpos_source).all() or not np.isfinite(qvel_source).all():
        raise ValueError("final base state must be finite")
    if not np.isclose(np.linalg.norm(qpos_source[3:7]), 1.0, atol=1e-6):
        raise ValueError("final base quaternion must be normalized")
    scenarios: list[dict[str, Any]] = []
    for index, seed in enumerate(range(20000, 20100)):
        category = _CATEGORIES[index // 25]
        rng = np.random.default_rng(seed)
        qpos = qpos_source.copy()
        qvel = qvel_source.copy()
        qpos[:2] += rng.uniform(-0.02, 0.02, 2)
        qpos[3:7] = _yaw_quaternion(qpos_source[3:7], float(rng.uniform(-0.10, 0.10)))
        qpos[7:] += rng.uniform(-0.01, 0.01, 29)
        qvel[:] += rng.uniform(-0.02, 0.02, 35)
        scenarios.append(
            {
                "name": f"final-{index:03d}-{category}",
                "seed": seed,
                "category": category,
                "initialization": "standing",
                "initial_phase": (index % 8) / 8,
                "initial_qpos": qpos.tolist(),
                "initial_qvel": qvel.tolist(),
                "segments": _segments(category),
            }
        )
    return {"schema_version": 3, "name": "walking-v1-final-v1", "scenarios": scenarios}


def assess_final_prerequisite(
    bundle_path: Path, visual_approval_path: Path | None
) -> dict[str, Any]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(bundle, dict):
        raise ValueError("walking bundle must be an object")
    domain = bundle.get("qualified_command_domain_m_s")
    approval: Any = None
    if visual_approval_path is not None and visual_approval_path.is_file():
        approval = json.loads(visual_approval_path.read_text(encoding="utf-8"))
    checks = {
        "development_qualified_policy": bundle.get("status") == "development_qualified",
        "qualified_speed_domain_0_to_0_8_m_s": isinstance(domain, list)
        and len(domain) == 2
        and float(domain[0]) <= 0.0
        and float(domain[1]) >= 0.8,
        "owner_visual_acceptance": isinstance(approval, dict)
        and approval.get("decision") == "accepted"
        and approval.get("checkpoint_sha256") == bundle.get("checkpoint_sha256"),
    }
    return {
        "schema_version": 1,
        "gate": "walking-v1-final-freeze-prerequisite",
        "final_execution_authorized": all(checks.values()),
        "checks": checks,
        "missing": [name for name, passed in checks.items() if not passed],
        "checkpoint_sha256": bundle.get("checkpoint_sha256"),
    }


def freeze_final_suite(
    bundle_dir: Path,
    base_scenarios: Path,
    acceptance_path: Path,
    visual_approval_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Freeze scenarios and all governing hashes before any final outcome is read."""
    bundle_path = bundle_dir / "walking-policy-bundle.json"
    gate = assess_final_prerequisite(bundle_path, visual_approval_path)
    if not gate["final_execution_authorized"]:
        raise ValueError(f"final walking execution is blocked: {gate['missing']}")
    if output.exists():
        raise FileExistsError(f"final freeze output already exists: {output}")
    base = json.loads(base_scenarios.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or not base.get("scenarios"):
        raise ValueError("base scenarios must contain at least one explicit state")
    first = base["scenarios"][0]
    scenarios = build_final_scenario_payload(first["initial_qpos"], first["initial_qvel"])
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    freeze = {
        "schema_version": 1,
        "status": "frozen_unseen",
        "scenario_seed_range": [20000, 20099],
        "scenario_sha256": _canonical(scenarios),
        "acceptance": acceptance,
        "hashes": {
            "checkpoint": bundle["checkpoint_sha256"],
            "onnx": bundle["files"]["policy.onnx"],
            "model": bundle["files"]["model.mjb"],
            "contract": bundle["files"]["contract.json"],
            "reference": bundle["files"]["reference.npz"],
            "command": bundle["files"]["host-profile.json"],
            "acceptance": sha256_file(acceptance_path),
            "visual_approval": sha256_file(visual_approval_path),
        },
        "tolerances": {
            "observation_absolute": 1e-5,
            "observation_relative": 1e-4,
            "action_absolute": 1e-4,
            "action_relative": 1e-4,
        },
    }
    output.mkdir(parents=True)
    (output / "scenarios.json").write_text(
        json.dumps(scenarios, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return freeze


def _backend_result(summary: dict[str, Any], minimum_stratum: int) -> dict[str, Any]:
    trials = summary.get("trials")
    if not isinstance(trials, list) or len(trials) != 100:
        raise ValueError("final backend summary must contain exactly 100 trials")
    passed = [
        bool(item.get("function_passed") is True and item.get("style_passed") is True)
        for item in trials
    ]
    strata = Counter(
        item.get("category") for item, accepted in zip(trials, passed, strict=True) if accepted
    )
    if set(item.get("category") for item in trials) != set(_CATEGORIES):
        raise ValueError("final backend summary has invalid strata")
    return {
        "overall_passes": sum(passed),
        "stratum_passes": {category: strata[category] for category in _CATEGORIES},
        "all_strata_pass": all(strata[category] >= minimum_stratum for category in _CATEGORIES),
    }


def qualify_final_walking(
    freeze_path: Path, mjlab_summary_path: Path, native_summary_path: Path, output: Path
) -> dict[str, Any]:
    """Bind the one-shot paired result; a failure remains a recorded failure."""
    if output.exists():
        raise FileExistsError(f"walking qualification output already exists: {output}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    mjlab = json.loads(mjlab_summary_path.read_text(encoding="utf-8"))
    native = json.loads(native_summary_path.read_text(encoding="utf-8"))
    acceptance = freeze["acceptance"]
    expected_checkpoint = freeze["hashes"]["checkpoint"]
    for name, summary in (("mjlab", mjlab), ("native", native)):
        if summary.get("checkpoint_sha256") != expected_checkpoint:
            raise ValueError(f"{name} result does not match frozen checkpoint")
    backend_results = {
        "mjlab": _backend_result(mjlab, int(acceptance["minimum_stratum_passes"])),
        "native": _backend_result(native, int(acceptance["minimum_stratum_passes"])),
    }
    qualified = all(
        result["overall_passes"] >= int(acceptance["minimum_overall_passes"])
        and result["all_strata_pass"]
        for result in backend_results.values()
    )
    result = {
        "schema_version": 1,
        "qualified": qualified,
        "policy_status": "qualified" if qualified else "failed_final_qualification",
        "freeze_sha256": sha256_file(freeze_path),
        "mjlab_summary_sha256": sha256_file(mjlab_summary_path),
        "native_summary_sha256": sha256_file(native_summary_path),
        "backends": backend_results,
        "checkpoint_sha256": expected_checkpoint,
        "reuse_rule": "a failed final suite becomes development evidence and cannot be rerun as final",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


_FINAL_IDENTITY_KEYS = frozenset(
    {
        "checkpoint",
        "onnx",
        "model",
        "controller",
        "contract",
        "reference",
        "host_profile",
        "evaluator",
        "acceptance",
        "scenarios",
    }
)


def _strict_hashes(value: Any, *, location: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _FINAL_IDENTITY_KEYS:
        raise ValueError(f"{location} identities do not match schema")
    result = {str(key): str(item) for key, item in value.items()}
    if any(
        len(item) != 64 or any(character not in "0123456789abcdef" for character in item)
        for item in result.values()
    ):
        raise ValueError(f"{location} identities must be lowercase SHA-256 values")
    return result


def _v2_backend_result(
    summary: dict[str, Any],
    *,
    backend: str,
    identities: dict[str, str],
    expected_trials: list[dict[str, Any]],
    criteria: FinalAcceptanceCriteria,
) -> dict[str, Any]:
    if summary.get("schema_version") != 2 or summary.get("backend") != backend:
        raise ValueError(f"{backend} final summary has an invalid identity")
    if _strict_hashes(summary.get("identities"), location=backend) != identities:
        raise ValueError(f"{backend} final summary artifact identities differ from the freeze")
    trials = summary.get("trials")
    if not isinstance(trials, list) or len(trials) != len(expected_trials):
        raise ValueError(f"{backend} final summary must contain the complete frozen trial set")
    fields = ("trial_id", "seed", "category", "initial_state_sha256")
    actual_identities = [tuple(trial.get(field) for field in fields) for trial in trials]
    expected_identities = [tuple(trial.get(field) for field in fields) for trial in expected_trials]
    if actual_identities != expected_identities or len(set(actual_identities)) != len(
        actual_identities
    ):
        raise ValueError(f"{backend} result does not contain the exact frozen trial identities")

    accepted: list[bool] = []
    failures: dict[str, list[str]] = {}
    strata = Counter({category: 0 for category in _CATEGORIES})
    for trial in trials:
        measurements = trial.get("measurements")
        if not isinstance(measurements, dict):
            raise ValueError(f"{backend} trial is missing raw final measurements")
        result = assess_final_trial(FinalTrialMeasurements.from_dict(measurements), criteria)
        accepted.append(result.accepted)
        if result.accepted:
            strata[str(trial["category"])] += 1
        else:
            failures[str(trial["trial_id"])] = list(result.violations)
    return {
        "overall_passes": sum(accepted),
        "stratum_passes": {category: strata[category] for category in _CATEGORIES},
        "all_strata_pass": all(
            strata[category] >= criteria.minimum_stratum_passes for category in _CATEGORIES
        ),
        "failures": failures,
    }


def qualify_final_walking_v2(
    freeze_path: Path, mjlab_summary_path: Path, native_summary_path: Path, output: Path
) -> dict[str, Any]:
    """Recompute strict trial acceptance and bind both backends to one frozen suite."""
    if output.exists():
        raise FileExistsError(f"walking qualification output already exists: {output}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not isinstance(freeze, dict) or freeze.get("schema_version") != 2:
        raise ValueError("final freeze must use schema 2")
    identities = _strict_hashes(freeze.get("identities"), location="freeze")
    criteria_raw = freeze.get("criteria")
    if not isinstance(criteria_raw, dict) or set(criteria_raw) != set(
        FinalAcceptanceCriteria().to_dict()
    ):
        raise ValueError("final freeze criteria do not match schema")
    criteria = FinalAcceptanceCriteria(**criteria_raw)
    expected_trials = freeze.get("trials")
    if (
        not isinstance(expected_trials, list)
        or len(expected_trials) != 4 * criteria.trials_per_stratum
    ):
        raise ValueError("final freeze does not contain the required trial count")
    expected_categories = Counter(trial.get("category") for trial in expected_trials)
    if expected_categories != Counter(
        {category: criteria.trials_per_stratum for category in _CATEGORIES}
    ):
        raise ValueError("final freeze trial strata are not balanced")
    summaries = {
        "mjlab": json.loads(mjlab_summary_path.read_text(encoding="utf-8")),
        "native": json.loads(native_summary_path.read_text(encoding="utf-8")),
    }
    if any(not isinstance(value, dict) for value in summaries.values()):
        raise ValueError("final backend summaries must be JSON objects")
    backend_results = {
        name: _v2_backend_result(
            summary,
            backend=name,
            identities=identities,
            expected_trials=expected_trials,
            criteria=criteria,
        )
        for name, summary in summaries.items()
    }
    qualified = all(
        result["overall_passes"] >= criteria.minimum_overall_passes and result["all_strata_pass"]
        for result in backend_results.values()
    )
    result = {
        "schema_version": 2,
        "qualified": qualified,
        "policy_status": "qualified" if qualified else "failed_final_qualification",
        "freeze_sha256": sha256_file(freeze_path),
        "mjlab_summary_sha256": sha256_file(mjlab_summary_path),
        "native_summary_sha256": sha256_file(native_summary_path),
        "identities": identities,
        "backends": backend_results,
        "reuse_rule": "a failed final suite becomes development evidence and cannot be rerun as final",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def qualify_final_walking_dispatch(
    freeze_path: Path, mjlab_summary_path: Path, native_summary_path: Path, output: Path
) -> dict[str, Any]:
    """Route frozen qualification artifacts without weakening either schema."""
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not isinstance(freeze, dict):
        raise ValueError("final freeze must be a JSON object")
    if freeze.get("schema_version") == 2:
        return qualify_final_walking_v2(
            freeze_path, mjlab_summary_path, native_summary_path, output
        )
    if freeze.get("schema_version") == 1:
        return qualify_final_walking(freeze_path, mjlab_summary_path, native_summary_path, output)
    raise ValueError("unsupported final freeze schema")
