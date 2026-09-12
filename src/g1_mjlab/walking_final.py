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
