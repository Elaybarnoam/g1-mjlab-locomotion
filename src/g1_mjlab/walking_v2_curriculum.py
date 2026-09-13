"""Fail-closed promotion gates for the walking-v2 curriculum."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .config import load_walking_v2_curriculum_profile

_SPEED_FRONTIERS = {
    "stand": (0.0,),
    "stand-walk-040": (0.0, 0.4),
    "add-060": (0.0, 0.4, 0.6),
    "add-080": (0.0, 0.4, 0.6, 0.8),
}


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    stage: str
    profile: Path
    profile_sha256: str
    updates: int
    gate: str


@dataclass(frozen=True, slots=True)
class CurriculumMethod:
    name: str
    seeds: tuple[int, ...]
    num_envs: int
    rollout_steps: int
    save_interval_updates: int
    fixed_speed_evaluation_horizon_s: float
    development_scenarios: Path
    development_criteria: Path
    stages: tuple[CurriculumStage, ...]


def _bound_path(root: Path, value: object, digest: object, name: str) -> Path:
    if not isinstance(value, str) or not isinstance(digest, str):
        raise ValueError(f"{name} path and digest must be strings")
    path = (root / value).resolve(strict=True)
    if len(digest) != 64 or sha256_file(path) != digest:
        raise ValueError(f"{name} SHA-256 mismatch")
    return path


def load_curriculum_method(path: Path) -> CurriculumMethod:
    """Load the frozen multi-seed curriculum and verify every bound input."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "name",
        "source_policy_role",
        "seeds",
        "num_envs",
        "rollout_steps",
        "save_interval_updates",
        "fixed_speed_evaluation_horizon_s",
        "development_scenarios",
        "development_criteria",
        "stages",
        "failure_policy",
        "qualification_claim",
    }
    if not isinstance(raw, dict) or set(raw) != expected or raw["schema_version"] != 2:
        raise ValueError("curriculum method fields or schema do not match")
    if raw["seeds"] != [42, 43, 44] or raw["qualification_claim"] is not False:
        raise ValueError("curriculum seeds or qualification claim drifted")
    root = path.resolve().parent
    bindings: dict[str, Path] = {}
    for name in ("development_scenarios", "development_criteria"):
        binding = raw[name]
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"{name} binding fields do not match")
        bindings[name] = _bound_path(root, binding["path"], binding["sha256"], name)
    raw_stages = raw["stages"]
    if not isinstance(raw_stages, list):
        raise ValueError("curriculum stages must be a list")
    stages: list[CurriculumStage] = []
    for item in raw_stages:
        fields = {"stage", "profile", "profile_sha256", "updates", "gate"}
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("curriculum stage fields do not match")
        profile = _bound_path(root, item["profile"], item["profile_sha256"], item["stage"])
        loaded = load_walking_v2_curriculum_profile(profile)
        if loaded.stage != item["stage"]:
            raise ValueError("curriculum stage and profile identity differ")
        if type(item["updates"]) is not int or item["updates"] < 0:
            raise ValueError("curriculum stage updates must be a non-negative integer")
        stages.append(
            CurriculumStage(
                item["stage"], profile, item["profile_sha256"], item["updates"], item["gate"]
            )
        )
    expected_stages = tuple((*_SPEED_FRONTIERS, "transitions", "robustness"))
    if tuple(stage.stage for stage in stages) != expected_stages:
        raise ValueError("curriculum stage order drifted")
    counts = (raw["num_envs"], raw["rollout_steps"], raw["save_interval_updates"])
    if counts != (256, 24, 100):
        raise ValueError("curriculum parallelism, rollout, or checkpoint cadence drifted")
    return CurriculumMethod(
        name=raw["name"],
        seeds=tuple(raw["seeds"]),
        num_envs=raw["num_envs"],
        rollout_steps=raw["rollout_steps"],
        save_interval_updates=raw["save_interval_updates"],
        fixed_speed_evaluation_horizon_s=float(raw["fixed_speed_evaluation_horizon_s"]),
        development_scenarios=bindings["development_scenarios"],
        development_criteria=bindings["development_criteria"],
        stages=tuple(stages),
    )


def assess_fixed_speed_stage(
    stage: str, rows: list[dict[str, Any]], *, horizon_s: float
) -> dict[str, Any]:
    """Apply the frozen safety, tracking, and gait gate to one speed frontier."""
    try:
        speeds = _SPEED_FRONTIERS[stage]
    except KeyError as error:
        raise ValueError(f"{stage!r} is not a fixed-speed curriculum stage") from error
    actual = tuple(float(row["requested_speed_m_s"]) for row in rows)
    if actual != speeds:
        raise ValueError(f"{stage} speed frontier must be {list(speeds)}, got {list(actual)}")
    if not math.isfinite(horizon_s) or horizon_s < 10.0:
        raise ValueError("fixed-speed promotion horizon must be at least 10 seconds")
    checks: list[dict[str, Any]] = []
    for row in rows:
        speed = float(row["requested_speed_m_s"])
        limits = {
            "finite": all(
                not isinstance(value, float) or math.isfinite(value) for value in row.values()
            ),
            "functional": bool(row["development_functional_passed"]),
            "not_terminated": not bool(row["terminated"]),
            "full_horizon": float(row["survived_seconds"]) + 1e-9 >= horizon_s,
            "command_tracking": float(row["settled_forward_command_rms_m_s"])
            <= (0.05 if speed == 0.0 else 0.25),
            "pelvis_height": float(row["minimum_pelvis_height_m"]) >= 0.62,
            "torso_tilt": float(row["maximum_torso_tilt_rad"]) <= 0.45,
            "torque_p95": float(row["torque_ratio_p95"]) <= 0.80,
            "torque_peak": float(row["torque_ratio_peak"]) <= 1.00,
        }
        if speed > 0.0:
            limits.update(
                {
                    "cadence": 0.6 <= float(row["cadence_steps_s"]) <= 3.0,
                    "alternation": float(row["alternation_fraction"]) >= 0.80,
                    "contact_chatter": float(row["contact_transition_rate_s"]) <= 6.0,
                    "minimum_steps": int(row["touchdown_count"]) >= 6,
                    "forward_progress": float(row["forward_progress_m"])
                    >= 0.25 * speed * horizon_s,
                }
            )
        checks.append({"speed_m_s": speed, "passed": all(limits.values()), "checks": limits})
    return {
        "schema_version": 1,
        "stage": stage,
        "required_speeds_m_s": list(speeds),
        "horizon_s": horizon_s,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "qualification_claim": False,
    }


def assess_transition_stage(stage: str, summary: dict[str, Any]) -> dict[str, Any]:
    """Require all four frozen development trials to pass functional and style gates."""
    if stage not in {"transitions", "robustness"}:
        raise ValueError("transition gate only applies to transitions or robustness")
    trials = summary.get("trials")
    if not isinstance(trials, list) or len(trials) != 4:
        raise ValueError("transition promotion requires exactly four frozen trials")
    checks = [
        {
            "trial_id": trial.get("trial_id", index),
            "functional_passed": bool(trial.get("functional_passed")),
            "style_passed": bool(trial.get("style_passed")),
            "sufficient_evidence": not bool(trial.get("insufficient_evidence", True)),
        }
        for index, trial in enumerate(trials)
    ]
    return {
        "schema_version": 1,
        "stage": stage,
        "passed": all(
            all(value for key, value in check.items() if key != "trial_id") for check in checks
        ),
        "checks": checks,
        "qualification_claim": False,
    }
