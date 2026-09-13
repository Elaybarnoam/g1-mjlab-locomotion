"""Frozen, hash-bound campaign declaration for walking-v2."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import sha256_file


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class WalkingV2Campaign:
    schema_version: int
    task_id: str
    objective: str
    method_id: str
    ppo: ArtifactBinding
    reward: ArtifactBinding
    reference_bank: ArtifactBinding
    training_seeds: tuple[int, ...]
    environment_candidates: tuple[int, ...]
    rollout_steps: int
    evaluation_interval_updates: int
    first_promotion_check_update: int
    maximum_acquisition_updates: int
    maximum_extension_updates: int
    aggregate_transition_budget: int
    curriculum_stages: tuple[str, ...]


def _binding(value: object, root: Path, name: str) -> ArtifactBinding:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} must contain exactly path and sha256")
    path_value, digest = value["path"], value["sha256"]
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise ValueError(f"{name} path and sha256 must be strings")
    path = (root / path_value).resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or sha256_file(path) != digest:
        raise ValueError(f"{name} SHA-256 mismatch")
    return ArtifactBinding(path, digest)


def load_walking_v2_campaign(path: Path) -> WalkingV2Campaign:
    """Load the complete Plan 06 campaign and reject drift before GPU execution."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = set(WalkingV2Campaign.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("walking-v2 campaign fields do not match schema")
    root = path.resolve().parent
    converted: dict[str, Any] = dict(raw)
    for name in ("ppo", "reward", "reference_bank"):
        converted[name] = _binding(raw[name], root, name)
    for name in ("training_seeds", "environment_candidates", "curriculum_stages"):
        if not isinstance(raw[name], list):
            raise ValueError(f"{name} must be a list")
        converted[name] = tuple(raw[name])
    campaign = WalkingV2Campaign(**converted)
    _validate_campaign(campaign)
    return campaign


def _validate_campaign(campaign: WalkingV2Campaign) -> None:
    if campaign.schema_version != 2 or campaign.task_id != "G1-Walking-Flat-v2":
        raise ValueError("campaign must use walking-v2 schema 2")
    if not campaign.objective.strip() or campaign.method_id != "soft-reference-policy-first-v1":
        raise ValueError("campaign identity is invalid")
    if campaign.training_seeds != (42, 43, 44):
        raise ValueError("development and replication seeds are frozen to 42, 43, and 44")
    if campaign.environment_candidates != (64, 128, 256):
        raise ValueError("resource candidates must be benchmarked in frozen order")
    counts = (
        campaign.rollout_steps,
        campaign.evaluation_interval_updates,
        campaign.first_promotion_check_update,
        campaign.maximum_acquisition_updates,
        campaign.aggregate_transition_budget,
    )
    if any(type(value) is not int or value <= 0 for value in counts):
        raise ValueError("campaign counts must be positive integers")
    if campaign.rollout_steps != 24 or campaign.evaluation_interval_updates != 100:
        raise ValueError("campaign rollout and evaluation cadence drifted")
    if campaign.first_promotion_check_update != 500:
        raise ValueError("first promotion check must remain update 500")
    if campaign.maximum_acquisition_updates != 4000 or campaign.maximum_extension_updates != 2000:
        raise ValueError("campaign acquisition budget drifted")
    expected_stages = ("stand", "stand-walk-040", "add-060", "add-080", "transitions", "robustness")
    if campaign.curriculum_stages != expected_stages:
        raise ValueError("campaign curriculum order drifted")
