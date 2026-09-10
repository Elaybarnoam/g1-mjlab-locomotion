"""Validated, immutable configuration for mjlab standing experiments."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResolvedRunConfig:
    schema_version: int
    task_id: str
    run_name: str
    seed: int
    num_envs: int
    max_iterations: int
    rollout_steps: int
    save_interval: int
    physics_dt: float
    decimation: int
    episode_length_s: float
    action_clip: float | None
    device: str
    logger: str
    video: bool

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.decimation

    @property
    def transitions_per_update(self) -> int:
        return self.num_envs * self.rollout_steps

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["control_dt"] = self.control_dt
        data["transitions_per_update"] = self.transitions_per_update
        return data

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


@dataclass(frozen=True)
class StandingRewardProfile:
    """Reward additions whose units remain explicit across control rates."""

    schema_version: int
    name: str
    alive_reward_rate: float
    termination_penalty: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def termination_weight(self, control_dt: float) -> float:
        """Convert a per-event penalty to mjlab's dt-scaled reward weight."""
        return self.termination_penalty / control_dt


@dataclass(frozen=True)
class PpoProfile:
    """Small, explicit PPO overrides used by controlled experiments."""

    schema_version: int
    name: str
    initial_action_std: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


_FIELDS = set(ResolvedRunConfig.__dataclass_fields__)


def _validate(data: Mapping[str, Any]) -> ResolvedRunConfig:
    unknown = set(data) - _FIELDS
    missing = _FIELDS - set(data)
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing configuration fields: {sorted(missing)}")
    cfg = ResolvedRunConfig(**data)
    if cfg.schema_version != 1:
        raise ValueError("only configuration schema_version 1 is supported")
    for name in (
        "seed",
        "num_envs",
        "max_iterations",
        "rollout_steps",
        "save_interval",
        "decimation",
    ):
        value = getattr(cfg, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if cfg.seed < 0:
        raise ValueError("seed must be non-negative")
    for name in ("num_envs", "max_iterations", "rollout_steps", "save_interval", "decimation"):
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("physics_dt", "episode_length_s", "action_clip"):
        value = getattr(cfg, name)
        if name == "action_clip" and value is None:
            continue
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive finite number")
    if cfg.transitions_per_update % 4:
        raise ValueError("num_envs * rollout_steps must be divisible by four PPO minibatches")
    if cfg.device not in {"cuda:0", "cpu"}:
        raise ValueError("device must be 'cuda:0' or 'cpu'")
    if cfg.logger != "tensorboard":
        raise ValueError("standing-v1 requires local TensorBoard logging")
    if not cfg.task_id:
        raise ValueError("task_id cannot be empty")
    if not cfg.run_name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in cfg.run_name
    ):
        raise ValueError("run_name must contain only letters, digits, '-' and '_'")
    return cfg


def load_config(path: Path, overrides: Mapping[str, object] | None = None) -> ResolvedRunConfig:
    """Load strict JSON configuration and apply explicit field overrides."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be an object")
    merged: dict[str, Any] = dict(raw)
    if overrides:
        merged.update(overrides)
    return _validate(merged)


def load_reward_profile(path: Path) -> StandingRewardProfile:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("reward profile root must be an object")
    expected = set(StandingRewardProfile.__dataclass_fields__)
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown or missing:
        raise ValueError(
            f"reward profile fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    profile = StandingRewardProfile(**raw)
    if profile.schema_version != 1:
        raise ValueError("only reward profile schema_version 1 is supported")
    if not profile.name or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for char in profile.name
    ):
        raise ValueError("reward profile name must contain only letters, digits, '-' and '_'")
    if not math.isfinite(profile.alive_reward_rate) or profile.alive_reward_rate < 0:
        raise ValueError("alive_reward_rate must be finite and non-negative")
    if not math.isfinite(profile.termination_penalty) or profile.termination_penalty > 0:
        raise ValueError("termination_penalty must be finite and non-positive")
    return profile


def load_ppo_profile(path: Path) -> PpoProfile:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("PPO profile root must be an object")
    expected = set(PpoProfile.__dataclass_fields__)
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown or missing:
        raise ValueError(
            f"PPO profile fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    profile = PpoProfile(**raw)
    if profile.schema_version != 1:
        raise ValueError("only PPO profile schema_version 1 is supported")
    if not profile.name or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for char in profile.name
    ):
        raise ValueError("PPO profile name must contain only letters, digits, '-' and '_'")
    if not math.isfinite(profile.initial_action_std) or profile.initial_action_std <= 0:
        raise ValueError("initial_action_std must be positive and finite")
    return profile
