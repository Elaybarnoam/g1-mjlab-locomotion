"""Validated, immutable configuration for mjlab standing experiments."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

STAGE19_REWARD_NAMES = frozenset(
    {
        "track_linear_velocity",
        "track_angular_velocity",
        "upright",
        "pose",
        "body_ang_vel",
        "angular_momentum",
        "dof_pos_limits",
        "action_rate_l2",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
        "self_collisions",
        "commanded_forward_progress",
        "reference_joint_pose",
        "reference_joint_velocity",
        "reference_foot_position",
        "reference_contact_timing",
        "crouch",
        "effort",
        "termination",
        "phase_contact_error",
        "extra_contact_event",
        "short_stance",
        "short_swing",
        "physical_stance_slip",
        "swing_clearance_error",
        "touchdown_placement",
        "bilateral_flight",
    }
)

STAGE19_EVENT_REWARDS = frozenset(
    {"extra_contact_event", "short_stance", "short_swing", "touchdown_placement", "termination"}
)

STAGE19_REWARD_PARAMETERS: dict[str, frozenset[str]] = {
    name: frozenset() for name in STAGE19_REWARD_NAMES
}
STAGE19_REWARD_PARAMETERS.update(
    {
        "track_linear_velocity": frozenset({"std"}),
        "track_angular_velocity": frozenset({"std"}),
        "upright": frozenset({"std"}),
        "phase_contact_error": frozenset({"walk_threshold_m_s"}),
        "extra_contact_event": frozenset({"phase_tolerance_cycle", "transition_grace_s"}),
        "short_stance": frozenset({"minimum_duration_s", "transition_grace_s"}),
        "short_swing": frozenset({"minimum_duration_s", "transition_grace_s"}),
        "physical_stance_slip": frozenset({"slip_scale_m_s", "squared_error_clip"}),
        "swing_clearance_error": frozenset(
            {"reference_clearance_m", "clearance_scale_m", "squared_error_clip"}
        ),
        "touchdown_placement": frozenset(
            {
                "minimum_root_progress_fraction",
                "minimum_step_reference_m",
                "minimum_swing_clearance_m",
                "placement_scale_m",
                "step_reference_m",
            }
        ),
        "bilateral_flight": frozenset({"minimum_duration_s"}),
    }
)


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
    entropy_coef: float | None = None
    reset_action_std_on_transfer: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class WalkingTrainingProfile:
    """One explicit command, reset, and reward curriculum stage for walking-v1."""

    schema_version: int
    name: str
    standing_fraction: float
    reference_initialization: bool
    randomize_phase: bool
    objective: str = "reference_style"
    forward_speed_range_m_s: tuple[float, float] | None = None
    velocity_tracking_std_m_s: float | None = None
    forward_progress_weight: float | None = None
    reference_foot_position_std_m: float | None = None
    host_semantics_version: int = 2
    domain_randomization: bool | None = None
    observation_noise: bool | None = None
    reference_ground_offset_m: float = 0.03

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class Stage19RewardTerm:
    """One allowlisted reward with explicit units and integration semantics."""

    name: str
    enabled: bool
    weight: float
    integration_kind: str
    parameters: tuple[tuple[str, float], ...]
    physical_unit: str
    mask_id: str
    formula_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "weight": self.weight,
            "integration_kind": self.integration_kind,
            "parameters": dict(self.parameters),
            "physical_unit": self.physical_unit,
            "mask_id": self.mask_id,
            "formula_id": self.formula_id,
        }

    def manager_weight(self, control_dt: float) -> float:
        return self.weight / control_dt if self.integration_kind == "per_event" else self.weight

    @property
    def parameter_dict(self) -> dict[str, float]:
        return dict(self.parameters)


@dataclass(frozen=True)
class Stage19RewardProfile:
    """Complete reward resolution for one controlled Stage 19 arm."""

    schema_version: int
    name: str
    terms: tuple[Stage19RewardTerm, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "terms": [term.to_dict() for term in self.terms],
        }

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def by_name(self) -> dict[str, Stage19RewardTerm]:
        return {term.name: term for term in self.terms}


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
    required = expected - {"entropy_coef", "reset_action_std_on_transfer"}
    unknown = set(raw) - expected
    missing = required - set(raw)
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
    if profile.entropy_coef is not None and (
        not math.isfinite(profile.entropy_coef) or profile.entropy_coef < 0
    ):
        raise ValueError("entropy_coef must be finite and non-negative")
    if not isinstance(profile.reset_action_std_on_transfer, bool):
        raise ValueError("reset_action_std_on_transfer must be a boolean")
    return profile


def load_walking_training_profile(path: Path) -> WalkingTrainingProfile:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("walking profile root must be an object")
    expected = set(WalkingTrainingProfile.__dataclass_fields__)
    required = expected - {
        "objective",
        "forward_speed_range_m_s",
        "velocity_tracking_std_m_s",
        "forward_progress_weight",
        "reference_foot_position_std_m",
        "host_semantics_version",
        "domain_randomization",
        "observation_noise",
        "reference_ground_offset_m",
    }
    if not required.issubset(raw) or not set(raw).issubset(expected):
        raise ValueError("walking profile fields do not match schema")
    values = dict(raw)
    if "forward_speed_range_m_s" in values:
        speed_range = values["forward_speed_range_m_s"]
        if not isinstance(speed_range, list) or len(speed_range) != 2:
            raise ValueError("forward_speed_range_m_s must contain [minimum, maximum]")
        values["forward_speed_range_m_s"] = tuple(speed_range)
    profile = WalkingTrainingProfile(**values)
    if profile.schema_version != 1:
        raise ValueError("only walking profile schema_version 1 is supported")
    if not profile.name or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for char in profile.name
    ):
        raise ValueError("walking profile name must contain only letters, digits, '-' and '_'")
    if (
        isinstance(profile.standing_fraction, bool)
        or not isinstance(profile.standing_fraction, (int, float))
        or not math.isfinite(profile.standing_fraction)
        or not 0 <= profile.standing_fraction <= 1
    ):
        raise ValueError("standing_fraction must be finite in [0, 1]")
    if not isinstance(profile.reference_initialization, bool) or not isinstance(
        profile.randomize_phase, bool
    ):
        raise ValueError("walking reset flags must be boolean")
    if profile.host_semantics_version not in {1, 2}:
        raise ValueError("host_semantics_version must be 1 or 2")
    if any(
        value is not None and not isinstance(value, bool)
        for value in (profile.domain_randomization, profile.observation_noise)
    ):
        raise ValueError("domain_randomization and observation_noise must be booleans or null")
    if (
        not math.isfinite(profile.reference_ground_offset_m)
        or profile.reference_ground_offset_m < 0
        or profile.reference_ground_offset_m > 0.05
    ):
        raise ValueError("reference_ground_offset_m must be finite in [0, 0.05]")
    if profile.objective not in {"locomotion_bootstrap", "reference_style"}:
        raise ValueError("walking objective must be 'locomotion_bootstrap' or 'reference_style'")
    if profile.forward_speed_range_m_s is not None:
        low, high = profile.forward_speed_range_m_s
        if (
            isinstance(low, bool)
            or isinstance(high, bool)
            or not all(isinstance(value, (int, float)) for value in (low, high))
            or not all(math.isfinite(value) for value in (low, high))
            or low <= 0
            or high < low
        ):
            raise ValueError("forward_speed_range_m_s must contain finite positive ordered values")
    if profile.velocity_tracking_std_m_s is not None and (
        not math.isfinite(profile.velocity_tracking_std_m_s)
        or profile.velocity_tracking_std_m_s <= 0
    ):
        raise ValueError("velocity_tracking_std_m_s must be positive and finite")
    if profile.forward_progress_weight is not None and (
        not math.isfinite(profile.forward_progress_weight) or profile.forward_progress_weight < 0
    ):
        raise ValueError("forward_progress_weight must be finite and non-negative")
    if profile.reference_foot_position_std_m is not None and (
        not math.isfinite(profile.reference_foot_position_std_m)
        or profile.reference_foot_position_std_m <= 0
    ):
        raise ValueError("reference_foot_position_std_m must be positive and finite")
    return profile


def load_stage19_reward_profile(path: Path) -> Stage19RewardProfile:
    """Load a complete, non-executable Stage 19 reward declaration."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "name", "terms"}:
        raise ValueError("stage19 reward profile fields do not match schema")
    if raw["schema_version"] != 1 or not isinstance(raw["name"], str) or not raw["name"]:
        raise ValueError("invalid stage19 reward profile identity")
    if not isinstance(raw["terms"], list):
        raise ValueError("stage19 reward terms must be a list")
    expected = set(Stage19RewardTerm.__dataclass_fields__)
    terms: list[Stage19RewardTerm] = []
    for value in raw["terms"]:
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("stage19 reward term fields do not match schema")
        if value["name"] not in STAGE19_REWARD_NAMES or value["formula_id"] != value["name"]:
            raise ValueError("stage19 reward term or formula is not allowlisted")
        if value["integration_kind"] not in {"rate", "per_event"}:
            raise ValueError("stage19 integration_kind must be rate or per_event")
        expected_kind = "per_event" if value["name"] in STAGE19_EVENT_REWARDS else "rate"
        if value["integration_kind"] != expected_kind:
            raise ValueError(f"{value['name']} must use {expected_kind} integration")
        parameters = value["parameters"]
        if not isinstance(parameters, dict) or any(
            not isinstance(key, str)
            or isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            for key, number in parameters.items()
        ):
            raise ValueError("stage19 parameters must be finite numeric values")
        expected_parameters = STAGE19_REWARD_PARAMETERS[value["name"]]
        if set(parameters) != expected_parameters:
            raise ValueError(
                f"{value['name']} parameters mismatch; "
                f"expected={sorted(expected_parameters)}, actual={sorted(parameters)}"
            )
        if any(number <= 0 for number in parameters.values()):
            raise ValueError("stage19 scales and thresholds must be positive")
        weight = value["weight"]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or not isinstance(value["enabled"], bool)
        ):
            raise ValueError("stage19 reward state and weight are invalid")
        if not value["enabled"] and weight != 0:
            raise ValueError("disabled stage19 rewards must have zero weight")
        if not value["physical_unit"] or not value["mask_id"]:
            raise ValueError("stage19 reward units and mask must be explicit")
        converted = dict(value)
        converted["parameters"] = tuple(
            sorted((key, float(number)) for key, number in parameters.items())
        )
        converted["weight"] = float(weight)
        terms.append(Stage19RewardTerm(**converted))
    names = [term.name for term in terms]
    if len(names) != len(set(names)) or set(names) != STAGE19_REWARD_NAMES:
        raise ValueError("stage19 profile must resolve every reward exactly once")
    return Stage19RewardProfile(1, raw["name"], tuple(terms))
