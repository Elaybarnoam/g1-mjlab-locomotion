"""Strict, shared walking scenario definitions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _fields(raw: dict[str, Any], expected: set[str], name: str) -> None:
    if set(raw) != expected:
        raise ValueError(f"{name} fields do not match schema")


@dataclass(frozen=True)
class ScenarioSegment:
    duration_s: float
    command: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("scenario segment duration must be finite and positive")
        if len(self.command) != 3 or any(not math.isfinite(value) for value in self.command):
            raise ValueError("scenario command must contain three finite values")
        if self.command[0] < 0 or self.command[1:] != (0.0, 0.0):
            raise ValueError("walking-v1 scenarios are forward-only")


@dataclass(frozen=True)
class WalkingScenario:
    name: str
    seed: int
    initialization: str
    segments: tuple[ScenarioSegment, ...]

    def __post_init__(self) -> None:
        if not self.name or self.seed < 0 or not self.segments:
            raise ValueError("scenario name, seed, and segments are required")
        if self.initialization not in {"standing", "reference", "reference-fixed"}:
            raise ValueError("unsupported scenario initialization")

    @property
    def horizon_s(self) -> float:
        return sum(segment.duration_s for segment in self.segments)


@dataclass(frozen=True)
class ScenarioSet:
    schema_version: int
    name: str
    scenarios: tuple[WalkingScenario, ...]
    sha256: str


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def load_scenario_set(path: Path, *, control_dt: float) -> ScenarioSet:
    """Load and validate scenarios before simulator allocation."""
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be finite and positive")
    raw_value: Any = json.loads(path.read_text(encoding="utf-8"))
    raw = _object(raw_value, "scenario set")
    _fields(raw, {"schema_version", "name", "scenarios"}, "scenario set")
    if raw["schema_version"] != 2 or not isinstance(raw["name"], str) or not raw["name"]:
        raise ValueError("unsupported scenario-set schema or name")
    raw_scenarios = raw["scenarios"]
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("scenario set must contain scenarios")
    scenarios: list[WalkingScenario] = []
    names: set[str] = set()
    for scenario_value in raw_scenarios:
        scenario_raw = _object(scenario_value, "scenario")
        _fields(scenario_raw, {"name", "seed", "initialization", "segments"}, "scenario")
        segment_values = scenario_raw["segments"]
        if not isinstance(segment_values, list) or not segment_values:
            raise ValueError("scenario segments must be a nonempty list")
        segments: list[ScenarioSegment] = []
        for segment_value in segment_values:
            segment_raw = _object(segment_value, "segment")
            _fields(segment_raw, {"duration_s", "command"}, "segment")
            command = segment_raw["command"]
            if not isinstance(command, list) or len(command) != 3:
                raise ValueError("scenario command must be a three-item array")
            segment = ScenarioSegment(
                duration_s=float(segment_raw["duration_s"]),
                command=(float(command[0]), float(command[1]), float(command[2])),
            )
            steps = round(segment.duration_s / control_dt)
            if not math.isclose(steps * control_dt, segment.duration_s, abs_tol=1e-9):
                raise ValueError("scenario durations must align to control_dt")
            segments.append(segment)
        scenario = WalkingScenario(
            name=str(scenario_raw["name"]),
            seed=int(scenario_raw["seed"]),
            initialization=str(scenario_raw["initialization"]),
            segments=tuple(segments),
        )
        if scenario.name in names:
            raise ValueError("scenario names must be unique")
        names.add(scenario.name)
        scenarios.append(scenario)
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return ScenarioSet(2, raw["name"], tuple(scenarios), hashlib.sha256(encoded).hexdigest())
