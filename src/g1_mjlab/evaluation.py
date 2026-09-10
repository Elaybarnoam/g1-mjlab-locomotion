"""Simulator-independent first-episode accounting and standing acceptance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StandingCriteria:
    """Versioned candidate development limits, not hardware safety limits."""

    max_drift_m: float = 0.20
    max_torso_tilt_deg: float = 15.0
    min_height_m: float = 0.60
    settling_seconds: float = 1.0
    max_unsupported_seconds: float = 0.10

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_torso_tilt_deg > 180:
            raise ValueError("tilt cannot exceed 180 degrees")


@dataclass
class TrialAccumulator:
    """Consume pre-reset frames; completed trials are immutable to later frames."""

    horizon_steps: int
    dt: float
    criteria: StandingCriteria
    steps: int = 0
    finished: bool = False
    terminated: bool = False
    truncated: bool = False
    max_drift_m: float = 0.0
    max_torso_tilt_deg: float = 0.0
    min_height_m: float | None = None
    unsupported_seconds: float = 0.0
    max_unsupported_seconds: float = 0.0
    violations: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.horizon_steps <= 0 or not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("positive horizon and finite positive dt required")

    def observe(
        self,
        *,
        terminated: bool,
        truncated: bool,
        drift_m: float,
        torso_tilt_deg: float,
        height_m: float,
        supported: bool,
        finite: bool,
    ) -> None:
        if self.finished:
            return
        self.steps += 1
        finite = finite and all(math.isfinite(x) for x in (drift_m, torso_tilt_deg, height_m))
        if not finite:
            self.violations.add("nonfinite_state")
        else:
            self.max_drift_m = max(self.max_drift_m, drift_m)
            if drift_m > self.criteria.max_drift_m:
                self.violations.add("drift")
            if self.steps * self.dt > self.criteria.settling_seconds:
                self.max_torso_tilt_deg = max(self.max_torso_tilt_deg, torso_tilt_deg)
                self.min_height_m = (
                    height_m if self.min_height_m is None else min(self.min_height_m, height_m)
                )
                self.unsupported_seconds = 0.0 if supported else self.unsupported_seconds + self.dt
                self.max_unsupported_seconds = max(
                    self.max_unsupported_seconds, self.unsupported_seconds
                )
                if torso_tilt_deg > self.criteria.max_torso_tilt_deg:
                    self.violations.add("torso_tilt")
                if height_m < self.criteria.min_height_m:
                    self.violations.add("height")
                if self.unsupported_seconds > self.criteria.max_unsupported_seconds + 1e-9:
                    self.violations.add("unsupported")
        self.terminated = terminated
        self.truncated = truncated
        if terminated:
            self.violations.add("environment_termination")
        if truncated and self.steps < self.horizon_steps:
            self.violations.add("early_timeout")
        self.finished = terminated or truncated or not finite or self.steps >= self.horizon_steps

    @property
    def survival_passed(self) -> bool:
        return (
            self.steps >= self.horizon_steps
            and not self.terminated
            and "nonfinite_state" not in self.violations
        )

    @property
    def passed(self) -> bool:
        return self.survival_passed and not self.violations and self.min_height_m is not None


def wilson_interval(passed: int, total: int) -> tuple[float, float]:
    """Two-sided 95% Wilson interval for independent Bernoulli trials."""
    if total <= 0 or not 0 <= passed <= total:
        raise ValueError("require 0 <= passed <= total and total > 0")
    z = 1.959963984540054
    p = passed / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)
