"""Simulator-independent contact hysteresis and gait-event state."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ContactProfile:
    """Force hysteresis and event-validity thresholds in SI units."""

    enter_force_n: float = 15.0
    exit_force_n: float = 8.0
    confirmation_duration_s: float = 0.06
    minimum_swing_duration_s: float = 0.12
    minimum_swing_clearance_m: float = 0.025

    def __post_init__(self) -> None:
        values = vars(self)
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("contact profile values must be finite and nonnegative")
        if self.enter_force_n <= self.exit_force_n:
            raise ValueError("enter force must be greater than exit force")
        if self.confirmation_duration_s <= 0:
            raise ValueError("confirmation duration must be positive")


@dataclass(frozen=True)
class ContactEvent:
    """One confirmed transition for one environment and one foot."""

    environment: int
    foot: int
    kind: str
    first_crossing_time_s: float
    confirmation_time_s: float
    prior_stable_duration_s: float
    swing_peak_clearance_m: float | None
    valid: bool


@dataclass(frozen=True)
class ContactUpdate:
    """Snapshot returned after a state-machine update."""

    raw_contact: npt.NDArray[np.bool_]
    stable_contact: npt.NDArray[np.bool_]
    stance_age_s: npt.NDArray[np.float64]
    swing_age_s: npt.NDArray[np.float64]
    raw_transition: npt.NDArray[np.bool_]
    events: tuple[ContactEvent, ...]


class ContactStateMachine:
    """Vectorized, reset-safe two-foot contact debouncer.

    The state lives in NumPy for trace analysis and ground-truth tests. The
    simulator adapter owns an equivalent Torch state for batched GPU use.
    """

    def __init__(self, num_environments: int, profile: ContactProfile) -> None:
        if num_environments <= 0:
            raise ValueError("num_environments must be positive")
        self.profile = profile
        self.num_environments = num_environments
        shape = (num_environments, 2)
        self.stable_contact = np.zeros(shape, dtype=bool)
        self.raw_contact = np.zeros(shape, dtype=bool)
        self.candidate_state = np.zeros(shape, dtype=bool)
        self.candidate_active = np.zeros(shape, dtype=bool)
        self.candidate_age_s = np.zeros(shape, dtype=float)
        self.candidate_start_time_s = np.zeros(shape, dtype=float)
        self.stable_age_s = np.zeros(shape, dtype=float)
        self.swing_peak_clearance_m = np.zeros(shape, dtype=float)
        self.elapsed_time_s = np.zeros(num_environments, dtype=float)
        self.initialized = np.zeros(shape, dtype=bool)

    def _ids(self, env_ids: npt.ArrayLike | None) -> npt.NDArray[np.int64]:
        if env_ids is None:
            return np.arange(self.num_environments, dtype=np.int64)
        result = np.asarray(env_ids, dtype=np.int64)
        if result.ndim != 1 or np.any(result < 0) or np.any(result >= self.num_environments):
            raise ValueError("env_ids must be a valid one-dimensional index")
        if len(np.unique(result)) != len(result):
            raise ValueError("env_ids must not contain duplicates")
        return result

    @staticmethod
    def _forces(value: npt.ArrayLike, rows: int) -> npt.NDArray[np.float64]:
        result = np.asarray(value, dtype=float)
        if result.shape != (rows, 2):
            raise ValueError(f"normal force must have shape ({rows}, 2)")
        if not np.isfinite(result).all():
            raise ValueError("normal force must be finite")
        return result

    def reset(self, normal_force_n: npt.ArrayLike, env_ids: npt.ArrayLike | None = None) -> None:
        """Initialize selected rows from measurement without emitting events."""
        ids = self._ids(env_ids)
        force = self._forces(normal_force_n, len(ids))
        measured = force >= self.profile.enter_force_n
        self.stable_contact[ids] = measured
        self.raw_contact[ids] = force > 0
        self.candidate_state[ids] = measured
        self.candidate_active[ids] = False
        self.candidate_age_s[ids] = 0
        self.candidate_start_time_s[ids] = 0
        self.stable_age_s[ids] = 0
        self.swing_peak_clearance_m[ids] = 0
        self.elapsed_time_s[ids] = 0
        self.initialized[ids] = True

    def update(
        self,
        normal_force_n: npt.ArrayLike,
        dt: float,
        *,
        sole_clearance_m: npt.ArrayLike | None = None,
    ) -> ContactUpdate:
        """Advance all environments once and emit confirmed transitions."""
        force = self._forces(normal_force_n, self.num_environments)
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        if not self.initialized.all():
            raise RuntimeError("all contact rows must be reset before update")
        clearance = (
            np.zeros_like(force)
            if sole_clearance_m is None
            else self._forces(sole_clearance_m, self.num_environments)
        )
        if dt == 0:
            return self._snapshot(np.zeros_like(self.raw_contact), ())

        new_raw = force > 0
        raw_transition = new_raw != self.raw_contact
        self.raw_contact[:] = new_raw
        desired = np.where(
            self.stable_contact,
            force >= self.profile.exit_force_n,
            force >= self.profile.enter_force_n,
        )
        self.stable_age_s += dt
        airborne = ~self.stable_contact
        self.swing_peak_clearance_m[airborne] = np.maximum(
            self.swing_peak_clearance_m[airborne], clearance[airborne]
        )

        differs = desired != self.stable_contact
        new_candidate = differs & (~self.candidate_active | (self.candidate_state != desired))
        self.candidate_state[new_candidate] = desired[new_candidate]
        self.candidate_active[new_candidate] = True
        self.candidate_age_s[new_candidate] = dt
        start_grid = np.broadcast_to((self.elapsed_time_s + dt)[:, None], force.shape)
        self.candidate_start_time_s[new_candidate] = start_grid[new_candidate]

        continuing = differs & ~new_candidate
        self.candidate_age_s[continuing] += dt
        cancelled = ~differs
        self.candidate_active[cancelled] = False
        self.candidate_age_s[cancelled] = 0

        events: list[ContactEvent] = []
        confirmed = self.candidate_active & (
            self.candidate_age_s + 1e-12 >= self.profile.confirmation_duration_s
        )
        for environment, foot in np.argwhere(confirmed):
            was_contact = bool(self.stable_contact[environment, foot])
            prior_duration = float(self.stable_age_s[environment, foot])
            peak = float(self.swing_peak_clearance_m[environment, foot])
            kind = "liftoff" if was_contact else "touchdown"
            valid = kind == "liftoff" or (
                prior_duration >= self.profile.minimum_swing_duration_s
                and peak >= self.profile.minimum_swing_clearance_m
            )
            events.append(
                ContactEvent(
                    environment=int(environment),
                    foot=int(foot),
                    kind=kind,
                    first_crossing_time_s=float(self.candidate_start_time_s[environment, foot]),
                    confirmation_time_s=float(self.elapsed_time_s[environment] + dt),
                    prior_stable_duration_s=prior_duration,
                    swing_peak_clearance_m=peak if kind == "touchdown" else None,
                    valid=valid,
                )
            )
        self.stable_contact[confirmed] = self.candidate_state[confirmed]
        self.stable_age_s[confirmed] = 0
        touchdown = confirmed & self.stable_contact
        self.swing_peak_clearance_m[touchdown] = 0
        self.candidate_active[confirmed] = False
        self.candidate_age_s[confirmed] = 0
        self.elapsed_time_s += dt
        return self._snapshot(raw_transition, tuple(events))

    def _snapshot(
        self,
        raw_transition: npt.NDArray[np.bool_],
        events: tuple[ContactEvent, ...],
    ) -> ContactUpdate:
        stable = self.stable_contact.copy()
        return ContactUpdate(
            raw_contact=self.raw_contact.copy(),
            stable_contact=stable,
            stance_age_s=np.where(stable, self.stable_age_s, 0.0),
            swing_age_s=np.where(stable, 0.0, self.stable_age_s),
            raw_transition=raw_transition.copy(),
            events=events,
        )
