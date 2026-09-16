"""Simulator-independent contact hysteresis and gait-event state."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

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
class Stage19ContactProfile:
    """Frozen mechanics shared by every Stage 19 ablation arm."""

    schema_version: int
    enter_force_n: float
    exit_force_n: float
    confirmation_duration_s: float
    minimum_stance_duration_s: float
    minimum_swing_duration_s: float
    minimum_swing_clearance_m: float
    phase_tolerance_cycle: float
    transition_grace_s: float
    bilateral_flight_duration_s: float
    slip_scale_m_s: float
    swing_clearance_reference_m: float
    clearance_scale_m: float
    placement_scale_m: float
    step_reference_m: float
    minimum_root_progress_fraction: float
    minimum_step_reference_m: float
    squared_error_clip: float
    contact_slots: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.contact_slots not in {1, 2, 3, 4}:
            raise ValueError("invalid Stage 19 contact profile identity")
        numeric = [
            value
            for name, value in vars(self).items()
            if name not in {"schema_version", "contact_slots"}
        ]
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise ValueError("Stage 19 contact parameters must be finite and nonnegative")
        ContactProfile(
            self.enter_force_n,
            self.exit_force_n,
            self.confirmation_duration_s,
            self.minimum_swing_duration_s,
            self.minimum_swing_clearance_m,
        )
        if not 0 <= self.phase_tolerance_cycle <= 0.5:
            raise ValueError("phase tolerance must be in [0, 0.5]")
        for name in (
            "minimum_stance_duration_s",
            "minimum_swing_duration_s",
            "slip_scale_m_s",
            "clearance_scale_m",
            "placement_scale_m",
            "step_reference_m",
            "squared_error_clip",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def runtime_parameters(self) -> dict[str, float]:
        return {
            name: float(value) for name, value in vars(self).items() if name != "schema_version"
        }


def load_stage19_contact_profile(path: Path) -> Stage19ContactProfile:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(Stage19ContactProfile.__dataclass_fields__):
        raise ValueError("Stage 19 contact profile fields do not match schema")
    return Stage19ContactProfile(**raw)


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
    event_confirmation_ids: npt.NDArray[np.int64]
    last_valid_touchdown_time_s: npt.NDArray[np.float64]
    last_valid_touchdown_position_m: npt.NDArray[np.float64]
    last_opposite_touchdown_time_s: npt.NDArray[np.float64]
    swing_peak_clearance_m: npt.NDArray[np.float64]
    previous_expected_contact: npt.NDArray[np.bool_]
    expected_transition_phase: npt.NDArray[np.float64]
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
        self.event_confirmation_ids = np.zeros(shape, dtype=np.int64)
        self.last_valid_touchdown_time_s = np.zeros(shape)
        self.last_valid_touchdown_position_m = np.zeros((*shape, 3))
        self.last_opposite_touchdown_time_s = np.zeros(shape)
        self.previous_expected_contact = np.zeros(shape, dtype=bool)
        self.expected_transition_phase = np.zeros(shape)

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
        self.event_confirmation_ids[ids] = 0
        self.last_valid_touchdown_time_s[ids] = 0
        self.last_valid_touchdown_position_m[ids] = 0
        self.last_opposite_touchdown_time_s[ids] = 0
        self.previous_expected_contact[ids] = measured
        self.expected_transition_phase[ids] = 0

    def update(
        self,
        normal_force_n: npt.ArrayLike,
        dt: float | npt.ArrayLike,
        *,
        sole_clearance_m: npt.ArrayLike | None = None,
        sole_position_m: npt.ArrayLike | None = None,
        expected_contact: npt.ArrayLike | None = None,
        phase: npt.ArrayLike | None = None,
    ) -> ContactUpdate:
        """Advance all environments once and emit confirmed transitions."""
        force = self._forces(normal_force_n, self.num_environments)
        dt_array = np.asarray(dt, dtype=float)
        if dt_array.ndim == 0:
            dt_array = np.full(self.num_environments, float(dt_array))
        if (
            dt_array.shape != (self.num_environments,)
            or np.any(~np.isfinite(dt_array))
            or np.any(dt_array < 0)
        ):
            raise ValueError("dt must be finite, nonnegative, and scalar or per-environment")
        dt_grid = dt_array[:, None]
        dt_matrix = np.broadcast_to(dt_grid, force.shape)
        if not self.initialized.all():
            raise RuntimeError("all contact rows must be reset before update")
        clearance = (
            np.zeros_like(force)
            if sole_clearance_m is None
            else self._forces(sole_clearance_m, self.num_environments)
        )
        position = (
            np.zeros((self.num_environments, 2, 3))
            if sole_position_m is None
            else np.asarray(sole_position_m, dtype=float)
        )
        if position.shape != (self.num_environments, 2, 3):
            raise ValueError("sole position must have shape (num_environments, 2, 3)")
        expected = (
            self.previous_expected_contact
            if expected_contact is None
            else np.asarray(expected_contact, dtype=bool)
        )
        phase_array = (
            np.zeros(self.num_environments) if phase is None else np.asarray(phase, dtype=float)
        )
        if expected.shape != self.stable_contact.shape or phase_array.shape != (
            self.num_environments,
        ):
            raise ValueError("expected contact or phase has invalid shape")
        expected_changed = (expected != self.previous_expected_contact) & (dt_array[:, None] > 0)
        self.expected_transition_phase[expected_changed] = np.broadcast_to(
            phase_array[:, None], expected.shape
        )[expected_changed]
        self.previous_expected_contact[:] = np.where(
            dt_array[:, None] > 0, expected, self.previous_expected_contact
        )
        if np.all(dt_array == 0):
            return self._snapshot(np.zeros_like(self.raw_contact), ())

        advancing = dt_matrix > 0
        new_raw = np.where(advancing, force > 0, self.raw_contact)
        raw_transition = new_raw != self.raw_contact
        self.raw_contact[:] = new_raw
        desired = np.where(
            self.stable_contact,
            force >= self.profile.exit_force_n,
            force >= self.profile.enter_force_n,
        )
        self.stable_age_s += dt_grid
        airborne = ~self.stable_contact
        self.swing_peak_clearance_m[airborne] = np.maximum(
            self.swing_peak_clearance_m[airborne], clearance[airborne]
        )

        differs = (desired != self.stable_contact) & advancing
        new_candidate = differs & (~self.candidate_active | (self.candidate_state != desired))
        self.candidate_state[new_candidate] = desired[new_candidate]
        self.candidate_active[new_candidate] = True
        self.candidate_age_s[new_candidate] = dt_matrix[new_candidate]
        start_grid = np.broadcast_to((self.elapsed_time_s + dt_array)[:, None], force.shape)
        self.candidate_start_time_s[new_candidate] = start_grid[new_candidate]

        continuing = differs & ~new_candidate
        self.candidate_age_s[continuing] += dt_matrix[continuing]
        cancelled = ~differs
        self.candidate_active[cancelled] = False
        self.candidate_age_s[cancelled] = 0

        events: list[ContactEvent] = []
        confirmed = (
            self.candidate_active
            & (self.candidate_age_s + 1e-12 >= self.profile.confirmation_duration_s)
            & advancing
        )
        event_swing_peak = self.swing_peak_clearance_m.copy()
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
                    confirmation_time_s=float(
                        self.elapsed_time_s[environment] + dt_array[environment]
                    ),
                    prior_stable_duration_s=prior_duration,
                    swing_peak_clearance_m=peak if kind == "touchdown" else None,
                    valid=valid,
                )
            )
            self.event_confirmation_ids[environment, foot] += 1
            if kind == "touchdown" and valid:
                opposite = 1 - int(foot)
                self.last_opposite_touchdown_time_s[environment, foot] = (
                    self.last_valid_touchdown_time_s[environment, opposite]
                )
                self.last_valid_touchdown_time_s[environment, foot] = (
                    self.elapsed_time_s[environment] + dt_array[environment]
                )
                self.last_valid_touchdown_position_m[environment, foot] = position[
                    environment, foot
                ]
        self.stable_contact[confirmed] = self.candidate_state[confirmed]
        self.stable_age_s[confirmed] = 0
        touchdown = confirmed & self.stable_contact
        self.swing_peak_clearance_m[touchdown] = 0
        self.candidate_active[confirmed] = False
        self.candidate_age_s[confirmed] = 0
        self.elapsed_time_s += dt_array
        return self._snapshot(raw_transition, tuple(events), event_swing_peak)

    def _snapshot(
        self,
        raw_transition: npt.NDArray[np.bool_],
        events: tuple[ContactEvent, ...],
        event_swing_peak: npt.NDArray[np.float64] | None = None,
    ) -> ContactUpdate:
        stable = self.stable_contact.copy()
        return ContactUpdate(
            raw_contact=self.raw_contact.copy(),
            stable_contact=stable,
            stance_age_s=np.where(stable, self.stable_age_s, 0.0),
            swing_age_s=np.where(stable, 0.0, self.stable_age_s),
            raw_transition=raw_transition.copy(),
            event_confirmation_ids=self.event_confirmation_ids.copy(),
            last_valid_touchdown_time_s=self.last_valid_touchdown_time_s.copy(),
            last_valid_touchdown_position_m=self.last_valid_touchdown_position_m.copy(),
            last_opposite_touchdown_time_s=self.last_opposite_touchdown_time_s.copy(),
            swing_peak_clearance_m=(
                self.swing_peak_clearance_m.copy() if event_swing_peak is None else event_swing_peak
            ),
            previous_expected_contact=self.previous_expected_contact.copy(),
            expected_transition_phase=self.expected_transition_phase.copy(),
            events=events,
        )
