"""Simulator-independent deterministic walking playback schedule state."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .gait import CommandSchedule


@dataclass(frozen=True, slots=True)
class PlaybackFrame:
    step: int
    requested_forward_speed_m_s: float
    schedule_complete: bool


class WalkingScheduleCursor:
    """Map policy steps to a frozen schedule, then hold standing."""

    def __init__(self, schedule: CommandSchedule, control_dt: float) -> None:
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        boundaries: list[tuple[int, float]] = []
        end = 0
        for segment in schedule.segments:
            count = round(segment.duration_s / control_dt)
            if count <= 0 or not math.isclose(count * control_dt, segment.duration_s, abs_tol=1e-9):
                raise ValueError("schedule segments must align to control_dt")
            end += count
            boundaries.append((end, segment.forward_speed_m_s))
        self.boundaries = tuple(boundaries)
        self.horizon_steps = end
        self.control_dt = control_dt
        self.step = 0

    @property
    def complete(self) -> bool:
        return self.step >= self.horizon_steps

    def next(self) -> PlaybackFrame:
        current = self.step
        speed = next(
            (speed for boundary, speed in self.boundaries if current < boundary),
            0.0,
        )
        self.step += 1
        return PlaybackFrame(current, speed, self.complete)

    def reset(self) -> None:
        self.step = 0
