from __future__ import annotations

import pytest

from g1_mjlab.motion.gait import CommandSchedule, CommandSegment
from g1_mjlab.motion.playback import WalkingScheduleCursor


def schedule() -> CommandSchedule:
    return CommandSchedule(
        schema_version=1,
        name="test",
        reference_id="reference",
        segments=(CommandSegment(0.04, 0.0), CommandSegment(0.06, 0.6), CommandSegment(0.02, 0.0)),
    )


def test_cursor_runs_schedule_then_holds_standing_and_resets() -> None:
    cursor = WalkingScheduleCursor(schedule(), 0.02)
    frames = [cursor.next() for _ in range(8)]
    assert [frame.requested_forward_speed_m_s for frame in frames] == [
        0.0,
        0.0,
        0.6,
        0.6,
        0.6,
        0.0,
        0.0,
        0.0,
    ]
    assert frames[5].schedule_complete
    cursor.reset()
    assert cursor.next().step == 0


def test_cursor_rejects_schedule_not_aligned_to_control_period() -> None:
    broken = CommandSchedule(1, "broken", "reference", (CommandSegment(0.03, 0.0),))
    with pytest.raises(ValueError, match="align"):
        WalkingScheduleCursor(broken, 0.02)
