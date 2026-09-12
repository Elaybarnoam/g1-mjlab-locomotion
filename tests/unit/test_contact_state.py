from __future__ import annotations

import numpy as np
import pytest

from g1_mjlab.motion.contact import ContactProfile, ContactStateMachine


def test_force_hysteresis_rejects_short_blips_and_gaps() -> None:
    machine = ContactStateMachine(1, ContactProfile())
    machine.reset(np.array([[0.0, 20.0]]))

    for force in (20.0, 0.0, 20.0):
        update = machine.update(np.array([[force, 20.0]]), 0.02)
    assert not update.stable_contact[0, 0]
    assert update.stable_contact[0, 1]
    assert update.events == ()

    events = ()
    for _ in range(3):
        events += machine.update(np.array([[20.0, 0.0]]), 0.02).events
    assert machine.stable_contact.tolist() == [[True, False]]
    assert {event.kind for event in events} == {"touchdown", "liftoff"}


def test_valid_touchdown_requires_real_swing_and_clearance() -> None:
    machine = ContactStateMachine(1, ContactProfile())
    machine.reset(np.array([[20.0, 20.0]]))
    for index in range(9):
        machine.update(
            np.array([[0.0, 20.0]]),
            0.02,
            sole_clearance_m=np.array([[0.04 if index == 4 else 0.01, 0.0]]),
        )
    touchdown = ()
    for _ in range(3):
        touchdown = machine.update(np.array([[20.0, 20.0]]), 0.02).events
    event = next(event for event in touchdown if event.kind == "touchdown")
    assert event.valid
    assert event.prior_stable_duration_s >= 0.12
    assert event.swing_peak_clearance_m == pytest.approx(0.04)
    assert event.confirmation_time_s - event.first_crossing_time_s == pytest.approx(0.04)


def test_reset_is_indexed_and_dt_zero_is_a_noop() -> None:
    machine = ContactStateMachine(3, ContactProfile())
    machine.reset(np.full((3, 2), 20.0))
    before = machine.stable_contact.copy()
    machine.update(np.zeros((3, 2)), 0.0)
    np.testing.assert_array_equal(machine.stable_contact, before)

    machine.reset(np.zeros((1, 2)), env_ids=np.array([1]))
    assert machine.stable_contact.tolist() == [[True, True], [False, False], [True, True]]
    assert machine.elapsed_time_s.tolist() == [0.0, 0.0, 0.0]


def test_profile_and_input_validation_are_strict() -> None:
    with pytest.raises(ValueError, match="enter"):
        ContactProfile(enter_force_n=5.0, exit_force_n=8.0)
    machine = ContactStateMachine(1, ContactProfile())
    with pytest.raises(ValueError, match="shape"):
        machine.reset(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        machine.reset(np.array([[np.nan, 0.0]]))
