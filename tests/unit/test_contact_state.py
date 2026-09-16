from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from g1_mjlab.motion.contact import (
    ContactProfile,
    ContactStateMachine,
    load_stage19_contact_profile,
)
from g1_mjlab.tasks.walking_contact import TorchContactState


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


def test_stage19_contact_profile_loads_strict_frozen_parameters() -> None:
    profile = load_stage19_contact_profile(Path("configs/walking-v1/stage19/contact-profile.json"))
    assert profile.contact_slots == 4
    assert profile.minimum_stance_duration_s == pytest.approx(0.2)
    assert profile.runtime_parameters()["step_reference_m"] == pytest.approx(0.6064067445483481)


def test_stage19_contact_profile_rejects_unknown_fields(tmp_path: Path) -> None:
    source = Path("configs/walking-v1/stage19/contact-profile.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["unreviewed_threshold"] = 1.0
    path = tmp_path / "contact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        load_stage19_contact_profile(path)


@pytest.mark.parametrize("control_dt", [0.005, 0.02, 0.04])
def test_numpy_and_torch_contact_state_are_equivalent(control_dt: float) -> None:
    torch = pytest.importorskip("torch")
    numpy_state = ContactStateMachine(3, ContactProfile())
    torch_state = TorchContactState(3, ContactProfile(), "cpu")
    initial = np.array([[20.0, 20.0], [0.0, 20.0], [20.0, 0.0]])
    numpy_state.reset(initial)
    torch_state.reset(torch.tensor(initial, dtype=torch.float32))

    steps = max(4, round(0.24 / control_dt))
    for index in range(steps):
        force = np.array(
            [
                [0.0 if index < steps // 2 else 20.0, 20.0],
                [20.0, 20.0],
                [20.0, 0.0 if index < steps // 2 else 20.0],
            ]
        )
        dt = np.array([control_dt, 0.0 if index == 1 else control_dt, control_dt])
        clearance = np.full((3, 2), 0.04)
        position = np.zeros((3, 2, 3))
        position[:, 0, 0] = index * 0.01
        expected = np.array([[False, True], [True, True], [True, index >= steps // 2]])
        phase = np.full(3, index / steps)
        actual_numpy = numpy_state.update(
            force,
            dt,
            sole_clearance_m=clearance,
            sole_position_m=position,
            expected_contact=expected,
            phase=phase,
        )
        actual_torch = torch_state.update(
            torch.tensor(force, dtype=torch.float32),
            torch.tensor(dt, dtype=torch.float32),
            torch.tensor(clearance, dtype=torch.float32),
            sole_position_m=torch.tensor(position, dtype=torch.float32),
            expected_contact=torch.tensor(expected),
            phase=torch.tensor(phase, dtype=torch.float32),
        )
        for field in (
            "raw_contact",
            "stable_contact",
            "stance_age_s",
            "swing_age_s",
            "raw_transition",
            "event_confirmation_ids",
            "last_valid_touchdown_time_s",
            "last_valid_touchdown_position_m",
            "last_opposite_touchdown_time_s",
            "swing_peak_clearance_m",
            "previous_expected_contact",
            "expected_transition_phase",
        ):
            np.testing.assert_allclose(
                getattr(actual_torch, field).numpy(),
                getattr(actual_numpy, field),
                atol=1e-6,
                err_msg=field,
            )
