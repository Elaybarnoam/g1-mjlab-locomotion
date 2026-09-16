from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from g1_mjlab.tasks.walking_contact import WalkingContactRuntime

torch = pytest.importorskip("torch")


def _parameters() -> dict[str, float]:
    return {
        "enter_force_n": 15.0,
        "exit_force_n": 8.0,
        "confirmation_duration_s": 0.06,
        "minimum_stance_duration_s": 0.2,
        "minimum_swing_duration_s": 0.12,
        "minimum_swing_clearance_m": 0.025,
        "phase_tolerance_cycle": 0.1,
        "transition_grace_s": 1.0,
        "bilateral_flight_duration_s": 0.06,
        "slip_scale_m_s": 0.12,
        "swing_clearance_reference_m": 0.03,
        "clearance_scale_m": 0.03,
        "placement_scale_m": 0.1,
        "step_reference_m": 0.6064067445483481,
        "minimum_root_progress_fraction": 0.5,
        "minimum_step_reference_m": 0.18,
        "squared_error_clip": 4.0,
        "contact_slots": 1.0,
    }


def _runtime(
    initial_contact: tuple[bool, bool] = (True, False), num_envs: int = 1
) -> tuple[Any, Any, Any]:
    device = "cpu"
    sensor_data = SimpleNamespace(
        found=torch.zeros((num_envs, 2), device=device),
        force=torch.zeros((num_envs, 2, 3), device=device),
        pos=torch.zeros((num_envs, 2, 3), device=device),
        normal=torch.zeros((num_envs, 2, 3), device=device),
    )
    sensor_data.normal[..., 2] = 1
    robot_data = SimpleNamespace(
        body_link_pos_w=torch.zeros((num_envs, 2, 3), device=device),
        body_link_lin_vel_w=torch.zeros((num_envs, 2, 3), device=device),
        body_link_ang_vel_w=torch.zeros((num_envs, 2, 3), device=device),
        site_pos_w=torch.zeros((num_envs, 2, 3), device=device),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_envs, 1),
        root_link_pos_w=torch.zeros((num_envs, 3), device=device),
    )
    robot = SimpleNamespace(
        body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        site_names=("left_foot", "right_foot"),
        data=robot_data,
    )
    sensor = SimpleNamespace(data=sensor_data)
    env = SimpleNamespace(
        num_envs=num_envs,
        device=device,
        step_dt=0.02,
        scene={"stage19_feet_contact": sensor, "robot": robot},
    )
    reference = torch.tensor(
        [[True, False], [True, False], [False, True], [False, True], [True, False]],
        device=device,
    )
    command = SimpleNamespace(
        reference_contact=reference,
        phase=torch.zeros(num_envs, device=device),
        foot_contact=torch.tensor([initial_contact], device=device).repeat(num_envs, 1),
        command=torch.tensor([[0.6, 0.0, 0.0]], device=device).repeat(num_envs, 1),
        blend=torch.ones(num_envs, device=device),
    )
    runtime = WalkingContactRuntime(env, command, _parameters())
    _set_contact(sensor_data, initial_contact)
    runtime.reset()
    runtime.walking_active[:] = True
    return runtime, command, sensor_data


def _set_contact(sensor_data: Any, contact: tuple[bool, bool]) -> None:
    values = torch.tensor([contact], device=sensor_data.found.device).expand_as(sensor_data.found)
    sensor_data.found[:] = values
    sensor_data.force.zero_()
    sensor_data.force[..., 2] = values * 20.0


def test_reward_snapshot_is_pure_and_phase_matched_contact_scores_better() -> None:
    runtime, command, _ = _runtime()
    matched = runtime.update()
    ids_before = matched.event_confirmation_ids.clone()
    first_read = matched.phase_contact_error
    second_read = runtime.snapshot.phase_contact_error
    assert torch.equal(first_read, second_read)
    assert torch.equal(runtime.snapshot.event_confirmation_ids, ids_before)
    assert first_read.item() == pytest.approx(0.0)

    command.foot_contact[:] = torch.tensor([[False, True]])
    shifted = runtime.update()
    assert shifted.phase_contact_error.item() > first_read.item()


def test_partial_reset_preserves_other_world_snapshot_and_state() -> None:
    runtime, command, _ = _runtime(num_envs=2)
    command.foot_contact[1] = torch.tensor([False, True])
    before = runtime.update()
    assert before.phase_contact_error.tolist() == [0.0, 1.0]

    runtime.reset(torch.tensor([0]))

    assert runtime.snapshot.phase_contact_error.tolist() == [0.0, 1.0]
    assert runtime.contact.initialized.tolist() == [[True, True], [True, True]]


def test_idle_has_no_placement_and_static_leg_cycle_cannot_earn_it() -> None:
    runtime, command, sensor = _runtime((True, True))
    command.command.zero_()
    command.blend.zero_()
    assert runtime.update().touchdown_placement.item() == 0

    command.command[:, 0] = 0.6
    command.blend[:] = 1
    runtime.walking_active[:] = True
    for foot in (0, 1):
        other = 1 - foot
        for _ in range(9):
            state = [False, False]
            state[other] = True
            _set_contact(sensor, tuple(state))
            runtime.robot.data.site_pos_w[:, foot, 2] = 0.04
            runtime.update()
        for _ in range(3):
            _set_contact(sensor, (True, True))
            snapshot = runtime.update()
    assert snapshot.touchdown_placement.item() == 0


def test_hopping_triggers_flight_and_contact_slip_is_monotonic() -> None:
    runtime, command, sensor = _runtime((True, True))
    command.foot_contact[:] = True
    slip_values = []
    for speed in (0.0, 0.06, 0.12):
        runtime.robot.data.body_link_lin_vel_w.zero_()
        runtime.robot.data.body_link_lin_vel_w[..., 0] = speed
        slip_values.append(runtime.update().physical_stance_slip.item())
    assert slip_values[0] <= slip_values[1] <= slip_values[2]

    _set_contact(sensor, (False, False))
    flight = 0.0
    for _ in range(7):
        flight = runtime.update().bilateral_flight.item()
    assert flight > 0


def test_missing_contacts_and_nonfinite_contact_measurements_produce_finite_rewards() -> None:
    runtime, _, sensor = _runtime((False, False))
    sensor.force[:] = torch.nan
    sensor.pos[:] = torch.inf
    runtime.robot.data.site_pos_w[:] = torch.nan

    snapshot = runtime.update()

    for field in (
        "phase_contact_error",
        "physical_stance_slip",
        "swing_clearance_error",
        "bilateral_flight",
    ):
        assert torch.isfinite(getattr(snapshot, field)).all()


def _integrated_phase_error(control_dt: float) -> float:
    total = 0.0
    steps = round(4.0 / control_dt)
    for index in range(steps):
        time_s = (index + 0.5) * control_dt
        actual = (int(time_s / 0.4) % 2) == 0
        expected = (int((time_s + 0.08) / 0.4) % 2) == 0
        error = abs(float(actual) - float(expected))
        total += error * control_dt
    return total


def test_rate_reward_integration_is_control_rate_stable() -> None:
    values = [_integrated_phase_error(dt) for dt in (0.01, 0.02, 0.04)]
    assert max(values) - min(values) <= 0.02 * max(values)
