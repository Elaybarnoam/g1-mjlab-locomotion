"""Torch contact state and physical contact-point diagnostics for walking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..motion.contact import ContactProfile


@dataclass(frozen=True)
class TorchContactUpdate:
    raw_contact: Any
    stable_contact: Any
    stance_age_s: Any
    swing_age_s: Any
    raw_transition: Any
    touchdown: Any
    liftoff: Any
    valid_touchdown: Any
    first_crossing_time_s: Any
    confirmation_time_s: Any
    swing_peak_clearance_m: Any
    prior_stable_duration_s: Any
    event_confirmation_ids: Any
    last_valid_touchdown_time_s: Any
    last_valid_touchdown_position_m: Any
    last_opposite_touchdown_time_s: Any
    previous_expected_contact: Any
    expected_transition_phase: Any


class TorchContactState:
    """GPU-resident equivalent of :class:`ContactStateMachine`."""

    def __init__(self, num_environments: int, profile: ContactProfile, device: str) -> None:
        if num_environments <= 0:
            raise ValueError("num_environments must be positive")
        import torch

        self.profile = profile
        self.num_environments = num_environments
        self.device = device
        shape = (num_environments, 2)
        self.stable_contact = torch.zeros(shape, dtype=torch.bool, device=device)
        self.raw_contact = torch.zeros(shape, dtype=torch.bool, device=device)
        self.candidate_state = torch.zeros(shape, dtype=torch.bool, device=device)
        self.candidate_active = torch.zeros(shape, dtype=torch.bool, device=device)
        self.candidate_age_s = torch.zeros(shape, device=device)
        self.candidate_start_time_s = torch.zeros(shape, device=device)
        self.stable_age_s = torch.zeros(shape, device=device)
        self.swing_peak_clearance_m = torch.zeros(shape, device=device)
        self.elapsed_time_s = torch.zeros(num_environments, device=device)
        self.initialized = torch.zeros(shape, dtype=torch.bool, device=device)
        self._all_initialized = False
        self.event_confirmation_ids = torch.zeros(shape, dtype=torch.long, device=device)
        self.last_valid_touchdown_time_s = torch.zeros(shape, device=device)
        self.last_valid_touchdown_position_m = torch.zeros((*shape, 3), device=device)
        self.last_opposite_touchdown_time_s = torch.zeros(shape, device=device)
        self.previous_expected_contact = torch.zeros(shape, dtype=torch.bool, device=device)
        self.expected_transition_phase = torch.zeros(shape, device=device)

    def _validate_force(self, force: Any, rows: int, *, finite: bool = True) -> None:
        import torch

        if tuple(force.shape) != (rows, 2):
            raise ValueError(f"normal force must have shape ({rows}, 2)")
        if force.device != self.stable_contact.device:
            raise ValueError("normal force must be on the state device")
        if finite and not bool(torch.isfinite(force).all()):
            raise ValueError("normal force must be finite")

    def reset(self, normal_force_n: Any, env_ids: Any | None = None) -> None:
        import torch

        ids = (
            torch.arange(self.num_environments, device=self.device)
            if env_ids is None
            else env_ids.to(device=self.device, dtype=torch.long)
        )
        if ids.ndim != 1 or len(torch.unique(ids)) != len(ids):
            raise ValueError("env_ids must be a unique one-dimensional tensor")
        if len(ids) and (bool(torch.any(ids < 0)) or bool(torch.any(ids >= self.num_environments))):
            raise ValueError("env_ids are out of range")
        self._validate_force(normal_force_n, len(ids))
        measured = normal_force_n >= self.profile.enter_force_n
        self.stable_contact[ids] = measured
        self.raw_contact[ids] = normal_force_n > 0
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
        if env_ids is None:
            self._all_initialized = True

    def update(
        self,
        normal_force_n: Any,
        dt: Any,
        sole_clearance_m: Any | None = None,
        *,
        sole_position_m: Any | None = None,
        expected_contact: Any | None = None,
        phase: Any | None = None,
    ) -> TorchContactUpdate:
        import torch

        self._validate_force(normal_force_n, self.num_environments, finite=False)
        if not self._all_initialized:
            raise RuntimeError("all contact rows must be reset before update")
        dt_tensor = torch.as_tensor(dt, dtype=normal_force_n.dtype, device=self.device)
        if dt_tensor.ndim == 0:
            dt_tensor = dt_tensor.expand(self.num_environments)
        if tuple(dt_tensor.shape) != (self.num_environments,):
            raise ValueError("dt must be scalar or per-environment")
        dt_grid = dt_tensor[:, None].expand_as(normal_force_n)
        clearance = (
            torch.zeros_like(normal_force_n) if sole_clearance_m is None else sole_clearance_m
        )
        self._validate_force(clearance, self.num_environments, finite=False)
        position = (
            torch.zeros((self.num_environments, 2, 3), device=self.device)
            if sole_position_m is None
            else sole_position_m
        )
        if tuple(position.shape) != (self.num_environments, 2, 3):
            raise ValueError("sole position must have shape (num_environments, 2, 3)")
        expected = self.previous_expected_contact if expected_contact is None else expected_contact
        phase_tensor = (
            torch.zeros(self.num_environments, device=self.device) if phase is None else phase
        )
        if tuple(expected.shape) != (self.num_environments, 2) or tuple(phase_tensor.shape) != (
            self.num_environments,
        ):
            raise ValueError("expected contact or phase has invalid shape")
        expected_changed = (expected != self.previous_expected_contact) & (dt_grid > 0)
        self.expected_transition_phase = torch.where(
            expected_changed, phase_tensor[:, None], self.expected_transition_phase
        )
        self.previous_expected_contact[:] = torch.where(
            dt_grid > 0, expected, self.previous_expected_contact
        )
        advancing = dt_grid > 0
        new_raw = torch.where(advancing, normal_force_n > 0, self.raw_contact)
        raw_transition = new_raw != self.raw_contact
        self.raw_contact[:] = new_raw
        desired = torch.where(
            self.stable_contact,
            normal_force_n >= self.profile.exit_force_n,
            normal_force_n >= self.profile.enter_force_n,
        )
        self.stable_age_s += dt_grid
        airborne = ~self.stable_contact
        self.swing_peak_clearance_m[airborne] = torch.maximum(
            self.swing_peak_clearance_m[airborne], clearance[airborne]
        )
        differs = (desired != self.stable_contact) & advancing
        new_candidate = differs & (~self.candidate_active | (self.candidate_state != desired))
        self.candidate_state[new_candidate] = desired[new_candidate]
        self.candidate_active[new_candidate] = True
        self.candidate_age_s[new_candidate] = dt_grid[new_candidate]
        sample_time = (self.elapsed_time_s + dt_tensor)[:, None].expand_as(normal_force_n)
        self.candidate_start_time_s[new_candidate] = sample_time[new_candidate]
        continuing = differs & ~new_candidate
        self.candidate_age_s[continuing] += dt_grid[continuing]
        cancelled = ~differs
        self.candidate_active[cancelled] = False
        self.candidate_age_s[cancelled] = 0
        confirmed = (
            self.candidate_active
            & (self.candidate_age_s + 1e-6 >= self.profile.confirmation_duration_s)
            & advancing
        )
        liftoff = confirmed & self.stable_contact
        touchdown = confirmed & ~self.stable_contact
        valid_touchdown = (
            touchdown
            & (self.stable_age_s + 1e-6 >= self.profile.minimum_swing_duration_s)
            & (self.swing_peak_clearance_m >= self.profile.minimum_swing_clearance_m)
        )
        confirmation_time = sample_time.clone()
        first_crossing = self.candidate_start_time_s.clone()
        swing_peak = self.swing_peak_clearance_m.clone()
        prior_stable_duration = self.stable_age_s.clone()
        self.event_confirmation_ids += confirmed.long()
        opposite_time = self.last_valid_touchdown_time_s.flip(1)
        self.last_opposite_touchdown_time_s = torch.where(
            valid_touchdown, opposite_time, self.last_opposite_touchdown_time_s
        )
        self.last_valid_touchdown_time_s = torch.where(
            valid_touchdown, sample_time, self.last_valid_touchdown_time_s
        )
        self.last_valid_touchdown_position_m = torch.where(
            valid_touchdown[:, :, None], position, self.last_valid_touchdown_position_m
        )
        self.stable_contact[confirmed] = self.candidate_state[confirmed]
        self.stable_age_s[confirmed] = 0
        self.swing_peak_clearance_m[touchdown] = 0
        self.candidate_active[confirmed] = False
        self.candidate_age_s[confirmed] = 0
        self.elapsed_time_s += dt_tensor
        return self._result(
            raw_transition,
            touchdown,
            liftoff,
            valid_touchdown,
            first_crossing,
            confirmation_time,
            swing_peak,
            prior_stable_duration,
        )

    def _result(
        self,
        raw_transition: Any,
        touchdown: Any,
        liftoff: Any,
        valid_touchdown: Any,
        first_crossing: Any | None = None,
        confirmation_time: Any | None = None,
        swing_peak: Any | None = None,
        prior_stable_duration: Any | None = None,
    ) -> TorchContactUpdate:
        import torch

        stable = self.stable_contact.clone()
        return TorchContactUpdate(
            raw_contact=self.raw_contact.clone(),
            stable_contact=stable,
            stance_age_s=torch.where(stable, self.stable_age_s, 0),
            swing_age_s=torch.where(stable, 0, self.stable_age_s),
            raw_transition=raw_transition,
            touchdown=touchdown,
            liftoff=liftoff,
            valid_touchdown=valid_touchdown,
            first_crossing_time_s=(
                torch.zeros_like(self.candidate_start_time_s)
                if first_crossing is None
                else first_crossing
            ),
            confirmation_time_s=(
                torch.zeros_like(self.candidate_start_time_s)
                if confirmation_time is None
                else confirmation_time
            ),
            swing_peak_clearance_m=(
                self.swing_peak_clearance_m.clone() if swing_peak is None else swing_peak
            ),
            prior_stable_duration_s=(
                self.stable_age_s.clone()
                if prior_stable_duration is None
                else prior_stable_duration
            ),
            event_confirmation_ids=self.event_confirmation_ids.clone(),
            last_valid_touchdown_time_s=self.last_valid_touchdown_time_s.clone(),
            last_valid_touchdown_position_m=self.last_valid_touchdown_position_m.clone(),
            last_opposite_touchdown_time_s=self.last_opposite_touchdown_time_s.clone(),
            previous_expected_contact=self.previous_expected_contact.clone(),
            expected_transition_phase=self.expected_transition_phase.clone(),
        )


def contact_point_velocity(
    body_origin_w: Any, body_lin_vel_w: Any, body_ang_vel_w: Any, point_w: Any
) -> Any:
    """Return point velocity from body-origin spatial velocity."""
    import torch

    return body_lin_vel_w + torch.cross(body_ang_vel_w, point_w - body_origin_w, dim=-1)


def tangential_slip_terms(velocity_w: Any, normal_w: Any, normal_force_n: Any) -> tuple[Any, Any]:
    """Return force-weighted tangential-speed-square numerator and denominator."""
    import torch

    normal = torch.nn.functional.normalize(normal_w, dim=-1, eps=1e-12)
    tangent_velocity = velocity_w - torch.sum(velocity_w * normal, dim=-1, keepdim=True) * normal
    denominator = torch.clamp(normal_force_n, min=0)
    numerator = denominator * torch.sum(torch.square(tangent_velocity), dim=-1)
    return numerator, denominator


def add_diagnostic_contact_sensor(env_cfg: Any, *, num_slots: int = 4) -> None:
    """Add a contact-level sensor only to a bounded diagnostic environment."""
    if not 1 <= num_slots <= 4:
        raise ValueError("diagnostic contact sensor supports 1..4 slots")
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    sensor = ContactSensorCfg(
        name="feet_ground_diagnostic_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "pos", "normal", "tangent"),
        reduce="none",
        num_slots=num_slots,
        global_frame=True,
    )
    env_cfg.scene.sensors = (env_cfg.scene.sensors or ()) + (sensor,)


class PhysicsContactRecorder:
    """Retain contact-level values at each physics substep on the device."""

    def __init__(self, env: Any, *, num_slots: int = 4) -> None:
        import torch

        self.env = env
        self.num_slots = num_slots
        self.robot = env.scene["robot"]
        self.sensor = env.scene["feet_ground_diagnostic_contact"]
        self.body_ids = torch.tensor(
            [
                self.robot.body_names.index("left_ankle_roll_link"),
                self.robot.body_names.index("right_ankle_roll_link"),
            ],
            device=env.device,
        ).repeat_interleave(num_slots)
        self.site_ids = torch.tensor(
            [self.robot.site_names.index("left_foot"), self.robot.site_names.index("right_foot")],
            device=env.device,
        )
        names = (
            "contact",
            "normal_force_n",
            "tangential_speed_square_numerator",
            "contact_force_denominator",
            "contact_count",
            "lowest_sole_clearance_m",
            "ankle_position_w",
            "sole_position_w",
        )
        self.records: dict[str, list[Any]] = {name: [] for name in names}
        self.steps = 0

    def capture(self) -> None:
        import torch

        data = self.sensor.data
        if any(value is None for value in (data.found, data.force, data.pos, data.normal)):
            raise RuntimeError("diagnostic contact sensor did not expose contact-level fields")
        found = data.found > 0
        force = data.force
        point = data.pos
        normal = data.normal
        origin = self.robot.data.body_link_pos_w[:, self.body_ids]
        linear = self.robot.data.body_link_lin_vel_w[:, self.body_ids]
        angular = self.robot.data.body_link_ang_vel_w[:, self.body_ids]
        point_velocity = contact_point_velocity(origin, linear, angular, point)
        normal_force = torch.abs(torch.sum(force * normal, dim=-1)) * found
        numerator, denominator = tangential_slip_terms(point_velocity, normal, normal_force)
        shape = (self.env.num_envs, 2, self.num_slots)
        sole_position = self.robot.data.site_pos_w[:, self.site_ids]
        self.records["contact"].append(found.view(shape).any(dim=2).clone())
        self.records["normal_force_n"].append(normal_force.view(shape).sum(dim=2).clone())
        self.records["tangential_speed_square_numerator"].append(
            numerator.view(shape).sum(dim=2).clone()
        )
        self.records["contact_force_denominator"].append(denominator.view(shape).sum(dim=2).clone())
        self.records["contact_count"].append(found.view(shape).sum(dim=2).clone())
        self.records["lowest_sole_clearance_m"].append(sole_position[:, :, 2].clone())
        self.records["ankle_position_w"].append(
            self.robot.data.body_link_pos_w[:, self.body_ids[:: self.num_slots]].clone()
        )
        self.records["sole_position_w"].append(sole_position.clone())
        self.steps += 1

    def arrays_for_environment(self, environment: int, physics_dt: float) -> dict[str, Any]:
        import numpy as np
        import torch

        if not self.steps:
            raise RuntimeError("physics recorder has no samples")
        arrays = {
            name: torch.stack(values)[:, environment].cpu().numpy()
            for name, values in self.records.items()
        }
        arrays["time_s"] = (np.arange(self.steps, dtype=float) + 1) * physics_dt
        return arrays


@dataclass(frozen=True)
class WalkingContactSnapshot:
    """Cached raw terms from exactly one control transition."""

    phase_contact_error: Any
    unmatched_events: Any
    short_stance: Any
    short_swing: Any
    physical_stance_slip: Any
    swing_clearance_error: Any
    touchdown_placement: Any
    bilateral_flight: Any
    raw_transition_count: Any
    event_confirmation_ids: Any


class WalkingContactRuntime:
    """Own all stateful Stage 19 contact mechanics and update it once per transition."""

    def __init__(self, env: Any, command: Any, parameters: dict[str, float]) -> None:
        import torch

        self.env = env
        self.command = command
        self.dt = float(env.step_dt)
        self.parameters = parameters
        profile = ContactProfile(
            enter_force_n=parameters["enter_force_n"],
            exit_force_n=parameters["exit_force_n"],
            confirmation_duration_s=parameters["confirmation_duration_s"],
            minimum_swing_duration_s=parameters["minimum_swing_duration_s"],
            minimum_swing_clearance_m=parameters["minimum_swing_clearance_m"],
        )
        self.contact = TorchContactState(env.num_envs, profile, str(env.device))
        self.sensor = env.scene["stage19_feet_contact"]
        self.robot = env.scene["robot"]
        slots = int(parameters["contact_slots"])
        self.slots = slots
        ankle_ids = [
            self.robot.body_names.index("left_ankle_roll_link"),
            self.robot.body_names.index("right_ankle_roll_link"),
        ]
        self.body_ids = torch.tensor(ankle_ids, device=env.device).repeat_interleave(slots)
        self.site_ids = torch.tensor(
            [self.robot.site_names.index("left_foot"), self.robot.site_names.index("right_foot")],
            device=env.device,
        )
        shape = (env.num_envs, 2)
        self.previous_phase = torch.zeros(env.num_envs, device=env.device)
        self.cycle_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.last_matched_cycle = torch.full(
            (env.num_envs, 2, 2), -1, dtype=torch.long, device=env.device
        )
        self.event_confirmation_ids = torch.zeros(shape, dtype=torch.long, device=env.device)
        self.walking_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.grace_remaining_s = torch.zeros(env.num_envs, device=env.device)
        self.bilateral_airborne_age_s = torch.zeros(env.num_envs, device=env.device)
        self.sustained_flight_since_touchdown = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.last_touchdown_valid = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.last_touchdown_foot_xy = torch.zeros((env.num_envs, 2, 2), device=env.device)
        self.last_touchdown_root_xy = torch.zeros((env.num_envs, 2, 2), device=env.device)
        self._transition_phase = self._reference_transition_phases(command.reference_contact)
        self.snapshot = self._zero_snapshot()
        self._initialized = False

    def _reference_transition_phases(self, contact: Any) -> Any:
        import torch

        interval_count = contact.shape[0] - 1
        cycle = contact[:interval_count]
        previous = torch.roll(cycle, 1, dims=0)
        phases = torch.arange(interval_count, device=contact.device) / interval_count
        result = torch.zeros((2, 2, interval_count), device=contact.device)
        validity = torch.zeros_like(result, dtype=torch.bool)
        for kind, transition in enumerate((~cycle & previous, cycle & ~previous)):
            for foot in range(2):
                selected = phases[transition[:, foot]]
                result[kind, foot, : len(selected)] = selected
                validity[kind, foot, : len(selected)] = True
        return result, validity

    def _zero_snapshot(self) -> WalkingContactSnapshot:
        import torch

        zeros = torch.zeros(self.env.num_envs, device=self.env.device)
        return WalkingContactSnapshot(
            phase_contact_error=zeros.clone(),
            unmatched_events=zeros.clone(),
            short_stance=zeros.clone(),
            short_swing=zeros.clone(),
            physical_stance_slip=zeros.clone(),
            swing_clearance_error=zeros.clone(),
            touchdown_placement=zeros.clone(),
            bilateral_flight=zeros.clone(),
            raw_transition_count=zeros.clone(),
            event_confirmation_ids=self.event_confirmation_ids.clone(),
        )

    def _measure_contacts(self) -> tuple[Any, Any, Any]:
        import torch

        data = self.sensor.data
        if data.found is None or data.force is None or data.pos is None or data.normal is None:
            raise RuntimeError("Stage 19 contact sensor fields are unavailable")
        found = data.found > 0
        normal_force = (
            torch.nan_to_num(
                torch.abs(torch.sum(data.force * data.normal, dim=-1)),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            * found
        )
        shape = (self.env.num_envs, 2, self.slots)
        force_by_foot = normal_force.view(shape).sum(dim=2)
        origin = self.robot.data.body_link_pos_w[:, self.body_ids]
        linear = self.robot.data.body_link_lin_vel_w[:, self.body_ids]
        angular = self.robot.data.body_link_ang_vel_w[:, self.body_ids]
        velocity = contact_point_velocity(origin, linear, angular, data.pos)
        numerator, denominator = tangential_slip_terms(velocity, data.normal, normal_force)
        slip_square = numerator.view(shape).sum(dim=2) / torch.clamp(
            denominator.view(shape).sum(dim=2), min=1e-9
        )
        sole_position = torch.nan_to_num(
            self.robot.data.site_pos_w[:, self.site_ids], nan=0.0, posinf=0.0, neginf=0.0
        )
        return force_by_foot, torch.nan_to_num(slip_square), sole_position

    def reset(self, env_ids: Any | None = None) -> None:
        import torch

        ids = (
            torch.arange(self.env.num_envs, device=self.env.device) if env_ids is None else env_ids
        )
        force, _, _ = self._measure_contacts()
        if env_ids is None or len(ids) == self.env.num_envs:
            self.contact.reset(force)
        else:
            self.contact.reset(force[ids], ids)
        for value in (
            self.last_matched_cycle,
            self.event_confirmation_ids,
            self.last_touchdown_valid,
            self.last_touchdown_foot_xy,
            self.last_touchdown_root_xy,
        ):
            value[ids] = 0
        self.last_matched_cycle[ids] = -1
        self.previous_phase[ids] = self.command.phase[ids]
        self.cycle_id[ids] = 0
        self.walking_active[ids] = False
        self.grace_remaining_s[ids] = 0
        self.bilateral_airborne_age_s[ids] = 0
        self.sustained_flight_since_touchdown[ids] = False
        full_reset = env_ids is None or len(ids) == self.env.num_envs
        if full_reset:
            self.snapshot = self._zero_snapshot()
            self._initialized = True
        else:

            def cleared(field: str) -> Any:
                value = getattr(self.snapshot, field).clone()
                value[ids] = 0
                return value

            self.snapshot = WalkingContactSnapshot(
                phase_contact_error=cleared("phase_contact_error"),
                unmatched_events=cleared("unmatched_events"),
                short_stance=cleared("short_stance"),
                short_swing=cleared("short_swing"),
                physical_stance_slip=cleared("physical_stance_slip"),
                swing_clearance_error=cleared("swing_clearance_error"),
                touchdown_placement=cleared("touchdown_placement"),
                bilateral_flight=cleared("bilateral_flight"),
                raw_transition_count=cleared("raw_transition_count"),
                event_confirmation_ids=self.event_confirmation_ids.clone(),
            )

    def _event_matches(self, events: Any, kind: int, phase: Any) -> Any:
        import torch

        transition_phase, valid = self._transition_phase
        candidates = transition_phase[kind][None, :, :]
        distance = torch.abs(phase[:, None, None] - candidates)
        circular = torch.minimum(distance, 1.0 - distance)
        circular = torch.where(valid[kind][None, :, :], circular, torch.full_like(circular, 2.0))
        near = torch.amin(circular, dim=2) <= self.parameters["phase_tolerance_cycle"]
        unused = self.last_matched_cycle[:, :, kind] != self.cycle_id[:, None]
        matched = events & near & unused
        self.last_matched_cycle[:, :, kind] = torch.where(
            matched, self.cycle_id[:, None], self.last_matched_cycle[:, :, kind]
        )
        return matched

    def update(self) -> WalkingContactSnapshot:
        """Advance cached mechanics once; reward readers are pure after this call."""
        import torch

        force, slip_square, sole_position = self._measure_contacts()
        if not self._initialized:
            self.reset()
            return self.snapshot
        phase = self.command.phase
        expected = self.command.foot_contact
        contact = self.contact.update(
            force,
            self.dt,
            sole_position[:, :, 2],
            sole_position_m=sole_position,
            expected_contact=expected,
            phase=phase,
        )
        wrapped = phase + 1e-7 < self.previous_phase
        self.cycle_id += wrapped.long()
        self.previous_phase[:] = phase
        active = (self.command.command[:, 0] >= 0.15) & (self.command.blend > 0)
        changed = active != self.walking_active
        self.grace_remaining_s = torch.where(
            changed,
            torch.full_like(self.grace_remaining_s, self.parameters["transition_grace_s"]),
            torch.clamp(self.grace_remaining_s - self.dt, min=0),
        )
        self.walking_active[:] = active
        gait_mask = self.command.blend * active * (self.grace_remaining_s <= 0)
        touchdown_match = self._event_matches(contact.touchdown, 1, phase)
        liftoff_match = self._event_matches(contact.liftoff, 0, phase)
        event_mask = (self.grace_remaining_s <= 0)[:, None]
        unmatched = (contact.touchdown & ~touchdown_match) | (contact.liftoff & ~liftoff_match)
        unmatched = (unmatched & event_mask).float().sum(dim=1)
        self.event_confirmation_ids[:] = contact.event_confirmation_ids
        short_stance = (
            contact.liftoff.float()
            * torch.clamp(
                (self.parameters["minimum_stance_duration_s"] - contact.prior_stable_duration_s)
                / self.parameters["minimum_stance_duration_s"],
                0,
                1,
            )
            * event_mask
        ).sum(dim=1)
        short_swing = (
            contact.touchdown.float()
            * torch.clamp(
                (self.parameters["minimum_swing_duration_s"] - contact.prior_stable_duration_s)
                / self.parameters["minimum_swing_duration_s"],
                0,
                1,
            )
            * event_mask
        ).sum(dim=1)
        airborne = ~contact.stable_contact.any(dim=1)
        self.bilateral_airborne_age_s = torch.where(
            airborne, self.bilateral_airborne_age_s + self.dt, 0
        )
        sustained_flight = (
            self.bilateral_airborne_age_s >= self.parameters["bilateral_flight_duration_s"]
        )
        self.sustained_flight_since_touchdown |= sustained_flight
        quaternion = torch.nan_to_num(
            self.robot.data.root_link_quat_w, nan=0.0, posinf=0.0, neginf=0.0
        )
        yaw = torch.atan2(
            2 * (quaternion[:, 0] * quaternion[:, 3] + quaternion[:, 1] * quaternion[:, 2]),
            1 - 2 * (torch.square(quaternion[:, 2]) + torch.square(quaternion[:, 3])),
        )
        heading = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=1)
        opposite_foot = self.last_touchdown_foot_xy.flip(1)
        opposite_root = self.last_touchdown_root_xy.flip(1)
        opposite_valid = self.last_touchdown_valid.flip(1)
        root_xy = torch.nan_to_num(
            self.robot.data.root_link_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0
        )
        sole_xy = sole_position[:, :, :2]
        step_length = torch.abs(torch.sum((sole_xy - opposite_foot) * heading[:, None, :], dim=2))
        root_progress = torch.sum(
            (root_xy[:, None, :] - opposite_root) * heading[:, None, :], dim=2
        )
        step_reference = self.parameters["step_reference_m"]
        eligible = (
            contact.valid_touchdown
            & opposite_valid
            & ~self.sustained_flight_since_touchdown[:, None]
            & (
                root_progress
                >= self.parameters["minimum_root_progress_fraction"]
                * max(step_reference, self.parameters["minimum_step_reference_m"])
            )
        )
        placement = (
            eligible.float()
            * torch.exp(
                -torch.square((step_length - step_reference) / self.parameters["placement_scale_m"])
            )
        ).sum(dim=1)
        valid = contact.valid_touchdown
        self.last_touchdown_foot_xy = torch.where(
            valid[:, :, None], sole_xy, self.last_touchdown_foot_xy
        )
        self.last_touchdown_root_xy = torch.where(
            valid[:, :, None], root_xy[:, None, :], self.last_touchdown_root_xy
        )
        self.last_touchdown_valid |= valid
        self.sustained_flight_since_touchdown = torch.where(
            valid.any(dim=1),
            torch.zeros_like(self.sustained_flight_since_touchdown),
            self.sustained_flight_since_touchdown,
        )
        phase_error = gait_mask * torch.abs(contact.stable_contact.float() - expected.float()).mean(
            dim=1
        )
        physical_slip = gait_mask * torch.mean(
            contact.stable_contact
            * torch.clamp(
                slip_square / self.parameters["slip_scale_m_s"] ** 2,
                0,
                self.parameters["squared_error_clip"],
            ),
            dim=1,
        )
        clearance_error = gait_mask * torch.mean(
            (~expected)
            * torch.clamp(
                torch.square(
                    (sole_position[:, :, 2] - self.parameters["swing_clearance_reference_m"])
                    / self.parameters["clearance_scale_m"]
                ),
                0,
                self.parameters["squared_error_clip"],
            ),
            dim=1,
        )
        self.snapshot = WalkingContactSnapshot(
            phase_error,
            gait_mask * unmatched,
            gait_mask * short_stance,
            gait_mask * short_swing,
            physical_slip,
            clearance_error,
            placement,
            gait_mask * sustained_flight,
            contact.raw_transition.float().sum(dim=1),
            self.event_confirmation_ids.clone(),
        )
        return self.snapshot
