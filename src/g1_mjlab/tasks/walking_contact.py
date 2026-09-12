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

    def _validate_force(self, force: Any, rows: int) -> None:
        import torch

        if tuple(force.shape) != (rows, 2):
            raise ValueError(f"normal force must have shape ({rows}, 2)")
        if force.device != self.stable_contact.device or not bool(torch.isfinite(force).all()):
            raise ValueError("normal force must be finite and on the state device")

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

    def update(
        self, normal_force_n: Any, dt: float, sole_clearance_m: Any | None = None
    ) -> TorchContactUpdate:
        import torch

        self._validate_force(normal_force_n, self.num_environments)
        if dt < 0 or not torch.isfinite(torch.tensor(dt)):
            raise ValueError("dt must be finite and nonnegative")
        if not bool(self.initialized.all()):
            raise RuntimeError("all contact rows must be reset before update")
        clearance = (
            torch.zeros_like(normal_force_n) if sole_clearance_m is None else sole_clearance_m
        )
        self._validate_force(clearance, self.num_environments)
        empty = torch.zeros_like(self.stable_contact)
        if dt == 0:
            return self._result(empty, empty, empty, empty)
        new_raw = normal_force_n > 0
        raw_transition = new_raw != self.raw_contact
        self.raw_contact[:] = new_raw
        desired = torch.where(
            self.stable_contact,
            normal_force_n >= self.profile.exit_force_n,
            normal_force_n >= self.profile.enter_force_n,
        )
        self.stable_age_s += dt
        airborne = ~self.stable_contact
        self.swing_peak_clearance_m[airborne] = torch.maximum(
            self.swing_peak_clearance_m[airborne], clearance[airborne]
        )
        differs = desired != self.stable_contact
        new_candidate = differs & (~self.candidate_active | (self.candidate_state != desired))
        self.candidate_state[new_candidate] = desired[new_candidate]
        self.candidate_active[new_candidate] = True
        self.candidate_age_s[new_candidate] = dt
        sample_time = (self.elapsed_time_s + dt)[:, None].expand_as(normal_force_n)
        self.candidate_start_time_s[new_candidate] = sample_time[new_candidate]
        continuing = differs & ~new_candidate
        self.candidate_age_s[continuing] += dt
        cancelled = ~differs
        self.candidate_active[cancelled] = False
        self.candidate_age_s[cancelled] = 0
        confirmed = self.candidate_active & (
            self.candidate_age_s + 1e-12 >= self.profile.confirmation_duration_s
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
        self.stable_contact[confirmed] = self.candidate_state[confirmed]
        self.stable_age_s[confirmed] = 0
        self.swing_peak_clearance_m[touchdown] = 0
        self.candidate_active[confirmed] = False
        self.candidate_age_s[confirmed] = 0
        self.elapsed_time_s += dt
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
