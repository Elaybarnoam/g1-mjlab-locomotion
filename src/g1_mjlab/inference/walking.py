"""Native walking policy state, observation composition, and ONNX inference."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from ..artifacts import sha256_file
from ..motion.gait import CommandProfile, GaitState, step_gait_numpy
from ..qualification import canonical_hash

FloatArray = npt.NDArray[np.float32]

_ACTOR_LAYOUT = (
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 29),
    ("joint_vel", 29),
    ("actions", 29),
    ("command", 3),
    ("phase_sin", 1),
    ("phase_cos", 1),
    ("walk_blend", 1),
)


def _vector(value: npt.ArrayLike, size: int, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


@dataclass(frozen=True, slots=True)
class WalkingSensorState:
    base_linear_velocity: npt.ArrayLike
    base_angular_velocity: npt.ArrayLike
    projected_gravity: npt.ArrayLike
    joint_position: npt.ArrayLike
    joint_velocity: npt.ArrayLike

    def vectors(self) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        return (
            _vector(self.base_linear_velocity, 3, "base linear velocity"),
            _vector(self.base_angular_velocity, 3, "base angular velocity"),
            _vector(self.projected_gravity, 3, "projected gravity"),
            _vector(self.joint_position, 29, "joint position"),
            _vector(self.joint_velocity, 29, "joint velocity"),
        )


@dataclass(frozen=True, slots=True)
class WalkingCommandDiagnostics:
    requested_forward_speed_m_s: float
    observed_applied_forward_speed_m_s: float
    observed_phase: float
    observed_blend: float
    next_applied_forward_speed_m_s: float
    next_phase: float
    next_blend: float


@dataclass(frozen=True, slots=True)
class WalkingPolicyStep:
    action: FloatArray
    observation: FloatArray
    command: WalkingCommandDiagnostics


def _validate_contract(contract: dict[str, Any]) -> None:
    if (
        contract.get("schema_version") != 2
        or contract.get("task_id") != "G1-Walking-Flat-v1"
        or contract.get("layout_id") != "g1-walking-actor-v1"
    ):
        raise ValueError("unsupported walking contract identity")
    fields = contract.get("actor_fields")
    if not isinstance(fields, list) or [
        (field.get("name"), field.get("size")) for field in fields
    ] != list(_ACTOR_LAYOUT):
        raise ValueError("walking actor layout does not match the 102-value contract")
    offset = 0
    for field in fields:
        if field.get("offset") != offset:
            raise ValueError("walking actor offsets are not contiguous")
        offset += int(field["size"])
        term = field.get("resolved_term")
        if not isinstance(term, dict) or any(
            term.get(name) is not None and term.get(name) != 0
            for name in ("scale", "clip", "history_length", "delay_max_lag")
        ):
            raise ValueError("walking actor contains an unsupported transform or history")
    if offset != 102:
        raise ValueError("walking actor must contain exactly 102 values")


def compose_walking_actor_observation(
    contract: dict[str, Any],
    sensor_state: WalkingSensorState,
    previous_action: npt.ArrayLike,
    gait_state: GaitState,
) -> FloatArray:
    """Compose the raw 102-value actor vector; ONNX owns frozen normalization."""
    _validate_contract(contract)
    gait_state.validate()
    if gait_state.phase.shape != (1,):
        raise ValueError("native walking inference supports one robot per policy session")
    base_linear, base_angular, gravity, joint_position, joint_velocity = sensor_state.vectors()
    previous = _vector(previous_action, 29, "previous action")
    nominal = _vector(contract["nominal_joint_position"], 29, "nominal joint position")
    bias = _vector(contract["encoder_bias"], 29, "encoder bias")
    result = np.concatenate(
        (
            base_linear,
            base_angular,
            gravity,
            joint_position - nominal + bias,
            joint_velocity,
            previous,
            gait_state.applied_command[0].astype(np.float32),
            gait_state.policy_features[0].astype(np.float32),
        )
    ).astype(np.float32, copy=False)
    if result.shape != (102,) or not np.isfinite(result).all():
        raise RuntimeError("composed walking actor observation is invalid")
    return result


class WalkingPolicySession:
    """One inference-only robot session with explicit state and fail-closed policy status."""

    def __init__(
        self,
        contract: dict[str, Any],
        bundle: dict[str, Any],
        command_profile: CommandProfile,
        runtime: Any,
    ) -> None:
        self.contract = contract
        self.bundle = bundle
        self.command_profile = command_profile
        self.runtime = runtime
        self.previous_action = np.zeros(29, dtype=np.float32)
        self.gait_state = GaitState.zeros(1)

    @classmethod
    def from_components(
        cls,
        contract: dict[str, Any],
        bundle: dict[str, Any],
        command_profile: CommandProfile,
        runtime: Any,
        *,
        allow_unqualified: bool = False,
    ) -> Self:
        _validate_contract(contract)
        command_profile.validate()
        if bundle.get("status") != "qualified" and not allow_unqualified:
            raise ValueError(
                "walking policy is unqualified; explicit development opt-in is required"
            )
        inputs = runtime.get_inputs()
        outputs = runtime.get_outputs()
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or inputs[0].shape[-1] != 102
            or outputs[0].shape[-1] != 29
            or inputs[0].type != "tensor(float)"
            or outputs[0].type != "tensor(float)"
        ):
            raise ValueError("ONNX walking signature must be float32 [batch,102] -> [batch,29]")
        return cls(contract, bundle, command_profile, runtime)

    @classmethod
    def load(cls, bundle_dir: Path, *, allow_unqualified: bool = False) -> Self:
        """Validate every bundled byte stream before constructing CPU ONNX inference."""
        import onnxruntime as ort

        root = bundle_dir.resolve(strict=True)
        bundle = json.loads((root / "walking-policy-bundle.json").read_text(encoding="utf-8"))
        expected_fields = {
            "schema_version",
            "task_id",
            "layout_id",
            "status",
            "checkpoint",
            "checkpoint_sha256",
            "source_run",
            "development_command_domain_m_s",
            "qualified_command_domain_m_s",
            "files",
        }
        if (
            not isinstance(bundle, dict)
            or set(bundle) != expected_fields
            or bundle["schema_version"] != 2
        ):
            raise ValueError("walking policy bundle fields do not match schema 2")
        expected_files = {
            "contract.json",
            "host-profile.json",
            "model.mjb",
            "policy.onnx",
            "reference.npz",
        }
        if set(bundle["files"]) != expected_files:
            raise ValueError("walking policy bundle file inventory is incomplete")
        for name, digest in bundle["files"].items():
            if sha256_file(root / name) != digest:
                raise ValueError(f"walking policy bundle integrity mismatch: {name}")
        contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
        if canonical_hash(contract) != contract.get("sha256"):
            raise ValueError("walking contract canonical hash mismatch")
        host = json.loads((root / "host-profile.json").read_text(encoding="utf-8"))
        host_fields = {
            "schema_version",
            "reference_speed_m_s",
            "cycle_duration_s",
            "acceleration_m_s2",
            "deceleration_m_s2",
            "blend_rate_s",
            "stand_threshold_m_s",
            "walk_threshold_m_s",
            "reference_file",
        }
        if not isinstance(host, dict) or set(host) != host_fields or host["schema_version"] != 1:
            raise ValueError("walking host profile fields do not match schema")
        if host["reference_file"] != "reference.npz":
            raise ValueError("walking host reference must resolve inside the bundle")
        profile = CommandProfile(
            host["reference_speed_m_s"],
            host["cycle_duration_s"],
            host["acceleration_m_s2"],
            host["deceleration_m_s2"],
            host["blend_rate_s"],
            host["stand_threshold_m_s"],
            host["walk_threshold_m_s"],
        )
        runtime = ort.InferenceSession(
            str(root / "policy.onnx"), providers=["CPUExecutionProvider"]
        )
        return cls.from_components(
            contract,
            bundle,
            profile,
            runtime,
            allow_unqualified=allow_unqualified,
        )

    def reset(self, initial_state: WalkingSensorState, *, phase: float = 0.0) -> None:
        initial_state.vectors()
        if not math.isfinite(phase) or not 0 <= phase < 1:
            raise ValueError("initial walking phase must be finite in [0, 1)")
        self.previous_action.fill(0)
        self.gait_state = GaitState.zeros(1)
        self.gait_state.phase[0] = phase

    def step(self, requested_command: float, sensor_state: WalkingSensorState) -> WalkingPolicyStep:
        if not math.isfinite(requested_command):
            raise ValueError("requested walking command must be finite")
        domain_name = (
            "qualified_command_domain_m_s"
            if self.bundle.get("status") == "qualified"
            else "development_command_domain_m_s"
        )
        domain = self.bundle.get(domain_name)
        if (
            not isinstance(domain, list)
            or len(domain) != 2
            or requested_command < float(domain[0])
            or requested_command > float(domain[1])
        ):
            raise ValueError(f"requested command is outside the bundle {domain_name} domain")
        observed = self.gait_state
        observation = compose_walking_actor_observation(
            self.contract, sensor_state, self.previous_action, observed
        )
        output = self.runtime.run(None, {self.runtime.get_inputs()[0].name: observation[None, :]})[
            0
        ]
        raw = np.asarray(output, dtype=np.float32)
        if raw.shape != (1, 29) or not np.isfinite(raw).all():
            raise RuntimeError("ONNX walking action must be finite with shape [1,29]")
        clip = self.contract.get("action_clip")
        action = raw[0].copy() if clip is None else np.clip(raw[0], -clip, clip)
        next_state = step_gait_numpy(
            observed,
            np.asarray([[requested_command, 0.0, 0.0]]),
            self.command_profile,
            dt=float(self.contract["control_dt"]),
        )
        diagnostics = WalkingCommandDiagnostics(
            requested_forward_speed_m_s=requested_command,
            observed_applied_forward_speed_m_s=float(observed.applied_command[0, 0]),
            observed_phase=float(observed.phase[0]),
            observed_blend=float(observed.blend[0]),
            next_applied_forward_speed_m_s=float(next_state.applied_command[0, 0]),
            next_phase=float(next_state.phase[0]),
            next_blend=float(next_state.blend[0]),
        )
        self.previous_action = action.astype(np.float32, copy=False)
        self.gait_state = next_state
        return WalkingPolicyStep(self.previous_action.copy(), observation, diagnostics)


def walking_joint_targets(contract: dict[str, Any], action: npt.ArrayLike) -> FloatArray:
    """Map normalized actions to the 29 actuator position targets exactly once."""
    _validate_contract(contract)
    if contract.get("target_clip") is not None:
        raise ValueError("walking target clipping is not supported by this bundle schema")
    normalized = _vector(action, 29, "walking action")
    clip = contract.get("action_clip")
    if clip is not None:
        normalized = np.clip(normalized, -float(clip), float(clip))
    nominal = _vector(contract["nominal_joint_position"], 29, "nominal joint position")
    scale = _vector(contract["action_scale"], 29, "action scale")
    bias = _vector(contract["encoder_bias"], 29, "encoder bias")
    targets = nominal + scale * normalized - bias
    if not np.isfinite(targets).all():
        raise RuntimeError("walking actuator targets are non-finite")
    return targets.astype(np.float32, copy=False)
