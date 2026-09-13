"""Immutable speed-indexed walking references with NumPy/Torch sampling parity."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from ..artifacts import sha256_file, write_atomic_json

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ReferenceSample:
    joint_position: FloatArray
    joint_partial_phase: FloatArray
    joint_partial_speed: FloatArray
    local_foot_position: FloatArray
    contact: BoolArray
    root_velocity: FloatArray
    phase_rate_hz: FloatArray


@dataclass(frozen=True, slots=True)
class TorchReferenceSample:
    joint_position: Any
    joint_partial_phase: Any
    joint_partial_speed: Any
    local_foot_position: Any
    contact: Any
    root_velocity: Any
    phase_rate_hz: Any


@dataclass(frozen=True, slots=True)
class ReferenceTargets:
    joint_position: FloatArray
    joint_velocity: FloatArray


@dataclass(frozen=True, slots=True)
class ReferenceBankMetadata:
    schema_version: int
    reference_id: str
    source_license: str
    source_sha256: str
    source_sha256_by_speed: tuple[str, ...]
    model_sha256: str
    controller_sha256: str
    npz: str
    npz_sha256: str
    joint_names: tuple[str, ...]
    frame: str
    units: str
    speed_knots_m_s: tuple[float, ...]
    cycle_period_s: tuple[float, ...]
    interpolation: str
    derivative_rule: str
    preprocessing_revision: str
    action_semantics: str

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = {field.name for field in fields(cls)}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("reference bank metadata fields do not match schema 1")
        converted = dict(value)
        for name in (
            "joint_names",
            "source_sha256_by_speed",
            "speed_knots_m_s",
            "cycle_period_s",
        ):
            raw = converted[name]
            if not isinstance(raw, list):
                raise ValueError(f"{name} must be a list")
            converted[name] = tuple(raw)
        metadata = cls(**converted)
        metadata.validate()
        return metadata

    def validate(self) -> None:
        if self.schema_version != 1 or not self.reference_id or not self.source_license:
            raise ValueError("reference bank identity is invalid")
        for value in (
            self.source_sha256,
            self.model_sha256,
            self.controller_sha256,
            self.npz_sha256,
            *self.source_sha256_by_speed,
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("reference bank hashes must be lowercase SHA-256")
        if Path(self.npz).name != self.npz or Path(self.npz).suffix != ".npz":
            raise ValueError("reference bank npz must be a sibling filename")
        if len(self.joint_names) != 29 or len(set(self.joint_names)) != 29:
            raise ValueError("reference bank requires 29 unique joints")
        if self.frame != "heading-local" or self.units != "SI":
            raise ValueError("reference bank requires heading-local SI conventions")
        if self.speed_knots_m_s != (0.4, 0.6, 0.8):
            raise ValueError("reference bank speed knots are frozen at 0.4/0.6/0.8 m/s")
        if len(self.cycle_period_s) != 3 or any(
            not math.isfinite(value) or value <= 0 for value in self.cycle_period_s
        ):
            raise ValueError("reference cycle periods must be finite and positive")
        if len(self.source_sha256_by_speed) != 3:
            raise ValueError("reference bank requires one source hash per speed knot")


class ReferenceBank:
    """Stateless immutable sampler; callers own phase and blend state."""

    _ARRAYS = {
        "phase",
        "joint_position_rad",
        "joint_partial_phase",
        "joint_partial_speed",
        "local_foot_position_m",
        "contact",
        "root_velocity_m_s",
        "nominal_joint_position_rad",
        "speed_knots_m_s",
        "cycle_period_s",
    }

    def __init__(self, metadata: ReferenceBankMetadata, arrays: dict[str, FloatArray]) -> None:
        self.metadata = metadata
        self.phase = arrays["phase"]
        self.joint_position = arrays["joint_position_rad"]
        self.joint_partial_phase = arrays["joint_partial_phase"]
        self.joint_partial_speed = arrays["joint_partial_speed"]
        self.local_foot_position = arrays["local_foot_position_m"]
        self.contact = arrays["contact"].astype(bool)
        self.root_velocity = arrays["root_velocity_m_s"]
        self.nominal_joint_position = arrays["nominal_joint_position_rad"]
        self.speed_knots = arrays["speed_knots_m_s"]
        self.cycle_period = arrays["cycle_period_s"]
        self._torch_cache: dict[tuple[int, tuple[int, ...], str, str], Any] = {}
        self._validate_arrays()

    @classmethod
    def load(cls, metadata_path: Path) -> Self:
        metadata = ReferenceBankMetadata.from_dict(
            json.loads(metadata_path.read_text(encoding="utf-8"))
        )
        npz_path = metadata_path.parent / metadata.npz
        if sha256_file(npz_path) != metadata.npz_sha256:
            raise ValueError("reference bank npz hash mismatch")
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != cls._ARRAYS:
                raise ValueError("reference bank arrays do not match schema")
            if any(archive[name].dtype.kind not in "buifc" for name in archive.files):
                raise ValueError("reference bank npz may contain only numeric arrays")
            arrays = {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}
        return cls(metadata, arrays)

    def _validate_arrays(self) -> None:
        frames = len(self.phase)
        expected = {
            "joint_position": (3, frames, 29),
            "joint_partial_phase": (3, frames, 29),
            "joint_partial_speed": (3, frames, 29),
            "local_foot_position": (3, frames, 2, 3),
            "contact": (3, frames, 2),
            "root_velocity": (3, frames, 3),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"reference bank {name} must be finite with shape {shape}")
        if self.nominal_joint_position.shape != (29,):
            raise ValueError("reference nominal pose must have shape (29,)")
        if not np.allclose(self.speed_knots, self.metadata.speed_knots_m_s):
            raise ValueError("reference speed knots differ from metadata")
        if not np.allclose(self.cycle_period, self.metadata.cycle_period_s):
            raise ValueError("reference periods differ from metadata")
        if frames < 4 or not np.allclose(self.phase, np.arange(frames) / frames):
            raise ValueError("reference phase grid must be uniform, periodic and half-open")

    def _coordinates(
        self, phase: FloatArray
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], FloatArray]:
        coordinate = np.mod(phase, 1.0) * len(self.phase)
        lower = np.floor(coordinate).astype(np.int64) % len(self.phase)
        upper = (lower + 1) % len(self.phase)
        return lower, upper, coordinate - np.floor(coordinate)

    def _phase_sample(self, values: FloatArray, phase: FloatArray) -> FloatArray:
        lower, upper, fraction = self._coordinates(phase)
        return (
            values[:, lower, ...] * (1 - fraction)[None, ..., None]
            + values[:, upper, ...] * fraction[None, ..., None]
        )

    def _speed_sample(
        self, values: FloatArray, derivatives: FloatArray, speed: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        low_domain = speed <= self.speed_knots[0]
        interval = np.clip(np.searchsorted(self.speed_knots, speed, side="right") - 1, 0, 1)
        left = self.speed_knots[interval]
        right = self.speed_knots[interval + 1]
        width = right - left
        blend = (speed - left) / width
        y0 = np.take_along_axis(values, interval[None, ..., None], axis=0)[0]
        y1 = np.take_along_axis(values, (interval + 1)[None, ..., None], axis=0)[0]
        m0 = np.take_along_axis(derivatives, interval[None, ..., None], axis=0)[0]
        m1 = np.take_along_axis(derivatives, (interval + 1)[None, ..., None], axis=0)[0]
        t = blend[..., None]
        result = (
            (2 * t**3 - 3 * t**2 + 1) * y0
            + (t**3 - 2 * t**2 + t) * width[..., None] * m0
            + (-2 * t**3 + 3 * t**2) * y1
            + (t**3 - t**2) * width[..., None] * m1
        )
        derivative = (
            (6 * t**2 - 6 * t) * y0 / width[..., None]
            + (3 * t**2 - 4 * t + 1) * m0
            + (-6 * t**2 + 6 * t) * y1 / width[..., None]
            + (3 * t**2 - 2 * t) * m1
        )
        result = np.where(low_domain[..., None], values[0], result)
        derivative = np.where(low_domain[..., None], 0.0, derivative)
        return result, derivative

    def sample_numpy(self, phase: npt.ArrayLike, speed_m_s: npt.ArrayLike) -> ReferenceSample:
        phase_array, speed = np.broadcast_arrays(
            np.asarray(phase, dtype=np.float64), np.asarray(speed_m_s, dtype=np.float64)
        )
        if not np.isfinite(phase_array).all() or not np.isfinite(speed).all():
            raise ValueError("reference phase and speed must be finite")
        if np.any(speed < 0) or np.any(speed > 0.8):
            raise ValueError("reference speed must be in [0,0.8] m/s")
        joint_by_speed = self._phase_sample(self.joint_position, phase_array)
        phase_derivative_by_speed = self._phase_sample(self.joint_partial_phase, phase_array)
        speed_derivative_by_speed = self._phase_sample(self.joint_partial_speed, phase_array)
        foot_by_speed = self._phase_sample(
            self.local_foot_position.reshape(3, len(self.phase), 6), phase_array
        )
        root_by_speed = self._phase_sample(self.root_velocity, phase_array)
        joint, joint_speed_derivative = self._speed_sample(
            joint_by_speed, speed_derivative_by_speed, speed
        )
        phase_derivative, _ = self._speed_sample(
            phase_derivative_by_speed, np.zeros_like(phase_derivative_by_speed), speed
        )
        foot, _ = self._speed_sample(foot_by_speed, np.zeros_like(foot_by_speed), speed)
        root, _ = self._speed_sample(root_by_speed, np.zeros_like(root_by_speed), speed)
        lower, _, fraction = self._coordinates(phase_array)
        speed_index = np.clip(np.searchsorted(self.speed_knots, speed), 0, 2)
        nearest = (lower + (fraction >= 0.5)) % len(self.phase)
        contact = self.contact[speed_index, nearest, :]
        base_frequency = 1.0 / self.cycle_period
        frequency = np.interp(speed, self.speed_knots, base_frequency)
        frequency = np.where(speed <= 0.4, speed / 0.4 * base_frequency[0], frequency)
        return ReferenceSample(
            joint,
            phase_derivative,
            joint_speed_derivative,
            foot.reshape((*phase_array.shape, 2, 3)),
            contact,
            root,
            frequency,
        )

    def sample_torch(self, phase: Any, speed_m_s: Any) -> TorchReferenceSample:
        import torch

        phase_array, speed = torch.broadcast_tensors(phase, speed_m_s)  # type: ignore[no-untyped-call]
        if not bool(torch.isfinite(phase_array).all()) or not bool(torch.isfinite(speed).all()):
            raise ValueError("reference phase and speed must be finite")
        if bool((speed < 0).any()) or bool((speed > 0.8).any()):
            raise ValueError("reference speed must be in [0,0.8] m/s")
        device, dtype = phase_array.device, phase_array.dtype

        def tensor(value: npt.NDArray[Any], *, target_dtype: Any = dtype) -> Any:
            pointer = int(value.__array_interface__["data"][0])
            key = (pointer, tuple(value.shape), str(device), str(target_dtype))
            cached = self._torch_cache.get(key)
            if cached is None:
                cached = torch.as_tensor(value, device=device, dtype=target_dtype)
                self._torch_cache[key] = cached
            return cached

        knots = tensor(self.speed_knots)
        coordinate = torch.remainder(phase_array, 1.0) * len(self.phase)
        lower = torch.floor(coordinate).long() % len(self.phase)
        upper = (lower + 1) % len(self.phase)
        fraction = coordinate - torch.floor(coordinate)

        def phase_sample(value: FloatArray) -> Any:
            data = tensor(value)
            return (
                data[:, lower, ...] * (1 - fraction)[None, ..., None]
                + data[:, upper, ...] * fraction[None, ..., None]
            )

        interval = torch.clamp(torch.searchsorted(knots, speed, right=True) - 1, 0, 1)
        left, right = knots[interval], knots[interval + 1]
        width = right - left
        blend = (speed - left) / width

        def speed_sample(value: Any, derivative: Any) -> tuple[Any, Any]:
            flat_interval = interval.reshape(-1)
            trailing = value.shape[len(phase_array.shape) + 1 :]
            flat_value = value.reshape((3, -1, *trailing))
            row = torch.arange(flat_interval.numel(), device=device)
            y0 = flat_value[flat_interval, row].reshape((*phase_array.shape, *trailing))
            y1 = flat_value[flat_interval + 1, row].reshape((*phase_array.shape, *trailing))
            flat_derivative = derivative.reshape((3, -1, *trailing))
            m0 = flat_derivative[flat_interval, row].reshape((*phase_array.shape, *trailing))
            m1 = flat_derivative[flat_interval + 1, row].reshape((*phase_array.shape, *trailing))
            expand = (None,) * len(trailing)
            t, h = blend[(..., *expand)], width[(..., *expand)]
            result = (
                (2 * t**3 - 3 * t**2 + 1) * y0
                + (t**3 - 2 * t**2 + t) * h * m0
                + (-2 * t**3 + 3 * t**2) * y1
                + (t**3 - t**2) * h * m1
            )
            gradient = (
                (6 * t**2 - 6 * t) * y0 / h
                + (3 * t**2 - 4 * t + 1) * m0
                + (-6 * t**2 + 6 * t) * y1 / h
                + (3 * t**2 - 2 * t) * m1
            )
            low = (speed <= 0.4)[(..., *expand)]
            first = value[0]
            return torch.where(low, first, result), torch.where(low, 0.0, gradient)

        joint_phase = phase_sample(self.joint_position)
        phase_partial = phase_sample(self.joint_partial_phase)
        speed_partial = phase_sample(self.joint_partial_speed)
        foot_phase = phase_sample(self.local_foot_position.reshape(3, len(self.phase), 6))
        root_phase = phase_sample(self.root_velocity)
        joint, joint_speed = speed_sample(joint_phase, speed_partial)
        joint_phase_partial, _ = speed_sample(phase_partial, torch.zeros_like(phase_partial))
        foot, _ = speed_sample(foot_phase, torch.zeros_like(foot_phase))
        root, _ = speed_sample(root_phase, torch.zeros_like(root_phase))
        speed_index = torch.clamp(torch.searchsorted(knots, speed), 0, 2)
        contact_data = tensor(self.contact, target_dtype=torch.bool)
        nearest = (lower + (fraction >= 0.5).long()) % len(self.phase)
        contact = contact_data[speed_index, nearest]
        periods = tensor(self.cycle_period)
        frequency = torch.lerp(1 / periods[interval], 1 / periods[interval + 1], blend)
        frequency = torch.where(speed <= 0.4, speed / 0.4 / periods[0], frequency)
        return TorchReferenceSample(
            joint,
            joint_phase_partial,
            joint_speed,
            foot.reshape((*phase_array.shape, 2, 3)),
            contact,
            root,
            frequency,
        )

    def compose_targets_numpy(
        self,
        phase: npt.ArrayLike,
        speed_m_s: npt.ArrayLike,
        *,
        blend: npt.ArrayLike,
        blend_rate_s: npt.ArrayLike,
        speed_rate_m_s2: npt.ArrayLike,
    ) -> ReferenceTargets:
        sample = self.sample_numpy(phase, speed_m_s)
        blend_array, blend_rate, speed_rate = np.broadcast_arrays(
            np.asarray(blend, dtype=np.float64),
            np.asarray(blend_rate_s, dtype=np.float64),
            np.asarray(speed_rate_m_s2, dtype=np.float64),
        )
        if np.any(blend_array < 0) or np.any(blend_array > 1):
            raise ValueError("reference blend must be in [0,1]")
        position = self.nominal_joint_position + blend_array[..., None] * (
            sample.joint_position - self.nominal_joint_position
        )
        velocity = blend_rate[..., None] * (
            sample.joint_position - self.nominal_joint_position
        ) + blend_array[..., None] * (
            sample.joint_partial_phase * sample.phase_rate_hz[..., None]
            + sample.joint_partial_speed * speed_rate[..., None]
        )
        return ReferenceTargets(position, velocity)


def build_reference_bank(
    source_paths: tuple[Path, Path, Path],
    metadata_path: Path,
    *,
    reference_id: str,
    source_license: str,
    model_sha256: str,
    controller_sha256: str,
    joint_names: tuple[str, ...],
    nominal_joint_position: npt.ArrayLike,
    local_foot_position: tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike],
) -> ReferenceBankMetadata:
    joints: list[FloatArray] = []
    joint_velocities: list[FloatArray] = []
    contacts_list: list[FloatArray] = []
    periods_list: list[float] = []
    for source_path in source_paths:
        with np.load(source_path, allow_pickle=False) as source:
            joint = np.asarray(source["joint_pos"], dtype=np.float64)[:-1]
            joints.append(joint)
            joint_velocities.append(np.asarray(source["joint_vel"], dtype=np.float64)[:-1])
            contacts_list.append(np.asarray(source["foot_contact"], dtype=np.float64)[:-1])
            fps = float(np.asarray(source["fps"]).reshape(-1)[0])
            periods_list.append(len(joint) / fps)
    joint_position = np.stack(joints)
    joint_velocity = np.stack(joint_velocities)
    contacts = np.stack(contacts_list)
    foot_position = np.stack(
        [np.asarray(value, dtype=np.float64)[:-1] for value in local_foot_position]
    )
    nominal = np.asarray(nominal_joint_position, dtype=np.float64)
    if (
        joint_position.ndim != 3
        or joint_position.shape[0] != 3
        or joint_position.shape[2] != 29
        or foot_position.shape != (3, joint_position.shape[1], 2, 3)
        or contacts.shape != (3, joint_position.shape[1], 2)
    ):
        raise ValueError("source reference arrays do not match the G1 bank schema")
    speed_knots = np.array([0.4, 0.6, 0.8], dtype=np.float64)
    periods = np.asarray(periods_list, dtype=np.float64)
    phase = np.arange(joint_position.shape[1], dtype=np.float64) / joint_position.shape[1]
    joint_partial_phase = joint_velocity * periods[:, None, None]
    joint_partial_speed = np.gradient(joint_position, speed_knots, axis=0)
    joint_partial_speed[0] = 0.0
    root_velocity = np.zeros((3, joint_position.shape[1], 3), dtype=np.float64)
    root_velocity[:, :, 0] = speed_knots[:, None]
    npz_path = metadata_path.with_suffix(".npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        phase=phase,
        joint_position_rad=joint_position,
        joint_partial_phase=joint_partial_phase,
        joint_partial_speed=joint_partial_speed,
        local_foot_position_m=foot_position,
        contact=contacts,
        root_velocity_m_s=root_velocity,
        nominal_joint_position_rad=nominal,
        speed_knots_m_s=speed_knots,
        cycle_period_s=periods,
    )
    source_hashes = tuple(sha256_file(path) for path in source_paths)
    aggregate_hash = hashlib.sha256("".join(source_hashes).encode("ascii")).hexdigest()
    metadata = ReferenceBankMetadata(
        schema_version=1,
        reference_id=reference_id,
        source_license=source_license,
        source_sha256=aggregate_hash,
        source_sha256_by_speed=source_hashes,
        model_sha256=model_sha256,
        controller_sha256=controller_sha256,
        npz=npz_path.name,
        npz_sha256=sha256_file(npz_path),
        joint_names=joint_names,
        frame="heading-local",
        units="SI",
        speed_knots_m_s=tuple(float(value) for value in speed_knots),
        cycle_period_s=tuple(float(value) for value in periods),
        interpolation="periodic-linear-phase-c1-cubic-hermite-speed",
        derivative_rule="analytic-composed-target-v1",
        preprocessing_revision="plan05-p04-time-warp-v1",
        action_semantics="reference-centered-residual",
    )
    metadata.validate()
    write_atomic_json(metadata_path, asdict(metadata))
    return metadata
