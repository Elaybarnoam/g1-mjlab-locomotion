"""Serializable robot and policy contracts shared by training and reports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VectorField:
    name: str
    offset: int
    size: int
    unit: str
    frame: str
    source: str


@dataclass(frozen=True)
class PolicyContract:
    schema_version: int
    robot: str
    model_revision: str
    joint_names: tuple[str, ...]
    actor_fields: tuple[VectorField, ...]
    critic_fields: tuple[VectorField, ...]
    action_names: tuple[str, ...]
    physics_dt: float
    control_dt: float
    action_semantics: str
    actuator_semantics: str
    task_id: str = ""
    layout_id: str = ""
    command_semantics: str = ""
    phase_semantics: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def fields_from_sizes(specs: Iterable[tuple[str, int, str, str, str]]) -> tuple[VectorField, ...]:
    result: list[VectorField] = []
    offset = 0
    for name, size, unit, frame, source in specs:
        if size <= 0:
            raise ValueError(f"field {name!r} must have positive size")
        result.append(VectorField(name, offset, size, unit, frame, source))
        offset += size
    return tuple(result)


def validate_contract(contract: PolicyContract) -> None:
    if contract.schema_version != 1:
        raise ValueError("unsupported contract schema")
    if len(set(contract.joint_names)) != len(contract.joint_names):
        raise ValueError("joint names must be unique")
    if len(contract.action_names) != len(set(contract.action_names)):
        raise ValueError("action names must be unique")
    if set(contract.action_names) - set(contract.joint_names):
        raise ValueError("every action must reference a model joint")
    if bool(contract.task_id) != bool(contract.layout_id):
        raise ValueError("task_id and layout_id must either both be set or both be empty")
    for fields in (contract.actor_fields, contract.critic_fields):
        expected = 0
        for field in fields:
            if field.offset != expected:
                raise ValueError(f"non-contiguous vector field {field.name!r}")
            expected += field.size
