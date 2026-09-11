"""CPU-only registry for task identity, ownership, and runtime capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from types import ModuleType
from typing import Any

STANDING_TASK_ID = "G1-Standing-Flat-v1"
WALKING_TASK_ID = "G1-Walking-Flat-v1"


class UnknownTaskError(ValueError):
    """Raised before simulator startup when a task identity is not registered."""


class TaskCapabilityError(ValueError):
    """Raised before simulator startup when a workflow is unavailable for a task."""


class TaskCapability(StrEnum):
    TRAIN = "train"
    STANDING_EVALUATION = "standing_evaluation"
    NATIVE_INFERENCE = "native_inference"
    CONTROLLER_QUALIFICATION = "controller_qualification"


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """Stable metadata plus lazy hooks for one versioned learning problem."""

    task_id: str
    config_directory: str
    experiment_name: str
    implementation_module: str
    layout_id: str
    actor_size: int
    critic_size: int
    capabilities: frozenset[TaskCapability]
    unavailable_reason: str | None = None
    termination_penalty: float | None = None

    def supports(self, capability: TaskCapability) -> bool:
        return capability in self.capabilities

    def require(self, capability: TaskCapability) -> TaskDefinition:
        if not self.supports(capability):
            detail = self.unavailable_reason or "the workflow has not been implemented"
            raise TaskCapabilityError(
                f"Task {self.task_id!r} does not support {capability.value!r}: {detail}."
            )
        return self

    def _implementation(self) -> ModuleType:
        return import_module(self.implementation_module)

    def register(self) -> None:
        self._implementation().register_task()

    def configure_environment(self, env: Any, *, randomized_reset: bool = True) -> None:
        self._implementation().configure_environment(
            env,
            randomized_reset=randomized_reset,
        )


_TASKS = {
    STANDING_TASK_ID: TaskDefinition(
        task_id=STANDING_TASK_ID,
        config_directory="standing-v1",
        experiment_name="g1_standing",
        implementation_module="g1_mjlab.tasks.standing_v1",
        layout_id="g1-standing-actor-v1",
        actor_size=99,
        critic_size=111,
        capabilities=frozenset(
            {
                TaskCapability.TRAIN,
                TaskCapability.STANDING_EVALUATION,
                TaskCapability.NATIVE_INFERENCE,
                TaskCapability.CONTROLLER_QUALIFICATION,
            }
        ),
        termination_penalty=-5.0,
    ),
    WALKING_TASK_ID: TaskDefinition(
        task_id=WALKING_TASK_ID,
        config_directory="walking-v1",
        experiment_name="g1_walking",
        implementation_module="g1_mjlab.tasks.walking_v1",
        layout_id="g1-walking-actor-v1",
        actor_size=102,
        critic_size=114,
        capabilities=frozenset({TaskCapability.TRAIN}),
        unavailable_reason=(
            "walking evaluation, native inference, and controller qualification are not yet implemented"
        ),
        termination_penalty=-10.0,
    ),
}


def get_task(task_id: str) -> TaskDefinition:
    """Resolve a task without importing mjlab, Torch, CUDA, or task implementations."""
    try:
        return _TASKS[task_id]
    except KeyError as error:
        known = ", ".join(sorted(_TASKS))
        raise UnknownTaskError(f"Unknown task {task_id!r}; registered tasks: {known}.") from error
