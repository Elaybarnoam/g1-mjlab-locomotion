"""Versioned task identities and lazy runtime dispatch."""

from .registry import (
    STANDING_TASK_ID,
    WALKING_TASK_ID,
    TaskCapability,
    TaskCapabilityError,
    TaskDefinition,
    UnknownTaskError,
    get_task,
)

__all__ = [
    "STANDING_TASK_ID",
    "WALKING_TASK_ID",
    "TaskCapability",
    "TaskCapabilityError",
    "TaskDefinition",
    "UnknownTaskError",
    "get_task",
]
