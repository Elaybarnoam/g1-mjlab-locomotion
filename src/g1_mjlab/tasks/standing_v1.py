"""Standing-v1 hooks exposed through the task registry."""

from __future__ import annotations

from typing import Any

from ..standing_task import configure_stance, register_standing_task


def configure_environment(env: Any, *, randomized_reset: bool = True) -> None:
    configure_stance(env, randomized_reset=randomized_reset)


def register_task() -> None:
    register_standing_task()
