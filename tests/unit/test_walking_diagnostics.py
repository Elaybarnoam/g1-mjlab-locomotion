from __future__ import annotations

import pytest

from g1_mjlab.walking_diagnostics import select_parallelism


def test_parallelism_selects_fastest_repeatable_batch_with_headroom() -> None:
    results = [
        {"num_envs": 64, "stable": True, "steps_per_second": 1000, "minimum_free_ratio": 0.6},
        {"num_envs": 64, "stable": True, "steps_per_second": 980, "minimum_free_ratio": 0.59},
        {"num_envs": 128, "stable": True, "steps_per_second": 1700, "minimum_free_ratio": 0.3},
        {"num_envs": 128, "stable": True, "steps_per_second": 1680, "minimum_free_ratio": 0.29},
        {"num_envs": 256, "stable": True, "steps_per_second": 1900, "minimum_free_ratio": 0.15},
        {"num_envs": 256, "stable": True, "steps_per_second": 1880, "minimum_free_ratio": 0.14},
    ]

    selection = select_parallelism(results, repeats=2, minimum_free_ratio=0.2)

    assert selection["selected_num_envs"] == 128
    assert selection["reason"] == "fastest repeatable stable batch with required memory headroom"


def test_parallelism_requires_every_repeat_to_be_stable() -> None:
    results = [
        {"num_envs": 16, "stable": True, "steps_per_second": 100, "minimum_free_ratio": 0.8},
        {"num_envs": 16, "stable": False, "steps_per_second": 0, "minimum_free_ratio": 0.8},
    ]

    with pytest.raises(ValueError, match="No environment count"):
        select_parallelism(results, repeats=2, minimum_free_ratio=0.2)
