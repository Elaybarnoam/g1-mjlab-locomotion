from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_mjlab.walking_v2_resources import qualify_resource_profile


def _probe(root: Path, count: int, fps: tuple[float, float]) -> tuple[int, Path, Path]:
    run, evaluation = root / f"run-{count}", root / f"eval-{count}"
    (run / "metrics").mkdir(parents=True)
    evaluation.mkdir()
    values = {
        "manifest.json": {"status": "completed", "wall_seconds": 10.0},
        "config.json": {"num_envs": count},
        "optimization.json": {
            "all_kl_samples_finite": True,
            "all_recorded_gradients_finite": True,
            "total_transitions": count * 72,
        },
        "memory.json": {
            "device_total_bytes": 1000,
            "device_free_at_end_bytes": 800,
            "torch_peak_allocated_bytes": 100,
            "torch_peak_reserved_bytes": 120,
        },
    }
    for name, value in values.items():
        (run / name).write_text(json.dumps(value), encoding="utf-8")
    lines = [
        {"metric": "Perf/total_fps", "update": 0, "value": 1},
        {"metric": "Perf/total_fps", "update": 1, "value": fps[0]},
        {"metric": "Perf/total_fps", "update": 2, "value": fps[1]},
    ]
    (run / "metrics/metrics.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in lines), encoding="utf-8"
    )
    (evaluation / "summary.json").write_text(
        json.dumps(
            {
                "recorded_steps": 25,
                "planned_steps": 25,
                "wall_seconds_including_startup": 2.0,
            }
        ),
        encoding="utf-8",
    )
    return count, run, evaluation


def test_resource_profile_selects_sustained_throughput(tmp_path: Path) -> None:
    probes = [
        _probe(tmp_path, 64, (1000, 1100)),
        _probe(tmp_path, 128, (2000, 2200)),
        _probe(tmp_path, 256, (4000, 4200)),
    ]

    result = qualify_resource_profile(probes, tmp_path / "profile.json")

    assert result["selected_num_envs"] == 256
    assert result["transitions_per_update"] == 6144
    assert result["estimated_4000_update_wall_seconds"]["lower"] > 0


def test_resource_profile_rejects_reordered_candidates(tmp_path: Path) -> None:
    probes = [
        _probe(tmp_path, 128, (1, 1)),
        _probe(tmp_path, 64, (1, 1)),
        _probe(tmp_path, 256, (1, 1)),
    ]

    with pytest.raises(ValueError, match="frozen"):
        qualify_resource_profile(probes, tmp_path / "profile.json")
