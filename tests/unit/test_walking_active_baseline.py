"""Guard the selected unqualified walking development baseline."""

from __future__ import annotations

import json
from pathlib import Path

from g1_mjlab.artifacts import sha256_file


def test_active_walking_baseline_is_hash_bound_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[2]
    baseline = json.loads(
        (root / "configs/walking-v1/active-development-baseline.json").read_text(
            encoding="utf-8"
        )
    )

    assert baseline["task_id"] == "G1-Walking-Flat-v1"
    assert baseline["role"] == "active_development_baseline"
    assert baseline["status"] == "unqualified_development"
    assert baseline["policy"]["action_center"] == "nominal_joint_position"
    assert baseline["qualification_claim"] is False
    assert baseline["release_authorized"] is False
    assert baseline["walking_v2_role"] == "experimental_history_only"
    replay = baseline["evidence"]["deterministic_replay"]
    assert sha256_file(root / replay["file"]) == replay["sha256"]
    assert baseline["evidence"]["later_strict_measurement"]["functional_passed"] == 0
