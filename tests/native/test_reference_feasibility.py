from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from g1_mjlab.motion.reference_feasibility import (
    audit_reference_speed_map,
    load_reference_speed_map,
)


def test_reference_audit_freezes_time_warp_cadence_and_hashes() -> None:
    root = Path(__file__).parents[2]
    models = list(
        Path.home().glob(
            "AppData/Local/uv/cache/git-v0/checkouts/*/8ee51fb/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
        )
    )
    controller = root / ".runtime/walking-bootstrap-001/contract.json"
    if not models or not controller.exists():
        pytest.skip("pinned local mjlab model and baseline controller evidence are required")
    model = models[0]
    result = audit_reference_speed_map(
        root / "configs/walking-v1/reference/g1-walk-a057-cycle.npz",
        model,
        controller,
        reference_id="nvidia-soma-g1-neutral-walk-a057-cycle-v1",
        source_speed_m_s=1.16381159304071,
        cycle_duration_s=1.06,
    )
    assert (
        result.reference_sha256
        == "028eae4a2a8163dae515891cb31a8f015b9e47e2f3e13d84669dd3fa7e91d06e"
    )
    assert result.model_sha256 == "febdcbeffbbf84051556ae41a5ac1b43fb479a5d76bdb3f54824dbc2721c20aa"
    assert [entry.expected_cadence_steps_s for entry in result.entries] == pytest.approx(
        [0.6485, 0.9728, 1.2971], abs=5e-4
    )
    assert all(entry.passed for entry in result.entries)
    assert result.supported_speed_range_m_s == (0.4, 0.8)
    assert min(entry.sole_clearance_range_m[0] for entry in result.entries) > 0
    assert result.passed


def test_committed_reference_map_loads_strictly() -> None:
    root = Path(__file__).parents[2]
    result = load_reference_speed_map(root / "configs/walking-v1/reference-map-v2.json")
    assert result.schema_version == 2
    assert result.supported_speed_range_m_s == (0.4, 0.8)
