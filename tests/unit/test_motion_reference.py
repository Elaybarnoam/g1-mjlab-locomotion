from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from g1_mjlab.motion import (
    G1_JOINT_NAMES,
    MotionManifest,
    MotionValidationError,
    audit_periodic_motion,
    derive_foot_contacts,
    load_soma_csv,
    resample_motion,
)


def _write_soma_csv(path: Path, rows: list[list[float]]) -> None:
    columns = [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
        *(f"{name}_dof" for name in G1_JOINT_NAMES),
    ]
    path.write_text(
        ",".join(columns) + "\n" + "\n".join(",".join(map(str, row)) for row in rows),
        encoding="utf-8",
    )


def test_soma_csv_converts_declared_units_and_joint_order(tmp_path: Path) -> None:
    csv_path = tmp_path / "motion.csv"
    row0 = [0, 100, 200, 80, 0, 0, 90, *range(29)]
    row1 = [1, 101, 200, 80, 0, 0, 90, *range(1, 30)]
    _write_soma_csv(csv_path, [row0, row1])

    motion = load_soma_csv(csv_path, fps=120.0)

    np.testing.assert_allclose(motion.root_position_m[0], [1.0, 2.0, 0.8])
    np.testing.assert_allclose(motion.root_quaternion_wxyz[0], [2**-0.5, 0, 0, 2**-0.5])
    np.testing.assert_allclose(motion.joint_position_rad[0], np.deg2rad(range(29)))
    assert motion.joint_names == G1_JOINT_NAMES
    assert motion.fps == 120.0


def test_soma_csv_rejects_reordered_or_missing_joint_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "motion.csv"
    _write_soma_csv(csv_path, [[0, 0, 0, 80, 0, 0, 0, *range(29)]])
    text = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(
        text.replace(
            "left_hip_pitch_joint_dof,left_hip_roll_joint_dof",
            "left_hip_roll_joint_dof,left_hip_pitch_joint_dof",
        ),
        encoding="utf-8",
    )

    with pytest.raises(MotionValidationError, match="joint column order"):
        load_soma_csv(csv_path, fps=120.0)


def test_resampling_preserves_endpoints_and_normalizes_quaternions(tmp_path: Path) -> None:
    csv_path = tmp_path / "motion.csv"
    _write_soma_csv(
        csv_path,
        [
            [0, 0, 0, 80, 0, 0, 0, *([0] * 29)],
            [1, 2, 0, 80, 0, 0, 9, *([9] * 29)],
            [2, 4, 0, 80, 0, 0, 18, *([18] * 29)],
        ],
    )
    source = load_soma_csv(csv_path, fps=2.0)

    sampled = resample_motion(source, output_fps=4.0)

    assert sampled.frame_count == 5
    np.testing.assert_allclose(sampled.root_position_m[[0, -1]], source.root_position_m[[0, -1]])
    np.testing.assert_allclose(np.linalg.norm(sampled.root_quaternion_wxyz, axis=1), 1.0)


def test_contact_derivation_uses_hysteresis_and_minimum_duration() -> None:
    height = np.array([0.01, 0.01, 0.04, 0.01, 0.01, 0.01])
    speed = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

    contacts = derive_foot_contacts(
        height,
        speed,
        enter_height_m=0.02,
        exit_height_m=0.03,
        max_stance_speed_m_s=0.2,
        minimum_frames=2,
    )

    np.testing.assert_array_equal(contacts, [True, True, True, True, True, True])


def test_periodic_audit_removes_forward_root_displacement_from_seam() -> None:
    joint = np.zeros((5, 29))
    joint[-1] = 0.01
    root = np.zeros((5, 3))
    root[:, 0] = np.linspace(0, 0.8, 5)
    root_quat = np.zeros((5, 4))
    root_quat[:, 0] = 1.0
    contacts = np.array([[True, False], [True, False], [False, True], [False, True], [True, False]])

    audit = audit_periodic_motion(joint, root, root_quat, contacts, fps=50.0)

    assert audit.duration_s == pytest.approx(0.08)
    assert audit.forward_speed_m_s == pytest.approx(10.0)
    assert audit.root_orientation_seam_rad == pytest.approx(0.0)
    assert audit.joint_pose_seam_rms_rad == pytest.approx(0.01)
    assert audit.contact_seam_matches


def test_manifest_is_hash_bound_and_rejects_tampering(tmp_path: Path) -> None:
    asset = tmp_path / "motion.npz"
    asset.write_bytes(b"motion")
    import hashlib

    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reference_id": "test-walk-v1",
                "redistribution": "approved-derivative",
                "assets": [{"path": "motion.npz", "sha256": digest, "bytes": 6}],
            }
        ),
        encoding="utf-8",
    )

    manifest = MotionManifest.load(manifest_path)
    manifest.verify_assets(tmp_path)
    asset.write_bytes(b"notion")

    with pytest.raises(MotionValidationError, match="SHA-256"):
        manifest.verify_assets(tmp_path)


def test_public_walking_reference_matches_manifest_and_audit() -> None:
    root = Path(__file__).parents[2]
    config = root / "configs" / "walking-v1"
    manifest = MotionManifest.load(config / "motion-manifest.json")

    manifest.verify_assets(config)
    with np.load(config / manifest.assets[0].path, allow_pickle=False) as reference:
        assert reference["joint_pos"].shape == (54, 29)
        assert reference["body_pos_w"].shape == (54, 30, 3)
        assert reference["foot_contact"].shape == (54, 2)
        assert tuple(reference["joint_names"].tolist()) == G1_JOINT_NAMES
        assert np.all(np.isfinite(reference["joint_vel"]))
    audit = json.loads((config / manifest.raw["audit_path"]).read_text(encoding="utf-8"))
    assert audit["acceptance"]["passed"] is True
    assert audit["npz_sha256"] == manifest.assets[0].sha256
