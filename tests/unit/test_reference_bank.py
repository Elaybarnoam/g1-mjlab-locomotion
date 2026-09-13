from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from g1_mjlab.motion.reference_bank import ReferenceBank, build_reference_bank


def _source(path: Path) -> None:
    phase = np.arange(8, dtype=np.float64) / 8
    joint = np.sin(2 * np.pi * phase)[:, None] * np.linspace(0.1, 0.3, 29)
    np.savez_compressed(
        path,
        phase=phase,
        joint_position_rad=np.stack([joint] * 3),
        joint_partial_phase=np.stack(
            [np.cos(2 * np.pi * phase)[:, None] * 2 * np.pi * np.linspace(0.1, 0.3, 29)]
            * 3
        ),
        joint_partial_speed=np.zeros((3, 8, 29)),
        local_foot_position_m=np.zeros((3, 8, 2, 3)),
        contact=np.ones((3, 8, 2), dtype=np.uint8),
        root_velocity_m_s=np.zeros((3, 8, 3)),
        nominal_joint_position_rad=np.zeros(29),
        speed_knots_m_s=np.array([0.4, 0.6, 0.8]),
        cycle_period_s=np.array([2.0, 1.5, 1.0]),
    )


def _metadata(path: Path, npz: Path) -> None:
    import hashlib

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reference_id": "test-bank",
                "source_license": "test-only",
                "source_sha256": "a" * 64,
                "source_sha256_by_speed": ["a" * 64, "a" * 64, "a" * 64],
                "model_sha256": "b" * 64,
                "controller_sha256": "c" * 64,
                "npz": npz.name,
                "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                "joint_names": [f"joint-{index}" for index in range(29)],
                "frame": "heading-local",
                "units": "SI",
                "speed_knots_m_s": [0.4, 0.6, 0.8],
                "cycle_period_s": [2.0, 1.5, 1.0],
                "interpolation": "periodic-linear-phase-c1-cubic-hermite-speed",
                "derivative_rule": "analytic-composed-target-v1",
                "preprocessing_revision": "test-v1",
                "action_semantics": "reference-centered-residual",
            }
        ),
        encoding="utf-8",
    )


def test_reference_bank_numpy_torch_parity_and_arbitrary_batch(tmp_path: Path) -> None:
    npz = tmp_path / "bank.npz"
    metadata = tmp_path / "bank.json"
    _source(npz)
    _metadata(metadata, npz)
    bank = ReferenceBank.load(metadata)
    phase = np.array([[0.0, 0.125], [0.99, 0.5]])
    speed = np.array([[0.4, 0.5], [0.7, 0.8]])

    numpy_sample = bank.sample_numpy(phase, speed)
    torch_sample = bank.sample_torch(torch.tensor(phase), torch.tensor(speed))

    assert numpy_sample.joint_position.shape == (2, 2, 29)
    np.testing.assert_allclose(torch_sample.joint_position.numpy(), numpy_sample.joint_position)
    np.testing.assert_allclose(torch_sample.joint_partial_phase.numpy(), numpy_sample.joint_partial_phase)
    np.testing.assert_array_equal(torch_sample.contact.numpy(), numpy_sample.contact)


def test_reference_targets_stand_exactly_and_include_blend_derivative(tmp_path: Path) -> None:
    npz = tmp_path / "bank.npz"
    metadata = tmp_path / "bank.json"
    _source(npz)
    _metadata(metadata, npz)
    bank = ReferenceBank.load(metadata)

    standing = bank.compose_targets_numpy(
        np.array([0.25]),
        np.array([0.0]),
        blend=np.array([0.0]),
        blend_rate_s=np.array([0.0]),
        speed_rate_m_s2=np.array([0.0]),
    )
    transitioning = bank.compose_targets_numpy(
        np.array([0.25]),
        np.array([0.4]),
        blend=np.array([0.5]),
        blend_rate_s=np.array([1.0]),
        speed_rate_m_s2=np.array([0.0]),
    )

    np.testing.assert_allclose(standing.joint_position, 0.0)
    np.testing.assert_allclose(standing.joint_velocity, 0.0)
    assert np.linalg.norm(transitioning.joint_velocity) > 0


def test_reference_bank_rejects_out_of_domain_or_tampered_data(tmp_path: Path) -> None:
    npz = tmp_path / "bank.npz"
    metadata = tmp_path / "bank.json"
    _source(npz)
    _metadata(metadata, npz)
    bank = ReferenceBank.load(metadata)

    with pytest.raises(ValueError, match="speed"):
        bank.sample_numpy(np.array([0.0]), np.array([0.81]))
    npz.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        ReferenceBank.load(metadata)


def test_build_reference_bank_keeps_speed_specific_shapes_and_numeric_npz(tmp_path: Path) -> None:
    frames = 9
    joint = np.zeros((frames, 29))
    joint[:, 0] = np.sin(np.linspace(0, 2 * np.pi, frames))
    sources = tuple(tmp_path / f"source-{index}.npz" for index in range(3))
    for index, source in enumerate(sources):
        np.savez_compressed(
            source,
            joint_pos=joint + index * 0.01,
            joint_vel=np.gradient(joint, 0.02, axis=0),
            foot_contact=np.tile([[1, 0], [1, 1], [0, 1]], (3, 1))[:frames],
            fps=np.array([50.0 + index]),
        )
    metadata = tmp_path / "bank.json"

    build_reference_bank(
        sources,
        metadata,
        reference_id="test",
        source_license="test-only",
        model_sha256="b" * 64,
        controller_sha256="c" * 64,
        joint_names=tuple(f"joint-{index}" for index in range(29)),
        nominal_joint_position=np.zeros(29),
        local_foot_position=tuple(np.zeros((frames, 2, 3)) for _ in range(3)),
    )

    bank = ReferenceBank.load(metadata)
    assert bank.joint_position.shape == (3, 8, 29)
    assert len(bank.metadata.source_sha256_by_speed) == 3
    assert not np.allclose(bank.joint_position[0], bank.joint_position[2])
    with np.load(metadata.with_suffix(".npz"), allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind in "buifc" for name in archive.files)
