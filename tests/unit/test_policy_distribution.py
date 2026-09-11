from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from g1_mjlab.artifacts import sha256_file
from g1_mjlab.policy_distribution import install_policy, validate_policy
from g1_mjlab.qualification import canonical_hash


def _policy_archive(tmp_path: Path, *, unsafe_member: str | None = None) -> Path:
    source = tmp_path / "source"
    (source / "checkpoints").mkdir(parents=True)
    (source / "LICENSES").mkdir()
    model = source / "model.mjb"
    onnx = source / "checkpoints" / "policy.onnx"
    model.write_bytes(b"model")
    onnx.write_bytes(b"onnx")
    contract = {"schema_version": 2, "joint_names": ["joint"]}
    contract["sha256"] = canonical_hash(contract)
    (source / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    bundle = {
        "schema_version": 1,
        "contract_sha256": contract["sha256"],
        "model_sha256": sha256_file(model),
        "onnx_sha256": sha256_file(onnx),
        "checkpoint_sha256": "checkpoint",
    }
    (source / "policy-bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    (source / "scenarios.json").write_text(
        json.dumps({"schema_version": 2, "checkpoint_sha256": "checkpoint"}), encoding="utf-8"
    )
    (source / "README.md").write_text("policy", encoding="utf-8")
    (source / "LICENSE").write_text("license", encoding="utf-8")
    (source / "THIRD_PARTY_NOTICES.md").write_text("notices", encoding="utf-8")
    (source / "LICENSES" / "UNITREE_G1_LICENSE.txt").write_text("license", encoding="utf-8")
    archive_path = tmp_path / "policy.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for file in sorted(path for path in source.rglob("*") if path.is_file()):
            archive.write(file, file.relative_to(source).as_posix())
        if unsafe_member:
            archive.writestr(unsafe_member, "unsafe")
    return archive_path


def test_install_policy_verifies_and_extracts_atomically(tmp_path: Path) -> None:
    archive = _policy_archive(tmp_path)
    destination = tmp_path / "installed"

    result = install_policy(
        "standing-v1",
        destination,
        archive_url=archive.as_uri(),
        archive_sha256=sha256_file(archive),
    )

    assert result["status"] == "installed"
    assert result["verified"] is True
    assert validate_policy(destination)["verified"] is True
    assert install_policy("standing-v1", destination)["status"] == "already-installed"


def test_install_policy_rejects_hash_mismatch_without_destination(tmp_path: Path) -> None:
    archive = _policy_archive(tmp_path)
    destination = tmp_path / "installed"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        install_policy(
            "standing-v1",
            destination,
            archive_url=archive.as_uri(),
            archive_sha256="0" * 64,
        )

    assert not destination.exists()


@pytest.mark.parametrize("member", ["../escape", "/absolute", "C:/escape"])
def test_install_policy_rejects_unsafe_archive_members(tmp_path: Path, member: str) -> None:
    archive = _policy_archive(tmp_path, unsafe_member=member)

    with pytest.raises(ValueError, match="unsafe archive member"):
        install_policy(
            "standing-v1",
            tmp_path / "installed",
            archive_url=archive.as_uri(),
            archive_sha256=sha256_file(archive),
        )
