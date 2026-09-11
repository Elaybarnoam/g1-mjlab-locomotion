"""Install and verify published policy bundles without importing simulator dependencies."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import sha256_file
from .qualification import canonical_hash

MAX_DOWNLOAD_BYTES = 150 * 1024 * 1024
MAX_EXTRACTED_BYTES = 200 * 1024 * 1024
REQUIRED_FILES = frozenset(
    {
        "LICENSE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "LICENSES/UNITREE_G1_LICENSE.txt",
        "checkpoints/policy.onnx",
        "contract.json",
        "model.mjb",
        "policy-bundle.json",
        "scenarios.json",
    }
)


@dataclass(frozen=True)
class PublishedPolicy:
    """Immutable metadata for a downloadable policy artifact."""

    name: str
    version: str
    archive_url: str
    archive_sha256: str


POLICIES = {
    "standing-v1": PublishedPolicy(
        name="standing-v1",
        version="1.0.0",
        archive_url=(
            "https://github.com/Elaybarnoam/g1-mjlab-locomotion/releases/download/"
            "v0.2.0/standing-v1-inference.zip"
        ),
        archive_sha256="4c7208ce29dae1106b031a06e37416f56c232054cc33a937f900e9a85662df92",
    )
}


def get_policy(name: str) -> PublishedPolicy:
    try:
        return POLICIES[name]
    except KeyError as error:
        supported = ", ".join(sorted(POLICIES))
        raise ValueError(f"unknown policy {name!r}; supported policies: {supported}") from error


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    names: set[str] = set()
    extracted_bytes = 0
    for member in members:
        path = PurePosixPath(member.filename)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in member.filename
            or (path.parts and ":" in path.parts[0])
        ):
            raise ValueError(f"unsafe archive member: {member.filename}")
        if member.is_dir():
            continue
        if member.filename in names:
            raise ValueError(f"duplicate archive member: {member.filename}")
        if (member.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f"symbolic links are not allowed: {member.filename}")
        names.add(member.filename)
        extracted_bytes += member.file_size
    missing = REQUIRED_FILES - names
    if missing:
        raise ValueError(f"policy archive is missing required files: {sorted(missing)}")
    if extracted_bytes > MAX_EXTRACTED_BYTES:
        raise ValueError("policy archive exceeds the extracted-size limit")
    return members


def validate_policy(path: Path) -> dict[str, Any]:
    """Validate a policy's schemas and cryptographic bindings without loading MuJoCo or ONNX."""
    root = path.resolve(strict=True)
    missing = sorted(name for name in REQUIRED_FILES if not (root / name).is_file())
    if missing:
        raise ValueError(f"installed policy is missing required files: {missing}")
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    bundle = json.loads((root / "policy-bundle.json").read_text(encoding="utf-8"))
    scenarios = json.loads((root / "scenarios.json").read_text(encoding="utf-8"))
    if not all(isinstance(value, dict) for value in (contract, bundle, scenarios)):
        raise ValueError("policy metadata roots must be objects")
    if contract.get("schema_version") != 2 or bundle.get("schema_version") != 1:
        raise ValueError("unsupported policy metadata schema")
    if scenarios.get("schema_version") != 2:
        raise ValueError("unsupported scenario schema")
    if canonical_hash(contract) != contract.get("sha256"):
        raise ValueError("controller contract integrity mismatch")
    if bundle.get("contract_sha256") != contract.get("sha256"):
        raise ValueError("policy bundle does not match the controller contract")
    if sha256_file(root / "model.mjb") != bundle.get("model_sha256"):
        raise ValueError("compiled model integrity mismatch")
    if sha256_file(root / "checkpoints" / "policy.onnx") != bundle.get("onnx_sha256"):
        raise ValueError("ONNX policy integrity mismatch")
    if scenarios.get("checkpoint_sha256") != bundle.get("checkpoint_sha256"):
        raise ValueError("scenarios do not match the policy checkpoint")
    return {
        "name": "standing-v1",
        "path": str(root),
        "checkpoint_sha256": bundle["checkpoint_sha256"],
        "onnx_sha256": bundle["onnx_sha256"],
        "model_sha256": bundle["model_sha256"],
        "verified": True,
    }


def _download(url: str, destination: Path) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "g1-mjlab-locomotion"})
    digest_bytes = 0
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("xb") as stream:
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > MAX_DOWNLOAD_BYTES:
            raise ValueError("policy archive exceeds the download-size limit")
        while chunk := response.read(1024 * 1024):
            digest_bytes += len(chunk)
            if digest_bytes > MAX_DOWNLOAD_BYTES:
                raise ValueError("policy archive exceeds the download-size limit")
            stream.write(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(destination)


def install_policy(
    name: str,
    destination: Path,
    *,
    archive_url: str | None = None,
    archive_sha256: str | None = None,
) -> dict[str, Any]:
    """Download, verify, and atomically install one published policy."""
    policy = get_policy(name)
    destination = destination.resolve()
    if destination.exists():
        result = validate_policy(destination)
        return {**result, "status": "already-installed", "version": policy.version}
    if (archive_url is None) != (archive_sha256 is None):
        raise ValueError("archive URL and SHA-256 override must be supplied together")
    url = archive_url or policy.archive_url
    expected_sha256 = archive_sha256 or policy.archive_sha256
    if len(expected_sha256) != 64:
        raise ValueError("published policy does not yet have a valid archive SHA-256")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{name}-", suffix=".zip", dir=destination.parent
    )
    os.close(descriptor)
    temporary_archive = Path(temporary_name)
    temporary_archive.unlink()
    staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=destination.parent))
    try:
        actual_sha256 = _download(url, temporary_archive)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"policy archive SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
        with zipfile.ZipFile(temporary_archive) as archive:
            members = _safe_members(archive)
            archive.extractall(staging, members=members)
        result = validate_policy(staging)
        staging.replace(destination)
        return {
            **result,
            "path": str(destination),
            "status": "installed",
            "version": policy.version,
            "archive_sha256": actual_sha256,
        }
    finally:
        temporary_archive.unlink(missing_ok=True)
        if staging.exists():
            shutil.rmtree(staging)
