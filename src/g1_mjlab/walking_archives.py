"""Build deterministic development-only walking inference and PPO-resume archives."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .artifacts import sha256_file


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.resolve(strict=True), destination)


def _manifest(root: Path, *, policy_id: str, artifact_type: str) -> dict[str, Any]:
    files = {
        path.relative_to(root).as_posix(): {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    return {
        "schema_version": 1,
        "task_id": "G1-Walking-Flat-v1",
        "policy_id": policy_id,
        "policy_version": "0.0.0-development",
        "artifact_type": artifact_type,
        "qualified": False,
        "qualified_command_range_m_s": [],
        "files": files,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _archive(root: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"archive already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=6)
    return {
        "path": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "size": destination.stat().st_size,
    }


def build_walking_archives(
    bundle: Path,
    source_run: Path,
    checkpoint: Path,
    scenarios: Path,
    policy_spec: Path,
    output: Path,
) -> dict[str, Any]:
    """Create separate portable inference and fresh-environment resume archives."""
    bundle = bundle.resolve(strict=True)
    source_run = source_run.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    walking_bundle = json.loads((bundle / "walking-policy-bundle.json").read_text(encoding="utf-8"))
    if walking_bundle.get("status") == "qualified":
        raise ValueError("this development archive builder cannot publish qualified policy assets")
    if sha256_file(checkpoint) != walking_bundle.get("checkpoint_sha256"):
        raise ValueError("resume checkpoint does not match inference bundle")
    if output.exists():
        raise FileExistsError(f"walking archive output already exists: {output}")
    output.mkdir(parents=True)
    project = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="g1-walking-archive-") as temporary:
        staging = Path(temporary)
        inference = staging / "inference"
        for name in (
            "contract.json",
            "host-profile.json",
            "model.mjb",
            "reference.npz",
            "walking-policy-bundle.json",
        ):
            _copy(bundle / name, inference / name)
        _copy(bundle / "walking-policy-bundle.json", inference / "policy-bundle.json")
        _copy(bundle / "policy.onnx", inference / "checkpoints" / "policy.onnx")
        _copy(scenarios, inference / "scenarios.json")
        _copy(policy_spec, inference / "policy-spec.json")
        _copy(project / "LICENSE", inference / "LICENSE")
        _copy(project / "THIRD_PARTY_NOTICES.md", inference / "THIRD_PARTY_NOTICES.md")
        _copy(
            project / "release" / "UNITREE_G1_LICENSE.txt",
            inference / "LICENSES" / "UNITREE_G1_LICENSE.txt",
        )
        (inference / "model-card.md").write_text(
            "# walking-v1 development policy\n\nThis artifact is unqualified development evidence. "
            "It passed 16/16 function and 0/16 combined function/style scenarios. "
            "Use requires explicit `--allow-unqualified-development`; no sim-to-real claim is made.\n",
            encoding="utf-8",
        )
        inference_manifest = _manifest(
            inference, policy_id="walking-v1", artifact_type="native_cpu_inference"
        )
        _write_json(inference / "manifest.json", inference_manifest)

        resume = staging / "resume"
        _copy(checkpoint, resume / "checkpoints" / checkpoint.name)
        for name in (
            "config.json",
            "algorithm.json",
            "mdp.json",
            "ppo-profile.json",
            "walking-profile.json",
            "walking-reward-profile.json",
            "runtime.json",
            "source.json",
            "source.zip",
            "contract.json",
        ):
            _copy(source_run / name, resume / name)
        _copy(bundle / "reference.npz", resume / "reference.npz")
        _copy(project / "LICENSE", resume / "LICENSE")
        _copy(project / "THIRD_PARTY_NOTICES.md", resume / "THIRD_PARTY_NOTICES.md")
        resume_manifest = _manifest(
            resume,
            policy_id="walking-v1-resume",
            artifact_type="ppo_fresh_environment_continuation",
        )
        resume_manifest["checkpoint_sha256"] = sha256_file(checkpoint)
        resume_manifest["semantics"] = (
            "complete learner/optimizer/normalizer state with fresh environments and RNG; "
            "not bitwise trajectory replay"
        )
        resume_manifest["commands"] = {
            "resume": "g1-mjlab train --resume checkpoints/" + checkpoint.name,
            "fine_tune": "g1-mjlab train --fine-tune checkpoints/" + checkpoint.name,
        }
        _write_json(resume / "manifest.json", resume_manifest)
        inference_result = _archive(inference, output / "walking-v1-development-inference.zip")
        resume_result = _archive(resume, output / "walking-v1-development-resume.zip")
    result = {
        "schema_version": 1,
        "status": "development_only_not_published",
        "checkpoint_sha256": walking_bundle["checkpoint_sha256"],
        "inference": inference_result,
        "resume": resume_result,
    }
    _write_json(output / "archives.json", result)
    return result
