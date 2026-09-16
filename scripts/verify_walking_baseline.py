"""Verify local walking run identities before a new development campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def checkpoint_audit(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("checkpoint verification requires the training runtime") from error

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {missing}")

    tensor_count = 0
    scalar_count = 0
    nonfinite: list[str] = []

    def visit(value: Any, location: str) -> None:
        nonlocal tensor_count, scalar_count
        if isinstance(value, torch.Tensor):
            tensor_count += 1
            scalar_count += value.numel()
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                nonfinite.append(location)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{location}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{location}[{index}]")

    visit(checkpoint, "checkpoint")
    if nonfinite:
        raise FloatingPointError(f"non-finite checkpoint tensors: {nonfinite}")

    actor = checkpoint["actor_state_dict"]
    critic = checkpoint["critic_state_dict"]
    normalizers: dict[str, dict[str, Any]] = {}
    for name, state, width in (("actor", actor, 102), ("critic", critic, 114)):
        keys = ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std")
        if any(key not in state for key in keys) or "obs_normalizer.count" not in state:
            raise ValueError(f"{name} normalizer is incomplete")
        if any(tuple(state[key].shape) != (1, width) for key in keys):
            raise ValueError(f"{name} normalizer width does not match the walking contract")
        normalizers[name] = {
            "width": width,
            "count": float(state["obs_normalizer.count"]),
            "minimum_variance": float(state["obs_normalizer._var"].min()),
            "minimum_std": float(state["obs_normalizer._std"].min()),
            "zero_variance_features": int((state["obs_normalizer._var"] == 0).sum()),
            "zero_std_features": int((state["obs_normalizer._std"] == 0).sum()),
        }
        if (
            normalizers[name]["count"] <= 0
            or normalizers[name]["minimum_variance"] < 0
            or normalizers[name]["minimum_std"] < 0
        ):
            raise ValueError(f"{name} normalizer is not usable")

    return {
        "iteration": int(checkpoint["iter"]),
        "tensor_count": tensor_count,
        "scalar_count": scalar_count,
        "all_floating_tensors_finite": True,
        "normalizers": normalizers,
    }


ALLOWED_CONTRACT_DIFFERENCES = frozenset({"config_sha256", "sha256"})

HISTORICAL_METRICS: dict[str, dict[str, tuple[float, int]]] = {
    "bootstrap": {
        "survived_seconds": (15.0, 0),
        "command_rms_m_s": (0.0843, 4),
        "stop_speed_rms_m_s": (0.0246, 4),
        "step_length_median_m": (0.0385, 4),
        "cadence_steps_s": (5.87, 2),
        "alternation_ratio": (0.792, 3),
        "contact_transitions_s": (12.67, 2),
        "stance_slip_rms_m_s": (0.2191, 4),
        "tiny_step_fraction": (0.674, 3),
    },
    "stage18": {
        "survived_seconds": (15.0, 0),
        "command_rms_m_s": (0.0940, 4),
        "stop_speed_rms_m_s": (0.0266, 4),
        "step_length_median_m": (0.0536, 4),
        "cadence_steps_s": (5.72, 2),
        "alternation_ratio": (0.820, 3),
        "contact_transitions_s": (11.99, 2),
        "stance_slip_rms_m_s": (0.2000, 4),
        "tiny_step_fraction": (0.741, 3),
    },
}


def evaluation_audit(path: Path, baseline: str) -> dict[str, Any]:
    summary = load_object(path.resolve(strict=True))
    trials = summary.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError(f"{path} has no evaluation trials")
    expected = HISTORICAL_METRICS[baseline]
    means = {
        field: sum(float(trial[field]) for trial in trials) / len(trials) for field in expected
    }
    mismatches = {
        field: {"actual": means[field], "expected": value, "decimal_places": places}
        for field, (value, places) in expected.items()
        if round(means[field], places) != value
    }
    functional_passes = sum(bool(trial["functional_passed"]) for trial in trials)
    style_passes = sum(bool(trial["style_passed"]) for trial in trials)
    if mismatches or functional_passes != 16 or style_passes != 0 or len(trials) != 16:
        raise ValueError(f"{baseline} evaluation differs from historical evidence: {mismatches}")
    return {
        "path": str(path),
        "schema_version": summary.get("schema_version"),
        "checkpoint": summary.get("checkpoint"),
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
        "trial_count": len(trials),
        "functional_passes": functional_passes,
        "style_passes": style_passes,
        "means": means,
        "historical_values_reproduced": True,
    }


def audit_run(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run = run.resolve(strict=True)
    bundle = load_object(run / "policy-bundle.json")
    contract = load_object(run / "contract.json")
    checkpoint = run / "checkpoints" / str(bundle["checkpoint"])
    files = {
        "checkpoint": (checkpoint, str(bundle["checkpoint_sha256"])),
        "onnx": (run / "checkpoints" / "policy.onnx", str(bundle["onnx_sha256"])),
        "model": (run / "model.mjb", str(bundle["model_sha256"])),
        "source": (run / "source.zip", str(bundle["source_sha256"])),
    }
    hashes: dict[str, dict[str, Any]] = {}
    for name, (path, expected) in files.items():
        actual = sha256_file(path.resolve(strict=True))
        hashes[name] = {"path": str(path), "expected": expected, "actual": actual}
        if actual != expected:
            raise ValueError(f"{name} hash differs from policy-bundle.json")
    actual_contract = canonical_hash(contract)
    if actual_contract != contract.get("sha256") or actual_contract != bundle.get(
        "contract_sha256"
    ):
        raise ValueError("contract identity is inconsistent")
    manifest = load_object(run / "manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError(f"run is not complete: {manifest.get('status')}")
    return (
        {
            "run_id": manifest.get("run_id"),
            "status": manifest["status"],
            "hashes": hashes,
            "contract_sha256": actual_contract,
            "checkpoint": checkpoint_audit(checkpoint),
            "runtime": load_object(run / "runtime.json"),
            "memory": load_object(run / "memory.json"),
            "algorithm": load_object(run / "algorithm.json"),
        },
        contract,
    )


def git_value(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def installed_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in (
        "g1-mjlab-locomotion",
        "mjlab",
        "mujoco",
        "mujoco-warp",
        "warp-lang",
        "torch",
        "rsl-rl-lib",
        "onnxruntime",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def build_inventory(
    repository: Path,
    bootstrap: Path,
    stage18: Path,
    bootstrap_evaluation: Path,
    stage18_evaluation: Path,
) -> dict[str, Any]:
    bootstrap_result, bootstrap_contract = audit_run(bootstrap)
    stage18_result, stage18_contract = audit_run(stage18)
    contract_differences = {
        field: {
            "bootstrap": bootstrap_contract.get(field),
            "stage18": stage18_contract.get(field),
        }
        for field in sorted(set(bootstrap_contract) | set(stage18_contract))
        if bootstrap_contract.get(field) != stage18_contract.get(field)
    }
    unexpected_differences = sorted(set(contract_differences) - ALLOWED_CONTRACT_DIFFERENCES)
    if unexpected_differences:
        raise ValueError(
            f"deployment semantics changed between bootstrap and Stage 18: {unexpected_differences}"
        )
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    hardware = {
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0)
        if torch is not None and torch.cuda.is_available()
        else None,
    }
    return {
        "schema_version": 1,
        "result": "verified",
        "source": {
            "commit": git_value(repository, "rev-parse", "HEAD"),
            "branch": git_value(repository, "branch", "--show-current"),
            "dirty": bool(git_value(repository, "status", "--porcelain")),
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            **hardware,
        },
        "packages": installed_versions(),
        "bootstrap": bootstrap_result,
        "stage18": stage18_result,
        "evaluations": {
            "bootstrap": evaluation_audit(bootstrap_evaluation, "bootstrap"),
            "stage18": evaluation_audit(stage18_evaluation, "stage18"),
        },
        "contract_comparison": {
            "fields_compared": len(set(bootstrap_contract) | set(stage18_contract)),
            "allowed_differences": contract_differences,
            "unexpected_differences": unexpected_differences,
            "deployment_semantics_equal": True,
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--repository", required=True, type=Path)
    result.add_argument("--bootstrap-run", required=True, type=Path)
    result.add_argument("--stage18-run", required=True, type=Path)
    result.add_argument("--bootstrap-evaluation", required=True, type=Path)
    result.add_argument("--stage18-evaluation", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    inventory = build_inventory(
        arguments.repository.resolve(strict=True),
        arguments.bootstrap_run,
        arguments.stage18_run,
        arguments.bootstrap_evaluation,
        arguments.stage18_evaluation,
    )
    if not all(
        math.isfinite(value)
        for value in (
            inventory["bootstrap"]["checkpoint"]["normalizers"]["actor"]["count"],
            inventory["stage18"]["checkpoint"]["normalizers"]["actor"]["count"],
        )
    ):
        raise FloatingPointError("normalizer counts are non-finite")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(arguments.output)
    print(json.dumps({"result": inventory["result"], "output": str(arguments.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
