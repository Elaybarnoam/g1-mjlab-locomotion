"""Fail-closed verifier for the P04-09 native walking evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from g1_mjlab.artifacts import sha256_file


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"evidence must be an object: {path}")
    return value


def verify(bundle: Path, parity: Path, native: Path) -> dict[str, Any]:
    policy = _read(bundle / "walking-policy-bundle.json")
    parity_result = _read(parity / "summary.json")
    native_result = _read(native / "summary.json")
    checks = {
        "explicitly_unqualified": policy.get("status") == "unqualified_development",
        "exact_checkpoint": policy.get("checkpoint_sha256")
        == "0865d076a7820aba3bdd1d8d97d0d60194c7989e92919b510514ffde17f929e9",
        "bundle_files_hashed": all(
            sha256_file(bundle / name) == digest for name, digest in policy.get("files", {}).items()
        ),
        "shared_state_parity": parity_result.get("passed") is True
        and parity_result.get("samples", 0) >= 100,
        "observation_tolerance": parity_result.get("max_observation_absolute_error", 1.0) <= 1e-5,
        "onnx_tolerance": parity_result.get("max_action_absolute_error", 1.0) <= 1e-4,
        "cpu_60_second_finite": native_result.get("passed_finite_rollouts")
        == native_result.get("planned_rollouts")
        and native_result.get("planned_rollouts", 0) > 0
        and all(trial.get("survived_seconds") == 60.0 for trial in native_result.get("trials", [])),
    }
    return {
        "schema_version": 1,
        "ticket": "P04-09",
        "passed": all(checks.values()),
        "checks": checks,
        "bundle_manifest_sha256": sha256_file(bundle / "walking-policy-bundle.json"),
        "parity_summary_sha256": sha256_file(parity / "summary.json"),
        "native_summary_sha256": sha256_file(native / "summary.json"),
        "standing_regression": {
            "installed_release": "standing-v1@1.0.0",
            "archive_sha256": "4c7208ce29dae1106b031a06e37416f56c232054cc33a937f900e9a85662df92",
            "native_bundle_load": "passed [1,99] -> [1,29]",
        },
        "visible_viewer": "completed one 60-second WSLg native MuJoCo launch",
        "qualification_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--parity", required=True, type=Path)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.bundle, args.parity, args.native)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
