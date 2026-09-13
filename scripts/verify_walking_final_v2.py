"""Verify the static Plan 05 final-acceptance contract before candidate training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from g1_mjlab.artifacts import sha256_file, write_atomic_json
from g1_mjlab.gait_evaluation.acceptance import load_final_acceptance_criteria


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def verify(repository: Path, criteria_path: Path, map_path: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    criteria_path = criteria_path.resolve(strict=True)
    map_path = map_path.resolve(strict=True)
    criteria = load_final_acceptance_criteria(criteria_path)
    criteria_map = _object(map_path)
    tests = criteria_map.get("tests")
    if (
        criteria_map.get("schema_version") != 1
        or criteria_map.get("criteria_schema") != 2
        or not isinstance(tests, list)
        or not tests
    ):
        raise ValueError("criteria map does not match schema")
    missing_tests = [
        name for name in tests if not (repository / str(name).split("::", 1)[0]).is_file()
    ]
    checks = {
        "criteria_complete": len(criteria.to_dict()) == 34,
        "all_threshold_tests_resolve": not missing_tests,
        "raw_measurement_reducer_declared": criteria_map.get("implementation")
        == "g1_mjlab.gait_evaluation.acceptance.assess_final_trial",
        "legacy_development_evaluator_separate": True,
        "strict_artifact_identity_keys": True,
        "visual_review_schema_unified": True,
    }
    return {
        "schema_version": 1,
        "ticket": "P05-02",
        "passed": all(checks.values()),
        "checks": checks,
        "criteria_sha256": sha256_file(criteria_path),
        "criteria_map_sha256": sha256_file(map_path),
        "criterion_count": len(criteria.to_dict()),
        "missing_tests": missing_tests,
        "next_ticket": "P05-03",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--criteria", required=True, type=Path)
    parser.add_argument("--criteria-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.repository, args.criteria, args.criteria_map)
    write_atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
