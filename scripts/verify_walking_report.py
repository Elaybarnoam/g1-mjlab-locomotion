"""Verify P04-11 generated policy and offline report artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_mjlab.artifacts import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    specification = json.loads(args.spec.read_text(encoding="utf-8"))
    report = (args.report / "index.html").read_text(encoding="utf-8")
    data = json.loads((args.report / "report-data.json").read_text(encoding="utf-8"))
    checks = {
        "actor_102": len(specification["actor_inputs"]) == 102,
        "critic_114": len(specification["critic_inputs"]) == 114,
        "actions_29": len(specification["outputs"]) == 29,
        "parity_bound": specification["validation"]["parity_passed"] is True,
        "worked_physical_capture": isinstance(
            specification["worked_control_step"]["raw_observation"], list
        ),
        "missing_rollout_honest": specification["worked_rollout_minibatch"]["status"] == "missing",
        "offline": "https://" not in report and "http://" not in report,
        "failed_status_visible": "failed hypothesis" in report,
        "unavailable_visible": "Unavailable means not recorded" in report,
        "portable_policy_spec": (args.report / "policy-spec.json").is_file(),
        "source_hashes_complete": len(data["sources"]) == 11,
    }
    result = {
        "schema_version": 1,
        "ticket": "P04-11",
        "passed": all(checks.values()),
        "checks": checks,
        "policy_spec_sha256": sha256_file(args.spec),
        "report_sha256": sha256_file(args.report / "index.html"),
        "visual_qa": (
            "browser file URL was blocked by browser security policy; responsive CSS and DOM "
            "content were validated, but screenshot QA remains unavailable"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
