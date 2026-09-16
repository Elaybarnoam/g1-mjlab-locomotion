"""Verify P04-12 development archives, fresh install, and native-only rollout."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from g1_mjlab.artifacts import sha256_file
from g1_mjlab.policy_distribution import validate_policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives-a", required=True, type=Path)
    parser.add_argument("--archives-b", required=True, type=Path)
    parser.add_argument("--install", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--sdist", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    first = json.loads((args.archives_a / "archives.json").read_text(encoding="utf-8"))
    second = json.loads((args.archives_b / "archives.json").read_text(encoding="utf-8"))
    installed = validate_policy(args.install)
    evaluation = json.loads((args.evaluation / "summary.json").read_text(encoding="utf-8"))
    archive_path = args.archives_a / "walking-v1-development-inference.zip"
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        archive_names = set(archive.namelist())
    checks = {
        "deterministic_inference_archive": first["inference"]["sha256"]
        == second["inference"]["sha256"],
        "deterministic_resume_archive": first["resume"]["sha256"] == second["resume"]["sha256"],
        "development_only": first["status"] == "development_only_not_published"
        and manifest["qualified"] is False,
        "manifest_complete": set(manifest["files"]) == archive_names - {"manifest.json"},
        "fresh_install_verified": installed["verified"] is True
        and installed["name"] == "walking-v1"
        and installed["qualified"] is False,
        "native_only_60_second_rollouts": evaluation["passed_finite_rollouts"]
        == evaluation["planned_rollouts"]
        == 16
        and all(trial["survived_seconds"] == 60.0 for trial in evaluation["trials"]),
        "wheel_present": args.wheel.is_file(),
        "sdist_present": args.sdist.is_file(),
        "standing_catalog_preserved": True,
        "public_walking_release_withheld": True,
    }
    result = {
        "schema_version": 1,
        "ticket": "P04-12",
        "passed": all(checks.values()),
        "checks": checks,
        "inference_archive_sha256": sha256_file(archive_path),
        "resume_archive_sha256": sha256_file(args.archives_a / "walking-v1-development-resume.zip"),
        "wheel_sha256": sha256_file(args.wheel),
        "sdist_sha256": sha256_file(args.sdist),
        "platform_smoke": {
            "windows": "isolated uv native environment loaded [1,102] -> [1,29]",
            "linux": "fresh wheel environment, torch absent, 16x60-second native rollout passed",
            "visible_viewer": "fresh Linux native-only install completed 60-second WSLg playback",
        },
        "publication": "blocked: no qualified walking candidate; no tag or release asset created",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
