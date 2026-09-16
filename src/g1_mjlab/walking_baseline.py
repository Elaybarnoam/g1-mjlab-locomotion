"""Read-only inventory of the Plan 04 C/100 walking baseline."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from argparse import ArgumentParser
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import median
from typing import Any

EXPECTED_C100_CHECKPOINT_SHA256 = "0865d076a7820aba3bdd1d8d97d0d60194c7989e92919b510514ffde17f929e9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def source_inventory(repository: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    changes = git("status", "--porcelain=v1").splitlines()
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "changes": changes,
        "tracked_changes": [item for item in changes if not item.startswith("??")],
        "untracked_changes": [item for item in changes if item.startswith("??")],
    }


def host_inventory() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
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
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
    }


def _single_identity(metadata: list[dict[str, Any]], field: str) -> str:
    values = {str(item[field]) for item in metadata if item.get(field)}
    if len(values) != 1:
        raise ValueError(f"evaluation does not have one {field}: {sorted(values)}")
    return values.pop()


def _required_gaps() -> dict[str, Any]:
    gaps = [
        {
            "id": "fresh_gpu_reproduction",
            "status": "not_run",
            "required_evidence": (
                "C/100 rerun on all 16 Stage 19 scenarios with evaluator schema 2"
            ),
        },
        {
            "id": "quality_recheck",
            "status": "not_run",
            "required_evidence": "canonical format, lint, mypy, unit/native tests and repository policy",
        },
        {
            "id": "natural_walking_candidate",
            "status": "missing",
            "required_evidence": "checkpoint passing both function and style criteria",
        },
        {
            "id": "method_audits",
            "status": "missing",
            "required_evidence": "reference feasibility, phase, observation and reward-scale audits",
        },
        {
            "id": "strict_qualification_engine",
            "status": "missing",
            "required_evidence": "identity-bound numerical acceptance and unified review schema",
        },
        {
            "id": "walking_v2_contract",
            "status": "missing",
            "required_evidence": "versioned reference-conditioned task and inference contract",
        },
        {
            "id": "walking_v2_curriculum",
            "status": "missing",
            "required_evidence": "stand-walk-stop speed scheduler and independent seed results",
        },
        {
            "id": "walking_v2_native_parity",
            "status": "missing",
            "required_evidence": "checkpoint-bound export, native parity and deterministic playback",
        },
        {
            "id": "owner_visual_acceptance",
            "status": "missing",
            "required_evidence": "owner decision on exact checkpoint-bound videos",
        },
        {
            "id": "final_qualification_and_release",
            "status": "missing",
            "required_evidence": "paired backend qualification, report and public fresh-install proof",
        },
    ]
    return {"schema_version": 1, "status": "gaps_present", "gaps": gaps}


def verify_fresh_evaluation(path: Path) -> dict[str, Any]:
    """Verify a fresh evaluator-v2 run preserves the baseline scientific decision."""
    summary = _load_object(path.resolve(strict=True))
    trials = summary.get("trials")
    if (
        summary.get("schema_version") != 2
        or summary.get("checkpoint_sha256") != EXPECTED_C100_CHECKPOINT_SHA256
        or summary.get("completed") != 16
        or not isinstance(trials, list)
        or len(trials) != 16
    ):
        raise ValueError("fresh evaluation does not match the frozen C/100 cohort")
    required = {
        "functional_passed",
        "style_passed",
        "command_rms_m_s",
        "physical_slip_rms_m_s",
        "step_length_median_m",
        "raw_transition_rate_s",
    }
    if any(not required.issubset(trial) for trial in trials):
        raise ValueError("fresh evaluation is missing required gait metrics")
    counts = {
        "trials": len(trials),
        "functional_passes": sum(bool(trial["functional_passed"]) for trial in trials),
        "style_passes": sum(bool(trial["style_passed"]) for trial in trials),
        "combined_passes": sum(
            bool(trial["functional_passed"] and trial["style_passed"]) for trial in trials
        ),
    }
    reproduced = counts == {
        "trials": 16,
        "functional_passes": 16,
        "style_passes": 0,
        "combined_passes": 0,
    }
    if not reproduced:
        raise ValueError(f"fresh evaluation changed the baseline decision: {counts}")
    return {
        "path": str(path.resolve()),
        "checkpoint_sha256": EXPECTED_C100_CHECKPOINT_SHA256,
        "counts": counts,
        "median_command_rms_m_s": median(float(x["command_rms_m_s"]) for x in trials),
        "median_physical_slip_rms_m_s": median(float(x["physical_slip_rms_m_s"]) for x in trials),
        "median_step_length_m": median(float(x["step_length_median_m"]) for x in trials),
        "median_raw_transition_rate_s": median(float(x["raw_transition_rate_s"]) for x in trials),
        "reproduced_decision": True,
        "trajectory_equality_expected": False,
        "reason": "the configured CUDA backends are performance-oriented and nondeterministic",
    }


def verify_baseline(
    *,
    repository: Path,
    segment: Path,
    evaluation: Path,
    source: dict[str, Any] | None = None,
    host: dict[str, Any] | None = None,
    checkpoint_file_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Inventory the immutable Plan 04 C/100 baseline without running a simulator."""
    repository = repository.resolve(strict=True)
    segment = segment.resolve(strict=True)
    evaluation = evaluation.resolve(strict=True)
    bundle = _load_object(segment / "policy-bundle.json")
    manifest = _load_object(segment / "manifest.json")
    summary = _load_object(evaluation / "summary.json")
    if manifest.get("status") != "completed":
        raise ValueError("C/100 training segment is not complete")
    if bundle.get("checkpoint") != "model_99.pt" or manifest.get("final_checkpoint") not in (
        None,
        "model_99.pt",
    ):
        raise ValueError("baseline is not the Plan 04 C/100 checkpoint")
    declared_checkpoint_hashes = {
        str(bundle.get("checkpoint_sha256")),
        str(summary.get("checkpoint_sha256")),
    }
    if manifest.get("final_checkpoint_sha256") is not None:
        declared_checkpoint_hashes.add(str(manifest["final_checkpoint_sha256"]))
    if declared_checkpoint_hashes != {EXPECTED_C100_CHECKPOINT_SHA256}:
        raise ValueError("C/100 checkpoint identity differs from the frozen Plan 05 baseline")
    checkpoint = segment / "checkpoints" / "model_99.pt"
    actual_checkpoint_sha256 = checkpoint_file_sha256 or sha256_file(
        checkpoint.resolve(strict=True)
    )
    if actual_checkpoint_sha256 != EXPECTED_C100_CHECKPOINT_SHA256:
        raise ValueError("C/100 checkpoint file hash differs from the frozen Plan 05 baseline")

    model = (segment / "model.mjb").resolve(strict=True)
    model_sha256 = sha256_file(model)
    if model_sha256 != bundle.get("model_sha256"):
        raise ValueError("C/100 model hash differs from policy-bundle.json")
    source_record = _load_object(segment / "source.json")
    reference_record = next(
        (
            item
            for item in source_record.get("files", [])
            if item.get("path") == "configs/walking-v1/reference-map-v2.json"
        ),
        None,
    )
    if reference_record is None:
        raise ValueError("C/100 source inventory is missing reference-map-v2.json")
    reference = (repository / str(reference_record["path"])).resolve(strict=True)
    reference_file_sha256 = sha256_file(reference)
    if reference_file_sha256 != reference_record.get("sha256"):
        raise ValueError("reference map hash differs from the C/100 source inventory")

    trials = summary.get("trials")
    if summary.get("schema_version") != 2 or not isinstance(trials, list) or len(trials) != 16:
        raise ValueError("C/100 evaluation must contain 16 evaluator-schema-2 trials")
    trial_ids = [int(trial["trial_id"]) for trial in trials]
    if sorted(trial_ids) != list(range(16)):
        raise ValueError("C/100 evaluation trial ids must be exactly 0 through 15")
    metadata = [
        _load_object(evaluation / f"trace-{trial_id:03d}.metadata.json") for trial_id in trial_ids
    ]
    if any(item.get("schema_version") != 2 for item in metadata):
        raise ValueError("C/100 trace metadata must use schema 2")
    if _single_identity(metadata, "checkpoint_sha256") != EXPECTED_C100_CHECKPOINT_SHA256:
        raise ValueError("C/100 trace metadata references a different checkpoint")

    functional_passes = sum(bool(trial["functional_passed"]) for trial in trials)
    style_passes = sum(bool(trial["style_passed"]) for trial in trials)
    combined_passes = sum(
        bool(trial["functional_passed"] and trial["style_passed"]) for trial in trials
    )
    if (functional_passes, combined_passes) != (16, 0):
        raise ValueError(
            "C/100 evaluation differs from the frozen 16/16 function, 0/16 style result"
        )
    nominal = next(
        (int(trial["trial_id"]) for trial in trials if "nominal" in trial["scenario_name"]),
        0,
    )
    worst_slip = int(max(trials, key=lambda trial: trial["physical_slip_rms_m_s"])["trial_id"])
    worst_step = int(min(trials, key=lambda trial: trial["step_length_median_m"])["trial_id"])
    baseline = {
        "schema_version": 1,
        "status": "inventoried",
        "source": source if source is not None else source_inventory(repository),
        "host": host if host is not None else host_inventory(),
        "candidate": {
            "arm": "C",
            "completed_updates": 100,
            "checkpoint": "model_99.pt",
            "checkpoint_sha256": actual_checkpoint_sha256,
        },
        "training_evidence": {
            "run_id": manifest.get("run_id"),
            "source_commit": manifest.get("source_commit"),
            "runtime": _load_object(segment / "runtime.json"),
        },
        "artifacts": {
            "checkpoint": {"path": str(checkpoint), "sha256": actual_checkpoint_sha256},
            "reference_map": {"path": str(reference), "sha256": reference_file_sha256},
            "reference": {"sha256": _single_identity(metadata, "reference_sha256")},
            "model": {"path": str(model), "sha256": model_sha256},
            "controller": {"sha256": _single_identity(metadata, "controller_sha256")},
        },
        "evaluation": {
            "provenance": "Plan 04 retained evidence; no fresh GPU evaluation was run",
            "schema_version": 2,
            "counts": {
                "trials": len(trials),
                "functional_passes": functional_passes,
                "style_passes": style_passes,
                "combined_passes": combined_passes,
            },
            "median_command_rms_m_s": median(float(trial["command_rms_m_s"]) for trial in trials),
            "median_raw_transition_rate_s": median(
                float(trial["raw_transition_rate_s"]) for trial in trials
            ),
            "trials": trials,
            "retained_trajectory_trials": {
                "nominal": nominal,
                "worst_slip": worst_slip,
                "worst_step": worst_step,
            },
        },
        "interpretation": {
            "historical": (
                "Older walking-v1 functional claims do not establish natural gait under corrected "
                "physical-contact metrics."
            ),
            "current": "unqualified",
            "reason": "16/16 functional trials but 0/16 combined function/style trials",
        },
    }
    return baseline, _required_gaps()


def write_baseline_outputs(
    *,
    repository: Path,
    segment: Path,
    retained_evaluation: Path,
    fresh_evaluation: Path,
    output: Path,
    quality_passed: bool,
) -> dict[str, Any]:
    """Write the three machine-readable P05-01 handoff artifacts."""
    baseline, gaps = verify_baseline(
        repository=repository,
        segment=segment,
        evaluation=retained_evaluation,
    )
    fresh = verify_fresh_evaluation(fresh_evaluation)
    for gap in gaps["gaps"]:
        if gap["id"] == "fresh_gpu_reproduction":
            gap["status"] = "passed"
            gap["evidence"] = fresh["path"]
        elif gap["id"] == "quality_recheck":
            gap["status"] = "passed" if quality_passed else "failed"
    verification = {
        "schema_version": 1,
        "ticket": "P05-01",
        "passed": quality_passed and fresh["reproduced_decision"],
        "baseline_status": baseline["status"],
        "fresh_evaluation": fresh,
        "quality_passed": quality_passed,
        "remaining_gap_count": sum(gap["status"] == "missing" for gap in gaps["gaps"]),
        "next_ticket": "P05-02",
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("baseline.json", baseline),
        ("gap-inventory.json", gaps),
        ("verification.json", verification),
    ):
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return verification


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Verify the Plan 05 walking baseline")
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--segment", required=True, type=Path)
    parser.add_argument("--retained-evaluation", required=True, type=Path)
    parser.add_argument("--fresh-evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quality-passed", action="store_true")
    arguments = parser.parse_args(argv)
    result = write_baseline_outputs(
        repository=arguments.repository,
        segment=arguments.segment,
        retained_evaluation=arguments.retained_evaluation,
        fresh_evaluation=arguments.fresh_evaluation,
        output=arguments.output,
        quality_passed=arguments.quality_passed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
