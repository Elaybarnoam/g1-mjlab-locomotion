"""Bounded, crash-readable Stage 19 walking campaign orchestration."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import sha256_file, utc_now, write_atomic_json
from .config import (
    ResolvedRunConfig,
    Stage19RewardProfile,
    load_config,
    load_ppo_profile,
    load_stage19_reward_profile,
    load_walking_training_profile,
)

_CAMPAIGN_TRANSITIONS = {
    "planned": {"smoke_running", "failed", "stopped"},
    "smoke_running": {"smoke_passed", "failed", "stopped"},
    "smoke_passed": {"training", "failed", "stopped"},
    "training": {"evaluating", "failed", "stopped"},
    "evaluating": {"training", "completed", "failed", "stopped"},
}


@dataclass(frozen=True)
class ArtifactReference:
    path: Path
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class CampaignStopRules:
    segment_updates: int
    smoke_updates: int
    max_recovery_attempts: int
    fatal_nonfinite: bool
    cuda_error_codes: tuple[int, ...]


@dataclass(frozen=True)
class WalkingCampaignManifest:
    schema_version: int
    hypothesis: str
    baseline_commit: str
    source_checkpoint: ArtifactReference
    run_config: ArtifactReference
    walking_profile: ArtifactReference
    reward_profile: ArtifactReference
    ppo_profile: ArtifactReference
    training_seed: int
    scenario_set: ArtifactReference
    checkpoint_interval: int
    max_updates: int
    environment_count: int
    stop_rules: CampaignStopRules
    metric_schema_version: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in (
            "source_checkpoint",
            "run_config",
            "walking_profile",
            "reward_profile",
            "ppo_profile",
            "scenario_set",
        ):
            value[name] = getattr(self, name).to_dict()
        value["stop_rules"]["cuda_error_codes"] = list(self.stop_rules.cuda_error_codes)
        return value


@dataclass
class CampaignStore:
    root: Path

    @classmethod
    def create(cls, root: Path, manifest: WalkingCampaignManifest) -> CampaignStore:
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=False)
        for name in ("configs", "smoke", "segments", "evaluations", "logs"):
            (root / name).mkdir()
        write_atomic_json(root / "resolved-manifest.json", manifest.to_dict())
        now = utc_now()
        write_atomic_json(
            root / "state.json",
            {
                "schema_version": 1,
                "status": "planned",
                "revision": 0,
                "completed_updates": 0,
                "transition_count": 0,
                "recovery_attempts": 0,
                "latest_checkpoint": None,
                "created_at": now,
                "updated_at": now,
            },
        )
        store = cls(root)
        store.record(
            "campaign_created", {"manifest_sha256": sha256_file(root / "resolved-manifest.json")}
        )
        return store

    def state(self) -> dict[str, Any]:
        value = json.loads((self.root / "state.json").read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("campaign state root must be an object")
        return value

    def record(self, event: str, fields: dict[str, Any] | None = None) -> None:
        if not event:
            raise ValueError("campaign event name cannot be empty")
        state = self.state()
        value = {
            "schema_version": 1,
            "event": event,
            "status": state["status"],
            "revision": state["revision"],
            "time_utc": utc_now(),
            **(fields or {}),
        }
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def transition(self, status: str, **fields: Any) -> None:
        state = self.state()
        current = str(state["status"])
        if status not in _CAMPAIGN_TRANSITIONS.get(current, set()):
            raise ValueError(f"invalid campaign transition {current!r} -> {status!r}")
        state.update(fields)
        state["status"] = status
        state["revision"] = int(state["revision"]) + 1
        state["updated_at"] = utc_now()
        write_atomic_json(self.root / "state.json", state)
        self.record("state_transition", {"from": current, "to": status})

    def update(self, **fields: Any) -> None:
        """Atomically update progress without inventing a lifecycle transition."""
        state = self.state()
        state.update(fields)
        state["revision"] = int(state["revision"]) + 1
        state["updated_at"] = utc_now()
        write_atomic_json(self.root / "state.json", state)
        self.record("state_updated", {"fields": sorted(fields)})


def _artifact_reference(value: Any, root: Path, field: str) -> ArtifactReference:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{field} must contain exactly path and sha256")
    if not isinstance(value["path"], str) or not isinstance(value["sha256"], str):
        raise ValueError(f"{field} path and sha256 must be strings")
    path = Path(value["path"])
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    expected = value["sha256"].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(path) != expected:
        raise ValueError(f"{field} SHA-256 mismatch")
    return ArtifactReference(path, expected)


def load_walking_campaign_manifest(path: Path) -> WalkingCampaignManifest:
    """Load a complete campaign declaration and verify every referenced byte stream."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = set(WalkingCampaignManifest.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("walking campaign manifest fields do not match schema")
    root = path.resolve().parent
    artifact_names = {
        "source_checkpoint",
        "run_config",
        "walking_profile",
        "reward_profile",
        "ppo_profile",
        "scenario_set",
    }
    converted = dict(raw)
    for name in artifact_names:
        converted[name] = _artifact_reference(raw[name], root, name)
    stop_fields = set(CampaignStopRules.__dataclass_fields__)
    stop_rules = raw["stop_rules"]
    if not isinstance(stop_rules, dict) or set(stop_rules) != stop_fields:
        raise ValueError("campaign stop_rules fields do not match schema")
    cuda_codes = stop_rules["cuda_error_codes"]
    if not isinstance(cuda_codes, list) or any(
        not isinstance(code, int) or isinstance(code, bool) for code in cuda_codes
    ):
        raise ValueError("cuda_error_codes must be integer values")
    converted["stop_rules"] = CampaignStopRules(
        **{**stop_rules, "cuda_error_codes": tuple(cuda_codes)}
    )
    manifest = WalkingCampaignManifest(**converted)
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: WalkingCampaignManifest) -> None:
    if manifest.schema_version != 1 or manifest.metric_schema_version != 2:
        raise ValueError("only campaign schema 1 with metric schema 2 is supported")
    if not manifest.hypothesis.strip() or not re.fullmatch(
        r"[0-9a-f]{40}", manifest.baseline_commit.lower()
    ):
        raise ValueError("campaign identity is invalid")
    integer_values = {
        "training_seed": manifest.training_seed,
        "checkpoint_interval": manifest.checkpoint_interval,
        "max_updates": manifest.max_updates,
        "environment_count": manifest.environment_count,
        "segment_updates": manifest.stop_rules.segment_updates,
        "smoke_updates": manifest.stop_rules.smoke_updates,
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in integer_values.values()
    ):
        raise ValueError("campaign counts must be positive integers")
    if manifest.stop_rules.max_recovery_attempts not in {0, 1}:
        raise ValueError("campaign allows at most one recovery attempt")
    if not manifest.stop_rules.fatal_nonfinite:
        raise ValueError("non-finite values must be fatal")
    if set(manifest.stop_rules.cuda_error_codes) != {700, 719}:
        raise ValueError("campaign CUDA recovery codes must be exactly 700 and 719")
    if manifest.checkpoint_interval > manifest.stop_rules.segment_updates:
        raise ValueError("checkpoint interval cannot exceed segment budget")
    config: ResolvedRunConfig = load_config(manifest.run_config.path)
    if config.task_id != "G1-Walking-Flat-v1":
        raise ValueError("campaign run_config must target walking-v1")
    if config.seed != manifest.training_seed or config.num_envs != manifest.environment_count:
        raise ValueError("campaign seed or environment count differs from run_config")
    if config.save_interval != manifest.checkpoint_interval:
        raise ValueError("campaign checkpoint interval differs from run_config")
    if config.max_iterations != manifest.max_updates:
        raise ValueError("campaign max_updates differs from run_config")
    load_walking_training_profile(manifest.walking_profile.path)
    reward: Stage19RewardProfile = load_stage19_reward_profile(manifest.reward_profile.path)
    load_ppo_profile(manifest.ppo_profile.path)
    if not reward.name:
        raise ValueError("campaign reward profile has no identity")
    scenario = json.loads(manifest.scenario_set.path.read_text(encoding="utf-8"))
    if not isinstance(scenario, dict):
        raise ValueError("campaign scenario_set root must be an object")
    if not math.isfinite(float(config.control_dt)):
        raise ValueError("campaign control period must be finite")


ProcessRunner = Callable[..., Any]


def _write_run_config(
    manifest: WalkingCampaignManifest,
    destination: Path,
    *,
    run_name: str,
    updates: int,
) -> ResolvedRunConfig:
    raw = json.loads(manifest.run_config.path.read_text(encoding="utf-8"))
    raw.update(
        {
            "run_name": run_name,
            "max_iterations": updates,
            "save_interval": min(manifest.checkpoint_interval, updates),
        }
    )
    write_atomic_json(destination, raw)
    return load_config(destination)


def _run_logged(command: list[str], log: Path, *, process_runner: ProcessRunner) -> tuple[int, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        completed = process_runner(
            command,
            check=False,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    return_code = int(completed.returncode)
    return return_code, log.read_text(encoding="utf-8", errors="replace")


def _training_command(
    manifest: WalkingCampaignManifest,
    config: Path,
    output: Path,
    checkpoint: Path,
    *,
    resume: bool,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "g1_mjlab.cli",
        "train",
        "--config",
        str(config),
        "--walking-profile",
        str(manifest.walking_profile.path),
        "--walking-reward-profile",
        str(manifest.reward_profile.path),
        "--ppo-profile",
        str(manifest.ppo_profile.path),
        "--resume" if resume else "--fine-tune",
        str(checkpoint),
        "--output",
        str(output),
    ]


def _evaluation_command(
    manifest: WalkingCampaignManifest,
    config: Path,
    checkpoint: Path,
    output: Path,
    *,
    criteria: Path | None = None,
) -> list[str]:
    criteria = config.parent / "evaluation-v2.json" if criteria is None else criteria
    criteria.resolve(strict=True)
    return [
        sys.executable,
        "-m",
        "g1_mjlab.cli",
        "diagnose-walking",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--scenarios",
        str(manifest.scenario_set.path),
        "--criteria",
        str(criteria),
        "--output",
        str(output),
        "--physics-trace",
    ]


def _completed_checkpoint(run: Path, expected_iteration: int) -> tuple[Path, dict[str, Any]]:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError(f"training process did not complete: {run}")
    index = json.loads((run / "checkpoints/index.json").read_text(encoding="utf-8"))
    latest = index.get("latest")
    records = {record["name"]: record for record in index.get("checkpoints", [])}
    if latest not in records:
        raise ValueError("checkpoint index latest entry is missing")
    record = records[latest]
    checkpoint = run / "checkpoints" / latest
    if (
        record.get("completed_iteration") != expected_iteration
        or record.get("validity") != "complete"
        or record.get("size_bytes") != checkpoint.stat().st_size
        or record.get("sha256") != sha256_file(checkpoint)
    ):
        raise ValueError("latest checkpoint fails completion or integrity checks")
    return checkpoint, record


def _failure_classification(return_code: int, log_text: str) -> str | None:
    if not return_code:
        return None
    lowered = log_text.lower()
    if "non-finite" in lowered or "nonfinite" in lowered:
        return "nonfinite"
    cuda_codes = [code for code in (700, 719) if str(code) in lowered and "cuda" in lowered]
    return f"cuda-{cuda_codes[0]}" if cuda_codes else "process"


def _require_process_success(return_code: int, log_text: str, operation: str) -> None:
    classification = _failure_classification(return_code, log_text)
    if classification is not None:
        raise RuntimeError(f"{operation} failed ({classification}, exit {return_code})")


def _latest_valid_checkpoint(run: Path) -> tuple[Path, dict[str, Any]] | None:
    index_path = run / "checkpoints/index.json"
    if not index_path.exists():
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = {record["name"]: record for record in index.get("checkpoints", [])}
    record = records.get(index.get("latest"))
    if record is None or record.get("validity") != "complete":
        return None
    checkpoint = run / "checkpoints" / record["name"]
    if (
        not checkpoint.is_file()
        or record.get("size_bytes") != checkpoint.stat().st_size
        or record.get("sha256") != sha256_file(checkpoint)
    ):
        return None
    return checkpoint, record


def _verify_checkpoint_on_cpu(
    checkpoint: Path,
    log: Path,
    *,
    process_runner: ProcessRunner,
) -> None:
    script = (
        "import sys,torch;"
        "p=torch.load(sys.argv[1],map_location='cpu',weights_only=False);"
        "required={'actor_state_dict','critic_state_dict','optimizer_state_dict'};"
        "assert isinstance(p,dict) and required.issubset(p)"
    )
    command = [sys.executable, "-c", script, str(checkpoint)]
    return_code, log_text = _run_logged(command, log, process_runner=process_runner)
    _require_process_success(return_code, log_text, "CPU checkpoint verification")


def run_walking_campaign(
    manifest_path: Path,
    output: Path,
    *,
    process_runner: ProcessRunner = subprocess.run,
) -> dict[str, Any]:
    """Run smoke, bounded train segments, and isolated evaluation without GPU overlap."""
    manifest = load_walking_campaign_manifest(manifest_path)
    store = CampaignStore.create(output, manifest)
    output = store.root
    template = load_config(manifest.run_config.path)
    try:
        criteria_source = (
            Path(__file__).resolve().parents[2] / "configs/walking-v1/evaluation-v2.json"
        )
        criteria = output / "configs/evaluation-v2.json"
        criteria.write_bytes(criteria_source.read_bytes())
        store.update(
            evaluation_criteria=str(criteria),
            evaluation_criteria_sha256=sha256_file(criteria),
        )
        smoke_config_path = output / "configs/smoke.json"
        _write_run_config(
            manifest,
            smoke_config_path,
            run_name="campaign-stage19-smoke",
            updates=manifest.stop_rules.smoke_updates,
        )
        smoke_output = output / "smoke/initial"
        store.transition("smoke_running")
        smoke_command = _training_command(
            manifest,
            smoke_config_path,
            smoke_output,
            manifest.source_checkpoint.path,
            resume=False,
        )
        return_code, log_text = _run_logged(
            smoke_command, output / "logs/smoke-initial.log", process_runner=process_runner
        )
        _require_process_success(return_code, log_text, "initial smoke")
        smoke_iteration = manifest.stop_rules.smoke_updates - 1
        smoke_checkpoint, smoke_record = _completed_checkpoint(smoke_output, smoke_iteration)
        store.transition(
            "smoke_passed",
            smoke_checkpoint=str(smoke_checkpoint),
            smoke_checkpoint_sha256=smoke_record["sha256"],
        )

        completed_updates = 0
        source_checkpoint = manifest.source_checkpoint.path
        source_iteration = -1
        segment = 0
        store.transition("training")
        while completed_updates < manifest.max_updates:
            updates = min(
                manifest.stop_rules.segment_updates, manifest.max_updates - completed_updates
            )
            config_path = output / "configs" / f"segment-{segment:03d}.json"
            _write_run_config(
                manifest,
                config_path,
                run_name=f"campaign-stage19-segment-{segment:03d}",
                updates=updates,
            )
            segment_output = output / "segments" / f"segment-{segment:03d}"
            command = _training_command(
                manifest,
                config_path,
                segment_output,
                source_checkpoint,
                resume=segment > 0,
            )
            return_code, log_text = _run_logged(
                command,
                output / "logs" / f"segment-{segment:03d}.log",
                process_runner=process_runner,
            )
            expected_iteration = source_iteration + updates
            classification = _failure_classification(return_code, log_text)
            if classification is not None:
                state = store.state()
                can_recover = (
                    classification in {"cuda-700", "cuda-719"}
                    and state["recovery_attempts"] < manifest.stop_rules.max_recovery_attempts
                )
                if not can_recover:
                    _require_process_success(return_code, log_text, f"training segment {segment}")
                recovered = _latest_valid_checkpoint(segment_output)
                recovered_checkpoint = source_checkpoint
                recovered_iteration = source_iteration
                if recovered is not None:
                    recovered_checkpoint, recovered_record = recovered
                    recovered_iteration = int(recovered_record["completed_iteration"])
                if not source_iteration <= recovered_iteration < expected_iteration:
                    raise ValueError("recovery checkpoint lies outside the failed segment")
                recovery_attempt = int(state["recovery_attempts"]) + 1
                store.update(recovery_attempts=recovery_attempt)
                store.record(
                    "cuda_recovery_started",
                    {
                        "classification": classification,
                        "segment": segment,
                        "checkpoint": str(recovered_checkpoint),
                        "checkpoint_sha256": sha256_file(recovered_checkpoint),
                        "discarded_uncheckpointed_updates": "unknown",
                    },
                )
                _verify_checkpoint_on_cpu(
                    recovered_checkpoint,
                    output / "logs" / f"recovery-{recovery_attempt:03d}-cpu.log",
                    process_runner=process_runner,
                )
                recovery_config = output / "configs" / f"recovery-{recovery_attempt:03d}.json"
                _write_run_config(
                    manifest,
                    recovery_config,
                    run_name=f"campaign-stage19-recovery-{recovery_attempt:03d}",
                    updates=manifest.stop_rules.smoke_updates,
                )
                recovery_output = output / "smoke" / f"recovery-{recovery_attempt:03d}"
                recovery_command = _training_command(
                    manifest,
                    recovery_config,
                    recovery_output,
                    recovered_checkpoint,
                    resume=recovered_iteration >= 0,
                )
                recovery_code, recovery_text = _run_logged(
                    recovery_command,
                    output / "logs" / f"recovery-{recovery_attempt:03d}-smoke.log",
                    process_runner=process_runner,
                )
                _require_process_success(
                    recovery_code, recovery_text, f"recovery smoke {recovery_attempt}"
                )
                recovery_final_iteration = (
                    recovered_iteration + manifest.stop_rules.smoke_updates
                    if recovered_iteration >= 0
                    else manifest.stop_rules.smoke_updates - 1
                )
                _completed_checkpoint(recovery_output, recovery_final_iteration)
                remaining_updates = expected_iteration - recovered_iteration
                retry_config = output / "configs" / f"segment-{segment:03d}-recovery.json"
                _write_run_config(
                    manifest,
                    retry_config,
                    run_name=f"campaign-stage19-segment-{segment:03d}-recovery",
                    updates=remaining_updates,
                )
                segment_output = output / "segments" / f"segment-{segment:03d}-recovery"
                retry_command = _training_command(
                    manifest,
                    retry_config,
                    segment_output,
                    recovered_checkpoint,
                    resume=recovered_iteration >= 0,
                )
                retry_code, retry_text = _run_logged(
                    retry_command,
                    output / "logs" / f"segment-{segment:03d}-recovery.log",
                    process_runner=process_runner,
                )
                _require_process_success(
                    retry_code, retry_text, f"recovered training segment {segment}"
                )
                config_path = retry_config
                store.record(
                    "cuda_recovery_completed",
                    {"segment": segment, "recovery_attempt": recovery_attempt},
                )
            checkpoint, record = _completed_checkpoint(segment_output, expected_iteration)
            completed_updates += updates
            transition_count = completed_updates * template.transitions_per_update
            store.record(
                "checkpoint_completed",
                {
                    "segment": segment,
                    "completed_updates": completed_updates,
                    "transition_count": transition_count,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": record["sha256"],
                },
            )
            store.transition(
                "evaluating",
                completed_updates=completed_updates,
                transition_count=transition_count,
                latest_checkpoint=str(checkpoint),
                latest_checkpoint_sha256=record["sha256"],
            )
            evaluation_output = output / "evaluations" / f"segment-{segment:03d}"
            evaluation_command = _evaluation_command(
                manifest, config_path, checkpoint, evaluation_output
            )
            return_code, log_text = _run_logged(
                evaluation_command,
                output / "logs" / f"evaluation-{segment:03d}.log",
                process_runner=process_runner,
            )
            _require_process_success(return_code, log_text, f"evaluation segment {segment}")
            if not (evaluation_output / "summary.json").exists():
                raise RuntimeError("evaluation completed without summary.json")
            store.record(
                "evaluation_completed",
                {
                    "segment": segment,
                    "checkpoint_sha256": record["sha256"],
                    "summary": str(evaluation_output / "summary.json"),
                },
            )
            if completed_updates == manifest.max_updates:
                store.transition("completed")
                break
            store.transition("training")
            source_checkpoint = checkpoint
            source_iteration = expected_iteration
            segment += 1
        return store.state()
    except KeyboardInterrupt:
        current = store.state()["status"]
        if "stopped" in _CAMPAIGN_TRANSITIONS.get(current, set()):
            store.transition("stopped", failure="KeyboardInterrupt")
        raise
    except Exception as exc:
        current = store.state()["status"]
        if "failed" in _CAMPAIGN_TRANSITIONS.get(current, set()):
            store.transition("failed", failure=f"{type(exc).__name__}: {exc}")
        raise


def evaluate_walking_checkpoints(
    manifest_path: Path,
    output: Path,
    *,
    process_runner: ProcessRunner = subprocess.run,
) -> dict[str, Any]:
    """Evaluate every completed campaign checkpoint in fresh sequential processes."""
    manifest = load_walking_campaign_manifest(manifest_path)
    output = output.resolve(strict=True)
    store = CampaignStore(output)
    if store.state()["status"] not in {"training", "evaluating", "completed", "stopped"}:
        raise ValueError("campaign has no train checkpoints ready for evaluation")
    expected_manifest = manifest.to_dict()
    actual_manifest = json.loads((output / "resolved-manifest.json").read_text(encoding="utf-8"))
    if actual_manifest != expected_manifest:
        raise ValueError("campaign output belongs to a different manifest")
    jobs: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    candidates = [(manifest.source_checkpoint.path, manifest.run_config.path, "source")]
    for segment in sorted((output / "segments").glob("segment-*")):
        index = json.loads((segment / "checkpoints/index.json").read_text(encoding="utf-8"))
        config = output / "configs" / f"{segment.name}.json"
        for record in index["checkpoints"]:
            if record.get("validity") == "complete":
                candidates.append((segment / "checkpoints" / record["name"], config, segment.name))
    evaluation_root = output / "checkpoint-evaluations"
    evaluation_root.mkdir(exist_ok=True)
    for checkpoint, config, lineage in candidates:
        digest = sha256_file(checkpoint)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        job_output = evaluation_root / f"{checkpoint.stem}-{digest[:12]}"
        log = output / "logs" / f"checkpoint-evaluation-{digest[:12]}.log"
        if (job_output / "summary.json").exists():
            return_code = 0
        else:
            command = _evaluation_command(
                manifest,
                config,
                checkpoint,
                job_output,
                criteria=output / "configs/evaluation-v2.json",
            )
            return_code, log_text = _run_logged(command, log, process_runner=process_runner)
            _require_process_success(return_code, log_text, f"checkpoint evaluation {checkpoint}")
        if not (job_output / "summary.json").exists():
            raise RuntimeError(f"checkpoint evaluation omitted summary: {checkpoint}")
        jobs.append(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
                "lineage": lineage,
                "output": str(job_output),
                "return_code": return_code,
            }
        )
    result = {
        "schema_version": 1,
        "execution": "one fresh sequential process per checkpoint; no concurrent GPU work",
        "jobs": jobs,
    }
    write_atomic_json(output / "checkpoint-evaluations.json", result)
    store.record("checkpoint_evaluation_batch_completed", {"job_count": len(jobs)})
    return result
