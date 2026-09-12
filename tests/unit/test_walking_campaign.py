from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from g1_mjlab.artifacts import read_jsonl, sha256_file
from g1_mjlab.checkpoints import checkpoint_iteration, publish_checkpoint
from g1_mjlab.walking_campaign import (
    CampaignStore,
    evaluate_walking_checkpoints,
    load_walking_campaign_manifest,
    run_walking_campaign,
)


def _copy(source: Path, destination: Path) -> Path:
    destination.write_bytes(source.read_bytes())
    return destination


def _campaign_manifest(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "model_13398.pt"
    source.write_bytes(b"trusted checkpoint")
    references = {}
    files = {
        "run_config": root / "configs/walking-v1/stage19/arm-c-train.json",
        "walking_profile": root / "configs/walking-v1/stage19/walking-profile.json",
        "reward_profile": root / "configs/walking-v1/stage19/arm-c-rewards.json",
        "ppo_profile": root / "configs/walking-v1/stage19/ppo.json",
        "scenario_set": root / "configs/walking-v1/measurement-scenarios-v2.json",
    }
    references["source_checkpoint"] = {
        "path": source.name,
        "sha256": sha256_file(source),
    }
    for name, original in files.items():
        copied = _copy(original, tmp_path / f"{name}.json")
        references[name] = {"path": copied.name, "sha256": sha256_file(copied)}
    payload = {
        "schema_version": 1,
        "hypothesis": "Physical contact slip improves walking mechanics.",
        "baseline_commit": "a" * 40,
        **references,
        "training_seed": 42,
        "checkpoint_interval": 100,
        "max_updates": 300,
        "environment_count": 64,
        "stop_rules": {
            "segment_updates": 100,
            "smoke_updates": 2,
            "max_recovery_attempts": 1,
            "fatal_nonfinite": True,
            "cuda_error_codes": [700, 719],
        },
        "metric_schema_version": 2,
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_campaign_manifest_verifies_all_references_and_budget(tmp_path: Path) -> None:
    manifest = load_walking_campaign_manifest(_campaign_manifest(tmp_path))
    assert manifest.max_updates == 300
    assert manifest.stop_rules.segment_updates == 100
    assert manifest.source_checkpoint.path.is_absolute()


def test_campaign_manifest_fails_after_referenced_file_changes(tmp_path: Path) -> None:
    path = _campaign_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    (tmp_path / payload["reward_profile"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_walking_campaign_manifest(path)


def test_campaign_store_is_new_atomic_and_append_only(tmp_path: Path) -> None:
    manifest = load_walking_campaign_manifest(_campaign_manifest(tmp_path))
    output = tmp_path / "output"
    store = CampaignStore.create(output, manifest)
    store.transition("smoke_running")
    store.transition("smoke_passed")
    store.transition("training", completed_updates=0)
    store.transition("evaluating", completed_updates=100, transition_count=153_600)
    store.transition("completed")

    state = store.state()
    records, truncated = read_jsonl(output / "events.jsonl")
    assert state["status"] == "completed"
    assert state["revision"] == 5
    assert [record["event"] for record in records].count("state_transition") == 5
    assert not truncated
    with pytest.raises(FileExistsError):
        CampaignStore.create(output, manifest)


def test_synthetic_segments_preserve_lineage_and_iteration_boundaries(tmp_path: Path) -> None:
    manifest_path = _campaign_manifest(tmp_path)
    calls: list[list[str]] = []

    def run(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        if "train" in command:
            config = json.loads(Path(command[command.index("--config") + 1]).read_text())
            updates = config["max_iterations"]
            if "--resume" in command:
                source = Path(command[command.index("--resume") + 1])
                final_iteration = checkpoint_iteration(source) + updates
            else:
                final_iteration = updates - 1
            output.mkdir(parents=True)
            (output / "manifest.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            upstream = output / "upstream" / f"model_{final_iteration}.pt"
            upstream.parent.mkdir()
            upstream.write_bytes(f"checkpoint-{final_iteration}".encode())
            publish_checkpoint(
                upstream,
                output / "checkpoints",
                transitions_per_update=1536,
                source_lineage=None,
            )
        else:
            output.mkdir(parents=True)
            (output / "summary.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    state = run_walking_campaign(manifest_path, tmp_path / "campaign-output", process_runner=run)

    training = [command for command in calls if "train" in command]
    assert state["status"] == "completed"
    assert state["completed_updates"] == 300
    assert state["transition_count"] == 460_800
    assert len(training) == 4  # one smoke and three independent 100-update segments
    assert "--fine-tune" in training[1]
    assert "--resume" in training[2]
    assert Path(training[2][training[2].index("--resume") + 1]).name == "model_99.pt"
    assert Path(training[3][training[3].index("--resume") + 1]).name == "model_199.pt"
    assert Path(state["latest_checkpoint"]).name == "model_299.pt"

    result = evaluate_walking_checkpoints(
        manifest_path, tmp_path / "campaign-output", process_runner=run
    )
    assert len(result["jobs"]) == 4
    assert result["jobs"][0]["lineage"] == "source"


def test_cuda_failure_uses_latest_complete_checkpoint_for_one_recovery(tmp_path: Path) -> None:
    manifest_path = _campaign_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = tmp_path / payload["run_config"]["path"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["max_iterations"] = 100
    config_path.write_text(json.dumps(config), encoding="utf-8")
    payload["run_config"]["sha256"] = sha256_file(config_path)
    payload["max_updates"] = 100
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    failed_once = False

    def complete_training(command: list[str], final_iteration: int) -> None:
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        (output / "manifest.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        upstream = output / "upstream" / f"model_{final_iteration}.pt"
        upstream.parent.mkdir()
        upstream.write_bytes(f"checkpoint-{final_iteration}".encode())
        publish_checkpoint(
            upstream,
            output / "checkpoints",
            transitions_per_update=1536,
            source_lineage=None,
        )

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal failed_once
        if "-c" in command:
            return SimpleNamespace(returncode=0)
        output = Path(command[command.index("--output") + 1])
        if "diagnose-walking" in command:
            output.mkdir(parents=True)
            (output / "summary.json").write_text("{}", encoding="utf-8")
            return SimpleNamespace(returncode=0)
        config = json.loads(Path(command[command.index("--config") + 1]).read_text())
        updates = config["max_iterations"]
        source_iteration = -1
        if "--resume" in command:
            source_iteration = checkpoint_iteration(Path(command[command.index("--resume") + 1]))
        if "segment-000" in output.name and "recovery" not in output.name and not failed_once:
            failed_once = True
            output.mkdir(parents=True)
            partial = output / "upstream/model_49.pt"
            partial.parent.mkdir()
            partial.write_bytes(b"trusted partial checkpoint")
            publish_checkpoint(
                partial,
                output / "checkpoints",
                transitions_per_update=1536,
                source_lineage=None,
            )
            stream = kwargs["stdout"]
            stream.write("CUDA error 700\n")
            return SimpleNamespace(returncode=1)
        complete_training(command, source_iteration + updates)
        return SimpleNamespace(returncode=0)

    state = run_walking_campaign(manifest_path, tmp_path / "recovery-output", process_runner=run)

    assert state["status"] == "completed"
    assert state["recovery_attempts"] == 1
    assert state["completed_updates"] == 100
    assert Path(state["latest_checkpoint"]).name == "model_99.pt"
    records, _ = read_jsonl(tmp_path / "recovery-output/events.jsonl")
    assert sum(record["event"] == "cuda_recovery_started" for record in records) == 1
    assert sum(record["event"] == "cuda_recovery_completed" for record in records) == 1


def test_interruption_preserves_campaign_state_and_journal(tmp_path: Path) -> None:
    manifest_path = _campaign_manifest(tmp_path)
    output = tmp_path / "interrupted-output"

    def interrupt(*_: object, **__: object) -> SimpleNamespace:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_walking_campaign(manifest_path, output, process_runner=interrupt)

    state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    records, truncated = read_jsonl(output / "events.jsonl")
    assert state["status"] == "stopped"
    assert state["failure"] == "KeyboardInterrupt"
    assert not truncated
    assert records[-1]["to"] == "stopped"
