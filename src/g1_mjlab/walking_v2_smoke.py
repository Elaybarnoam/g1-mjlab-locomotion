"""Fail-closed verification of the bounded walking-v2 PPO smoke."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .artifacts import sha256_file, write_atomic_json
from .walking_v2_campaign import load_walking_v2_campaign


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return False


def verify_walking_v2_smoke(
    *,
    campaign_path: Path,
    run: Path,
    resume_run: Path,
    rollout: Path,
    output: Path,
) -> dict[str, Any]:
    """Verify training, telemetry, checkpoint resume, and deterministic rollout evidence."""
    campaign = load_walking_v2_campaign(campaign_path)
    manifest = _json(run / "manifest.json")
    algorithm = _json(run / "algorithm.json")
    optimization = _json(run / "optimization.json")
    learning = _json(run / "learning-summary.json")
    checkpoint_index = _json(run / "checkpoints/index.json")
    resume_manifest = _json(resume_run / "manifest.json")
    resume = _json(resume_run / "resume.json")
    rollout_summary = _json(rollout / "summary.json")
    runtime = _json(run / "runtime.json")
    ppo = algorithm["algorithm"]
    expected_ppo = {
        "gamma": 0.99,
        "lam": 0.95,
        "clip_param": 0.2,
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "entropy_coef": 0.005,
        "learning_rate": 0.0003,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "optimizer": "adam",
    }
    latest = checkpoint_index["checkpoints"][-1]
    trace = rollout / str(rollout_summary["trace"])
    diagnostics = optimization.get("rollout_diagnostics", [])
    checks = {
        "campaign_schema_2": campaign.schema_version == 2,
        "training_completed": manifest.get("status") == "completed",
        "two_updates": optimization.get("updates") == 2,
        "samples_per_update": len(diagnostics) == 2
        and all(item.get("samples") == 1536 for item in diagnostics),
        "total_transitions": optimization.get("total_transitions") == 3072,
        "ppo_exact": all(ppo.get(name) == value for name, value in expected_ppo.items()),
        "network_exact": algorithm["actor"].get("hidden_dims") == [512, 256, 128]
        and algorithm["critic"].get("hidden_dims") == [512, 256, 128],
        "losses_finite": learning.get("all_losses_finite") is True and _finite(learning),
        "optimization_finite": optimization.get("all_kl_samples_finite") is True
        and optimization.get("all_recorded_gradients_finite") is True
        and _finite(optimization),
        "parameters_changed": all(
            component.get("parameters_changed") is True
            for component in optimization.get("components", {}).values()
        ),
        "full_state_checkpoint": latest.get("completed_updates") == 2
        and sha256_file(run / "checkpoints" / latest["name"]) == latest["sha256"],
        "onnx_exists": (run / "checkpoints/policy.onnx").is_file(),
        "resume_completed": resume_manifest.get("status") == "completed"
        and resume.get("restored_iteration") == 1
        and resume.get("first_new_iteration") == 2
        and len(resume.get("verified_restored_components", [])) >= 3,
        "deterministic_rollout": rollout_summary.get("schema_version") == 2
        and rollout_summary.get("recorded_steps", 0) > 0
        and rollout_summary.get("qualification_claim") is False
        and trace.is_file()
        and sha256_file(trace) == rollout_summary.get("trace_sha256"),
        "device_identified": runtime.get("device", {}).get("name")
        == "NVIDIA GeForce RTX 5060 Laptop GPU",
    }
    result = {
        "schema_version": 2,
        "phase": "P06-05",
        "campaign_sha256": sha256_file(campaign_path),
        "run": str(run),
        "run_manifest_sha256": sha256_file(run / "manifest.json"),
        "checkpoint_sha256": latest["sha256"],
        "resume_run": str(resume_run),
        "rollout": str(rollout),
        "checks": checks,
        "passed": all(checks.values()),
        "learning_claim": False,
        "limitations": [
            "Two PPO updates establish integration only.",
            "The deterministic smoke policy terminated after 1.14 seconds and is not qualified.",
        ],
    }
    write_atomic_json(output, result)
    return result
