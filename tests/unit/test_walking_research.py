from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from g1_mjlab.walking_research import (
    AuditArtifact,
    MethodAuditManifest,
    analyze_observations,
    analyze_phase_dependence,
    analyze_reward_returns,
    classify_hypothesis,
    synthetic_reward_ordering,
)


def _artifact(path: str = "artifact.bin") -> dict[str, str]:
    return {"path": path, "sha256": "a" * 64}


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "audit_id": "walking-v1-method-audit-001",
        "checkpoint": _artifact("checkpoint.pt"),
        "run_config": _artifact("run-config.json"),
        "walking_profile": _artifact("walking-profile.json"),
        "reward_profile": _artifact("reward-profile.json"),
        "ppo_profile": _artifact("ppo-profile.json"),
        "contract": _artifact("contract.json"),
        "reference": _artifact("reference.npz"),
        "reference_audit": _artifact("reference-audit.json"),
        "reference_speed_map": _artifact("reference-speed-map.json"),
        "controller": _artifact("controller.json"),
        "scenario_set": _artifact("scenarios.json"),
        "traces": [_artifact("trace-000.npz")],
        "reward_metrics": _artifact("metrics.jsonl"),
        "phase_sweep_count": 16,
        "observation_sample_count": 1024,
        "reward_rollout_batches": 10,
        "reward_num_envs": 64,
        "reward_rollout_steps": 24,
        "seeds": [51001, 51002],
    }


def test_method_audit_manifest_is_strict_and_hash_bound() -> None:
    manifest = MethodAuditManifest.from_dict(_manifest())

    assert manifest.phase_sweep_count == 16
    assert isinstance(manifest.traces[0], AuditArtifact)
    with pytest.raises(ValueError, match="fields"):
        MethodAuditManifest.from_dict({**_manifest(), "unexpected": True})
    with pytest.raises(ValueError, match="SHA-256"):
        MethodAuditManifest.from_dict(
            {**_manifest(), "checkpoint": {"path": "checkpoint.pt", "sha256": "bad"}}
        )


def test_phase_dependence_uses_only_phase_pair_and_reports_target_units() -> None:
    observations = np.zeros((4, 102), dtype=np.float64)
    output_weight = np.eye(29, 102)
    output_weight[:, 99] = 1.0
    output_weight[:, 100] = -0.5
    weights = [
        (np.eye(102), np.zeros(102)),
        (np.eye(102), np.zeros(102)),
        (np.eye(102), np.zeros(102)),
        (output_weight, np.zeros(29)),
    ]
    normalizer_mean = np.zeros(102)
    normalizer_std = np.ones(102)
    scale = np.full(29, 0.25)

    result, raw = analyze_phase_dependence(
        observations,
        weights,
        normalizer_mean,
        normalizer_std,
        scale,
        phase_sweep_count=16,
    )

    assert raw.shape == (4, 16, 29)
    assert result["changed_observation_indices"] == [99, 100]
    assert result["phase_sweep_count"] == 16
    assert result["target_displacement_rms_rad"] > 0
    assert result["hypothesis"] == "refuted"


def test_observation_audit_fails_closed_on_stale_action_and_phase() -> None:
    observations = np.ones((32, 102), dtype=np.float64)
    observations[:, 67:96] = 0.0
    observations[:, 99:101] = 0.0
    result = analyze_observations(observations, np.ones(102), actor_size=102, critic_size=114)

    assert result["actor_dimension_verified"] is True
    assert result["critic_dimension_declared"] is True
    assert result["phase_unit_circle_error_max"] == pytest.approx(1.0)
    assert "phase_not_unit_circle" in result["flags"]
    assert "stale_previous_action" in result["flags"]


def test_reward_return_audit_distinguishes_aggregate_from_sample_level(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        {"metric": "Episode_Reward/reference_joint_pose", "value": value}
        for value in (0.0, 0.1, 0.2)
    ]
    rows += [{"metric": "Loss/value", "value": 0.3}]
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = analyze_reward_returns(metrics, gamma=0.99, gae_lambda=0.95)

    assert result["sample_level_returns_available"] is False
    assert result["gamma"] == pytest.approx(0.99)
    assert result["metrics"]["Episode_Reward/reference_joint_pose"]["zero_fraction"] == pytest.approx(
        1 / 3
    )
    assert result["hypothesis"] == "inconclusive"


def test_synthetic_reward_ordering_exposes_disabled_reference_style_terms() -> None:
    names = ("reference_joint_pose", "extra_contact_event", "phase_contact_error")

    disabled = synthetic_reward_ordering(names, np.array([0.0, -1.0, -1.0]))
    enabled = synthetic_reward_ordering(names, np.array([1.0, -1.0, -1.0]))

    assert disabled["desirable_above_phase_shuffle"] is False
    assert disabled["desirable_above_contact_chatter"] is True
    assert disabled["passed"] is False
    assert enabled["passed"] is True


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        ({"passed": True}, "supported"),
        ({"passed": False}, "refuted"),
        ({"passed": None}, "inconclusive"),
    ],
)
def test_hypothesis_classification_is_tristate(
    evidence: dict[str, bool | None], expected: str
) -> None:
    assert classify_hypothesis(evidence["passed"]) == expected
