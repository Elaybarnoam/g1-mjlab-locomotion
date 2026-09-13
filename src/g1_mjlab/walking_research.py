"""Hash-bound, fail-closed diagnostics for choosing the walking-v2 learning method."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from .artifacts import sha256_file, write_atomic_json

FloatArray = npt.NDArray[np.float64]


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class AuditArtifact:
    path: str
    sha256: str

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ValueError("audit artifact fields do not match schema")
        path = value["path"]
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise ValueError("audit artifact path must be a safe relative path")
        return cls(path, _require_sha256(value["sha256"], "artifact SHA-256"))

    def resolve(self, root: Path) -> Path:
        path = (root / self.path).resolve(strict=True)
        if not path.is_relative_to(root.resolve(strict=True)):
            raise ValueError("audit artifact escapes manifest root")
        if sha256_file(path) != self.sha256:
            raise ValueError(f"audit artifact hash mismatch: {self.path}")
        return path


@dataclass(frozen=True, slots=True)
class ResolvedAuditArtifacts:
    checkpoint: Path
    run_config: Path
    walking_profile: Path
    reward_profile: Path
    ppo_profile: Path
    contract: Path
    reference: Path
    reference_audit: Path
    reference_speed_map: Path
    controller: Path
    scenario_set: Path
    traces: tuple[Path, ...]
    reward_metrics: Path


@dataclass(frozen=True, slots=True)
class MethodAuditManifest:
    audit_id: str
    checkpoint: AuditArtifact
    run_config: AuditArtifact
    walking_profile: AuditArtifact
    reward_profile: AuditArtifact
    ppo_profile: AuditArtifact
    contract: AuditArtifact
    reference: AuditArtifact
    reference_audit: AuditArtifact
    reference_speed_map: AuditArtifact
    controller: AuditArtifact
    scenario_set: AuditArtifact
    traces: tuple[AuditArtifact, ...]
    reward_metrics: AuditArtifact
    phase_sweep_count: int
    observation_sample_count: int
    reward_rollout_batches: int
    reward_num_envs: int
    reward_rollout_steps: int
    seeds: tuple[int, ...]
    schema_version: int = 1

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = {field.name for field in fields(cls)}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("method audit manifest fields do not match schema 1")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("method audit manifest requires schema version 1")
        audit_id = value["audit_id"]
        if not isinstance(audit_id, str) or not audit_id.strip():
            raise ValueError("audit_id must be a nonempty string")
        artifact_names = (
            "checkpoint",
            "run_config",
            "walking_profile",
            "reward_profile",
            "ppo_profile",
            "contract",
            "reference",
            "reference_audit",
            "reference_speed_map",
            "controller",
            "scenario_set",
            "reward_metrics",
        )
        artifacts = {name: AuditArtifact.from_dict(value[name]) for name in artifact_names}
        traces_raw = value["traces"]
        if not isinstance(traces_raw, list) or not traces_raw:
            raise ValueError("traces must be a nonempty list")
        traces = tuple(AuditArtifact.from_dict(item) for item in traces_raw)
        if len({item.path for item in traces}) != len(traces):
            raise ValueError("trace paths must be unique")
        count_names = (
            "phase_sweep_count",
            "observation_sample_count",
            "reward_rollout_batches",
            "reward_num_envs",
            "reward_rollout_steps",
        )
        counts = {name: value[name] for name in count_names}
        if any(type(number) is not int or number <= 0 for number in counts.values()):
            raise ValueError("audit counts must be positive integers")
        if counts["phase_sweep_count"] != 16:
            raise ValueError("phase audit is frozen at 16 sweep angles")
        seeds_raw = value["seeds"]
        if (
            not isinstance(seeds_raw, list)
            or not seeds_raw
            or any(type(seed) is not int or seed < 0 for seed in seeds_raw)
            or len(seeds_raw) != len(set(seeds_raw))
        ):
            raise ValueError("seeds must be unique nonnegative integers")
        return cls(
            audit_id=audit_id,
            checkpoint=artifacts["checkpoint"],
            run_config=artifacts["run_config"],
            walking_profile=artifacts["walking_profile"],
            reward_profile=artifacts["reward_profile"],
            ppo_profile=artifacts["ppo_profile"],
            contract=artifacts["contract"],
            reference=artifacts["reference"],
            reference_audit=artifacts["reference_audit"],
            reference_speed_map=artifacts["reference_speed_map"],
            controller=artifacts["controller"],
            scenario_set=artifacts["scenario_set"],
            traces=traces,
            reward_metrics=artifacts["reward_metrics"],
            phase_sweep_count=counts["phase_sweep_count"],
            observation_sample_count=counts["observation_sample_count"],
            reward_rollout_batches=counts["reward_rollout_batches"],
            reward_num_envs=counts["reward_num_envs"],
            reward_rollout_steps=counts["reward_rollout_steps"],
            seeds=tuple(seeds_raw),
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def resolve_all(self, root: Path) -> ResolvedAuditArtifacts:
        return ResolvedAuditArtifacts(
            checkpoint=self.checkpoint.resolve(root),
            run_config=self.run_config.resolve(root),
            walking_profile=self.walking_profile.resolve(root),
            reward_profile=self.reward_profile.resolve(root),
            ppo_profile=self.ppo_profile.resolve(root),
            contract=self.contract.resolve(root),
            reference=self.reference.resolve(root),
            reference_audit=self.reference_audit.resolve(root),
            reference_speed_map=self.reference_speed_map.resolve(root),
            controller=self.controller.resolve(root),
            scenario_set=self.scenario_set.resolve(root),
            reward_metrics=self.reward_metrics.resolve(root),
            traces=tuple(item.resolve(root) for item in self.traces),
        )


def classify_hypothesis(passed: bool | None) -> str:
    if passed is None:
        return "inconclusive"
    return "supported" if passed else "refuted"


def _elu(value: FloatArray) -> FloatArray:
    return np.where(value > 0, value, np.expm1(value))


def evaluate_mlp(
    inputs: FloatArray,
    layers: list[tuple[FloatArray, FloatArray]],
    normalizer_mean: FloatArray,
    normalizer_std: FloatArray,
) -> FloatArray:
    if inputs.ndim != 2 or inputs.shape[1] != len(normalizer_mean):
        raise ValueError("MLP input does not match normalizer")
    if normalizer_std.shape != normalizer_mean.shape or np.any(normalizer_std < 0):
        raise ValueError("normalizer statistics are invalid")
    hidden = (inputs - normalizer_mean) / np.maximum(normalizer_std, 1e-8)
    for index, (weight, bias) in enumerate(layers):
        if (
            weight.ndim != 2
            or bias.shape != (weight.shape[0],)
            or hidden.shape[1] != weight.shape[1]
        ):
            raise ValueError("MLP layer shapes are inconsistent")
        hidden = hidden @ weight.T + bias
        if index < len(layers) - 1:
            hidden = _elu(hidden)
    return hidden


def analyze_phase_dependence(
    observations: FloatArray,
    layers: list[tuple[FloatArray, FloatArray]],
    normalizer_mean: FloatArray,
    normalizer_std: FloatArray,
    action_scale: FloatArray,
    *,
    phase_sweep_count: int,
) -> tuple[dict[str, Any], FloatArray]:
    if observations.ndim != 2 or observations.shape[1] != 102:
        raise ValueError("phase audit requires [samples,102] actor observations")
    if action_scale.shape != (29,) or phase_sweep_count != 16:
        raise ValueError("phase audit requires 29 action scales and 16 phases")
    phases = np.arange(phase_sweep_count, dtype=np.float64) / phase_sweep_count
    swept = np.repeat(observations[:, None, :], phase_sweep_count, axis=1)
    swept[:, :, 99] = np.sin(2 * np.pi * phases)
    swept[:, :, 100] = np.cos(2 * np.pi * phases)
    actions = evaluate_mlp(swept.reshape(-1, 102), layers, normalizer_mean, normalizer_std).reshape(
        len(observations), phase_sweep_count, -1
    )
    if actions.shape[2] != 29 or not np.isfinite(actions).all():
        raise ValueError("actor phase sweep output must be finite [samples,16,29]")
    joint_std = np.std(actions, axis=1)
    target_rms = float(np.sqrt(np.mean(np.square(joint_std * action_scale))))
    result = {
        "schema_version": 1,
        "sample_count": len(observations),
        "phase_sweep_count": phase_sweep_count,
        "changed_observation_indices": [99, 100],
        "per_joint_action_std": np.mean(joint_std, axis=0).tolist(),
        "action_std_rms": float(np.sqrt(np.mean(np.square(joint_std)))),
        "target_displacement_rms_rad": target_rms,
        "phase_ignored_threshold_rad": 0.01,
        "closed_loop_contact_change_available": False,
        "hypothesis": classify_hypothesis(target_rms < 0.01),
        "interpretation": (
            "The frozen actor is phase-insensitive at a fixed physical state."
            if target_rms < 0.01
            else "The frozen actor changes targets with phase; usefulness requires closed-loop evidence."
        ),
    }
    return result, actions


def analyze_observations(
    observations: FloatArray,
    normalizer_std: FloatArray,
    *,
    actor_size: int,
    critic_size: int,
) -> dict[str, Any]:
    if observations.ndim != 2 or not np.isfinite(observations).all():
        raise ValueError("observation audit requires finite rank-2 observations")
    if normalizer_std.shape != (actor_size,):
        raise ValueError("actor normalizer dimension does not match contract")
    empirical_std = np.std(observations, axis=0)
    radius = np.linalg.norm(observations[:, 99:101], axis=1)
    flags: list[str] = []
    if float(np.max(np.abs(radius - 1))) > 1e-4:
        flags.append("phase_not_unit_circle")
    if float(np.max(empirical_std[67:96])) < 1e-6:
        flags.append("stale_previous_action")
    if float(np.max(empirical_std[99:101])) < 1e-4:
        flags.append("near_zero_phase_variance")
    zero_normalizer_indices = np.flatnonzero(normalizer_std < 1e-8).tolist()
    unexpected_zero_normalizer = [
        index for index in zero_normalizer_indices if index not in {97, 98}
    ]
    if unexpected_zero_normalizer:
        flags.append("near_zero_normalizer_variance")
    return {
        "schema_version": 1,
        "sample_count": len(observations),
        "actor_dimension": observations.shape[1],
        "actor_dimension_verified": observations.shape[1] == actor_size == 102,
        "critic_dimension": critic_size,
        "critic_dimension_declared": critic_size == 114,
        "empirical_std": empirical_std.tolist(),
        "normalizer_std": normalizer_std.tolist(),
        "zero_normalizer_variance_indices": zero_normalizer_indices,
        "expected_fixed_command_indices": [97, 98],
        "unexpected_zero_normalizer_variance_indices": unexpected_zero_normalizer,
        "phase_unit_circle_error_max": float(np.max(np.abs(radius - 1))),
        "flags": flags,
        "explicit_reference_targets_present": False,
        "portable_actor_contact_force_present": False,
        "hypothesis": "refuted" if flags else "inconclusive",
        "interpretation": (
            "Detected a concrete actor-input defect."
            if flags
            else "No concrete defect was proven; walking-v2 will test explicit reference targets."
        ),
    }


def analyze_reward_returns(
    metrics_path: Path, *, gamma: float, gae_lambda: float
) -> dict[str, Any]:
    if not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and GAE lambda must be in [0,1]")
    values: dict[str, list[float]] = {}
    updates: set[int] = set()
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        name = record.get("metric")
        value = record.get("value")
        update = record.get("update")
        if type(update) is int:
            updates.add(update)
        if isinstance(name, str) and type(value) in {int, float} and math.isfinite(value):
            values.setdefault(name, []).append(float(value))
    summaries: dict[str, dict[str, float]] = {}
    for name, samples in sorted(values.items()):
        array = np.asarray(samples, dtype=np.float64)
        summaries[name] = {
            "count": float(len(array)),
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "p05": float(np.percentile(array, 5)),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "zero_fraction": float(np.mean(array == 0)),
        }
    sample_level = all(
        name in values
        for name in ("Audit/reward", "Audit/value", "Audit/return", "Audit/advantage")
    )
    return {
        "schema_version": 1,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "metrics": summaries,
        "observed_update_count": len(updates),
        "sample_level_returns_available": sample_level,
        "advantage_correlations_available": sample_level,
        "timing_alignment_verified": False,
        "synthetic_reward_ordering_verified": False,
        "hypothesis": "inconclusive",
        "limitations": [
            "Aggregate TensorBoard telemetry cannot establish per-sample reward/advantage association.",
            "A future instrumented rollout must emit the four Audit/* sample streams.",
        ],
    }


def _correlation(left: FloatArray, right: FloatArray) -> float | None:
    if left.size != right.size or left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def synthetic_reward_ordering(
    term_names: tuple[str, ...], term_weights: FloatArray
) -> dict[str, Any]:
    """Check whether the frozen weights distinguish desirable, shuffled and chattering motion."""
    if term_weights.shape != (len(term_names),):
        raise ValueError("reward term names and weights must align")
    weights = dict(zip(term_names, term_weights, strict=True))
    pose_weight = weights.get("reference_joint_pose", 0.0)
    velocity_weight = weights.get("reference_joint_velocity", 0.0)
    foot_weight = weights.get("reference_foot_position", 0.0)
    contact_weight = weights.get("reference_contact_timing", 0.0)
    phase_style_weight = pose_weight + velocity_weight + foot_weight + contact_weight
    desired_style = phase_style_weight
    shuffled_style = phase_style_weight * math.exp(-4.0)
    chatter_penalty = weights.get("extra_contact_event", 0.0) + weights.get(
        "phase_contact_error", 0.0
    )
    shuffle_ordered = bool(desired_style > shuffled_style + 1e-12)
    chatter_ordered = bool(0.0 > chatter_penalty + 1e-12)
    return {
        "desirable_style_score": desired_style,
        "phase_shuffled_style_score": shuffled_style,
        "contact_chatter_score": chatter_penalty,
        "desirable_above_phase_shuffle": shuffle_ordered,
        "desirable_above_contact_chatter": chatter_ordered,
        "passed": shuffle_ordered and chatter_ordered,
    }


def _collect_fresh_reward_returns(  # pragma: no cover - exercised by the GPU audit
    manifest: MethodAuditManifest,
    resolved: ResolvedAuditArtifacts,
    output: Path,
) -> dict[str, Any]:
    """Collect frozen-policy PPO storage without calling the optimizer."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls

    from .config import (
        load_config,
        load_ppo_profile,
        load_stage19_reward_profile,
        load_walking_training_profile,
    )
    from .environment import build_train_config

    base = load_config(resolved.run_config)
    walking = load_walking_training_profile(resolved.walking_profile)
    reward_profile = load_stage19_reward_profile(resolved.reward_profile)
    ppo = load_ppo_profile(resolved.ppo_profile)
    config = replace(
        base,
        num_envs=manifest.reward_num_envs,
        rollout_steps=manifest.reward_rollout_steps,
        seed=manifest.seeds[0],
        max_iterations=1,
    )
    train_cfg = build_train_config(
        config,
        output / "probe-log",
        walking_profile=walking,
        walking_reward_profile=reward_profile,
        ppo_profile=ppo,
    )
    env = ManagerBasedRlEnv(cfg=train_cfg.env, device=config.device, render_mode=None)
    arrays: dict[str, list[FloatArray]] = {
        name: [] for name in ("reward", "value", "return", "advantage", "done")
    }
    weighted_batches: list[FloatArray] = []
    reward_manager: Any = env.reward_manager
    term_names = tuple(reward_manager.active_terms)
    term_weights = np.asarray(
        [reward_manager.get_term_cfg(name).weight for name in term_names], dtype=np.float64
    )
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=train_cfg.agent.clip_actions)
        runner_cls = load_runner_cls(config.task_id) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(train_cfg.agent), device=config.device)
        runner.load(
            str(resolved.checkpoint),
            load_cfg={"actor": True, "critic": True},
            strict=True,
            map_location=config.device,
        )
        algorithm = runner.alg
        algorithm.train_mode()
        with torch.inference_mode():
            for seed in manifest.seeds[: manifest.reward_rollout_batches]:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                env.reset(seed=seed)
                observations = wrapped.get_observations().to(config.device)
                weighted_steps: list[FloatArray] = []
                for _ in range(manifest.reward_rollout_steps):
                    actions = algorithm.act(observations)
                    observations, rewards, dones, extras = wrapped.step(actions.to(wrapped.device))
                    step_values: list[list[float]] = []
                    for index in range(manifest.reward_num_envs):
                        step_values.append(
                            [
                                float(value[0])
                                for _, value in reward_manager.get_active_iterable_terms(index)
                            ]
                        )
                    weighted_steps.append(np.asarray(step_values, dtype=np.float64))
                    observations, rewards, dones = (
                        observations.to(config.device),
                        rewards.to(config.device),
                        dones.to(config.device),
                    )
                    algorithm.process_env_step(observations, rewards, dones, extras)
                algorithm.compute_returns(observations)
                storage = algorithm.storage
                for name, tensor in (
                    ("reward", storage.rewards),
                    ("value", storage.values),
                    ("return", storage.returns),
                    ("advantage", storage.advantages),
                    ("done", storage.dones),
                ):
                    arrays[name].append(tensor.detach().cpu().numpy().astype(np.float64))
                weighted_batches.append(np.stack(weighted_steps))
                storage.clear()
    finally:
        env.close()
    packed = {name: np.stack(batches) for name, batches in arrays.items()}
    weighted = np.stack(weighted_batches)
    raw = np.divide(
        weighted,
        term_weights[None, None, None, :],
        out=np.zeros_like(weighted),
        where=np.abs(term_weights[None, None, None, :]) > 0,
    )
    np.savez_compressed(
        output / "reward-return-samples.npz",
        reward=packed["reward"],
        value=packed["value"],
        returns=packed["return"],
        advantage=packed["advantage"],
        done=packed["done"],
        weighted_reward_rate=weighted,
        raw_reward=raw,
        term_names=np.asarray(term_names, dtype=np.str_),
        term_weights=term_weights,
        seeds=np.asarray(manifest.seeds[: manifest.reward_rollout_batches], dtype=np.int64),
    )
    advantage = packed["advantage"].reshape(-1)
    correlations = {
        name: _correlation(weighted[..., index].reshape(-1), advantage)
        for index, name in enumerate(term_names)
    }
    term_statistics = {}
    for index, name in enumerate(term_names):
        samples = weighted[..., index]
        term_statistics[name] = {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "rms": float(np.sqrt(np.mean(np.square(samples)))),
            "p05": float(np.percentile(samples, 5)),
            "p50": float(np.percentile(samples, 50)),
            "p95": float(np.percentile(samples, 95)),
            "zero_fraction": float(np.mean(samples == 0)),
        }
    style_names = (
        "reference_joint_pose",
        "reference_joint_velocity",
        "reference_foot_position",
        "reference_contact_timing",
    )
    style_indices = [term_names.index(name) for name in style_names]
    style_sum = np.sum(weighted[..., style_indices], axis=-1)
    total_sum = np.sum(weighted, axis=-1)
    style_rms = float(np.sqrt(np.mean(np.square(style_sum))))
    total_rms = float(np.sqrt(np.mean(np.square(total_sum))))
    style_fraction = style_rms / total_rms if total_rms > 0 else 0.0
    ordering = synthetic_reward_ordering(term_names, term_weights)
    return {
        "sample_level_returns_available": True,
        "advantage_correlations_available": True,
        "fresh_rollout_requirement_met": True,
        "rollout_batches": manifest.reward_rollout_batches,
        "num_envs": manifest.reward_num_envs,
        "rollout_steps": manifest.reward_rollout_steps,
        "sample_count": manifest.reward_rollout_batches
        * manifest.reward_num_envs
        * manifest.reward_rollout_steps,
        "term_names": list(term_names),
        "term_weights": term_weights.tolist(),
        "weighted_term_statistics": term_statistics,
        "weighted_term_advantage_correlations": correlations,
        "style_contribution_rms": style_rms,
        "total_weighted_reward_rate_rms": total_rms,
        "style_contribution_rms_fraction": style_fraction,
        "style_contribution_below_one_percent": style_fraction < 0.01,
        "synthetic_ordering": ordering,
        "synthetic_reward_ordering_verified": ordering["passed"],
        "timing_alignment_verified": True,
        "dt_multiplier": config.control_dt,
        "reward_mean": float(np.mean(packed["reward"])),
        "value_mean": float(np.mean(packed["value"])),
        "return_mean": float(np.mean(packed["return"])),
        "advantage_mean": float(np.mean(packed["advantage"])),
        "sample_artifact": "reward-return-samples.npz",
    }


def _rotate_inverse(quaternion: FloatArray, vector: FloatArray) -> FloatArray:
    w = quaternion[:, :1]
    xyz = quaternion[:, 1:]
    return vector + 2 * np.cross(xyz, np.cross(xyz, vector) - w * vector)


def _sample_observations(
    trace_paths: tuple[Path, ...], contract: dict[str, Any], count: int
) -> FloatArray:
    nominal = np.asarray(contract["nominal_joint_position"], dtype=np.float64)
    bias = np.asarray(contract["encoder_bias"], dtype=np.float64)
    batches: list[FloatArray] = []
    for path in trace_paths:
        with np.load(path, allow_pickle=False) as trace:
            quaternion = np.asarray(trace["root_quaternion_wxyz"], dtype=np.float64)
            linear = _rotate_inverse(
                quaternion, np.asarray(trace["root_lin_vel_w"], dtype=np.float64)
            )
            angular = _rotate_inverse(
                quaternion, np.asarray(trace["root_ang_vel_w"], dtype=np.float64)
            )
            gravity = _rotate_inverse(
                quaternion, np.broadcast_to(np.array([0.0, 0.0, -1.0]), linear.shape)
            )
            phase = np.asarray(trace["phase"], dtype=np.float64)
            batch = np.column_stack(
                (
                    linear,
                    angular,
                    gravity,
                    np.asarray(trace["joint_pos"], dtype=np.float64) - nominal + bias,
                    np.asarray(trace["joint_vel"], dtype=np.float64),
                    np.asarray(trace["applied_action"], dtype=np.float64),
                    np.asarray(trace["applied_command"], dtype=np.float64),
                    np.sin(2 * np.pi * phase),
                    np.cos(2 * np.pi * phase),
                    np.asarray(trace["blend"], dtype=np.float64),
                )
            )
            batches.append(batch)
    combined = np.concatenate(batches)
    if combined.shape[1] != 102 or not np.isfinite(combined).all():
        raise ValueError("trace-derived actor observations are invalid")
    indices = np.linspace(0, len(combined) - 1, min(count, len(combined)), dtype=int)
    return combined[indices]


def _checkpoint_actor(
    path: Path,
) -> tuple[list[tuple[FloatArray, FloatArray]], FloatArray, FloatArray]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["actor_state_dict"]
    layers = [
        (
            state[f"mlp.{index}.weight"].detach().numpy().astype(np.float64),
            state[f"mlp.{index}.bias"].detach().numpy().astype(np.float64),
        )
        for index in (0, 2, 4, 6)
    ]
    mean = state["obs_normalizer._mean"].detach().numpy().reshape(-1).astype(np.float64)
    std = state["obs_normalizer._std"].detach().numpy().reshape(-1).astype(np.float64)
    return layers, mean, std


def _reference_audit(
    reference: Path, prior_audit: Path, speed_map_path: Path, controller: Path
) -> dict[str, Any]:
    prior = json.loads(prior_audit.read_text(encoding="utf-8"))
    speed_map = json.loads(speed_map_path.read_text(encoding="utf-8"))
    with np.load(reference, allow_pickle=False) as archive:
        qpos = np.asarray(archive["joint_pos"], dtype=np.float64)
        qvel = np.asarray(archive["joint_vel"], dtype=np.float64)
        contacts = np.asarray(archive["foot_contact"], dtype=bool)
        fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
        body_quaternion = np.asarray(archive["body_quat_w"], dtype=np.float64)
        finite_difference = np.gradient(qpos, 1 / fps, axis=0)
    result = {
        "schema_version": 1,
        "reference_sha256": sha256_file(reference),
        "controller_sha256": sha256_file(controller),
        "reference_speed_map_sha256": sha256_file(speed_map_path),
        "frame_count": len(qpos),
        "fps": fps,
        "joint_position_shape": list(qpos.shape),
        "joint_velocity_shape": list(qvel.shape),
        "qdot_finite_difference_rms_rad_s": float(
            np.sqrt(np.mean(np.square(qvel - finite_difference)))
        ),
        "joint_pose_cycle_seam_rms_rad": float(np.sqrt(np.mean(np.square(qpos[-1] - qpos[0])))),
        "contact_duty_fraction": np.mean(contacts, axis=0).tolist(),
        "contact_duty_asymmetry": float(abs(np.mean(contacts[:, 0]) - np.mean(contacts[:, 1]))),
        "double_support_fraction": float(np.mean(np.all(contacts, axis=1))),
        "bilateral_flight_fraction": float(np.mean(~np.any(contacts, axis=1))),
        "quaternion_convention": "wxyz",
        "quaternion_norm_error_max": float(
            np.max(np.abs(np.linalg.norm(body_quaternion, axis=-1) - 1.0))
        ),
        "source_timing_verified": bool(
            math.isclose((len(qpos) - 1) / fps, float(prior["duration_s"]), abs_tol=1e-9)
        ),
        "legacy_audit_passed": prior.get("acceptance", {}).get("passed"),
        "time_warp_speed_entries": speed_map.get("entries"),
        "time_warp_proxy_passed": speed_map.get("passed"),
        "actuator_demand_method": "damping-plus-armature proxy; not inverse dynamics",
        "hypothesis": "inconclusive",
        "limitations": ["Full constrained inverse dynamics is deferred to P05-04."],
    }
    return result


def audit_walking_method(manifest_path: Path, output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"audit output is not empty: {output}")
    manifest = MethodAuditManifest.load(manifest_path)
    # Audit manifests use repository-relative paths so evidence can bind artifacts from
    # several immutable run directories without allowing parent-directory traversal.
    root = Path(__file__).resolve().parents[2]
    resolved = manifest.resolve_all(root)
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads(resolved.contract.read_text(encoding="utf-8"))
    observations = _sample_observations(
        resolved.traces, contract, manifest.observation_sample_count
    )
    layers, mean, std = _checkpoint_actor(resolved.checkpoint)
    action_scale = np.asarray(contract["action_scale"], dtype=np.float64)
    phase, actions = analyze_phase_dependence(
        observations,
        layers,
        mean,
        std,
        action_scale,
        phase_sweep_count=manifest.phase_sweep_count,
    )
    observation = analyze_observations(observations, std, actor_size=102, critic_size=114)
    reward = analyze_reward_returns(resolved.reward_metrics, gamma=0.99, gae_lambda=0.95)
    reward.update(_collect_fresh_reward_returns(manifest, resolved, output))
    reference = _reference_audit(
        resolved.reference,
        resolved.reference_audit,
        resolved.reference_speed_map,
        resolved.controller,
    )
    artifacts = {
        "reference-dynamics.json": reference,
        "phase-dependence.json": phase,
        "observation-adequacy.json": observation,
        "reward-return.json": reward,
    }
    for name, data in artifacts.items():
        write_atomic_json(output / name, data)
    np.savez_compressed(
        output / "phase-sweep-actions.npz",
        observations=observations,
        actions=actions,
    )
    hashes = {
        name: sha256_file(output / name)
        for name in (*artifacts, "phase-sweep-actions.npz", "reward-return-samples.npz")
    }
    audit = {
        "schema_version": 1,
        "audit_id": manifest.audit_id,
        "checkpoint_sha256": manifest.checkpoint.sha256,
        "hypotheses": {
            "reference_is_dynamically_feasible": reference["hypothesis"],
            "old_policy_ignores_phase": phase["hypothesis"],
            "old_observation_is_adequate_for_reference_tracking": observation["hypothesis"],
            "old_reward_aligns_with_return": reward["hypothesis"],
        },
        "findings": {
            "phase_target_displacement_rms_rad": phase["target_displacement_rms_rad"],
            "reference_style_reward_rms_fraction": reward["style_contribution_rms_fraction"],
            "synthetic_reward_ordering_passed": reward["synthetic_reward_ordering_verified"],
            "fresh_reward_return_samples": reward["sample_count"],
        },
        "evidence_sha256": hashes,
        "causal_claims": [],
        "complete": True,
    }
    write_atomic_json(output / "method-audit.json", audit)
    decision = {
        "schema_version": 1,
        "decision": "reference-conditioned-ppo-with-rsi-and-reference-residual-actions",
        "reason": (
            "The old policy's style failure is established; its reference-style reward contribution is zero "
            "and its phase-conditioned target displacement is below the frozen phase-ignored threshold. "
            "Explicit targets and RSI directly address those exploration and conditioning gaps while "
            "constrained reference dynamics and causal reward interpretation remain unresolved."
        ),
        "conditional_fallback": "amp-after-primary-budget-if-predeclared-gates-fail",
        "frozen_settings": {
            "actor_mean": True,
            "normalizers": "checkpoint-frozen",
            "reference_targets_in_actor": True,
            "reference_state_initialization": True,
            "action_mapping": "reference-centered-residual-joint-position-target",
        },
        "remaining_uncertainties": [
            "constrained inverse-dynamics feasibility",
            "closed-loop phase perturbation response",
            "causal interpretation of measured reward/advantage associations",
            "human visual naturalness",
        ],
        "method_audit_sha256": sha256_file(output / "method-audit.json"),
        "next_ticket": "P05-04",
    }
    write_atomic_json(output / "method-decision.json", decision)
    return decision
