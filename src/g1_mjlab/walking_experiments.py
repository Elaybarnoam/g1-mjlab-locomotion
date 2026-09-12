"""Frozen development scenarios and predeclared Stage 19 comparison rules."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import numpy.typing as npt

from .artifacts import sha256_file, write_atomic_json
from .config import ResolvedRunConfig, WalkingTrainingProfile

FloatArray = npt.NDArray[np.float64]

_SCHEDULE = (
    {"duration_s": 2.0, "command": [0.0, 0.0, 0.0]},
    {"duration_s": 10.0, "command": [0.6, 0.0, 0.0]},
    {"duration_s": 3.0, "command": [0.0, 0.0, 0.0]},
)


@dataclass(frozen=True)
class Stage19DecisionRule:
    """Predeclared partial-progress and early-stop thresholds for Stage 19."""

    schema_version: int = 1
    required_trials: int = 16
    expected_cadence_steps_s: float = 0.972719696969697
    expected_step_length_m: float = 0.6063803444314848
    comparison_walk_duration_s: float = 7.0
    slip_scale_m_s: float = 0.12
    error_denominator_floor: float = 0.05
    minimum_relative_improvement: float = 0.15
    maximum_relative_worsening: float = 0.10
    maximum_command_rms_worsening_m_s: float = 0.03
    maximum_chatter_ratio: float = 1.05
    low_function_threshold: int = 12
    low_function_patience: int = 2
    no_progress_patience: int = 3

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.required_trials != 16:
            raise ValueError("unsupported Stage 19 decision-rule schema")
        numeric = (
            self.expected_cadence_steps_s,
            self.expected_step_length_m,
            self.comparison_walk_duration_s,
            self.slip_scale_m_s,
            self.error_denominator_floor,
            self.minimum_relative_improvement,
            self.maximum_relative_worsening,
            self.maximum_command_rms_worsening_m_s,
            self.maximum_chatter_ratio,
        )
        if any(not math.isfinite(value) or value <= 0 for value in numeric):
            raise ValueError("decision-rule numeric thresholds must be finite and positive")
        if self.low_function_patience <= 0 or self.no_progress_patience <= 0:
            raise ValueError("decision-rule patience must be positive")


def load_stage19_decision_rule(path: Path) -> Stage19DecisionRule:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = set(Stage19DecisionRule.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("Stage 19 decision-rule fields do not match schema")
    return Stage19DecisionRule(**raw)


def _valid_median(trials: list[dict[str, Any]], field: str) -> float | None:
    values = [float(trial[field]) for trial in trials if trial.get(field) is not None]
    return median(values) if values else None


def _cohort(summary: dict[str, Any], rule: Stage19DecisionRule) -> dict[str, Any]:
    trials = summary.get("trials")
    if (
        summary.get("schema_version") != 2
        or summary.get("planned") != rule.required_trials
        or summary.get("completed") != rule.required_trials
        or not isinstance(trials, list)
        or len(trials) != rule.required_trials
    ):
        raise ValueError("development evaluation must contain exactly 16 schema-2 trials")
    typed_trials: list[dict[str, Any]] = []
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("development trial must be an object")
        typed_trials.append(trial)
    expected_events = rule.expected_cadence_steps_s * rule.comparison_walk_duration_s
    cadence = _valid_median(typed_trials, "cadence_steps_s")
    step = _valid_median(typed_trials, "step_length_median_m")
    slip = _valid_median(typed_trials, "physical_slip_rms_m_s")
    repeats = _valid_median(typed_trials, "same_side_repeat_count")
    errors = {
        "cadence": abs(cadence / rule.expected_cadence_steps_s - 1)
        if cadence is not None
        else None,
        "step": abs(step / rule.expected_step_length_m - 1) if step is not None else None,
        "slip": slip / rule.slip_scale_m_s if slip is not None else None,
        "repeat": repeats / max(expected_events, 1.0) if repeats is not None else None,
    }
    function_count = sum(bool(trial.get("functional_passed")) for trial in typed_trials)
    style_count = sum(
        bool(trial.get("functional_passed")) and bool(trial.get("style_passed"))
        for trial in typed_trials
    )
    return {
        "function_pass_count": function_count,
        "function_gate_passed": function_count == rule.required_trials,
        "function_style_pass_count": style_count,
        "median_command_rms_m_s": _valid_median(typed_trials, "command_rms_m_s"),
        "median_raw_transition_rate_s": _valid_median(typed_trials, "raw_transition_rate_s"),
        "errors": errors,
    }


def compare_development_evaluation(
    source_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    rule: Stage19DecisionRule,
) -> dict[str, Any]:
    """Compare one checkpoint to the common source using only declared physical metrics."""
    source = _cohort(source_summary, rule)
    candidate = _cohort(candidate_summary, rule)
    relative_changes: dict[str, float | None] = {}
    improved = 0
    worsened: list[str] = []
    for name, source_error in source["errors"].items():
        candidate_error = candidate["errors"][name]
        if source_error is None or candidate_error is None:
            relative_changes[name] = None
            worsened.append(name)
            continue
        change = (candidate_error - source_error) / max(source_error, rule.error_denominator_floor)
        relative_changes[name] = change
        improved += change <= -rule.minimum_relative_improvement
        if change > rule.maximum_relative_worsening:
            worsened.append(name)
    source_command = source["median_command_rms_m_s"]
    candidate_command = candidate["median_command_rms_m_s"]
    command_preserved = (
        source_command is not None
        and candidate_command is not None
        and candidate_command <= source_command + rule.maximum_command_rms_worsening_m_s
    )
    source_chatter = source["median_raw_transition_rate_s"]
    candidate_chatter = candidate["median_raw_transition_rate_s"]
    chatter_preserved = (
        source_chatter is not None
        and candidate_chatter is not None
        and candidate_chatter <= source_chatter * rule.maximum_chatter_ratio
    )
    extension_eligible = bool(
        candidate["function_gate_passed"]
        and improved >= 2
        and not worsened
        and command_preserved
        and chatter_preserved
    )
    return {
        "schema_version": 1,
        **candidate,
        "source": source,
        "relative_error_changes": relative_changes,
        "improved_metric_count": improved,
        "worsened_metrics": worsened,
        "command_tracking_preserved": command_preserved,
        "raw_chatter_preserved": chatter_preserved,
        "extension_eligible": extension_eligible,
    }


def stop_decision(comparisons: list[dict[str, Any]], rule: Stage19DecisionRule) -> dict[str, Any]:
    """Apply the predeclared consecutive-evaluation stop rules to one arm."""
    if len(comparisons) >= rule.low_function_patience and all(
        int(item["function_pass_count"]) <= rule.low_function_threshold
        for item in comparisons[-rule.low_function_patience :]
    ):
        return {"stop": True, "reason": "two_low_function_evaluations"}
    if len(comparisons) >= rule.no_progress_patience and all(
        not bool(item["extension_eligible"]) for item in comparisons[-rule.no_progress_patience :]
    ):
        return {"stop": True, "reason": "three_no_progress"}
    return {"stop": False, "reason": None}


def _trial_max_error(trial: dict[str, Any], rule: Stage19DecisionRule) -> float:
    values = (
        trial.get("cadence_steps_s"),
        trial.get("step_length_median_m"),
        trial.get("physical_slip_rms_m_s"),
        trial.get("same_side_repeat_count"),
    )
    if any(value is None for value in values):
        return math.inf
    cadence_value, step_value, slip_value, repeats_value = values
    assert cadence_value is not None
    assert step_value is not None
    assert slip_value is not None
    assert repeats_value is not None
    cadence = float(cadence_value)
    step = float(step_value)
    slip = float(slip_value)
    repeats = float(repeats_value)
    return max(
        abs(cadence / rule.expected_cadence_steps_s - 1),
        abs(step / rule.expected_step_length_m - 1),
        slip / rule.slip_scale_m_s,
        repeats / max(rule.expected_cadence_steps_s * rule.comparison_walk_duration_s, 1.0),
    )


def _selection_metrics(summary: dict[str, Any], rule: Stage19DecisionRule) -> dict[str, Any]:
    cohort = _cohort(summary, rule)
    trials: list[dict[str, Any]] = summary["trials"]
    strata: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        name = str(trial.get("scenario_name", ""))
        category = name.split("-", 2)[1] if name.count("-") >= 2 else "unknown"
        strata.setdefault(category, []).append(trial)
    stratum_passes = {
        name: sum(
            bool(trial.get("functional_passed")) and bool(trial.get("style_passed"))
            for trial in values
        )
        for name, values in strata.items()
    }
    finite_errors = [
        value for trial in trials if math.isfinite(value := _trial_max_error(trial, rule))
    ]
    return {
        **cohort,
        "stratum_function_style_passes": stratum_passes,
        "worst_stratum_function_style_passes": min(stratum_passes.values()),
        "median_trial_max_error": median(finite_errors) if finite_errors else None,
    }


def build_stage19_experiment_table(
    source_evaluation_path: Path,
    arm_campaigns: dict[str, Path],
    decision_rule_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Freeze the complete A/B/C evidence table and deterministic development decision."""
    if output.exists():
        raise FileExistsError(f"experiment table already exists: {output}")
    if set(arm_campaigns) != {"a", "b", "c"}:
        raise ValueError("Stage 19 experiment table requires exactly arms a, b, and c")
    source = json.loads(source_evaluation_path.read_text(encoding="utf-8"))
    rule = load_stage19_decision_rule(decision_rule_path)
    rows: list[dict[str, Any]] = []
    for arm, campaign in sorted(arm_campaigns.items()):
        state = json.loads((campaign / "state.json").read_text(encoding="utf-8"))
        for evaluation_dir in sorted((campaign / "evaluations").glob("segment-*")):
            comparison_path = evaluation_dir / "development-comparison.json"
            summary_path = evaluation_dir / "summary.json"
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "arm": arm,
                    "campaign": str(campaign.resolve()),
                    "campaign_status": state["status"],
                    "evaluation": str(evaluation_dir.resolve()),
                    "evaluation_sha256": sha256_file(summary_path),
                    "completed_updates": comparison["completed_updates"],
                    "checkpoint": comparison["checkpoint"],
                    "checkpoint_sha256": comparison["checkpoint_sha256"],
                    "extension_eligible": comparison["extension_eligible"],
                    "comparison": comparison,
                    "selection": _selection_metrics(summary, rule),
                }
            )
    if not rows:
        raise ValueError("arm campaigns contain no development evaluations")

    def selection_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
        selection = row["selection"]
        error = selection["median_trial_max_error"]
        return (
            -float(selection["function_style_pass_count"]),
            -float(selection["worst_stratum_function_style_passes"]),
            float(error) if error is not None else math.inf,
            int(row["completed_updates"]),
        )

    best = min(rows, key=selection_key)
    final_by_arm = {
        arm: max(
            (row for row in rows if row["arm"] == arm),
            key=lambda row: row["completed_updates"],
        )
        for arm in arm_campaigns
    }
    arm_b_slip = final_by_arm["b"]["comparison"]["errors"]["slip"]
    arm_c_slip = final_by_arm["c"]["comparison"]["errors"]["slip"]
    d_justified = bool(
        arm_b_slip is not None
        and arm_c_slip is not None
        and arm_c_slip <= arm_b_slip * (1 - rule.minimum_relative_improvement)
        and final_by_arm["c"]["comparison"]["function_gate_passed"]
    )
    extension_rows = [row for row in rows if row["extension_eligible"]]
    development_qualified = any(
        row["selection"]["function_style_pass_count"] == rule.required_trials for row in rows
    )
    result = {
        "schema_version": 1,
        "outcome": "candidate" if development_qualified else "failed_hypothesis",
        "development_qualified": development_qualified,
        "selected_candidate": best if development_qualified else None,
        "best_observed": best,
        "extension_authorized": bool(extension_rows),
        "extension_candidates": extension_rows,
        "arm_d_authorized": d_justified,
        "arm_d_decision": (
            "authorized_by_at_least_15_percent_slip_improvement"
            if d_justified
            else "not_authorized_physical_slip_did_not_improve_over_arm_b"
        ),
        "method_decision_required": not development_qualified,
        "method_decision": {
            "further_ppo_reward_training_authorized": False,
            "reason": "two distinct mechanical hypotheses failed the extension rule",
            "required_next_audits": [
                "reference_feasibility",
                "policy_phase_dependence",
                "observation_adequacy",
                "reward_signal_scale",
            ],
            "amp_status": "not_specified_and_not_authorized",
        },
        "source_evaluation": str(source_evaluation_path.resolve()),
        "source_evaluation_sha256": sha256_file(source_evaluation_path),
        "source_selection": _selection_metrics(source, rule),
        "decision_rule": str(decision_rule_path.resolve()),
        "decision_rule_sha256": sha256_file(decision_rule_path),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic_json(output, result)
    return result


def _quaternion_multiply(left: FloatArray, right: FloatArray) -> FloatArray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _scenario(
    index: int,
    category: str,
    qpos: FloatArray,
    qvel: FloatArray,
    *,
    seed: int,
) -> dict[str, Any]:
    return {
        "name": f"{index:02d}-{category}-stand-walk-stop-060",
        "category": category,
        "seed": seed,
        "initialization": "standing",
        "initial_phase": (index % 4) * 0.25,
        "initial_qpos": qpos.tolist(),
        "initial_qvel": qvel.tolist(),
        "segments": [dict(segment) for segment in _SCHEDULE],
    }


def build_development_scenario_payload(
    nominal_qpos: npt.ArrayLike,
    nominal_qvel: npt.ArrayLike,
    joint_position_limits: npt.ArrayLike,
    *,
    seed: int,
) -> dict[str, Any]:
    """Build the exact 16-state development cohort from one canonical standing reset."""
    qpos = np.asarray(nominal_qpos, dtype=np.float64)
    qvel = np.asarray(nominal_qvel, dtype=np.float64)
    limits = np.asarray(joint_position_limits, dtype=np.float64)
    if qpos.shape != (36,) or qvel.shape != (35,) or limits.shape != (29, 2):
        raise ValueError("nominal qpos/qvel or joint limits have the wrong G1 shape")
    if seed < 0 or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("scenario seed and nominal state must be valid")
    # mjlab's soft limits may be narrower than the model's declared default pose.
    # Never pull a default farther outside that envelope, but preserve it exactly.
    lower = np.minimum(limits[:, 0], qpos[7:])
    upper = np.maximum(limits[:, 1], qpos[7:])
    rng = np.random.default_rng(seed)
    scenarios: list[dict[str, Any]] = []
    for _ in range(4):
        scenarios.append(_scenario(len(scenarios), "nominal", qpos.copy(), qvel.copy(), seed=seed))
    for yaw_degrees in (-5.0, -2.5, 2.5, 5.0):
        perturbed = qpos.copy()
        half = math.radians(yaw_degrees) / 2
        yaw = np.asarray((math.cos(half), 0.0, 0.0, math.sin(half)))
        perturbed[3:7] = _quaternion_multiply(yaw, qpos[3:7])
        perturbed[3:7] /= np.linalg.norm(perturbed[3:7])
        scenarios.append(_scenario(len(scenarios), "yaw", perturbed, qvel.copy(), seed=seed))
    for _ in range(4):
        perturbed = qpos.copy()
        perturbation = rng.uniform(-0.015, 0.015, size=29)
        perturbed[7:] = np.clip(qpos[7:] + perturbation, lower, upper)
        scenarios.append(
            _scenario(len(scenarios), "joint_position", perturbed, qvel.copy(), seed=seed)
        )
    for _ in range(4):
        perturbed_velocity = qvel.copy()
        angle = rng.uniform(0.0, 2 * math.pi)
        norm = rng.uniform(0.0, 0.03)
        perturbed_velocity[:2] += norm * np.asarray((math.cos(angle), math.sin(angle)))
        scenarios.append(
            _scenario(len(scenarios), "root_velocity", qpos.copy(), perturbed_velocity, seed=seed)
        )
    return {
        "schema_version": 3,
        "name": "stage19-development-frozen-v3",
        "scenarios": scenarios,
    }


def freeze_development_scenarios(
    config: ResolvedRunConfig, output: Path, *, seed: int = 10042
) -> dict[str, Any]:
    """Capture nominal simulator state, freeze 16 perturbations, and validate them in mjlab."""
    if output.exists():
        raise FileExistsError(f"scenario output already exists: {output}")
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    from .environment import build_train_config
    from .tasks.walking_mdp import WalkingCommand

    eval_config = replace(config, seed=seed, num_envs=16)
    profile = WalkingTrainingProfile(1, "scenario-freeze-standing", 1.0, False, False)
    train_cfg = build_train_config(
        eval_config,
        output.parent / "unused-scenario-freeze",
        randomized_reset=False,
        walking_profile=profile,
    )
    env = ManagerBasedRlEnv(cfg=train_cfg.env, device=config.device, render_mode=None)
    try:
        env.reset(seed=seed)
        robot = env.scene["robot"]
        command = env.command_manager.get_term("twist")
        if not isinstance(command, WalkingCommand):
            raise TypeError("scenario freeze requires WalkingCommand")
        origin = env.scene.env_origins[0]
        root = (
            torch.cat((robot.data.root_link_pose_w[0], robot.data.root_link_vel_w[0]))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        root[:3] -= origin.detach().cpu().numpy()
        qpos = np.concatenate((root[:7], robot.data.joint_pos[0].detach().cpu().numpy()))
        qvel = np.concatenate((root[7:13], robot.data.joint_vel[0].detach().cpu().numpy()))
        limits = robot.data.soft_joint_pos_limits[0].detach().cpu().numpy()
        payload = build_development_scenario_payload(qpos, qvel, limits, seed=seed)

        scenarios = payload["scenarios"]
        qpos_tensor = torch.tensor(
            [item["initial_qpos"] for item in scenarios],
            device=config.device,
            dtype=robot.data.joint_pos.dtype,
        )
        qvel_tensor = torch.tensor(
            [item["initial_qvel"] for item in scenarios],
            device=config.device,
            dtype=robot.data.joint_vel.dtype,
        )
        root_state = torch.cat((qpos_tensor[:, :7], qvel_tensor[:, :6]), dim=1)
        root_state[:, :3] += env.scene.env_origins
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(qpos_tensor[:, 7:], qvel_tensor[:, 6:])
        command.phase[:] = torch.tensor(
            [item["initial_phase"] for item in scenarios], device=config.device
        )
        env.sim.forward()
        terminated = env.termination_manager.compute()
        collision_cfg = env.reward_manager.get_term_cfg("self_collisions")
        self_collisions = collision_cfg.func(env, **collision_cfg.params)
        finite = bool(
            torch.isfinite(robot.data.root_link_pose_w).all()
            and torch.isfinite(robot.data.root_link_vel_w).all()
            and torch.isfinite(robot.data.joint_pos).all()
            and torch.isfinite(robot.data.joint_vel).all()
        )
        validation = {
            "all_finite": finite,
            "termination_count": int(torch.sum(terminated)),
            "self_collision_count": int(torch.sum(self_collisions > 0)),
            "minimum_pelvis_height_m": float(torch.min(robot.data.root_link_pos_w[:, 2])),
        }
        if not finite or validation["termination_count"] or validation["self_collision_count"]:
            raise RuntimeError(f"generated development scenario failed validation: {validation}")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_atomic_json(output, payload)
        verification_path = output.with_suffix(".verification.json")
        write_atomic_json(
            verification_path,
            {
                "schema_version": 1,
                "generation_seed": seed,
                "scenario_sha256": sha256_file(output),
                "validation": validation,
            },
        )
    finally:
        env.close()
    return {
        "schema_version": 1,
        "scenario_path": str(output.resolve()),
        "scenario_sha256": sha256_file(output),
        "scenario_count": 16,
        "verification_path": str(verification_path.resolve()),
        "validation": validation,
    }
