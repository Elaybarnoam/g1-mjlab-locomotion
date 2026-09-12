"""Bounded simulator adapter for walking measurement schema 2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import sha256_file
from .config import ResolvedRunConfig, WalkingTrainingProfile
from .gait_evaluation.diagnostic_plot import render_measurement_svg
from .gait_evaluation.scenarios import ScenarioSet, load_scenario_set
from .gait_evaluation.walking_v2 import (
    CONTROL_FIELDS,
    PhysicsTraceV2,
    TraceMetadataV2,
    WalkingTraceV2,
    evaluate_walking_v2,
    load_walking_criteria_v2,
    save_trace_v2,
)
from .motion.contact import ContactProfile
from .tasks import TaskCapability, get_task


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _run_identities(checkpoint: Path) -> dict[str, str]:
    run = checkpoint.parent.parent
    contract_path = run / "contract.json"
    source_path = run / "source.zip"
    bundle_path = run / "policy-bundle.json"
    for path in (contract_path, source_path, bundle_path):
        path.resolve(strict=True)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("checkpoint") == checkpoint.name and sha256_file(checkpoint) != bundle.get(
        "checkpoint_sha256"
    ):
        raise ValueError("checkpoint identity differs from its policy bundle")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    controller_fields = {
        key: contract.get(key)
        for key in (
            "action_semantics",
            "actuator_semantics",
            "action_scale",
            "stiffness",
            "damping",
            "armature",
            "force_limits",
            "encoder_bias",
        )
    }
    return {
        "checkpoint": sha256_file(checkpoint),
        "source": sha256_file(source_path),
        "controller": _canonical_hash(controller_fields),
    }


def _require_parallel_scenario(scenarios: ScenarioSet) -> None:
    if not 1 <= len(scenarios.scenarios) <= 4:
        raise ValueError("detailed diagnosis requires 1..4 scenarios")
    first = scenarios.scenarios[0]
    signature = (first.initialization, first.seed, first.segments)
    if any(
        (item.initialization, item.seed, item.segments) != signature for item in scenarios.scenarios
    ):
        raise ValueError(
            "parallel diagnostic scenarios must share seed, initialization, and segments"
        )


def _schedule_speed(scenarios: ScenarioSet, step: int, control_dt: float) -> float:
    end = 0
    for segment in scenarios.scenarios[0].segments:
        end += round(segment.duration_s / control_dt)
        if step < end:
            return segment.command[0]
    raise IndexError("diagnostic schedule is shorter than its declared horizon")


def _initial_state_hash(robot: Any, environment: int) -> str:
    import torch

    values = (
        torch.cat(
            (
                robot.data.root_link_pose_w[environment],
                robot.data.root_link_vel_w[environment],
                robot.data.joint_pos[environment],
                robot.data.joint_vel[environment],
            )
        )
        .detach()
        .cpu()
        .numpy()
    )
    return hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()


def _append(records: list[dict[str, list[Any]]], name: str, value: Any, active: list[bool]) -> None:
    array = value.detach().cpu().numpy()
    for environment, record in enumerate(records):
        if active[environment]:
            record[name].append(array[environment].copy())


def diagnose_walking(
    config: ResolvedRunConfig,
    checkpoint: Path,
    scenarios_path: Path,
    output: Path,
    *,
    criteria_path: Path,
    physics_trace: bool = False,
    video: bool = False,
) -> dict[str, Any]:
    """Replay up to four worlds and write schema-2 contact diagnostics."""
    get_task(config.task_id).require(TaskCapability.WALKING_EVALUATION)
    checkpoint = checkpoint.resolve(strict=True)
    scenarios = load_scenario_set(scenarios_path.resolve(strict=True), control_dt=config.control_dt)
    _require_parallel_scenario(scenarios)
    criteria = load_walking_criteria_v2(criteria_path.resolve(strict=True))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"diagnostic output is not empty: {output}")
    identities = _run_identities(checkpoint)
    reference_path = (
        Path(__file__).resolve().parents[2] / "configs/walking-v1/reference/g1-walk-a057-cycle.npz"
    )
    reference_sha256 = sha256_file(reference_path.resolve(strict=True))
    output.mkdir(parents=True, exist_ok=True)

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner

    from .environment import build_train_config
    from .rl_adapter import MjlabVecEnvWrapper
    from .tasks.walking_contact import (
        PhysicsContactRecorder,
        TorchContactState,
        add_diagnostic_contact_sensor,
    )
    from .tasks.walking_mdp import WalkingCommand, heading_frame_delta

    first = scenarios.scenarios[0]
    horizon_s = first.horizon_s
    horizon_steps = round(horizon_s / config.control_dt)
    eval_config = replace(
        config,
        seed=first.seed,
        num_envs=len(scenarios.scenarios),
        episode_length_s=horizon_s + config.control_dt,
    )
    reference_initialization = first.initialization != "standing"
    evaluation_profile = WalkingTrainingProfile(
        1,
        f"diagnostic-{first.initialization}",
        0.0 if reference_initialization else 1.0,
        reference_initialization,
        first.initialization == "reference",
    )
    train_cfg = build_train_config(
        eval_config, output, randomized_reset=False, walking_profile=evaluation_profile
    )
    train_cfg.env.auto_reset = False
    if physics_trace:
        add_diagnostic_contact_sensor(train_cfg.env)
    env = ManagerBasedRlEnv(
        cfg=train_cfg.env, device=config.device, render_mode="rgb_array" if video else None
    )
    num_envs = len(scenarios.scenarios)
    records: list[dict[str, list[Any]]] = [
        {name: [] for name in CONTROL_FIELDS} for _ in range(num_envs)
    ]
    events: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
    active = [True] * num_envs
    reasons: list[str | None] = [None] * num_envs
    reset_counter = torch.zeros(num_envs, dtype=torch.int64, device=config.device)
    video_frames: list[np.ndarray] = []
    recorder = PhysicsContactRecorder(env) if physics_trace else None
    if recorder is not None:
        original_compute_substep = env.metrics_manager.compute_substep

        def compute_substep_with_trace() -> None:
            original_compute_substep()
            recorder.capture()

        env.metrics_manager.compute_substep = compute_substep_with_trace
    try:
        wrapped = MjlabVecEnvWrapper(env, clip_actions=config.action_clip)
        runner = MjlabOnPolicyRunner(wrapped, asdict(train_cfg.agent), device=config.device)
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=config.device
        )
        policy = runner.get_inference_policy(device=config.device)
        env.reset(seed=first.seed)
        observations = wrapped.get_observations()
        command = env.command_manager.get_term("twist")
        if not isinstance(command, WalkingCommand):
            raise TypeError("walking diagnostic requires WalkingCommand")
        robot = env.scene["robot"]
        ankle_ids = torch.tensor(
            [
                robot.body_names.index("left_ankle_roll_link"),
                robot.body_names.index("right_ankle_roll_link"),
            ],
            device=config.device,
        )
        site_ids = torch.tensor(
            [robot.site_names.index("left_foot"), robot.site_names.index("right_foot")],
            device=config.device,
        )
        initial_hashes = [_initial_state_hash(robot, index) for index in range(num_envs)]
        contact_sensor = env.scene["feet_ground_contact"]
        if contact_sensor.data.force is None:
            raise RuntimeError("feet contact sensor does not expose force")
        force = torch.abs(contact_sensor.data.force[:, :, 2])
        contact_state = TorchContactState(num_envs, ContactProfile(), config.device)
        contact_state.reset(force)
        with torch.inference_mode():
            for step in range(horizon_steps):
                command.set_requested_forward_speed(
                    _schedule_speed(scenarios, step, config.control_dt)
                )
                raw_actions = policy(observations)
                observations, _, _, _ = wrapped.step(raw_actions)
                action_term = env.action_manager.get_term("joint_pos")
                applied_actions = action_term.raw_action
                q_target = action_term.offset + action_term.scale * applied_actions
                force = torch.abs(contact_sensor.data.force[:, :, 2])
                sole_position = robot.data.site_pos_w[:, site_ids]
                contact = contact_state.update(force, config.control_dt, sole_position[:, :, 2])
                reference_body = command.interpolate(command.reference_body_position)
                reference_ids = torch.tensor(
                    [
                        command.reference_body_names.index("left_ankle_roll_link"),
                        command.reference_body_names.index("right_ankle_roll_link"),
                    ],
                    device=config.device,
                )
                expected_foot = heading_frame_delta(
                    reference_body[:, reference_ids] - reference_body[:, 0:1],
                    command.interpolate(command.reference_body_quaternion)[:, 0],
                )
                terminated = env.termination_manager.terminated.clone()
                truncated = env.termination_manager.time_outs.clone()
                finite = (
                    torch.isfinite(raw_actions).all(dim=1)
                    & torch.isfinite(robot.data.joint_pos).all(dim=1)
                    & torch.isfinite(robot.data.root_link_pos_w).all(dim=1)
                )
                values = {
                    "time_s": torch.full(
                        (num_envs,), step * config.control_dt, device=config.device
                    ),
                    "requested_command": command.requested_command,
                    "applied_command": command.command,
                    "root_position_w": robot.data.root_link_pos_w,
                    "root_quaternion_wxyz": robot.data.root_link_quat_w,
                    "root_lin_vel_w": robot.data.root_link_lin_vel_w,
                    "root_ang_vel_w": robot.data.root_link_ang_vel_w,
                    "phase": command.phase,
                    "blend": command.blend,
                    "expected_contact": command.foot_contact,
                    "joint_pos": robot.data.joint_pos,
                    "joint_vel": robot.data.joint_vel,
                    "raw_action": raw_actions,
                    "applied_action": applied_actions,
                    "q_target": q_target,
                    "actuator_torque": robot.data.qfrc_actuator,
                    "ankle_position_w": robot.data.body_link_pos_w[:, ankle_ids],
                    "sole_position_w": sole_position,
                    "sole_quaternion_wxyz": robot.data.site_quat_w[:, site_ids],
                    "raw_contact": contact.raw_contact,
                    "debounced_contact": contact.stable_contact,
                    "normal_force_n": force,
                    "stance_age_s": contact.stance_age_s,
                    "swing_age_s": contact.swing_age_s,
                    "expected_foot_position_heading": expected_foot,
                    "swing_peak_height_m": contact.swing_peak_clearance_m,
                    "finite": finite,
                    "terminated": terminated,
                    "truncated": truncated,
                    "reset_counter": reset_counter,
                }
                for name, value in values.items():
                    _append(records, name, value, active)
                for environment in range(num_envs):
                    if not active[environment]:
                        continue
                    for foot in range(2):
                        for kind, mask in (
                            ("touchdown", contact.touchdown),
                            ("liftoff", contact.liftoff),
                        ):
                            if bool(mask[environment, foot]):
                                events[environment].append(
                                    {
                                        "foot": foot,
                                        "kind": kind,
                                        "first_crossing_time_s": float(
                                            contact.first_crossing_time_s[environment, foot]
                                        ),
                                        "confirmation_time_s": float(
                                            contact.confirmation_time_s[environment, foot]
                                        ),
                                        "valid": bool(contact.valid_touchdown[environment, foot])
                                        if kind == "touchdown"
                                        else True,
                                        "prior_stable_duration_s": float(
                                            contact.prior_stable_duration_s[environment, foot]
                                        ),
                                        "swing_peak_clearance_m": float(
                                            contact.swing_peak_clearance_m[environment, foot]
                                        )
                                        if kind == "touchdown"
                                        else None,
                                    }
                                )
                if video and step % 2 == 0:
                    frame = env.render()
                    if frame is not None:
                        array = np.asarray(frame)
                        video_frames.append(array[0] if array.ndim == 4 else array)
                done = terminated | truncated | ~finite
                for environment in torch.nonzero(done, as_tuple=False).flatten().cpu().tolist():
                    active[environment] = False
                    if not bool(finite[environment]):
                        reasons[environment] = "nonfinite_state"
                    elif bool(truncated[environment]):
                        reasons[environment] = "time_out"
                    else:
                        for name in train_cfg.env.terminations:
                            if name != "time_out" and bool(
                                env.termination_manager.get_term(name)[environment]
                            ):
                                reasons[environment] = name
                                break
                        reasons[environment] = reasons[environment] or "terminated"
                done_ids = torch.nonzero(done, as_tuple=False).flatten()
                if len(done_ids):
                    env.reset(env_ids=done_ids)
                    reset_counter[done_ids] += 1
                    observations = wrapped.get_observations()
                if not any(active):
                    break
    finally:
        env.close()

    trial_summaries: list[dict[str, Any]] = []
    for environment, record in enumerate(records):
        arrays = {name: np.asarray(values) for name, values in record.items()}
        frames = len(arrays["time_s"])
        arrays["time_s"] = np.arange(frames, dtype=float) * config.control_dt
        metadata = TraceMetadataV2(
            schema_version=2,
            checkpoint_sha256=identities["checkpoint"],
            source_sha256=identities["source"],
            controller_sha256=identities["controller"],
            reference_sha256=reference_sha256,
            scenario_sha256=scenarios.sha256,
            control_dt=config.control_dt,
            physics_dt=config.physics_dt,
            seed=first.seed,
            initialization=first.initialization,
            terminated=bool(arrays["terminated"].any()),
            reason=reasons[environment],
            completed_horizon_s=frames * config.control_dt,
            field_definitions={name: "schema-2 SI field" for name in CONTROL_FIELDS},
            initial_state_sha256=initial_hashes[environment],
        )
        trace = WalkingTraceV2(metadata, arrays)
        trace_path = output / f"trace-{environment:03d}.npz"
        save_trace_v2(trace, trace_path, output / f"trace-{environment:03d}.metadata.json")
        physics = PhysicsTraceV2.unsupported("physics trace was not requested")
        if recorder is not None:
            physics_arrays = recorder.arrays_for_environment(environment, config.physics_dt)
            physics_arrays = {
                name: value[: frames * config.decimation] for name, value in physics_arrays.items()
            }
            physics = PhysicsTraceV2(True, None, physics_arrays)
            physics.validate()
            np.savez_compressed(output / f"physics-trace-{environment:03d}.npz", **physics_arrays)
        render_measurement_svg(trace, physics, output / f"diagnostic-{environment:03d}.svg")
        summary = evaluate_walking_v2(trace, criteria, physics, planned_horizon_s=horizon_s)
        (output / f"events-{environment:03d}.json").write_text(
            json.dumps(events[environment], indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        trial_summaries.append(
            {
                "trial_id": environment,
                "trace": trace_path.name,
                "physics_trace": f"physics-trace-{environment:03d}.npz" if recorder else None,
                **asdict(summary),
            }
        )
    result: dict[str, Any] = {
        "schema_version": 2,
        "task_id": config.task_id,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": identities["checkpoint"],
        "scenario_set": scenarios.name,
        "scenario_sha256": scenarios.sha256,
        "planned": num_envs,
        "completed": len(trial_summaries),
        "trials": trial_summaries,
    }
    physical = [trial["physical_slip_rms_m_s"] for trial in trial_summaries]
    ankle = [trial["ankle_speed_proxy_rms_m_s"] for trial in trial_summaries]
    raw_rates = [trial["raw_transition_rate_s"] for trial in trial_summaries]
    causes: list[str] = []
    if any(value is None for value in physical):
        causes.append("insufficient_physical_contact_evidence")
    else:
        if float(np.mean(physical)) > criteria.maximum_physical_slip_rms_m_s:
            causes.append("physical_slip")
        if (
            float(np.mean(ankle)) > criteria.maximum_physical_slip_rms_m_s
            and float(np.mean(physical)) <= criteria.maximum_physical_slip_rms_m_s
        ):
            causes.append("ankle_sensor_artifact")
    if float(np.mean(raw_rates)) > criteria.maximum_raw_transition_rate_s:
        causes.append("physical_contact_chatter")
    audit = {
        "schema_version": 2,
        "checkpoint_sha256": identities["checkpoint"],
        "causes": causes or ["no_threshold_exceedance"],
        "conclusion": "mixed_causes" if len(causes) > 1 else (causes[0] if causes else "none"),
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "measurement-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    if video:
        if not video_frames:
            raise RuntimeError("diagnostic video requested but renderer produced no frames")
        import imageio.v2 as imageio

        with imageio.get_writer(
            output / "deterministic.mp4", fps=25, codec="libx264", quality=8, macro_block_size=None
        ) as writer:
            for frame in video_frames:
                writer.append_data(frame)  # type: ignore[attr-defined]
    return result
