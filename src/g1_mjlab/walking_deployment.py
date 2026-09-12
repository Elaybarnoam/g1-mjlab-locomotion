"""Exact walking export and repository-independent native MuJoCo execution."""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .config import (
    ResolvedRunConfig,
    load_stage19_reward_profile,
    load_walking_training_profile,
)
from .gait_evaluation.scenarios import ScenarioSegment, WalkingScenario, load_scenario_set
from .inference.walking import (
    WalkingPolicySession,
    WalkingSensorState,
    compose_walking_actor_observation,
    walking_joint_targets,
)
from .motion.gait import GaitState


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _host_profile(run: Path) -> tuple[dict[str, Any], Path]:
    mdp = json.loads((run / "mdp.json").read_text(encoding="utf-8"))
    command = mdp["commands"]["twist"]
    profile = {
        "schema_version": 1,
        "reference_speed_m_s": command["reference_speed_m_s"],
        "cycle_duration_s": command["cycle_duration_s"],
        "acceleration_m_s2": command["acceleration_m_s2"],
        "deceleration_m_s2": command["deceleration_m_s2"],
        "blend_rate_s": command["blend_rate_s"],
        "stand_threshold_m_s": command["stand_threshold_m_s"],
        "walk_threshold_m_s": command["walk_threshold_m_s"],
        "reference_file": "reference.npz",
    }
    return profile, Path(command["motion_file"]).resolve(strict=True)


def _resolved_run_config(path: Path) -> ResolvedRunConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("resolved run configuration must be an object")
    fields = set(ResolvedRunConfig.__dataclass_fields__)
    if not fields.issubset(raw) or set(raw) - fields != {"control_dt", "transitions_per_update"}:
        raise ValueError("resolved run configuration fields do not match schema")
    return ResolvedRunConfig(**{name: raw[name] for name in fields})


def export_walking_policy(run: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    """Export one named checkpoint and bind every native dependency by hash."""
    import onnxruntime as ort
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner

    from .environment import build_train_config
    from .rl_adapter import MjlabVecEnvWrapper

    run = run.resolve(strict=True)
    checkpoint = checkpoint if checkpoint.is_absolute() else run / "checkpoints" / checkpoint
    checkpoint = checkpoint.resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"walking bundle output already exists: {output}")
    output.mkdir(parents=True)
    try:
        config = replace(_resolved_run_config(run / "config.json"), num_envs=1)
        walking_profile = load_walking_training_profile(run / "walking-profile.json")
        reward_profile = load_stage19_reward_profile(run / "walking-reward-profile.json")
        train_cfg = build_train_config(
            config,
            output / ".export",
            randomized_reset=False,
            walking_profile=walking_profile,
            walking_reward_profile=reward_profile,
        )
        env = ManagerBasedRlEnv(cfg=train_cfg.env, device=config.device, render_mode=None)
        try:
            wrapped = MjlabVecEnvWrapper(env, clip_actions=config.action_clip)
            runner = MjlabOnPolicyRunner(wrapped, asdict(train_cfg.agent), device=config.device)
            runner.load(
                str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=config.device
            )
            runner.export_policy_to_onnx(str(output), filename="policy.onnx")
        finally:
            env.close()

        shutil.copy2(run / "contract.json", output / "contract.json")
        shutil.copy2(run / "model.mjb", output / "model.mjb")
        host, reference = _host_profile(run)
        shutil.copy2(reference, output / "reference.npz")
        _write_json(output / "host-profile.json", host)
        session = ort.InferenceSession(
            str(output / "policy.onnx"), providers=["CPUExecutionProvider"]
        )
        if session.get_inputs()[0].shape[-1] != 102 or session.get_outputs()[0].shape[-1] != 29:
            raise ValueError("exported walking ONNX signature is not [batch,102] -> [batch,29]")
        files = {
            name: sha256_file(output / name)
            for name in (
                "contract.json",
                "host-profile.json",
                "model.mjb",
                "policy.onnx",
                "reference.npz",
            )
        }
        bundle = {
            "schema_version": 2,
            "task_id": "G1-Walking-Flat-v1",
            "layout_id": "g1-walking-actor-v1",
            "status": "unqualified_development",
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256_file(checkpoint),
            "source_run": str(run),
            "development_command_domain_m_s": [0.0, 0.6],
            "qualified_command_domain_m_s": [],
            "files": files,
        }
        _write_json(output / "walking-policy-bundle.json", bundle)
        WalkingPolicySession.load(output, allow_unqualified=True)
        return bundle
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def _sensor(model: Any, data: Any, contract: dict[str, Any]) -> WalkingSensorState:
    import numpy as np

    pelvis = model.body("robot/pelvis").id
    return WalkingSensorState(
        data.sensor("robot/imu_lin_vel").data.copy(),
        data.sensor("robot/imu_ang_vel").data.copy(),
        data.xmat[pelvis].reshape(3, 3).T @ np.asarray([0.0, 0.0, -1.0]),
        data.qpos[contract["qpos_addresses"]].copy(),
        data.qvel[contract["dof_addresses"]].copy(),
    )


def _reset(model: Any, data: Any, scenario: WalkingScenario, contract: dict[str, Any]) -> None:
    import mujoco

    if scenario.initial_qpos is None or scenario.initial_qvel is None:
        raise ValueError("native walking requires explicit schema-3 scenario state")
    if len(scenario.initial_qpos) != model.nq or len(scenario.initial_qvel) != model.nv:
        raise ValueError("scenario state dimensions do not match bundled model")
    data.qpos[:] = scenario.initial_qpos
    data.qvel[:] = scenario.initial_qvel
    data.ctrl[contract["actuator_ids"]] = data.qpos[contract["qpos_addresses"]]
    mujoco.mj_forward(model, data)


def _requested(segment: tuple[ScenarioSegment, ...], step: int, dt: float) -> float:
    cursor = 0
    for item in segment:
        cursor += round(item.duration_s / dt)
        if step < cursor:
            return item.command[0]
    return segment[-1].command[0]


def evaluate_native_walking(
    policy: Path,
    scenarios: Path,
    output: Path,
    *,
    allow_unqualified: bool = False,
    minimum_horizon_s: float | None = None,
) -> dict[str, Any]:
    """Run explicit physical states through the native-only CPU controller loop."""
    import mujoco
    import numpy as np

    policy = policy.resolve(strict=True)
    session = WalkingPolicySession.load(policy, allow_unqualified=allow_unqualified)
    model = mujoco.MjModel.from_binary_path(str(policy / "model.mjb"))
    scenario_set = load_scenario_set(
        scenarios.resolve(strict=True), control_dt=float(session.contract["control_dt"])
    )
    if output.exists():
        raise FileExistsError(f"native walking output already exists: {output}")
    output.mkdir(parents=True)
    dt = float(session.contract["control_dt"])
    decimation = round(dt / model.opt.timestep)
    if not math.isclose(decimation * model.opt.timestep, dt, abs_tol=1e-9):
        raise ValueError("walking control period is not an integral physics decimation")
    records: list[dict[str, Any]] = []
    for scenario in scenario_set.scenarios:
        data = mujoco.MjData(model)
        _reset(model, data, scenario, session.contract)
        session.reset(
            _sensor(model, data, session.contract), phase=float(scenario.initial_phase or 0.0)
        )
        finite = True
        survived = 0
        pelvis = model.body("robot/pelvis").id
        trace: list[dict[str, Any]] = []
        horizon_s = max(scenario.horizon_s, minimum_horizon_s or 0.0)
        horizon_steps = round(horizon_s / dt)
        for step in range(horizon_steps):
            speed = _requested(scenario.segments, step, dt)
            inference = session.step(speed, _sensor(model, data, session.contract))
            targets = walking_joint_targets(session.contract, inference.action)
            data.ctrl[session.contract["actuator_ids"]] = targets
            for _ in range(decimation):
                mujoco.mj_step(model, data)
            finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
            survived = step + 1
            if scenario is scenario_set.scenarios[0]:
                trace.append(
                    {
                        "step": step,
                        "time_s": (step + 1) * dt,
                        "requested_speed_m_s": speed,
                        "applied_speed_m_s": inference.command.next_applied_forward_speed_m_s,
                        "phase": inference.command.next_phase,
                        "pelvis_height_m": float(data.xpos[pelvis, 2]),
                        "action": inference.action.tolist(),
                        "joint_target_rad": targets.tolist(),
                    }
                )
            if not finite or data.xpos[pelvis, 2] < 0.35:
                break
        if trace:
            (output / "trial-0.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in trace), encoding="utf-8"
            )
        records.append(
            {
                "name": scenario.name,
                "survived_seconds": survived * dt,
                "planned_seconds": horizon_s,
                "finite": finite,
                "completed": survived == horizon_steps,
            }
        )
    result = {
        "schema_version": 1,
        "backend": "native_mujoco_onnx_cpu",
        "policy_status": session.bundle["status"],
        "scenario_sha256": scenario_set.sha256,
        "checkpoint_sha256": session.bundle["checkpoint_sha256"],
        "passed_finite_rollouts": sum(row["finite"] and row["completed"] for row in records),
        "planned_rollouts": len(records),
        "trials": records,
    }
    _write_json(output / "summary.json", result)
    return result


def check_walking_native_parity(
    run: Path,
    checkpoint: Path,
    policy: Path,
    output: Path,
    *,
    samples: int = 100,
) -> dict[str, Any]:
    """Compare physical actor fields and actions on shared live mjlab states."""
    import mujoco
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner

    from .environment import build_train_config
    from .rl_adapter import MjlabVecEnvWrapper
    from .tasks.walking_mdp import WalkingCommand

    if samples < 100:
        raise ValueError("walking parity requires at least 100 shared states")
    run = run.resolve(strict=True)
    policy = policy.resolve(strict=True)
    checkpoint = checkpoint if checkpoint.is_absolute() else run / "checkpoints" / checkpoint
    checkpoint = checkpoint.resolve(strict=True)
    session = WalkingPolicySession.load(policy, allow_unqualified=True)
    if sha256_file(checkpoint) != session.bundle["checkpoint_sha256"]:
        raise ValueError("parity checkpoint does not match walking bundle")
    if output.exists():
        raise FileExistsError(f"walking parity output already exists: {output}")
    output.mkdir(parents=True)
    config = replace(_resolved_run_config(run / "config.json"), num_envs=1, seed=10043)
    profile = load_walking_training_profile(run / "walking-profile.json")
    rewards = load_stage19_reward_profile(run / "walking-reward-profile.json")
    train_cfg = build_train_config(
        config,
        output / ".mjlab",
        randomized_reset=False,
        walking_profile=profile,
        walking_reward_profile=rewards,
    )
    env = ManagerBasedRlEnv(cfg=train_cfg.env, device=config.device, render_mode=None)
    native_model = mujoco.MjModel.from_binary_path(str(policy / "model.mjb"))
    records: list[dict[str, Any]] = []
    try:
        wrapped = MjlabVecEnvWrapper(env, clip_actions=config.action_clip)
        runner = MjlabOnPolicyRunner(wrapped, asdict(train_cfg.agent), device=config.device)
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=config.device
        )
        actor = runner.get_inference_policy(device=config.device)
        observation = wrapped.get_observations()
        command = env.command_manager.get_term("twist")
        if not isinstance(command, WalkingCommand):
            raise TypeError("walking parity requires WalkingCommand")
        simulator_data: Any = env.sim.data
        for step in range(samples):
            requested = 0.6 if 25 <= step < 85 else 0.0
            command.set_requested_forward_speed(requested)
            with torch.inference_mode():
                expected_action_tensor = actor(observation)
                expected_action = expected_action_tensor[0].detach().cpu().numpy()
            data = mujoco.MjData(native_model)
            data.qpos[:] = simulator_data.qpos[0].detach().cpu().numpy()
            data.qvel[:] = simulator_data.qvel[0].detach().cpu().numpy()
            mujoco.mj_forward(native_model, data)
            gait = GaitState(
                command.command.detach().cpu().numpy().copy(),
                command.phase.detach().cpu().numpy().copy(),
                command.blend.detach().cpu().numpy().copy(),
                command.walking.detach().cpu().numpy().copy(),
                command.reference_distance_m.detach().cpu().numpy().copy(),
            )
            previous = env.action_manager.action[0].detach().cpu().numpy()
            reconstructed = compose_walking_actor_observation(
                session.contract, _sensor(native_model, data, session.contract), previous, gait
            )
            expected_observation = observation["actor"][0].detach().cpu().numpy()
            actual_action = session.runtime.run(
                None,
                {
                    session.runtime.get_inputs()[0].name: expected_observation[None].astype(
                        np.float32
                    )
                },
            )[0][0]
            records.append(
                {
                    "step": step,
                    "observation_max_abs_error": float(
                        np.max(np.abs(reconstructed - expected_observation))
                    ),
                    "action_max_abs_error": float(np.max(np.abs(actual_action - expected_action))),
                    "observation_passed": bool(
                        np.allclose(reconstructed, expected_observation, atol=1e-5, rtol=1e-4)
                    ),
                    "action_passed": bool(
                        np.allclose(actual_action, expected_action, atol=1e-4, rtol=1e-4)
                    ),
                }
            )
            observation, _, _, _ = wrapped.step(expected_action_tensor)
    finally:
        env.close()
    result = {
        "schema_version": 1,
        "samples": len(records),
        "observation_tolerance": {"absolute": 1e-5, "relative": 1e-4},
        "action_tolerance": {"absolute": 1e-4, "relative": 1e-4},
        "max_observation_absolute_error": max(row["observation_max_abs_error"] for row in records),
        "max_action_absolute_error": max(row["action_max_abs_error"] for row in records),
        "passed": all(row["observation_passed"] and row["action_passed"] for row in records),
        "checkpoint_sha256": session.bundle["checkpoint_sha256"],
        "records": records,
    }
    _write_json(output / "summary.json", result)
    return result


def play_native_walking(
    policy: Path,
    *,
    forward_speed_m_s: float,
    allow_unqualified: bool = False,
    duration_s: float | None = None,
) -> None:
    """Open a real-time native MuJoCo viewer with deterministic ONNX inference."""
    import mujoco
    import mujoco.viewer

    session = WalkingPolicySession.load(
        policy.resolve(strict=True), allow_unqualified=allow_unqualified
    )
    model = mujoco.MjModel.from_binary_path(str(policy / "model.mjb"))
    data = mujoco.MjData(model)
    data.qpos[2] = float(session.contract["initial_root_position"][2])
    data.qpos[3] = 1.0
    data.qpos[session.contract["qpos_addresses"]] = session.contract["nominal_joint_position"]
    data.ctrl[session.contract["actuator_ids"]] = data.qpos[session.contract["qpos_addresses"]]
    mujoco.mj_forward(model, data)
    session.reset(_sensor(model, data, session.contract))
    dt = float(session.contract["control_dt"])
    decimation = round(dt / model.opt.timestep)
    deadline = None if duration_s is None else time.monotonic() + duration_s
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("robot/pelvis").id
        viewer.cam.distance = 3.0
        while viewer.is_running() and (deadline is None or time.monotonic() < deadline):
            started = time.monotonic()
            action = session.step(forward_speed_m_s, _sensor(model, data, session.contract)).action
            data.ctrl[session.contract["actuator_ids"]] = walking_joint_targets(
                session.contract, action
            )
            for _ in range(decimation):
                mujoco.mj_step(model, data)
            viewer.sync()
            remaining = dt - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
