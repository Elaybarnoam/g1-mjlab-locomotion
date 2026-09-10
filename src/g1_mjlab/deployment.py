"""Independent native MuJoCo/ONNX standing loop consuming a frozen run bundle."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .artifacts import sha256_file, snapshot_source
from .config import ResolvedRunConfig
from .evaluation import StandingCriteria, TrialAccumulator, wilson_interval
from .qualification import canonical_hash


def load_bundle(run: Path) -> tuple[dict[str, Any], Any, Any]:
    import mujoco
    import onnxruntime as ort

    contract = json.loads((run / "contract.json").read_text(encoding="utf-8"))
    bundle = json.loads((run / "policy-bundle.json").read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("controller contract must be an object")
    if not isinstance(bundle, dict):
        raise ValueError("policy bundle must be an object")
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported controller contract schema version")
    if bundle.get("schema_version") != 1:
        raise ValueError("unsupported policy bundle schema version")
    if (
        canonical_hash(contract) != contract["sha256"]
        or bundle["contract_sha256"] != contract["sha256"]
    ):
        raise ValueError("controller contract integrity mismatch")
    if sha256_file(run / "model.mjb") != bundle["model_sha256"]:
        raise ValueError("compiled model integrity mismatch")
    onnx = run / "checkpoints" / "policy.onnx"
    if sha256_file(onnx) != bundle["onnx_sha256"]:
        raise ValueError("ONNX policy/normalizer integrity mismatch")
    model = mujoco.MjModel.from_binary_path(str(run / "model.mjb"))
    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    return contract, model, session


def freeze_selected_policy(config: ResolvedRunConfig, run: Path) -> dict[str, Any]:
    """Export the development-selected checkpoint and bind all deployment hashes."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from .environment import build_standing_train_config

    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    checkpoint = run / "checkpoints" / selection["selected_checkpoint"]
    checkpoint.resolve(strict=True)
    train_cfg = build_standing_train_config(replace(config, num_envs=1), run / "freeze")
    env = ManagerBasedRlEnv(cfg=train_cfg.env, device=config.device, render_mode=None)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=config.action_clip)
        runner = MjlabOnPolicyRunner(wrapped, asdict(train_cfg.agent), device=config.device)
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=config.device
        )
        output = run / "checkpoints" / "policy.onnx"
        if output.exists():
            final_copy = run / "checkpoints" / "policy-final-budget.onnx"
            if not final_copy.exists():
                shutil.copy2(output, final_copy)
        runner.export_policy_to_onnx(str(output.parent), filename=output.name)
    finally:
        env.close()
    bundle = json.loads((run / "policy-bundle.json").read_text(encoding="utf-8"))
    if not isinstance(bundle, dict):
        raise ValueError("policy bundle must be an object")
    bundle.update(
        {
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256_file(checkpoint),
            "onnx_sha256": sha256_file(run / "checkpoints" / "policy.onnx"),
            "selection": selection["rule"],
            "selection_phase": "development",
            "final_test_qualified": False,
        }
    )
    (run / "policy-bundle.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return bundle


def qualify_final_bundle(run: Path) -> dict[str, Any]:
    """Bind the immutable final evidence only when both backends pass the frozen gate."""
    mjlab_path = run / "evaluation" / "final-100x60" / "summary.json"
    native_path = run / "native-final-100x60" / "summary.json"
    mjlab_result: dict[str, Any] = json.loads(mjlab_path.read_text(encoding="utf-8"))
    native_result: dict[str, Any] = json.loads(native_path.read_text(encoding="utf-8"))
    bundle: dict[str, Any] = json.loads((run / "policy-bundle.json").read_text(encoding="utf-8"))
    if not all(isinstance(value, dict) for value in (mjlab_result, native_result, bundle)):
        raise ValueError("final evidence and bundle must be JSON objects")
    required = (100, 60.0)
    for backend, result in (("mjlab", mjlab_result), ("native_mujoco", native_result)):
        if (
            result.get("phase") != "final"
            or (result.get("planned"), result.get("horizon_seconds")) != required
            or result.get("passed", 0) < 95
        ):
            raise ValueError(f"{backend} has not passed the frozen final gate")
    if mjlab_result["checkpoint_sha256"] != bundle["checkpoint_sha256"]:
        raise ValueError("final mjlab result does not match the selected checkpoint")
    if native_result["policy_bundle"]["onnx_sha256"] != bundle["onnx_sha256"]:
        raise ValueError("final native result does not match the selected ONNX policy")
    candidate = run / "source-qualified-candidate.zip"
    qualified_source = snapshot_source(Path(__file__).resolve().parents[2], candidate)
    archive = run / f"source-qualified-{qualified_source['sha256'][:12]}.zip"
    if archive.exists():
        if sha256_file(archive) != qualified_source["sha256"]:
            raise ValueError("qualified source archive hash-prefix collision")
        candidate.unlink()
    else:
        candidate.rename(archive)
    qualified_source["archive"] = archive.name
    bundle.update(
        {
            "final_test_qualified": True,
            "final_gate": "at least 95/100 strict 60-second trials in both backends",
            "final_mjlab_passed": mjlab_result["passed"],
            "final_native_passed": native_result["passed"],
            "final_mjlab_evidence_sha256": sha256_file(mjlab_path),
            "final_native_evidence_sha256": sha256_file(native_path),
            "source_archive": qualified_source["archive"],
            "source_sha256": qualified_source["sha256"],
        }
    )
    (run / "policy-bundle.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return bundle


def native_observation(
    model: Any, data: Any, contract: dict[str, Any], previous_action: Any
) -> Any:
    """Reconstruct the v1 actor vector in physical units, before ONNX normalization."""
    import numpy as np

    expected = [
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
    ]
    if [field["name"] for field in contract["actor_fields"]] != expected:
        raise ValueError("unsupported actor layout; explicit deployment adapter required")
    for field in contract["actor_fields"]:
        term = field["resolved_term"]
        if (
            term["scale"] is not None
            or term["clip"] is not None
            or term["history_length"]
            or term["delay_max_lag"]
        ):
            raise ValueError("unsupported transformed/history observation")
    root = model.body("robot/pelvis").id
    return np.concatenate(
        (
            data.sensor("robot/imu_lin_vel").data,
            data.sensor("robot/imu_ang_vel").data,
            data.xmat[root].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0]),
            data.qpos[contract["qpos_addresses"]]
            - np.asarray(contract["nominal_joint_position"])
            + np.asarray(contract["encoder_bias"]),
            data.qvel[contract["dof_addresses"]],
            previous_action,
            np.zeros(3),
        )
    ).astype(np.float32)


def apply_action(data: Any, contract: dict[str, Any], raw: Any) -> Any:
    import numpy as np

    if contract["target_clip"] is not None:
        raise ValueError("target-clipping deployment is not implemented")
    clip = contract["action_clip"]
    applied = raw.copy() if clip is None else np.clip(raw, -clip, clip)
    data.ctrl[contract["actuator_ids"]] = (
        np.asarray(contract["nominal_joint_position"])
        + np.asarray(contract["action_scale"]) * applied
        - np.asarray(contract["encoder_bias"])
    )
    return applied


def evaluate_native(run: Path, scenarios: Path, output: Path) -> dict[str, Any]:
    """Replay development initial states, with no training managers or extra PD."""
    import mujoco
    import numpy as np

    reference = json.loads(scenarios.read_text(encoding="utf-8"))
    if (
        reference.get("phase") not in {"development", "final"}
        or reference.get("schema_version") != 2
    ):
        raise ValueError("requires version-2 development or final scenario evidence")
    contract, model, session = load_bundle(run)
    bundle = json.loads((run / "policy-bundle.json").read_text(encoding="utf-8"))
    if reference["checkpoint_sha256"] != bundle["checkpoint_sha256"]:
        raise ValueError("scenarios must evaluate the exact frozen bundle checkpoint")
    criteria = StandingCriteria(**reference["criteria"])
    horizon = reference["horizon_seconds"]
    control_dt = contract["control_dt"]
    decimation = round(control_dt / model.opt.timestep)
    if not math.isclose(decimation * model.opt.timestep, control_dt, abs_tol=1e-9):
        raise ValueError("nonintegral control decimation")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for scenario in reference["trials"]:
        data = mujoco.MjData(model)
        initial = scenario["initial_state"]
        data.qpos[:3] = initial["root_position_w"]
        data.qpos[3:7] = initial["root_quaternion_wxyz"]
        data.qpos[contract["qpos_addresses"]] = initial["joint_position"]
        data.qvel[contract["dof_addresses"]] = initial["joint_velocity"]
        mujoco.mj_forward(model, data)
        previous = np.zeros(29, dtype=np.float32)
        torso = model.body("robot/torso_link").id
        root = model.body("robot/pelvis").id
        initial_xy = data.xpos[root, :2].copy()
        trial = TrialAccumulator(round(horizon / control_dt), control_dt, criteria)
        trace = []
        for step in range(trial.horizon_steps):
            obs = native_observation(model, data, contract, previous)
            raw = session.run(None, {session.get_inputs()[0].name: obs[None]})[0][0]
            if not np.isfinite(raw).all():
                raise RuntimeError("nonfinite ONNX output")
            previous = apply_action(data, contract, raw)
            for _ in range(decimation):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            tilt = float(np.degrees(np.arccos(np.clip(data.xmat[torso, 8], -1, 1))))
            pelvis_tilt = float(np.degrees(np.arccos(np.clip(data.xmat[root, 8], -1, 1))))
            support = False
            for contact in data.contact:
                names = [model.geom(int(gid)).name for gid in contact.geom]
                support |= any("foot" in n for n in names) and any("terrain" in n for n in names)
            drift = float(np.linalg.norm(data.xpos[root, :2] - initial_xy))
            trial.observe(
                terminated=pelvis_tilt > 70,
                truncated=step + 1 == trial.horizon_steps,
                drift_m=drift,
                torso_tilt_deg=tilt,
                height_m=float(data.xpos[root, 2]),
                supported=support,
                finite=bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
            )
            if scenario["trial_id"] == 0:
                trace.append(
                    {
                        "time_s": (step + 1) * control_dt,
                        "height_m": float(data.xpos[root, 2]),
                        "torso_tilt_deg": tilt,
                        "drift_m": drift,
                        "raw_action": raw.tolist(),
                        "applied_action": previous.tolist(),
                    }
                )
            if trial.finished:
                break
        if trace:
            (output / "trial-0.jsonl").write_text(
                "".join(json.dumps(r, allow_nan=False) + "\n" for r in trace), encoding="utf-8"
            )
        records.append(
            {
                "trial_id": scenario["trial_id"],
                "passed": trial.passed,
                "survival_passed": trial.survival_passed,
                "survived_seconds": trial.steps * control_dt,
                "max_drift_m": trial.max_drift_m,
                "failure_reason": ",".join(sorted(trial.violations)) or None,
            }
        )
    passed = sum(record["passed"] for record in records)
    result = {
        "schema_version": 1,
        "backend": "native_mujoco_onnx_cpu",
        "phase": reference["phase"],
        "passed": passed,
        "planned": len(records),
        "horizon_seconds": horizon,
        "criteria": reference["criteria"],
        "scenario_sha256": sha256_file(scenarios),
        "policy_bundle": bundle,
        "trials": records,
        "wilson_95": wilson_interval(passed, len(records)),
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def check_transfer_parity(config: ResolvedRunConfig, run: Path, output: Path) -> dict[str, Any]:
    """Replay 31 live physical states and actor vectors through independent paths."""
    import mujoco
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from .environment import build_standing_train_config

    contract, native_model, onnx = load_bundle(run)
    bundle = json.loads((run / "policy-bundle.json").read_text(encoding="utf-8"))
    checkpoint = run / "checkpoints" / bundle["checkpoint"]
    if sha256_file(checkpoint) != bundle["checkpoint_sha256"]:
        raise ValueError("PyTorch checkpoint integrity mismatch")
    output.mkdir(parents=True, exist_ok=False)
    cfg = build_standing_train_config(replace(config, num_envs=1, seed=10043), output)
    cfg.env.auto_reset = True
    env = ManagerBasedRlEnv(cfg=cfg.env, device=config.device, render_mode=None)
    samples = []
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=config.action_clip)
        runner = MjlabOnPolicyRunner(wrapped, asdict(cfg.agent), device=config.device)
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=config.device
        )
        policy = runner.get_inference_policy(device=config.device)
        obs = wrapped.get_observations()
        with torch.inference_mode():
            for step in range(301):
                raw = policy(obs)
                if step % 10 == 0:
                    data = mujoco.MjData(native_model)
                    data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
                    data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
                    previous = env.action_manager.action[0].cpu().numpy()
                    apply_action(data, contract, previous)
                    mujoco.mj_forward(native_model, data)
                    reconstructed = native_observation(native_model, data, contract, previous)
                    expected_obs = obs["actor"][0].cpu().numpy()
                    actual_action = onnx.run(None, {onnx.get_inputs()[0].name: expected_obs[None]})[
                        0
                    ][0]
                    expected_action = raw[0].cpu().numpy()
                    field_errors = {
                        field["name"]: float(
                            np.max(
                                np.abs(
                                    reconstructed[field["offset"] : field["offset"] + field["size"]]
                                    - expected_obs[
                                        field["offset"] : field["offset"] + field["size"]
                                    ]
                                )
                            )
                        )
                        for field in contract["actor_fields"]
                    }
                    samples.append(
                        {
                            "step": step,
                            "actor_observation": expected_obs.tolist(),
                            "raw_action": expected_action.tolist(),
                            "observation_field_max_abs_error": field_errors,
                            "inference_max_abs_error": float(
                                np.max(np.abs(actual_action - expected_action))
                            ),
                            "observation_passed": bool(
                                np.allclose(reconstructed, expected_obs, atol=1e-4, rtol=1e-4)
                            ),
                            "inference_passed": bool(
                                np.allclose(actual_action, expected_action, atol=1e-4, rtol=1e-4)
                            ),
                        }
                    )
                if step < 300:
                    obs, _, _, _ = wrapped.step(raw)
    finally:
        env.close()
    result = {
        "schema_version": 1,
        "samples": samples,
        "tolerance": {"absolute": 1e-4, "relative": 1e-4},
        "passed": all(
            sample["observation_passed"] and sample["inference_passed"] for sample in samples
        ),
        "contract_sha256": contract["sha256"],
        "onnx_sha256": bundle["onnx_sha256"],
        "limitation": "state/vector parity only; not closed-loop standing qualification",
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def record_native_video(
    run: Path,
    scenarios: Path,
    output: Path,
    *,
    trial_id: int = 0,
    duration_s: float = 15.0,
    fps: int = 30,
    width: int = 960,
    height: int = 720,
) -> dict[str, Any]:
    """Compatibility entry point; implementation lives in :mod:`native_media`."""
    from .native_media import record_native_video as record

    return record(
        run,
        scenarios,
        output,
        trial_id=trial_id,
        duration_s=duration_s,
        fps=fps,
        width=width,
        height=height,
    )


def play_native(run: Path, scenarios: Path, *, trial_id: int = 0) -> None:
    """Compatibility entry point; implementation lives in :mod:`native_media`."""
    from .native_media import play_native as play

    play(run, scenarios, trial_id=trial_id)
