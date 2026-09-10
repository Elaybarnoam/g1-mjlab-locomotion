"""Bounded standing and checkpoint diagnostics for qualified mjlab runtimes."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median
from typing import Any

from .artifacts import sha256_file
from .config import ResolvedRunConfig
from .runtime import _standing_train_config


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n" for record in records
        ),
        encoding="utf-8",
    )


def _diagnostic_case(
    config: ResolvedRunConfig,
    checkpoint: Path,
    output_dir: Path,
    *,
    case_name: str,
    use_policy: bool,
    randomized_reset: bool,
    trials: int,
    horizon_s: float,
) -> dict[str, Any]:
    """Run one bounded control probe and retain terminal-state actuator traces."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls

    case_config = replace(config, num_envs=trials, episode_length_s=horizon_s)
    train_cfg = _standing_train_config(
        case_config, output_dir / case_name, randomized_reset=randomized_reset
    )
    train_cfg.env.auto_reset = False
    env = ManagerBasedRlEnv(cfg=train_cfg.env, device=config.device, render_mode=None)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=train_cfg.agent.clip_actions)
        policy = None
        if use_policy:
            runner_cls = load_runner_cls(config.task_id) or MjlabOnPolicyRunner
            runner = runner_cls(wrapped, asdict(train_cfg.agent), device=config.device)
            runner.load(
                str(checkpoint),
                load_cfg={"actor": True},
                strict=True,
                map_location=config.device,
            )
            policy = runner.get_inference_policy(device=config.device)

        obs = wrapped.get_observations()
        robot = env.scene["robot"]
        initial_xy = robot.data.root_link_pos_w[:, :2].clone()
        active = torch.ones(trials, dtype=torch.bool, device=config.device)
        survived_steps = torch.zeros(trials, dtype=torch.int64, device=config.device)
        max_drift = torch.zeros(trials, device=config.device)
        failed = torch.zeros(trials, dtype=torch.bool, device=config.device)
        horizon_steps = int(round(horizon_s / config.control_dt))
        trace: list[dict[str, Any]] = []
        initial_actor_observation: list[float] | None = None
        initial_action: list[float] | None = None
        with torch.inference_mode():
            for step in range(horizon_steps):
                actions = (
                    policy(obs)
                    if policy is not None
                    else torch.zeros((trials, wrapped.num_actions), device=config.device)
                )
                actions = torch.where(active[:, None], actions, torch.zeros_like(actions))
                if step == 0:
                    initial_actor_observation = obs["actor"][0].detach().cpu().tolist()
                    initial_action = actions[0].detach().cpu().tolist()
                obs, rewards, dones, extras = wrapped.step(actions)
                time_outs = extras.get("time_outs")
                if time_outs is None:
                    time_outs = torch.zeros_like(dones, dtype=torch.bool)
                failures = active & env.termination_manager.terminated
                failed |= failures
                drift = torch.linalg.vector_norm(
                    robot.data.root_link_pos_w[:, :2] - initial_xy, dim=1
                )
                max_drift = torch.where(active, torch.maximum(max_drift, drift), max_drift)
                survived_steps += active.to(torch.int64)

                snapshots = {
                    "height_m": robot.data.root_link_pos_w[:, 2].detach().cpu().tolist(),
                    "projected_gravity": robot.data.projected_gravity_b.detach().cpu().tolist(),
                    "base_lin_vel": robot.data.root_link_lin_vel_b.detach().cpu().tolist(),
                    "base_ang_vel": robot.data.root_link_ang_vel_b.detach().cpu().tolist(),
                    "joint_pos": robot.data.joint_pos.detach().cpu().tolist(),
                    "joint_vel": robot.data.joint_vel.detach().cpu().tolist(),
                    "joint_target": robot.data.joint_pos_target.detach().cpu().tolist(),
                    "actuator_force": robot.data.qfrc_actuator.detach().cpu().tolist(),
                    "action": actions.detach().cpu().tolist(),
                    "applied_action": (
                        actions
                        if config.action_clip is None
                        else actions.clamp(-config.action_clip, config.action_clip)
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    "reward": rewards.detach().cpu().tolist(),
                    "done": dones.detach().cpu().tolist(),
                    "time_out": time_outs.detach().cpu().tolist(),
                }
                for trial_id in range(trials):
                    if not bool(active[trial_id].item()):
                        continue
                    trace.append(
                        {
                            "schema_version": 1,
                            "case": case_name,
                            "step": step + 1,
                            "time_s": (step + 1) * config.control_dt,
                            "trial_id": trial_id,
                            **{key: value[trial_id] for key, value in snapshots.items()},
                        }
                    )
                active &= ~dones.bool()
                done_ids = torch.nonzero(dones.bool(), as_tuple=False).flatten()
                if done_ids.numel():
                    env.reset(env_ids=done_ids)
                    obs = wrapped.get_observations()
                if not bool(active.any().item()):
                    break
    finally:
        env.close()

    survival = [float(value) * config.control_dt for value in survived_steps.cpu().tolist()]
    drift_values = [float(value) for value in max_drift.cpu().tolist()]
    trace_path = output_dir / f"{case_name}.jsonl"
    _write_jsonl(trace_path, trace)
    return {
        "case": case_name,
        "control": "policy_mean" if use_policy else "zero_action_nominal_target",
        "randomized_reset": randomized_reset,
        "trials": trials,
        "horizon_seconds": horizon_s,
        "passed": sum(
            value >= horizon_s and not bool(failed[index].item())
            for index, value in enumerate(survival)
        ),
        "survival_seconds": survival,
        "median_survival_seconds": median(survival),
        "max_drift_m": drift_values,
        "trace": trace_path.name,
        "trace_records": len(trace),
        "initial_actor_observation": initial_actor_observation,
        "initial_action": initial_action,
    }


def diagnose_standing(
    config: ResolvedRunConfig,
    checkpoint: Path,
    output_dir: Path,
    *,
    onnx_path: Path | None = None,
    trials: int = 4,
    horizon_s: float = 3.0,
) -> dict[str, Any]:
    """Compare learned and passive controls under matched bounded conditions."""
    if not 1 <= trials <= 4 or not math.isfinite(horizon_s) or not 0 < horizon_s <= 10:
        raise ValueError("diagnostics require 1..4 trials and a finite horizon in (0, 10]")
    output_dir.mkdir(parents=True, exist_ok=False)
    cases = [
        _diagnostic_case(
            config,
            checkpoint,
            output_dir,
            case_name="policy-randomized",
            use_policy=True,
            randomized_reset=True,
            trials=trials,
            horizon_s=horizon_s,
        ),
        _diagnostic_case(
            config,
            checkpoint,
            output_dir,
            case_name="policy-nominal",
            use_policy=True,
            randomized_reset=False,
            trials=trials,
            horizon_s=horizon_s,
        ),
        _diagnostic_case(
            config,
            checkpoint,
            output_dir,
            case_name="zero-nominal",
            use_policy=False,
            randomized_reset=False,
            trials=trials,
            horizon_s=horizon_s,
        ),
    ]
    parity: dict[str, Any] | None = None
    if onnx_path is not None:
        import numpy as np
        import onnxruntime as ort

        actor_observation = cases[0]["initial_actor_observation"]
        expected_action = cases[0]["initial_action"]
        if actor_observation is None or expected_action is None:
            raise RuntimeError("diagnostic did not capture initial inference tensors")
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        actual_action = session.run(
            None, {session.get_inputs()[0].name: np.asarray([actor_observation], dtype=np.float32)}
        )[0][0]
        error = np.abs(actual_action - np.asarray(expected_action, dtype=np.float32))
        parity = {
            "onnx": onnx_path.name,
            "onnx_sha256": sha256_file(onnx_path),
            "input_name": session.get_inputs()[0].name,
            "output_name": session.get_outputs()[0].name,
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
            "passed_at_1e-5": bool(error.max() <= 1.0e-5),
        }

    result = {
        "schema_version": 1,
        "config_sha256": config.sha256,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "control_dt": config.control_dt,
        "onnx_parity": parity,
        "cases": cases,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def diagnose_checkpoints(
    config: ResolvedRunConfig,
    checkpoints: list[Path],
    output_dir: Path,
    *,
    horizon_s: float = 3.0,
) -> dict[str, Any]:
    """Compare deterministic checkpoints from the same nominal initial state."""
    if not 1 <= len(checkpoints) <= 16 or not math.isfinite(horizon_s) or not 0 < horizon_s <= 10:
        raise ValueError("diagnostics require 1..16 checkpoints and a finite horizon in (0, 10]")
    output_dir.mkdir(parents=True, exist_ok=False)
    cases = [
        _diagnostic_case(
            config,
            checkpoint,
            output_dir,
            case_name=f"{checkpoint.stem}-nominal",
            use_policy=True,
            randomized_reset=False,
            trials=1,
            horizon_s=horizon_s,
        )
        for checkpoint in checkpoints
    ]
    result = {
        "schema_version": 1,
        "config_sha256": config.sha256,
        "control_dt": config.control_dt,
        "checkpoints": [
            {"name": checkpoint.name, "sha256": sha256_file(checkpoint)}
            for checkpoint in checkpoints
        ],
        "cases": cases,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
