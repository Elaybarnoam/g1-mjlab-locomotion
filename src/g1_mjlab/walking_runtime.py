"""Lazy simulator adapter for deterministic walking development evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import sha256_file
from .config import ResolvedRunConfig
from .motion.gait import load_command_profile, load_command_schedule
from .tasks import TaskCapability, get_task
from .walking_evaluation import (
    WalkingCriteria,
    WalkingTrace,
    evaluate_walking_trace,
    load_walking_criteria,
)


def evaluate_walking(
    config: ResolvedRunConfig,
    checkpoint: Path,
    schedule_path: Path,
    output: Path,
    *,
    trials: int,
    seed: int,
    criteria: WalkingCriteria | None = None,
) -> dict[str, Any]:
    """Run deterministic actor means and retain each first episode before reset."""
    get_task(config.task_id).require(TaskCapability.WALKING_EVALUATION)
    if not 1 <= trials <= 256 or seed < 0 or seed == config.seed:
        raise ValueError("walking evaluation requires 1..256 trials and a held-out seed")
    checkpoint = checkpoint.resolve(strict=True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evaluation output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config_root = Path(__file__).resolve().parents[2] / "configs" / "walking-v1"
    command_profile = load_command_profile(config_root / "commands.json")
    schedule = load_command_schedule(schedule_path, command_profile)
    horizon_s = sum(segment.duration_s for segment in schedule.segments)
    horizon_steps = round(horizon_s / config.control_dt)
    if abs(horizon_steps * config.control_dt - horizon_s) > 1e-9:
        raise ValueError("schedule duration must align to the control period")

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner

    from .environment import build_train_config
    from .rl_adapter import MjlabVecEnvWrapper
    from .tasks.walking_mdp import WalkingCommand

    eval_config = replace(
        config, seed=seed, num_envs=trials, episode_length_s=horizon_s + config.control_dt
    )
    train_cfg = build_train_config(eval_config, output, randomized_reset=False)
    train_cfg.env.auto_reset = False
    env = ManagerBasedRlEnv(cfg=train_cfg.env, device=config.device, render_mode=None)
    records: list[dict[str, Any]] = [
        {
            "command": [],
            "speed": [],
            "height": [],
            "tilt": [],
            "left_position": [],
            "right_position": [],
            "left_contact": [],
            "right_contact": [],
            "finite": [],
            "terminated": False,
        }
        for _ in range(trials)
    ]
    try:
        wrapped = MjlabVecEnvWrapper(env, clip_actions=config.action_clip)
        runner = MjlabOnPolicyRunner(wrapped, asdict(train_cfg.agent), device=config.device)
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=config.device
        )
        policy = runner.get_inference_policy(device=config.device)
        env.reset(seed=seed)
        observations = wrapped.get_observations()
        command = env.command_manager.get_term("twist")
        if not isinstance(command, WalkingCommand):
            raise TypeError("walking evaluator requires WalkingCommand")
        robot = env.scene["robot"]
        left_id = robot.body_names.index("left_ankle_roll_link")
        right_id = robot.body_names.index("right_ankle_roll_link")
        boundaries: list[tuple[int, float]] = []
        end = 0
        for segment in schedule.segments:
            end += round(segment.duration_s / config.control_dt)
            boundaries.append((end, segment.forward_speed_m_s))
        with torch.inference_mode():
            for step in range(horizon_steps):
                requested = next(speed for end, speed in boundaries if step < end)
                command.set_requested_forward_speed(requested)
                actions = policy(observations)
                observations, _, dones, _ = wrapped.step(actions)
                root_quaternion = robot.data.root_link_quat_w
                upright_z = 1 - 2 * (
                    torch.square(root_quaternion[:, 1]) + torch.square(root_quaternion[:, 2])
                )
                tilt = torch.arccos(torch.clamp(upright_z, -1.0, 1.0))
                found = env.scene["feet_ground_contact"].data.found
                contacts = (found > 0).reshape(trials, 2, -1).any(dim=2)
                finite = (
                    torch.isfinite(actions).all(dim=1)
                    & torch.isfinite(robot.data.joint_pos).all(dim=1)
                    & torch.isfinite(robot.data.root_link_pos_w).all(dim=1)
                )
                frame_values = (
                    command.command[:, 0].cpu().tolist(),
                    robot.data.root_link_lin_vel_b[:, 0].cpu().tolist(),
                    robot.data.root_link_pos_w[:, 2].cpu().tolist(),
                    tilt.cpu().tolist(),
                    robot.data.body_link_pos_w[:, left_id].cpu().tolist(),
                    robot.data.body_link_pos_w[:, right_id].cpu().tolist(),
                    contacts[:, 0].cpu().tolist(),
                    contacts[:, 1].cpu().tolist(),
                    finite.cpu().tolist(),
                    env.termination_manager.terminated.cpu().tolist(),
                )
                for index, record in enumerate(records):
                    if record["terminated"]:
                        continue
                    keys = (
                        "command",
                        "speed",
                        "height",
                        "tilt",
                        "left_position",
                        "right_position",
                        "left_contact",
                        "right_contact",
                        "finite",
                    )
                    for key, values in zip(keys, frame_values[:-1], strict=True):
                        record[key].append(values[index])
                    record["terminated"] = bool(frame_values[-1][index])
                if all(bool(record["terminated"]) for record in records):
                    break
    finally:
        env.close()

    criteria = criteria or load_walking_criteria(config_root / "evaluation.json")
    trials_out = []
    for trial_id, record in enumerate(records):
        trace = WalkingTrace(
            dt=config.control_dt,
            command_speed=np.asarray(record["command"]),
            forward_speed=np.asarray(record["speed"]),
            pelvis_height=np.asarray(record["height"]),
            torso_tilt_rad=np.asarray(record["tilt"]),
            left_foot_position=np.asarray(record["left_position"]),
            right_foot_position=np.asarray(record["right_position"]),
            left_contact=np.asarray(record["left_contact"]),
            right_contact=np.asarray(record["right_contact"]),
            finite=np.asarray(record["finite"]),
            terminated=bool(record["terminated"]),
        )
        np.savez_compressed(
            output / f"trace-{trial_id:03d}.npz",
            command_speed=trace.command_speed,
            forward_speed=trace.forward_speed,
            pelvis_height=trace.pelvis_height,
            torso_tilt_rad=trace.torso_tilt_rad,
            left_foot_position=trace.left_foot_position,
            right_foot_position=trace.right_foot_position,
            left_contact=trace.left_contact,
            right_contact=trace.right_contact,
            finite=trace.finite,
        )
        trials_out.append(
            {
                "trial_id": trial_id,
                "trace": f"trace-{trial_id:03d}.npz",
                "frames": trace.frame_count,
                "survived_seconds": trace.frame_count * trace.dt,
                **asdict(evaluate_walking_trace(trace, criteria)),
            }
        )
    passed = sum(trial["functional_passed"] and trial["style_passed"] for trial in trials_out)
    result = {
        "schema_version": 1,
        "phase": "development",
        "task_id": config.task_id,
        "seed": seed,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "schedule": schedule.name,
        "schedule_sha256": sha256_file(schedule_path),
        "criteria": asdict(criteria),
        "criteria_status": "frozen before pilot; development only",
        "planned": trials,
        "completed": len(trials_out),
        "passed_both": passed,
        "trials": trials_out,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result
