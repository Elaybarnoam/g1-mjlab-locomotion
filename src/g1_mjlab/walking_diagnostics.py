"""Bounded controller-feasibility and parallelism probes for walking-v1."""

from __future__ import annotations

import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .config import ResolvedRunConfig


def select_parallelism(
    results: list[dict[str, Any]], *, repeats: int, minimum_free_ratio: float
) -> dict[str, Any]:
    if repeats <= 0 or not 0 <= minimum_free_ratio < 1:
        raise ValueError("invalid parallelism selection requirements")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[int(result["num_envs"])].append(result)
    candidates: list[tuple[float, int, list[dict[str, Any]]]] = []
    for num_envs, group in grouped.items():
        if (
            len(group) == repeats
            and all(bool(item["stable"]) for item in group)
            and min(float(item["minimum_free_ratio"]) for item in group) >= minimum_free_ratio
        ):
            candidates.append(
                (mean(float(item["steps_per_second"]) for item in group), num_envs, group)
            )
    if not candidates:
        raise ValueError("No environment count passed every repeat and memory-headroom gate")
    throughput, selected, group = max(candidates)
    return {
        "schema_version": 1,
        "selected_num_envs": selected,
        "mean_steps_per_second": throughput,
        "minimum_free_ratio": min(float(item["minimum_free_ratio"]) for item in group),
        "repeats": repeats,
        "required_minimum_free_ratio": minimum_free_ratio,
        "reason": "fastest repeatable stable batch with required memory headroom",
    }


def run_walking_probe(
    config: ResolvedRunConfig,
    output: Path,
    *,
    steps: int,
    reference_actions: bool,
) -> dict[str, Any]:
    """Run a fresh bounded simulator process and save measured controller/runtime telemetry."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    from .environment import build_train_config
    from .tasks.walking_mdp import WalkingCommand

    train_cfg = build_train_config(config, output.parent / "unused", randomized_reset=True)
    command_cfg = train_cfg.env.commands["twist"]
    command_cfg.standing_fraction = 0.0
    command_cfg.randomize_phase = False
    env = ManagerBasedRlEnv(cfg=train_cfg.env, device=config.device, render_mode=None)
    started = time.perf_counter()
    reset_seconds = math.nan
    minimum_free = math.inf
    total_memory = math.nan
    max_action = 0.0
    max_applied_action = 0.0
    max_tracking_error = 0.0
    tracking_square_sum = 0.0
    tracking_samples = 0
    max_effort = 0.0
    max_effort_by_joint: Any = None
    max_tracking_by_joint: Any = None
    minimum_pelvis_height = math.inf
    maximum_torso_tilt = 0.0
    first_termination: dict[str, Any] | None = None
    termination_count = 0
    termination_by_term = {name: 0 for name in train_cfg.env.terminations if name != "time_out"}
    finite = True
    try:
        reset_started = time.perf_counter()
        observations, _ = env.reset(seed=config.seed)
        torch.cuda.synchronize() if config.device.startswith("cuda") else None
        reset_seconds = time.perf_counter() - reset_started
        command = env.command_manager.get_term("twist")
        if not isinstance(command, WalkingCommand):
            raise TypeError("walking diagnostic did not resolve WalkingCommand")
        action_term = env.action_manager.get_term("joint_pos")
        robot = env.scene["robot"]
        loop_started = time.perf_counter()
        for index in range(steps):
            target = (
                robot.data.default_joint_pos * (1 - command.blend[:, None])
                + command.joint_position * command.blend[:, None]
            )
            raw_action = (target - action_term.offset) / action_term.scale
            action = raw_action if reference_actions else torch.zeros_like(raw_action)
            if config.action_clip is not None:
                action = torch.clamp(action, -config.action_clip, config.action_clip)
            observations, _, terminated, _, _ = env.step(action)
            error = robot.data.joint_pos - target
            max_action = max(max_action, float(torch.max(torch.abs(raw_action))))
            max_applied_action = max(max_applied_action, float(torch.max(torch.abs(action))))
            max_tracking_error = max(max_tracking_error, float(torch.max(torch.abs(error))))
            per_joint_tracking = torch.max(torch.abs(error), dim=0).values
            max_tracking_by_joint = (
                per_joint_tracking
                if max_tracking_by_joint is None
                else torch.maximum(max_tracking_by_joint, per_joint_tracking)
            )
            tracking_square_sum += float(torch.sum(torch.square(error)))
            tracking_samples += error.numel()
            max_effort = max(max_effort, float(torch.max(torch.abs(robot.data.qfrc_actuator))))
            per_joint_effort = torch.max(torch.abs(robot.data.qfrc_actuator), dim=0).values
            max_effort_by_joint = (
                per_joint_effort
                if max_effort_by_joint is None
                else torch.maximum(max_effort_by_joint, per_joint_effort)
            )
            termination_count += int(torch.sum(terminated))
            for name in termination_by_term:
                count = int(torch.sum(env.termination_manager.get_term(name)))
                termination_by_term[name] += count
                if count and first_termination is None:
                    first_termination = {
                        "step": index + 1,
                        "time_s": (index + 1) * env.step_dt,
                        "term": name,
                    }
            minimum_pelvis_height = min(
                minimum_pelvis_height, float(torch.min(robot.data.root_link_pos_w[:, 2]))
            )
            root_quaternion = robot.data.root_link_quat_w
            upright_z = 1 - 2 * (
                torch.square(root_quaternion[:, 1]) + torch.square(root_quaternion[:, 2])
            )
            tilt = torch.arccos(torch.clamp(upright_z, -1.0, 1.0))
            maximum_torso_tilt = max(maximum_torso_tilt, float(torch.max(tilt)))
            finite = finite and bool(
                torch.isfinite(observations["actor"]).all()
                and torch.isfinite(observations["critic"]).all()
                and torch.isfinite(robot.data.joint_pos).all()
                and torch.isfinite(robot.data.qfrc_actuator).all()
            )
            if config.device.startswith("cuda") and index % 10 == 0:
                free, total = torch.cuda.mem_get_info()
                minimum_free = min(minimum_free, float(free))
                total_memory = float(total)
        torch.cuda.synchronize() if config.device.startswith("cuda") else None
        loop_seconds = time.perf_counter() - loop_started
    finally:
        env.close()
    stable = finite and termination_count == 0
    result = {
        "schema_version": 1,
        "task_id": config.task_id,
        "num_envs": config.num_envs,
        "seed": config.seed,
        "steps": steps,
        "reference_actions": reference_actions,
        "stable": stable,
        "all_values_finite": finite,
        "termination_count": termination_count,
        "termination_by_term": termination_by_term,
        "first_termination": first_termination,
        "reset_seconds": reset_seconds,
        "loop_seconds": loop_seconds,
        "steps_per_second": config.num_envs * steps / loop_seconds,
        "max_absolute_raw_action": max_action,
        "max_absolute_applied_action": max_applied_action,
        "action_clip": config.action_clip,
        "joint_tracking_rms_rad": math.sqrt(tracking_square_sum / tracking_samples),
        "max_absolute_joint_tracking_error_rad": max_tracking_error,
        "max_absolute_actuator_force_nm": max_effort,
        "max_actuator_force_nm_by_joint": dict(
            zip(robot.joint_names, max_effort_by_joint.cpu().tolist(), strict=True)
        ),
        "max_tracking_error_rad_by_joint": dict(
            zip(robot.joint_names, max_tracking_by_joint.cpu().tolist(), strict=True)
        ),
        "minimum_pelvis_height_m": minimum_pelvis_height,
        "maximum_torso_tilt_rad": maximum_torso_tilt,
        "minimum_free_bytes": minimum_free,
        "total_device_bytes": total_memory,
        "minimum_free_ratio": minimum_free / total_memory,
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if config.device.startswith("cuda") else None,
        },
        "elapsed_seconds_including_compile": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
