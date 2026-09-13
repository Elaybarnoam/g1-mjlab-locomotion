"""Record one deterministic walking-v2 actor rollout as numeric evidence."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from g1_mjlab.artifacts import sha256_file, write_atomic_json
from g1_mjlab.config import WalkingV2CurriculumProfile, load_config
from g1_mjlab.environment import build_train_config
from g1_mjlab.rl_adapter import MjlabVecEnvWrapper
from g1_mjlab.tasks.walking_v2_mdp import ReferenceResidualAction, WalkingV2Command
from g1_mjlab.walking_v2_acquisition import summarize_acquisition_trace


def main() -> int:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner

    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--speed", type=float, default=0.4)
    parser.add_argument("--horizon-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=10042)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    steps = round(args.horizon_seconds / config.control_dt)
    if steps <= 0 or abs(steps * config.control_dt - args.horizon_seconds) > 1e-9:
        raise ValueError("horizon must be positive and align to the policy interval")
    eval_config = replace(
        config,
        seed=args.seed,
        num_envs=1,
        max_iterations=1,
        episode_length_s=args.horizon_seconds + config.control_dt,
    )
    evaluation_profile = WalkingV2CurriculumProfile(
        schema_version=2,
        name="deterministic-evaluation-v2",
        stage="transitions",
        standing_fraction=1.0,
        forward_speed_range_m_s=(0.4, 0.8),
        resampling_time_range_s=(1.5, 4.0),
        reference_initialization=False,
        observation_noise=False,
        startup_domain_randomization=False,
        push_disturbance=False,
        terminate_reference_deviation=False,
        gait_contact_weight=2.0,
        gait_alternation_event_weight=1.0,
        fall_penalty=-10.0,
    )
    train_cfg = build_train_config(
        eval_config,
        args.output,
        randomized_reset=False,
        walking_v2_curriculum_profile=evaluation_profile,
    )
    train_cfg.env.auto_reset = False
    train_cfg.env.commands["twist"].standing_fraction = 1.0
    train_cfg.env.commands["twist"].randomize_phase = False
    train_cfg.env.commands["twist"].reference_initialization = False
    env = ManagerBasedRlEnv(cfg=train_cfg.env, device=config.device, render_mode=None)
    records: dict[str, list[np.ndarray | float | bool]] = {
        name: []
        for name in (
            "actor_observation",
            "action",
            "reward",
            "root_position_w",
            "root_quaternion_wxyz",
            "root_linear_velocity_body",
            "joint_position",
            "joint_velocity",
            "joint_target",
            "actuator_torque",
            "command",
            "phase",
            "blend",
            "contact",
            "expected_contact",
            "reference_joint_velocity",
            "terminated",
            "truncated",
        )
    }
    try:
        wrapped = MjlabVecEnvWrapper(env, clip_actions=config.action_clip)
        runner = MjlabOnPolicyRunner(wrapped, asdict(train_cfg.agent), device=config.device)
        runner.load(
            str(args.checkpoint.resolve(strict=True)),
            load_cfg={"actor": True},
            strict=True,
            map_location=config.device,
        )
        policy = runner.get_inference_policy(device=config.device)
        env.reset(seed=args.seed)
        observation = wrapped.get_observations()
        command = env.command_manager.get_term("twist")
        action_term = env.action_manager.get_term("joint_pos")
        if not isinstance(command, WalkingV2Command) or not isinstance(
            action_term, ReferenceResidualAction
        ):
            raise TypeError("rollout requires walking-v2 command and residual action terms")
        command.set_requested_forward_speed(args.speed)
        robot = env.scene["robot"]
        sensor = env.scene["feet_ground_contact"]
        with torch.inference_mode():
            for _ in range(steps):
                action = policy(observation)
                observation, reward, done, extras = wrapped.step(action)
                del extras
                terminated = bool(env.termination_manager.terminated[0])
                truncated = bool(env.termination_manager.time_outs[0])
                found = sensor.data.found
                if found is None:
                    raise RuntimeError("feet contact sensor has no found field")
                values = {
                    "actor_observation": observation["actor"][0],
                    "action": action[0],
                    "reward": reward[0],
                    "root_position_w": robot.data.root_link_pos_w[0],
                    "root_quaternion_wxyz": robot.data.root_link_quat_w[0],
                    "root_linear_velocity_body": robot.data.root_link_lin_vel_b[0],
                    "joint_position": robot.data.joint_pos[0],
                    "joint_velocity": robot.data.joint_vel[0],
                    "joint_target": action_term.joint_target[0],
                    "actuator_torque": robot.data.qfrc_actuator[0],
                    "command": command.command[0],
                    "phase": command.phase[0],
                    "blend": command.blend[0],
                    "contact": (found.reshape(1, -1)[0, :2] > 0),
                    "expected_contact": command.expected_contact[0],
                    "reference_joint_velocity": command.target_joint_velocity[0],
                    "terminated": terminated,
                    "truncated": truncated,
                }
                for name, value in values.items():
                    if isinstance(value, torch.Tensor):
                        records[name].append(value.detach().cpu().numpy())
                    else:
                        records[name].append(value)
                if bool(done[0]):
                    break
    finally:
        env.close()
    arrays = {name: np.asarray(values) for name, values in records.items()}
    if not arrays["action"].size or not all(
        np.isfinite(value).all() for value in arrays.values() if value.dtype.kind != "b"
    ):
        raise FloatingPointError("deterministic rollout is empty or non-finite")
    trace_path = args.output / "deterministic-rollout.npz"
    # NumPy's stubs do not model dynamically named array members, although the
    # runtime API explicitly supports them through ``**kwds``.
    np.savez_compressed(trace_path, **arrays)  # type: ignore[arg-type]
    summary = {
        "schema_version": 2,
        "task_id": config.task_id,
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "trace": trace_path.name,
        "trace_sha256": sha256_file(trace_path),
        "seed": args.seed,
        "requested_speed_m_s": args.speed,
        "planned_steps": steps,
        "recorded_steps": int(len(arrays["reward"])),
        "survived_seconds": float(len(arrays["reward"]) * config.control_dt),
        "terminated": bool(arrays["terminated"].any()),
        "truncated": bool(arrays["truncated"].any()),
        "mean_reward": float(arrays["reward"].mean()),
        "wall_seconds_including_startup": time.monotonic() - started,
        "qualification_claim": False,
        "acquisition_metrics": summarize_acquisition_trace(
            arrays, control_dt=config.control_dt, requested_speed_m_s=args.speed
        ),
    }
    write_atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
