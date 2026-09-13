"""Run the bounded P06-04 GPU task and manager-order smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from g1_mjlab.artifacts import sha256_file, write_atomic_json
from g1_mjlab.tasks.walking_v2 import TASK_ID, register_task, walking_v2_policy_contract
from g1_mjlab.tasks.walking_v2_mdp import ReferenceResidualAction, WalkingV2Command


def main() -> int:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    register_task()
    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    cfg.events = {}
    cfg.commands["twist"].standing_fraction = 0.0
    cfg.commands["twist"].randomize_phase = True
    cfg.commands["twist"].reference_initialization = True
    device = torch.device(args.device)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode=None)
    try:
        observation, _ = env.reset(seed=42)
        command = env.command_manager.get_term("twist")
        action = env.action_manager.get_term("joint_pos")
        if not isinstance(command, WalkingV2Command) or not isinstance(
            action, ReferenceResidualAction
        ):
            raise TypeError("walking-v2 manager term types differ from the frozen task")
        robot = env.scene["robot"]
        rsi_position_error = float(
            torch.max(torch.abs(robot.data.joint_pos - command.target_joint_position)).item()
        )
        rsi_velocity_error = float(
            torch.max(torch.abs(robot.data.joint_vel - command.target_joint_velocity)).item()
        )
        rsi_root_velocity_error = float(
            torch.max(
                torch.abs(robot.data.root_link_lin_vel_w[:, 0] - command.command[:, 0])
            ).item()
        )
        contact_sensor = env.scene["feet_ground_contact"]
        if contact_sensor.data.found is None:
            raise RuntimeError("feet contact sensor did not produce its required found field")
        rsi_contact_agreement = float(
            ((contact_sensor.data.found.reshape(1, -1)[:, :2] > 0) == command.expected_contact)
            .float()
            .mean()
            .item()
        )
        target_before_step = command.target_joint_position.clone()
        observation, reward, terminated, truncated, extras = env.step(
            torch.zeros((1, 29), device=args.device)
        )
        zero_residual_center_error = float(
            torch.max(torch.abs(action.joint_target - target_before_step)).item()
        )
        raw_logs = sorted(name for name in extras["log"] if name.startswith("WalkingV2Raw/"))
        weighted_logs = sorted(
            name for name in extras["log"] if name.startswith("WalkingV2Weighted/")
        )
        checks = {
            "actor_shape": tuple(observation["actor"].shape) == (1, 160),
            "critic_shape": tuple(observation["critic"].shape) == (1, 172),
            "finite_reward": bool(torch.isfinite(reward).all()),
            "not_terminated": not bool(terminated[0]),
            "not_truncated": not bool(truncated[0]),
            "rsi_pose_coherent": rsi_position_error <= 1e-6,
            "rsi_velocity_coherent": rsi_velocity_error <= 1e-6,
            "rsi_root_velocity_coherent": rsi_root_velocity_error <= 1e-6,
            "rsi_contact_state_coherent": rsi_contact_agreement == 1.0,
            "zero_residual_uses_cached_center": zero_residual_center_error <= 1e-7,
            "all_raw_reward_terms_logged": len(raw_logs) == 13,
            "all_weighted_reward_terms_logged": len(weighted_logs) == 13,
        }
        result = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "policy_contract_sha256": walking_v2_policy_contract().sha256,
            "reference_bank_metadata_sha256": sha256_file(Path(cfg.commands["twist"].bank_file)),
            "device": args.device,
            "physics_dt_s": env.physics_dt,
            "policy_dt_s": env.step_dt,
            "rsi_position_max_abs_error": rsi_position_error,
            "rsi_velocity_max_abs_error": rsi_velocity_error,
            "rsi_root_velocity_max_abs_error": rsi_root_velocity_error,
            "rsi_contact_agreement": rsi_contact_agreement,
            "zero_residual_center_max_abs_error": zero_residual_center_error,
            "reward": float(reward[0]),
            "raw_reward_logs": raw_logs,
            "weighted_reward_logs": weighted_logs,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device_index)),
            "checks": checks,
            "passed": all(checks.values()),
        }
    finally:
        env.close()
    write_atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
