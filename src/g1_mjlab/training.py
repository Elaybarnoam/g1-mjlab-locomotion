"""Single-device RSL-RL execution, with explicit ownership and artifact capture."""

from __future__ import annotations

import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .artifacts import RunStore, sha256_file
from .checkpoints import checkpoint_iteration
from .config import PpoProfile, ResolvedRunConfig, StandingRewardProfile
from .qualification import describe, resolved_contract


def assert_state_equal(actual: Any, expected: Any, path: str = "learner") -> None:
    """Fail closed if any restored tensor or optimizer counter differs."""
    import torch

    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual.cpu(), expected.cpu()):
            raise ValueError(f"resume tensor mismatch: {path}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"resume mapping mismatch: {path}")
        for key in expected:
            assert_state_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            raise ValueError(f"resume sequence mismatch: {path}")
        for index, value in enumerate(expected):
            assert_state_equal(actual[index], value, f"{path}[{index}]")
    elif actual != expected:
        raise ValueError(f"resume value mismatch: {path}")


def validate_resume(
    config: ResolvedRunConfig,
    checkpoint: Path,
    reward_profile: StandingRewardProfile | None,
    ppo_profile: PpoProfile | None,
) -> None:
    """Only full-state continuation of the same declared learning problem is allowed."""
    checkpoint.resolve(strict=True)
    root = checkpoint.parent.parent
    previous = json.loads((root / "config.json").read_text(encoding="utf-8"))
    mutable = {
        "run_name",
        "seed",
        "num_envs",
        "max_iterations",
        "save_interval",
        "transitions_per_update",
    }
    for key, value in config.to_dict().items():
        if key not in mutable and previous.get(key) != value:
            raise ValueError(
                f"resume changes {key}; use a separately declared fine-tuning workflow"
            )
    for name, profile in (("reward-profile", reward_profile), ("ppo-profile", ppo_profile)):
        path = root / f"{name}.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if old != (profile.to_dict() if profile else None):
            raise ValueError(f"resume changes {name}")


def execute_training(
    config: ResolvedRunConfig,
    train_cfg: Any,
    run_dir: Path,
    resume: Path | None = None,
) -> None:
    """Resume learner state into fresh worlds; never claim exact trajectory replay."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    from .rl_adapter import MjlabVecEnvWrapper
    from .runtime import harvest_tensorboard

    configure_torch_backends(allow_tf32=True, deterministic=False)
    (run_dir / "runtime.json").write_text(
        json.dumps(
            {
                "packages": {
                    name: version(name)
                    for name in (
                        "torch",
                        "mujoco",
                        "mujoco-warp",
                        "warp-lang",
                        "rsl-rl-lib",
                        "onnxruntime",
                    )
                },
                "allow_tf32": True,
                "deterministic_backends": False,
                "timeout_bootstrap": "stored-current-value; true termination takes precedence",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log_dir = run_dir / "upstream" / train_cfg.agent.experiment_name / config.run_name
    log_dir.mkdir(parents=True, exist_ok=False)
    agent_cfg = asdict(train_cfg.agent)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)
    dump_yaml(log_dir / "params" / "env.yaml", asdict(train_cfg.env))
    env = ManagerBasedRlEnv(cfg=train_cfg.env, device=config.device, render_mode=None)
    runner: Any = None
    try:
        wrapped = MjlabVecEnvWrapper(env, clip_actions=config.action_clip)
        resolved_contract(env, config, run_dir)
        (run_dir / "mdp.json").write_text(
            json.dumps(
                describe(
                    {
                        "rewards": train_cfg.env.rewards,
                        "terminations": train_cfg.env.terminations,
                        "events": train_cfg.env.events,
                        "commands": train_cfg.env.commands,
                    }
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        class RecordingRunner(MjlabOnPolicyRunner):  # type: ignore[misc]
            def save(self, path: str, infos: Any = None) -> None:
                super().save(path, infos)
                if self.logger.writer is not None:
                    self.logger.writer.flush()
                harvest_tensorboard(log_dir, RunStore.open(run_dir))

        runner = RecordingRunner(wrapped, agent_cfg, str(log_dir), device=config.device)
        if resume is not None:
            previous_mdp = json.loads(
                (resume.parent.parent / "mdp.json").read_text(encoding="utf-8")
            )
            if previous_mdp != json.loads((run_dir / "mdp.json").read_text(encoding="utf-8")):
                raise ValueError("resolved MDP differs from resume source")
            runner.load(str(resume), strict=True, map_location=config.device)
            saved = torch.load(resume, weights_only=False, map_location="cpu")
            restored_state = runner.alg.save()
            for key, state in restored_state.items():
                assert_state_equal(state, saved[key], key)
            runner.alg.learning_rate = saved["optimizer_state_dict"]["param_groups"][0]["lr"]
            # Upstream stores the just-completed zero-based iteration.
            restored = int(runner.current_learning_iteration)
            if restored != checkpoint_iteration(resume):
                raise ValueError("checkpoint filename and restored iteration disagree")
            runner.current_learning_iteration = restored + 1
            (run_dir / "resume.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(resume),
                        "sha256": sha256_file(resume),
                        "restored_iteration": restored,
                        "first_new_iteration": restored + 1,
                        "mode": "full learner state; fresh environments and RNG; not exact replay",
                        "additional_updates": config.max_iterations,
                        "verified_restored_components": list(restored_state),
                        "restored_learning_rate": runner.alg.learning_rate,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if config.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        runner.learn(config.max_iterations, init_at_random_ep_len=True)
        runner.export_policy_to_onnx(str(log_dir), filename="policy.onnx")
        if config.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            (run_dir / "memory.json").write_text(
                json.dumps(
                    {
                        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "device_free_at_end_bytes": free,
                        "device_total_bytes": total,
                        "limitation": (
                            "Torch allocator excludes Warp allocations; "
                            "end free is not peak device headroom"
                        ),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    finally:
        try:
            if runner is not None and runner.logger.writer is not None:
                runner.logger.stop_logging_writer()
        finally:
            env.close()
