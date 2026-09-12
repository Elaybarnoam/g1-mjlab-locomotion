"""Single-device RSL-RL execution, with explicit ownership and artifact capture."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .artifacts import RunStore, sha256_file
from .checkpoints import checkpoint_iteration, publish_checkpoint
from .config import (
    PpoProfile,
    ResolvedRunConfig,
    Stage19RewardProfile,
    StandingRewardProfile,
    WalkingTrainingProfile,
)
from .qualification import describe, resolved_contract


def configure_transferred_action_std(
    actor: Any, initial_action_std: float, *, reset: bool
) -> str | None:
    """Optionally reset exploration noise after loading a transferred actor."""
    if not reset:
        return None
    import torch

    with torch.no_grad():
        actor.distribution.std_param.fill_(initial_action_std)
    return "actor action distribution standard deviation"


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
    walking_profile: WalkingTrainingProfile | None = None,
    walking_reward_profile: Stage19RewardProfile | None = None,
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
    path = root / "walking-profile.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    current = (
        json.loads(json.dumps(walking_profile.to_dict())) if walking_profile is not None else None
    )
    if old != current:
        raise ValueError("resume changes walking-profile")
    path = root / "walking-reward-profile.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    current_reward = (
        walking_reward_profile.to_dict() if walking_reward_profile is not None else None
    )
    if old != current_reward:
        raise ValueError("resume changes walking-reward-profile")


def validate_actor_initialization(config: ResolvedRunConfig, checkpoint: Path) -> dict[str, Any]:
    """Validate a new-run actor initialization without restoring learner state."""
    checkpoint = checkpoint.resolve(strict=True)
    source_root = checkpoint.parent.parent
    previous = json.loads((source_root / "config.json").read_text(encoding="utf-8"))
    if previous.get("task_id") != config.task_id:
        raise ValueError(
            "actor initialization currently requires the same task and policy layout; "
            "cross-task transfer needs an explicit observation remap"
        )
    return {
        "mode": "actor-and-actor-normalizer-only",
        "checkpoint": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "source_task_id": previous["task_id"],
        "fresh_components": ["critic", "optimizer", "iteration", "environment_state"],
    }


def validate_fine_tune_initialization(
    config: ResolvedRunConfig, checkpoint: Path
) -> dict[str, Any]:
    """Validate explicit same-task fine-tuning from a complete learner checkpoint."""
    metadata = validate_actor_initialization(config, checkpoint)
    return {
        "mode": "full-learner-state-fine-tune",
        "checkpoint": metadata["checkpoint"],
        "sha256": metadata["sha256"],
        "source_task_id": metadata["source_task_id"],
        "source_iteration": checkpoint_iteration(checkpoint),
        "restored_components": ["actor", "critic", "optimizer"],
        "fresh_components": ["iteration", "environment_state"],
    }


def execute_training(
    config: ResolvedRunConfig,
    train_cfg: Any,
    run_dir: Path,
    resume: Path | None = None,
    initialize_actor: Path | None = None,
    fine_tune: Path | None = None,
    ppo_profile: PpoProfile | None = None,
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

        lineage_checkpoint = resume if resume is not None else fine_tune
        checkpoint_lineage = (
            {
                "mode": "resume" if resume is not None else "fine_tune",
                "checkpoint": str(lineage_checkpoint),
                "checkpoint_sha256": sha256_file(lineage_checkpoint),
            }
            if lineage_checkpoint is not None
            else None
        )

        class RecordingRunner(MjlabOnPolicyRunner):
            def save(self, path: str, infos: Any = None) -> None:
                super().save(path, infos)
                publish_checkpoint(
                    Path(path),
                    run_dir / "checkpoints",
                    transitions_per_update=config.transitions_per_update,
                    source_lineage=checkpoint_lineage,
                )
                if self.logger.writer is not None:
                    self.logger.writer.flush()
                harvest_tensorboard(log_dir, RunStore.open(run_dir))

        runner = RecordingRunner(wrapped, agent_cfg, str(log_dir), device=config.device)
        kl_samples: list[Any] = []
        original_kl_divergence = runner.alg.actor.get_kl_divergence

        def record_kl_divergence(*args: Any, **kwargs: Any) -> Any:
            value = original_kl_divergence(*args, **kwargs)
            kl_samples.append(torch.mean(value).detach())
            return value

        runner.alg.actor.get_kl_divergence = record_kl_divergence
        if sum(value is not None for value in (resume, initialize_actor, fine_tune)) > 1:
            raise ValueError("resume, initialize_actor, and fine_tune are mutually exclusive")
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
        elif fine_tune is not None:
            initialization = validate_fine_tune_initialization(config, fine_tune)
            runner.load(str(fine_tune), strict=True, map_location=config.device)
            saved = torch.load(fine_tune, weights_only=False, map_location="cpu")
            restored_state = runner.alg.save()
            for key, state in restored_state.items():
                assert_state_equal(state, saved[key], key)
            runner.alg.learning_rate = saved["optimizer_state_dict"]["param_groups"][0]["lr"]
            runner.current_learning_iteration = 0
            initialization["restored_learning_rate"] = runner.alg.learning_rate
            (run_dir / "fine-tune.json").write_text(
                json.dumps(initialization, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif initialize_actor is not None:
            initialization = validate_actor_initialization(config, initialize_actor)
            saved = torch.load(initialize_actor, weights_only=False, map_location="cpu")
            runner.alg.actor.load_state_dict(saved["actor_state_dict"], strict=True)
            if ppo_profile is not None:
                reset_component = configure_transferred_action_std(
                    runner.alg.actor,
                    ppo_profile.initial_action_std,
                    reset=ppo_profile.reset_action_std_on_transfer,
                )
                initialization["reset_components"] = (
                    [reset_component] if reset_component is not None else []
                )
                initialization["preserved_components"] = (
                    ["actor action distribution standard deviation"]
                    if reset_component is None
                    else []
                )
            (run_dir / "actor-initialization.json").write_text(
                json.dumps(initialization, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        initial_actor = {
            name: value.detach().cpu().clone()
            for name, value in runner.alg.actor.named_parameters()
        }
        initial_critic = {
            name: value.detach().cpu().clone()
            for name, value in runner.alg.critic.named_parameters()
        }
        if config.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        runner.learn(config.max_iterations, init_at_random_ep_len=True)
        changes: dict[str, dict[str, float | bool]] = {}
        for component, before, after in (
            ("actor", initial_actor, dict(runner.alg.actor.named_parameters())),
            ("critic", initial_critic, dict(runner.alg.critic.named_parameters())),
        ):
            squared_delta = 0.0
            max_delta = 0.0
            finite = True
            for name, initial in before.items():
                final = after[name].detach().cpu()
                finite = finite and bool(torch.isfinite(final).all())
                if final.is_floating_point():
                    delta = final - initial
                    squared_delta += float(torch.sum(torch.square(delta)))
                    max_delta = max(max_delta, float(torch.max(torch.abs(delta))))
            changes[component] = {
                "all_parameters_finite": finite,
                "parameter_delta_l2": squared_delta**0.5,
                "parameter_delta_max_abs": max_delta,
                "parameters_changed": max_delta > 0,
            }
        gradients = [
            parameter.grad
            for parameter in (*runner.alg.actor.parameters(), *runner.alg.critic.parameters())
            if parameter.grad is not None
        ]
        gradient_finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        gradient_global_norm = (
            float(
                torch.sqrt(
                    torch.stack(
                        [torch.sum(torch.square(value.detach())) for value in gradients]
                    ).sum()
                ).cpu()
            )
            if gradients
            else 0.0
        )
        kl_values = torch.stack(kl_samples).detach().cpu().tolist() if kl_samples else []
        algorithm = agent_cfg["algorithm"]
        kl_samples_per_update = int(algorithm["num_learning_epochs"]) * int(
            algorithm["num_mini_batches"]
        )
        if not all(bool(item["all_parameters_finite"]) for item in changes.values()):
            raise FloatingPointError("PPO produced non-finite model parameters")
        if not all(bool(item["parameters_changed"]) for item in changes.values()):
            raise RuntimeError("PPO completed without changing actor and critic parameters")
        if not gradient_finite:
            raise FloatingPointError("PPO produced non-finite gradients")
        (run_dir / "optimization.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "updates": config.max_iterations,
                    "transitions_per_update": config.transitions_per_update,
                    "total_transitions": config.max_iterations * config.transitions_per_update,
                    "components": changes,
                    "recorded_gradient_tensor_count": len(gradients),
                    "all_recorded_gradients_finite": gradient_finite,
                    "final_gradient_global_norm": gradient_global_norm,
                    "kl_divergence_samples": kl_values,
                    "kl_samples_per_update": kl_samples_per_update,
                    "all_kl_samples_finite": all(math.isfinite(value) for value in kl_values),
                    "optimizer_state_entries": len(runner.alg.optimizer.state),
                    "final_learning_rate": runner.alg.learning_rate,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        normalizers: dict[str, Any] = {"schema_version": 1}
        for name, model in (("actor", runner.alg.actor), ("critic", runner.alg.critic)):
            state = model.obs_normalizer.state_dict()
            normalizers[name] = {
                key: {
                    "shape": list(value.shape),
                    "values": value.detach().cpu().tolist(),
                    "finite": bool(torch.isfinite(value).all()),
                }
                for key, value in state.items()
            }
        (run_dir / "normalizer-state.json").write_text(
            json.dumps(normalizers, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
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
