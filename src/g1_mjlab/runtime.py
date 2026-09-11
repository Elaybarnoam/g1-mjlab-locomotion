"""Heavy mjlab adapters. Import only inside a qualified Linux runtime."""

# ruff: noqa: E501

from __future__ import annotations

import json
import math
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .artifacts import RunStore, read_jsonl, sha256_file, snapshot_source
from .checkpoints import checkpoint_index, ordered_checkpoints
from .config import PpoProfile, ResolvedRunConfig, StandingRewardProfile
from .environment import build_train_config
from .evaluation import StandingCriteria, TrialAccumulator, wilson_interval
from .tasks import TaskCapability, get_task

MJLAB_REVISION = "8ee51fbcf806a7419189f706d9e394cbeb7790fa"


def doctor(output: Path) -> dict[str, Any]:
    """Exercise Torch and mjlab imports and write a machine-readable diagnosis."""
    try:
        import mjlab  # noqa: F401
        import mujoco
        import torch
        import warp
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "doctor requires the Linux training stack; install with `uv sync --extra train`"
        ) from error

    started = time.monotonic()
    cuda = torch.cuda.is_available()
    tensor = torch.arange(8, device="cuda:0" if cuda else "cpu", dtype=torch.float32)
    result = {
        "schema_version": 1,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mjlab": version("mjlab"),
        "mujoco": mujoco.__version__,
        "torch": torch.__version__,
        "warp": getattr(warp, "__version__", "unknown"),
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "finite_tensor": bool(torch.isfinite(tensor).all().item()),
        "elapsed_seconds": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def harvest_tensorboard(log_dir: Path, store: RunStore) -> int:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    existing, truncated = read_jsonl(store.root / "metrics" / "metrics.jsonl")
    if truncated:
        raise ValueError("metric journal has a partial tail; preserve it before recovery")
    seen = {(r.get("metric"), r.get("update"), r.get("event_wall_time_unix")) for r in existing}
    event_files = sorted(log_dir.rglob("events.out.tfevents.*"))
    records: list[dict[str, Any]] = []
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                key = (tag, event.step, event.wall_time)
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    {
                        "phase": "train",
                        "update": event.step,
                        "metric": tag,
                        "value": event.value,
                        "unit": "upstream",
                        "event_wall_time_unix": event.wall_time,
                    }
                )
    store.append_metrics(records)
    return len(existing) + len(records)


def train(
    config: ResolvedRunConfig,
    run_dir: Path,
    source: dict[str, Any],
    reward_profile: StandingRewardProfile | None = None,
    ppo_profile: PpoProfile | None = None,
    resume: Path | None = None,
) -> Path:
    """Execute one bounded upstream training run and finalize local evidence."""
    task = get_task(config.task_id).require(TaskCapability.TRAIN)
    from .training import execute_training

    if resume is not None:
        from .training import validate_resume

        validate_resume(config, resume, reward_profile, ppo_profile)

    store = RunStore.create(
        run_dir,
        {
            "run_id": config.run_name,
            "run_type": "training",
            "config_sha256": config.sha256,
            "reward_profile_sha256": reward_profile.sha256 if reward_profile else None,
            "ppo_profile_sha256": ppo_profile.sha256 if ppo_profile else None,
            "max_iterations": config.max_iterations,
            **source,
        },
    )
    (run_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if reward_profile is not None:
        (run_dir / "reward-profile.json").write_text(
            json.dumps(reward_profile.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if ppo_profile is not None:
        (run_dir / "ppo-profile.json").write_text(
            json.dumps(ppo_profile.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    store.transition("starting")
    try:
        snapshot = snapshot_source(
            Path(__file__).resolve().parents[2],
            run_dir / "source.zip",
            config_directory=task.config_directory,
        )
        (run_dir / "source.json").write_text(
            json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
        )
        train_cfg = build_train_config(
            config,
            run_dir / "upstream",
            reward_profile=reward_profile,
            ppo_profile=ppo_profile,
        )
        (run_dir / "algorithm.json").write_text(
            json.dumps(asdict(train_cfg.agent), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        store.transition("running")
        execute_training(config, train_cfg, run_dir, resume=resume)
        logs = sorted((run_dir / "upstream").rglob("params/agent.yaml"))
        if not logs:
            raise RuntimeError("training returned without an upstream run directory")
        log_dir = logs[-1].parent.parent
        metric_count = harvest_tensorboard(log_dir, store)
        checkpoints = ordered_checkpoints(log_dir)
        for checkpoint in checkpoints:
            shutil.copy2(checkpoint, run_dir / "checkpoints" / checkpoint.name)
        for exported in sorted(log_dir.glob("*.onnx")):
            shutil.copy2(exported, run_dir / "checkpoints" / exported.name)
        final = checkpoints[-1] if checkpoints else None
        if final is None:
            raise RuntimeError("training returned without a checkpoint")
        contract = json.loads((run_dir / "contract.json").read_text(encoding="utf-8"))
        (run_dir / "policy-bundle.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "contract_sha256": contract["sha256"],
                    "model_sha256": contract["model_sha256"],
                    "checkpoint": final.name,
                    "checkpoint_sha256": sha256_file(final),
                    "normalizer_identity": "actor obs_normalizer tensors inside hashed checkpoint and ONNX",
                    "onnx_sha256": sha256_file(run_dir / "checkpoints" / "policy.onnx"),
                    "selection": "final budget checkpoint, not best-development or deployment-qualified",
                    "source_sha256": snapshot["sha256"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "checkpoints" / "index.json").write_text(
            json.dumps(checkpoint_index(run_dir / "checkpoints"), indent=2) + "\n",
            encoding="utf-8",
        )
        store.transition(
            "completed",
            metric_records=metric_count,
            upstream_log_dir=str(log_dir),
            final_checkpoint=final.name if final else None,
            final_checkpoint_sha256=sha256_file(final) if final else None,
        )
        return run_dir
    except KeyboardInterrupt:
        store.transition(
            "interrupted", failure="KeyboardInterrupt", **salvage_training(run_dir, store)
        )
        raise
    except Exception as exc:
        store.transition(
            "failed", failure=f"{type(exc).__name__}: {exc}", **salvage_training(run_dir, store)
        )
        raise


def salvage_training(run_dir: Path, store: RunStore) -> dict[str, Any]:
    """Best-effort exception finalization must never hide the primary error."""
    try:
        logs = sorted((run_dir / "upstream").rglob("params/agent.yaml"))
        if not logs:
            return {"recovery": "no upstream log directory created"}
        log_dir = logs[-1].parent.parent
        metric_count = harvest_tensorboard(log_dir, store)
        for checkpoint in ordered_checkpoints(log_dir):
            shutil.copy2(checkpoint, run_dir / "checkpoints" / checkpoint.name)
        index = checkpoint_index(run_dir / "checkpoints")
        (run_dir / "checkpoints" / "index.json").write_text(
            json.dumps(index, indent=2) + "\n", encoding="utf-8"
        )
        return {
            "metric_records": metric_count,
            "latest_checkpoint": index["latest"],
            "upstream_log_dir": str(log_dir),
        }
    except Exception as exc:
        return {"recovery_error": f"{type(exc).__name__}: {exc}"}


def git_source_metadata(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            args, cwd=project_root, check=True, text=True, capture_output=True
        ).stdout.strip()

    commit = run("git", "rev-parse", "HEAD")
    dirty = bool(run("git", "status", "--porcelain"))
    return {
        "source_commit": commit,
        "source_dirty": dirty,
        "mjlab_revision": MJLAB_REVISION,
    }


def evaluate(
    config: ResolvedRunConfig,
    checkpoint: Path,
    output_dir: Path,
    *,
    trials: int = 20,
    horizon_s: float = 60.0,
    seed: int = 10042,
    criteria: StandingCriteria | None = None,
    phase: str = "development",
) -> dict[str, Any]:
    """Measure first episodes before reset; these are development trials."""
    get_task(config.task_id).require(TaskCapability.STANDING_EVALUATION)
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls

    if not 1 <= trials <= 256 or not math.isfinite(horizon_s) or not 0 < horizon_s <= 120:
        raise ValueError("evaluation requires 1..256 trials and a finite horizon in (0, 120]")
    if seed < 0 or seed == config.seed:
        raise ValueError("development seed must be nonnegative and distinct from training")
    if phase not in {"development", "final"}:
        raise ValueError("evaluation phase must be development or final")
    if phase == "final" and (trials != 100 or not math.isclose(horizon_s, 60.0)):
        raise ValueError("final evaluation is frozen at exactly 100 trials x 60 seconds")
    criteria = criteria or StandingCriteria()
    horizon_steps = round(horizon_s / config.control_dt)
    if not math.isclose(horizon_steps * config.control_dt, horizon_s, abs_tol=1e-8):
        raise ValueError("horizon must be an integral number of control steps")
    if horizon_s <= criteria.settling_seconds:
        raise ValueError("evaluation horizon must exceed the settling interval")
    checkpoint = checkpoint.resolve(strict=True)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_config = replace(config, seed=seed, num_envs=trials, episode_length_s=horizon_s)
    train_cfg = build_train_config(eval_config, output_dir)
    train_cfg.env.auto_reset = False
    env = ManagerBasedRlEnv(cfg=train_cfg.env, device=config.device, render_mode=None)
    accumulators = [
        TrialAccumulator(horizon_steps, config.control_dt, criteria) for _ in range(trials)
    ]
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=train_cfg.agent.clip_actions)
        runner_cls = load_runner_cls(config.task_id) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(train_cfg.agent), device=config.device)
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=config.device
        )
        policy = runner.get_inference_policy(device=config.device)
        # Runner construction can consume RNG; explicitly reset the scenario batch afterward.
        env.reset(seed=seed)
        obs = wrapped.get_observations()
        robot = env.scene["robot"]
        torso_id = robot.body_names.index("torso_link")
        initial_xy = robot.data.root_link_pos_w[:, :2].clone()
        initial_states = {
            "root_position_w": robot.data.root_link_pos_w.cpu().tolist(),
            "root_quaternion_wxyz": robot.data.root_link_quat_w.cpu().tolist(),
            "joint_position": robot.data.joint_pos.cpu().tolist(),
            "joint_velocity": robot.data.joint_vel.cpu().tolist(),
        }
        with torch.inference_mode():
            for _ in range(horizon_steps):
                actions = policy(obs)
                obs, _, dones, _ = wrapped.step(actions)
                # Read independent flags before any explicit reset mutates manager buffers.
                terminated = env.termination_manager.terminated.cpu().tolist()
                truncated = env.termination_manager.time_outs.cpu().tolist()
                drift = torch.linalg.vector_norm(
                    robot.data.root_link_pos_w[:, :2] - initial_xy, dim=1
                )
                quat = robot.data.body_link_quat_w[:, torso_id]
                tilt = torch.rad2deg(
                    torch.acos((1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)).clamp(-1, 1))
                )
                found = env.scene["feet_ground_contact"].data.found
                support = (found > 0).reshape(trials, -1).any(dim=1)
                finite = (
                    torch.isfinite(robot.data.joint_pos).all(dim=1)
                    & torch.isfinite(robot.data.joint_vel).all(dim=1)
                    & torch.isfinite(robot.data.qfrc_actuator).all(dim=1)
                    & torch.isfinite(robot.data.root_link_pos_w).all(dim=1)
                    & torch.isfinite(actions).all(dim=1)
                )
                frames = (
                    torch.stack(
                        (drift, tilt, robot.data.root_link_pos_w[:, 2], support, finite), dim=1
                    )
                    .cpu()
                    .tolist()
                )
                for index, trial in enumerate(accumulators):
                    d, t, h, s, f = frames[index]
                    trial.observe(
                        terminated=bool(terminated[index]),
                        truncated=bool(truncated[index]),
                        drift_m=d,
                        torso_tilt_deg=t,
                        height_m=h,
                        supported=bool(s),
                        finite=bool(f),
                    )
                if all(trial.finished for trial in accumulators):
                    break
                done_ids = torch.nonzero(dones.bool() | ~finite, as_tuple=False).flatten()
                if done_ids.numel():
                    env.reset(env_ids=done_ids)
                    obs = wrapped.get_observations()
    finally:
        env.close()
    trial_records = [
        {
            "trial_id": index,
            "batch_seed": seed,
            "initial_state": {key: values[index] for key, values in initial_states.items()},
            "passed": trial.passed,
            "survival_passed": trial.survival_passed,
            "terminated": trial.terminated,
            "truncated": trial.truncated,
            "survived_seconds": trial.steps * config.control_dt,
            "max_drift_m": trial.max_drift_m,
            "max_torso_tilt_deg": trial.max_torso_tilt_deg,
            "min_height_after_settling_m": trial.min_height_m,
            "max_unsupported_seconds": trial.max_unsupported_seconds,
            "failure_reason": ",".join(sorted(trial.violations)) or None,
        }
        for index, trial in enumerate(accumulators)
    ]
    passed_count = sum(trial.passed for trial in accumulators)
    result = {
        "schema_version": 2,
        "phase": phase,
        "seed": seed,
        "config_sha256": config.sha256,
        "criteria": asdict(criteria),
        "criteria_status": "candidate development thresholds; not final-test qualification",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "planned": trials,
        "completed": trials,
        "passed": passed_count,
        "survival_passed": sum(trial.survival_passed for trial in accumulators),
        "standing_success_wilson_95": wilson_interval(passed_count, trials),
        "horizon_seconds": horizon_s,
        "deterministic_actor_mean": True,
        "trials": trial_records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def diagnose_standing(
    config: ResolvedRunConfig,
    checkpoint: Path,
    output_dir: Path,
    *,
    onnx_path: Path | None = None,
    trials: int = 4,
    horizon_s: float = 3.0,
) -> dict[str, Any]:
    """Compatibility entry point; implementation lives in :mod:`diagnostics`."""
    from .diagnostics import diagnose_standing as diagnose

    return diagnose(
        config,
        checkpoint,
        output_dir,
        onnx_path=onnx_path,
        trials=trials,
        horizon_s=horizon_s,
    )


def diagnose_checkpoints(
    config: ResolvedRunConfig,
    checkpoints: list[Path],
    output_dir: Path,
    *,
    horizon_s: float = 3.0,
) -> dict[str, Any]:
    """Compatibility entry point; implementation lives in :mod:`diagnostics`."""
    from .diagnostics import diagnose_checkpoints as diagnose

    return diagnose(config, checkpoints, output_dir, horizon_s=horizon_s)
