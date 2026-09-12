"""Interactive deterministic walking playback in mjlab's native viewer."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import VerbosityLevel

from .artifacts import sha256_file
from .config import ResolvedRunConfig, WalkingTrainingProfile
from .motion.gait import CommandSchedule, load_command_profile, load_command_schedule
from .motion.playback import PlaybackFrame, WalkingScheduleCursor
from .tasks import TaskCapability, get_task
from .tasks.walking_mdp import WalkingCommand


class ScheduledWalkingPolicy:
    """Set the requested command immediately before deterministic actor inference."""

    def __init__(
        self,
        actor: Callable[[torch.Tensor], torch.Tensor],
        command: WalkingCommand,
        schedule: CommandSchedule,
        control_dt: float,
    ) -> None:
        self.actor = actor
        self.command = command
        self.cursor = WalkingScheduleCursor(schedule, control_dt)
        self.last_frame = PlaybackFrame(0, 0.0, False)

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        self.last_frame = self.cursor.next()
        self.command.set_requested_forward_speed(self.last_frame.requested_forward_speed_m_s)
        return self.actor(observation)

    def reset(self) -> None:
        self.cursor.reset()
        self.last_frame = PlaybackFrame(0, 0.0, False)


class WalkingNativeViewer(NativeMujocoViewer):
    """Walking-specific controls and evidence overlay over mjlab's GPU-state viewer."""

    def __init__(
        self,
        env: Any,
        policy: ScheduledWalkingPolicy,
        *,
        checkpoint: str,
        checkpoint_sha256: str,
        loop: bool,
    ) -> None:
        super().__init__(
            env,
            policy,  # type: ignore[arg-type]
            frame_rate=60.0,
            enable_perturbations=False,
            verbosity=VerbosityLevel.INFO,
        )
        self.walking_policy = policy
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = checkpoint_sha256
        self.loop = loop
        self.episode = 0
        self.fall_reason: str | None = None

    def setup(self) -> None:
        import mujoco

        super().setup()
        assert self.viewer is not None and self.mjm is not None
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = self.mjm.body("robot/pelvis").id
        self.viewer.cam.distance = 3.0
        self.viewer.cam.azimuth = 135.0
        self.viewer.cam.elevation = -12.0

    def _termination_reason(self) -> str:
        manager = self.env.unwrapped.termination_manager
        for name in manager.active_terms:
            if name != "time_out" and bool(manager.get_term(name)[0]):
                return str(name)
        return "terminated"

    def _execute_step(self) -> bool:
        """Run one actor-mean step and expose terminal state without auto restart."""
        try:
            observation = self.env.get_observations()
            with torch.inference_mode():
                actions = self.walking_policy(observation)
            # MJLab updates resettable contact history during step(). Creating that state in
            # inference mode turns it into immutable inference tensors and breaks R/loop reset.
            self.env.step(actions)
            self._step_count += 1
            self._stats_steps += 1
            if bool(self.env.unwrapped.termination_manager.terminated[0]):
                self.fall_reason = self._termination_reason()
                self.pause()
            elif self.loop and self.walking_policy.cursor.complete:
                self.request_reset()
            return True
        except Exception as error:
            self._last_error = f"{type(error).__name__}: {error}"
            self.pause()
            return False

    def reset_environment(self) -> None:
        super().reset_environment()
        self.episode += 1
        self.fall_reason = None

    def _safe_key_callback(self, key: int) -> None:
        """Queue UI actions; simulator state is touched only by the main thread."""
        from mjlab.viewer.native.keys import (
            KEY_EQUAL,
            KEY_MINUS,
            KEY_R,
            KEY_RIGHT,
            KEY_SPACE,
        )

        if key == KEY_SPACE:
            self.request_toggle_pause()
        elif key == KEY_R:
            self.request_reset()
        elif key == KEY_MINUS:
            self.request_speed_down()
        elif key == KEY_EQUAL:
            self.request_speed_up()
        elif key == KEY_RIGHT:
            self.request_single_step()

    def _set_status_overlay(self, viewer: Any) -> None:
        import mujoco

        command = self.walking_policy.command
        requested = float(command.requested_command[0, 0].detach().cpu())
        applied = float(command.command[0, 0].detach().cpu())
        phase = float(command.phase[0].detach().cpu())
        status = self.get_status()
        state = (
            f"FALLEN: {self.fall_reason}"
            if self.fall_reason
            else ("PAUSED" if status.paused else "RUNNING")
        )
        labels = (
            "Mode\nCheckpoint\nHash\nEpisode\nState\nSim time\nRequested vx\nApplied vx\nPhase\nRTF\nControls"
        )
        values = (
            "deterministic actor mean; learning disabled\n"
            f"{self.checkpoint}\n{self.checkpoint_sha256[:12]}\n{self.episode}\n{state}\n"
            f"{status.step_count * self.env.unwrapped.step_dt:.2f} s\n"
            f"{requested:.3f} m/s\n{applied:.3f} m/s\n{phase:.4f}\n"
            f"{status.actual_realtime:.2f}x / {status.target_realtime:.2f}x\n"
            "Space pause | R reset | +/- speed"
        )
        viewer.set_texts(
            (
                mujoco.mjtFontScale.mjFONTSCALE_150.value,
                mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                labels,
                values,
            )
        )

    def close(self) -> None:
        """Let the asynchronous GL swap finish before destroying the native window."""
        if self.viewer is not None and self.viewer.is_running():
            time.sleep(max(0.05, 2 * self.frame_time))
        super().close()


def play_walking(
    config: ResolvedRunConfig,
    checkpoint: Path,
    schedule_path: Path,
    *,
    seed: int,
    loop: bool = False,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Launch one visible mjlab-backed deterministic walking simulation."""
    get_task(config.task_id).require(TaskCapability.WALKING_EVALUATION)
    if seed < 0 or (duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0)):
        raise ValueError("seed must be nonnegative and duration must be finite and positive")
    checkpoint = checkpoint.resolve(strict=True)
    config_root = Path(__file__).resolve().parents[2] / "configs" / "walking-v1"
    command_profile = load_command_profile(config_root / "commands.json")
    schedule = load_command_schedule(schedule_path.resolve(strict=True), command_profile)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner

    from .environment import build_train_config
    from .rl_adapter import MjlabVecEnvWrapper

    viewer_horizon = duration_s + config.control_dt if duration_s is not None else 86_400.0
    viewer_config = replace(
        config,
        seed=seed,
        num_envs=1,
        episode_length_s=max(config.episode_length_s, viewer_horizon),
    )
    profile = WalkingTrainingProfile(
        schema_version=1,
        name="deterministic-live-viewer-v2",
        standing_fraction=1.0,
        reference_initialization=False,
        randomize_phase=False,
        host_semantics_version=2,
        domain_randomization=False,
        observation_noise=False,
    )
    train_cfg = build_train_config(
        viewer_config,
        Path(".runtime/walking-viewer"),
        randomized_reset=False,
        walking_profile=profile,
    )
    train_cfg.env.auto_reset = False
    env = ManagerBasedRlEnv(cfg=train_cfg.env, device=config.device, render_mode=None)
    checkpoint_hash = sha256_file(checkpoint)
    try:
        wrapped = MjlabVecEnvWrapper(env, clip_actions=config.action_clip)
        runner = MjlabOnPolicyRunner(wrapped, asdict(train_cfg.agent), device=config.device)
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=config.device
        )
        actor = runner.get_inference_policy(device=config.device)
        env.reset(seed=seed)
        command = env.command_manager.get_term("twist")
        if not isinstance(command, WalkingCommand):
            raise TypeError("walking viewer requires WalkingCommand")
        policy = ScheduledWalkingPolicy(actor, command, schedule, config.control_dt)
        viewer = WalkingNativeViewer(
            wrapped,
            policy,
            checkpoint=checkpoint.name,
            checkpoint_sha256=checkpoint_hash,
            loop=loop,
        )
        num_steps = None if duration_s is None else math.ceil(duration_s / config.control_dt)
        viewer.run(num_steps=num_steps)
        status = viewer.get_status()
        return {
            "schema_version": 1,
            "backend": "mjlab NativeMujocoViewer; GPU physics with passive CPU visualization",
            "learning_enabled": False,
            "deterministic_actor_mean": True,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": checkpoint_hash,
            "schedule": schedule.name,
            "seed": seed,
            "completed_steps": status.step_count,
            "episode": viewer.episode,
            "fall_reason": viewer.fall_reason,
            "actual_realtime_factor": status.actual_realtime,
        }
    finally:
        env.close()
