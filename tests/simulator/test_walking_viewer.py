from __future__ import annotations

import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from g1_mjlab.config import WalkingTrainingProfile, load_config  # noqa: E402
from g1_mjlab.motion.gait import (  # noqa: E402
    CommandSchedule,
    CommandSegment,
    load_command_profile,
    load_command_schedule,
)
from g1_mjlab.motion.playback import WalkingScheduleCursor  # noqa: E402
from g1_mjlab.walking_viewer import ScheduledWalkingPolicy, WalkingNativeViewer  # noqa: E402


class FakeCommand:
    def __init__(self) -> None:
        self.requested_command = torch.zeros((1, 3))
        self.command = torch.zeros((1, 3))
        self.phase = torch.zeros(1)

    def set_requested_forward_speed(self, value: float) -> None:
        self.requested_command[:, 0] = value


class FakeTerminationManager:
    def __init__(self) -> None:
        self.terminated = torch.zeros(1, dtype=torch.bool)
        self.active_terms = ("fell_over",)

    def get_term(self, name: str):
        assert name == "fell_over"
        return self.terminated


class FakeEnvironment:
    def __init__(self) -> None:
        self.num_envs = 1
        self.device = "cpu"
        self.cfg = SimpleNamespace(viewer=SimpleNamespace(env_idx=0))
        self.unwrapped = self
        self.step_dt = 0.02
        self.termination_manager = FakeTerminationManager()
        self.steps = 0
        self.resets = 0
        self.terminate_next = False

    def get_observations(self):
        return torch.zeros((1, 102))

    def step(self, actions):
        assert actions.shape == (1, 29)
        self.steps += 1
        self.termination_manager.terminated[:] = self.terminate_next
        return self.get_observations(), torch.zeros(1), self.termination_manager.terminated, {}

    def reset(self):
        self.resets += 1
        self.termination_manager.terminated.zero_()
        return self.get_observations(), {}

    def close(self) -> None:
        pass


def _viewer(*, loop: bool = False, duration_s: float = 0.04):
    env = FakeEnvironment()
    command = FakeCommand()
    schedule = CommandSchedule(1, "test", "reference", (CommandSegment(duration_s, 0.6),))
    policy = ScheduledWalkingPolicy(
        lambda observation: torch.zeros((len(observation), 29)),
        command,  # type: ignore[arg-type]
        schedule,
        0.02,
    )
    viewer = WalkingNativeViewer(
        env,
        policy,
        checkpoint="model.pt",
        checkpoint_sha256="a" * 64,
        loop=loop,
    )
    return env, command, policy, viewer


def test_pause_advances_neither_policy_schedule_nor_environment() -> None:
    env, command, policy, viewer = _viewer()
    viewer.pause()
    viewer._last_tick_time = time.perf_counter() - 0.05
    viewer._time_until_next_render = 1.0
    viewer.tick()
    assert env.steps == 0
    assert policy.cursor.step == 0
    assert command.phase.item() == 0


def test_fall_pauses_on_terminal_state_without_automatic_reset() -> None:
    env, _, _, viewer = _viewer()
    env.terminate_next = True
    assert viewer._execute_step()
    assert viewer.get_status().paused
    assert viewer.fall_reason == "fell_over"
    assert env.resets == 0


def test_loop_queues_reset_only_after_complete_schedule() -> None:
    env, _, policy, viewer = _viewer(loop=True, duration_s=0.02)
    assert viewer._execute_step()
    assert policy.cursor.complete
    assert env.resets == 0
    viewer._process_actions()
    assert env.resets == 1
    assert policy.cursor.step == 0
    assert viewer.episode == 1


@pytest.mark.gpu
def test_first_100_viewer_actions_match_headless_actor_mean() -> None:
    """Prove that the viewer adapter does not create a second policy path."""
    root = Path(__file__).resolve().parents[2]
    checkpoint = root / ".runtime/walking-bootstrap-resume-10600-64/checkpoints/model_13398.pt"
    if not checkpoint.exists():
        pytest.skip("local qualified walking checkpoint is unavailable")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner

    from g1_mjlab.environment import build_train_config
    from g1_mjlab.rl_adapter import MjlabVecEnvWrapper
    from g1_mjlab.tasks.walking_mdp import WalkingCommand

    config = load_config(root / "configs/walking-v1/bootstrap-train.json")
    config = replace(config, num_envs=1, seed=10042, episode_length_s=60.0)
    profile = WalkingTrainingProfile(
        1,
        "deterministic-live-viewer-v2",
        1.0,
        False,
        False,
        host_semantics_version=2,
        domain_randomization=False,
        observation_noise=False,
    )
    train_cfg = build_train_config(
        config,
        root / ".runtime/walking-viewer-parity",
        randomized_reset=False,
        walking_profile=profile,
    )
    train_cfg.env.auto_reset = False
    env = ManagerBasedRlEnv(cfg=train_cfg.env, device=config.device, render_mode=None)
    try:
        wrapped = MjlabVecEnvWrapper(env, clip_actions=config.action_clip)
        runner = MjlabOnPolicyRunner(wrapped, asdict(train_cfg.agent), device=config.device)
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=config.device
        )
        actor = runner.get_inference_policy(device=config.device)
        command = env.command_manager.get_term("twist")
        assert isinstance(command, WalkingCommand)
        command_profile = load_command_profile(root / "configs/walking-v1/commands.json")
        schedule = load_command_schedule(
            root / "configs/walking-v1/bootstrap-development-schedule.json", command_profile
        )

        def headless_trace() -> tuple[torch.Tensor, list[torch.Tensor]]:
            env.reset(seed=10042)
            observations = wrapped.get_observations()
            cursor = WalkingScheduleCursor(schedule, config.control_dt)
            actions = []
            actor_inputs = []
            for _ in range(100):
                command.set_requested_forward_speed(cursor.next().requested_forward_speed_m_s)
                actor_inputs.append(observations.detach().clone())
                with torch.inference_mode():
                    action = actor(observations)
                actions.append(action.detach().cpu().clone())
                observations, _, _, _ = wrapped.step(action)
            return torch.stack(actions), actor_inputs

        expected, actor_inputs = headless_trace()
        # This also proves contact-history tensors created during stepping remain resettable.
        env.reset(seed=10042)
        viewer_policy = ScheduledWalkingPolicy(actor, command, schedule, config.control_dt)
        actual_actions = []
        for observation in actor_inputs:
            with torch.inference_mode():
                action = viewer_policy(observation)
            actual_actions.append(action.detach().cpu().clone())
        actual = torch.stack(actual_actions)

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    finally:
        env.close()
