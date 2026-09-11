from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from g1_mjlab.config import (  # noqa: E402
    WalkingTrainingProfile,
    load_config,
    load_walking_training_profile,
)
from g1_mjlab.environment import build_train_config  # noqa: E402
from g1_mjlab.tasks.walking_mdp import heading_frame_delta  # noqa: E402


def _config():
    root = Path(__file__).parents[2]
    run = load_config(root / "configs" / "walking-v1" / "mdp-probe.json")
    return build_train_config(run, root / ".runtime" / "walking-mdp-probe")


def test_reference_foot_delta_is_rotated_from_world_into_heading_frame() -> None:
    root_quaternion = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]])
    world_delta = torch.tensor([[[1.0, 0.0, 0.0]]])

    actual = heading_frame_delta(world_delta, root_quaternion)

    torch.testing.assert_close(actual, torch.tensor([[[0.0, -1.0, 0.0]]]), atol=1e-6, rtol=0)


def test_walking_mdp_resolves_only_declared_terms() -> None:
    cfg = _config().env

    assert list(cfg.observations["actor"].terms)[-3:] == [
        "phase_sin",
        "phase_cos",
        "walk_blend",
    ]
    assert "base_motion" not in cfg.rewards
    assert set(cfg.rewards) == {
        "track_linear_velocity",
        "commanded_forward_progress",
        "track_angular_velocity",
        "upright",
        "reference_joint_pose",
        "reference_joint_velocity",
        "reference_foot_position",
        "reference_contact_timing",
        "foot_slip",
        "crouch",
        "dof_pos_limits",
        "action_rate_l2",
        "effort",
        "self_collisions",
        "termination",
        "pose",
        "body_ang_vel",
        "angular_momentum",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "soft_landing",
    }
    assert "forbidden_ground_contact" in cfg.terminations
    control_dt = cfg.sim.mujoco.timestep * cfg.decimation
    assert cfg.rewards["termination"].weight * control_dt == pytest.approx(-10.0)


def test_bootstrap_profile_uses_nominal_actions_and_upstream_locomotion_rewards() -> None:
    root = Path(__file__).parents[2]
    run = load_config(root / "configs" / "walking-v1" / "bootstrap-smoke.json")
    profile = load_walking_training_profile(
        root / "configs" / "walking-v1" / "bootstrap-profile.json"
    )

    cfg = build_train_config(run, root / ".runtime", walking_profile=profile).env

    assert cfg.actions["joint_pos"].use_default_offset is True
    assert cfg.rewards["reference_joint_pose"].weight == 0
    assert cfg.rewards["reference_contact_timing"].weight == 0
    assert cfg.rewards["track_linear_velocity"].weight == pytest.approx(2.0)
    assert cfg.rewards["track_linear_velocity"].params["std"] == pytest.approx(0.5)
    assert cfg.rewards["track_angular_velocity"].weight == pytest.approx(2.0)
    assert cfg.rewards["pose"].weight == pytest.approx(1.0)
    assert cfg.rewards["body_ang_vel"].weight == pytest.approx(-0.05)
    assert cfg.rewards["angular_momentum"].weight == pytest.approx(-0.02)
    assert cfg.rewards["action_rate_l2"].weight == pytest.approx(-0.1)
    assert cfg.rewards["foot_clearance"].weight == pytest.approx(-2.0)
    assert cfg.rewards["foot_swing_height"].weight == pytest.approx(-0.25)
    assert cfg.rewards["foot_slip"].weight == pytest.approx(-0.1)
    assert cfg.rewards["soft_landing"].weight == pytest.approx(-1e-5)
    assert cfg.rewards["termination"].weight == 0


def test_reference_only_profile_changes_only_command_reset_curriculum() -> None:
    root = Path(__file__).parents[2]
    run = load_config(root / "configs" / "walking-v1" / "stage2-train.json")
    profile = load_walking_training_profile(
        root / "configs" / "walking-v1" / "stage2-reference-only.json"
    )
    cfg = build_train_config(run, root / ".runtime", walking_profile=profile).env
    assert cfg.commands["twist"].standing_fraction == 0
    assert cfg.commands["twist"].randomize_phase
    assert cfg.commands["twist"].reference_initialization is True


def test_walking_profile_can_enable_forward_progress_reward() -> None:
    root = Path(__file__).parents[2]
    run = load_config(root / "configs" / "walking-v1" / "stage13-progressive-velocity-train.json")
    profile = WalkingTrainingProfile(
        schema_version=1,
        name="forward-progress",
        standing_fraction=0.5,
        reference_initialization=False,
        randomize_phase=False,
        forward_progress_weight=1.0,
        reference_foot_position_std_m=0.3,
    )

    cfg = build_train_config(run, root / ".runtime", walking_profile=profile).env

    assert cfg.rewards["commanded_forward_progress"].weight == 1.0
    assert cfg.rewards["reference_foot_position"].params["std_m"] == 0.3


@pytest.mark.gpu
def test_reference_profile_resets_moving_worlds_to_sampled_reference() -> None:
    root = Path(__file__).parents[2]
    run = load_config(root / "configs" / "walking-v1" / "stage2-train.json")
    profile = load_walking_training_profile(
        root / "configs" / "walking-v1" / "stage2-reference-only.json"
    )
    cfg = build_train_config(run, root / ".runtime", walking_profile=profile).env
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    try:
        env.reset(seed=42)
        command = env.command_manager.get_term("twist")
        robot = env.scene["robot"]
        assert torch.all(command.requested_command[:, 0] > 1.0)
        assert torch.all(command.blend == 1)
        torch.testing.assert_close(robot.data.joint_pos, command.joint_position, atol=2e-5, rtol=0)
        torch.testing.assert_close(robot.data.joint_vel, command.joint_velocity, atol=2e-5, rtol=0)
        action = env.action_manager.get_term("joint_pos")
        torch.testing.assert_close(
            action.offset,
            robot.data.default_joint_pos,
            atol=2e-5,
            rtol=0,
        )
    finally:
        env.close()


@pytest.mark.gpu
def test_walking_mdp_has_finite_batched_short_rollout() -> None:
    root = Path(__file__).parents[2]
    run = load_config(root / "configs" / "walking-v1" / "mdp-probe.json")
    standing_profile = WalkingTrainingProfile(
        schema_version=1,
        name="finite-standing-reset",
        standing_fraction=1.0,
        reference_initialization=False,
        randomize_phase=False,
    )
    cfg = build_train_config(
        run, root / ".runtime" / "walking-mdp-probe", walking_profile=standing_profile
    ).env
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    try:
        observations, _ = env.reset(seed=42)
        assert observations["actor"].shape == (4, 102)
        assert observations["critic"].shape == (4, 114)
        termination_count = 0
        for _ in range(25):
            observations, rewards, terminated, truncated, _ = env.step(
                torch.zeros((4, 29), device="cuda:0")
            )
            assert torch.isfinite(observations["actor"]).all()
            assert torch.isfinite(observations["critic"]).all()
            assert torch.isfinite(rewards).all()
            assert not torch.any(terminated & truncated)
            termination_count += int(terminated.sum())
        reported_rate = sum(
            values[0] for _, values in env.reward_manager.get_active_iterable_terms(0)
        )
        assert reported_rate * env.step_dt == pytest.approx(float(rewards[0]), abs=1e-5)
        assert termination_count < 4
        assert "Errors/base_linear_velocity_rms_m_s" in env.extras["log"]
        command = env.command_manager.get_term("twist")
        assert "command_filter_error_m_s" in command.metrics
        assert "command_error_m_s" not in command.metrics
    finally:
        env.close()
