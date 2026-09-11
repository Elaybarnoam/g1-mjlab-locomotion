from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from g1_mjlab.config import load_config  # noqa: E402
from g1_mjlab.environment import build_train_config  # noqa: E402


def _config():
    root = Path(__file__).parents[2]
    run = load_config(root / "configs" / "walking-v1" / "mdp-probe.json")
    return build_train_config(run, root / ".runtime" / "walking-mdp-probe")


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
    }
    assert "forbidden_ground_contact" in cfg.terminations
    control_dt = cfg.sim.mujoco.timestep * cfg.decimation
    assert cfg.rewards["termination"].weight * control_dt == pytest.approx(-10.0)


@pytest.mark.gpu
def test_walking_mdp_has_finite_batched_short_rollout() -> None:
    cfg = _config().env
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
