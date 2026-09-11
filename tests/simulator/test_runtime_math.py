"""Simulator-adapter tests using the pinned learning runtime."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from g1_mjlab.rl_adapter import StandingVecEnvWrapper  # noqa: E402
from g1_mjlab.standing_task import healthy_standing, mean_squared_effort_cost  # noqa: E402


def test_healthy_reward_cannot_be_earned_lying_kneeling_or_flying() -> None:
    data = SimpleNamespace(
        body_link_quat_w=torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0]],
            ]
        ),
        root_link_pos_w=torch.tensor(
            [[0.0, 0.0, 0.75], [0.0, 0.0, 0.75], [0.0, 0.0, 0.4], [0.0, 0.0, 0.75]]
        ),
        qfrc_actuator=torch.tensor([[2.0, -2.0]]),
    )
    env = SimpleNamespace(
        device="cpu",
        scene={
            "robot": SimpleNamespace(body_names=["torso_link"], data=data),
            "feet_ground_contact": SimpleNamespace(
                data=SimpleNamespace(found=torch.tensor([[1, 1], [1, 1], [1, 1], [0, 0]]))
            ),
        },
        termination_manager=SimpleNamespace(terminated=torch.zeros(4, dtype=torch.bool)),
    )
    assert healthy_standing(env).tolist() == [1.0, 0.0, 0.0, 0.0]
    assert mean_squared_effort_cost(env).item() == 4.0


def test_wrapper_and_real_ppo_timeout_compensation() -> None:
    from unittest.mock import patch

    from mjlab.rl import RslRlVecEnvWrapper
    from rsl_rl.algorithms import PPO
    from rsl_rl.storage import RolloutStorage

    terminated = torch.tensor([False, False, True, True])
    truncated = torch.tensor([False, True, False, True])
    done = terminated | truncated
    wrapper = StandingVecEnvWrapper.__new__(StandingVecEnvWrapper)
    wrapper.env = SimpleNamespace(reset_terminated=terminated)
    with patch.object(
        RslRlVecEnvWrapper,
        "step",
        return_value=(None, torch.ones(4), done, {"time_outs": truncated}),
    ):
        obs, reward, dones, extras = wrapper.step(None)
    assert extras["time_outs"].tolist() == [False, True, False, False]
    learner = PPO.__new__(PPO)
    learner.transition = RolloutStorage.Transition()
    learner.transition.values = torch.full((4, 1), 2.0)
    learner.rnd = None
    learner.gamma = 0.99
    learner.device = "cpu"
    recorded = []
    learner.storage = SimpleNamespace(
        add_transition=lambda transition: recorded.append(transition.rewards.clone())
    )
    learner.actor = Mock()
    learner.critic = Mock()
    learner.process_env_step(obs, reward, dones, extras)
    torch.testing.assert_close(recorded[0], torch.tensor([1.0, 2.98, 1.0, 1.0]))
    torch.testing.assert_close(reward, torch.ones(4))


def test_native_observation_and_applied_action_order() -> None:
    import numpy as np

    from g1_mjlab.deployment import apply_action, native_observation

    fields = [
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
    ]
    contract = {
        "actor_fields": [
            {
                "name": name,
                "resolved_term": {
                    "scale": None,
                    "clip": None,
                    "history_length": 0,
                    "delay_max_lag": 0,
                },
            }
            for name in fields
        ],
        "qpos_addresses": list(range(7, 36)),
        "dof_addresses": list(range(6, 35)),
        "nominal_joint_position": [0.1] * 29,
        "encoder_bias": [0.0] * 29,
        "action_scale": [0.25] * 29,
        "actuator_ids": list(range(29)),
        "action_clip": 1.0,
        "target_clip": None,
    }
    model = SimpleNamespace(body=lambda name: SimpleNamespace(id=0))
    data = SimpleNamespace(
        qpos=np.zeros(36),
        qvel=np.zeros(35),
        ctrl=np.zeros(29),
        xmat=np.eye(3).reshape(1, 9),
        sensor=lambda name: SimpleNamespace(data=np.array([1.0, 2.0, 3.0])),
    )
    previous = apply_action(data, contract, np.full(29, 2.0))
    np.testing.assert_allclose(data.ctrl, 0.35)
    obs = native_observation(model, data, contract, previous)
    assert obs.shape == (99,)
    np.testing.assert_allclose(obs[6:9], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(obs[9:38], -0.1)
    np.testing.assert_allclose(obs[67:96], 1.0)
    np.testing.assert_allclose(obs[96:], 0.0)


def test_interruption_salvages_flushed_metrics_and_checkpoint(tmp_path, monkeypatch) -> None:
    import json
    from pathlib import Path

    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    import g1_mjlab.training
    from g1_mjlab.config import load_config
    from g1_mjlab.runtime import train

    config = load_config(Path(__file__).resolve().parents[2] / "configs/standing-v1/smoke.json")

    def interrupted(config, train_cfg, run_dir, resume=None, initialize_actor=None):
        log = run_dir / "upstream" / "fake"
        (log / "params").mkdir(parents=True)
        (log / "params" / "agent.yaml").write_text("{}", encoding="utf-8")
        torch.save({"iter": 1}, log / "model_1.pt")
        writer = EventFileWriter(str(log))
        writer.add_event(
            Event(
                wall_time=1.0,
                step=1,
                summary=Summary(value=[Summary.Value(tag="Loss/value", simple_value=0.5)]),
            )
        )
        writer.close()
        raise KeyboardInterrupt()

    monkeypatch.setattr(g1_mjlab.training, "execute_training", interrupted)
    run = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        train(config, run, {"source_commit": "test"})
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["metric_records"] == 1
    assert (run / "checkpoints" / "model_1.pt").exists()
