from __future__ import annotations

import json
from pathlib import Path

from g1_mjlab.artifacts import RunStore
from g1_mjlab.reporting import render_report


def test_report_is_offline_and_escapes_metadata(tmp_path: Path) -> None:
    store = RunStore.create(
        tmp_path / "run",
        {
            "run_id": "</script><script>alert(1)</script>",
            "config_sha256": "abc",
            "max_iterations": 2,
        },
    )
    store.transition("starting")
    store.transition("running")
    store.append_metric({"update": 0, "metric": "Loss/value", "value": 1.2})
    store.transition("completed")
    (store.root / "contract.json").write_text(
        json.dumps({"actor_fields": [{"name": "gravity", "offset": 0, "size": 3, "unit": "1"}]}),
        encoding="utf-8",
    )
    (store.root / "reward-profile.json").write_text(
        json.dumps({"termination_penalty": -5.0}), encoding="utf-8"
    )
    report = render_report(store.root)
    content = report.read_text(encoding="utf-8")
    assert "https://" not in content
    assert "</script><script>alert(1)</script>" not in content
    assert "Loss/value" in content
    assert "reward_profile.termination_penalty" in content


def test_failed_run_does_not_claim_completion(tmp_path: Path) -> None:
    store = RunStore.create(tmp_path / "run", {"run_id": "failed", "max_iterations": 1000})
    store.transition("failed", failure="test")
    content = render_report(store.root).read_text(encoding="utf-8")
    assert "budget completion is not established" in content
    assert "completed its declared update budget" not in content
    assert "p.x-a[0].x" in content


def test_report_links_recorded_video(tmp_path: Path) -> None:
    store = RunStore.create(tmp_path / "run", {"run_id": "video-run"})
    video = store.root / "videos" / "deterministic.mp4"
    video.write_bytes(b"video")

    content = render_report(store.root).read_text(encoding="utf-8")

    assert '<video controls preload="metadata"' in content
    assert "../videos/deterministic.mp4" in content
    assert "Deterministic playback was captured" in content


def test_curve_points_are_ordered_by_real_update() -> None:
    from g1_mjlab.reporting.render import _series

    points = _series(
        [{"metric": "loss", "update": 100, "value": 2}, {"metric": "loss", "update": 2, "value": 3}]
    )
    assert [p["x"] for p in points["loss"]] == [2, 100]


def test_walking_report_uses_gait_language_not_standing_claims(tmp_path: Path) -> None:
    store = RunStore.create(tmp_path / "run", {"run_id": "walking", "max_iterations": 10})
    store.transition("starting")
    store.transition("running")
    store.transition("completed")
    (store.root / "config.json").write_text(
        json.dumps({"task_id": "G1-Walking-Flat-v1"}), encoding="utf-8"
    )
    evaluation = store.root / "evaluation"
    evaluation.mkdir(exist_ok=True)
    (evaluation / "summary.json").write_text(
        json.dumps(
            {
                "phase": "development",
                "planned": 1,
                "passed_both": 0,
                "trials": [
                    {
                        "trial_id": 0,
                        "functional_passed": False,
                        "style_passed": False,
                        "survived_seconds": 10,
                        "completed_steps": 0,
                        "command_rms_m_s": 1.0,
                        "classification": "insufficient_steps",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    content = render_report(store.root).read_text(encoding="utf-8")
    assert "Walking function and style" in content
    assert "Strict standing passes" not in content
