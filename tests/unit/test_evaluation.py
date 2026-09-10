import math

import pytest

from g1_mjlab.evaluation import StandingCriteria, TrialAccumulator, wilson_interval


def frame(trial: TrialAccumulator, **changes: object) -> None:
    values = dict(
        terminated=False,
        truncated=False,
        drift_m=0.0,
        torso_tilt_deg=0.0,
        height_m=0.75,
        supported=True,
        finite=True,
    )
    values.update(changes)
    trial.observe(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "terminated,truncated,passed",
    [
        (False, False, True),
        (False, True, True),
        (True, False, False),
        (True, True, False),
    ],
)
def test_terminal_precedence(terminated: bool, truncated: bool, passed: bool) -> None:
    trial = TrialAccumulator(1, 0.02, StandingCriteria(settling_seconds=0))
    frame(trial, terminated=terminated, truncated=truncated)
    assert trial.passed is passed
    assert trial.survival_passed is passed


def test_terminal_frame_retained_and_reset_ignored() -> None:
    trial = TrialAccumulator(10, 0.02, StandingCriteria())
    frame(trial, terminated=True, drift_m=0.8)
    frame(trial, drift_m=0.0)
    assert trial.max_drift_m == 0.8
    assert trial.steps == 1


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"drift_m": 0.3}, "drift"),
        ({"height_m": 0.4}, "height"),
        ({"torso_tilt_deg": 20}, "torso_tilt"),
        ({"height_m": math.nan}, "nonfinite_state"),
        ({"supported": False}, "unsupported"),
    ],
)
def test_survival_is_not_standing(changes: dict[str, object], reason: str) -> None:
    trial = TrialAccumulator(
        1, 0.02, StandingCriteria(settling_seconds=0, max_unsupported_seconds=0)
    )
    frame(trial, **changes)
    assert not trial.passed
    assert reason in trial.violations


def test_early_timeout_and_no_quality_samples_fail() -> None:
    trial = TrialAccumulator(10, 0.02, StandingCriteria())
    frame(trial, truncated=True)
    assert not trial.passed
    assert "early_timeout" in trial.violations


def test_wilson_interval() -> None:
    low, high = wilson_interval(95, 100)
    assert 0.88 < low < 0.90
    assert 0.97 < high < 0.99
