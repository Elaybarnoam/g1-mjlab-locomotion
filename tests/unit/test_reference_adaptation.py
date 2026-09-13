from __future__ import annotations

import numpy as np
import pytest

from g1_mjlab.motion.reference_adaptation import AdaptationSettings, periodic_fourier_resample


def test_periodic_fourier_resample_preserves_supported_harmonics() -> None:
    phase = np.arange(32) / 32
    values = np.stack((np.sin(2 * np.pi * phase), np.cos(4 * np.pi * phase)), axis=1)

    result = periodic_fourier_resample(values, 64, maximum_harmonic=2)
    target = np.arange(64) / 64

    np.testing.assert_allclose(result[:, 0], np.sin(2 * np.pi * target), atol=1e-12)
    np.testing.assert_allclose(result[:, 1], np.cos(4 * np.pi * target), atol=1e-12)


def test_adaptation_settings_reject_invalid_deployment_speed() -> None:
    with pytest.raises(ValueError, match="speed"):
        AdaptationSettings(speed_m_s=0.9, cycle_period_s=1.0).validate()
