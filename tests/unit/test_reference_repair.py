from __future__ import annotations

import pytest

from g1_mjlab.motion.reference_repair import CollisionRepairSettings


def test_collision_repair_settings_freeze_stricter_targets() -> None:
    settings = CollisionRepairSettings()

    settings.validate()
    assert settings.target_ground_penetration_m == 0.001
    assert settings.target_self_penetration_m == 0.001
    assert settings.maximum_root_lift_m == 0.080
    assert settings.arm_blend_increment == 0.01
    assert settings.source_joint_margin_rad == 0.023


@pytest.mark.parametrize(
    "settings",
    (
        CollisionRepairSettings(target_ground_penetration_m=0.006),
        CollisionRepairSettings(target_self_penetration_m=0.003),
        CollisionRepairSettings(maximum_root_lift_m=0.081),
        CollisionRepairSettings(arm_blend_increment=0.051),
        CollisionRepairSettings(source_joint_margin_rad=0.018),
    ),
)
def test_collision_repair_cannot_weaken_admission(settings: CollisionRepairSettings) -> None:
    with pytest.raises(ValueError):
        settings.validate()
