import math

import mujoco


def test_native_mujoco_can_step_a_bounded_model() -> None:
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body pos='0 0 1'><freejoint/><geom size='.1'/></body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert math.isfinite(data.time)
    assert data.time > 0
