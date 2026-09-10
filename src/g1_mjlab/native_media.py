"""Deterministic native-MuJoCo playback and video recording."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .deployment import apply_action, load_bundle, native_observation


def record_native_video(
    run: Path,
    scenarios: Path,
    output: Path,
    *,
    trial_id: int = 0,
    duration_s: float = 15.0,
    fps: int = 30,
    width: int = 960,
    height: int = 720,
) -> dict[str, Any]:
    """Record a deterministic ONNX rollout without constructing a learner."""
    import imageio.v2 as imageio
    import mujoco
    import numpy as np

    if duration_s <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("duration, fps, width and height must be positive")
    reference = json.loads(scenarios.read_text(encoding="utf-8"))
    if reference.get("schema_version") != 2:
        raise ValueError("requires version-2 scenario evidence")
    try:
        scenario = next(item for item in reference["trials"] if item["trial_id"] == trial_id)
    except StopIteration as error:
        raise ValueError(f"trial {trial_id} is not present in the scenario evidence") from error

    contract, model, session = load_bundle(run)
    bundle = json.loads((run / "policy-bundle.json").read_text(encoding="utf-8"))
    if reference["checkpoint_sha256"] != bundle["checkpoint_sha256"]:
        raise ValueError("scenario and frozen policy checkpoint do not match")
    control_dt = float(contract["control_dt"])
    decimation = round(control_dt / model.opt.timestep)
    if not math.isclose(decimation * model.opt.timestep, control_dt, abs_tol=1e-9):
        raise ValueError("nonintegral control decimation")
    duration_s = min(duration_s, float(reference["horizon_seconds"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"video already exists: {output}")
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    if partial.exists():
        raise FileExistsError(f"partial video already exists: {partial}")

    data = mujoco.MjData(model)
    initial = scenario["initial_state"]
    data.qpos[:3] = initial["root_position_w"]
    data.qpos[3:7] = initial["root_quaternion_wxyz"]
    data.qpos[contract["qpos_addresses"]] = initial["joint_position"]
    data.qvel[contract["dof_addresses"]] = initial["joint_velocity"]
    mujoco.mj_forward(model, data)
    previous = np.zeros(len(contract["actuator_ids"]), dtype=np.float32)
    renderer = mujoco.Renderer(model, height=height, width=width)
    frame_period = 1.0 / fps
    next_frame_time = 0.0
    frame_count = 0
    max_raw_action = 0.0
    max_applied_action = 0.0
    try:
        with imageio.get_writer(
            partial,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:
            while data.time < duration_s - 1e-12:
                observation = native_observation(model, data, contract, previous)
                raw = session.run(None, {session.get_inputs()[0].name: observation[None]})[0][0]
                if not np.isfinite(raw).all():
                    raise RuntimeError("nonfinite ONNX output during recording")
                previous = apply_action(data, contract, raw)
                max_raw_action = max(max_raw_action, float(np.max(np.abs(raw))))
                max_applied_action = max(max_applied_action, float(np.max(np.abs(previous))))
                for _ in range(decimation):
                    mujoco.mj_step(model, data)
                while next_frame_time <= data.time + 1e-12 and next_frame_time < duration_s:
                    renderer.update_scene(data, camera="robot/tracking")
                    # ImageIO's public writer type omits this backend-provided method.
                    writer.append_data(renderer.render())  # type: ignore[attr-defined]
                    frame_count += 1
                    next_frame_time += frame_period
    finally:
        renderer.close()
    partial.rename(output)

    metadata = {
        "schema_version": 1,
        "mode": "deterministic_inference_only",
        "learning_enabled": False,
        "backend": "native_mujoco_onnx_cpu",
        "trial_id": trial_id,
        "scenario_sha256": sha256_file(scenarios),
        "checkpoint": bundle["checkpoint"],
        "checkpoint_sha256": bundle["checkpoint_sha256"],
        "onnx_sha256": bundle["onnx_sha256"],
        "duration_seconds": duration_s,
        "fps": fps,
        "frame_count": frame_count,
        "resolution": [width, height],
        "max_absolute_raw_action": max_raw_action,
        "max_absolute_applied_action": max_applied_action,
        "video": output.name,
        "video_sha256": sha256_file(output),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def play_native(run: Path, scenarios: Path, *, trial_id: int = 0) -> None:
    """Open an interactive native-MuJoCo viewer with a frozen deterministic actor."""
    import mujoco
    import mujoco.viewer
    import numpy as np

    reference = json.loads(scenarios.read_text(encoding="utf-8"))
    if reference.get("schema_version") != 2:
        raise ValueError("requires version-2 scenario evidence")
    try:
        scenario = next(item for item in reference["trials"] if item["trial_id"] == trial_id)
    except StopIteration as error:
        raise ValueError(f"trial {trial_id} is not present in the scenario evidence") from error

    contract, model, session = load_bundle(run)
    bundle = json.loads((run / "policy-bundle.json").read_text(encoding="utf-8"))
    if reference["checkpoint_sha256"] != bundle["checkpoint_sha256"]:
        raise ValueError("scenario and frozen policy checkpoint do not match")
    control_dt = float(contract["control_dt"])
    decimation = round(control_dt / model.opt.timestep)
    if not math.isclose(decimation * model.opt.timestep, control_dt, abs_tol=1e-9):
        raise ValueError("nonintegral control decimation")

    data = mujoco.MjData(model)
    initial = scenario["initial_state"]
    data.qpos[:3] = initial["root_position_w"]
    data.qpos[3:7] = initial["root_quaternion_wxyz"]
    data.qpos[contract["qpos_addresses"]] = initial["joint_position"]
    data.qvel[contract["dof_addresses"]] = initial["joint_velocity"]
    mujoco.mj_forward(model, data)
    previous = np.zeros(len(contract["actuator_ids"]), dtype=np.float32)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("robot/pelvis").id
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -12.0
        while viewer.is_running():
            started = time.perf_counter()
            observation = native_observation(model, data, contract, previous)
            raw = session.run(None, {session.get_inputs()[0].name: observation[None]})[0][0]
            if not np.isfinite(raw).all():
                raise RuntimeError("nonfinite ONNX output during playback")
            previous = apply_action(data, contract, raw)
            for _ in range(decimation):
                mujoco.mj_step(model, data)
            viewer.sync()
            remaining = control_dt - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
