"""Bounded resolved-controller export and supported native actuator probes.

Simulator imports remain lazy so artifact validation is usable without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .config import ResolvedRunConfig


def canonical_hash(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def describe(value: Any) -> Any:
    """Serialize resolved configuration, preserving callable identities."""
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if is_dataclass(value) and not isinstance(value, type):
        return describe(asdict(value))
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if hasattr(value, "tolist"):
        return value.tolist()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported contract value: {type(value).__name__}")


def resolved_contract(env: Any, config: ResolvedRunConfig, output: Path) -> dict[str, Any]:
    """Export exact live action order, observation terms, model and PD arrays."""
    import mujoco

    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    model = env.sim.mj_model
    if env.action_manager.active_terms != ["joint_pos"]:
        raise ValueError("contract exporter requires one position-action term")
    names = list(action.target_names)
    if len(set(names)) != 29 or names != list(robot.joint_names):
        raise ValueError("G1 action/joint order does not match the declared 29-DOF interface")
    model_path = output / "model.mjb"
    mujoco.mj_saveModel(model, str(model_path), None)
    model_joint_ids = [model.joint(f"robot/{name}").id for name in names]
    actuator_ids = []
    for jid in model_joint_ids:
        matches = [index for index in range(model.nu) if int(model.actuator_trnid[index, 0]) == jid]
        if len(matches) != 1:
            raise ValueError(f"joint {jid} does not have exactly one actuator")
        actuator_ids.append(matches[0])

    def action_array(value: Any) -> list[float]:
        return (
            [float(value)] * len(names)
            if isinstance(value, (int, float))
            else value[0].cpu().tolist()
        )

    observations: dict[str, Any] = {}
    for group, terms in env.observation_manager.active_terms.items():
        offset = 0
        fields = []
        for name, dims in zip(
            terms, env.observation_manager.group_obs_term_dim[group], strict=True
        ):
            size = math.prod(dims)
            cfg = env.observation_manager.get_term_cfg(group, name)
            fields.append(
                {"name": name, "offset": offset, "size": size, "resolved_term": describe(cfg)}
            )
            offset += size
        observations[group] = fields
    if sum(item["size"] for item in observations["actor"]) != 99:
        raise ValueError("actor dimensions changed; version the deployment interface")
    sensors = []
    for index in range(model.nsensor):
        sensors.append(
            {
                "name": model.sensor(index).name,
                "type": int(model.sensor_type[index]),
                "object_id": int(model.sensor_objid[index]),
                "object_type": int(model.sensor_objtype[index]),
                "address": int(model.sensor_adr[index]),
                "dimension": int(model.sensor_dim[index]),
            }
        )
    contract = {
        "schema_version": 2,
        "robot": "Unitree G1 29-DOF",
        "config_sha256": config.sha256,
        "joint_names": names,
        "action_names": names,
        "actor_fields": observations["actor"],
        "critic_fields": observations["critic"],
        "action_target_ids": action.target_ids.cpu().tolist(),
        "action_scale": action_array(action.scale),
        "nominal_joint_position": action_array(action.offset),
        "action_clip": config.action_clip,
        "target_clip": describe(action.cfg.clip),
        "encoder_bias": robot.data.encoder_bias[0].cpu().tolist(),
        "physics_dt": float(model.opt.timestep),
        "control_dt": config.control_dt,
        "actuator_ids": actuator_ids,
        "model_joint_ids": model_joint_ids,
        "qpos_addresses": model.jnt_qposadr[model_joint_ids].tolist(),
        "dof_addresses": model.jnt_dofadr[model_joint_ids].tolist(),
        "hard_joint_limits": model.jnt_range[model_joint_ids].tolist(),
        "soft_joint_limits": robot.data.soft_joint_pos_limits[0].cpu().tolist(),
        "actuator_gain_parameters": model.actuator_gainprm[actuator_ids].tolist(),
        "actuator_bias_parameters": model.actuator_biasprm[actuator_ids].tolist(),
        "actuator_force_limits": model.actuator_forcerange[actuator_ids].tolist(),
        "actuator_force_limited": model.actuator_forcelimited[actuator_ids].tolist(),
        "actuator_control_limits": model.actuator_ctrlrange[actuator_ids].tolist(),
        "actuator_control_limited": model.actuator_ctrllimited[actuator_ids].tolist(),
        "actuator_gear": model.actuator_gear[actuator_ids].tolist(),
        "armature": model.dof_armature[model.jnt_dofadr[model_joint_ids]].tolist(),
        "friction_loss": model.dof_frictionloss[model.jnt_dofadr[model_joint_ids]].tolist(),
        "sensors": sensors,
        "site_positions": model.site_pos.tolist(),
        "site_quaternions": model.site_quat.tolist(),
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "normalizer": (
            "contained in each checkpoint and exported ONNX; bind its hash when selecting a policy"
        ),
        "initial_root_position": robot.data.root_link_pos_w[0].cpu().tolist(),
        "initial_root_quaternion_wxyz": robot.data.root_link_quat_w[0].cpu().tolist(),
        "action_semantics": (
            "raw policy -> optional symmetric action clip -> scale + nominal -> "
            "built-in position actuator; no extra PD"
        ),
    }
    contract["sha256"] = canonical_hash(contract)
    (output / "contract.json").write_text(
        json.dumps(contract, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return contract


def supported_step_probe(
    spec: Any,
    contract: dict[str, Any],
    config: ResolvedRunConfig,
    output: Path,
    options: Any,
) -> dict[str, Any]:
    """Signed 0.02-rad probes, each compared with a matched unexcited rollout.

    Remove only the free joint and suspend the pelvis; joint dynamics, gravity,
    transmissions and force limits remain enabled. This is native MuJoCo evidence.
    """
    import mujoco
    import numpy as np

    fixed = spec.copy()
    for key in list(fixed.keys):
        fixed.delete(key)
    for joint in list(fixed.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            fixed.delete(joint)
    pelvis = fixed.body("robot/pelvis")
    pelvis.pos = [0, 0, 1.2]
    pelvis.quat = contract["initial_root_quaternion_wxyz"]
    model = fixed.compile()
    # Match all configured solver/integrator settings, not only the time step.
    options.apply(model)
    names = contract["joint_names"]
    jids = [model.joint(f"robot/{name}").id for name in names]
    qids = model.jnt_qposadr[jids]
    dids = model.jnt_dofadr[jids]
    aids = contract["actuator_ids"]
    nominal = np.asarray(contract["nominal_joint_position"])
    steps = round(1.0 / config.physics_dt)
    step_start = round(0.25 / config.physics_dt)
    traces: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    def rollout(joint_index: int | None, amplitude: float) -> tuple[Any, Any]:
        data = mujoco.MjData(model)
        data.qpos[qids] = nominal
        data.ctrl[aids] = nominal
        mujoco.mj_forward(model, data)
        positions, forces = [], []
        for step in range(steps):
            if joint_index is not None and step >= step_start:
                data.ctrl[aids[joint_index]] = nominal[joint_index] + amplitude
            mujoco.mj_step(model, data)
            positions.append(data.qpos[qids].copy())
            forces.append(data.qfrc_actuator[dids].copy())
            if joint_index is not None:
                traces.append(
                    {
                        "joint": names[joint_index],
                        "amplitude_rad": amplitude,
                        "time_s": (step + 1) * config.physics_dt,
                        "position_rad": float(data.qpos[qids[joint_index]]),
                        "velocity_rad_s": float(data.qvel[dids[joint_index]]),
                        "target_rad": float(data.ctrl[aids[joint_index]]),
                        "generalized_force_nm": float(data.qfrc_actuator[dids[joint_index]]),
                    }
                )
        return np.asarray(positions), np.asarray(forces)

    baseline, _ = rollout(None, 0)
    for index, name in enumerate(names):
        for sign in (-1, 1):
            amplitude = sign * 0.02
            low, high = contract["hard_joint_limits"][index]
            if not low + 0.01 < nominal[index] + amplitude < high - 0.01:
                raise ValueError(f"probe target lacks joint margin: {name}")
            positions, forces = rollout(index, amplitude)
            response = positions[-20:, index].mean() - baseline[-20:, index].mean()
            limit = max(abs(x) for x in contract["actuator_force_limits"][index])
            occupancy = float(np.mean(np.abs(forces[step_start:, index]) >= 0.98 * limit))
            finite = bool(np.isfinite(positions).all() and np.isfinite(forces).all())
            direction = bool(response * sign > 0.002)
            results.append(
                {
                    "joint": name,
                    "amplitude_rad": amplitude,
                    "steady_delta_rad": float(response),
                    "finite": finite,
                    "direction_passed": direction,
                    "torque_limit_occupancy": occupancy,
                    "passed": finite and direction and occupancy < 0.10,
                }
            )
    (output / "supported-steps.jsonl").write_text(
        "".join(json.dumps(x, allow_nan=False) + "\n" for x in traces), encoding="utf-8"
    )
    return {
        "backend": "native_mujoco",
        "fixture": "pelvis fixed at 1.2m; gravity enabled; matched zero-step subtraction",
        "physics_dt": config.physics_dt,
        "passed": all(x["passed"] for x in results),
        "cases": results,
    }


def qualify_controller(config: ResolvedRunConfig, output: Path) -> dict[str, Any]:
    from mjlab.envs import ManagerBasedRlEnv

    from .environment import build_standing_train_config

    output.mkdir(parents=True, exist_ok=False)
    cfg = build_standing_train_config(replace(config, num_envs=1), output, randomized_reset=False)
    cfg.env.auto_reset = False
    env = ManagerBasedRlEnv(cfg=cfg.env, device=config.device, render_mode=None)
    try:
        env.reset(seed=config.seed)
        contract = resolved_contract(env, config, output)
        result = supported_step_probe(env.scene.spec, contract, config, output, cfg.env.sim.mujoco)
        floating = floating_probe(contract, output)
        result["floating"] = floating
        result["passed"] = result["passed"] and floating["passed"]
        result["contract_sha256"] = contract["sha256"]
        result["scope"] = (
            "native supported actuator and floating contact/timestep checks; "
            "not policy standing or backend trajectory-parity qualification"
        )
        (output / "summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return result
    finally:
        env.close()


def floating_probe(contract: dict[str, Any], output: Path) -> dict[str, Any]:
    """Characterize powered nominal hold and short numerical sensitivity."""
    import mujoco
    import numpy as np

    cases = []
    for divisor in (1, 2):
        model = mujoco.MjModel.from_binary_path(str(output / "model.mjb"))
        model.opt.timestep /= divisor
        data = mujoco.MjData(model)
        data.qpos[:3] = contract["initial_root_position"]
        data.qpos[3:7] = contract["initial_root_quaternion_wxyz"]
        data.qpos[contract["qpos_addresses"]] = contract["nominal_joint_position"]
        data.ctrl[contract["actuator_ids"]] = contract["nominal_joint_position"]
        mujoco.mj_forward(model, data)
        torso = model.body("robot/torso_link").id
        root = model.body("robot/pelvis").id
        trace = []
        first_fall = None
        supported_frames = 0
        penetration = 0.0
        for step in range(round(3.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            time_s = (step + 1) * model.opt.timestep
            contacts = []
            for contact in data.contact:
                names = [model.geom(int(gid)).name for gid in contact.geom]
                if any("foot" in name for name in names) and any(
                    "terrain" in name for name in names
                ):
                    contacts.append(contact.pos.tolist())
                    if time_s <= 0.5:
                        penetration = min(penetration, float(contact.dist))
            if contacts and time_s <= 0.5:
                supported_frames += 1
            tilt = float(np.degrees(np.arccos(np.clip(data.xmat[torso, 8], -1, 1))))
            if first_fall is None and tilt > 70:
                first_fall = time_s
            trace.append(
                {
                    "time_s": time_s,
                    "root_position": data.xpos[root].tolist(),
                    "com_position": data.subtree_com[root].tolist(),
                    "torso_tilt_deg": tilt,
                    "foot_ground_contact_positions": contacts,
                    "joint_position": data.qpos[contract["qpos_addresses"]].tolist(),
                    "joint_velocity": data.qvel[contract["dof_addresses"]].tolist(),
                    "generalized_force_nm": data.qfrc_actuator[contract["dof_addresses"]].tolist(),
                }
            )
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("nonfinite floating qualification state")
        filename = f"floating-{divisor}.jsonl"
        (output / filename).write_text(
            "".join(json.dumps(x, allow_nan=False) + "\n" for x in trace), encoding="utf-8"
        )
        cases.append(
            {
                "physics_dt": float(model.opt.timestep),
                "trace": filename,
                "first_torso_70deg_seconds": first_fall,
                "early_supported_frames": supported_frames,
                "early_min_contact_distance_m": penetration,
                "root_position_at_1s": trace[round(1 / model.opt.timestep) - 1]["root_position"],
            }
        )
    difference = float(
        np.linalg.norm(
            np.asarray(cases[0]["root_position_at_1s"]) - cases[1]["root_position_at_1s"]
        )
    )
    return {
        "cases": cases,
        "root_position_difference_at_1s_m": difference,
        "passed": all(
            x["early_supported_frames"] > 0 and x["early_min_contact_distance_m"] >= -0.01
            for x in cases
        )
        and difference < 0.05,
        "acceptance": (
            "foot-ground contact in first 0.5s; penetration <=1cm; 1s timestep root difference <5cm"
        ),
        "interpretation": (
            "Powered nominal hold is not a balancing controller; "
            "a later fall does not fail this gate."
        ),
    }
