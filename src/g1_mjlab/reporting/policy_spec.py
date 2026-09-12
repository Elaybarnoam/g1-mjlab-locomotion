"""Generate the exact walking policy specification from artifacts and tensors."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, cast

from ..artifacts import sha256_file

_FIELD_METADATA = {
    "base_lin_vel": ("m/s", "pelvis/body"),
    "base_ang_vel": ("rad/s", "pelvis/body"),
    "projected_gravity": ("1", "pelvis/body"),
    "joint_pos": ("rad", "joint order"),
    "joint_vel": ("rad/s", "joint order"),
    "actions": ("normalized", "previous policy output"),
    "command": ("m/s,m/s,rad/s", "pelvis heading"),
    "phase_sin": ("1", "host gait clock"),
    "phase_cos": ("1", "host gait clock"),
    "walk_blend": ("1", "host state"),
    "foot_height": ("m", "world"),
    "foot_air_time": ("s", "contact history"),
    "foot_contact": ("bool", "contact sensor"),
    "foot_contact_forces": ("N", "world"),
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "positive_infinity" if value > 0 else "negative_infinity" if value < 0 else "nan"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _scalar_layout(
    fields: list[dict[str, Any]], *, used_by_inference: bool
) -> list[dict[str, Any]]:
    result = []
    for field in fields:
        unit, frame = _FIELD_METADATA.get(field["name"], ("missing", "missing"))
        term = field.get("resolved_term", {})
        for local_index in range(int(field["size"])):
            result.append(
                {
                    "index": int(field["offset"]) + local_index,
                    "field": field["name"],
                    "field_index": local_index,
                    "unit": unit,
                    "frame": frame,
                    "transform": {
                        "scale": term.get("scale"),
                        "clip": term.get("clip"),
                        "noise": term.get("noise"),
                        "history_length": term.get("history_length", 0),
                    },
                    "training": True,
                    "inference": used_by_inference,
                }
            )
    return result


def _network(state: dict[str, Any], prefix: str, sample: Any | None = None) -> dict[str, Any]:
    import torch.nn.functional as functional

    layers = []
    current = sample
    for index in (0, 2, 4, 6):
        weight = state[f"mlp.{index}.weight"]
        bias = state[f"mlp.{index}.bias"]
        entry: dict[str, Any] = {
            "name": f"{prefix}.mlp.{index}",
            "weight_shape": list(weight.shape),
            "bias_shape": list(bias.shape),
            "parameters": int(weight.numel() + bias.numel()),
            "activation": "linear" if index == 6 else "elu",
        }
        if current is not None:
            current = functional.linear(current, weight, bias)
            entry["worked_output"] = current.tolist()
            if index != 6:
                current = functional.elu(current)
        layers.append(entry)
    return {
        "layers": layers,
        "parameter_count": sum(layer["parameters"] for layer in layers),
        "worked_final_output": current.tolist() if current is not None else "missing",
    }


def generate_walking_policy_spec(
    run: Path, checkpoint: Path, parity_summary: Path, output: Path
) -> dict[str, Any]:
    """Create JSON, Markdown, and HTML from one validated specification object."""
    import torch

    run = run.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    parity = _read(parity_summary)
    contract = _read(run / "contract.json")
    algorithm = _read(run / "algorithm.json")
    mdp = _read(run / "mdp.json")
    bundle = _read(run / "policy-bundle.json")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor_state = saved["actor_state_dict"]
    critic_state = saved["critic_state_dict"]
    capture = parity.get("records", [{}])[0].get("capture")
    raw_observation = capture.get("actor_observation") if isinstance(capture, dict) else None
    normalized = None
    if raw_observation is not None:
        vector = torch.as_tensor(raw_observation, dtype=torch.float32)
        normalized = (vector - actor_state["obs_normalizer._mean"][0]) / actor_state[
            "obs_normalizer._std"
        ][0]
    actor = _network(actor_state, "actor", normalized)
    critic = _network(critic_state, "critic")
    normalizer = {
        "mean": actor_state["obs_normalizer._mean"][0].tolist(),
        "variance": actor_state["obs_normalizer._var"][0].tolist(),
        "standard_deviation": actor_state["obs_normalizer._std"][0].tolist(),
        "count": float(actor_state["obs_normalizer.count"]),
        "epsilon": "missing from serialized tensors; fixed by pinned upstream implementation",
        "mode": "frozen inside ONNX for inference",
    }
    optimizer_groups = saved["optimizer_state_dict"].get("param_groups", [])
    effective_lr = optimizer_groups[0].get("lr") if optimizer_groups else "missing"
    actions = []
    for index, name in enumerate(contract["action_names"]):
        actions.append(
            {
                "index": index,
                "joint": name,
                "scale_rad": contract["action_scale"][index],
                "nominal_rad": contract["nominal_joint_position"][index],
                "encoder_bias_rad": contract["encoder_bias"][index],
                "qpos_address": contract["qpos_addresses"][index],
                "dof_address": contract["dof_addresses"][index],
                "actuator_id": contract["actuator_ids"][index],
                "kp": contract["actuator_gain_parameters"][index][0],
                "kd": -contract["actuator_bias_parameters"][index][2],
                "armature": contract["armature"][index],
                "force_limit": contract["actuator_force_limits"][index],
                "action_clip": contract["action_clip"],
                "target_clip": contract["target_clip"],
            }
        )
    specification = {
        "schema_version": 1,
        "task_id": contract["task_id"],
        "layout_id": contract["layout_id"],
        "status": "unqualified_development",
        "checkpoint": {"file": checkpoint.name, "sha256": sha256_file(checkpoint)},
        "source_bundle": bundle,
        "actor_inputs": _scalar_layout(contract["actor_fields"], used_by_inference=True),
        "critic_inputs": _scalar_layout(contract["critic_fields"], used_by_inference=False),
        "outputs": actions,
        "normalizer": normalizer,
        "networks": {"actor": actor, "critic": critic},
        "algorithm": {
            "configured": algorithm,
            "effective_learning_rate": effective_lr,
            "effective_action_std": actor_state["distribution.std_param"].tolist(),
            "timeout_semantics": (
                "RSL-RL 5.5.0 adds gamma times the stored current transition value to reward "
                "for timeout-only dones; true termination takes precedence in MjlabVecEnvWrapper"
            ),
            "equations": {
                "td_residual": "delta_t = r_t + gamma * V_next - V_t",
                "gae": "A_t = delta_t + gamma * lambda * continuation_mask * A_(t+1)",
                "return_target": "R_t = A_t + V_t",
                "ppo_actor": "min(ratio*A, clip(ratio,1-epsilon,1+epsilon)*A)",
                "value": "maximum of unclipped and clipped squared value error when enabled",
                "entropy": "entropy_coef * Gaussian entropy; configured coefficient is recorded",
            },
        },
        "worked_control_step": {
            "source": "first shared physical state from the P04-09 parity capture",
            "raw_observation": raw_observation or "missing",
            "normalized_observation": normalized.tolist() if normalized is not None else "missing",
            "layer_outputs": actor["layers"],
            "action": capture.get("actor_action", "missing")
            if isinstance(capture, dict)
            else "missing",
            "joint_target_rad": capture.get("joint_target_rad", "missing")
            if isinstance(capture, dict)
            else "missing",
            "physics": "four 0.005 s MuJoCo steps under built-in position actuators",
            "next_observation": "available in full trace, not duplicated into this compact object",
        },
        "worked_rollout_minibatch": {
            "status": "missing",
            "reason": "the retained checkpoint does not serialize rollout tensors or terminal observations",
            "no_values_invented": True,
        },
        "reference_and_mdp": mdp,
        "validation": {
            "actor_dimension": len(
                _scalar_layout(contract["actor_fields"], used_by_inference=True)
            ),
            "critic_dimension": len(
                _scalar_layout(contract["critic_fields"], used_by_inference=False)
            ),
            "action_dimension": len(actions),
            "parity_summary_sha256": sha256_file(parity_summary),
            "parity_passed": parity.get("passed") is True,
        },
    }
    if (
        specification["validation"]["actor_dimension"] != 102
        or specification["validation"]["critic_dimension"] != 114
        or len(actions) != 29
    ):
        raise ValueError("generated walking policy dimensions do not match versioned interface")
    specification = _json_safe(specification)
    encoded_specification = json.dumps(specification, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    (output / "policy-spec.json").write_text(encoded_specification, encoding="utf-8")
    rows = "\n".join(
        f"| {item['index']} | {item['field']}[{item['field_index']}] | {item['unit']} | {item['frame']} |"
        for item in specification["actor_inputs"]
    )
    markdown = f"""# Walking-v1 generated policy specification

Status: **unqualified development**. Checkpoint: `{specification["checkpoint"]["sha256"]}`.

Actor: 102 → 512 → 256 → 128 → 29 with ELU. Critic: 114 → 512 → 256 → 128 → 1.
Actor parameters: {actor["parameter_count"]:,}. Critic parameters: {critic["parameter_count"]:,}.

## Actor input vector

| Index | Value | Unit | Frame/source |
| ---: | --- | --- | --- |
{rows}

## Action and PD equation

`q_target = q_nominal + action_scale × clipped_action − encoder_bias`.

The 29 output mappings, normalizer tensors, layer tensors derived shapes, PPO settings, equations,
reference lineage, and worked captured control step are available in `policy-spec.json`. The missing
worked rollout minibatch is explicitly labeled; no optimizer or rollout history was reconstructed.
"""
    (output / "walking-policy-spec.md").write_text(markdown, encoding="utf-8")
    escaped = html.escape(encoded_specification)
    (output / "walking-policy-spec.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        "<title>Walking policy specification</title><style>body{max-width:1000px;margin:auto;padding:2rem;"
        "font:15px/1.5 system-ui;background:#f7f7f2;color:#111}pre{white-space:pre-wrap;background:white;"
        "padding:1rem;border:1px solid #ddd}code{font-family:ui-monospace,monospace}</style></head>"
        f"<body><h1>Walking-v1 policy specification</h1><p>Unqualified development artifact.</p><pre>{escaped}</pre></body></html>",
        encoding="utf-8",
    )
    return cast(dict[str, Any], specification)
