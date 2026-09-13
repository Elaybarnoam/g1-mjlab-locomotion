"""Build a bank from three hash-bound, training-authorized references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from g1_mjlab.artifacts import sha256_file, write_atomic_json
from g1_mjlab.motion.reference_bank import build_reference_bank
from g1_mjlab.motion.soft_reference_admission import authorized_reference_hashes


def _rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w = quaternion[:, :1]
    xyz = quaternion[:, 1:]
    return np.asarray(
        vector + 2 * np.cross(xyz, np.cross(xyz, vector) - w * vector),
        dtype=np.float64,
    )


def local_foot_positions(reference_path: Path, model_path: Path) -> np.ndarray:
    import mujoco

    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    with np.load(reference_path, allow_pickle=False) as reference:
        joint_position = np.asarray(reference["joint_pos"], dtype=np.float64)
        body_position = np.asarray(reference["body_pos_w"], dtype=np.float64)
        body_quaternion = np.asarray(reference["body_quat_w"], dtype=np.float64)
        joint_names = tuple(str(name) for name in reference["joint_names"].tolist())
        body_names = tuple(str(name) for name in reference["body_names"].tolist())
    pelvis = body_names.index("pelvis")

    def model_id(kind: Any, name: str) -> int:
        result = int(mujoco.mj_name2id(model, kind, name))
        return result if result >= 0 else int(mujoco.mj_name2id(model, kind, f"robot/{name}"))

    joint_ids = [model_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    qpos_addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    site_ids = [model_id(mujoco.mjtObj.mjOBJ_SITE, name) for name in ("left_foot", "right_foot")]
    if any(value < 0 for value in (*joint_ids, *site_ids)):
        raise ValueError("model does not satisfy reference joint/site contract")
    world = np.empty((len(joint_position), 2, 3), dtype=np.float64)
    for frame in range(len(joint_position)):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = body_position[frame, pelvis]
        data.qpos[3:7] = body_quaternion[frame, pelvis]
        data.qpos[qpos_addresses] = joint_position[frame]
        mujoco.mj_forward(model, data)
        world[frame] = data.site_xpos[site_ids]
    relative = world - body_position[:, pelvis, None, :]
    quaternion = np.repeat(body_quaternion[:, pelvis], 2, axis=0)
    return _rotate_inverse(quaternion, relative.reshape(-1, 3)).reshape(world.shape)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        required=True,
        action="append",
        metavar="SPEED=PATH",
    )
    parser.add_argument("--qualification", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--motion-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    motion = json.loads(args.motion_manifest.read_text(encoding="utf-8"))
    qualification = json.loads(args.qualification.read_text(encoding="utf-8"))
    references: dict[float, Path] = {}
    for value in args.reference:
        speed_text, separator, path_text = value.partition("=")
        if not separator:
            raise SystemExit("--reference must use SPEED=PATH")
        references[float(speed_text)] = Path(path_text)
    if set(references) != {0.4, 0.6, 0.8}:
        raise SystemExit("exactly one reference is required for 0.4, 0.6 and 0.8 m/s")
    selected_paths = (references[0.4], references[0.6], references[0.8])
    selected_hashes = tuple(sha256_file(path) for path in selected_paths)
    try:
        expected_hashes = authorized_reference_hashes(qualification, (0.4, 0.6, 0.8))
    except ValueError as error:
        raise SystemExit(f"{error}; refusing to build bank") from error
    if selected_hashes != expected_hashes:
        raise SystemExit("selected references do not match the admitted candidate hashes")
    metadata = build_reference_bank(
        selected_paths,
        args.output,
        reference_id="g1-natural-walking-bank-v1",
        source_license=motion["source"]["license"],
        model_sha256=sha256_file(args.model),
        controller_sha256=sha256_file(args.model),
        joint_names=tuple(contract["joint_names"]),
        nominal_joint_position=np.asarray(contract["nominal_joint_position"], dtype=np.float64),
        local_foot_position=(
            local_foot_positions(selected_paths[0], args.model),
            local_foot_positions(selected_paths[1], args.model),
            local_foot_positions(selected_paths[2], args.model),
        ),
    )
    binding_path = args.output.with_name(f"{args.output.stem}-admission.json")
    write_atomic_json(
        binding_path,
        {
            "schema_version": 1,
            "semantics": "soft-reference-admission-bound-bank",
            "reference_bank_metadata": args.output.name,
            "reference_bank_metadata_sha256": sha256_file(args.output),
            "reference_bank_npz_sha256": metadata.npz_sha256,
            "admission_decision_sha256": sha256_file(args.qualification),
            "source_reference_sha256_by_speed": list(expected_hashes),
            "training_authorized": True,
            "policy_qualification_required": True,
        },
    )
    print(
        json.dumps(
            {
                "metadata": str(args.output),
                "binding": str(binding_path),
                "npz_sha256": metadata.npz_sha256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
