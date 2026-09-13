"""Verify walking-v2 target-state semantics against an immutable reference bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from g1_mjlab.artifacts import sha256_file, write_atomic_json
from g1_mjlab.motion.reference_bank import ReferenceBank
from g1_mjlab.motion.walking_target_state import (
    WalkingTargetProfile,
    WalkingTargetState,
    predict_walking_target_numpy,
    predict_walking_target_torch,
    reset_walking_target_numpy,
    reset_walking_target_torch,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    profile = WalkingTargetProfile(
        policy_dt_s=float(raw["policy_dt_s"]),
        maximum_forward_speed_m_s=float(raw["maximum_forward_speed_m_s"]),
        full_walk_blend_speed_m_s=float(raw["full_walk_blend_speed_m_s"]),
        acceleration_m_s2=float(raw["acceleration_m_s2"]),
        deceleration_m_s2=float(raw["deceleration_m_s2"]),
        blend_rate_s=float(raw["blend_rate_s"]),
    )
    profile.validate()
    bank = ReferenceBank.load(args.bank)
    requested = np.asarray(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0], [0.8, 0.0, 0.0]],
        dtype=np.float64,
    )
    numpy_state = WalkingTargetState.zeros(4)
    torch_command = torch.zeros((4, 3), dtype=torch.float64)
    torch_phase = torch.zeros(4, dtype=torch.float64)
    torch_blend = torch.zeros(4, dtype=torch.float64)
    torch_requested = torch.tensor(requested, dtype=torch.float64)
    maximum_error = 0.0
    finite = True
    for _ in range(100):
        numpy_transition = predict_walking_target_numpy(
            numpy_state, requested, bank, profile
        )
        torch_transition = predict_walking_target_torch(
            torch_command,
            torch_phase,
            torch_blend,
            torch_requested,
            bank,
            profile,
        )
        comparisons = (
            (numpy_transition.next_state.applied_command, torch_transition.next_applied_command),
            (numpy_transition.next_state.phase, torch_transition.next_phase),
            (numpy_transition.next_state.blend, torch_transition.next_blend),
            (numpy_transition.targets.joint_position, torch_transition.target_joint_position),
            (numpy_transition.targets.joint_velocity, torch_transition.target_joint_velocity),
        )
        for numpy_value, torch_value in comparisons:
            converted = torch_value.detach().cpu().numpy()
            maximum_error = max(
                maximum_error, float(np.max(np.abs(numpy_value - converted)))
            )
            finite = finite and bool(np.isfinite(numpy_value).all())
        numpy_state = numpy_transition.next_state
        torch_command = torch_transition.next_applied_command
        torch_phase = torch_transition.next_phase
        torch_blend = torch_transition.next_blend
    before_reset = WalkingTargetState(
        numpy_state.applied_command.copy(),
        numpy_state.phase.copy(),
        numpy_state.blend.copy(),
    )
    numpy_reset = reset_walking_target_numpy(numpy_state, [1, 3], phase=[0.25, 0.75])
    torch_reset = reset_walking_target_torch(
        torch_command,
        torch_phase,
        torch_blend,
        torch.tensor([1, 3]),
        reset_phase=torch.tensor([0.25, 0.75], dtype=torch.float64),
    )
    unchanged = np.array_equal(
        numpy_reset.applied_command[[0, 2]], before_reset.applied_command[[0, 2]]
    ) and np.array_equal(numpy_reset.phase[[0, 2]], before_reset.phase[[0, 2]])
    reset_error = max(
        float(np.max(np.abs(numpy_reset.applied_command - torch_reset[0].numpy()))),
        float(np.max(np.abs(numpy_reset.phase - torch_reset[1].numpy()))),
        float(np.max(np.abs(numpy_reset.blend - torch_reset[2].numpy()))),
    )
    passed = finite and unchanged and maximum_error <= 1e-10 and reset_error <= 1e-12
    result = {
        "schema_version": 2,
        "semantics": "walking-v2-target-state-verification",
        "bank_metadata_sha256": sha256_file(args.bank),
        "bank_npz_sha256": bank.metadata.npz_sha256,
        "config_sha256": sha256_file(args.config),
        "steps": 100,
        "environment_count": 4,
        "requested_forward_speeds_m_s": requested[:, 0].tolist(),
        "maximum_numpy_torch_error": maximum_error,
        "partial_reset_numpy_torch_error": reset_error,
        "partial_reset_unselected_unchanged": unchanged,
        "all_targets_finite": finite,
        "passed": passed,
    }
    write_atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
