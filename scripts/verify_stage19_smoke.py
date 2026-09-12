"""Verify a trusted local Stage 19 two-update smoke artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from g1_mjlab.artifacts import sha256_file


def _load(path: Path) -> dict[str, Any]:
    value = torch.load(path, weights_only=False, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return value


def _changed_tensors(before: dict[str, Any], after: dict[str, Any], key: str) -> int:
    before_state = before[key]
    after_state = after[key]
    if before_state.keys() != after_state.keys():
        raise ValueError(f"{key} layout changed")
    return sum(
        isinstance(before_state[name], torch.Tensor)
        and not torch.equal(before_state[name], after_state[name])
        for name in before_state
    )


def verify(source: Path, run: Path, resume_run: Path | None = None) -> dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    target = run / "checkpoints" / manifest["final_checkpoint"]
    before = _load(source)
    after = _load(target)
    required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict"}
    if not required.issubset(after):
        raise ValueError("final checkpoint omits learner state")
    actor_changed = _changed_tensors(before, after, "actor_state_dict")
    critic_changed = _changed_tensors(before, after, "critic_state_dict")
    if actor_changed == 0 or critic_changed == 0:
        raise ValueError("two-update smoke did not change actor and critic parameters")
    summary = json.loads((run / "learning-summary.json").read_text(encoding="utf-8"))
    memory = json.loads((run / "memory.json").read_text(encoding="utf-8"))
    contract = json.loads((run / "contract.json").read_text(encoding="utf-8"))
    transfer = json.loads((run / "fine-tune.json").read_text(encoding="utf-8"))
    optimization = json.loads((run / "optimization.json").read_text(encoding="utf-8"))
    actor_size = sum(field["size"] for field in contract["actor_fields"])
    critic_size = sum(field["size"] for field in contract["critic_fields"])
    action_size = len(contract["action_names"])
    numeric_memory = [value for value in memory.values() if isinstance(value, (int, float))]
    checks = {
        "status_completed": manifest["status"] == "completed",
        "checkpoint_hash_matches": sha256_file(target) == manifest["final_checkpoint_sha256"],
        "two_updates": summary["update_indices"] == [0, 1],
        "losses_finite": summary["all_losses_finite"],
        "memory_finite": all(math.isfinite(value) for value in numeric_memory),
        "actor_size_102": actor_size == 102,
        "critic_size_114": critic_size == 114,
        "action_size_29": action_size == 29,
        "full_learner_state_present": required.issubset(after),
        "actor_parameters_changed": actor_changed > 0,
        "critic_parameters_changed": critic_changed > 0,
        "source_hash_matches": transfer["sha256"] == sha256_file(source),
        "full_state_transfer_declared": set(transfer["restored_components"])
        == {"actor", "critic", "optimizer"},
        "optimization_finite": optimization["all_recorded_gradients_finite"]
        and all(
            component["all_parameters_finite"] for component in optimization["components"].values()
        ),
    }
    if resume_run is not None:
        resume = json.loads((resume_run / "resume.json").read_text(encoding="utf-8"))
        resume_manifest = json.loads((resume_run / "manifest.json").read_text(encoding="utf-8"))
        resume_summary = json.loads(
            (resume_run / "learning-summary.json").read_text(encoding="utf-8")
        )
        restored_iteration = int(target.stem.rsplit("_", 1)[1])
        expected_updates = list(
            range(restored_iteration + 1, restored_iteration + 1 + resume["additional_updates"])
        )
        checks.update(
            {
                "resume_status_completed": resume_manifest["status"] == "completed",
                "resume_source_hash_matches": resume["sha256"] == sha256_file(target),
                "resume_full_state_verified": set(resume["verified_restored_components"])
                == required,
                "resume_iteration_is_contiguous": resume["restored_iteration"] == restored_iteration
                and resume["first_new_iteration"] == restored_iteration + 1
                and resume_summary["update_indices"] == expected_updates,
                "resume_losses_finite": resume_summary["all_losses_finite"],
            }
        )
    return {
        "schema_version": 1,
        "passed": all(checks.values()),
        "checks": checks,
        "actor_tensors_changed": actor_changed,
        "critic_tensors_changed": critic_changed,
        "checkpoint": str(target),
        "checkpoint_sha256": sha256_file(target),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.source, args.run, args.resume_run)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
