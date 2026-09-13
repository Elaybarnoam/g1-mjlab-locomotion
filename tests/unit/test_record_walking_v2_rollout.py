from __future__ import annotations

import ast
from pathlib import Path


def test_fixed_speed_rollout_reasserts_command_inside_every_policy_step() -> None:
    """Command-manager resampling must not turn a fixed-speed trial into a stop trial."""
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse(
        (root / "scripts" / "record_walking_v2_rollout.py").read_text(encoding="utf-8")
    )
    policy_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "policy"
            for child in ast.walk(node)
        )
    ]

    assert len(policy_loops) == 1
    calls = [
        child
        for child in ast.walk(policy_loops[0])
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "set_requested_forward_speed"
    ]
    assert len(calls) == 1
