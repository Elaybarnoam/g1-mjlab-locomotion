from __future__ import annotations

from g1_mjlab.contracts import validate_contract
from g1_mjlab.tasks.walking_v1 import walking_policy_contract


def test_walking_contract_has_exact_actor_critic_and_action_dimensions() -> None:
    contract = walking_policy_contract()

    validate_contract(contract)
    assert sum(field.size for field in contract.actor_fields) == 102
    assert sum(field.size for field in contract.critic_fields) == 114
    assert len(contract.action_names) == 29
    assert [(field.name, field.offset, field.size) for field in contract.actor_fields[-4:]] == [
        ("command", 96, 3),
        ("phase_sin", 99, 1),
        ("phase_cos", 100, 1),
        ("walk_blend", 101, 1),
    ]
    assert contract.task_id == "G1-Walking-Flat-v1"
    assert contract.layout_id == "g1-walking-actor-v1"
    assert "nominal" in contract.action_semantics
    assert "reference" not in contract.action_semantics
