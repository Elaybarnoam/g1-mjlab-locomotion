from __future__ import annotations

import pytest

from g1_mjlab.contracts import PolicyContract, fields_from_sizes, validate_contract


def test_fields_have_contiguous_offsets() -> None:
    fields = fields_from_sizes(
        (("gravity", 3, "1", "body", "imu"), ("joints", 2, "rad", "joint", "encoder"))
    )
    assert [field.offset for field in fields] == [0, 3]


def test_contract_rejects_action_for_missing_joint() -> None:
    contract = PolicyContract(
        1, "G1", "rev", ("hip",), (), (), ("knee",), 0.005, 0.02, "position", "PD"
    )
    with pytest.raises(ValueError, match="model joint"):
        validate_contract(contract)
