from dataclasses import dataclass

from g1_mjlab.qualification import canonical_hash, describe


def test_hash_covers_gains_limits_and_scales() -> None:
    contract = {"gain": [40.0], "scale": [0.25], "limit": [88.0]}
    original = canonical_hash(contract)
    assert canonical_hash({**contract, "sha256": original}) == original
    for name in contract:
        assert canonical_hash({**contract, name: [0.0]}) != original


def test_callable_config_serialization() -> None:
    @dataclass
    class Config:
        scale: float = 0.2

    assert describe(Config()) == {"scale": 0.2}
    assert describe(canonical_hash) == "g1_mjlab.qualification.canonical_hash"
