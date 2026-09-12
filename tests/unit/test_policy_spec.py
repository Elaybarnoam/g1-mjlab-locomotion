from __future__ import annotations

from g1_mjlab.reporting.policy_spec import _scalar_layout


def test_scalar_layout_expands_fields_to_exact_indices_and_marks_critic_private() -> None:
    fields = [
        {
            "name": "base_lin_vel",
            "offset": 0,
            "size": 3,
            "resolved_term": {"noise": None, "scale": None, "clip": None},
        },
        {
            "name": "walk_blend",
            "offset": 3,
            "size": 1,
            "resolved_term": {"history_length": 0},
        },
    ]

    result = _scalar_layout(fields, used_by_inference=False)

    assert [item["index"] for item in result] == [0, 1, 2, 3]
    assert result[0]["unit"] == "m/s"
    assert result[-1]["field"] == "walk_blend"
    assert all(item["training"] and not item["inference"] for item in result)
