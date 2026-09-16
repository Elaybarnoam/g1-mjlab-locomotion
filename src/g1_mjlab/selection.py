"""Deterministic development-only checkpoint ranking."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from statistics import median
from typing import Any

from .checkpoints import checkpoint_iteration


def score(summary: dict[str, Any]) -> tuple[int, int, float, float, int]:
    """Strict passes, survival, lower drift/tilt, then later iteration."""
    trials = summary["trials"]
    if not trials or summary.get("phase") != "development":
        raise ValueError("checkpoint selection requires development trials")
    checkpoint = Path(summary["checkpoint"])
    return (
        int(summary["passed"]),
        int(summary["survival_passed"]),
        -median(float(item["max_drift_m"]) for item in trials),
        -median(float(item["max_torso_tilt_deg"]) for item in trials),
        checkpoint_iteration(checkpoint),
    )


def score_walking(summary: dict[str, Any]) -> tuple[int, int, float, float, float, float, int]:
    """Required gates, command tracking, declared gait quality, then iteration."""
    trials = summary["trials"]
    if not trials or summary.get("phase") != "development":
        raise ValueError("walking checkpoint selection requires development trials")

    def values(name: str, *, missing: float) -> list[float]:
        return [float(item[name]) if item.get(name) is not None else missing for item in trials]

    return (
        int(summary["passed_both"]),
        sum(bool(item["functional_passed"]) for item in trials),
        -median(values("command_rms_m_s", missing=1e9)),
        median(values("alternation_ratio", missing=0.0)),
        -median(values("tiny_step_fraction", missing=1.0)),
        -median(values("stance_slip_rms_m_s", missing=1e9)),
        checkpoint_iteration(Path(summary["checkpoint"])),
    )


def select_checkpoint(run: Path) -> dict[str, Any]:
    all_summaries = []
    for path in sorted((run / "evaluation").glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        all_summaries.append((path, summary))
    walking = any(summary.get("task_id") == "G1-Walking-Flat-v1" for _, summary in all_summaries)
    score_fn = score_walking if walking else score
    evaluations = [
        (path, summary)
        for path, summary in all_summaries
        if (walking and summary.get("task_id") == "G1-Walking-Flat-v1")
        or (not walking and summary.get("schema_version") == 2)
    ]
    if not evaluations:
        raise ValueError("no compatible development evaluations found")
    selected_path, selected = max(evaluations, key=lambda item: score_fn(item[1]))
    rule = (
        "combined functional/style gates, functional gates, command error, gait quality, later iteration"
        if walking
        else "strict passes, survival passes, lower median drift, lower median tilt, later iteration"
    )
    result = {
        "schema_version": 1,
        "rule": rule,
        "selected_checkpoint": selected["checkpoint"],
        "selected_score": score_fn(selected),
        "evaluations": [
            {
                "path": str(path.relative_to(run)),
                "checkpoint": summary["checkpoint"],
                "score": score_fn(summary),
            }
            for path, summary in evaluations
        ],
        "phase": "development",
        "not_final_test": True,
        "qualified": (
            int(selected.get("passed_both", -1)) == len(selected["trials"])
            if walking
            else int(selected.get("passed", -1)) == len(selected["trials"])
        ),
    }
    shutil.copy2(selected_path, run / "evaluation" / "summary.json")
    (run / "selection.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    index_path = run / "checkpoints" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["best_development"] = selected["checkpoint"]
    index["selection_rule"] = result["rule"]
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return result
