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


def select_checkpoint(run: Path) -> dict[str, Any]:
    evaluations = []
    for path in sorted((run / "evaluation").glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("schema_version") == 2:
            evaluations.append((path, summary))
    if not evaluations:
        raise ValueError("no version-2 development evaluations found")
    selected_path, selected = max(evaluations, key=lambda item: score(item[1]))
    result = {
        "schema_version": 1,
        "rule": (
            "strict passes, survival passes, lower median drift, lower median tilt, later iteration"
        ),
        "selected_checkpoint": selected["checkpoint"],
        "selected_score": score(selected),
        "evaluations": [
            {
                "path": str(path.relative_to(run)),
                "checkpoint": summary["checkpoint"],
                "score": score(summary),
            }
            for path, summary in evaluations
        ],
        "phase": "development",
        "not_final_test": True,
    }
    shutil.copy2(selected_path, run / "evaluation" / "summary.json")
    (run / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    index_path = run / "checkpoints" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["best_development"] = selected["checkpoint"]
    index["selection_rule"] = result["rule"]
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return result
