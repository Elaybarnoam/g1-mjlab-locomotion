"""Dependency-free synchronized SVG plots for walking diagnostics."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .walking_v2 import PhysicsTraceV2, WalkingTraceV2


def _points(values: npt.ArrayLike, x: float, y: float, width: float, height: float) -> str:
    array = np.asarray(values, dtype=float).reshape(-1)
    if len(array) > 600:
        array = array[np.linspace(0, len(array) - 1, 600).astype(int)]
    finite = array[np.isfinite(array)]
    if not len(finite):
        return ""
    low = float(np.min(finite))
    high = float(np.max(finite))
    span = max(high - low, 1e-9)
    return " ".join(
        f"{x + width * index / max(len(array) - 1, 1):.2f},{y + height * (1 - (value - low) / span):.2f}"
        for index, value in enumerate(array)
    )


def render_measurement_svg(trace: WalkingTraceV2, physics: PhysicsTraceV2, output: Path) -> None:
    """Write six synchronized diagnostic panels without plotting dependencies."""
    trace.validate()
    physics.validate()
    arrays = trace.arrays
    dt = trace.metadata.control_dt
    ankle_velocity = np.diff(np.asarray(arrays["ankle_position_w"], dtype=float), axis=0) / dt
    ankle_speed = np.sqrt(np.mean(np.sum(np.square(ankle_velocity[:, :, :2]), axis=2), axis=1))
    if physics.supported:
        numerator = np.sum(
            np.asarray(physics.arrays["tangential_speed_square_numerator"], dtype=float), axis=1
        )
        denominator = np.sum(
            np.asarray(physics.arrays["contact_force_denominator"], dtype=float), axis=1
        )
        physical_speed = np.sqrt(
            np.maximum(
                np.divide(
                    numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
                ),
                0,
            )
        )
    else:
        physical_speed = np.zeros(1)
    target_error = np.asarray(arrays["q_target"], dtype=float) - np.asarray(
        arrays["joint_pos"], dtype=float
    )
    panels = (
        (
            "Command and root speed",
            arrays["requested_command"][:, 0],
            arrays["applied_command"][:, 0],
            arrays["root_lin_vel_w"][:, 0],
        ),
        (
            "Contact and normal force",
            arrays["normal_force_n"][:, 0],
            arrays["normal_force_n"][:, 1],
            arrays["debounced_contact"].mean(axis=1),
        ),
        (
            "Phase and sole height",
            arrays["phase"],
            arrays["sole_position_w"][:, 0, 2],
            arrays["sole_position_w"][:, 1, 2],
        ),
        ("Physical contact slip", physical_speed, ankle_speed, np.zeros(1)),
        (
            "Root displacement",
            arrays["root_position_w"][:, 0],
            arrays["root_position_w"][:, 1],
            arrays["root_position_w"][:, 2],
        ),
        (
            "Target tracking",
            np.sqrt(np.mean(np.square(target_error), axis=1)),
            np.max(np.abs(arrays["actuator_torque"]), axis=1),
            arrays["blend"],
        ),
    )
    width = 1000
    panel_height = 145
    body: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{len(panels) * panel_height}">',
        '<rect width="100%" height="100%" fill="#07111f"/>',
        "<style>text{font-family:system-ui;fill:#dce7f5;font-size:13px}.grid{stroke:#203147;stroke-width:1}.line{fill:none;stroke-width:2}</style>",
    ]
    colors = ("#47d7ac", "#ffb454", "#7aa2f7")
    for index, panel in enumerate(panels):
        title, *series = panel
        top = index * panel_height
        body.append(f'<text x="18" y="{top + 22}">{html.escape(title)}</text>')
        body.append(
            f'<rect class="grid" fill="none" x="15" y="{top + 32}" width="970" height="100"/>'
        )
        for color, values in zip(colors, series, strict=True):
            points = _points(values, 15, top + 32, 970, 100)
            body.append(f'<polyline class="line" stroke="{color}" points="{points}"/>')
    body.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(body) + "\n", encoding="utf-8")
    temporary.replace(output)
