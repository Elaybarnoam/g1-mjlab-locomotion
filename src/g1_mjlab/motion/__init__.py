"""Validated, simulator-independent walking-motion interfaces."""

from .reference import (
    G1_JOINT_NAMES,
    MotionAudit,
    MotionManifest,
    MotionSeries,
    MotionValidationError,
    audit_periodic_motion,
    derive_foot_contacts,
    load_soma_csv,
    resample_motion,
)

__all__ = [
    "G1_JOINT_NAMES",
    "MotionAudit",
    "MotionManifest",
    "MotionSeries",
    "MotionValidationError",
    "audit_periodic_motion",
    "derive_foot_contacts",
    "load_soma_csv",
    "resample_motion",
]
