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
from .reference_feasibility import (
    ReferenceAuditCriteria,
    ReferenceSpeedEntry,
    ReferenceSpeedMap,
    audit_reference_speed_map,
    load_reference_speed_map,
)

__all__ = [
    "G1_JOINT_NAMES",
    "MotionAudit",
    "MotionManifest",
    "MotionSeries",
    "MotionValidationError",
    "ReferenceAuditCriteria",
    "ReferenceSpeedEntry",
    "ReferenceSpeedMap",
    "audit_periodic_motion",
    "audit_reference_speed_map",
    "derive_foot_contacts",
    "load_soma_csv",
    "load_reference_speed_map",
    "resample_motion",
]
