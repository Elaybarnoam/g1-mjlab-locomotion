# Soft-reference policy-first walking

Plan 06 does not change the result of the Plan 05 reference-feasibility gate. Those candidate
animations remain dynamically unqualified. It introduces a second, deliberately narrower concept:
**soft-reference admission** determines whether a kinematic cycle is coherent and safe enough to
serve as an observation/reward target during bounded PPO acquisition.

Admission requires finite, hash-bound arrays; the exact G1 joint contract; joint margin; periodic
seam; normalized root quaternion; correct speed-times-period displacement; no flight gaps; foot-IK
accuracy; and strict collision-geometry limits. The previous constrained-wrench audit must be
attached, but may be labeled `failed`. A failed dynamics diagnostic is never presented as passing.

The learned policy receives the reference pose and velocity and emits residual joint-position
targets around it. The simulator and PD controller determine the actual trajectory. Strict torque,
contact, slip, fall and gait requirements are evaluated on closed-loop policy rollouts before any
candidate can be qualified or released.

`SoftReferenceAdmission` is fail-closed: missing evidence, mismatched hashes or one missing speed
sets `training_authorized` to false. `admitted` means only “permitted to guide bounded learning.” It
does not mean balanced, dynamically feasible, humanlike, deployable or qualified.

The checked-in criteria are in `configs/walking-v2/soft-reference-admission-v2.json`. Native audits
are produced by `scripts/audit_soft_walking_reference.py`; the final hash-bound decision is produced
by `scripts/admit_soft_walking_references.py`.
