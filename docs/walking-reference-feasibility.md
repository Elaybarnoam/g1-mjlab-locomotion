# Walking-v2 reference feasibility

Phase P05-04 is intentionally fail-closed. A walking reference must pass kinematic and
constrained floating-base inverse-dynamics checks at each deployment speed before the immutable
reference bank, gait acceptance targets, closed-loop probes, or PPO training can be created.

The checked-in contract is `configs/walking-v2/reference-feasibility-v1.json`. The audit uses the
pinned G1 model and maps actuator limits by transmission joint identity. It solves two six-axis
foot wrenches under unilateral normal force, a linear friction pyramid, rectangular support
polygons, and torsional bounds. It reports floating-base residuals and gravity, inertia,
velocity/passive, contact, and per-joint torque ratios separately.

## Current decision

No walking-v2 reference bank is qualified. The original SOMA A057 cycle failed the constrained
dynamics gate after a corrected joint-to-actuator limit mapping. Adaptation attempt 1 generated
speed-specific, periodic cycles using contact-preserving foot-path scaling, bounded pelvis
correction, G1 leg IK, and arm acceleration smoothing. All three passed its 1 cm IK gate, but only
76.6–78.1% of frames were dynamically feasible (required: 95%); normalized base residual p95 was
0.284–0.407 (maximum: 0.05), and peak torque ratios were 1.54–1.94 (maximum: 1.0).

Attempt 2 added stronger smoothing, temporal pelvis regularization, and foot-derived symmetric
support labels. It failed IK at 0.4 and 0.6 m/s with 3.95 cm and 3.53 cm peak foot errors. Its
0.8 m/s cycle passed IK but failed dynamics: 68.75% feasible, base-residual p95 0.429, torque p95
0.857, and peak ratio 1.665. The two-attempt budget is exhausted. Thresholds were not changed.

Consequently, reference-only native/Warp PD probes, gait-target freezing, task-v2 integration,
and PPO training are blocked. Running those stages would violate the plan because their targets
would come from a rejected reference.

The next valid input is either a licensed speed-specific G1 motion set with physically consistent
root, foot-contact, and joint trajectories, or a newly measured/versioned controller contract
that changes effective torque limits. Either change starts a new P05-04 reference version and
reruns the same gates. A conservative diagnostic may be replaced only after a documented physics
model correction—not to make a candidate pass.

## Reproduction

Use `scripts/adapt_walking_reference.py` for one declared adaptation and
`scripts/audit_walking_reference_dynamics.py` for its native-speed dynamics audit. Then run
`scripts/qualify_walking_references.py` with every candidate report. The command exits with code 2
and writes `status: blocked` unless one candidate passes both checks at every required speed.
Large motion/evidence NPZ and JSON files belong under `.runtime/`; they are intentionally excluded
from Git.
