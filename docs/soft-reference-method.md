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

## Collision repair

`scripts/repair_walking_reference_collisions.py` implements a bounded preprocessing step. It applies
the minimum uniform root lift needed to clear the model's actual foot collision hulls, then searches
in 0.01 increments for the smallest blend of retargeted arm joints toward the model nominal pose
that removes hand/wrist collisions. It reserves a 0.023 rad source-joint margin so cubic speed
interpolation remains above the 0.019 rad admission margin. The script regenerates FK-dependent
arrays and records old/new hashes, corrections, collision measurements and added foot-path error.

Repair is not dynamics qualification. Its output must pass a fresh kinematic audit and retain a
fresh failed/passed dynamics diagnostic. The reference bank builder accepts only exact hashes from
an explicitly training-authorized decision and emits a separate admission-to-bank binding manifest.

## Interval-end target state

`WalkingTargetState` owns only the applied command, periodic phase and actual stand/walk blend. Its
pure transition computes the next rate-limited command, the smoothstep desired blend, the next
rate-limited actual blend and phase, then composes the exact `k+1` joint-position and joint-velocity
targets. The returned transition is the cache shared by action centering and post-step rewards; its
`next_state` is committed only after that interval. This prevents observations, actions and rewards
from silently using different reference times. NumPy and Torch adapters implement the same contract,
and indexed resets return copies while leaving every unselected environment unchanged.
