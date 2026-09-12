# Stage 19 controlled experiment matrix

Stage 19 compares three PPO fine-tuning hypotheses from the same bootstrap checkpoint. It is a
development experiment, not release qualification. Arm A is the audited common baseline, Arm B
adds contact timing and event-duration terms, and Arm C adds physical stance-slip shaping. Every
arm restores the actor, critic, observation normalizers, optimizer, learned action standard
deviation, and effective learning rate. No arm inherits another arm's result.

The frozen development suite contains 16 explicit generalized states: four nominal starts, four
yaw offsets, four bounded joint-position perturbations, and four bounded horizontal root-velocity
perturbations. Each stores all 36 qpos values, all 35 qvel values, and phase 0, .25, .5, or .75.
The simulator-side freeze command verifies finite forward kinematics, no initial termination, and
no self-collision before publishing the file.

```bash
g1-mjlab freeze-walking-scenarios \
  --config configs/walking-v1/stage19/arm-c-train.json \
  --output configs/walking-v1/stage19/development-scenarios-v3.json \
  --seed 10042
```

Campaign schema 2 references the immutable source evaluation and decision rule by SHA-256. Each
100-update segment is evaluated on all 16 states. An arm stops after two consecutive evaluations
with at most 12 functional passes, or after three evaluations without a qualifying improvement.
An extension requires 16/16 function, at least two physical error improvements of 15% or more, no
error worsening over 10%, command-RMS preservation, and raw-chatter preservation. Reward is never
a selection key.

The September 12, 2026 matrix outcome is a failed hypothesis, not a walking-policy success claim.
Arm A stopped after 200 updates; Arms B and C completed 300. B and C reached 16/16 functional
passes but 0/16 combined function-and-style passes. Neither qualified for another 300 updates, and
C did not improve physical slip over B enough to authorize conditional Arm D. The best-observed
development checkpoint is retained for diagnosis only. The evidence requires a method decision
covering reference feasibility, phase dependence, observation adequacy, and reward scale before
more PPO reward tuning or any AMP proposal.

Raw campaigns, traces, and the machine-readable `experiment-table.json` live under ignored
`.runtime` directories. Verify them with:

```bash
python scripts/verify_stage19_experiments.py \
  --source-evaluation .runtime/plan04-p07-source-eval-001/summary.json \
  --scenarios configs/walking-v1/stage19/development-scenarios-v3.json \
  --scenario-verification configs/walking-v1/stage19/development-scenarios-v3.verification.json \
  --experiment-table .runtime/plan04-p07-experiment-table-001/experiment-table.json \
  --arm-a .runtime/plan04-p07-arm-a-001 \
  --arm-b .runtime/plan04-p07-arm-b-001 \
  --arm-c .runtime/plan04-p07-arm-c-001 \
  --output .runtime/plan04-p07-experiment-table-001/verification.json
```
