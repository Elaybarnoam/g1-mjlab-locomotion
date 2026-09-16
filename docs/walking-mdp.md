# Walking-v1 MDP and incentive design

Walking-v1 now resolves to a real mjlab task, but it has not yet been trained. The environment uses
the pinned G1 flat model at 200 Hz physics and 50 Hz policy control. Each episode samples either the
audited 1.164 m/s reference command or standing; moving training resets can start from a randomized
reference phase with the complete floating-base and joint velocity state.

The MDP contains only explicit walking terms. Standing-v1's world-position/base-motion penalty is
absent because forward travel is the task. The exact initial equations, units, masks, and weights
are machine-readable in [`configs/walking-v1/rewards.json`](../configs/walking-v1/rewards.json).
Reward-manager values are rates multiplied by the 0.02 s control interval. The `-10` termination
value is converted to a manager weight of `-10 / 0.02` so it is applied once rather than weakened by
the timestep.

Positive terms cover applied-command velocity, zero angular velocity/upright torso, reference joint
pose and velocity, root-relative foot placement, and reference contact timing. Costs cover stance
slip, crouching, soft joint-limit violations, action changes, actuator effort, and self-collision.
Walking-only style terms are multiplied by the stand/walk blend. At blend zero, joint targets return
to the model's standing pose and reference joint velocity returns to zero.

Episodes terminate separately for excessive tilt, pelvis height below 0.50 m, any non-finite root or
joint state, and ground contact by robot bodies other than the two foot-bearing ankle links. This
prevents knee-supported crawling from earning velocity/reference reward. Time-limit truncation
remains distinct from true termination for PPO bootstrapping.

The reference command owns one GPU-resident phase state per robot and loads the hash-verified NPZ
with `allow_pickle=False`. Simulator joint order must exactly match the 29 names embedded in the
reference. Reference interpolation, reward evaluation, command ramps, and phase updates are batched
Torch operations; there is no per-robot Python loop in the control path.

Pure behavior probes classify ideal following, standing at a nonzero command, shuffling,
stance-slip, crouching, falling, and forbidden support contacts. A four-world GPU integration test
also checks exact 102/114 observation shapes, finite rewards/observations, short-rollout stability,
and equality between the reward returned by the environment and the sum of reported term
contributions. These tests validate wiring and incentive direction. They are not evidence that PPO
has learned a humanlike gait.
