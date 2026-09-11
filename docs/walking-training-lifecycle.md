# Walking PPO lifecycle

Walking-v1 uses RSL-RL 5.5.0 PPO through mjlab. The resolved actor is a normalized 102-input MLP
with hidden widths 512, 256, and 128, ELU activations, and 29 Gaussian position-target outputs. The
critic is a separate normalized 114-input MLP with the same hidden widths and one value output.
Deterministic evaluation uses the actor mean; stochastic actions are used while collecting training
rollouts.

The original reference-guided smoke configuration is
[`configs/walking-v1/smoke.json`](../configs/walking-v1/smoke.json):
256 worlds times 24 steps gives 6,144 transitions per update. PPO uses four minibatches, five epochs,
gamma 0.99, GAE lambda 0.95, clip 0.2, Adam at an initial 0.001 learning rate with adaptive KL target
0.01, value coefficient 1.0, entropy coefficient 0.01, and gradient-norm cap 1.0. These are baseline
settings, not yet tuned walking results.

## Locomotion bootstrap

The active replacement campaign uses
[`bootstrap-train.json`](../configs/walking-v1/bootstrap-train.json) with
[`bootstrap-profile.json`](../configs/walking-v1/bootstrap-profile.json). It trains the same
102→512→256→128→29 actor and nominal-centered action contract, but sets all reference imitation
weights to zero. Its locomotion objective reproduces the pinned mjlab G1 flat-task weights for
linear/angular command tracking, upright posture, command-dependent pose, body angular velocity,
angular momentum, joint limits, action rate, foot clearance, swing height, slip, soft landing, and
self-collision. Ordinary reset, push, friction, encoder-bias, and center-of-mass randomization remain
enabled during training and are disabled in deterministic evaluation.

Bootstrap terminates on timeout, a 70-degree fall, or non-finite simulator state. Minimum pelvis
height and non-foot ground contacts remain measured qualification failures but do not truncate early
learning rollouts; applying those final gates during the first attempt collapsed the mean episode to
roughly 14 control steps by update 534 and prevented useful credit assignment.

The declared 6,000-update run contains 36,864,000 transitions at 256 environments × 24 steps. It is
a bounded campaign, not evidence of success. Checkpoints are evaluated on the 0→0.6→0 m/s schedule;
only a functional locomotion checkpoint may become the source of reference-style fine-tuning.

## Verified smoke result

The two-update smoke completed 12,288 transitions. Every recorded value, surrogate, and entropy
loss was finite; recorded gradients and parameters were finite. Learnable actor parameters changed
by L2 0.714 and critic parameters by L2 0.712. A checkpoint and ONNX export were produced. The
reviewed identities are in [`evidence/walking-v1/ppo-smoke.json`](../evidence/walking-v1/ppo-smoke.json).

This verifies the data and optimization path only. It does **not** show a trained walking policy.
At two updates, reward and termination telemetry remain poor. W07 must define behavior metrics
before the 1,000-update pilot is inspected.

## Continuation modes

`--resume CHECKPOINT` continues the same task with actor, critic, optimizer, iteration, and saved
learner state. Configuration, reward profile, PPO profile, and resolved MDP are checked. Worlds and
random number state are fresh, so this is learner continuation rather than bit-exact trajectory
replay.

`--initialize-actor CHECKPOINT` creates a new experiment and copies only the actor and its observation
normalizer. The critic, optimizer, iteration counter, and environment state remain fresh. The two
flags are mutually exclusive. W06 permits only same-task initialization; standing-to-walking
transfer requires the explicit 99-to-102 observation remap planned as a later ablation.

`--fine-tune CHECKPOINT` starts a new, explicitly labeled run while restoring actor, critic,
optimizer, observation normalizers, action distribution, and learned learning rate. Iteration and
environment state restart. This is the intended bootstrap-to-style transition because actor-only
transfer caused immediate catastrophic forgetting in the W09 experiments.
