# Walking PPO lifecycle

Walking-v1 uses RSL-RL 5.5.0 PPO through mjlab. The resolved actor is a normalized 102-input MLP
with hidden widths 512, 256, and 128, ELU activations, and 29 Gaussian position-target outputs. The
critic is a separate normalized 114-input MLP with the same hidden widths and one value output.
Deterministic evaluation uses the actor mean; stochastic actions are used while collecting training
rollouts.

The first smoke configuration is [`configs/walking-v1/smoke.json`](../configs/walking-v1/smoke.json):
256 worlds times 24 steps gives 6,144 transitions per update. PPO uses four minibatches, five epochs,
gamma 0.99, GAE lambda 0.95, clip 0.2, Adam at an initial 0.001 learning rate with adaptive KL target
0.01, value coefficient 1.0, entropy coefficient 0.01, and gradient-norm cap 1.0. These are baseline
settings, not yet tuned walking results.

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
