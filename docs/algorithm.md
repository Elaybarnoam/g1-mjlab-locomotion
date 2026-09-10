# PPO, GAE, and the policy

Standing-v1 uses RSL-RL 5.5.0 Proximal Policy Optimization with generalized advantage estimation.
The project integrates the upstream implementation and tests the environment/learner seam instead
of reimplementing PPO.

## Collection and update

Each update collects 24 control steps from 256 independent worlds: 6,144 transitions. Five epochs
reuse the rollout through four shuffled minibatches. One 1,000-update seed therefore collects
6,144,000 new transitions. Episodes can span rollout boundaries; the rollout length is not the
episode horizon.

Reference temporal-difference residual and GAE recursion:

```text
delta_t = r_t + gamma * b_t * V(s_t+1) - V(s_t)
A_t     = delta_t + gamma * lambda * c_t * A_t+1
```

`b_t` disables bootstrap on true termination. `c_t` stops recursion at any episode boundary. The
pinned RSL-RL implementation compensates timeouts by adding gamma times its stored transition value
to reward before done-masked GAE; it is not described as textbook final-observation bootstrap.

The clipped surrogate uses the new/old policy probability ratio `rho`, independently from actuator
action clipping:

```text
L_clip = E[min(rho_t A_t, clip(rho_t, 1-epsilon, 1+epsilon) A_t)]
```

Standing-v1 selected `action_clip = null`, matching the upstream Gaussian-action interface. PPO
ratio clipping remains `epsilon = 0.2`.

## Frozen configuration

| Setting | Value |
| --- | ---: |
| gamma | 0.99 |
| GAE lambda | 0.95 |
| PPO clip | 0.2 |
| learning epochs | 5 |
| minibatches | 4 |
| initial learning rate | 0.001 |
| adaptive KL target | 0.01 |
| entropy coefficient | 0.01 |
| value coefficient | 1.0 |
| gradient norm cap | 1.0 |

Actor and critic are separate ELU MLPs. The actor maps 99 observations through 512/256/128 hidden
units to 29 Gaussian action means. The critic maps 111 observations through the same hidden widths
to one value. Observation normalization is part of the hashed checkpoint and exported ONNX graph.

Deterministic evaluation and playback use the actor mean. They create no optimizer, do not sample
the training Gaussian, and do not update policy or normalization state.
