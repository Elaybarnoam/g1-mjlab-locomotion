# Walking controller feasibility and training scale

Ticket W05 qualifies the existing G1 position controller for a bounded PPO smoke test. It does not
claim that replaying the reference joint angles is itself a walking controller.

## Result

The selected first-training batch is **256 environments** with 24 control steps per rollout, or
6,144 transitions per PPO update. Two fresh-process 0.5-second launches passed at every tested
batch size. The selected batch was the fastest tested count and retained 83.86% free GPU memory,
well above the 20% gate.

| Environments | Mean transitions/s | Minimum free VRAM | Max tracking RMS (rad) | Max force (N m) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 30.9 | 84.28% | 0.0678 | 18.40 |
| 16 | 407.4 | 84.28% | 0.0678 | 18.40 |
| 64 | 1,662.9 | 83.88% | 0.0678 | 18.40 |
| 128 | 3,422.2 | 83.88% | 0.0678 | 18.40 |
| 256 | 6,493.2 | 83.86% | 0.0678 | 18.40 |

The machine was Ubuntu 24.04 under WSL2, Python 3.13.15, PyTorch 2.9.0+cu128, mjlab 1.6.0 at
`8ee51fbcf806a7419189f706d9e394cbeb7790fa`, MuJoCo/MuJoCo Warp 3.11.0, RSL-RL 5.5.0, and an
8,151 MiB NVIDIA GeForce RTX 5060 Laptop GPU with driver 595.95.

## Important limitation

An additional 1.4-second, single-robot open-loop replay was deliberately run beyond the selection
window. The robot first fell at step 67 (1.34 seconds). The left knee reached its configured 139 N m
limit and maximum joint tracking error reached 0.831 rad. All values remained finite.

This is a real training risk, not a hidden pass. The reference supplies desired style and phase; the
PPO actor must learn balance corrections from observations. Therefore W06 is limited to a smoke
run and must verify finite optimization, changing parameters, checkpoints, and bounded actions.
The first pilot may start only after the gait evaluator exists, and success requires learned closed-
loop behavior rather than reference reward or short-horizon survival alone.

The reviewed machine-readable summary is
[`evidence/walking-v1/controller-feasibility.json`](../evidence/walking-v1/controller-feasibility.json).
Raw per-launch telemetry stays under ignored `.runtime/` and is not release evidence.
