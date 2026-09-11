# Walking method decision: locomotion before style

Status: accepted development decision; no walking policy is qualified yet.

Implementation status: the nominal-action bootstrap profile, 0.4–0.8 m/s command sampling, pinned
locomotion rewards, ordinary dynamic resets, and deterministic 0→0.6→0 m/s schedule are implemented.
A two-update GPU smoke completed with finite gradients and changing actor/critic parameters. This
proves the learning path only; the 6,000-update campaign and gait qualification remain outstanding.

## Decision

Train a robust command-conditioned G1 locomotion policy using the pinned mjlab velocity-task
mechanics before fine-tuning toward the licensed NVIDIA SOMA A057 human walking reference. Do not
extend the current reference-heavy PPO objective unchanged, and do not publish its checkpoints as
working walking policies.

The locomotion bootstrap must preserve the public walking controller equation:

```text
q_target = q_nominal + action_scale * actor_output
```

The human-style phase, reference targets, and rewards are introduced as a recorded fine-tuning
stage after the bootstrap passes forward-velocity, balance, contact, and stopping gates. The
reference-residual action center remains an experiment, not the release contract.

## Evidence

The W09 campaign produced three useful but insufficient behaviors:

| Run/checkpoint | Survival | Command RMS | Median stride | Finding |
| --- | ---: | ---: | ---: | --- |
| Stage 13 / 1199 | 6.06 s | 1.01 m/s | -0.02 m | rapid in-place/backward stepping |
| Stage 15 / 1199 | 4.44 s | 1.35 m/s | 0.15 m | more humanlike foot alternation, but backward and unstable |
| Stage 17 / 300 | 4.93 s | 0.97 m/s | 0.01 m | full-state fine-tuning preserved the source policy but did not solve locomotion |

The campaign also found concrete implementation defects. These are fixed and covered by tests:

- moving resets now apply the sampled command before physical reference initialization;
- a reset with `dt=0` no longer advances gait phase;
- corrected contact labels contain zero flight frames and four double-support frames;
- reference foot positions are transformed from world coordinates into the reference heading frame;
- transferred exploration variance can be explicitly preserved;
- `--fine-tune` restores actor, critic, optimizer, normalizers, and learned learning rate while
  starting a new iteration lineage.

Actor-only reward fine-tuning was rejected because one early PPO update reduced deterministic
survival from about 4.4 seconds to about 1.5 seconds. Full-state fine-tuning prevented that immediate
collapse. The foot-position imitation kernel was also saturated: a 0.12 m kernel at 0.23–0.28 m
error produced effectively zero reward. A 0.30 m development kernel restored a usable learning
signal but did not replace missing locomotion mechanics.

## Bootstrap requirements

The replacement bootstrap must reproduce the pinned mjlab G1 flat-velocity ingredients before
style tuning:

- exponential linear and angular velocity tracking;
- torso upright, body angular-velocity, and angular-momentum control;
- command-dependent posture tolerance;
- foot clearance, swing-height, slip, and soft-landing terms;
- joint-limit, self-collision, and action-rate costs;
- command curriculum and ordinary dynamic resets;
- deterministic stand→walk→stand evaluation with the existing anti-shuffle gates.

The pinned upstream runner defaults to 30,000 PPO updates. Official cloud examples use 6,000
updates with 4,096 environments. This repository measured 256 environments as the safe local batch
on the available 8 GiB laptop GPU, so the replacement campaign must declare its transition budget
and expected multi-hour runtime instead of pretending that another 600-update run is sufficient.

## Promotion rule

Bootstrap checkpoints may advance to style fine-tuning only after they sustain the complete
development timeline, follow forward commands, stop, and pass basic contact/slip checks. The final
walking release still requires the stricter humanlike stride, cadence, alternation, torso, and video
review gates. Standing-v1 remains independent and unchanged.
