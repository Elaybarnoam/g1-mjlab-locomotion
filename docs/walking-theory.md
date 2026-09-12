# Walking-v1 theory and implementation

## What the policy learns

At each 20 ms control sample, the actor chooses a residual posture for all 29 G1 joints. It does not
directly choose torque, footstep locations, or a trajectory. The environment converts action
`a_t` into position targets:

`q_target = q_nominal + action_scale × clip(a_t) − encoder_bias`.

MuJoCo's built-in position actuators then produce forces from target error and joint velocity using
the model's Kp/Kd and force limits. This PD loop runs at 200 Hz while the policy runs at 50 Hz.

## Actor input: 102 values

The deployed actor receives body-frame pelvis linear velocity (3), body-frame angular velocity (3),
gravity projected into the pelvis frame (3), offset joint position (29), joint velocity (29), the
previous applied action (29), applied planar command (3), phase sine/cosine (2), and walk blend (1).
The frozen normalizer transforms every component using checkpoint mean and standard deviation.

The phase is not an animation command. It is a clock feature that lets one stationary neural network
represent different parts of a periodic gait. Its rate is proportional to applied forward speed:

`phase_(t+1) = (phase_t + dt × applied_speed / (reference_speed × cycle_duration)) mod 1`.

Acceleration/deceleration filtering and stand/walk hysteresis make transitions continuous. The actor
at sample `t` sees the current applied speed/phase/blend. After inference, the host advances these
values for the next observation, maps the action once, and advances four physics steps.

## Critic input: 114 values

The critic receives the actor information plus privileged foot height, air time, contact booleans,
and contact-force vectors. This asymmetric design can improve value estimation during training
without requiring privileged sensors in the deployed policy. The critic is discarded for playback.

## PPO and GAE

Sixty-four robots collect 24 transitions each, producing 1,536 transitions per update. GAE computes
temporal-difference residuals and advantages backwards:

- `δ_t = r_t + γ V_(t+1) − V_t`
- `A_t = δ_t + γ λ continuation_mask × A_(t+1)`
- `return_target_t = A_t + V_t`

PPO reuses the rollout for five epochs and four shuffled minibatches. The actor objective uses the
smaller of the normal probability-ratio objective and its clipped version with ε=0.2. The critic uses
clipped value loss; the optimizer clips global gradient norm at 1.0 and adapts learning rate toward a
KL target of 0.01. This run uses γ=0.99, λ=0.95, and entropy coefficient 0.

The pinned RSL-RL wrapper's timeout behavior is important: timeout-only transitions receive γ times
the stored current transition value as reward compensation before done-masked GAE. The project
adapter ensures a simultaneous true termination takes precedence. This is not the same as claiming
that a separately recorded final observation is bootstrapped.

## Rewards versus evaluation

Reward shapes learning; it does not prove success. Stage 19 has 30 versioned terms for tracking,
uprightness, pose, motion smoothness, contact timing, physical slip, stance/swing duration, clearance,
touchdown placement, flight, collision, and termination. Rates and events use explicit integration
semantics so a control-rate change cannot silently alter their meaning.

Checkpoint selection ignores training return. Frozen physical evaluations separately test survival,
velocity tracking, stopping, alternating contacts, cadence, step/stride length, asymmetry, foot slip,
clearance, pelvis height, tilt, and contact chatter. That separation exposed the present local
optimum: it moves and tracks command, but does not produce the intended human-like gait.

## What must happen next

Before changing methods, audit reference feasibility, whether the policy uses phase, observation
adequacy, and reward signal scales. If evidence confirms a phase-ignored local optimum, imitation
pretraining or adversarial motion priors may be appropriate—but either requires a new specification
and controlled baseline. More PPO updates alone are not authorized by the current experiment.

