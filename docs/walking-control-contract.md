# Walking-v1 command, phase, and policy contract

Walking-v1 is a separate policy identity from standing-v1. The first actor will have 102 inputs and
29 outputs; the critic used only during PPO will have 114 inputs. These dimensions are executable
declarations in `g1_mjlab.tasks.walking_v1.walking_policy_contract`, not manually inferred from a
diagram.

## Control-step ordering

At each 20 ms policy step the host performs one canonical update:

1. Read the requested command `[vx, vy, yaw_rate]` in the yaw-aligned body frame.
2. Validate that `vy = 0`, `yaw_rate = 0`, and `vx` is either 0 or 1.163811593 m/s.
3. Rate-limit the applied command: acceleration is 0.6 m/s² and deceleration is 0.8 m/s².
4. Update the walking mode with hysteresis: enter at 0.15 m/s and leave at 0.05 m/s.
5. Move the stand/walk blend toward that mode by at most 0.02 per policy step (1.0/s).
6. Advance phase by `dt * applied_vx / (reference_speed * cycle_duration)` and wrap modulo one.
7. Increase the unwrapped reference distance by `dt * applied_vx`.
8. Build the observation, run the actor, scale its 29 outputs, and apply one position controller.

Phase does not jump to zero on stop or restart. At zero applied speed it freezes. The separate
unwrapped distance keeps the reference root continuous across any number of phase wraps; a clip
boundary must never reset or teleport simulator state.

The public command profile is `configs/walking-v1/commands.json`. The deterministic development
timeline is `configs/walking-v1/development-schedule.json`: stand for 2 s, request nominal walking
for 10 s, then request standing for 3 s. Intermediate applied speeds exist only during the ramps.
They are not yet claimed as sustained supported speeds.

## Actor vector (102 float32 scalars)

| Slice | Field | Unit and frame | Source |
| --- | --- | --- | --- |
| `[0:3]` | base linear velocity | m/s, body | deployable pelvis velocity estimate |
| `[3:6]` | base angular velocity | rad/s, body | deployable pelvis angular estimate |
| `[6:9]` | projected gravity | unitless, body | pelvis orientation and world gravity |
| `[9:38]` | joint positions | rad, joint order | encoder positions relative to nominal pose |
| `[38:67]` | joint velocities | rad/s, joint order | encoders |
| `[67:96]` | previous action | unitless, joint order | action applied at the previous policy step |
| `[96:99]` | applied command | m/s, m/s, rad/s; yaw-aligned body | rate-limited host state |
| `[99:100]` | phase sine | unitless | `sin(2π phase)` |
| `[100:101]` | phase cosine | unitless | `cos(2π phase)` |
| `[101:102]` | walk blend | unitless `[0,1]` | rate-limited host state |

The actor never receives privileged contact force or perfect simulator-only foot state. Requested
and applied commands are both logged, but the actor receives the applied command because it matches
the gait cadence actually being requested.

## Critic vector (114 float32 scalars)

The critic begins with actor fields `[0:99]` through applied command. It then receives privileged
simulation signals: foot height `[99:101]`, foot air time `[101:103]`, binary contact
`[103:105]`, and two world-frame force vectors `[105:111]`. Phase sine, phase cosine, and blend are
`[111:114]`. Those privileged values improve value estimation during PPO but are unavailable to the
deployed actor.

## Action and state reconstruction

The output order is the exact 29-joint order in the motion manifest and G1 model. Each output is a
normalized joint-position offset; W04/W05 will resolve the per-joint scales, nominal positions,
limits and PD feasibility. No torque policy and no second PD layer are introduced.

A nonstationary scenario must save root position and normalized WXYZ orientation, all three
world-frame root linear velocity components, all three world-frame angular velocity components,
29 joint positions, 29 joint velocities, phase, and blend. Omitting floating-base velocity would
make a moving initial state impossible to reproduce faithfully in native MuJoCo.

NumPy and vectorized Torch implementations share these equations. The GPU step contains no Python
loop over robots and no tensor-to-host validation transfer; requested schedules are validated before
they are installed on the device.
