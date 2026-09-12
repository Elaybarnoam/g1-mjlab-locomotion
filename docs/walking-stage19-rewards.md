# Walking Stage 19 contact state and reward profiles

Stage 19 is controlled reward development for `G1-Walking-Flat-v1`. It does not claim a qualified
walking policy. Every experiment arm starts independently from the same trusted bootstrap
checkpoint so a reward comparison cannot silently inherit learning from another arm.

## Runtime data flow

At each 20 ms control transition, `WalkingCommand` advances the command and phase exactly once and
then calls its `WalkingContactRuntime`. The runtime reads the dedicated contact-level sensor,
updates the preallocated GPU state, and publishes one immutable `WalkingContactSnapshot`. Reward
terms only read that snapshot. Calling one or every reward reader repeatedly cannot advance event
IDs, durations, touchdown history, or phase state.

The state is maintained independently for every environment and foot. It includes raw and
hysteretic contact, candidate and stable-state ages, swing peak clearance, confirmed-event IDs,
valid touchdown time and position, preceding opposite-foot touchdown, previous expected contact,
and expected transition phase. Resetting selected environments clears only their rows. A zero `dt`
does not advance state.

Contact entry requires 15 N, contact exit uses 8 N, and either change must persist for 60 ms.
Touchdown validity additionally requires 120 ms of swing and a 25 mm swing peak. These values and
all placement, slip, flight, and grace thresholds are frozen in
`configs/walking-v1/stage19/contact-profile.json`.

## Reward integration

The strict reward profile resolves all 30 allowlisted terms exactly once. Unknown terms, formulas,
parameters, non-finite values, missing terms, and non-positive scales are rejected. Disabled terms
must have zero weight. The resolved JSON and SHA-256 are copied into every run artifact and checked
on strict resume.

Rate rewards are returned as instantaneous values and mjlab applies `weight * value * control_dt`.
Per-event weights are divided by `control_dt` once when the manager configuration is built, so one
event contributes the declared profile weight after the manager's time scaling. The profile records
the integration kind, unit, mask, formula identifier, and typed parameters for every term.

The new cached terms are:

| Term | Purpose | Integration |
| --- | --- | --- |
| `phase_contact_error` | Stable contact disagreement with the expected gait phase | rate |
| `extra_contact_event` | Confirmed events outside the matched phase window | per event |
| `short_stance` | Liftoff before the minimum stance duration | per event |
| `short_swing` | Touchdown before the minimum swing duration | per event |
| `physical_stance_slip` | Force-weighted tangential contact-point speed squared | rate |
| `swing_clearance_error` | Swing sole-height error relative to the frozen target | rate |
| `touchdown_placement` | Eligible alternating step placement quality | per event |
| `bilateral_flight` | Sustained periods with neither foot in stable contact | rate |

The one-second walking transition grace suppresses event penalties but does not hide raw transition
telemetry. Initial contacts are state initialization, not scored events. Placement requires a valid
alternating touchdown, sufficient swing clearance, no sustained flight, and forward root progress;
standing in place or cycling legs without root progress cannot earn it.

## Controlled arms

- Arm A is the explicit bootstrap controller baseline with corrected host semantics.
- Arm B changes only contact timing: phase error, unmatched events, short durations, and flight.
- Arm C changes only physical slip relative to B and disables the inherited ankle-proxy slip term.
- Arm D is intentionally absent until evaluation shows that clearance and placement are the next
  justified mechanical hypothesis.

Each arm has a two-update smoke config and a 300-update train config under
`configs/walking-v1/stage19/`. All use 64 environments, 24 rollout steps, the same command/reset
profile, and the same PPO profile. A two-update smoke is a wiring and numerical-safety test, not
evidence that the gait improved.

## Reproduction

```bash
g1-mjlab train \
  --config configs/walking-v1/stage19/arm-c-smoke.json \
  --walking-profile configs/walking-v1/stage19/walking-profile.json \
  --walking-reward-profile configs/walking-v1/stage19/arm-c-rewards.json \
  --ppo-profile configs/walking-v1/stage19/ppo.json \
  --fine-tune /trusted/path/model_13398.pt \
  --output /new/output/directory
```

The output directory must be new. The run records the complete resolved MDP, source checkpoint
lineage, reward and walking profile hashes, full learner checkpoint, finite-loss summary, parameter
deltas, memory evidence, and immutable policy contract. Use `scripts/verify_stage19_smoke.py` only
with trusted local PyTorch checkpoints; PyTorch checkpoint loading is executable deserialization.
