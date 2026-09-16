# Walking development evaluation

Walking-v1 separates functional success from gait style. A robot cannot pass by merely surviving,
moving its arms, sliding, hopping, or taking tiny rapid steps. The frozen development thresholds are
machine-readable in [`configs/walking-v1/evaluation.json`](../configs/walking-v1/evaluation.json).
They were committed before the pilot and are not final-release claims.

Each deterministic trial retains only the first episode before an automatic reset could hide a
failure. The evaluator records an NPZ trace and measures the applied command, body-frame forward
speed, pelvis height, torso tilt, both foot positions, debounced foot contact, finite state, and
termination. Contacts shorter than 0.06 seconds are rejected as chatter.

Functional development gates require at least four commanded walking seconds, six completed steps,
forward-speed RMS error no greater than 0.25 m/s, no fall/non-finite state, and a settled zero-command
speed RMS no greater than 0.10 m/s when a stop is present.

Style gates require alternating feet on at least 80% of steps, median step length from 0.18 to 0.65
m, no more than 20% tiny steps, cadence from 1 to 3 steps/s, stance-foot slip RMS no greater than
0.12 m/s, pelvis height at least 0.62 m, torso tilt no greater than 0.35 rad, and no more than six raw
contact transitions/s. Insufficient walking or steps also fails style because absent evidence cannot
certify natural motion.

Checkpoint ranking is lexicographic: combined functional/style passes, functional passes, lower
command error, stronger alternation, fewer tiny steps, lower slip, then later update. Training reward
is deliberately absent from selection.

The evaluator was exercised against the W06 two-update checkpoint on four held-out stand→walk→stand
trials. All four correctly failed at 1.36 seconds before walking began. This is expected and confirms
that the evaluator does not convert resets or missing gait samples into a favorable result. See
[`evidence/walking-v1/evaluator-smoke.json`](../evidence/walking-v1/evaluator-smoke.json).

## Final qualification protocol

Final thresholds are separately frozen in
[`configs/walking-v1/final-acceptance-v1.json`](../configs/walking-v1/final-acceptance-v1.json).
The final suite contains 100 serialized 60-second trials: 25 each at 0.4, 0.6, and 0.8 m/s and 25
with repeated stand/walk transitions. Seeds 20000–20099 are reserved exclusively for generation.
Both mjlab and native MuJoCo must pass at least 95/100 overall and 23/25 in every stratum, using the
same checkpoint, ONNX, model, controller, reference, command, criteria, and initial-state hashes.

`check-walking-final-prerequisite` runs before scenarios are frozen or outcomes are accessed. It
requires a development-qualified policy covering 0–0.8 m/s and checkpoint-bound owner visual
acceptance. The currently retained P04-09 policy is explicitly unqualified and covers only the
0–0.6 m/s development domain, so final execution is correctly blocked. A failed final suite is
recorded as failure and becomes development evidence; it cannot silently be reused as a final test.
