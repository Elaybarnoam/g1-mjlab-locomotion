# Walking evaluation schema 2

Schema 2 measures the Unitree G1's physical gait rather than treating ankle-link translation as
foot slip or every contact switch as a step. Schema 1 remains available for reproducing historical
results; the two schemas must not be mixed in one comparison table.

## Contact and event definitions

Each sole uses force hysteresis. Contact begins above 15 N and ends below 8 N. A new state must be
observed continuously for 0.06 seconds before confirmation. The state records both the first force
crossing and confirmation time. Reset samples initialize from measured force and never create a
synthetic touchdown.

A touchdown is valid only after at least 0.12 seconds of confirmed swing and a sole-clearance peak
of at least 0.025 m. Adjacent valid touchdowns on the same side are repeats, not alternating steps;
simultaneous bilateral touchdowns are one hop. Step length is the heading-projected distance from
the opposite foot's preceding valid touchdown. Stride length is the distance between successive
valid touchdowns of the same foot. These values are intentionally separate.

Physical slip is evaluated at every retained contact point:

```text
v_contact = v_body_origin + omega_body × (p_contact - p_body_origin)
v_tangent = v_contact - dot(v_contact, n_terrain) n_terrain
slip_rms = sqrt(mean_t(sum_contacts(Fn |v_tangent|²) / sum_contacts(Fn)))
```

The installed mjlab API reports body-link velocity at the link origin and contact positions and
normals in world coordinates. A diagnostic-only four-slot sensor retains contact-level values at
all four 200 Hz physics substeps. The policy and command remain at 50 Hz. Ankle-link speed is
reported separately as a compatibility proxy and is never substituted for physical slip.

Speed metrics use the pelvis yaw heading after the applied command has changed by at most 0.01 m/s
for one second and the walking blend is at least 0.9. Every post-walk zero-command segment is scored
after command deceleration reaches zero, 0.5 seconds of settling, and at least one second of data.
Missing contact samples, an incomplete stop, or a short trace yields `insufficient_evidence`; it is
not converted into a passing zero.

The frozen development function gate also requires at least 4.0 seconds of eligible walking, six
valid alternating steps, pelvis height at least 0.62 m, and the complete planned horizon. Style
requires median step length 0.18--0.65 m, no more than 20% tiny steps, torso tilt no more than
0.35 rad, step asymmetry no more than 0.20 when measurable, and sustained bilateral-flight fraction
no more than 0.05. These criteria prevent a speed-tracking shuffle from being labeled functional.

## Diagnostic command

Run at most four detailed worlds because contact-level physics traces are intentionally expensive:

```console
g1-mjlab diagnose-walking \
  --config configs/walking-v1/stage18-bootstrap-to-style-train.json \
  --checkpoint RUN/checkpoints/model_1199.pt \
  --scenarios configs/walking-v1/measurement-scenarios-v2.json \
  --criteria configs/walking-v1/evaluation-v2.json \
  --output .runtime/measurement-v2 \
  --physics-trace \
  --video
```

The command writes strict control-rate traces, 200 Hz contact traces, confirmed event journals,
six-panel SVG diagnostics, a deterministic MP4, `summary.json`, and `measurement-audit.json`. NPZ
loading always uses `allow_pickle=False`; JSON rejects NaN and Infinity. Checkpoint, source,
controller, reference, scenario, and initial-state identities are embedded in metadata.

The thresholds in `evaluation-v2.json` are frozen development criteria, not final qualification
claims. New reward experiments may begin only after their scenarios and this evaluator identity are
recorded. Raw local traces stay under `.runtime/` and are not distributed with the source package.

## P04-02 historical audit

The four-world 0→0.6→0 m/s replay finds mixed physical causes, not just a sensor artifact. The
bootstrap checkpoint has roughly 0.426 m/s steady-walk physical slip and 13.04 raw transitions/s.
Stage18/900 reduces slip to about 0.325 m/s but still chatters. Stage18/1199 reduces slip to about
0.300 m/s and produces more valid steps, but alternation remains poor and raw transitions remain
about 12–13/s. All three checkpoints track forward speed and complete the final stop, but none
passes the frozen four-world function gate (Stage18/1199 passes one of four trials). Exact values and
immutable identities are in the local P04-02 outputs; these rounded observations are not
qualification claims.
