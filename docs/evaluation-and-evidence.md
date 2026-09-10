# Evaluation and evidence

## Protocol

Development checkpoint selection ranks strict passes, survival passes, lower median drift, lower
median tilt, and then later update. Final-test scenarios use a distinct seed and are evaluated only
after selection. The frozen final gate requires at least 95 of 100 strict 60-second trials in both
mjlab/MuJoCo Warp and native MuJoCo/ONNX.

A strict pass requires finite state, no true fall termination, continuous foot support except for at
most 0.10 seconds, root drift no greater than 0.20 m, and—after one settling second—torso tilt no
greater than 15 degrees and pelvis height at least 0.60 m. Survival is reported separately so a
motion-quality failure cannot be disguised as standing success.

Auto-reset is disabled while measuring. The terminal frame is recorded before reset, termination
wins over a simultaneous timeout, and trial state becomes immutable after its first episode ends.

## Results

Three independently trained seeds produced development-qualified policies. The selected seed-42
`model_999.pt` checkpoint passed 100/100 final trials in each backend. The two-sided 95% Wilson
interval for 100/100 is 0.9630–1.0. Median/worst mjlab root drift was 0.0215/0.0407 m; maximum torso
tilt was 3.38 degrees; minimum post-settling height was 0.701 m.

The native adapter replayed the same saved initial states with the exported ONNX actor. A separate
31-state parity corpus passed with maximum observation-field error `5.97e-8` and maximum inference
error `4.18e-7`.

[`evidence/standing-v1/summary.json`](../evidence/standing-v1/summary.json) is the small public result
record. It contains hashes for the private executable evidence but not the checkpoint, ONNX model,
compiled model, or raw scenario states themselves.

## Interpretation

The result qualifies nominal flat-ground standing within the tested reset distribution. It does not
measure cross-engine variation, broad physics variation, disturbances, terrain, sensor estimation,
latency, actuator calibration, thermal/current limits, or real-robot safety. The deterministic video
is supporting visual evidence, not a substitute for the final trial table.
